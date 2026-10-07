"""Phase 4: grounded answer with a citation per claim, validated against what was retrieved.

Usage:
    python -m kgrag.answer.generate "Which suppliers do NVIDIA and AMD share?"

The model returns claims, each citing source ids from the context. A draft is rejected when
a claim has no citation, cites an id that was not in the context (an invented source), or
states a number its cited text does not contain;
the model then regenerates with the problems listed. After MAX_ATTEMPTS, claims that still
fail are dropped and reported rather than shown, so every citation in the final answer
resolves to a chunk that was actually retrieved.
"""

import re
import sys
import time
from dataclasses import asdict, dataclass, field

from pydantic import BaseModel, Field

from kgrag import llm
from kgrag.answer.context import Context, build_context
from kgrag.retrieval.retriever import retrieve

MAX_ATTEMPTS = 3
REFUSAL = "The filings in this corpus do not contain the answer to this question."

SYSTEM_PROMPT = """You answer questions about semiconductor companies using ONLY the context provided,
which comes from their SEC 10-K filings.

Context has two labeled parts:
- Graph facts: relationships extracted from the filings, each with the quote that supports it.
- Retrieved passages: verbatim filing text.
Each item has a source id like S3.

Rules:
1. Every claim is one factual sentence and cites the source ids that state it, e.g. ["S3", "S7"].
2. Cite only source ids that appear in the context. Never cite anything else.
3. Use no outside knowledge. If the context does not answer the question, set answerable to false
   and return no claims. Do not guess.
4. Put the direct answer to the question first, then supporting detail.
5. When combining facts across hops (A supplies B, B competes with C), cite every fact used."""


class Claim(BaseModel):
    text: str = Field(description="One factual sentence answering part of the question")
    citations: list[str] = Field(description="Source ids from the context that state this claim, e.g. S3")


class AnswerDraft(BaseModel):
    answerable: bool = Field(description="false when the context does not contain the answer")
    claims: list[Claim]


@dataclass
class Citation:
    source_id: str
    kind: str
    chunk_ids: list[str]
    quote: str


@dataclass
class Answer:
    question: str
    answerable: bool
    answer: str
    claims: list[dict]
    route_used: str
    attempts: int
    dropped_claims: list[dict] = field(default_factory=list)
    rejections: list[list[str]] = field(default_factory=list)  # problems found in each rejected draft
    retrieval_notes: list[str] = field(default_factory=list)
    latency_ms: int = 0


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def check(draft: AnswerDraft, ctx: Context) -> tuple[list[Claim], list[str]]:
    """Return (claims that pass every check, problems found).

    Checks: each claim cites at least one source; every cited id exists in the context
    (no invented sources); every number in the claim appears in the text it cites
    (no invented or misremembered figures).
    """
    known = ctx.by_id()
    valid, problems = [], []
    if draft.answerable and not draft.claims:
        problems.append("answerable is true but there are no claims")
    for i, claim in enumerate(draft.claims if draft.answerable else [], 1):
        cites = [c.strip().strip("[]") for c in claim.citations]
        unknown = [c for c in cites if c not in known]
        if not cites:
            problems.append(f"claim {i} has no citation: {claim.text!r}")
            continue
        if unknown:
            problems.append(f"claim {i} cites ids not in the context {unknown}: {claim.text!r}")
            continue
        cited_text = " ".join(known[c].text + " " + " ".join(known[c].evidence) for c in cites)
        missing = _numbers(claim.text) - _numbers(cited_text)
        if missing:
            problems.append(f"claim {i} states numbers {sorted(missing)} that its cited sources do not contain: "
                            f"{claim.text!r}")
            continue
        valid.append(Claim(text=claim.text, citations=cites))
    return valid, problems


def _render_claims(claims: list[Claim], ctx: Context) -> tuple[str, list[dict]]:
    known = ctx.by_id()
    rendered, out = [], []
    for c in claims:
        rendered.append(f"{c.text} [{', '.join(c.citations)}]")
        out.append({"text": c.text, "citations": [asdict(Citation(
            sid, known[sid].kind, known[sid].chunk_ids,
            known[sid].evidence[0] if known[sid].evidence else known[sid].text[:300])) for sid in c.citations]})
    return " ".join(rendered), out


def answer(question: str) -> Answer:
    start = time.perf_counter()
    retrieval = retrieve(question)
    ctx = build_context(retrieval)
    result = Answer(question, False, REFUSAL, [], retrieval.route_used, 0,
                    retrieval_notes=list(retrieval.notes))

    if ctx.sources:
        base = f"Question: {question}\n\n{ctx.render()}"
        message = base
        claims: list[Claim] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result.attempts = attempt
            try:
                draft, _ = llm.generate_structured(SYSTEM_PROMPT, message, AnswerDraft)
            except llm.LLMError as e:
                result.rejections.append([f"generation failed: {e}"])
                continue
            claims, problems = check(draft, ctx)
            if not problems:
                result.answerable = draft.answerable and bool(claims)
                break
            result.rejections.append(problems)
            message = (base + "\n\nYour previous answer was rejected:\n- " + "\n- ".join(problems)
                       + "\nAnswer again following the rules.")
        else:
            # Out of attempts: keep only claims whose citations resolve; report the rest.
            result.answerable = bool(claims)
            result.dropped_claims = [{"problem": p} for p in result.rejections[-1]] if result.rejections else []

        if result.answerable:
            result.answer, result.claims = _render_claims(claims, ctx)

    result.latency_ms = round((time.perf_counter() - start) * 1000)
    return result


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Which suppliers do NVIDIA and AMD share?"
    a = answer(question)
    print(f"Q: {a.question}\nroute: {a.route_used}  attempts: {a.attempts}  latency: {a.latency_ms} ms")
    for r in a.rejections:
        print(f"rejected draft: {r}")
    print(f"\nA: {a.answer}\n")
    for c in a.claims:
        for cit in c["citations"]:
            print(f"  [{cit['source_id']}] {cit['kind']:7} {', '.join(cit['chunk_ids'][:2])}  \"{cit['quote'][:100]}\"")
    if a.dropped_claims:
        print(f"dropped: {a.dropped_claims}")


if __name__ == "__main__":
    main()
