"""Chunk every downloaded 10-K into data/processed/chunks.jsonl.

Usage:
    python -m kgrag.ingest.chunker

Each chunk gets a deterministic id: "{doc_id}:{item}:{index}", e.g. "NVDA-10K-2026-01-25:1A:0007".
This chunk_id is the shared key between Neo4j (every edge stores it) and pgvector
(every embedding row is keyed by it). Re-chunking the same file yields the same ids.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from kgrag.config import PROCESSED_DIR, RAW_DIR
from kgrag.ingest.companies import SECTION_MARKERS
from kgrag.ingest.parse_10k import html_to_text, split_sections, split_sections_by_markers

CHUNK_SIZE = 3000     # characters, roughly 700 tokens
CHUNK_OVERLAP = 300   # so a sentence on a boundary appears whole in one chunk
SECTIONS = ("1", "1A", "7")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_hash: str        # sha256 of the source document text, used to cache extraction
    ticker: str
    company: str
    report_date: str
    filing_date: str
    section_item: str
    section_path: str    # "10-K > Item 1A Risk Factors"
    index: int
    text: str


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_document(html: bytes, meta: dict) -> list[Chunk]:
    text = html_to_text(html)
    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = []
    markers = SECTION_MARKERS.get(meta["ticker"])
    sections = (split_sections_by_markers(text, markers, SECTIONS) if markers
                else split_sections(text, SECTIONS))
    if not sections:
        print(f"[warn] {meta['doc_id']}: no sections found; add markers in companies.py")
    for section in sections:
        for i, piece in enumerate(_splitter.split_text(section.text)):
            chunks.append(Chunk(
                chunk_id=f"{meta['doc_id']}:{section.item}:{i:04d}",
                doc_id=meta["doc_id"],
                doc_hash=doc_hash,
                ticker=meta["ticker"],
                company=meta["company"],
                report_date=meta["report_date"],
                filing_date=meta["filing_date"],
                section_item=section.item,
                section_path=f"10-K > Item {section.item} {section.title}",
                index=i,
                text=piece,
            ))
    return chunks


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "chunks.jsonl"
    total = 0
    with out_path.open("w", encoding="utf-8") as out:
        for meta_path in sorted(RAW_DIR.glob("*.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            html = (RAW_DIR / f"{meta['doc_id']}.htm").read_bytes()
            chunks = chunk_document(html, meta)
            by_section = {}
            for c in chunks:
                by_section[c.section_item] = by_section.get(c.section_item, 0) + 1
                out.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            total += len(chunks)
            chars = sum(len(c.text) for c in chunks)
            print(f"{meta['doc_id']}: {len(chunks)} chunks {by_section}, {chars / 1e3:.0f}k chars")
    print(f"\nWrote {total} chunks to {out_path}")


if __name__ == "__main__":
    main()
