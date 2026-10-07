from kgrag.ingest.chunker import chunk_document
from kgrag.ingest.parse_10k import html_to_text, split_sections
from kgrag.ontology import EntityType as E, RelationType as R, is_allowed

BUSINESS = "We design GPUs. Our foundry partner TSMC supplies our wafers. " * 200
RISKS = "We depend on a limited number of suppliers including TSMC. " * 150
MDNA = "Revenue grew due to data center demand. " * 100

FILING = f"""
<html><body>
<div style="display:none"><ix:header>hidden xbrl junk</ix:header></div>
<p>TABLE OF CONTENTS</p>
<p>Item 1. Business 4</p>
<p>Item 1A. Risk Factors 12</p>
<p>Item 1B. Unresolved Staff Comments 30</p>
<p>Item 7. Management's Discussion 40</p>
<p>Item 7A. Quantitative Disclosures 50</p>
<p>Item 10. Directors 60</p>
<table><tr><td>Item 1.</td><td>Business</td></tr></table>
<p>{BUSINESS}</p>
<table><tr><td>Revenue</td><td>$1,000</td><td>$2,000</td></tr></table>
<p>Item 1A. Risk Factors</p>
<p>{RISKS}</p>
<p>Item 1B. Unresolved Staff Comments</p>
<p>None.</p>
<p>Item 7. Management&#8217;s Discussion and Analysis</p>
<p>{MDNA}</p>
<p>Item 7A. Quantitative and Qualitative Disclosures</p>
<p>Interest rate risk.</p>
<p>Item 10. Directors</p>
<p>See proxy.</p>
</body></html>
"""

META = {
    "doc_id": "TEST-10K-2026-01-01", "ticker": "TEST", "company": "Test Corp",
    "report_date": "2026-01-01", "filing_date": "2026-02-01",
}


def test_html_to_text_drops_xbrl_and_financial_tables():
    text = html_to_text(FILING)
    assert "hidden xbrl junk" not in text
    assert "$1,000" not in text


def test_split_sections_picks_real_section_not_toc():
    sections = {s.item: s for s in split_sections(html_to_text(FILING))}
    assert set(sections) == {"1", "1A", "7"}
    # The real Item 1 heading lives in a layout table; if it were dropped, this would fail.
    assert sections["1"].text.startswith("We design GPUs")
    assert "limited number of suppliers" in sections["1A"].text
    assert "Unresolved" not in sections["1A"].text   # stops at Item 1B
    assert "data center demand" in sections["7"].text
    assert "Interest rate" not in sections["7"].text  # Item 7A is not Item 7


def test_chunk_ids_are_deterministic_and_unique():
    first = chunk_document(FILING.encode(), META)
    second = chunk_document(FILING.encode(), META)
    ids = [c.chunk_id for c in first]
    assert ids == [c.chunk_id for c in second]
    assert len(ids) == len(set(ids))
    assert ids[0] == "TEST-10K-2026-01-01:1:0000"
    assert all(len(c.text) <= 3000 for c in first)


def test_ontology_rejects_wrong_direction():
    assert is_allowed(R.EXECUTIVE_OF, E.PERSON, E.COMPANY)
    assert not is_allowed(R.EXECUTIVE_OF, E.COMPANY, E.PERSON)
    assert not is_allowed(R.SUPPLIES, E.COMPANY, E.PRODUCT)
