"""Turn a 10-K HTML filing into clean text split by Item (section).

10-Ks repeat every "Item X." heading twice: once in the table of contents and once
at the real section. We keep, for each item, the occurrence with the longest body,
which is the real section rather than the one-line TOC entry.
"""

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Inline XBRL filings are XHTML; the HTML parser handles them fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SECTION_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "7": "Management's Discussion and Analysis",
}

BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"]

# Longest alternatives first so "Item 10" is not read as "Item 1".
ITEM_HEADING = re.compile(
    r"^\s*item\s+(1[0-6]|1[abc]|7a|9[abc]|[1-9])\b\.?",
    re.IGNORECASE | re.MULTILINE,
)


# Page furniture repeated on every page: "Table of Contents" links, page numbers, "2025 10-K".
PAGE_FOOTER = re.compile(r"^(Table of Contents|\d{1,3}|\d{4} (Form )?10-K)$", re.IGNORECASE)


@dataclass
class Section:
    item: str      # "1", "1A", "7"
    title: str
    text: str


def html_to_text(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Inline XBRL hidden header (machine-readable tags, not prose).
    for tag in soup.find_all(["ix:header", "script", "style"]):
        tag.decompose()
    # Tables in 10-Ks are almost all financial figures: noise for relationship extraction.
    # Exception: some filers lay out "Item 1A. Risk Factors" headings inside small tables.
    for table in soup.find_all("table"):
        table_text = table.get_text(" ", strip=True)
        if len(table_text) < 300 and ITEM_HEADING.match(table_text):
            table.replace_with(BeautifulSoup(f"<p>{table_text}</p>", "lxml").p)
        else:
            table.decompose()
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_after("\n")

    text = soup.get_text().replace("\xa0", " ")
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line and not PAGE_FOOTER.match(line))


def split_sections_by_markers(
    text: str, markers: dict[str, tuple[str, str]], items: tuple[str, ...]
) -> list[Section]:
    """For filers that do not use "Item X." headings in the body (e.g. Intel).

    markers: item -> (start regex, end regex), each matched at the start of a line.
    The start line is kept, since it may be prose rather than a heading.
    """
    sections = []
    for item in items:
        if item not in markers:
            continue
        start_re, end_re = markers[item]
        start = re.search(start_re, text, re.MULTILINE)
        if not start:
            continue
        end = re.search(end_re, text[start.end():], re.MULTILINE)
        stop = start.end() + end.start() if end else len(text)
        sections.append(Section(item=item, title=SECTION_TITLES.get(item, f"Item {item}"),
                                text=text[start.start():stop].strip()))
    return sections


def split_sections(text: str, items: tuple[str, ...] = ("1", "1A", "7")) -> list[Section]:
    matches = list(ITEM_HEADING.finditer(text))
    best: dict[str, tuple[int, int]] = {}  # item -> (start, end) of longest occurrence

    for i, m in enumerate(matches):
        item = m.group(1).upper()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if item not in best or (end - start) > (best[item][1] - best[item][0]):
            best[item] = (start, end)

    sections = []
    for item in items:
        if item not in best:
            continue
        start, end = best[item]
        body = text[start:end].split("\n", 1)[-1].strip()  # drop the heading line
        sections.append(Section(item=item, title=SECTION_TITLES.get(item, f"Item {item}"), text=body))
    return sections
