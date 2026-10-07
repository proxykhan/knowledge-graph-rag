"""Download the latest 10-K filing for each ticker from SEC EDGAR.

Usage:
    python -m kgrag.ingest.edgar                 # all tickers in companies.py
    python -m kgrag.ingest.edgar NVDA AMD INTC   # just these

SEC rules: identify yourself with a User-Agent (SEC_USER_AGENT in .env) and stay
under 10 requests/second. Files land in data/raw/ and are skipped if already present.
"""

import json
import sys
import time

import requests

from kgrag.config import RAW_DIR, settings
from kgrag.ingest.companies import TICKERS

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
REQUEST_INTERVAL_S = 0.2  # 5 req/s, half the SEC limit


class EdgarClient:
    def __init__(self, user_agent: str):
        if not user_agent.strip():
            raise SystemExit("Set SEC_USER_AGENT in .env (e.g. 'Your Name you@example.com').")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._last_request = 0.0

    def get(self, url: str) -> requests.Response:
        wait = REQUEST_INTERVAL_S - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp

    def ticker_to_cik(self) -> dict[str, tuple[int, str]]:
        data = self.get(TICKER_MAP_URL).json()
        return {row["ticker"].upper(): (row["cik_str"], row["title"]) for row in data.values()}

    def latest_10k(self, cik: int) -> dict | None:
        recent = self.get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        for i, form in enumerate(recent["form"]):
            if form == "10-K":
                return {
                    "accession": recent["accessionNumber"][i],
                    "filing_date": recent["filingDate"][i],
                    "report_date": recent["reportDate"][i],
                    "primary_document": recent["primaryDocument"][i],
                }
        return None


def download(tickers: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = EdgarClient(settings.sec_user_agent)
    cik_map = client.ticker_to_cik()

    for ticker in tickers:
        ticker = ticker.upper()
        if ticker not in cik_map:
            print(f"[skip] {ticker}: not in SEC ticker map")
            continue
        cik, company = cik_map[ticker]
        filing = client.latest_10k(cik)
        if filing is None:
            print(f"[skip] {ticker}: no 10-K found (foreign filers use 20-F)")
            continue

        doc_id = f"{ticker}-10K-{filing['report_date']}"
        html_path = RAW_DIR / f"{doc_id}.htm"
        meta_path = RAW_DIR / f"{doc_id}.json"
        if html_path.exists() and meta_path.exists():
            print(f"[cached] {doc_id}")
            continue

        url = ARCHIVE_URL.format(
            cik=cik,
            accession=filing["accession"].replace("-", ""),
            document=filing["primary_document"],
        )
        html_path.write_bytes(client.get(url).content)
        meta = {"doc_id": doc_id, "ticker": ticker, "company": company, "cik": cik, "url": url, **filing}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[ok] {doc_id}  ({html_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    download(sys.argv[1:] or TICKERS)
