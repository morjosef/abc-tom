"""
Minimal Finviz screener client.

Replaces the `finviz` PyPI package (2.0.0), whose HTML parser broke when Finviz
redesigned the screener page. The new page embeds the full result set as JSON
inside the server-rendered HTML, so we read that directly instead of scraping
table cells.

Returns rows shaped like the old library's dicts — {"Ticker": ..., "Price": ...,
"RSI": ...} — so callers need no column-shift workarounds.
"""

import json
import re
import time

import requests

BASE_URL = "https://finviz.com/screener.ashx"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
PAGE_SIZE = 20
TOTAL_RE = re.compile(r"#\d+ / (\d+) Total")


class FinvizError(RuntimeError):
    """Raised when Finviz is reachable but the response is not usable."""


class FinvizBlocked(FinvizError):
    """Raised when Finviz served a block/challenge page instead of results."""


def _fetch(filters, order, offset, timeout):
    params = {
        "v": "171",
        "f": ",".join(filters),
        "o": order,
    }
    if offset:
        params["r"] = str(offset + 1)

    resp = requests.get(
        BASE_URL,
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.text


def _parse_rows(html):
    """Extract the embedded result rows. Returns (rows, total)."""
    marker = '"rows":['
    idx = html.find(marker)
    if idx == -1:
        # Distinguish a genuine empty result from a block page: the toolbar
        # count is present on both empty and populated result pages.
        if TOTAL_RE.search(html) or "fv-table_toolbar-count" in html:
            return [], 0
        raise FinvizBlocked(
            "Finviz returned a page with no result payload. This is usually a "
            "block or challenge page, not an empty screener result."
        )

    decoder = json.JSONDecoder()
    raw, _ = decoder.raw_decode(html[idx + len('"rows":'):])

    total_match = TOTAL_RE.search(html)
    total = int(total_match.group(1)) if total_match else len(raw)

    rows = []
    for item in raw:
        text = item.get("text", {})
        nums = item.get("numbers", {})
        ticker = text.get("ticker")
        if not ticker:
            continue
        rows.append(
            {
                "Ticker": ticker,
                "Company": text.get("company", ""),
                "Industry": text.get("industry", ""),
                "Country": text.get("country", ""),
                "Price": nums.get("price"),
                "Change": nums.get("change"),
                "Volume": nums.get("volume"),
                "RSI": nums.get("rsi"),
                "Beta": nums.get("beta"),
                "SMA20": nums.get("sma20"),
                "SMA50": nums.get("sma50"),
                "SMA200": nums.get("sma200"),
                "High52W": nums.get("high52w"),
                "Low52W": nums.get("low52w"),
                "MarketCap": nums.get("marketCap"),
                "ATR": nums.get("averageTrueRange"),
                "Gap": nums.get("gap"),
                "PrevClose": nums.get("prevClose"),
                "Open": nums.get("open"),
            }
        )
    return rows, total


def screen(filters, order="ticker", max_rows=None, timeout=30, retries=3,
           pause=1.5):
    """Run a Finviz screen and return a list of row dicts.

    filters: list of Finviz filter codes, e.g. ["cap_midover", "ta_rsi_u50"].
    max_rows: stop after this many rows (None = fetch everything).
    """
    rows = []
    offset = 0
    total = None

    while True:
        html = None
        last_error = None
        for attempt in range(retries):
            try:
                html = _fetch(filters, order, offset, timeout)
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(pause * (attempt + 1))
        if html is None:
            raise FinvizError(
                f"Finviz request failed after {retries} attempts: {last_error}"
            )

        page_rows, page_total = _parse_rows(html)
        if total is None:
            total = page_total

        if not page_rows:
            break

        rows.extend(page_rows)

        if max_rows is not None and len(rows) >= max_rows:
            return rows[:max_rows]

        offset += PAGE_SIZE
        if offset >= total:
            break

        time.sleep(pause)

    return rows
