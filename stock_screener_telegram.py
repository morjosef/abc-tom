"""
Daily Stock Screener → Telegram
Runs automatically via GitHub Actions every weekday.
"""

import os
import io
import json
import requests
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from finviz.screener import Screener
from datetime import date

# ========== טעינת הגדרות ==========
def load_settings() -> dict:
    try:
        with open("settings.json") as f:
            return json.load(f)
    except Exception:
        return {"rsi_threshold": 50, "sma50": "above", "sma200": "above"}

def build_filters(s: dict) -> list:
    filters = ["cap_midover", "sh_avgvol_o1000"]
    rsi = s.get("rsi_threshold", 50)
    filters.append(f"ta_rsi_u{rsi}")
    sma50 = s.get("sma50", "above")
    if sma50 == "above":
        filters.append("ta_sma50_pa")
    elif sma50 == "below":
        filters.append("ta_sma50_pb")
    sma200 = s.get("sma200", "above")
    if sma200 == "above":
        filters.append("ta_sma200_pa")
    elif sma200 == "below":
        filters.append("ta_sma200_pb")
    return filters

SETTINGS   = load_settings()
FILTERS    = build_filters(SETTINGS)
PERIOD     = "6mo"
MAX_CHARTS = 100
PAGE_SIZE  = 12
COLS       = 3
# ==============================

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]


# ---------- Telegram helpers ----------

def send_message(text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": text,
                              "parse_mode": "HTML"}, timeout=30)


def send_photo(image_bytes: bytes, caption: str = ""):
    # Telegram caption limit for photos is 1024 characters
    if len(caption) > 1024:
        caption = caption[:1021] + "..."
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    resp = requests.post(
        url,
        data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"},
        files={"photo": ("chart.png", image_bytes, "image/png")},
        timeout=60,
    )
    if not resp.ok:
        print(f"  ⚠️  Telegram sendPhoto error {resp.status_code}: {resp.text}")
        # Fallback: send as document so image is never lost silently
        doc_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        resp2 = requests.post(
            doc_url,
            data={"chat_id": CHAT_ID, "caption": caption[:1024], "parse_mode": "HTML"},
            files={"document": ("chart.png", image_bytes, "image/png")},
            timeout=60,
        )
        if not resp2.ok:
            print(f"  ⚠️  Telegram sendDocument fallback error {resp2.status_code}: {resp2.text}")


# ---------- Chart helpers ----------

def make_style():
    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge="inherit", wick="inherit", volume="inherit",
    )
    return mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        gridcolor="#2a2e39",
        facecolor="#131722",
        figcolor="#131722",
        rc={
            "axes.labelcolor": "#d1d4dc",
            "xtick.color": "#787b86",
            "ytick.color": "#787b86",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.titlecolor": "#d1d4dc",
            "axes.titlesize": 9,
        },
    )


def render_chart(ticker: str, style) -> tuple[str, bytes] | None:
    """Returns (ticker, image_bytes) or None on failure."""
    try:
        data = yf.download(ticker, period=PERIOD, interval="1d",
                           progress=False, auto_adjust=True)
        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
        data.index = pd.DatetimeIndex(data.index)

        close = data["Close"]
        plots = []
        if close.rolling(50).mean().notna().any():
            plots.append(mpf.make_addplot(close.rolling(50).mean(),
                                          color="#2962ff", width=1.0))
        if close.rolling(200).mean().notna().any():
            plots.append(mpf.make_addplot(close.rolling(200).mean(),
                                          color="#ff6d00", width=1.0))

        kwargs = dict(type="candle", style=style, volume=True,
                      title=f"\n{ticker}", figsize=(5, 3.2),
                      tight_layout=True, returnfig=True)
        if plots:
            kwargs["addplot"] = plots

        fig, _ = mpf.plot(data, **kwargs)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90,
                    bbox_inches="tight", facecolor="#131722")
        plt.close(fig)
        buf.seek(0)
        return ticker, buf.read()

    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return None


def build_grid(batch: list[tuple[str, bytes]]) -> bytes:
    """Combine multiple chart images into one grid image."""
    n    = len(batch)
    rows = max(1, (n + COLS - 1) // COLS)

    fig, axes = plt.subplots(rows, COLS,
                             figsize=(COLS * 5.5, rows * 3.5),
                             facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # נרמול axes
    if rows == 1 and COLS == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif COLS == 1:
        axes = [[ax] for ax in axes]

    for i, (ticker, img_bytes) in enumerate(batch):
        r, c = i // COLS, i % COLS
        img_arr = plt.imread(io.BytesIO(img_bytes))
        axes[r][c].imshow(img_arr)
        axes[r][c].axis("off")

    for i in range(n, rows * COLS):
        r, c = i // COLS, i % COLS
        axes[r][c].set_visible(False)

    plt.subplots_adjust(hspace=0.06, wspace=0.04)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------- Caption builder ----------

def build_caption(page: list[tuple[str, bytes]], rows_map: dict,
                  idx: int, total_pages: int) -> str:
    """Build an HTML caption with TradingView links for each ticker in the page.
    Header is sent separately to stay under Telegram's 1024-char caption limit.
    """
    MAX = 1024
    page_line = f"עמוד {idx}/{total_pages}\n"
    links: list[str] = []

    for ticker, _ in page:
        url = f"https://www.tradingview.com/chart/?symbol={ticker}"
        row = rows_map.get(ticker, {})
        price = row.get("Price") or row.get("price") or "—"
        rsi   = row.get("RSI") or row.get("RSI (14)") or row.get("Relative Strength Index") or "—"
        link  = f'<a href="{url}">{ticker}</a> ${price} {rsi}'

        candidate = page_line + " · ".join(links + [link])
        if len(candidate) > MAX:
            break
        links.append(link)

    return page_line + " · ".join(links)


# ---------- Main ----------

def main():
    today = date.today().strftime("%d/%m/%Y")
    print(f"📅 {today} — מריץ סקרינר...")

    # 1. Finviz screener (table=Technical for RSI + price data)
    screener     = Screener(filters=FILTERS, table="Technical", order="ticker")

    # Debug: print the actual keys and first few rows from finviz
    if hasattr(screener, 'data') and screener.data:
        print(f"🔍 finviz row keys: {list(screener.data[0].keys())}")
        for i, r in enumerate(screener.data[:3]):
            print(f"   row[{i}]: {dict(r)}")
    if hasattr(screener, 'headers'):
        print(f"🔍 finviz headers: {screener.headers}")

    all_rows     = list(screener)
    total_raw    = len(all_rows)

    # Deduplicate — finviz sometimes returns the same ticker on multiple pages
    seen: set[str] = set()
    unique_rows: list = []
    for r in all_rows:
        t = r.get("Ticker") or r.get("ticker") or ""
        if t and t not in seen:
            seen.add(t)
            unique_rows.append(r)

    total        = len(unique_rows)
    rows         = unique_rows[:MAX_CHARTS]

    def get_ticker(r):
        return r.get("Ticker") or r.get("ticker") or ""

    tickers      = [get_ticker(r) for r in rows]

    print(f"✅ נמצאו {total_raw} שורות גולמיות → {total} ייחודיות | מציג {len(tickers)}")
    if tickers:
        print(f"   דוגמה: {tickers[:5]}")

    if not tickers:
        send_message(f"📊 <b>Stock Screener — {today}</b>\n\nלא נמצאו מניות לפי הפילטרים.")
        return

    rows_map = {get_ticker(r): r for r in rows}

    # 2. Render charts
    style = make_style()
    charts = []
    for ticker in tickers:
        print(f"  📊 {ticker}...")
        result = render_chart(ticker, style)
        if result:
            charts.append(result)

    if not charts:
        send_message(f"📊 <b>Stock Screener — {today}</b>\n\n⚠️ לא הצלחתי לרנדר גרפים.")
        return

    # 3. Send header message (separately — caption limit is 1024 chars)
    send_message(
        f"📊 <b>Stock Screener — {today}</b>\n"
        f"✅ {total} מניות | מוצגות {len(charts)} | 🔵SMA50 🟠SMA200"
    )

    # 4. Send grid pages with linked captions
    pages = [charts[i:i + PAGE_SIZE] for i in range(0, len(charts), PAGE_SIZE)]
    for idx, page in enumerate(pages, 1):
        print(f"  📤 שולח עמוד {idx}/{len(pages)}...")
        grid_bytes = build_grid(page)
        caption    = build_caption(page, rows_map, idx, len(pages))
        send_photo(grid_bytes, caption)

    print("✅ הושלם!")


if __name__ == "__main__":
    main()
