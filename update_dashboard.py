import requests
import json
import os
from datetime import datetime, timezone

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ── Asset definitions ────────────────────────────────────────────
CRYPTO_ASSETS = {
    "btc": {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin"},
    "eth": {"id": "ethereum", "symbol": "ETH", "name": "Ethereum"},
    "sui": {"id": "sui", "symbol": "SUI", "name": "SUI"},
}

# ── Data fetchers ────────────────────────────────────────────────
def fetch_crypto_prices():
    ids = ",".join([a["id"] for a in CRYPTO_ASSETS.values()])
    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={ids}&vs_currencies=usd"
        f"&include_24hr_change=true&include_market_cap=true"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_crypto_ohlc(coin_id):
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        f"/ohlc?vs_currency=usd&days=14"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_gold_tradingview():
    """Fetch XAU/USD data from TradingView via tradingview_ta."""
    from tradingview_ta import TA_Handler, Interval

    handler = TA_Handler(
        symbol="XAUUSD",
        screener="forex",
        exchange="OANDA",
        interval=Interval.INTERVAL_1_DAY,
    )
    analysis = handler.get_analysis()
    indicators = analysis.indicators

    price = indicators.get("close", 0)
    change_pct = indicators.get("change", 0)
    rsi = indicators.get("RSI", None)
    high = indicators.get("high", 0)
    low = indicators.get("low", 0)

    support_1 = indicators.get("Pivot.M.Classic.S1", None)
    support_2 = indicators.get("Pivot.M.Classic.S2", None)
    resistance_1 = indicators.get("Pivot.M.Classic.R1", None)
    resistance_2 = indicators.get("Pivot.M.Classic.R2", None)

    summary = analysis.summary
    recommendation = summary.get("RECOMMENDATION", "NEUTRAL")

    if rsi is not None:
        rsi = round(rsi, 2)

    return {
        "price": price,
        "change_pct": round(change_pct, 2) if change_pct else 0,
        "rsi": rsi,
        "high": high,
        "low": low,
        "support": [s for s in [support_1, support_2] if s],
        "resistance": [r for r in [resistance_1, resistance_2] if r],
        "tv_recommendation": recommendation,
    }


# ── RSI calculation (for crypto) ─────────────────────────────────
def calculate_rsi(ohlc_data, period=14):
    closes = [candle[4] for candle in ohlc_data[-period * 2 :]]
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


# ── AI analysis ──────────────────────────────────────────────────
def get_ai_analysis(symbol, price, change_24h, market_cap, rsi, asset_type="crypto", extra_context=""):
    rsi_label = "neutral"
    if rsi:
        rsi_label = "overbought" if rsi >= 70 else ("oversold" if rsi <= 30 else "neutral")

    if asset_type == "commodity":
        context = f"""You are a professional commodities market analyst. Analyze {symbol}/USD (Gold spot) and return ONLY a valid JSON object. No markdown, no explanation, just raw JSON.

Market data:
- Price: ${price:,.2f} per troy ounce
- 24h Change: {change_24h:.2f}%
- RSI (14): {rsi if rsi else 'N/A'} ({rsi_label})
{extra_context}

Note: Gold does not follow Elliott Wave theory in the same way crypto does. Focus on classical technical patterns, macro drivers (DXY, real yields, central bank policy), and safe-haven demand."""
    else:
        context = f"""You are a professional crypto market analyst. Analyze {symbol}/USD and return ONLY a valid JSON object. No markdown, no explanation, just raw JSON.

Market data:
- Price: ${price:,.2f}
- 24h Change: {change_24h:.2f}%
- Market Cap: ${market_cap:,.0f}
- RSI (14): {rsi if rsi else 'N/A'} ({rsi_label})"""

    prompt = f"""{context}

Return this exact JSON structure:
{{
  "elliott_wave": "string — current wave count / pattern assessment and what it implies (1-2 sentences)",
  "momentum": "string — momentum assessment based on price action and RSI (1-2 sentences)",
  "key_levels": {{
    "support": ["price1", "price2"],
    "resistance": ["price1", "price2"]
  }},
  "trend_bias": "bullish | bearish | neutral",
  "trade_plan": "string — concise actionable plan for the next 24-48h (2-3 sentences)",
  "risk_note": "string — one key risk to watch"
}}"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.3,
    }
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=20,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


# ── Main pipeline ────────────────────────────────────────────────
def main():
    os.makedirs("data", exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    # ── Crypto assets ────────────────────────────────────────────
    print("Fetching crypto prices...")
    prices = fetch_crypto_prices()

    for key, asset in CRYPTO_ASSETS.items():
        print(f"Processing {asset['symbol']}...")
        coin = prices.get(asset["id"], {})
        price = coin.get("usd", 0)
        change = coin.get("usd_24h_change", 0)
        mcap = coin.get("usd_market_cap", 0)

        ohlc = fetch_crypto_ohlc(asset["id"])
        rsi = calculate_rsi(ohlc)

        analysis = get_ai_analysis(asset["symbol"], price, change, mcap, rsi)

        output = {
            "symbol": asset["symbol"],
            "name": asset["name"],
            "price": price,
            "change_24h": round(change, 2),
            "market_cap": mcap,
            "rsi": rsi,
            "analysis": analysis,
            "updated_at": now,
        }
        with open(f"data/{key}.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"  {asset['symbol']} done — ${price:,.2f}")

    # ── Gold (XAU) via TradingView ───────────────────────────────
    print("Processing XAU (Gold) via TradingView...")
    gold = fetch_gold_tradingview()
    gold_price = gold["price"]
    gold_change = gold["change_pct"]
    gold_rsi = gold["rsi"]

    # Build extra context from TradingView data
    extra = f"- TradingView Signal: {gold['tv_recommendation']}"
    if gold["support"]:
        extra += f"\n- TV Pivot Support: {', '.join([f'${s:,.2f}' for s in gold['support']])}"
    if gold["resistance"]:
        extra += f"\n- TV Pivot Resistance: {', '.join([f'${r:,.2f}' for r in gold['resistance']])}"

    analysis = get_ai_analysis(
        "XAU", gold_price, gold_change, 0, gold_rsi,
        asset_type="commodity", extra_context=extra
    )

    # Override AI key_levels with TradingView's actual pivot levels
    if gold["support"] or gold["resistance"]:
        analysis["key_levels"] = {
            "support": [f"${s:,.2f}" for s in gold["support"]],
            "resistance": [f"${r:,.2f}" for r in gold["resistance"]],
        }

    output = {
        "symbol": "XAU",
        "name": "Gold",
        "price": gold_price,
        "change_24h": gold_change,
        "market_cap": 0,
        "rsi": gold_rsi,
        "analysis": analysis,
        "updated_at": now,
    }
    with open("data/xau.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  XAU done — ${gold_price:,.2f} | RSI: {gold_rsi} | TV: {gold['tv_recommendation']}")

    print("All done.")


if __name__ == "__main__":
    main()
