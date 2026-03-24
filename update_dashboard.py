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

GOLD_ASSET = {"key": "xau", "symbol": "XAU", "name": "Gold"}

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


def fetch_gold_price():
    """Fetch XAU/USD from gold-api.com (free, no key needed)."""
    url = "https://api.gold-api.com/price/XAU"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    return {
        "price": data.get("price", 0),
        "ch": data.get("ch", 0),          # absolute change
        "chp": data.get("chp", 0),         # percent change
        "high": data.get("high_price", 0),
        "low": data.get("low_price", 0),
    }


# ── RSI calculation ──────────────────────────────────────────────
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
def get_ai_analysis(symbol, price, change_24h, market_cap, rsi, asset_type="crypto"):
    rsi_label = "neutral"
    if rsi:
        rsi_label = "overbought" if rsi >= 70 else ("oversold" if rsi <= 30 else "neutral")

    if asset_type == "commodity":
        context = f"""You are a professional commodities market analyst. Analyze {symbol}/USD (Gold spot) and return ONLY a valid JSON object. No markdown, no explanation, just raw JSON.

Market data:
- Price: ${price:,.2f} per troy ounce
- 24h Change: {change_24h:.2f}%
- RSI (14): {rsi if rsi else 'N/A'} ({rsi_label})

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
        "model": "llama3-8b-8192",
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

    # ── Gold (XAU) ───────────────────────────────────────────────
    print("Processing XAU (Gold)...")
    gold = fetch_gold_price()
    gold_price = gold["price"]
    gold_change = gold["chp"]

    # No OHLC candles from gold-api.com, so RSI is null for now
    gold_rsi = None

    analysis = get_ai_analysis(
        "XAU", gold_price, gold_change, 0, gold_rsi, asset_type="commodity"
    )

    output = {
        "symbol": "XAU",
        "name": "Gold",
        "price": gold_price,
        "change_24h": round(gold_change, 2),
        "market_cap": 0,
        "rsi": gold_rsi,
        "analysis": analysis,
        "updated_at": now,
    }
    with open("data/xau.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  XAU done — ${gold_price:,.2f}")

    print("All done.")


if __name__ == "__main__":
    main()
