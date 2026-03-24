import requests
import json
import os
from datetime import datetime, timezone

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

ASSETS = {
    "btc": {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin"},
    "eth": {"id": "ethereum", "symbol": "ETH", "name": "Ethereum"},
    "sui": {"id": "sui", "symbol": "SUI", "name": "SUI"},
}

def fetch_prices():
    ids = ",".join([a["id"] for a in ASSETS.values()])
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true&include_market_cap=true"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

def fetch_ohlc(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=14"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

def calculate_rsi(ohlc_data, period=14):
    closes = [candle[4] for candle in ohlc_data[-period * 2:]]
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

def get_ai_analysis(symbol, price, change_24h, market_cap, rsi):
    rsi_label = "neutral"
    if rsi:
        if rsi >= 70:
            rsi_label = "overbought"
        elif rsi <= 30:
            rsi_label = "oversold"

    prompt = f"""You are a professional crypto market analyst. Analyze {symbol}/USD and return ONLY a valid JSON object. No markdown, no explanation, just raw JSON.

Market data:
- Price: ${price:,.2f}
- 24h Change: {change_24h:.2f}%
- Market Cap: ${market_cap:,.0f}
- RSI (14): {rsi if rsi else 'N/A'} ({rsi_label})

Return this exact JSON structure:
{{
  "elliott_wave": "string — current wave count and what it implies (1-2 sentences)",
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
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.3
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body, timeout=20)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)

def main():
    os.makedirs("data", exist_ok=True)
    print("Fetching prices...")
    prices = fetch_prices()

    for key, asset in ASSETS.items():
        print(f"Processing {asset['symbol']}...")
        coin_data = prices.get(asset["id"], {})
        price = coin_data.get("usd", 0)
        change_24h = coin_data.get("usd_24h_change", 0)
        market_cap = coin_data.get("usd_market_cap", 0)

        ohlc = fetch_ohlc(asset["id"])
        rsi = calculate_rsi(ohlc)

        analysis = get_ai_analysis(asset["symbol"], price, change_24h, market_cap, rsi)

        output = {
            "symbol": asset["symbol"],
            "name": asset["name"],
            "price": price,
            "change_24h": round(change_24h, 2),
            "market_cap": market_cap,
            "rsi": rsi,
            "analysis": analysis,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        with open(f"data/{key}.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"  {asset['symbol']} done — ${price:,.2f}")

    print("All done.")

if __name__ == "__main__":
    main()
