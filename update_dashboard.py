import requests
import json
import os
from datetime import datetime, timezone

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

CRYPTO_ASSETS = {
    "btc": {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin"},
    "eth": {"id": "ethereum", "symbol": "ETH", "name": "Ethereum"},
    "sui": {"id": "sui", "symbol": "SUI", "name": "SUI"},
}


def fetch_crypto_prices():
    ids = ",".join([a["id"] for a in CRYPTO_ASSETS.values()])
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={ids}&vs_currencies=usd"
        "&include_24hr_change=true&include_market_cap=true"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_crypto_ohlc(coin_id):
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        "/ohlc?vs_currency=usd&days=14"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_gold_tradingview():
    from tradingview_ta import TA_Handler, Interval

    # Try multiple exchange/screener combos until one works
    combos = [
        {"symbol": "XAUUSD", "exchange": "CAPITALCOM", "screener": "cfd"},
        {"symbol": "XAUUSD", "exchange": "PEPPERSTONE", "screener": "cfd"},
        {"symbol": "XAUUSD", "exchange": "FXOPEN", "screener": "cfd"},
        {"symbol": "XAUUSD", "exchange": "OANDA", "screener": "cfd"},
        {"symbol": "XAUUSD", "exchange": "FX_IDC", "screener": "cfd"},
        {"symbol": "XAUUSD", "exchange": "FOREXCOM", "screener": "forex"},
        {"symbol": "XAUUSD", "exchange": "OANDA", "screener": "forex"},
    ]

    analysis = None
    for combo in combos:
        try:
            print(f"    Trying {combo['exchange']}:{combo['symbol']} ({combo['screener']})...")
            handler = TA_Handler(
                symbol=combo["symbol"],
                screener=combo["screener"],
                exchange=combo["exchange"],
                interval=Interval.INTERVAL_1_DAY,
            )
            analysis = handler.get_analysis()
            print(f"    Success with {combo['exchange']}")
            break
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    if analysis is None:
        raise Exception("All TradingView exchange combos failed for XAUUSD")

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


def get_ai_analysis(symbol, price, change_24h, market_cap, rsi, asset_type="crypto", extra_context=""):
    rsi_label = "neutral"
    if rsi:
        rsi_label = "overbought" if rsi >= 70 else ("oversold" if rsi <= 30 else "neutral")

    if asset_type == "commodity":
        context = (
            f"You are a professional commodities market analyst. "
            f"Analyze {symbol}/USD (Gold spot) and return ONLY a valid JSON object. "
            f"No markdown, no explanation, just raw JSON.\n\n"
            f"Market data:\n"
            f"- Price: ${price:,.2f} per troy ounce\n"
            f"- 24h Change: {change_24h:.2f}%\n"
            f"- RSI (14): {rsi if rsi else 'N/A'} ({rsi_label})\n"
            f"{extra_context}\n\n"
            f"Note: Gold does not follow Elliott Wave theory in the same way crypto does. "
            f"Focus on classical technical patterns, macro drivers (DXY, real yields, central bank policy), "
            f"and safe-haven demand."
        )
    else:
        context = (
            f"You are a professional crypto market analyst. "
            f"Analyze {symbol}/USD and return ONLY a valid JSON object. "
            f"No markdown, no explanation, just raw JSON.\n\n"
            f"Market data:\n"
            f"- Price: ${price:,.2f}\n"
            f"- 24h Change: {change_24h:.2f}%\n"
            f"- Market Cap: ${market_cap:,.0f}\n"
            f"- RSI (14): {rsi if rsi else 'N/A'} ({rsi_label})"
        )

    prompt = (
        f"{context}\n\n"
        'Return this exact JSON structure:\n'
        '{\n'
        '  "elliott_wave": "string - current wave count / pattern assessment (1-2 sentences)",\n'
        '  "momentum": "string - momentum assessment based on price action and RSI (1-2 sentences)",\n'
        '  "key_levels": {\n'
        '    "support": ["price1", "price2"],\n'
        '    "resistance": ["price1", "price2"]\n'
        '  },\n'
        '  "trend_bias": "bullish | bearish | neutral",\n'
        '  "trade_plan": "string - concise actionable plan for the next 24-48h (2-3 sentences)",\n'
        '  "risk_note": "string - one key risk to watch"\n'
        '}'
    )

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

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def main():
    os.makedirs("data", exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

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
        print(f"  {asset['symbol']} done - ${price:,.2f}")

    print("Processing XAU (Gold) via TradingView...")
    gold = fetch_gold_tradingview()
    gold_price = gold["price"]
    gold_change = gold["change_pct"]
    gold_rsi = gold["rsi"]

    extra = f"- TradingView Signal: {gold['tv_recommendation']}"
    if gold["support"]:
        extra += "\n- TV Pivot Support: " + ", ".join([f"${s:,.2f}" for s in gold["support"]])
    if gold["resistance"]:
        extra += "\n- TV Pivot Resistance: " + ", ".join([f"${r:,.2f}" for r in gold["resistance"]])

    analysis = get_ai_analysis(
        "XAU", gold_price, gold_change, 0, gold_rsi,
        asset_type="commodity", extra_context=extra
    )

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
    print(f"  XAU done - ${gold_price:,.2f} | RSI: {gold_rsi} | TV: {gold['tv_recommendation']}")

    print("All done.")


if __name__ == "__main__":
    main()
