import requests
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

CRYPTO_ASSETS = {
    "btc": {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin"},
    "eth": {"id": "ethereum", "symbol": "ETH", "name": "Ethereum"},
    "sui": {"id": "sui", "symbol": "SUI", "name": "SUI"},
}

# ── History / Accuracy Settings ──────────────────────────
HISTORY_DIR = "data/history"
ACCURACY_FILE = "data/accuracy.json"
MAX_HISTORY_ENTRIES = 168  # 7 days of hourly snapshots
LOOKBACK_HOURS = 24        # Evaluate predictions from 24h ago
LOOKBACK_TOLERANCE_MIN = 30  # ±30 min tolerance when finding the 24h-old snapshot

# ── YouTube Feed Settings ────────────────────────────────
YOUTUBE_CHANNEL_ID = "UCngIhBkikUe6e7tZTjpKK7Q"  # More Crypto Online
YOUTUBE_RSS_URL = f"https://www.youtube.com/feeds/videos.xml?channel_id={YOUTUBE_CHANNEL_ID}"
YOUTUBE_FILE = "data/youtube.json"
YOUTUBE_MAX_VIDEOS = 7  # Keep the last 7 uploads

# Keywords to match video titles to dashboard assets
ASSET_KEYWORDS = {
    "btc": ["bitcoin", "btc"],
    "eth": ["ethereum", "eth"],
    "sui": ["sui"],
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


def parse_ai_json(content):
    """Safely parse JSON from AI response, handling bad escapes."""
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        cleaned = re.sub(r'\\(?!["\\/bfnrtu])', '', content)
        return json.loads(cleaned)


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
    return parse_ai_json(content)


# ══════════════════════════════════════════════════════════
#  HISTORY & ACCURACY — NEW
# ══════════════════════════════════════════════════════════

def save_history_snapshot(asset_key, data):
    """Append a prediction snapshot to the asset's history file."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(HISTORY_DIR, f"{asset_key}.json")

    # Load existing history
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    # Build the snapshot — just the fields we need for evaluation
    snapshot = {
        "timestamp": data["updated_at"],
        "price": data["price"],
        "trend_bias": data["analysis"].get("trend_bias", "neutral"),
        "trade_plan": data["analysis"].get("trade_plan", ""),
        "key_levels": data["analysis"].get("key_levels", {}),
        "rsi": data.get("rsi"),
    }

    history.append(snapshot)

    # Trim to max entries (oldest first)
    if len(history) > MAX_HISTORY_ENTRIES:
        history = history[-MAX_HISTORY_ENTRIES:]

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    return history


def find_snapshot_near(history, target_time, tolerance_minutes=LOOKBACK_TOLERANCE_MIN):
    """Find the snapshot closest to target_time within tolerance."""
    best = None
    best_delta = None
    for snap in history:
        try:
            snap_time = datetime.fromisoformat(snap["timestamp"])
        except (ValueError, KeyError):
            continue
        delta = abs((snap_time - target_time).total_seconds())
        if delta <= tolerance_minutes * 60:
            if best_delta is None or delta < best_delta:
                best = snap
                best_delta = delta
    return best


def evaluate_prediction(old_snapshot, current_price):
    """
    Score a 24h-old prediction against current reality.

    Returns a dict with the evaluation results.
    """
    predicted_bias = old_snapshot.get("trend_bias", "neutral")
    old_price = old_snapshot["price"]

    if old_price == 0:
        return None

    price_change = current_price - old_price
    price_change_pct = round((price_change / old_price) * 100, 2)

    # Determine actual direction
    # Use a ±0.5% dead zone for "neutral"
    if price_change_pct > 0.5:
        actual_direction = "bullish"
    elif price_change_pct < -0.5:
        actual_direction = "bearish"
    else:
        actual_direction = "neutral"

    # Score the bias prediction
    if predicted_bias == actual_direction:
        verdict = "CORRECT"
    elif predicted_bias == "neutral" and abs(price_change_pct) < 2.0:
        verdict = "CORRECT"  # Neutral call when price barely moved = fair
    elif actual_direction == "neutral":
        verdict = "PARTIAL"  # Market was flat, any call is a wash
    else:
        verdict = "INCORRECT"

    # Check if price hit any predicted support/resistance
    key_levels = old_snapshot.get("key_levels", {})
    levels_hit = []

    def parse_price(s):
        """Extract number from '$68,000' style strings."""
        try:
            return float(str(s).replace("$", "").replace(",", ""))
        except (ValueError, TypeError):
            return None

    for lvl in key_levels.get("support", []):
        p = parse_price(lvl)
        if p and current_price <= p:
            levels_hit.append({"level": lvl, "type": "support", "hit": True})

    for lvl in key_levels.get("resistance", []):
        p = parse_price(lvl)
        if p and current_price >= p:
            levels_hit.append({"level": lvl, "type": "resistance", "hit": True})

    return {
        "prediction_time": old_snapshot["timestamp"],
        "price_at_prediction": old_price,
        "price_now": current_price,
        "price_change_pct": price_change_pct,
        "predicted_bias": predicted_bias,
        "actual_direction": actual_direction,
        "verdict": verdict,
        "trade_plan": old_snapshot.get("trade_plan", ""),
        "levels_hit": levels_hit,
    }


def run_accuracy_evaluation(all_current_data):
    """
    For each asset, find the ~24h-old snapshot, compare to current price,
    and build the accuracy results.
    """
    now = datetime.now(timezone.utc)
    target_time = now - timedelta(hours=LOOKBACK_HOURS)

    accuracy = {}

    for asset_key in list(CRYPTO_ASSETS.keys()) + ["xau"]:
        history_file = os.path.join(HISTORY_DIR, f"{asset_key}.json")
        if not os.path.exists(history_file):
            print(f"  No history for {asset_key} yet — skipping accuracy")
            accuracy[asset_key] = {"evaluations": [], "stats": {}}
            continue

        with open(history_file, "r") as f:
            history = json.load(f)

        current_price = all_current_data[asset_key]["price"]

        # Find the 24h-old snapshot
        old_snap = find_snapshot_near(history, target_time)

        # Load existing accuracy results so we accumulate over time
        existing_accuracy = {}
        if os.path.exists(ACCURACY_FILE):
            try:
                with open(ACCURACY_FILE, "r") as f:
                    existing_accuracy = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_accuracy = {}

        prev_evals = existing_accuracy.get(asset_key, {}).get("evaluations", [])

        if old_snap:
            evaluation = evaluate_prediction(old_snap, current_price)
            if evaluation:
                # Deduplicate — don't re-evaluate the same prediction timestamp
                existing_times = {e["prediction_time"] for e in prev_evals}
                if evaluation["prediction_time"] not in existing_times:
                    prev_evals.append(evaluation)
                    print(f"  {asset_key.upper()} accuracy: predicted {evaluation['predicted_bias']}, "
                          f"actual {evaluation['actual_direction']} → {evaluation['verdict']} "
                          f"({evaluation['price_change_pct']:+.2f}%)")
                else:
                    print(f"  {asset_key.upper()} accuracy: already evaluated this snapshot")
            else:
                print(f"  {asset_key.upper()} accuracy: could not evaluate (price was 0)")
        else:
            print(f"  {asset_key.upper()} accuracy: no 24h-old snapshot found yet")

        # Trim to last 30 evaluations (30 days if daily, ~30 entries)
        prev_evals = prev_evals[-30:]

        # Calculate running stats
        total = len(prev_evals)
        correct = sum(1 for e in prev_evals if e["verdict"] == "CORRECT")
        partial = sum(1 for e in prev_evals if e["verdict"] == "PARTIAL")
        incorrect = sum(1 for e in prev_evals if e["verdict"] == "INCORRECT")

        accuracy[asset_key] = {
            "evaluations": prev_evals,
            "stats": {
                "total": total,
                "correct": correct,
                "partial": partial,
                "incorrect": incorrect,
                "accuracy_pct": round((correct / total) * 100, 1) if total > 0 else 0,
                "updated_at": now.isoformat(),
            }
        }

    # Save accuracy file
    with open(ACCURACY_FILE, "w") as f:
        json.dump(accuracy, f, indent=2)

    print("  Accuracy evaluation complete.")
    return accuracy


# ══════════════════════════════════════════════════════════
#  YOUTUBE FEED — NEW
# ══════════════════════════════════════════════════════════

def classify_video(title):
    """Match a video title to one of our dashboard assets (btc, eth, sui)."""
    title_lower = title.lower()
    matched = []
    for asset_key, keywords in ASSET_KEYWORDS.items():
        for kw in keywords:
            # Use word boundary matching to avoid partial matches
            if re.search(r'\b' + re.escape(kw) + r'\b', title_lower):
                matched.append(asset_key)
                break
    return matched if matched else ["other"]


def fetch_youtube_feed():
    """Fetch the latest videos from the More Crypto Online YouTube RSS feed."""
    print("\nFetching YouTube feed from More Crypto Online...")

    try:
        r = requests.get(YOUTUBE_RSS_URL, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  Failed to fetch YouTube feed: {e}")
        return

    # Parse the Atom XML feed
    # YouTube uses Atom namespace
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015",
        "media": "http://search.yahoo.com/mrss/",
    }

    root = ET.fromstring(r.text)
    entries = root.findall("atom:entry", ns)

    videos = []
    for entry in entries[:YOUTUBE_MAX_VIDEOS]:
        video_id = entry.find("yt:videoId", ns)
        title = entry.find("atom:title", ns)
        published = entry.find("atom:published", ns)
        # Get the media:group -> media:thumbnail for the thumbnail URL
        media_group = entry.find("media:group", ns)
        thumbnail_url = ""
        description = ""
        if media_group is not None:
            thumb = media_group.find("media:thumbnail", ns)
            if thumb is not None:
                thumbnail_url = thumb.get("url", "")
            desc = media_group.find("media:description", ns)
            if desc is not None and desc.text:
                # Trim description to first 200 chars
                description = desc.text[:200].strip()

        if video_id is None or title is None:
            continue

        vid = video_id.text
        title_text = title.text or ""
        pub_text = published.text if published is not None else ""

        # Classify which asset(s) this video covers
        assets = classify_video(title_text)

        # Use high-quality thumbnail if available
        hq_thumb = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"

        videos.append({
            "id": vid,
            "title": title_text,
            "published": pub_text,
            "thumbnail": hq_thumb,
            "description": description,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "assets": assets,
        })

    # Save to JSON
    output = {
        "channel": "More Crypto Online",
        "channel_url": "https://www.youtube.com/@morecryptoonline",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "videos": videos,
    }

    with open(YOUTUBE_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(videos)} videos to {YOUTUBE_FILE}")
    for v in videos:
        print(f"    [{','.join(v['assets'])}] {v['title'][:60]}...")

    return output


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    all_current = {}

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

        # ── Save history snapshot ──
        save_history_snapshot(key, output)
        all_current[key] = output

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

    # ── Save XAU history snapshot ──
    save_history_snapshot("xau", output)
    all_current["xau"] = output

    # ── Run accuracy evaluation ──
    print("\nEvaluating prediction accuracy...")
    run_accuracy_evaluation(all_current)

    # ── Fetch YouTube feed ──
    fetch_youtube_feed()

    print("\nAll done.")


if __name__ == "__main__":
    main()
