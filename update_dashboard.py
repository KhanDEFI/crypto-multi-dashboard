import requests
import json
import os
import re
import time
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY")

CRYPTO_ASSETS = {
    "btc": {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin"},
    "eth": {"id": "ethereum", "symbol": "ETH", "name": "Ethereum"},
    "sui": {"id": "sui", "symbol": "SUI", "name": "SUI"},
}

# ── History / Accuracy Settings ──────────────────────────
HISTORY_DIR = "data/history"
ACCURACY_FILE = "data/accuracy.json"
MAX_HISTORY_ENTRIES = 168  # 7 days of hourly snapshots
LOOKBACK_HOURS = 24        # Ideal age of a prediction when we score it
LOOKBACK_MIN_HOURS = 20    # Accept predictions from 20h old ...
LOOKBACK_MAX_HOURS = 28    # ... up to 28h old (GitHub run times drift, see F-12)

# ── YouTube Feed Settings ────────────────────────────────
YOUTUBE_CHANNEL_ID = "UCngIhBkikUe6e7tZTjpKK7Q"  # More Crypto Online
YOUTUBE_RSS_URL = f"https://www.youtube.com/feeds/videos.xml?channel_id={YOUTUBE_CHANNEL_ID}"
YOUTUBE_FILE = "data/youtube.json"
YOUTUBE_MAX_VIDEOS = 7  # Keep the last 7 uploads

# ── X Community Consensus Settings ───────────────────────
RSSHUB_URL = os.environ.get("RSSHUB_URL", "https://rsshub.example.app")
X_DATA_DIR = "data/x"
X_POSTS_FILE = "data/x/posts.json"
X_ANALYSIS_FILE = "data/x/analysis.json"
X_CONSENSUS_HISTORY_FILE = "data/x/consensus_history.json"
X_POSTS_SINCE = "2026-01-01T00:00:00+00:00"  # Collect from Jan 1 2026

# ── Curated X handles — UPDATE THESE WITH YOUR LIST ──────
X_HANDLES = [
    # "handle1",       # Uncomment and replace with your handles
    # "handle2",
    # "handle3",
    # Add more handles here — no @ prefix needed
]

# Keywords to match video titles to dashboard assets
ASSET_KEYWORDS = {
    "btc": ["bitcoin", "btc"],
    "eth": ["ethereum", "eth"],
    "sui": ["sui"],
}


def coingecko_get(path, params=None, retries=3):
    """GET from CoinGecko using the Demo key, retrying on 429 rate limits."""
    url = f"https://api.coingecko.com/api/v3/{path}"
    headers = {"x-cg-demo-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
    for attempt in range(1, retries + 1):
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 429 and attempt < retries:
            wait = 10 * attempt
            print(f"    CoinGecko rate limit (429) - waiting {wait}s, retry {attempt}/{retries - 1}...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()


def fetch_crypto_prices():
    ids = ",".join([a["id"] for a in CRYPTO_ASSETS.values()])
    return coingecko_get("simple/price", {
        "ids": ids,
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_market_cap": "true",
    })


def fetch_crypto_ohlc(coin_id):
    return coingecko_get(f"coins/{coin_id}/ohlc", {"vs_currency": "usd", "days": 14})


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
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 1500,
        "temperature": 0.3,
        "reasoning_effort": "low",
        "include_reasoning": False,
        "response_format": {"type": "json_object"},
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


def find_snapshot_to_score(history, now, already_scored=()):
    """
    Pick the prediction to score this run: the snapshot whose age is closest to
    LOOKBACK_HOURS, as long as it is between LOOKBACK_MIN_HOURS and LOOKBACK_MAX_HOURS
    old and hasn't been scored yet. The wide window copes with GitHub's irregular
    run times (see F-12).
    """
    best = None
    best_delta = None
    for snap in history:
        if snap.get("timestamp") in already_scored:
            continue
        try:
            snap_time = datetime.fromisoformat(snap["timestamp"])
        except (ValueError, KeyError, TypeError):
            continue
        age_hours = (now - snap_time).total_seconds() / 3600
        if LOOKBACK_MIN_HOURS <= age_hours <= LOOKBACK_MAX_HOURS:
            delta = abs(age_hours - LOOKBACK_HOURS)
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

        if asset_key not in all_current_data:
            # No fresh data for this asset this run: keep its previous scores untouched
            print(f"  {asset_key.upper()} accuracy: skipped (no fresh data this run)")
            try:
                with open(ACCURACY_FILE, "r") as f:
                    accuracy[asset_key] = json.load(f).get(asset_key, {"evaluations": [], "stats": {}})
            except (IOError, json.JSONDecodeError):
                accuracy[asset_key] = {"evaluations": [], "stats": {}}
            continue

        current_price = all_current_data[asset_key]["price"]

        # Load existing accuracy results so we accumulate over time
        existing_accuracy = {}
        if os.path.exists(ACCURACY_FILE):
            try:
                with open(ACCURACY_FILE, "r") as f:
                    existing_accuracy = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_accuracy = {}

        prev_evals = existing_accuracy.get(asset_key, {}).get("evaluations", [])
        already_scored = {e["prediction_time"] for e in prev_evals}

        # Find a prediction made 20-28h ago that hasn't been scored yet (see F-12)
        old_snap = find_snapshot_to_score(history, now, already_scored)

        if old_snap:
            evaluation = evaluate_prediction(old_snap, current_price)
            if evaluation:
                hours = (now - datetime.fromisoformat(old_snap["timestamp"])).total_seconds() / 3600
                evaluation["hours_elapsed"] = round(hours, 1)
                prev_evals.append(evaluation)
                print(f"  {asset_key.upper()} accuracy: predicted {evaluation['predicted_bias']}, "
                      f"actual {evaluation['actual_direction']} → {evaluation['verdict']} "
                      f"({evaluation['price_change_pct']:+.2f}% after {evaluation['hours_elapsed']}h)")
            else:
                print(f"  {asset_key.upper()} accuracy: could not evaluate (price was 0)")
        else:
            print(f"  {asset_key.upper()} accuracy: no unscored prediction from "
                  f"{LOOKBACK_MIN_HOURS}-{LOOKBACK_MAX_HOURS}h ago")
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
#  X COMMUNITY CONSENSUS — NEW
# ══════════════════════════════════════════════════════════

def classify_post_assets(text):
    """Detect which crypto assets are mentioned in a post."""
    text_lower = text.lower()
    found = []
    for asset_key, keywords in ASSET_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                found.append(asset_key)
                break
    return found


def fetch_x_posts():
    """
    Fetch posts from all curated X handles via self-hosted RSSHub.
    Stores posts in a single JSON file, deduplicating by post URL.
    """
    if not X_HANDLES:
        print("\n  No X handles configured — skipping X feed fetch.")
        print("  Edit X_HANDLES in update_dashboard.py to add your curated accounts.")
        return

    if RSSHUB_URL == "https://rsshub.example.app":
        print("\n  RSSHUB_URL not configured — skipping X feed fetch.")
        print("  Set the RSSHUB_URL secret in GitHub Actions or env var.")
        return

    os.makedirs(X_DATA_DIR, exist_ok=True)
    print(f"\nFetching X posts from {len(X_HANDLES)} handles via RSSHub...")

    # Load existing posts
    existing_posts = []
    existing_urls = set()
    if os.path.exists(X_POSTS_FILE):
        try:
            with open(X_POSTS_FILE, "r") as f:
                data = json.load(f)
                existing_posts = data.get("posts", [])
                existing_urls = {p["url"] for p in existing_posts}
        except (json.JSONDecodeError, IOError):
            existing_posts = []
            existing_urls = set()

    cutoff = datetime.fromisoformat(X_POSTS_SINCE)
    new_count = 0

    for handle in X_HANDLES:
        feed_url = f"{RSSHUB_URL}/twitter/user/{handle}/exclude_rts"
        print(f"  Fetching @{handle}...")

        try:
            r = requests.get(feed_url, timeout=20)
            r.raise_for_status()
        except Exception as e:
            print(f"    Failed: {e}")
            continue

        # RSSHub returns RSS 2.0 XML
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError as e:
            print(f"    XML parse error: {e}")
            continue

        # Try Atom namespace first, then plain RSS
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item")  # RSS 2.0
        if not items:
            items = root.findall(".//atom:entry", ns)  # Atom

        for item in items:
            # RSS 2.0 fields
            title_el = item.find("title")
            desc_el = item.find("description")
            link_el = item.find("link")
            pubdate_el = item.find("pubDate")

            # Atom fallback
            if title_el is None:
                title_el = item.find("atom:title", ns)
            if link_el is None:
                link_atom = item.find("atom:link", ns)
                link_text = link_atom.get("href", "") if link_atom is not None else ""
            else:
                link_text = link_el.text or ""

            if desc_el is None:
                desc_el = item.find("atom:content", ns)
                if desc_el is None:
                    desc_el = item.find("atom:summary", ns)

            title_text = title_el.text if title_el is not None else ""
            desc_text = desc_el.text if desc_el is not None else ""

            # Combine title and description for full post text
            # RSSHub often puts the tweet text in description
            post_text = desc_text or title_text or ""

            # Strip HTML tags from post text
            post_text_clean = re.sub(r'<[^>]+>', ' ', post_text).strip()
            post_text_clean = re.sub(r'\s+', ' ', post_text_clean)

            if not link_text or link_text in existing_urls:
                continue

            # Parse date
            pub_date = None
            if pubdate_el is not None and pubdate_el.text:
                try:
                    from email.utils import parsedate_to_datetime
                    pub_date = parsedate_to_datetime(pubdate_el.text)
                except Exception:
                    pub_date = None
            else:
                pub_el = item.find("atom:published", ns)
                if pub_el is not None and pub_el.text:
                    try:
                        pub_date = datetime.fromisoformat(pub_el.text)
                    except Exception:
                        pub_date = None

            if pub_date is None:
                continue

            # Make timezone-aware if needed
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)

            # Skip posts before our cutoff
            if pub_date < cutoff:
                continue

            # Classify which assets this post mentions
            assets = classify_post_assets(post_text_clean)

            post_entry = {
                "handle": handle,
                "text": post_text_clean[:500],  # Cap at 500 chars
                "published": pub_date.isoformat(),
                "url": link_text,
                "assets": assets,
            }

            existing_posts.append(post_entry)
            existing_urls.add(link_text)
            new_count += 1

        print(f"    @{handle} done")

    # Sort by published date (newest first for convenience)
    existing_posts.sort(key=lambda p: p.get("published", ""), reverse=True)

    output = {
        "handles": X_HANDLES,
        "total_posts": len(existing_posts),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "posts": existing_posts,
    }

    with open(X_POSTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Stored {len(existing_posts)} total posts ({new_count} new)")
    return existing_posts


def analyze_x_consensus(all_current_prices):
    """
    Run AI analysis on recent X posts to determine community consensus
    for each asset. Saves results + appends to consensus history.
    """
    if not os.path.exists(X_POSTS_FILE):
        print("  No X posts to analyze yet.")
        return

    with open(X_POSTS_FILE, "r") as f:
        data = json.load(f)

    all_posts = data.get("posts", [])
    if not all_posts:
        print("  No posts available for consensus analysis.")
        return

    now = datetime.now(timezone.utc)

    # Get posts from the last 48h for current consensus
    cutoff_recent = now - timedelta(hours=48)
    recent_posts = []
    for p in all_posts:
        try:
            ptime = datetime.fromisoformat(p["published"])
            if ptime >= cutoff_recent:
                recent_posts.append(p)
        except (ValueError, KeyError):
            continue

    if not recent_posts:
        print("  No recent posts (48h) for consensus — skipping analysis.")
        return

    print(f"  Analyzing {len(recent_posts)} recent posts for consensus...")

    # Group recent posts by asset
    # General posts (no specific coin mentioned) go into ALL asset buckets
    asset_posts = {"btc": [], "eth": [], "sui": []}
    general_posts = []
    for p in recent_posts:
        matched_assets = [a for a in p.get("assets", []) if a in asset_posts]
        if matched_assets:
            for a in matched_assets:
                asset_posts[a].append(p)
        else:
            # General market post — feed into all asset buckets
            general_posts.append(p)
            for a in asset_posts:
                asset_posts[a].append(p)

    if general_posts:
        print(f"    {len(general_posts)} general market posts factored into all assets")

    consensus = {}
    by_profile = {}

    for asset_key in ["btc", "eth", "sui"]:
        posts = asset_posts[asset_key]
        if not posts:
            consensus[asset_key] = {
                "bias": "no_data",
                "confidence": 0,
                "post_count": 0,
                "summary": "No recent posts about this asset.",
                "voices": [],
            }
            continue

        # Build a digest of posts for the AI
        digest_lines = []
        for p in posts[:20]:  # Cap at 20 posts per asset
            handle = p["handle"]
            text = p["text"][:200]
            digest_lines.append(f"@{handle}: {text}")

        digest = "\n".join(digest_lines)

        asset_name = CRYPTO_ASSETS.get(asset_key, {}).get("name", asset_key.upper())
        current_price = all_current_prices.get(asset_key, {}).get("price", 0)

        prompt = (
            f"You are a crypto market sentiment analyst. "
            f"Analyze the following {len(posts)} recent posts from curated crypto analysts. "
            f"Some posts mention {asset_name} specifically, others are general market commentary "
            f"that should be considered as indirect sentiment for {asset_name}. "
            f"Current {asset_name} price: ${current_price:,.2f}. "
            f"Determine the community CONSENSUS for {asset_name}.\n\n"
            f"Posts:\n{digest}\n\n"
            f"Return ONLY a valid JSON object. No markdown, no explanation:\n"
            '{\n'
            '  "bias": "bullish | bearish | neutral | mixed",\n'
            '  "confidence": 0-100,\n'
            '  "summary": "2-3 sentence consensus summary",\n'
            '  "key_themes": ["theme1", "theme2"],\n'
            '  "dissenting_view": "brief note on any contrarian voice, or null"\n'
            '}'
        )

        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.3,
            }
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=body, timeout=20,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            result = parse_ai_json(content)
        except Exception as e:
            print(f"    AI analysis failed for {asset_key}: {e}")
            result = {
                "bias": "error",
                "confidence": 0,
                "summary": f"Analysis failed: {e}",
                "key_themes": [],
                "dissenting_view": None,
            }

        result["post_count"] = len(posts)
        # Track which handles contributed
        voices = list(set(p["handle"] for p in posts))
        result["voices"] = voices
        consensus[asset_key] = result

        print(f"    {asset_key.upper()}: {result.get('bias', '?')} "
              f"({result.get('confidence', 0)}% confidence, "
              f"{len(posts)} posts, {len(voices)} voices)")

    # Build per-profile summary
    for p in recent_posts:
        handle = p["handle"]
        if handle not in by_profile:
            by_profile[handle] = {
                "handle": handle,
                "post_count": 0,
                "assets_mentioned": [],
                "latest_post": None,
                "latest_text": "",
            }
        by_profile[handle]["post_count"] += 1
        by_profile[handle]["assets_mentioned"] = list(
            set(by_profile[handle]["assets_mentioned"] + p.get("assets", []))
        )
        # Track latest post
        if (by_profile[handle]["latest_post"] is None
                or p["published"] > by_profile[handle]["latest_post"]):
            by_profile[handle]["latest_post"] = p["published"]
            by_profile[handle]["latest_text"] = p["text"][:200]

    # ── Append to consensus history ──
    history = []
    if os.path.exists(X_CONSENSUS_HISTORY_FILE):
        try:
            with open(X_CONSENSUS_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    history_entry = {
        "timestamp": now.isoformat(),
        "consensus": {
            k: {"bias": v.get("bias"), "confidence": v.get("confidence", 0)}
            for k, v in consensus.items()
        },
        "prices": {
            k: v.get("price", 0) for k, v in all_current_prices.items()
            if k in ["btc", "eth", "sui"]
        },
    }
    history.append(history_entry)

    # Keep max 720 entries (~30 days hourly)
    history = history[-720:]
    with open(X_CONSENSUS_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    # ── Save current analysis ──
    analysis_output = {
        "updated_at": now.isoformat(),
        "consensus": consensus,
        "by_profile": by_profile,
        "recent_post_count": len(recent_posts),
        "total_posts_stored": len(all_posts),
        "handles_tracked": X_HANDLES,
    }

    with open(X_ANALYSIS_FILE, "w") as f:
        json.dump(analysis_output, f, indent=2)

    print("  Consensus analysis complete.")
    return analysis_output


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

def unavailable_analysis(reason):
    """Placeholder used when the AI call fails, so the price and RSI still get saved."""
    return {
        "elliott_wave": "AI analysis unavailable this hour.",
        "momentum": "AI analysis unavailable this hour.",
        "key_levels": {"support": [], "resistance": []},
        "trend_bias": "unavailable",
        "trade_plan": "AI analysis unavailable this hour. Price and RSI above are still live.",
        "risk_note": f"AI error: {str(reason)[:150]}",
    }


def safe_ai_analysis(*args, **kwargs):
    """Call the AI. If it fails, return the placeholder instead of crashing the run."""
    try:
        return get_ai_analysis(*args, **kwargs), True
    except Exception as e:
        print(f"  ⚠ AI analysis failed: {e}")
        return unavailable_analysis(e), False


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    all_current = {}
    succeeded = []   # assets saved with fresh data
    ai_failed = []   # assets saved, but without an AI plan
    failed = {}      # assets skipped completely -> error message

    # ── Crypto prices (one call for all 3 coins) ──
    print("Fetching crypto prices...")
    try:
        prices = fetch_crypto_prices()
    except Exception as e:
        print(f"  ⚠ Crypto price fetch failed: {e}")
        prices = {}

    # ── Each crypto asset gets its own safety net ──
    for key, asset in CRYPTO_ASSETS.items():
        print(f"Processing {asset['symbol']}...")
        try:
            coin = prices.get(asset["id"])
            if not coin or not coin.get("usd"):
                raise ValueError("no price returned by CoinGecko")
            price = coin["usd"]
            change = coin.get("usd_24h_change") or 0
            mcap = coin.get("usd_market_cap") or 0

            try:
                ohlc = fetch_crypto_ohlc(asset["id"])
                rsi = calculate_rsi(ohlc)
            except Exception as e:
                print(f"  ⚠ RSI unavailable for {asset['symbol']}: {e}")
                rsi = None

            analysis, ai_ok = safe_ai_analysis(asset["symbol"], price, change, mcap, rsi)

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
            print(f"  {asset['symbol']} done - ${price:,.2f}" + ("" if ai_ok else " (no AI plan)"))

            # Only real AI predictions go into history, so accuracy is never scored on a placeholder
            if ai_ok:
                save_history_snapshot(key, output)
            else:
                ai_failed.append(key)
            all_current[key] = output
            succeeded.append(key)
        except Exception as e:
            print(f"  ✗ {asset['symbol']} skipped: {e}")
            failed[key] = str(e)

    # ── Gold gets its own safety net ──
    print("Processing XAU (Gold) via TradingView...")
    try:
        gold = fetch_gold_tradingview()
        gold_price = gold["price"]
        gold_change = gold["change_pct"]
        gold_rsi = gold["rsi"]

        extra = f"- TradingView Signal: {gold['tv_recommendation']}"
        if gold["support"]:
            extra += "\n- TV Pivot Support: " + ", ".join([f"${s:,.2f}" for s in gold["support"]])
        if gold["resistance"]:
            extra += "\n- TV Pivot Resistance: " + ", ".join([f"${r:,.2f}" for r in gold["resistance"]])

        analysis, ai_ok = safe_ai_analysis(
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
        print(f"  XAU done - ${gold_price:,.2f} | RSI: {gold_rsi} | TV: {gold['tv_recommendation']}"
              + ("" if ai_ok else " (no AI plan)"))

        if ai_ok:
            save_history_snapshot("xau", output)
        else:
            ai_failed.append("xau")
        all_current["xau"] = output
        succeeded.append("xau")
    except Exception as e:
        print(f"  ✗ XAU skipped: {e}")
        failed["xau"] = str(e)

    # ── Accuracy, YouTube: each on its own safety net ──
    print("\nEvaluating prediction accuracy...")
    try:
        run_accuracy_evaluation(all_current)
    except Exception as e:
        print(f"  ⚠ Accuracy evaluation failed: {e}")

    try:
        fetch_youtube_feed()
    except Exception as e:
        print(f"  ⚠ YouTube feed failed: {e}")

    # ── Fetch X posts & analyze consensus (parked feature, unchanged) ──
    fetch_x_posts()
    print("\nAnalyzing X community consensus...")
    analyze_x_consensus(all_current)

    # ── Run summary ──
    print("\n══ Run summary ══")
    print(f"  Saved:      {', '.join(a.upper() for a in succeeded) or 'none'}")
    if ai_failed:
        print(f"  No AI plan: {', '.join(a.upper() for a in ai_failed)}")
        print(f"::warning title=AI analysis unavailable::{', '.join(a.upper() for a in ai_failed)} saved without an AI plan")
    for a, err in failed.items():
        print(f"  Failed:     {a.upper()} - {err}")
        print(f"::warning title={a.upper()} skipped::{err[:200]}")

    if not succeeded:
        print("\n✗ Every asset failed - marking this run as FAILED.")
        sys.exit(1)

    print("\nAll done.")

if __name__ == "__main__":
    main()
