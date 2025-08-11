# signals_app_pro.py
import math, re, json, datetime as dt
from pathlib import Path
from typing import List, Dict, Optional
import os

import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import pytz
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional: precise exchange calendars (NYSE/XETRA/LSE/Paris)
try:
    import pandas_market_calendars as mcal
except Exception:
    mcal = None  # fallback to simple weekday/time check if not installed

# Optional: auto refresh (every 30 min)
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

APP_TITLE = "📈 Stock Signals PRO (Free-first)"
WATCHLIST_FILE = Path.home() / "signals_watchlist.json"

# -------------------- Market hours profiles --------------------
MARKETS = {
    "US — NYSE/Nasdaq (09:30—16:00 ET)": {"tz": "America/New_York", "open": (9,30), "close": (16,0), "cal": "XNYS"},
    "Germany — XETRA (09:00—17:30 DE)": {"tz": "Europe/Berlin", "open": (9,0),  "close": (17,30), "cal": "XETR"},
    "UK — LSE (08:00—16:30 UK)":        {"tz": "Europe/London", "open": (8,0),  "close": (16,30), "cal": "XLON"},
    "France — Euronext Paris (09:00—17:30 FR)": {"tz": "Europe/Paris", "open": (9,0), "close": (17,30), "cal": "XPAR"},
}

# -------------------- Market open check --------------------
def is_market_open(profile_key: str) -> bool:
    """Return True only if today is a trading day and current time is within official session.
       Uses pandas_market_calendars if available; else falls back to weekday/time window."""
    prof = MARKETS.get(profile_key)
    if not prof:
        return False
    tz = pytz.timezone(prof["tz"])
    now = dt.datetime.now(tz)

    # Prefer precise exchange calendar
    if mcal is not None and prof.get("cal"):
        try:
            cal = mcal.get_calendar(prof["cal"])
            sched = cal.schedule(start_date=now.date(), end_date=now.date())
            if sched.empty:
                return False  # weekend/holiday
            open_ts = sched.iloc[0]["market_open"].tz_convert(tz)
            close_ts = sched.iloc[0]["market_close"].tz_convert(tz)
            return open_ts <= now < close_ts
        except Exception:
            pass  # fall back if calendar not available

    # Fallback: Mon—Fri and within static hours
    if now.weekday() > 4:
        return False
    o_h, o_m = prof["open"]; c_h, c_m = prof["close"]
    open_t = now.replace(hour=o_h, minute=o_m, second=0, microsecond=0)
    close_t = now.replace(hour=c_h, minute=c_m, second=0, microsecond=0)
    return open_t <= now < close_t

def now_local(tz_name: str) -> dt.datetime:
    tz = pytz.timezone(tz_name)
    return dt.datetime.now(tz)

# -------------------- Data helpers --------------------
def load_watchlist() -> List[str]:
    if WATCHLIST_FILE.exists():
        try:
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [x.strip().upper() for x in data if isinstance(x, str) and x.strip()]
        except Exception:
            pass
    return ["MSFT", "AAPL", "GOOGL"]

def save_watchlist(tickers: List[str]):
    tickers = sorted(set([x.strip().upper() for x in tickers if x.strip()]))
    WATCHLIST_FILE.write_text(json.dumps(tickers, indent=2), encoding="utf-8")

def ema(series: pd.Series, span: int) -> pd.Series:
    result = series.ewm(span=span, adjust=False).mean()
    return result if isinstance(result, pd.Series) else pd.Series(result)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).abs()
    loss = (-delta.where(delta < 0, 0.0)).abs()
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Convert to Series if needed before using shift
    if not isinstance(avg_gain, pd.Series):
        avg_gain = pd.Series(avg_gain)
    if not isinstance(avg_loss, pd.Series):
        avg_loss = pd.Series(avg_loss)
    
    avg_gain = avg_gain.shift(1).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = avg_loss.shift(1).ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    result = 100 - (100 / (1 + rs))
    return result if isinstance(result, pd.Series) else pd.Series(result)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal = ema(macd_line, sig)
    hist = macd_line - signal
    return macd_line, signal, hist

def get_price_history(ticker: str, days: int, interval: str = "1d") -> pd.DataFrame:
    # Yahoo limits: for 30m bars ~60 days max; for daily use days param.
    if interval == "30m":
        yf_period = "60d"
        yf_interval = "30m"
    else:
        yf_period = f"{days}d"
        yf_interval = "1d"

    df = yf.download(
        ticker, period=yf_period, interval=yf_interval,
        auto_adjust=True, progress=False, group_by="column", threads=False
    )
    if df is None or df.empty:
        raise RuntimeError(f"No price data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if (x is not None and str(x).strip())]).strip() for tup in df.columns]
    df = df.rename(columns=lambda c: str(c).strip())

    if "Close" not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        wanted = None
        for k_lower, orig in lower_map.items():
            if "close" in k_lower and ticker.lower() in k_lower:
                wanted = orig; break
        if wanted is None:
            for k_lower, orig in lower_map.items():
                if re.search(r"\bclose\b", k_lower):
                    wanted = orig; break
        if wanted is None:
            for k_lower, orig in lower_map.items():
                if "adj" in k_lower and "close" in k_lower:
                    wanted = orig; break
        if wanted is None:
            raise RuntimeError(f"Missing Close column. Got: {list(df.columns)}")
        df["Close"] = df[wanted]

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna(subset=["Close"])

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].copy()
    if not isinstance(c, pd.Series):
        c = pd.Series(c)
    
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["RSI"] = rsi(c, 14)
    macd_line, macd_sig, macd_hist = macd(c, 12, 26, 9)
    df["MACD"] = macd_line
    df["MACD_SIG"] = macd_sig
    df["MACD_HIST"] = macd_hist
    window_52w = min(len(df), 252)
    df["HI52"] = c.rolling(window_52w).max()
    df["LO52"] = c.rolling(window_52w).min()
    return df

def last_cross(a: pd.Series, b: pd.Series) -> Optional[str]:
    if len(a) < 2 or len(b) < 2: return None
    prev = a.iloc[-2] - b.iloc[-2]
    curr = a.iloc[-1] - b.iloc[-1]
    if pd.isna(prev) or pd.isna(curr): return None
    if prev < 0 and curr > 0: return "up"
    if prev > 0 and curr < 0: return "down"
    return None

def google_news_titles(query: str, days: int = 5) -> List[str]:
    url = f"https://news.google.com/rss/search?q={query}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return [e.get("title","") for e in feed.entries[:30]]

def score_sentiment(headlines: List[str]) -> Dict[str, float]:
    if not headlines:
        return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0, "n": 0}
    an = SentimentIntensityAnalyzer()
    comp, pos, neu, neg = [], [], [], []
    for h in headlines:
        s = an.polarity_scores(h)
        comp.append(s["compound"]); pos.append(s["pos"]); neu.append(s["neu"]); neg.append(s["neg"])
    return {
        "compound": float(np.mean(comp)),
        "pos": float(np.mean(pos)),
        "neu": float(np.mean(neu)),
        "neg": float(np.mean(neg)),
        "n": len(headlines)
    }

def classify_reasoned(ticker: str, df: pd.DataFrame, sent: Optional[Dict], risk="balanced", near=0.02) -> Dict:
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else row
    rsi_ob, rsi_os = 70, 30
    if risk == "conservative": rsi_ob -= 2; rsi_os += 2
    if risk == "aggressive":   rsi_ob += 2; rsi_os -= 2

    price = float(row["Close"])
    sma50 = float(row["SMA50"]) if not math.isnan(row["SMA50"]) else None
    sma200 = float(row["SMA200"]) if not math.isnan(row["SMA200"]) else None
    rsi_now = float(row["RSI"]) if not math.isnan(row["RSI"]) else None
    rsi_prev = float(prev["RSI"]) if not math.isnan(prev["RSI"]) else None
    hi52 = float(row["HI52"]) if not math.isnan(row["HI52"]) else None
    lo52 = float(row["LO52"]) if not math.isnan(row["LO52"]) else None

    sma50_series = df["SMA50"] if isinstance(df["SMA50"], pd.Series) else pd.Series(df["SMA50"])
    sma200_series = df["SMA200"] if isinstance(df["SMA200"], pd.Series) else pd.Series(df["SMA200"])
    macd_series = df["MACD"] if isinstance(df["MACD"], pd.Series) else pd.Series(df["MACD"])
    macd_sig_series = df["MACD_SIG"] if isinstance(df["MACD_SIG"], pd.Series) else pd.Series(df["MACD_SIG"])
    
    sma_cross = last_cross(sma50_series, sma200_series)
    macd_cross = last_cross(macd_series, macd_sig_series)

    trend_up = (sma50 and sma200 and sma50 > sma200 and price > sma50)
    trend_down = (sma50 and sma200 and sma50 < sma200 and price < sma50)

    reasons, score = [], 50
    if hi52 and price >= hi52*(1-near): reasons.append("Near 52w HIGH"); score += 5
    if lo52 and price <= lo52*(1+near): reasons.append("Near 52w LOW"); score -= 5

    if rsi_now is not None and rsi_prev is not None:
        if rsi_prev < 50 <= rsi_now: reasons.append("RSI crossed up 50"); score += 6
        if rsi_prev > 50 >= rsi_now: reasons.append("RSI crossed down 50"); score -= 6
        if rsi_now >= rsi_ob: reasons.append(f"RSI overbought ({rsi_now:.1f})"); score -= 8
        if rsi_now <= rsi_os: reasons.append(f"RSI oversold ({rsi_now:.1f})"); score += 8

    if sma_cross == "up": reasons.append("SMA50 > SMA200 (golden cross)"); score += 10
    if sma_cross == "down": reasons.append("SMA50 < SMA200 (death cross)"); score -= 10
    if macd_cross == "up": reasons.append("MACD crossed up"); score += 7
    if macd_cross == "down": reasons.append("MACD crossed down"); score -= 7

    if sent:
        sc = sent.get("compound", 0.0)
        if sc >= 0.2: reasons.append(f"Positive news ({sc:+.2f})"); score += 6
        elif sc <= -0.2: reasons.append(f"Negative news ({sc:+.2f})"); score -= 6

    score = max(0, min(100, score))
    label = "WATCH"
    if trend_up and score >= 60: label = "BUY"
    if trend_down and score <= 40: label = "SELL"
    if hi52 and rsi_now and price >= hi52*(1- near/2) and rsi_now >= (70 if risk!="conservative" else 68):
        label = "TAKE-PROFIT"
    if lo52 and rsi_now and price <= lo52*(1+ near/2) and rsi_now <= (30 if risk!="conservative" else 32):
        label = "WATCH (oversold bounce?)"

    return {
        "ticker": ticker, "label": label, "confidence": score, "price": price,
        "sma50": sma50, "sma200": sma200, "rsi": rsi_now, "hi52": hi52, "lo52": lo52,
        "reasons": reasons[:6], "sentiment": sent if sent else None
    }

def format_alert(res_list: List[Dict], tz_name: str) -> str:
    ts = now_local(tz_name).strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>Stock signals — {ts}</b>"]
    for r in res_list:
        price = f"{r.get('price', 0):.2f}" if r.get('price') is not None else "-"
        rsi = f"{r.get('rsi', 0):.1f}" if r.get('rsi') is not None else "-"
        conf = f"{r.get('confidence', 0):.0f}"
        reasons = "; ".join(r.get("reasons", [])) if r.get("reasons") else "-"
        lines.append(f"<b>{r['ticker']}</b> — <b>{r['label']}</b> (conf {conf}/100) @ {price} | RSI {rsi}\n• {reasons}")
    return "\n".join(lines)

def send_telegram(token: str, chat_id: str, text: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

def scan_tickers(tickers: List[str], lookback_days: int, news_days: int, use_news: bool,
                 risk: str, near: float, interval: str):
    results, rows = [], []
    for t in tickers:
        try:
            df = get_price_history(t, lookback_days, interval=interval)
            df = compute_indicators(df)
            sent = None
            if use_news:
                titles = google_news_titles(t, news_days)
                sent = score_sentiment(titles)
            res = classify_reasoned(t, df, sent, risk=risk, near=near)
            results.append(res)
            rows.append({
                "ticker": t, "label": res["label"], "confidence": res["confidence"],
                "price": res["price"], "RSI": res["rsi"], "SMA50": res["sma50"], "SMA200": res["sma200"],
                "HI52": res["hi52"], "LO52": res["lo52"], "reasons": " | ".join(res["reasons"])
            })
        except Exception as e:
            st.error(f"Error processing {t}: {str(e)}")
            continue
    return results, rows

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    st.title(APP_TITLE)
    
    # Auto-refresh setup
    if st_autorefresh:
        count = st_autorefresh(interval=30*60*1000, key="datarefresh")  # 30 minutes
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Market selection
    market_key = st.sidebar.selectbox("Market Profile:", list(MARKETS.keys()), index=0)
    market_info = MARKETS[market_key]
    
    # Market status
    is_open = is_market_open(market_key)
    tz_name = market_info["tz"]
    local_time = now_local(tz_name)
    
    status_color = "🟢" if is_open else "🔴"
    status_text = "OPEN" if is_open else "CLOSED"
    st.sidebar.markdown(f"**Market Status:** {status_color} {status_text}")
    st.sidebar.markdown(f"**Local Time:** {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Risk profile
    risk_profile = st.sidebar.selectbox("Risk Profile:", ["conservative", "balanced", "aggressive"], index=1)
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    lookback_days = st.sidebar.slider("Lookback Days:", 30, 365, 90)
    interval = st.sidebar.selectbox("Data Interval:", ["1d", "30m"], index=0)
    use_news = st.sidebar.checkbox("Include News Sentiment", value=True)
    news_days = st.sidebar.slider("News Days:", 1, 14, 5) if use_news else 5
    near_threshold = st.sidebar.slider("52W High/Low Threshold:", 0.01, 0.1, 0.02, 0.01)
    
    # Watchlist management
    st.sidebar.subheader("📋 Watchlist Management")
    current_watchlist = load_watchlist()
    
    # Add new ticker
    new_ticker = st.sidebar.text_input("Add Ticker:", placeholder="e.g., AAPL").strip().upper()
    if st.sidebar.button("Add") and new_ticker:
        if new_ticker not in current_watchlist:
            current_watchlist.append(new_ticker)
            save_watchlist(current_watchlist)
            st.sidebar.success(f"Added {new_ticker}")
            st.rerun()
        else:
            st.sidebar.warning(f"{new_ticker} already in watchlist")
    
    # Remove ticker
    if current_watchlist:
        ticker_to_remove = st.sidebar.selectbox("Remove Ticker:", [""] + current_watchlist)
        if st.sidebar.button("Remove") and ticker_to_remove:
            current_watchlist.remove(ticker_to_remove)
            save_watchlist(current_watchlist)
            st.sidebar.success(f"Removed {ticker_to_remove}")
            st.rerun()
    
    # Display current watchlist
    st.sidebar.markdown(f"**Current Watchlist:** {', '.join(current_watchlist)}")
    
    # Telegram notifications
    st.sidebar.subheader("📱 Telegram Notifications")
    telegram_token = st.sidebar.text_input("Bot Token:", type="password", 
                                          value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id = st.sidebar.text_input("Chat ID:", 
                                            value=os.getenv("TELEGRAM_CHAT_ID", ""))
    
    # Main content area
    if not current_watchlist:
        st.warning("No tickers in watchlist. Please add some tickers to analyze.")
        return
    
    # Scan button
    if st.button("🔍 Scan Stocks", type="primary"):
        with st.spinner("Analyzing stocks..."):
            results, rows = scan_tickers(
                current_watchlist, lookback_days, news_days, use_news,
                risk_profile, near_threshold, interval
            )
        
        if results:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            buy_signals = sum(1 for r in results if r["label"] == "BUY")
            sell_signals = sum(1 for r in results if r["label"] == "SELL")
            watch_signals = sum(1 for r in results if "WATCH" in r["label"])
            avg_confidence = np.mean([r["confidence"] for r in results])
            
            col1.metric("BUY Signals", buy_signals)
            col2.metric("SELL Signals", sell_signals)
            col3.metric("WATCH Signals", watch_signals)
            col4.metric("Avg Confidence", f"{avg_confidence:.1f}/100")
            
            # Results table
            st.subheader("📊 Analysis Results")
            
            if rows:
                df_results = pd.DataFrame(rows)
                
                # Format the dataframe for display
                for col in ["price", "SMA50", "SMA200", "HI52", "LO52"]:
                    if col in df_results.columns:
                        df_results[col] = df_results[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else "-")
                
                if "RSI" in df_results.columns:
                    df_results["RSI"] = df_results["RSI"].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x is not None else "-")
                
                st.dataframe(df_results, use_container_width=True)
            
            # Detailed results
            st.subheader("📈 Detailed Analysis")
            for res in results:
                with st.expander(f"{res['ticker']} - {res['label']} (Confidence: {res['confidence']:.0f}/100)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Price:** ${res['price']:.2f}")
                        st.write(f"**SMA50:** ${res['sma50']:.2f}" if res['sma50'] else "**SMA50:** N/A")
                        st.write(f"**SMA200:** ${res['sma200']:.2f}" if res['sma200'] else "**SMA200:** N/A")
                        st.write(f"**RSI:** {res['rsi']:.1f}" if res['rsi'] else "**RSI:** N/A")
                    
                    with col2:
                        st.write(f"**52W High:** ${res['hi52']:.2f}" if res['hi52'] else "**52W High:** N/A")
                        st.write(f"**52W Low:** ${res['lo52']:.2f}" if res['lo52'] else "**52W Low:** N/A")
                        if res['sentiment']:
                            st.write(f"**News Sentiment:** {res['sentiment']['compound']:+.2f} ({res['sentiment']['n']} headlines)")
                    
                    if res['reasons']:
                        st.write("**Key Signals:**")
                        for reason in res['reasons']:
                            st.write(f"• {reason}")
            
            # Send Telegram notification if configured
            if telegram_token and telegram_chat_id:
                if st.button("📱 Send Telegram Alert"):
                    alert_text = format_alert(results, tz_name)
                    if send_telegram(telegram_token, telegram_chat_id, alert_text):
                        st.success("✅ Telegram alert sent successfully!")
                    else:
                        st.error("❌ Failed to send Telegram alert. Check your bot token and chat ID.")
        else:
            st.warning("No results to display. Check your watchlist and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("*This application provides educational analysis only. Not financial advice.*")

if __name__ == "__main__":
    main()
