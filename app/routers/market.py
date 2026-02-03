# ============================================================
# market.py — FULL SUPER MERGED VERSION (Option A)
# ============================================================

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import os
import sys
import time
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from app.routers.features import get_feature_flags
from fastapi import APIRouter, HTTPException, Query, Request





# Timezone
try:
    from zoneinfo import ZoneInfo
except:
    ZoneInfo = None

# ============================================================
# IST HELPERS (NO SIDE EFFECTS)
# ============================================================

if ZoneInfo:
    IST = ZoneInfo("Asia/Kolkata")
else:
    IST = None

from fastapi import WebSocket, WebSocketDisconnect
import asyncio

# NOTE:
# We intentionally avoid KiteTicker on Render because expired/invalid tokens cause endless
# 403 handshake retries (log spam). Instead, we stream ticks by polling Kite quote().
from app.services import kite_ws_manager as ws_manager
import math

def get_user_sig_paths(username: str):
    safe = username.strip().replace(" ", "_")

    base_render = "/data"
    base_local = os.path.join(BASE_DIR, "..", "data")
    base = base_render if os.path.exists(base_render) else base_local

    return (
        os.path.join(base, f"{safe}_sig_2mins.csv"),
        os.path.join(base, f"{safe}_sig_15mins.csv"),
    )



def is_today_ist(date_val):
    """
    Returns True if date_val belongs to TODAY in IST.
    Safe for:
      - string dates
      - naive datetimes
      - tz-aware datetimes
    Does NOT affect any other functionality.
    """
    try:
        if not date_val:
            return False

        dt = pd.to_datetime(date_val, errors="coerce")
        if pd.isna(dt):
            return False

        # Attach / convert timezone safely
        if IST:
            if dt.tzinfo is None:
                dt = dt.tz_localize(IST)
            else:
                dt = dt.tz_convert(IST)

            today = datetime.now(IST).date()
        else:
            # Fallback (should never happen on Python 3.9+)
            today = datetime.now().date()

        #return dt.date() == today
        return dt.date() >= today-timedelta(2)

    except Exception as e:
        print("⚠ is_today_ist error:", e)
        return False


# ============================================================
# WhatsApp handler
# ============================================================
try:
    from Whatsapp_Push_Module import run_whatsapp_alert
except Exception:
    def run_whatsapp_alert(symbol):
        return {"status": "error", "message": "Whatsapp module missing"}

# ============================================================
# Signal generator
# ============================================================
try:
    from signal_generator import generate_signal_backend, parse_to_unix
except Exception:
    def generate_signal_backend(*args, **kwargs):
        return {"status": "error", "message": "signal_generator not available"}

# ============================================================
# Zerodha SDK
# ============================================================
try:
    from kiteconnect import KiteConnect
except:
    KiteConnect = None

router = APIRouter(prefix="/market", tags=["market"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#RECO_CSV = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "Recommendation_Data.csv"))

FNO_SEGMENTS = {
    "NFO-FUT", "NFO-OPT",
    "BFO-FUT", "BFO-OPT",
    "CDS-FUT", "CDS-OPT",
    "MCX-FUT", "MCX-OPT",
    "NCO-FUT", "NCO-OPT",
}

# ============================================================
# LOAD INSTRUMENTS.CSV
# ============================================================
INSTRUMENTS_DF: Optional[pd.DataFrame] = None
def _load_instruments_csv_if_present():
    """
    Load instruments.csv using the same Render+Local path rules used elsewhere:

    Priority:
      1) INSTRUMENTS_CSV_PATH env (Render screenshot shows you set this)
      2) /data/instruments.csv on Render
      3) backend/app/data/instruments.csv locally
      4) best-effort fallbacks inside the repo (in case you copied it there)
    """
    global INSTRUMENTS_DF
    try:
        def _norm(p: str) -> str:
            return (p or "").strip().strip('"').replace("\\", "/")

        env_path = _norm(os.getenv("INSTRUMENTS_CSV_PATH", ""))
        candidates = []

        if env_path:
            candidates.append(env_path)

        # Render disk
        if os.path.exists("/data"):
            candidates.append("/data/instruments.csv")

        # Local default
        routers_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(routers_dir, ".."))
        candidates.append(os.path.join(app_dir, "data", "instruments.csv"))

        # Repo fallbacks (optional)
        candidates += [
            os.path.join(os.path.dirname(app_dir), "instruments.csv"),
            os.path.join(app_dir, "instruments.csv"),
            os.path.join(app_dir, "data", "instruments.csv"),
        ]

        csv_path = None
        for p in candidates:
            if p and os.path.exists(p):
                csv_path = p
                break

        print("Loading instruments from:", csv_path or "<NOT FOUND>")
        if not csv_path:
            INSTRUMENTS_DF = None
            return

        last_err = None
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                df = pd.read_csv(csv_path, dtype=str, low_memory=False, encoding=enc)
                print(f"✅ instruments.csv loaded: {csv_path} rows={len(df)} enc={enc}")
                break
            except Exception as e:
                last_err = e
                df = None

        if df is None:
            print("❌ ERROR loading instruments.csv:", last_err)
            INSTRUMENTS_DF = None
            return

        # normalize columns
        df.columns = [str(c).strip().lower() for c in df.columns]
        for col in ("tradingsymbol", "exchange", "segment", "name", "instrument_token"):
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        df["tradingsymbol"] = df["tradingsymbol"].str.upper()
        df["exchange"] = df["exchange"].str.upper()
        df["segment"] = df["segment"].str.upper()
        df["name"] = df["name"].astype(str)

        INSTRUMENTS_DF = df

    except Exception as e:
        print("❌ ERROR loading instruments.csv:", e)
        INSTRUMENTS_DF = None


# ============================================================
# KITE SESSION MAKER
# ============================================================
def _get_kite():
    if KiteConnect is None:
        raise HTTPException(500, "kiteconnect not installed")

    api_key = os.getenv("KITE_API_KEY") or ""
    access_token = os.getenv("KITE_ACCESS_TOKEN") or ""

    if not api_key or not access_token:
        raise HTTPException(500, "Missing KITE_API_KEY / KITE_ACCESS_TOKEN")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    return kite

# ============================================================
# INTERVAL MAP
# ============================================================
TF_MAP = {
    "1m": ("minute", 800),
    "2m": ("2minute", 800),
    "15m": ("15minute", 800),
    "1h": ("60minute", 800),
    "1d": ("day", 400),
}

def _lookup_token_from_zerodha(symbol: str) -> Optional[int]:
    """
    ❌ IMPORTANT:
    KiteConnect has NO `search()` method. The previous implementation calling `kite.search()`
    caused this Render log error:
        'KiteConnect' object has no attribute 'search'

    So we do NOT call any non-existent search API here.
    We resolve instrument_token using instruments.csv only (covers EQ + F&O + indices present in the file).
    """
    return None



# ============================================================
# SIMPLE IN-MEMORY CACHES (Render performance)
# ============================================================
_OHLC_CACHE = {}  # key -> (ts, data)
_QUOTES_CACHE = {}  # key -> (ts, data)

def _cache_get(cache: dict, key, ttl: float):
    now = time.time()
    v = cache.get(key)
    if not v:
        return None
    ts, data = v
    if now - ts > ttl:
        return None
    return data

def _cache_set(cache: dict, key, data):
    cache[key] = (time.time(), data)

def _calc_lookback_days(interval: str, bars: int) -> int:
    """Choose a minimal lookback window based on requested bars."""
    interval = interval.lower()
    minutes = {"1m":1, "2m":2, "15m":15, "1h":60, "1d":375}.get(interval, 2)
    if interval == "1d":
        return max(10, min(1200, bars + 5))
    # ~375 mins per trading day, add buffer
    est_days = int(math.ceil((bars * minutes) / 375.0)) + 2
    return max(1, min(60, est_days))

# ============================================================
# FIXED INSTRUMENT TOKEN LOOKUP (INDEX + EQUITY + F&O)
# ============================================================
def _lookup_instrument_token(symbol: str) -> int:
    """
    Resolve instrument_token from instruments.csv.

    Supports:
      - Equity
      - F&O
      - Common index aliases (NIFTY, BANKNIFTY, FINNIFTY, etc.)

    Notes:
      - We purposely avoid KiteTicker/Kite search calls to prevent 403 spam on Render.
    """
    global INSTRUMENTS_DF

    sym_raw = (symbol or "").upper().strip()
    sym = sym_raw.replace(" ", "")

    # Ensure instruments are loaded
    if INSTRUMENTS_DF is None or getattr(INSTRUMENTS_DF, "empty", True):
        _load_instruments_csv_if_present()

    if INSTRUMENTS_DF is None or INSTRUMENTS_DF.empty:
        raise HTTPException(500, "instruments.csv not loaded")

    df = INSTRUMENTS_DF

    # Build clean columns once
    trad_clean = df["tradingsymbol"].astype(str).str.upper().str.replace(" ", "", regex=False)
    name_clean = df["name"].astype(str).str.upper().str.replace(" ", "", regex=False)

    # Common aliases -> actual index names (as seen in Zerodha instruments)
    alias = {
        "NIFTY": "NIFTY50",
        "NIFTY50": "NIFTY50",
        "NIFTY_50": "NIFTY50",
        "BANKNIFTY": "NIFTYBANK",
        "NIFTYBANK": "NIFTYBANK",
        "FINNIFTY": "NIFTYFINSERVICE",
        "NIFTYFIN": "NIFTYFINSERVICE",
        "SENSEX": "SENSEX",
        "MIDCPNIFTY": "NIFTYMIDCAPSELECT",
    }

    candidates = [sym]
    if sym in alias:
        candidates.append(alias[sym])

    # also try with spaces removed but keeping words
    # (e.g., 'NIFTY 50' comes as 'NIFTY50' in our clean compare)
    if sym_raw and sym_raw != sym:
        candidates.append(sym_raw.replace(" ", ""))

    # 1) Exact tradingsymbol match
    for c in candidates:
        row = df.loc[trad_clean == c]
        if not row.empty:
            return int(float(row.iloc[0]["instrument_token"]))

    # 2) Exact name match (useful for indices)
    for c in candidates:
        row = df.loc[name_clean == c]
        if not row.empty:
            return int(float(row.iloc[0]["instrument_token"]))

    # 3) Startswith (ADANIENT -> ADANIENT-EQ, etc.)
    row = df.loc[trad_clean.str.startswith(sym)]
    if not row.empty:
        return int(float(row.iloc[0]["instrument_token"]))

    # 4) Contains fallback
    row = df.loc[trad_clean.str.contains(sym, na=False)]
    if not row.empty:
        return int(float(row.iloc[0]["instrument_token"]))

    raise HTTPException(404, f"instrument not found for {symbol}")


# ============================================================
# TIME WINDOW
# ============================================================
def _time_window(interval_key: str, bars: int):
    now_ist = _ist_now()

    if interval_key == "day":
        frm = now_ist - timedelta(days=bars * 2)
    else:
        minutes_per_bar = {
            "minute": 1,
            "2minute": 2,
            "5minute": 5,
            "15minute": 15,
            "60minute": 60,
        }.get(interval_key, 1)

        frm = now_ist - timedelta(minutes=minutes_per_bar * bars)

    return frm, now_ist

# ============================================================
# OHLC API (FULL MERGED)
# ============================================================
@router.get("/ohlc")
def ohlc(
    symbol: str,
    interval: str = "1m",
    limit: int = 500,
    before: Optional[int] = None
):

    interval = interval.lower()
    if interval not in TF_MAP:
        raise HTTPException(400, f"Invalid interval {interval}")

    kite_interval, default_bars = TF_MAP[interval]
    bars = min(limit, default_bars)

    token = _lookup_instrument_token(symbol)
    kite = _get_kite()

    IST = ZoneInfo("Asia/Kolkata")
    now = datetime.now(IST)
    to = now   # ✅ DO NOT subtract 1 minute


    if before:
        to = datetime.fromtimestamp(before, tz=IST)

    # LOOKBACK
    # LOOKBACK (dynamic, avoids fetching 30+ days for limit=1)
    lookback_days = _calc_lookback_days(interval, bars)
    frm = to - timedelta(days=lookback_days)

    # Cache: bucket 'to' in 10s windows to reduce Zerodha calls
    bucket = int(to.timestamp() // 10)
    cache_key = (symbol.upper(), interval, bucket, bars)
    cached = _cache_get(_OHLC_CACHE, cache_key, ttl=8.0)
    if cached is not None:
        return cached
    try:
        data = kite.historical_data(token, frm, to, kite_interval)
    except Exception as e:
        raise HTTPException(502, f"Zerodha error: {e}")

    out = []
    for c in data or []:
        candle_dt = c.get("date")
        if not isinstance(candle_dt, datetime):
            continue

        # ---------------------------------------------------
        # SPECIAL CASE: 1D – force timestamp to 09:15 IST
        # so it behaves like real market open time.
        # Other timeframes keep the original Zerodha timestamp.
        # ---------------------------------------------------
        if interval == "1d":
            try:
                if ZoneInfo:
                    ist = candle_dt.astimezone(ZoneInfo("Asia/Kolkata"))
                else:
                    ist = candle_dt  # fallback
                ist_open = ist.replace(
                    hour=9, minute=15, second=0, microsecond=0
                )
                ts = int(ist_open.timestamp())
            except Exception:
                ts = int(candle_dt.timestamp())
        else:
            ts = int(candle_dt.timestamp())

        out.append({
            "time": ts,
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c["volume"]),
        })

    _cache_set(_OHLC_CACHE, cache_key, out)
    return out

    

# ============================================================
# CSV SIGNAL PATHS
# ============================================================
BACKEND_DIR = "/data"
base_render = "/data"
base_local = os.path.join(BASE_DIR, "..", "data")
base = base_render if os.path.exists(base_render) else base_local
print ("base_local only for logging: {}".format(base_local))
print ("Base directory location for Signal files: {}".format(base))

SIG2 = os.path.join(base, "sig_2mins.csv")
SIG15 = os.path.join(base, "sig_15mins.csv")

# ============================================================
# GENERATE SIGNAL
# ============================================================
class GenReq(BaseModel):
    symbol: str
    username: str | None = None




@router.post("/generate-signal")
def generate_signal_api(req: GenReq, request: Request):
    # ✅ username from JSON body first (FastAPI already parsed JSON into req)
    username = (req.username or "").strip()

    # ✅ fallback: allow query param too (optional)
    if not username:
        username = (request.query_params.get("username") or "").strip()

    if not username:
        raise HTTPException(status_code=400, detail="username is required")

    # ✅ Feature lock check
    flags = get_feature_flags(username)
    if not flags.get("allow_generate_signals"):
        raise HTTPException(status_code=403, detail="Generate Signals is not enabled for this user")

    try:
        # ----------------------------------------------------
        # ABSOLUTE PATH TO REAL_TIME_ALERTS.py
        # ----------------------------------------------------
        BACKEND_ROOT = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )

        SIGNAL_SCRIPT = os.path.join(BACKEND_ROOT, "REAL_TIME_ALERTS.py")

        if not os.path.exists(SIGNAL_SCRIPT):
            raise FileNotFoundError(f"REAL_TIME_ALERTS.py not found at {SIGNAL_SCRIPT}")

        # ----------------------------------------------------
        # RUN SIGNAL GENERATOR
        # ----------------------------------------------------
        subprocess.run(
            [sys.executable, SIGNAL_SCRIPT, req.symbol, username],
            check=True
        )

        # ----------------------------------------------------
        # LOAD SIGNALS FROM CSV
        # ----------------------------------------------------
        SIG2_PATH, SIG15_PATH = get_user_sig_paths(username)

        # Guard: CSV may be empty during first run or partial write
        if (not os.path.exists(SIG2_PATH)) or os.path.getsize(SIG2_PATH) == 0:
            df2 = pd.DataFrame()
        else:
            df2 = pd.read_csv(SIG2_PATH)
        if (not os.path.exists(SIG15_PATH)) or os.path.getsize(SIG15_PATH) == 0:
            df15 = pd.DataFrame()
        else:
            df15 = pd.read_csv(SIG15_PATH)

        # ----------------------------------------------------
        # FILTER TODAY ONLY (IN-MEMORY)
        # ----------------------------------------------------
        df2_today = df2[df2["Date"].apply(is_today_ist)]
        df15_today = df15[df15["Date"].apply(is_today_ist)]

        return generate_signal_backend(req.symbol, df2_today, df15_today)

    except Exception as e:
        print("❌ GENERATE SIGNAL ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================
# GET SIGNAL
# ============================================================
@router.get("/get-signal")
def get_signal(symbol: str, username: str):
    try:
        SIG2, SIG15 = get_user_sig_paths(username)
        df2 = pd.read_csv(SIG2)
        df15 = pd.read_csv(SIG15)
        
        return generate_signal_backend(symbol, df2, df15)
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================
# ALL SIGNALS
# ============================================================
@router.get("/all-signals")
def all_signals(symbol: str, username: str):
    print("executing All Signals")

    try:
        def safe(x):
            if pd.isna(x):
                return None
            return x

        # ----------------------------------------------------
        # LOAD USER-SCOPED CSVs
        # ----------------------------------------------------
        SIG2, SIG15 = get_user_sig_paths(username)
        df2 = pd.read_csv(SIG2)
        df15 = pd.read_csv(SIG15)

        # ----------------------------------------------------
        # FILTER TODAY ONLY
        # ----------------------------------------------------
        df2 = df2[df2["Date"].apply(is_today_ist)]
        df15 = df15[df15["Date"].apply(is_today_ist)]

        df2["tf"] = "2m"
        df15["tf"] = "15m"

        # ----------------------------------------------------
        # FILTER SCRIPT
        # ----------------------------------------------------
        df2 = df2[df2["script"].astype(str).str.upper() == symbol.upper()].copy()
        df15 = df15[df15["script"].astype(str).str.upper() == symbol.upper()].copy()

        df2 = df2.fillna("")
        df15 = df15.fillna("")

        out_all = []

        # ----------------------------------------------------
        # PROCESS BOTH TIMEFRAMES
        # ----------------------------------------------------
        for _, r in pd.concat([df2, df15]).iterrows():

            # ✅ SINGLE SOURCE OF TRUTH
            # Date already contains candle timestamp
            try:
                dt = pd.to_datetime(r["Date"])
                if dt.tzinfo is None:
                    dt = dt.tz_localize("Asia/Kolkata")
                ts = int(dt.timestamp())  # 🔥 seconds (NOT ms)
            except Exception:
                continue

            alert = str(r.get("Alert_details", "")).upper().strip()
            if not alert:
                continue

            if "BULLISH" in alert:
                sig = "BUY"
            elif "BEARISH" in alert:
                sig = "SELL"
            else:
                continue

            out_all.append({
                "timestamp": ts,        # 🔥 candle-aligned
                "signal": sig,
                "tf": r["tf"],
                "close_price": safe(r.get("close_price")),
                "alert_details": safe(r.get("Alert_details")),
                "screener": safe(r.get("screener")),
                "screener_side": safe(r.get("screener_side")),
                "user_action": safe(r.get("user_actions")),
                "strategy": safe(r.get("Strategy")),
                "pattern": safe(r.get("pattern_identifier_trendbreak")),
                "support": safe(r.get("Price_Closer_Support")),
                "resistance": safe(r.get("Price_Closer_Resistance")),
            })

        out_all = sorted(out_all, key=lambda x: x["timestamp"])

        return {"status": "success", "signals": out_all}

    except Exception as e:
        return {"status": "error", "message": str(e)}





# ============================================================
# WHATSAPP ALERT
# ============================================================
@router.post("/send-whatsapp")
def send_whatsapp(req: GenReq):
    try:
        return run_whatsapp_alert(req.symbol)
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================
# RELOAD INSTRUMENTS
# ============================================================
@router.post("/reload_instruments")
def reload_instruments():
    global INSTRUMENTS_DF
    try:
        _load_instruments_csv_if_present()
        return {"ok": True, "rows": len(INSTRUMENTS_DF)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Auto path resolver for render + local
def get_reco_path():
    render_path = "/data/Recommendation_Data.csv"
    local_path = os.path.join(BASE_DIR, "..", "data", "Recommendation_Data.csv")
    return render_path if os.path.exists(render_path) else local_path


# Output file paths (strategy CSVs)
def get_output_paths():
    base_render = "/data"
    base_local = os.path.join(BASE_DIR, "..", "data")

    base = base_render if os.path.exists(base_render) else base_local

    return {
        "intraday": os.path.join(base, "sig_intraday.csv"),
        "btst": os.path.join(base, "sig_btst.csv"),
        "shortterm": os.path.join(base, "sig_shortterm.csv"),
    }
def ensure_signal_files():
    paths = get_output_paths()
    for p in paths.values():
        if not os.path.exists(p):
            print("⚠ Creating missing file:", p)
            pd.DataFrame([], columns=[
                "script","Date","Alert_details","screener","user_actions"
            ]).to_csv(p, index=False)

ensure_signal_files()


# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
@router.get("/reco-load")
def reco_load(symbol: str = Query(...), tf: str = Query(...)):
    """
    UNIVERSAL VERSION — Works with your uploaded CSV schema.
    Supports:
      ✔ ALERT column (renamed to Alert_details)
      ✔ Strategy names like "Intraday - Fast Alerts"
      ✔ Missing Resistance/Support/pivot columns
      ✔ BUY/SELL detection from Signal_type or Alert_details
      ✔ 15m = Intraday + BTST
      ✔ 1d = Short-term
    """

    import numpy as np

    try:
        symbol_clean = symbol.upper().strip()

        # ============================================================
        # LOAD MASTER CSV (Recommendation_Data.csv)
        # ============================================================
        reco_path = get_reco_path()
        df = pd.read_csv(reco_path)

        # Clean column names
        df.columns = [c.strip() for c in df.columns]

        # ------------------------------------------------------------
        # FIX COLUMN NAME MISMATCHES
        # ------------------------------------------------------------
        rename_map = {
            "ALERT": "Alert_details",
            "Data_Interval": "data_interval",
            "Signal_type": "signal_type",
            "Price_closeto": "price_closeto",
            "Strategy": "Strategy",
        }

        for old, new in rename_map.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)

        # Anything missing → create empty columns
        safe_cols = [
            "Date", "script", "close_price", "Alert_details",
            "Strategy", "data_interval", "signal_type",
            "screener", "screener_side", "user_actions",
            "Resistance", "Support", "pivot_upper", "pivot_lower"
        ]
        for col in safe_cols:
            if col not in df.columns:
                df[col] = None

        df = df.replace({np.nan: None})
        df["script"] = df["script"].astype(str).str.upper().str.strip()

        # ============================================================
        # FILTER SCRIPT
        # ============================================================
        df = df[df["script"] == symbol_clean]

        if df.empty:
            return {"status": "success", "markers": [], "rows": []}

                # ============================================================
        # NORMALIZE STRATEGY NAMES (Supports ALL Intraday variations)
        # ============================================================
        def normalize_strategy(s):
            if not s:
                return None

            s = str(s).upper().strip()

            # ALL intraday variants
            if (
                "INTRADAY" in s or
                "INTRA" in s or
                "FAST" in s or
                "U-TURN" in s or
                "GAP" in s or
                "OTHERS" in s or
                "N " in s       # matches "N GAP-UP", "N GAP-DOWN"
            ):
                return "INTRADAY"

            # BTST
            if "BTST" in s:
                return "BTST"

            # Short-term
            if "SHORT" in s:
                return "SHORTTERM"

            # Default → treat as intraday
            return "INTRADAY"


        df["Strategy_norm"] = df["Strategy"].apply(normalize_strategy)

        # ============================================================
        # FILTER BY TIMEFRAME
        # ============================================================
        if tf == "15m":
            df = df[df["Strategy_norm"].isin(["INTRADAY", "BTST"])]
        elif tf == "1d":
            df = df[df["Strategy_norm"] == "SHORTTERM"]
        else:
            return {"status": "success", "markers": [], "rows": []}

        if df.empty:
            return {"status": "success", "markers": [], "rows": []}

        # ============================================================
        # DETECT BUY / SELL
        # ============================================================
        def detect_side(r):
            # 1) Signal_type column exists in your CSV
            t = str(r.get("signal_type", "")).lower()
            if "buy" in t:
                return "BUY"
            if "sell" in t:
                return "SELL"

            # 2) Fallback to Alert_details column
            a = str(r.get("Alert_details", "")).lower()
            if "bull" in a or "buy" in a:
                return "BUY"
            if "bear" in a or "sell" in a:
                return "SELL"

            return None

        df["signal_type"] = df.apply(detect_side, axis=1)

        # ============================================================
        # TIMESTAMP PARSER
        # ============================================================
        def to_ts(d):
            try:
                if not d:
                    return None

                s = str(d).strip()

                # Case 1: ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM)
                if s[:4].isdigit() and s[4] == "-":
                    dt = pd.to_datetime(s)

                # Case 2: MM/DD/YYYY or MM/DD/YYYY HH:MM  ✅ YOUR CSV
                elif "/" in s:
                    dt = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
                    if pd.isna(dt):
                        dt = pd.to_datetime(s, format="%m/%d/%Y %H:%M", errors="coerce")

                # Case 3: DD-MM-YYYY fallback
                elif "-" in s:
                    dt = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")

                else:
                    dt = pd.to_datetime(s, errors="coerce")

                if pd.isna(dt):
                    return None

                # Force IST
                if dt.tzinfo is None:
                    dt = dt.tz_localize("Asia/Kolkata")

                return int(dt.timestamp())

            except Exception as e:
                print("❌ DATE PARSE ERROR:", d, e)
                return None


        df["timestamp"] = df["Date"].apply(to_ts)
        df = df[df["timestamp"].notna()]

        # ============================================================
        # BUILD MARKERS
        # ============================================================
        markers = []

        for _, r in df.iterrows():
            sig = r["signal_type"]
            if sig not in ["BUY", "SELL"]:
                continue

            price = float(r.get("close_price") or 0)

            label = f"{r['Strategy_norm']} | {sig} | {price:.2f}"

            markers.append({
                "time": int(r["timestamp"]),
                "price": price,
                "color": "#16a34a" if sig == "BUY" else "#dc2626",
                "position": "belowBar" if sig == "BUY" else "aboveBar",
                "shape": "circle",
                "text": label,
            })

        # ============================================================
        # LATEST 4 DESCRIPTION ROWS
        # ============================================================
        df = df.sort_values("timestamp", ascending=False)
        rows = df.head(4)[
            ["Date", "Alert_details", "screener", "user_actions",
             "Strategy_norm", "signal_type", "close_price"]
        ].rename(columns={"Strategy_norm": "Strategy"}).to_dict("records")

        return {
            "status": "success",
            "markers": markers,
            "rows": rows
        }

    except Exception as e:
        print("RECO LOAD ERROR:", e)
        return {"status": "error", "message": str(e)}


@router.get("/instrument-info")
def instrument_info(symbol: str):
    """
    Returns instrument info for Buy / Sell pages
    Supports EQ + ALL F&O segments
    """
    try:
        sym = symbol.upper().strip().replace(" ", "")

        # Ensure instruments.csv is loaded
        global INSTRUMENTS_DF
        if INSTRUMENTS_DF is None or INSTRUMENTS_DF.empty:
            _load_instruments_csv_if_present()

        df = INSTRUMENTS_DF.copy()

        # Normalize
        df["tradingsymbol_clean"] = (
            df["tradingsymbol"]
            .astype(str)
            .str.upper()
            .str.replace(" ", "")
        )

        row = df[df["tradingsymbol_clean"] == sym]

        if row.empty:
            raise HTTPException(404, f"Instrument not found: {symbol}")

        r = row.iloc[0]

        segment = str(r.get("segment", "")).upper()
        lot_size = int(r.get("lot_size") or 1)

        is_fno = segment in FNO_SEGMENTS

        return {
            "symbol": sym,
            "tradingsymbol": r["tradingsymbol"],
            "exchange": r.get("exchange", "NSE"),
            "segment": segment,
            "instrument_token": int(r["instrument_token"]),
            "is_fno": is_fno,
            "lot_size": lot_size if is_fno else 1,
        }

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    
@router.websocket("/ticks")
async def ticks_ws(ws: WebSocket, symbol: str):
    """
    Streams LTP ticks to the frontend.

    ✅ Implementation:
    - Poll Kite quote() every 1s and push to the client.
    - Avoids KiteTicker WebSocket (which spam-fails with 403 when token expires on Render).

    Frontend expects:
      { ltp: number, timestamp: int }
    """
    await ws.accept()

    try:
        while True:
            tick = await asyncio.to_thread(ws_manager.get_quote, symbol)  # cached; runs blocking call in thread
            ltp = tick.get("last_price")

            ts = tick.get("timestamp")
            unix_ts = int(time.time())
            if ts:
                try:
                    unix_ts = int(pd.to_datetime(ts).timestamp())
                except Exception:
                    pass

            await ws.send_json({
                "ltp": ltp,
                "timestamp": unix_ts,
            })
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        return
    except Exception as e:
        # Send one error to the client and close (prevents log spam)
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        return
