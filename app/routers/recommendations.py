# ----------------------------------------------------
# recommendations.py — CLEAN SIMPLIFIED VERSION
# Only CSV close_price decides CLOSED vs ACTIVE
# ----------------------------------------------------

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd, numpy as np, os, time, requests
from datetime import datetime
from zoneinfo import ZoneInfo

from app.routers.quotes import map_symbol, is_fno_symbol, _safe_float
from kiteconnect import KiteConnect
from app.routers.features import get_feature_flags

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# ----------------------------------------------------
# PATH
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_CANDIDATES = [
    "/data/Recommendation_Data.csv",  # Render disk
    os.path.abspath(os.path.join(BASE_DIR, "..", "data", "Recommendation_Data.csv")),  # local
    os.path.abspath(os.path.join(BASE_DIR, "..", "data", "Recommendation_Data_1.csv")),  # local alt (your old file)
]

CSV_PATH = next((p for p in CSV_CANDIDATES if p and os.path.exists(p)), CSV_CANDIDATES[0])

print("📌 Recommendations CSV_PATH =", CSV_PATH)


# ----------------------------------------------------
# Zerodha Init
# ----------------------------------------------------
API_KEY = os.getenv("KITE_API_KEY", "")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
BACKEND_HTTP = os.getenv("BACKEND_HTTP_BASE_URL", "http://127.0.0.1:8000")

kite = None
if API_KEY and ACCESS_TOKEN:
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(ACCESS_TOKEN)
        print("Kite Ready")
    except:
        print("Kite Init Failed")


# ----------------------------------------------------
# SAFE DATE PARSER
# ----------------------------------------------------
def parse_date_safe(raw):
    if not raw:
        return None

    raw = str(raw).strip()

    # ✅ MM/DD/YYYY comes FIRST
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",          # ✅ FIXED
        "%m-%d-%Y",          # ✅ FIXED
        "%Y-%m-%d",
    ]:
        try:
            return datetime.strptime(raw, fmt)
        except:
            pass

    # Final fallback (STRICT)
    try:
        dt = pd.to_datetime(raw, dayfirst=False, errors="coerce")
        if not pd.isna(dt):
            return dt.to_pydatetime()
    except:
        pass

    return None



# ----------------------------------------------------
# FETCH ALL LIVE PRICES (batch)
# ----------------------------------------------------
def fetch_batch_prices(symbols):
    if not kite:
        print("❌ Render KITE NOT INITIALIZED! No live prices fetched.")
        return {}

    final = {}
    keys, mapping = [], {}

    for s in symbols:
        mapped = map_symbol(s)
        exch = "NFO" if is_fno_symbol(mapped) else "NSE"
        key = f"{exch}:{mapped}"
        mapping[key] = s
        keys.append(key)

    chunks = [keys[i:i+90] for i in range(0, len(keys), 90)]

    for chunk in chunks:
        try:
            resp = kite.quote(chunk)
        except:
            time.sleep(0.2)
            resp = {}

        for k in chunk:
            item = resp.get(k, {})
            px = item.get("last_price") or item.get("last_traded_price") or item.get("ohlc", {}).get("close")
            final[mapping[k]] = _safe_float(px)


    return final


# ----------------------------------------------------
# FEATURE FLAGS (safe bool + fallback excel read)
# ----------------------------------------------------
def _as_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")

def _norm_user(u: str) -> str:
    return (u or "").strip().lower().replace(" ", "")

def _read_feature_flags_from_excel(username: str) -> dict:
    """Fallback: read feature_access.xlsx from common locations (local + Render)."""
    candidates = [
        os.getenv("FEATURE_ACCESS_FILE", "").strip(),
        "/data/feature_access.xlsx",
        "/data/feature_access.xlsm",
        os.path.abspath(os.path.join(BASE_DIR, "..", "data", "feature_access.xlsx")),
        os.path.abspath(os.path.join(BASE_DIR, "..", "data", "feature_access.xlsm")),
        os.path.abspath(os.path.join(BASE_DIR, "..", "data", "feature_access.csv")),
    ]
    candidates = [p for p in candidates if p]

    target = _norm_user(username)
    for path in candidates:
        try:
            if not os.path.exists(path):
                continue
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            if df is None or df.empty:
                continue

            # normalize columns
            df.columns = [str(c).strip().lower() for c in df.columns]
            if "username" not in df.columns:
                continue

            df["_u"] = df["username"].astype(str).map(_norm_user)
            hit = df[df["_u"] == target]
            if hit.empty:
                continue

            row = hit.iloc[0].to_dict()
            row["_source_file"] = path
            return row


        except Exception as e:
            print(f"⚠ feature_access read failed for {path}: {e}")
            continue
    return {}

def _get_flags(username: str) -> dict:
    """
    Primary: use shared get_feature_flags()
    Fallback: read excel
    IMPORTANT: Only trust primary if it actually contains allow_recommendation_page
    """
    try:
        f = get_feature_flags(username)

        # sometimes it may return nested dicts like {"features": {...}}
        if isinstance(f, dict):
            if "features" in f and isinstance(f["features"], dict):
                f = f["features"]
            elif "data" in f and isinstance(f["data"], dict):
                f = f["data"]

            # normalize keys
            f_norm = {str(k).strip().lower(): v for k, v in f.items()}

            # ✅ ONLY trust if it contains our required key
            if "allow_recommendation_page" in f_norm:
                return f_norm

    except Exception as e:
        print(f"⚠ get_feature_flags failed: {e}")

    # fallback excel
    return _read_feature_flags_from_excel(username) or {}


@router.get("/debug-flags")
def debug_flags(username: str):
    username = (username or "").strip()
    flags = _get_flags(username) or {}

    # normalize keys (very important)
    flags_norm = {str(k).strip().lower(): v for k, v in flags.items()}

    allow = _as_bool(
        flags_norm.get("allow_recommendation_page")
        or flags_norm.get("allow_recommendations_page")   # optional typo safe
    )

    return {
        "username": username,
        "normalized_user": _norm_user(username),
        "flags_keys": list(flags_norm.keys()),
        "allow_recommendation_page_raw": flags_norm.get("allow_recommendation_page"),
        "allow_recommendation_page_bool": allow,
        "csv_path": CSV_PATH,
        "csv_exists": os.path.exists(CSV_PATH),
        "feature_access_file_used": flags_norm.get("_source_file"),
    }

# ----------------------------------------------------
# MAIN API — SIMPLIFIED LOGIC
# ----------------------------------------------------
@router.get("/data")
def get_recommendations(username: str | None = None):

    # ⭐ DEBUG IMPORT CHECK (Render logs)
    print("🟢 Render IMPORT TEST:", map_symbol, is_fno_symbol, _safe_float)

    # ✅ STEP 3 — ADD RIGHT HERE (top of function)    # ✅ Access gating
    username = (username or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="username is required")

    flags = _get_flags(username)

    # Accept TRUE/True/"TRUE"/1 etc.
    if not _as_bool(flags.get("allow_recommendation_page")):
        raise HTTPException(status_code=403, detail="Recommendations are not enabled for this user")
    # ✅ then continue your existing logic...
    if not os.path.exists(CSV_PATH):
    # ✅ Don't block UI just because CSV isn't present
        return JSONResponse(
            content=[],
            headers={"Cache-Control": "no-store"}
        )


    # ----------------------------------------------------
    # READ CSV
    # ----------------------------------------------------
    df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # ⬇ ADD HERE
    if "price_closeto" not in df.columns:
        df["price_closeto"] = None

    # ----------------------------------------------------
    # NORMALIZE DATE COLUMN (MM/DD/YYYY → YYYY-MM-DD)
    # ----------------------------------------------------
# ----------------------------------------------------
# PRESERVE RAW DATETIME + NORMALIZE DATE (DO NOT LOSE TIME)
# ----------------------------------------------------

    # Detect original datetime column from CSV
    src_date_col = None
    for c in ["date", "signal_date", "raw_datetime"]:
        if c in df.columns:
            src_date_col = c
            break

    # 1️⃣ Preserve original value (date + time)
    if src_date_col:
        df["raw_datetime"] = df[src_date_col].astype(str)

    # 2️⃣ Create normalized date ONLY for filtering
    df["date"] = df["raw_datetime"].apply(
        lambda x: (
            parse_date_safe(x).strftime("%Y-%m-%d")
            if parse_date_safe(x)
            else None
        )
    )



    # Ensure columns
    if "signal_price" not in df.columns:
        df["signal_price"] = df.get("close_price")

    if "outcome" not in df.columns:
        df["outcome"] = None

    symbols = df["script"].astype(str).str.upper().tolist()
    live_map = fetch_batch_prices(symbols)

    output_prices = []

    # ----------------------------------------------------
    # PROCESS EACH ROW
    # ----------------------------------------------------
    for idx, row in df.iterrows():

        sym = str(row["script"]).upper().strip()
        signal_price_raw = row.get("signal_price")

        # safe signal price
        try:
            signal_price = float(signal_price_raw)
        except:
            signal_price = 0.0

        live_val = live_map.get(sym)

        # ⭐ ALWAYS prevent NaN
        if live_val is None or live_val == 0 or pd.isna(live_val):
            if pd.isna(signal_price_raw):
                live = 0.0
            else:
                live = signal_price
        else:
            live = live_val

        # ----------------------------------------------------
        # CHECK CSV CLOSE PRICE STATUS
        # ----------------------------------------------------
        csv_close_raw = row.get("close_price")
        try:
            csv_close = float(csv_close_raw)
        except:
            csv_close = None

        # Case 1: Missing close → ACTIVE
        # -------------------------------
        # CLOSED / ACTIVE (FINAL RULE)
        # -------------------------------
        if csv_close is not None and csv_close > 0:
            is_csv_closed = True
        else:
            is_csv_closed = False

        # CLOSED → override price with close price
        if is_csv_closed:
            df.at[idx, "outcome"] = "CSV_CLOSED"
            df.at[idx, "currentPrice"] = csv_close
            output_prices.append(csv_close)
            continue

        # ACTIVE → use live
        df.at[idx, "outcome"] = None
        df.at[idx, "currentPrice"] = live
        output_prices.append(live)

    df["currentPrice"] = output_prices
    df = df.replace({np.nan: None})

    return JSONResponse(
        content=df.to_dict("records"),
        headers={"Cache-Control": "no-store"}
    )
