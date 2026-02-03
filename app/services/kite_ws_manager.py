# backend/app/services/kite_ws_manager.py
import os
import time
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import requests

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None


# --------------------------------------------------------------------
# KiteConnect instance cache (avoid re-creating client on every quote())
# --------------------------------------------------------------------
_KITE_CLIENT = None
_KITE_CREDS = (None, None)

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
PREFERRED_EXCHANGE = os.getenv("PREFERRED_EXCHANGE", "NSE").upper()


def _is_render() -> bool:
    return bool(os.getenv("RENDER")) or os.path.exists("/data")


IS_RENDER = _is_render()


# --------------------------------------------------------------------
# Resolve instruments.csv path (Render + Local)
# --------------------------------------------------------------------
SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))          # .../backend/app/services
APP_DIR = os.path.abspath(os.path.join(SERVICES_DIR, ".."))        # .../backend/app


def _norm_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    s = str(p).strip().strip('"')
    if not s:
        return None
    return s.replace("\\", "/")


# ✅ Source of truth on Render is /data. No seeding from repo/app anymore.
CSV_PATH = _norm_path(os.getenv("INSTRUMENTS_CSV_PATH")) or (
    "/data/instruments.csv" if IS_RENDER else os.path.join(APP_DIR, "data", "instruments.csv")
)


# --------------------------------------------------------------------
# Kite helpers (for quotes etc.) - unchanged behavior
# --------------------------------------------------------------------
def _ensure_kite():
    """
    Returns a cached KiteConnect instance.

    Why: Creating a new KiteConnect client for every request is expensive and
    can slow down your entire API when /quotes is polled frequently.

    Note: If you update KITE_ACCESS_TOKEN / KITE_API_KEY and restart/redeploy,
    the cache resets automatically.
    """
    global _KITE_CLIENT, _KITE_CREDS

    if KiteConnect is None:
        raise RuntimeError("kiteconnect not installed. pip install kiteconnect")

    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")
    if not api_key or not access_token:
        raise RuntimeError("KITE_API_KEY / KITE_ACCESS_TOKEN not set in env")

    if _KITE_CLIENT is None or _KITE_CREDS != (api_key, access_token):
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        _KITE_CLIENT = kite
        _KITE_CREDS = (api_key, access_token)

    return _KITE_CLIENT


def _file_sig(path: str) -> Optional[Tuple[float, int]]:
    try:
        return (os.path.getmtime(path), os.path.getsize(path))
    except Exception:
        return None


def _download_instruments_to(path: str) -> Dict[str, Any]:
    """
    Download instruments.csv directly from Zerodha (public endpoint)
    and overwrite the given path atomically.
    """
    try:
        if IS_RENDER:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        url = "https://api.kite.trade/instruments"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
        resp = requests.get(url, headers=headers, timeout=60)

        if resp.status_code != 200:
            return {"status": "error", "message": f"Failed {resp.status_code}: {resp.text}", "path": path}

        content = resp.content
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(content)
        os.replace(tmp, path)

        sig = _file_sig(path)
        mb = round((sig[1] / (1024 * 1024)), 3) if sig else None
        return {"status": "ok", "message": "Downloaded instruments.csv from Zerodha", "path": path, "size_mb": mb}
    except Exception as e:
        return {"status": "error", "message": str(e), "path": path}


def _ensure_instruments_file_exists() -> None:
    """
    IMPORTANT: No seeding/copying from repo/app.
    If CSV_PATH is missing, we download directly from Zerodha.
    """
    if CSV_PATH and os.path.exists(CSV_PATH):
        return

    print(f"⚠️ instruments.csv missing at {CSV_PATH} — downloading directly from Zerodha...")
    res = _download_instruments_to(CSV_PATH)
    print(f"✅ instruments.csv ensure result: {res.get('status')} {res.get('message')}")


# --------------------------------------------------------------------
# Load instruments.csv once
# --------------------------------------------------------------------
def _read_csv_safely(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        print(f"⚠️ instruments.csv NOT FOUND at: {path}")
        return pd.DataFrame()

    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False, encoding=enc)
            print(f"✅ instruments.csv loaded: {path} rows={len(df)} enc={enc}")
            return df
        except Exception as e:
            last_err = e

    print(f"⚠️ Could not read instruments.csv: {path} err={last_err}")
    return pd.DataFrame()


def reload_instruments(path: Optional[str] = None, force_download: bool = False) -> Dict[str, Any]:
    """
    Reload instruments dataframe from disk.
    Optionally force-download from Zerodha before reading.
    """
    global CSV_PATH, INSTRUMENTS_DF, EQUITY_DF

    if path:
        CSV_PATH = _norm_path(path) or path

    if force_download:
        dl = _download_instruments_to(CSV_PATH)
        if dl.get("status") != "ok":
            return {"status": "error", "message": f"download failed: {dl.get('message')}", "csv_path": CSV_PATH}

    df = _read_csv_safely(CSV_PATH)

    if not df.empty:
        df.columns = [str(c).strip().lower() for c in df.columns]

        for col in ("tradingsymbol", "exchange", "segment", "instrument_type", "name"):
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        df["exchange"] = df["exchange"].str.upper()

    INSTRUMENTS_DF = df
    EQUITY_DF = df.copy()

    return {"status": "ok", "csv_path": CSV_PATH, "rows": int(len(df))}


# Ensure file exists (Render disk) without seeding from repo
_ensure_instruments_file_exists()

try:
    INSTRUMENTS_DF = _read_csv_safely(CSV_PATH)
    if not INSTRUMENTS_DF.empty:
        INSTRUMENTS_DF.columns = [str(c).strip().lower() for c in INSTRUMENTS_DF.columns]

        # normalize key cols so .str works without errors
        for col in ("tradingsymbol", "exchange", "segment", "instrument_type", "name"):
            if col not in INSTRUMENTS_DF.columns:
                INSTRUMENTS_DF[col] = ""
            INSTRUMENTS_DF[col] = INSTRUMENTS_DF[col].fillna("").astype(str)

        # exchange upper
        INSTRUMENTS_DF["exchange"] = INSTRUMENTS_DF["exchange"].str.upper()

    EQUITY_DF = INSTRUMENTS_DF.copy()
except Exception as e:
    print(f"⚠️ Could not load instruments.csv: {e}")
    INSTRUMENTS_DF = pd.DataFrame()
    EQUITY_DF = pd.DataFrame()


# --------------------------------------------------------------------
# Tick cache
# --------------------------------------------------------------------
_LAST_TICKS: Dict[str, Dict[str, Any]] = {}
_LAST_TS: Dict[str, float] = {}


# --------------------------------------------------------------------
# Index shortcuts
# --------------------------------------------------------------------
_INDEX_MAP_ZERODHA: Dict[str, str] = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "SENSEX": "BSE:SENSEX",
}


def _map_symbol_zerodha(symbol: str) -> Optional[str]:
    if not symbol:
        return None

    raw = symbol.strip()
    u = raw.upper()

    # Already qualified like 'NSE:RELIANCE' / 'NFO:NIFTY...'
    if ":" in u:
        return u

    # Known indices
    if u in _INDEX_MAP_ZERODHA:
        return _INDEX_MAP_ZERODHA[u]

    # Resolve via instruments.csv
    if not INSTRUMENTS_DF.empty and "tradingsymbol" in INSTRUMENTS_DF.columns:
        rows = INSTRUMENTS_DF.loc[INSTRUMENTS_DF["tradingsymbol"].str.upper() == u]

        if not rows.empty:
            if len(rows) > 1 and "exchange" in rows.columns:
                pref = rows.loc[rows["exchange"].str.upper() == PREFERRED_EXCHANGE]
                if not pref.empty:
                    rows = pref

            r = rows.iloc[0]
            ex = str(r.get("exchange", PREFERRED_EXCHANGE)).upper()
            ts = str(r.get("tradingsymbol", u))
            return f"{ex}:{ts}"

    return f"{PREFERRED_EXCHANGE}:{u}"


def subscribe_symbol(symbol: str) -> Dict[str, Any]:
    z = _map_symbol_zerodha(symbol)
    if not z:
        return {}
    kite = _ensure_kite()
    data = kite.quote([z]) or {}
    item = data.get(z) or {}
    tick = {
        "tradingsymbol": z,
        "last_price": item.get("last_price"),
        "ohlc": item.get("ohlc") or {},
        "timestamp": item.get("timestamp"),
    }
    key = symbol.upper().strip()
    _LAST_TICKS[key] = tick
    _LAST_TS[key] = time.time()
    return tick


def get_quote(symbol: str) -> Dict[str, Any]:
    key = symbol.upper().strip()
    ts = _LAST_TS.get(key)
    if ts and (time.time() - ts) <= 3:
        return _LAST_TICKS.get(key, {})
    return subscribe_symbol(symbol)


def get_instrument(symbol: str) -> Dict[str, Any]:
    if not symbol:
        return {}

    s = symbol.upper().strip()

    if s in _INDEX_MAP_ZERODHA:
        z = _INDEX_MAP_ZERODHA[s]
        return {
            "exchange": z.split(":")[0],
            "segment": "INDICES",
            "instrument_type": "INDEX",
            "lot_size": None,
            "tradingsymbol": z,
            "symbol": s,
            "name": s,
        }

    if INSTRUMENTS_DF.empty:
        return {}

    rows = INSTRUMENTS_DF.loc[INSTRUMENTS_DF["tradingsymbol"].str.upper() == s]

    if rows.empty:
        return {
            "exchange": PREFERRED_EXCHANGE,
            "segment": "",
            "instrument_type": "",
            "lot_size": None,
            "tradingsymbol": s,
            "symbol": s,
            "name": "",
        }

    if len(rows) > 1 and "exchange" in rows.columns:
        pref = rows.loc[rows["exchange"].str.upper() == PREFERRED_EXCHANGE]
        if not pref.empty:
            rows = pref

    r = rows.iloc[0].to_dict()
    return {
        "exchange": (r.get("exchange") or PREFERRED_EXCHANGE).upper(),
        "segment": r.get("segment", ""),
        "instrument_type": r.get("instrument_type", ""),
        "lot_size": r.get("lot_size"),
        "tradingsymbol": r.get("tradingsymbol", s),
        "symbol": s,
        "name": r.get("name", ""),
    }


def search_instruments(
    query: str,
    limit: int = 50,
    exchanges: Optional[List[str]] = None,
    segments: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if INSTRUMENTS_DF.empty or not query:
        return []

    df = INSTRUMENTS_DF
    q = query.strip().upper()

    name_series = df["name"] if "name" in df.columns else pd.Series(index=df.index, dtype=str)
    mask = (
        df["tradingsymbol"].str.upper().str.contains(q, na=False) |
        name_series.astype(str).str.upper().str.contains(q, na=False)
    )

    if exchanges and "exchange" in df.columns:
        mask &= df["exchange"].str.upper().isin([e.upper() for e in exchanges])

    if segments and "segment" in df.columns:
        mask &= df["segment"].astype(str).str.upper().isin([s.upper() for s in segments])

    out = df.loc[mask].head(limit).copy()
    results: List[Dict[str, Any]] = []
    for _, r in out.iterrows():
        results.append({
            "exchange": str(r.get("exchange", "")).upper(),
            "segment": r.get("segment", ""),
            "instrument_type": r.get("instrument_type", ""),
            "lot_size": r.get("lot_size"),
            "tradingsymbol": r.get("tradingsymbol", ""),
            "name": r.get("name", ""),
        })
    return results
