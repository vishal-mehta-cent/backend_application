# =============================================================
# app/routers/quotes.py — FULL MERGED + DAY RANGE FIXED
# =============================================================

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List, Tuple
import os, re, time
import pandas as pd

# -------------------------------------------------------------
# Tiny in-process cache to protect the API from aggressive polling
# (e.g., frontend calling /quotes every second).
# -------------------------------------------------------------
_QUOTES_CACHE: Dict[Tuple[str, Optional[str]], Tuple[float, Dict[str, Any]]] = {}
_QUOTES_TTL_SECONDS = float(os.getenv('QUOTES_TTL_SECONDS', '1.5'))

# -------------------------------------------------------------
# WS helpers
# -------------------------------------------------------------
try:
    from app.services.kite_ws_manager import (
        get_quote,
        get_instrument,
        subscribe_symbol,
    )
except Exception:
    def get_quote(*args, **kwargs): return None
    def get_instrument(*args, **kwargs): return None
    def subscribe_symbol(*args, **kwargs): return None


# -------------------------------------------------------------
# Zerodha REST
# -------------------------------------------------------------
try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

router = APIRouter(prefix="/quotes", tags=["quotes"])

API_KEY = os.getenv("KITE_API_KEY", "")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
kite: Optional["KiteConnect"] = None

if KiteConnect and API_KEY and ACCESS_TOKEN:
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(ACCESS_TOKEN)
        print("✅ KiteConnect initialized (REST + LTP)")
    except Exception as e:
        kite = None
        print("⚠️ Failed to init KiteConnect:", e)
else:
    print("⚠️ Missing KITE_API_KEY or KITE_ACCESS_TOKEN")


# -------------------------------------------------------------
# instruments.csv
# -------------------------------------------------------------
INSTR_PATH = "app/instruments.csv"
try:
    INSTR = pd.read_csv(INSTR_PATH)
    INSTR["tradingsymbol"] = INSTR["tradingsymbol"].astype(str).str.upper()
except:
    INSTR = pd.DataFrame()
    print("⚠️ instruments.csv missing")


# =============================================================
# SHARED HELPERS (compatibility with recommendations.py)
# =============================================================
def is_fno_symbol(sym: str) -> bool:
    s = sym.upper()
    patt = r"\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}"
    return ("CE" in s or "PE" in s or "FUT" in s or re.search(patt, s))


def map_symbol(sym: str) -> str:
    s = sym.strip().upper()

    if is_fno_symbol(s):
        return s

    if s.endswith("-EQ") or s.endswith("-BE"):
        return s

    if s in INSTR["tradingsymbol"].values:
        return s

    eq = s + "-EQ"
    if eq in INSTR["tradingsymbol"].values:
        return eq

    row = INSTR[INSTR["tradingsymbol"].str.contains(s)]
    if not row.empty:
        return row.iloc[0]["tradingsymbol"]

    return s


def _safe_float(x):
    try:
        f = float(x)
        return f if f != 0 else None
    except:
        return None


# =============================================================
# ADVANCED ENGINE HELPERS
# =============================================================
MONTHS = {"JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"}
DERIV_RE = re.compile(r"(CE|PE|FUT)$", re.I)


def _is_option(sym: str) -> bool:
    s = sym.upper()
    return bool(DERIV_RE.search(s)) or any(m in s for m in MONTHS)


def _guess_exchange(sym: str, hint: Optional[str] = None) -> str:
    if hint:
        return hint.upper()
    return "NFO" if _is_option(sym) else "NSE"


def _exchange_candidates(sym: str, hint: Optional[str]) -> List[str]:
    first = _guess_exchange(sym, hint)
    return ["NFO", "NSE", "BSE"] if first == "NFO" else ["NSE", "BSE", "NFO"]


def _find_real_symbol(sym: str) -> str:
    try:
        s = sym.upper()
        m = INSTR[INSTR["tradingsymbol"] == s]
        if not m.empty:
            return s
        m2 = INSTR[INSTR["tradingsymbol"].str.contains(s, regex=False)]
        if not m2.empty:
            mapped = m2.iloc[0]["tradingsymbol"]
            print(f"ℹ️ Auto-mapped {sym} -> {mapped}")
            return mapped
    except:
        pass
    return sym.upper()


def _fallback_price_from_depth(depth: Dict[str, Any]):
    if not isinstance(depth, dict):
        return None
    bids = [b.get("price") for b in depth.get("buy", []) if b.get("price")]
    asks = [a.get("price") for a in depth.get("sell", []) if a.get("price")]
    if bids and asks:
        return round((max(bids) + min(asks)) / 2, 2)
    if bids:
        return _safe_float(max(bids))
    if asks:
        return _safe_float(min(asks))
    return None


def _calc_change(price, ohlc):
    if price is None:
        return None, None
    prev = _safe_float((ohlc or {}).get("close"))
    if prev is None:
        h = _safe_float((ohlc or {}).get("high"))
        l = _safe_float((ohlc or {}).get("low"))
        if h and l:
            prev = (h + l) / 2
    if prev:
        chg = price - prev
        pct = (chg / prev * 100)
        return chg, pct
    return None, None


# =============================================================
# REST FALLBACKS
# =============================================================
def _kite_ltp_try(sym: str, candidates):
    if not kite:
        return None, None
    keys = [f"{ex}:{sym}" for ex in candidates]
    try:
        data = kite.ltp(keys)
        for k in keys:
            lp = _safe_float((data.get(k) or {}).get("last_price"))
            if lp is not None:
                return lp, k.split(":")[0]
    except:
        pass
    return None, None


def _kite_quote_try(sym: str, candidates):
    if not kite:
        return None, None, {}
    for ex in candidates:
        key = f"{ex}:{sym}"
        try:
            data = kite.quote(key)
            item = data.get(key) or {}
            lp = _safe_float(item.get("last_price") or item.get("last_traded_price"))
            if lp is None:
                lp = _fallback_price_from_depth(item.get("depth", {}))
            if lp:
                return lp, ex, item.get("ohlc") or {}
        except:
            pass
    return None, None, {}


# =============================================================
# MAIN QUOTES ENDPOINT
# =============================================================
@router.get("")
async def quotes(
    symbols: str = Query(...),
    exchange: Optional[str] = Query(None),
):
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(400, "No symbols provided")

    out = []

    for raw in syms:
        sym = _find_real_symbol(raw)

        # Cache hit? (per symbol + exchange hint)
        cache_key = (sym, (exchange.upper() if exchange else None))
        now = time.time()
        cached = _QUOTES_CACHE.get(cache_key)
        if cached and (now - cached[0]) <= _QUOTES_TTL_SECONDS:
            out.append(cached[1])
            continue

        candidates = _exchange_candidates(sym, exchange)

        price = None
        ex_used = None
        ohlc = {}
        day_high = None
        day_low = None
        change = None
        pct = None

        # A) WebSocket
        try:
            tick = get_quote(sym)
            if tick:
                p = _safe_float(tick.get("last_price"))
                if p:
                    price = p
                    ohlc = tick.get("ohlc") or {}
        except:
            pass

        # B) Quote()  (preferred: returns OHLC, so UI can show Change + Day's Range)
        if price is None:
            p2, ex2, ohlc2 = _kite_quote_try(sym, candidates)
            if p2:
                price = p2
                ex_used = ex2
                ohlc = ohlc2

        # C) LTP (fallback)
        if price is None:
            p, ex = _kite_ltp_try(sym, candidates)
            if p:
                price = p
                ex_used = ex

        # D) WebSocket subscribe (no blocking sleep in request)
        if price is None:
            try:
                subscribe_symbol(sym)
                tick = get_quote(sym)
                p = _safe_float(tick.get("last_price")) if tick else None
                if p:
                    price = p
                    ohlc = tick.get("ohlc") or {}
            except:
                pass

        # Final OHLC details
        if ohlc:
            day_high = _safe_float(ohlc.get("high"))
            day_low = _safe_float(ohlc.get("low"))
            change, pct = _calc_change(price, ohlc)

        # =====================================================
        # 📌 *** FIX: Day Range values for UI ***
        # Adds BOTH snake_case + camelCase keys
        # =====================================================
        payload = {
            "symbol": sym,
            "exchange": ex_used or _guess_exchange(sym),

            "price": round(price, 2) if price else None,
            "change": round(change, 2) if change else None,
            "pct_change": round(pct, 2) if pct else None,

            # Snake_case
            "day_high": round(day_high, 2) if isinstance(day_high, (float, int)) else None,
            "day_low": round(day_low, 2) if isinstance(day_low, (float, int)) else None,

            # CamelCase (UI compatibility)
            "dayHigh": round(day_high, 2) if isinstance(day_high, (float, int)) else None,
            "dayLow":  round(day_low, 2) if isinstance(day_low, (float, int)) else None,
        }

        _QUOTES_CACHE[cache_key] = (now, payload)

        out.append(payload)

    return out


# =============================================================
# Utility Live Price Reader
# =============================================================
def get_live_price(symbol: str, exchange: str = "NSE") -> Optional[float]:
    sym = _find_real_symbol(symbol)
    candidates = _exchange_candidates(sym, exchange)

    # WebSocket
    try:
        t = get_quote(sym)
        if t and t.get("last_price"):
            return _safe_float(t["last_price"])
    except:
        pass

    # Prefer quote() because it may include OHLC; fallback to LTP.
    p2, _, _ = _kite_quote_try(sym, candidates)
    if p2:
        return p2

    p, _ = _kite_ltp_try(sym, candidates)
    if p:
        return p

    return None
