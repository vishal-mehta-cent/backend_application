from fastapi import APIRouter, Query
from typing import List, Optional, Tuple, Dict, Any
import os
import pandas as pd
import re
import threading

router = APIRouter(prefix="/search", tags=["search"])

# ---- Optional: Zerodha-loaded DFs (if present) ----
try:
    from app.services.kite_ws_manager import EQUITY_DF  # type: ignore
except Exception:
    EQUITY_DF = None  # type: ignore

try:
    from app.services.kite_ws_manager import INSTRUMENTS_DF  # type: ignore
except Exception:
    INSTRUMENTS_DF = None  # type: ignore


# ---------- PATH HELPERS ----------
ROUTERS_DIR = os.path.dirname(os.path.abspath(__file__))           # .../app/routers
APP_DIR = os.path.abspath(os.path.join(ROUTERS_DIR, ".."))         # .../app
PROJECT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))         # .../backend
ROOT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, ".."))        # repo root (if backend/ exists)

NEEDED_COLS = ["tradingsymbol", "name", "segment", "instrument_type", "exchange"]
NEEDED_COLS_SET = set(NEEDED_COLS)

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

_LOCK = threading.Lock()

_MASTER: Optional[pd.DataFrame] = None
_MASTER_PATH: Optional[str] = None
_MASTER_SIG: Optional[Tuple[float, int]] = None  # (mtime, size)


def _is_render() -> bool:
    # Render sets RENDER env var; also /data generally exists only on the service with a disk mounted.
    return bool(os.getenv("RENDER")) or os.path.exists("/data")


def _norm_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    s = str(p).strip()
    if not s:
        return None
    # if user mistakenly writes \data\instruments.csv
    s = s.replace("\\", "/")
    # normalize repeated slashes
    s = re.sub(r"/{2,}", "/", s)
    return s


def _candidate_paths() -> List[str]:
    """
    IMPORTANT BEHAVIOR:
    - If INSTRUMENTS_CSV_PATH is set, it wins everywhere.
    - On Render, /data/instruments.csv is preferred BEFORE any repo/local fallback.
    - On Local, common repo paths are preferred.
    """
    env_path = _norm_path(os.getenv("INSTRUMENTS_CSV_PATH"))
    is_render = _is_render()

    candidates: List[str] = []
    if env_path:
        candidates.append(env_path)

    if is_render:
        # Render persistent disk FIRST
        candidates += [
            "/data/instruments.csv",
            "/data/instruments.csv.gz",
        ]

        # then fallbacks (repo)
        candidates += [
            os.path.join(APP_DIR, "data", "instruments.csv"),
            os.path.join(APP_DIR, "instruments.csv"),
            os.path.join(PROJECT_DIR, "app", "data", "instruments.csv"),
            os.path.join(PROJECT_DIR, "instruments.csv"),
            os.path.join(PROJECT_DIR, "data", "instruments.csv"),
            os.path.join(ROOT_DIR, "instruments.csv"),
            os.path.join(ROOT_DIR, "backend", "app", "data", "instruments.csv"),
        ]
    else:
        # Local common locations FIRST
        candidates += [
            os.path.join(APP_DIR, "data", "instruments.csv"),
            os.path.join(APP_DIR, "instruments.csv"),
            os.path.join(PROJECT_DIR, "app", "data", "instruments.csv"),
            os.path.join(PROJECT_DIR, "instruments.csv"),
            os.path.join(PROJECT_DIR, "data", "instruments.csv"),
            os.path.join(ROOT_DIR, "instruments.csv"),
            os.path.join(ROOT_DIR, "backend", "app", "data", "instruments.csv"),
            # allow /data even locally if user created it
            "/data/instruments.csv",
        ]

    # normalize + de-dup (preserve order)
    seen = set()
    out: List[str] = []
    for p in candidates:
        p2 = _norm_path(p)
        if not p2 or p2 in seen:
            continue
        seen.add(p2)
        out.append(p2)
    return out


def _pick_instruments_file() -> str:
    for p in _candidate_paths():
        if p and os.path.exists(p):
            return p

    # Fallback used in /search/meta if nothing exists:
    return "/data/instruments.csv" if _is_render() else os.path.join(APP_DIR, "data", "instruments.csv")


def _file_sig(path: str) -> Optional[Tuple[float, int]]:
    try:
        return (os.path.getmtime(path), os.path.getsize(path))
    except Exception:
        return None


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _alpha(s: str) -> str:
    return re.sub(r"[^a-z]+", "", (s or "").lower())


def _has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in (s or ""))


def _tokenize(term: str) -> List[str]:
    raw = (term or "").lower().strip()
    return [p for p in re.split(r"[\s\-_./]+", raw) if p]


def _safe_df(df) -> pd.DataFrame:
    out = pd.DataFrame(columns=NEEDED_COLS) if df is None else df.copy()

    # normalize col names
    out.columns = [str(c).strip() for c in out.columns]
    cmap = {str(c).strip().lower(): str(c).strip() for c in out.columns}

    def pick(col: str) -> pd.Series:
        src = cmap.get(col)
        if src is None:
            return pd.Series([""] * len(out))
        return out[src]

    safe = pd.DataFrame({c: pick(c) for c in NEEDED_COLS})
    for c in NEEDED_COLS:
        safe[c] = safe[c].fillna("").astype(str)

    safe["exchange"] = safe["exchange"].astype(str).str.upper()
    safe.loc[safe["exchange"] == "", "exchange"] = "NSE"
    return safe[NEEDED_COLS]


def _indices_from_ws_manager() -> pd.DataFrame:
    if INSTRUMENTS_DF is None:
        return _safe_df(None)
    df = _safe_df(INSTRUMENTS_DF)
    if df.empty:
        return df
    mask = (
        df["instrument_type"].str.upper().eq("INDEX")
        | df["segment"].str.upper().str.contains("INDICES", na=False)
    )
    out = df.loc[mask].copy()
    out["name"] = out["name"].where(out["name"].str.len() > 0, out["tradingsymbol"])
    return out.drop_duplicates(subset=["exchange", "tradingsymbol"]).reset_index(drop=True)


def _indices_fallback() -> pd.DataFrame:
    data = [
        {"tradingsymbol": "NIFTY", "name": "NIFTY 50", "segment": "INDICES", "instrument_type": "INDEX", "exchange": "NSE"},
        {"tradingsymbol": "BANKNIFTY", "name": "NIFTY BANK", "segment": "INDICES", "instrument_type": "INDEX", "exchange": "NSE"},
        {"tradingsymbol": "SENSEX", "name": "SENSEX", "segment": "INDICES", "instrument_type": "INDEX", "exchange": "BSE"},
        {"tradingsymbol": "FINNIFTY", "name": "NIFTY FINANCIAL SERVICES", "segment": "INDICES", "instrument_type": "INDEX", "exchange": "NSE"},
    ]
    return _safe_df(pd.DataFrame(data)).drop_duplicates(subset=["exchange", "tradingsymbol"]).reset_index(drop=True)


def _read_instruments_csv(csv_path: str) -> pd.DataFrame:
    if not csv_path or not os.path.exists(csv_path):
        print(f"❌ instruments.csv NOT FOUND: {csv_path}")
        return pd.DataFrame(columns=NEEDED_COLS)

    last_err = None
    df = None

    def _read_with_cols(enc: str) -> pd.DataFrame:
        # Try to read only needed cols for speed/memory. If file headers differ, fallback to full read.
        try:
            return pd.read_csv(
                csv_path,
                dtype=str,
                low_memory=False,
                encoding=enc,
                usecols=lambda c: str(c).strip().lower() in NEEDED_COLS_SET,
            )
        except ValueError:
            return pd.read_csv(csv_path, dtype=str, low_memory=False, encoding=enc)

    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = _read_with_cols(enc)
            break
        except Exception as e:
            last_err = e

    if df is None:
        print(f"❌ instruments.csv read failed: {csv_path} err={last_err}")
        return pd.DataFrame(columns=NEEDED_COLS)

    df.columns = [str(c).strip() for c in df.columns]

    # Ensure required columns exist
    for c in NEEDED_COLS:
        if c not in df.columns:
            df[c] = ""

    print(f"✅ instruments.csv loaded: {csv_path} rows={len(df)} cols={len(df.columns)}")
    return df[NEEDED_COLS].copy()


def _build_master_df(csv_path: str) -> pd.DataFrame:
    csv_df = _safe_df(_read_instruments_csv(csv_path))
    eq_df = _safe_df(EQUITY_DF)

    eq = pd.concat([eq_df, csv_df], ignore_index=True).drop_duplicates(
        subset=["exchange", "tradingsymbol"], keep="first"
    )

    # Ensure indices exist
    idx = pd.DataFrame(columns=eq.columns)
    if eq.empty or not (
        eq["instrument_type"].str.upper().eq("INDEX").any()
        or eq["segment"].str.upper().str.contains("INDICES", na=False).any()
    ):
        idx = _indices_from_ws_manager()
        if idx.empty:
            idx = _indices_fallback()

    frames = [d for d in (eq, idx) if not d.empty]
    master = _safe_df(None) if not frames else pd.concat(frames, ignore_index=True, sort=False)
    master = master.drop_duplicates(subset=["exchange", "tradingsymbol"], keep="first").reset_index(drop=True)

    master["exchange_l"] = master["exchange"].astype(str).str.lower()
    master["symbol_l"] = master["tradingsymbol"].astype(str).str.lower()
    master["name_l"] = master["name"].astype(str).str.lower()

    master["sym_alnum"] = master["symbol_l"].map(_alnum)
    master["sym_alpha"] = master["symbol_l"].map(_alpha)
    master["name_alnum"] = master["name_l"].map(_alnum)
    master["name_alpha"] = master["name_l"].map(_alpha)

    master["__blob"] = (
        master["exchange_l"]
        + " " + master["symbol_l"]
        + " " + master["name_l"]
        + " " + master["segment"].astype(str).str.lower()
        + " " + master["instrument_type"].astype(str).str.lower()
    )

    master["__alnum"] = (
        master["exchange_l"].map(_alnum)
        + " " + master["sym_alnum"]
        + " " + master["name_alnum"]
    )

    master["__alpha"] = master["sym_alpha"] + " " + master["name_alpha"]
    master["sym_len"] = master["tradingsymbol"].astype(str).str.len()

    return master


def _get_master(refresh: bool = False) -> pd.DataFrame:
    """
    Cache + auto-reload:
    - Reloads if refresh=1
    - Reloads if selected file path changes
    - Reloads if file mtime/size changes (e.g., you replaced /data/instruments.csv)
    """
    global _MASTER, _MASTER_PATH, _MASTER_SIG

    csv_path = _pick_instruments_file()
    sig = _file_sig(csv_path) if csv_path and os.path.exists(csv_path) else None

    with _LOCK:
        if (
            refresh
            or _MASTER is None
            or csv_path != _MASTER_PATH
            or sig != _MASTER_SIG
        ):
            _MASTER = _build_master_df(csv_path)
            _MASTER_PATH = csv_path
            _MASTER_SIG = sig

        return _MASTER


def _fuzzy_pairs(single_token: str) -> List[Tuple[str, str]]:
    a = _alpha(single_token)
    if not a or len(a) < 6:
        return []
    pairs: List[Tuple[str, str]] = []
    for m in MONTHS:
        if a.endswith(m) and len(a) > len(m) + 2:
            pairs.append((a[:-len(m)], m))
    return pairs[:8]


def _rank_results(sub: pd.DataFrame, term_l: str) -> pd.DataFrame:
    term_alnum = _alnum(term_l)
    term_alpha = _alpha(term_l)
    term_has_digit = _has_digit(term_l)

    sym = sub["symbol_l"]
    nm = sub["name_l"]
    sub["_score"] = 999

    # 0) exact symbol (or exact alnum)
    exact = (sym == term_l) | ((term_alnum != "") & (sub["sym_alnum"] == term_alnum))
    sub.loc[exact, "_score"] = 0

    # 10) symbol startswith (strict)
    starts_sym = sym.str.startswith(term_l)
    sub.loc[starts_sym & (sub["_score"] > 10), "_score"] = 10

    # 15) symbol alnum startswith (handles hyphen/space differences)
    if term_alnum:
        starts_sym_alnum = sub["sym_alnum"].str.startswith(term_alnum)
        sub.loc[starts_sym_alnum & (sub["_score"] > 15), "_score"] = 15

    # 20) name startswith
    starts_name = nm.str.startswith(term_l)
    sub.loc[starts_name & (sub["_score"] > 20), "_score"] = 20

    # 60/70) contains
    contains_sym = sym.str.contains(term_l, na=False, regex=False)
    sub.loc[contains_sym & (sub["_score"] > 60), "_score"] = 60

    contains_name = nm.str.contains(term_l, na=False, regex=False)
    sub.loc[contains_name & (sub["_score"] > 70), "_score"] = 70

    # IMPORTANT: do not alpha-rank digit queries (strike/expiry)
    if term_alpha and (not term_has_digit):
        starts_sym_alpha = sub["sym_alpha"].str.startswith(term_alpha)
        sub.loc[starts_sym_alpha & (sub["_score"] > 18), "_score"] = 18

    return sub


# ✅ quick test that router is mounted
@router.get("/ping")
@router.get("/ping/")
def ping():
    return {"ok": True, "router": "search"}


# ✅ Serve both /search and /search/
@router.get("", response_model=List[dict])
@router.get("/", response_model=List[dict])
def search_scripts(
    q: Optional[str] = Query(None),
    refresh: Optional[int] = None,
    limit: int = Query(50, ge=1, le=500),
):
    if not q:
        return []

    df = _get_master(refresh=bool(refresh))
    if df.empty:
        return []

    term = (q or "").strip().lower()
    if not term:
        return []

    tokens = _tokenize(term)
    if not tokens:
        return []

    # If query is 1-2 chars, only show prefix matches (reduces noise)
    t_alnum = _alnum(term)
    if len(t_alnum) <= 2 and t_alnum:
        sub = df[df["sym_alnum"].str.startswith(t_alnum, na=False)].copy()
        sub = _rank_results(sub, term)
        sub = sub.sort_values(by=["_score", "sym_len", "tradingsymbol", "exchange"]).head(limit)
        return [
            {
                "symbol": r["tradingsymbol"],
                "name": r.get("name", ""),
                "segment": r.get("segment", ""),
                "instrument_type": r.get("instrument_type", ""),
                "exchange": r.get("exchange", ""),
                "display_name": f"{r['tradingsymbol']} ({r.get('exchange','')}) | {r.get('segment','')} | {r.get('instrument_type','')}",
            }
            for _, r in sub.iterrows()
        ]

    mask = pd.Series(True, index=df.index)

    for tok in tokens:
        tok_l = tok.lower().strip()
        if not tok_l:
            continue

        tok_alnum = _alnum(tok_l)
        tok_alpha = _alpha(tok_l)
        tok_has_digit = _has_digit(tok_l)

        token_mask = df["__blob"].str.contains(tok_l, na=False, regex=False)

        if tok_alnum:
            token_mask |= df["__alnum"].str.contains(tok_alnum, na=False, regex=False)

        # KEY FIX: if token has digits, DO NOT alpha-match it
        if tok_alpha and (not tok_has_digit):
            token_mask |= df["__alpha"].str.contains(tok_alpha, na=False, regex=False)

        mask &= token_mask

    sub = df.loc[mask].copy()

    # Fuzzy fallback for single-token like "niftyjan"
    if sub.empty and len(tokens) == 1:
        pairs = _fuzzy_pairs(tokens[0])
        if pairs:
            m2 = pd.Series(False, index=df.index)
            for a, b in pairs:
                m2 |= (
                    df["__alpha"].str.contains(a, na=False, regex=False)
                    & df["__alpha"].str.contains(b, na=False, regex=False)
                )
            sub = df.loc[m2].copy()

    if sub.empty:
        return []

    sub = _rank_results(sub, term)

    sub = sub.sort_values(
        by=["_score", "sym_len", "tradingsymbol", "exchange"],
        ascending=[True, True, True, True],
    ).head(limit)

    return [
        {
            "symbol": r["tradingsymbol"],
            "name": r.get("name", ""),
            "segment": r.get("segment", ""),
            "instrument_type": r.get("instrument_type", ""),
            "exchange": r.get("exchange", ""),
            "display_name": f"{r['tradingsymbol']} ({r.get('exchange','')}) | {r.get('segment','')} | {r.get('instrument_type','')}",
        }
        for _, r in sub.iterrows()
    ]


@router.get("/scripts")
@router.get("/scripts/")
def list_scripts(
    refresh: Optional[int] = None,
    limit: int = Query(5000, ge=1, le=50000),
):
    df = _get_master(refresh=bool(refresh)).copy()
    if df.empty:
        return []
    df = df.sort_values(by=["tradingsymbol", "exchange"]).head(limit)
    return [
        {
            "symbol": r["tradingsymbol"],
            "name": r.get("name", ""),
            "segment": r.get("segment", ""),
            "instrument_type": r.get("instrument_type", ""),
            "exchange": r.get("exchange", ""),
            "display_name": f"{r['tradingsymbol']} ({r.get('exchange','')}) | {r.get('segment','')} | {r.get('instrument_type','')}",
        }
        for _, r in df.iterrows()
    ]


@router.get("/meta")
@router.get("/meta/")
def search_meta(refresh: Optional[int] = None, load: Optional[int] = None):
    """
    Lightweight metadata endpoint.

    NOTE: Previously this endpoint loaded the full instruments master DF, which can take time on Render
    (especially on cold start). Now it only loads the master DF when explicitly requested.

    - load=1   -> build/return master_rows
    - refresh=1 + load=1 -> force rebuild
    """

    df = None
    if bool(load):
        df = _get_master(refresh=bool(refresh))

    picked = _MASTER_PATH or _pick_instruments_file()
    sig = _MASTER_SIG or (_file_sig(picked) if picked and os.path.exists(picked) else None)
    mtime = sig[0] if sig else None
    size = sig[1] if sig else None

    return {
        "is_render": _is_render(),
        "env_INSTRUMENTS_CSV_PATH": _norm_path(os.getenv("INSTRUMENTS_CSV_PATH")),
        "picked_csv_path": picked,
        "picked_csv_exists": bool(picked and os.path.exists(picked)),
        "picked_csv_mtime": mtime,
        "picked_csv_size_bytes": size,
        "picked_csv_size_mb": (round(size / (1024 * 1024), 3) if size else 0),
        "master_rows": int(len(df)) if df is not None else 0,
        "candidates": _candidate_paths(),
    }
