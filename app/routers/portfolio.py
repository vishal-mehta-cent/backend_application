# backend/app/routers/portfolio.py
from fastapi import APIRouter, HTTPException, UploadFile, File
import sqlite3
from typing import Dict, Any
from datetime import datetime
import requests
import pandas as pd
from fastapi.responses import StreamingResponse
from io import BytesIO
import os
import re
from app.routers.orders import _ensure_orders_position_type, _ensure_tables, BrokerageSettings, _run_eod_if_due

# ---------- Market-close helpers ----------
# Portfolio should NOT show same-day DELIVERY trades during live market.
# Only after market close should delivery-based open positions move to Portfolio.
try:
    from app.routers.orders import is_after_market_close as _orders_is_after_market_close  # type: ignore
except Exception:
    _orders_is_after_market_close = None


def _market_today_str() -> str:
    """Return today's date in market timezone (default Asia/Kolkata) as YYYY-MM-DD."""
    try:
        from pytz import timezone as _pytz_timezone
        tz_name = os.getenv("MARKET_TZ", "Asia/Kolkata")
        tz = _pytz_timezone(tz_name)
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def _include_today_closed_orders() -> bool:
    """
    When False (market open), exclude today's delivery orders from seeding/reconcile so
    live trades don't leak into Portfolio. When True (after market close), allow today's
    delivery trades to carry-forward into Portfolio.
    """
    # Prefer orders.py's market-close logic if present
    try:
        if _orders_is_after_market_close:
            return bool(_orders_is_after_market_close())
    except Exception:
        pass

    # Fallback: IST 15:45 cutoff (configurable)
    try:
        from pytz import timezone as _pytz_timezone
        tz = _pytz_timezone(os.getenv("MARKET_TZ", "Asia/Kolkata"))
        now_tz = datetime.now(tz)
        cutoff_h = int(os.getenv("EOD_CUTOFF_HOUR", "15") or 15)
        cutoff_m = int(os.getenv("EOD_CUTOFF_MINUTE", "45") or 45)
        return (now_tz.hour, now_tz.minute) >= (cutoff_h, cutoff_m)
    except Exception:
        return False



router = APIRouter(prefix="/portfolio", tags=["portfolio"])

#DB_PATH = "/data/paper_trading.db"
IS_RENDER = bool(os.getenv("RENDER"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if IS_RENDER:
    # Render persistent disk
    DB_PATH = "/data/paper_trading.db"
else:
    # Local backend/app/data
    DB_PATH = os.path.join(BASE_DIR, "data", "paper_trading.db")

print("📍 Portfolio DB_PATH =", DB_PATH)


# ---------- Brokerage helpers (backend provides; matches Orders.jsx) ----------
def _to_f(v, fallback: float = 0.0) -> float:
    try:
        n = float(v)
        return n if (n == n and n not in (float("inf"), float("-inf"))) else fallback
    except Exception:
        return fallback


def _calc_additional_cost(rates: Dict[str, Any], segment: str, investment: float) -> float:
    seg = (segment or "delivery").lower()
    is_intra = seg == "intraday"

    mode = (rates.get("brokerage_mode") or "ABS").upper()

    if mode == "PCT":
        pct = _to_f(
            rates.get("brokerage_intraday_pct" if is_intra else "brokerage_delivery_pct"),
            0.0,
        )
        brokerage = investment * pct
    else:
        brokerage = _to_f(
            rates.get("brokerage_intraday_abs" if is_intra else "brokerage_delivery_abs"),
            0.0,
        )

    tax_pct = _to_f(rates.get("tax_intraday_pct" if is_intra else "tax_delivery_pct"), 0.0)
    tax = investment * tax_pct

    additional = brokerage + tax
    return round(additional, 2) if additional == additional else 0.0


def _get_brokerage_settings(conn: sqlite3.Connection, username: str) -> Dict[str, Any]:
    """Fetch brokerage settings for user; insert defaults if missing."""
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # ensure brokerage_settings table exists (created in orders._ensure_tables)
    _ensure_tables(c)

    c.execute("SELECT * FROM brokerage_settings WHERE username=?", (username,))
    row = c.fetchone()

    if not row:
        defaults = BrokerageSettings().dict()
        c.execute(
            """
            INSERT OR IGNORE INTO brokerage_settings
              (username, brokerage_mode,
               brokerage_intraday_pct, brokerage_intraday_abs,
               brokerage_delivery_pct, brokerage_delivery_abs,
               tax_intraday_pct, tax_delivery_pct, updated_at)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
            """,
            (
                username,
                defaults["brokerage_mode"],
                defaults["brokerage_intraday_pct"],
                defaults["brokerage_intraday_abs"],
                defaults["brokerage_delivery_pct"],
                defaults["brokerage_delivery_abs"],
                defaults["tax_intraday_pct"],
                defaults["tax_delivery_pct"],
            ),
        )
        conn.commit()
        c.execute("SELECT * FROM brokerage_settings WHERE username=?", (username,))
        row = c.fetchone()

    d = dict(row) if row else BrokerageSettings().dict()
    d.pop("updated_at", None)
    return d


# ✅ Auto-detect base URL for live backend (works on localhost + Render)
DEFAULT_BACKEND = "https://api.neurocrest.in"
QUOTES_API = os.getenv("VITE_BACKEND_BASE_URL", DEFAULT_BACKEND).rstrip("/") + "/quotes?symbols="


@router.get("/{username}/download")
def download_portfolio(username: str):
    """
    Download portfolio Excel with Instrument sheet populated from
    backend/app/instruments.csv (tradingsymbol + name only).
    """
    import logging
    logger = logging.getLogger("uvicorn.error")

    try:
        # ---------- DB READ (READ-ONLY) ----------
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute(
                "SELECT script, qty, avg_buy_price FROM portfolio WHERE username=? AND qty!=0",
                (username,),
            )
            rows = c.fetchall()

        # ---------- Instruction sheet ----------
        df_instructions = pd.DataFrame([
            ["** Please search the script you want to manually add in the Instruments tab"],
            ["** Copy the tradingsymbol and paste in the Portfolio sheet"],
            ["** If formulas are not applied automatically, fill-down from above cells"],
            ["** Segment defaults to BSE if same symbol exists in BSE/NSE"],
            ["** Only Yellow cells should be edited"],
        ])

        # ---------- Portfolio sheet ----------
        portfolio_data = []
        for r in rows:
            symbol = (r["script"] or "").upper().strip()
            qty = int(r["qty"] or 0)
            avg_price = float(r["avg_buy_price"] or 0.0)
            if not symbol or qty == 0 or avg_price <= 0:
                continue

            portfolio_data.append({
                "Symbol": symbol,
                "Name": "",
                "Segment": "BSE",
                "Side": "SELL" if qty < 0 else "BUY",
                "Qty": abs(qty),
                "Avg Price": avg_price,
                "Entry Price": avg_price,
                "Stoploss": 0,
                "Target": 0,
                "Live": avg_price,
                "Investment": abs(qty) * avg_price,
            })


        PORTFOLIO_COLUMNS = [
            "Symbol",
            "Name",
            "Segment",
            "Side",
            "Qty",
            "Avg Price",
            "Entry Price",
            "Stoploss",
            "Target",
            "Live",
            "Investment",
        ]

        df_portfolio = pd.DataFrame(portfolio_data, columns=PORTFOLIO_COLUMNS)

        
        # ---------- Instruments sheet ----------
        instruments_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "instruments.csv"
        )

        if not os.path.exists(instruments_path):
            raise HTTPException(
                status_code=500,
                detail=f"instruments.csv not found at {instruments_path}"
            )

        df_inst = pd.read_csv(instruments_path, usecols=["tradingsymbol", "name","instrument_type"])
        df_inst["tradingsymbol"] = df_inst["tradingsymbol"].astype(str).str.upper().str.strip()
        df_inst["name"] = df_inst["name"].astype(str).str.strip()
        df_inst = df_inst[(df_inst["name"] != "") & (df_inst["instrument_type"] == "EQ")]
        df_inst = df_inst.drop_duplicates(subset=["tradingsymbol"])

        # ---------- Write Excel ----------
        output = BytesIO()
        with pd.ExcelWriter(output) as writer:
            df_instructions.to_excel(writer, index=False, header=False, sheet_name="Instruction")
            df_portfolio.to_excel(writer, index=False, sheet_name="Portfolio")
            df_inst.to_excel(writer, index=False, sheet_name="Instrument")

        output.seek(0)

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="portfolio_{username}.xlsx"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ download_portfolio failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- DB helpers ----------
def _ensure_portfolio_schema(conn: sqlite3.Connection) -> None:
    c = conn.cursor()

    _dedupe_and_enforce_unique_portfolio(conn)

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        script TEXT NOT NULL,
        qty INTEGER NOT NULL,
        avg_buy_price REAL NOT NULL,
        current_price REAL NOT NULL DEFAULT 0,
        updated_at TEXT,
        UNIQUE(username, script)
        )
        """
    )

    c.execute("PRAGMA table_info(portfolio)")
    cols = {row[1].lower() for row in c.fetchall()}
    if "current_price" not in cols:
        c.execute("ALTER TABLE portfolio ADD COLUMN current_price REAL NOT NULL DEFAULT 0")
    if "updated_at" not in cols:
        c.execute("ALTER TABLE portfolio ADD COLUMN updated_at TEXT")
    conn.commit()

from datetime import datetime

def _parse_dt(s: str):
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None

def _dedupe_and_enforce_unique_portfolio(conn: sqlite3.Connection) -> None:
    c = conn.cursor()

    # pull everything once
    c.execute("""
        SELECT id, username, script, qty, avg_buy_price, current_price, updated_at
        FROM portfolio
    """)
    rows = c.fetchall()
    if not rows:
        return

    # group by (username, normalized script)
    grouped = {}
    for _id, username, script, qty, avg_buy_price, current_price, updated_at in rows:
        u = (username or "").strip()
        s_raw = (script or "").strip()
        s = s_raw.upper()

        key = (u, s)
        qty = int(qty or 0)
        abp = float(avg_buy_price or 0.0)
        cp = float(current_price or 0.0)
        dt = _parse_dt(updated_at)

        if key not in grouped:
            grouped[key] = {
                "username": u,
                "script": s,  # store normalized
                "qty_sum": 0,
                "w_sum": 0.0,
                "w_qty": 0.0,
                "current_price": cp,
                "latest_dt": dt,
                "latest_raw": updated_at,
            }

        g = grouped[key]
        g["qty_sum"] += qty

        w = abs(qty)
        g["w_sum"] += w * abp
        g["w_qty"] += w

        if cp > g["current_price"]:
            g["current_price"] = cp

        # keep latest updated_at by parsed datetime (fallback: keep existing if parse fails)
        if dt and (g["latest_dt"] is None or dt > g["latest_dt"]):
            g["latest_dt"] = dt
            g["latest_raw"] = updated_at

    # detect actual duplicates (same key appearing multiple times)
    keys_count = {}
    for _, username, script, *_ in rows:
        key = ((username or "").strip(), (script or "").strip().upper())
        keys_count[key] = keys_count.get(key, 0) + 1
    has_dups = any(n > 1 for n in keys_count.values())
    if not has_dups:
        return

    print("⚠️ Duplicates found in portfolio -> rebuilding with UNIQUE(username, script)")

    # rebuild table with UNIQUE(username, script)
    c.execute("DROP TABLE IF EXISTS portfolio_new")
    c.execute("""
        CREATE TABLE portfolio_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            script TEXT NOT NULL,
            qty INTEGER NOT NULL,
            avg_buy_price REAL NOT NULL,
            current_price REAL NOT NULL DEFAULT 0,
            updated_at TEXT,
            UNIQUE(username, script)
        )
    """)

    # insert merged rows
    for key, g in grouped.items():
        avg_price = (g["w_sum"] / g["w_qty"]) if g["w_qty"] > 0 else 0.0

        # if updated_at was unparseable or missing, stamp now (localtime)
        updated_at = g["latest_raw"]
        if not _parse_dt(updated_at):
            c.execute("SELECT datetime('now','localtime')")
            updated_at = c.fetchone()[0]

        c.execute("""
            INSERT INTO portfolio_new (username, script, qty, avg_buy_price, current_price, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (g["username"], g["script"], int(g["qty_sum"]), float(avg_price), float(g["current_price"]), updated_at))

    c.execute("DROP TABLE portfolio")
    c.execute("ALTER TABLE portfolio_new RENAME TO portfolio")
    conn.commit()



def _reconcile_portfolio_with_orders(conn: sqlite3.Connection, username: str) -> None:
    """
    Make portfolio reflect CLOSED DELIVERY trades:
      - If user sold fully -> DELETE portfolio row
      - If sold partially -> UPDATE qty
      - Also keeps avg_buy_price aligned with BUY history when available
    """
    c = conn.cursor()

    # Ensure portfolio table exists
    _ensure_portfolio_schema(conn)

    today_str = _market_today_str()

    # Seed missing portfolio rows from orders (post-market carry-forward safety)
    # _seed_portfolio_from_orders(conn, username)
    include_today_closed = _include_today_closed_orders()
    today_str = _market_today_str()
    _seed_portfolio_from_orders(conn, username, include_today_closed=include_today_closed)


    # Get all portfolio scripts for user
    c.execute("SELECT script, qty, avg_buy_price, updated_at FROM portfolio WHERE username=?", (username,))
    pf_rows = c.fetchall()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for r in pf_rows:
        script = (r[0] or "").upper().strip()
        pf_qty = int(r[1] or 0)
        pf_avg = float(r[2] or 0.0)
        pf_dt  = r[3] or now

        if not script:
            continue

        # ✅ IMPORTANT FIX:
        # Instead of recomputing net from entire orders history (which overwrites manual/existing holdings),
        # apply ONLY the NEW closed delivery orders since portfolio.updated_at.

        # Normalize updated_at for SQLite datetime()
        pf_dt_norm = str(pf_dt or now).replace("T", " ")[:19]

        if pf_qty > 0:
            # Apply NEW deltas after last sync
            c.execute("""
                SELECT
                  COALESCE(SUM(CASE WHEN UPPER(order_type)='BUY'  THEN qty ELSE 0 END), 0) AS buy_qty,
                  COALESCE(SUM(CASE WHEN UPPER(order_type)='SELL' THEN qty ELSE 0 END), 0) AS sell_qty,
                  COALESCE(SUM(CASE WHEN UPPER(order_type)='BUY'  THEN qty*price ELSE 0 END), 0) AS buy_value
                FROM orders
                WHERE username=?
                  AND UPPER(script)=?
                  AND UPPER(COALESCE(status,'')) IN ('CLOSED','SETTLED')
                  AND LOWER(COALESCE(segment,''))='delivery'
                  AND ( ? = 1 OR substr(datetime,1,10) < ? )
                  AND datetime(substr(datetime,1,19)) > datetime(?)
            """, (username, script, 1 if include_today_closed else 0, today_str, pf_dt_norm))

            buy_delta, sell_delta, buy_value_delta = c.fetchone() or (0, 0, 0)
            buy_delta = int(buy_delta or 0)
            sell_delta = int(sell_delta or 0)
            buy_value_delta = float(buy_value_delta or 0.0)

            net = pf_qty + buy_delta - sell_delta

            # Update avg only when there is BUY delta
            new_avg = pf_avg
            if buy_delta > 0:
                new_avg = ((pf_qty * pf_avg) + buy_value_delta) / float(pf_qty + buy_delta)

        else:
            # For shorts or non-manual rows, keep your old logic based on full history
            c.execute("""
                SELECT
                  COALESCE(SUM(CASE WHEN UPPER(order_type)='BUY'  THEN qty ELSE 0 END), 0) AS buy_qty,
                  COALESCE(SUM(CASE WHEN UPPER(order_type)='SELL' THEN qty ELSE 0 END), 0) AS sell_qty,
                  COALESCE(SUM(CASE WHEN UPPER(order_type)='BUY'  THEN qty*price ELSE 0 END), 0) AS buy_value
                FROM orders
                WHERE username=?
                  AND UPPER(script)=?
                  AND UPPER(COALESCE(status,'')) IN ('CLOSED','SETTLED')
                  AND LOWER(COALESCE(segment,''))='delivery'
                  AND ( ? = 1 OR substr(datetime,1,10) < ? )
            """, (username, script, 1 if include_today_closed else 0, today_str))
            buy_qty, sell_qty, buy_value = c.fetchone() or (0, 0, 0)

            buy_qty = int(buy_qty or 0)
            sell_qty = int(sell_qty or 0)
            buy_value = float(buy_value or 0.0)

            net = buy_qty - sell_qty

            new_avg = pf_avg
            if buy_qty > 0 and buy_value > 0:
                new_avg = float(buy_value) / float(buy_qty)


        # ✅ Delete only when fully flat (net == 0). Shorts have net < 0 and must remain.
        if net == 0:
            c.execute("DELETE FROM portfolio WHERE username=? AND script=?", (username, script))
        else:

            c.execute("""
                UPDATE portfolio
                SET qty=?, avg_buy_price=?, updated_at=?
                WHERE username=? AND script=?
            """, (int(net), float(new_avg), now, username, script))

            conn.commit()


def _seed_portfolio_from_orders(conn, username: str, include_today_closed: bool = False):
    
    """
    Seed missing portfolio rows from executed DELIVERY orders.

    Why:
    Sometimes EOD / post-market logic clears the "open positions" view,
    but the portfolio table doesn't get populated (so the Portfolio page looks empty).

    This helper *only* inserts symbols that are not already present in portfolio for the user.
    """
    today_str = _market_today_str()
    c = conn.cursor()
    _ensure_portfolio_schema(conn)

    # Existing scripts in portfolio (so we don't overwrite manual uploads)
    try:
        c.execute("SELECT UPPER(script) AS script FROM portfolio WHERE username=?", (username,))
        existing = set()
        for r in (c.fetchall() or []):
            s = r["script"] if isinstance(r, sqlite3.Row) else r[0]
            if s:
                existing.add(str(s).upper())
    except Exception:
        existing = set()

    # Aggregate executed DELIVERY orders -> net position
    try:
        c.execute(
            """
            SELECT
                UPPER(TRIM(script)) AS script,
                SUM(CASE WHEN UPPER(order_type)='BUY'  THEN CAST(qty AS REAL) ELSE 0 END) AS buy_qty,
                SUM(CASE WHEN UPPER(order_type)='SELL' THEN CAST(qty AS REAL) ELSE 0 END) AS sell_qty,
                SUM(CASE WHEN UPPER(order_type)='BUY'
                        THEN CAST(qty AS REAL) * CAST(COALESCE(price, 0) AS REAL)
                        ELSE 0 END) AS buy_value,
                SUM(CASE WHEN UPPER(order_type)='SELL'
                        THEN CAST(qty AS REAL) * CAST(COALESCE(price, 0) AS REAL)
                        ELSE 0 END) AS sell_value
            FROM orders
            WHERE username=?
            AND LOWER(COALESCE(segment,''))='delivery'
            AND LOWER(COALESCE(status,'')) IN ('closed','settled','complete','completed','executed','filled')
            AND ( ? = 1 OR substr(datetime,1,10) < ? )
            GROUP BY UPPER(TRIM(script))
            """,
            (username, 1 if include_today_closed else 0, today_str),
        )
        
        rows = c.fetchall() or []
    except Exception:
        return

    now = datetime.now().isoformat(timespec="seconds")

    for r in rows:
        script = r["script"] if isinstance(r, sqlite3.Row) else r[0]
        if not script:
            continue
        script = str(script).upper()

        if script in existing:
            continue

        buy_qty = float((r["buy_qty"] if isinstance(r, sqlite3.Row) else r[1]) or 0)
        sell_qty = float((r["sell_qty"] if isinstance(r, sqlite3.Row) else r[2]) or 0)
        buy_value = float((r["buy_value"] if isinstance(r, sqlite3.Row) else r[3]) or 0)
        sell_value = float((r["sell_value"] if isinstance(r, sqlite3.Row) else r[4]) or 0)

        net = buy_qty - sell_qty
        if abs(net) < 1e-9:
            continue

        # Entry price: longs -> avg buy, shorts -> avg sell
        if net > 0 and buy_qty > 0:
            entry = buy_value / buy_qty
        elif net < 0 and sell_qty > 0:
            entry = sell_value / sell_qty
        else:
            entry = 0.0

        # Upsert
        try:
            c.execute(
                """
                INSERT INTO portfolio (username, script, qty, avg_buy_price, current_price, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(username, script) DO UPDATE SET
                    qty=excluded.qty,
                    avg_buy_price=excluded.avg_buy_price,
                    current_price=excluded.current_price,
                    updated_at=excluded.updated_at
                """,
                (username, script, net, entry, entry, now),
            )
        except Exception:
            # Fallback if ON CONFLICT isn't available / unique constraint missing
            try:
                c.execute(
                    "INSERT INTO portfolio (username, script, qty, avg_buy_price, current_price, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (username, script, net, entry, entry, now),
                )
            except Exception:
                pass

    conn.commit()

# ---------- Live price fetch ----------
def _get_live_price(symbol: str) -> float:
    """
    Fetch live price from the backend /quotes endpoint.
    Works in both localhost and Render environments.
    """
    try:
        url = QUOTES_API + symbol
        r = requests.get(url, timeout=5)
        arr = r.json() or []
        if arr and isinstance(arr[0], dict):
            px = arr[0].get("price")
            if px is not None and px > 0:
                return float(px)
    except Exception as e:
        print(f"⚠️ Live price fetch failed for {symbol}: {e}")
    return 0.0


# ---------- Flexible header helpers (NEW) ----------
def _norm_col(s: str) -> str:
    """lowercase + remove spaces/underscores/punctuation so headers match flexibly."""
    if s is None:
        return ""
    s = s.strip().lower()
    return re.sub(r"[^a-z0-9]", "", s)

def _flex_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map many possible column names to canonical: symbol, qty, avg_price.
    Accepts case/space/underscore variants and common synonyms.
    """
    norm2orig = {_norm_col(c): c for c in df.columns}

    want = {
        "symbol": {
            "symbol","script","ticker","tradingsymbol","trading_symbol",
            "scrip","stock","code","name"
        },
        "qty": {
            "qty","quantity","shares","units","qnty","noofshares","noofunits"
        },
        "avg_price": {
            "avgprice","avg_price","averageprice","average_price","avgbuyprice",
            "avg buy price","avg price","buyprice","entryprice","entry","price"
        },
    }
    want_norm = {k: {_norm_col(x) for x in v} for k, v in want.items()}

    rename = {}
    missing = []
    for canonical, options in want_norm.items():
        match = None
        for opt in options:
            if opt in norm2orig:
                match = norm2orig[opt]
                break
        if match is None:
            missing.append(canonical)
        else:
            rename[match] = canonical

    if missing:
        raise HTTPException(
            status_code=400,
            detail="Excel must have columns: symbol, qty, avg_price",
        )

    df = df.rename(columns=rename)

    # Clean values
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    # numeric coercion with comma cleanup
    def _to_num(series):
        return pd.to_numeric(
            series.astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

    df["qty"] = _to_num(df["qty"]).fillna(0).astype(int)
    df["avg_price"] = _to_num(df["avg_price"]).fillna(0.0).astype(float)

    # keep only valid rows
    df = df[(df["symbol"] != "") & (df["qty"] > 0) & (df["avg_price"] > 0)].copy()
    return df


# ---------- API ----------
@router.get("/{username}")
def get_portfolio(username: str) -> Dict[str, Any]:
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            _ensure_portfolio_schema(conn)
            _ensure_orders_position_type(c)   # 🔴 ADD THIS
            # IMPORTANT: Only reconcile/seed from orders AFTER market close.
            # During live market, delivery trades (including SELL FIRST) must stay in Positions, not Portfolio.
            # _reconcile_portfolio_with_orders(conn, username)   # ✅ NEW: sync delete/update holdings
            if _include_today_closed_orders():
                _run_eod_if_due(username)
                _reconcile_portfolio_with_orders(conn, username)   # ✅ sync delete/update holdings

            # ✅ Load brokerage settings once (backend provides; UI can fall back)
            rates = _get_brokerage_settings(conn, username)

            # ---- Fetch funds correctly ----
            funds = 0.0
            try:
                c.execute("SELECT available_amount FROM funds WHERE username=?", (username,))
                row_funds = c.fetchone()
                if row_funds is not None:
                    try:
                        funds = float(row_funds["available_amount"])
                    except Exception:
                        funds = float(row_funds[0])
            except Exception:
                # Fallback for older schema
                try:
                    c.execute("SELECT funds FROM users WHERE username=?", (username,))
                    r2 = c.fetchone()
                    if r2 is not None:
                        try:
                            funds = float(r2["funds"])
                        except Exception:
                            funds = float(r2[0])
                except Exception:
                    funds = 0.0

            # ---- Fetch portfolio safely ----
            c.execute("""
                SELECT
                    script,
                    qty,
                    avg_buy_price,
                    current_price,
                    updated_at
                FROM portfolio
                WHERE username = ?
                ORDER BY updated_at DESC
            """, (username,))
            rows = c.fetchall()


            open_positions = []

            for r in rows:
                symbol = (r["script"] or "").upper()
                qty = int(r["qty"] or 0)

                # ✅ SKIP CLOSED POSITIONS
                # ✅ SKIP ONLY FULLY CLOSED POSITIONS
                if qty == 0:
                    continue



                c.execute("""
                    SELECT position_type,
                           COALESCE(segment, 'delivery') AS segment
                    FROM orders
                    WHERE username=? AND script=?
                    ORDER BY datetime DESC LIMIT 1
                """, (username, symbol))

                row_pt = c.fetchone()
                position_type = row_pt[0] if row_pt else None
                segment = (row_pt[1] if row_pt and len(row_pt) > 1 else 'delivery') or 'delivery'

                if qty < 0:
                    side = "SELL FIRST"
                else:
                    side = "BUY"





                abs_qty = abs(qty)
                entry_price = float(r["avg_buy_price"] or 0.0)

                # ✅ Brokerage fields (backend provides)
                investment = abs_qty * entry_price
                additional_cost = _calc_additional_cost(rates, segment, investment)
                net_investment = investment + additional_cost

                live = _get_live_price(symbol)
                if live <= 0:
                    live = entry_price
                live_value = live * abs_qty

                if side == "BUY":
                    pnl_total = live_value - net_investment
                else:
                    pnl_total = (investment - live_value) - additional_cost
                # ✅ P&L per share (includes cost impact per share)
                pnl_per_share = (pnl_total / abs_qty) if abs_qty else 0.0
                abs_ratio = ((live - entry_price) / entry_price) if entry_price else 0.0
                abs_pct = abs_ratio * 100.0

                open_positions.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "qty": abs_qty,
                        "signed_qty": qty,
                        "avg_price": round(entry_price, 2),
                        "current_price": round(live, 2),
                        "pnl_per_share": round(pnl_per_share, 4),
                        "pnl_total": round(pnl_total, 2),
                        "pct": round(abs_pct, 2),
                        "segment": (segment or "delivery").lower(),
                        "investment": round(investment, 2),
                        "additional_cost": round(additional_cost, 2),
                        "net_investment": round(net_investment, 2),
                        "datetime": r["updated_at"],
                    }
                )

            return {"funds": funds, "open": open_positions, "closed": []}

    except Exception as e:
        print("⚠️ Error in /portfolio:", e)
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/{username}/upload")
async def upload_portfolio(username: str, file: UploadFile = File(...)):
    """
    Accept .xlsx upload (multi-sheet supported) and APPEND valid rows to portfolio.
    Supported headers (case/spacing insensitive):
      - symbol | script | tradingsymbol
      - qty | quantity
      - avg_price | avg price | average price | entry price | buy price
    """
    import pandas as pd
    from datetime import datetime

    def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df

    def _choose_sheet(xl: pd.ExcelFile) -> str:
        # 1) Prefer a sheet literally called "Portfolio"
        for name in xl.sheet_names:
            if str(name).strip().lower() == "portfolio":
                return name
        # 2) Otherwise pick the first sheet that has required header synonyms
        synonyms = {
            "symbol": {"symbol", "script", "tradingsymbol", "trading symbol"},
            "qty": {"qty", "quantity"},
            "avg_price": {"avg_price", "avg price", "average price", "entry price", "buy price"},
        }
        for name in xl.sheet_names:
            probe = xl.parse(name, nrows=5)
            probe = _lower_cols(probe)
            cols = set(probe.columns)
            has_symbol = any(v in cols for v in synonyms["symbol"])
            has_qty    = any(v in cols for v in synonyms["qty"])
            has_avg    = any(v in cols for v in synonyms["avg_price"])
            if has_symbol and has_qty and has_avg:
                return name
        return ""  # not found

    try:
        xl = pd.ExcelFile(file.file)
        sheet = _choose_sheet(xl)
        if not sheet:
            # show what we found in each sheet to help the user
            found = {}
            for n in xl.sheet_names:
                probe = _lower_cols(xl.parse(n, nrows=3))
                found[n] = list(probe.columns)
            raise HTTPException(
                status_code=400,
                detail=f"Excel must have columns: symbol, qty, avg_price (found headers per sheet: {found})",
            )

        # Read the chosen sheet
        df = xl.parse(sheet)
        df = _lower_cols(df)

        # Normalize headers -> symbol, qty, avg_price
        colmap = {}
        for cand in ["symbol", "script", "tradingsymbol", "trading symbol"]:
            if cand in df.columns:
                colmap["symbol"] = cand
                break
        for cand in ["qty", "quantity"]:
            if cand in df.columns:
                colmap["qty"] = cand
                break
        for cand in ["avg_price", "avg price", "average price", "entry price", "buy price"]:
            if cand in df.columns:
                colmap["avg_price"] = cand
                break

        required = {"symbol", "qty", "avg_price"}
        if not required.issubset(set(colmap.keys())):
            raise HTTPException(
                status_code=400,
                detail=f"Excel must have columns: symbol, qty, avg_price (found: {list(df.columns)})",
            )

        # Build a normalized frame
        work = pd.DataFrame({
            "symbol": df[colmap["symbol"]],
            "qty": df[colmap["qty"]],
            "avg_price": df[colmap["avg_price"]],
        })

        # Clean values
        work["symbol"] = (
            work["symbol"].astype(str).str.strip().str.upper().replace({"": None})
        )
        work["qty"] = pd.to_numeric(work["qty"], errors="coerce").fillna(0).astype(int)
        work["avg_price"] = pd.to_numeric(work["avg_price"], errors="coerce").fillna(0.0).astype(float)

        # Keep valid rows only
        work = work[(work["symbol"].notna()) & (work["symbol"] != "") &
                    (work["qty"] > 0) & (work["avg_price"] > 0.0)]

        if work.empty:
            raise HTTPException(status_code=400, detail="No valid rows to insert")

        rows_to_insert = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for _, r in work.iterrows():
            sym = str(r["symbol"]).upper()
            q   = int(r["qty"])
            ap  = float(r["avg_price"])
            rows_to_insert.append((username, sym, q, ap, ap, now))

        # Insert
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            _ensure_portfolio_schema(conn)
            c = conn.cursor()
            c.executemany(
                """
                INSERT INTO portfolio (username, script, qty, avg_buy_price, current_price, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            conn.commit()

        return {"rows": len(rows_to_insert)}

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Upload error:", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@router.post("/{username}/cancel/{symbol}")
def cancel_position(username: str, symbol: str):
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT qty, avg_buy_price FROM portfolio WHERE username=? AND script=?",
                (username, symbol),
            )
            row = c.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Position not found")

            qty, avg_price = row
            refund = float(qty) * float(avg_price)

            c.execute(
                "DELETE FROM portfolio WHERE username=? AND script=?",
                (username, symbol),
            )
            c.execute(
                "UPDATE users SET funds = funds + ? WHERE username=?",
                (refund, username),
            )
            conn.commit()
            return {"success": True, "refund": refund}
    except Exception as e:
        print("❌ Cancel error:", e)
        raise HTTPException(status_code=500, detail="Server error in cancel")
