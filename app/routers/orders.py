
# app/routers/orders.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sqlite3
from datetime import datetime, time
import requests
from pytz import timezone
from fastapi_utils.tasks import repeat_every
from datetime import datetime, time, timezone as dt_timezone
from app.routers.quotes import get_live_price
import os
from app.routers.historical import get_user_history

def _get_user_cover_history(username: str) -> List[Dict[str, Any]]:
    """
    Build SHORT (SELL FIRST -> BUY to cover) history from executed orders.

    Why this exists:
      - `historical.get_user_history()` builds LONG history only (BUY -> SELL) and
        explicitly ignores SELL_FIRST / COVER rows.
      - The History page expects rows with the SAME keys as long history:
        symbol, time, buy_qty, buy_price, buy_date, sell_qty, sell_avg_price, sell_date, invested_value, pnl, segment, remaining_qty, is_closed

    For SHORTs we map:
      - Entry = SELL_FIRST (order_type='SELL', is_short=1 or position_type='SELL_FIRST')
      - Exit  = COVER/AUTO_BUY (order_type='BUY', position_type in ('COVER','AUTO_BUY'))

    We still return in the long-history schema so the existing frontend can render it.
    (Note: for shorts, `buy_price` represents the ENTRY SELL price, and `sell_avg_price`
    represents the EXIT BUY-to-cover price.)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)
        _ensure_orders_position_type(c)

        # Pull only SHORT-related legs (avoid mixing normal SELL exits of long trades)
        c.execute(
            """
            SELECT
                UPPER(TRIM(COALESCE(script,''))) AS sym,
                CAST(qty AS INTEGER)            AS qty,
                CAST(COALESCE(price,0) AS REAL) AS price,
                datetime                        AS dt,
                LOWER(COALESCE(segment,'intraday')) AS seg,
                UPPER(COALESCE(order_type,''))  AS side,
                COALESCE(is_short,0)            AS is_short,
                UPPER(COALESCE(position_type,'')) AS ptype
            FROM orders
            WHERE username=?
              AND status IN ('Closed','SETTLED')
              AND (
                    (UPPER(COALESCE(order_type,''))='SELL' AND (COALESCE(is_short,0)=1 OR UPPER(COALESCE(position_type,''))='SELL_FIRST'))
                 OR (UPPER(COALESCE(order_type,''))='BUY'  AND UPPER(COALESCE(position_type,'')) IN ('COVER','AUTO_BUY'))
              )
            ORDER BY dt ASC, id ASC
            """,
            (username,),
        )
        rows = c.fetchall()

        # FIFO queue of open short lots keyed by (sym, seg)
        open_shorts: Dict[tuple, List[Dict[str, Any]]] = {}
        out: List[Dict[str, Any]] = []

        for sym, qty, price, dt, seg, side, is_short, ptype in rows:
            sym = (sym or "").strip().upper()
            seg = (seg or "intraday").lower()
            q = int(qty or 0)
            p = float(price or 0.0)
            side = (side or "").upper()
            ptype = (ptype or "").upper()

            if not sym or q <= 0:
                continue

            key = (sym, seg)

            if side == "SELL":
                # Entry short lot
                open_shorts.setdefault(key, []).append({"qty": q, "price": p, "dt": dt})
                continue

            # BUY leg: consume from open short lots (cover)
            remaining = q
            lots = open_shorts.get(key, [])

            while remaining > 0 and lots:
                lot = lots[0]
                take = min(remaining, int(lot["qty"]))

                entry_price = float(lot["price"])
                exit_price = p

                # P&L for short = entry SELL - exit BUY
                pnl = (entry_price - exit_price) * take

                out.append(
                    {
                        "symbol": sym,
                        "time": dt,                 # show exit time in UI
                        "buy_qty": take,            # mapped: entry qty
                        "buy_price": round(entry_price, 2),   # mapped: ENTRY SELL price
                        "buy_date": lot.get("dt"),  # entry time
                        "sell_qty": take,           # mapped: exit qty
                        "sell_avg_price": round(exit_price, 2),  # mapped: EXIT BUY price
                        "sell_date": dt,            # exit time
                        "invested_value": round(entry_price * take, 2),
                        "pnl": round(pnl, 2),
                        "segment": seg,
                        "remaining_qty": 0,
                        "is_closed": True,
                        # Extra fields (safe for frontend to ignore)
                        "short_first": True,
                        "entry_side": "SELL_FIRST",
                        "exit_side": "AUTO_BUY" if ptype == "AUTO_BUY" else "COVER",
                    }
                )

                lot["qty"] = int(lot["qty"]) - take
                remaining -= take
                if int(lot["qty"]) <= 0:
                    lots.pop(0)

        return out
    finally:
        conn.close()


router = APIRouter(prefix="/orders", tags=["orders"])

from dotenv import load_dotenv
# Load .env from current working directory (local dev)
load_dotenv()


#DB_PATH = "/data/paper_trading.db"
# =====================================================
# DB PATH — LOCAL + RENDER SAFE
# =====================================================

IS_RENDER = bool(os.getenv("RENDER"))

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if IS_RENDER:
    DB_PATH = "/data/paper_trading.db"
else:
    DB_PATH = os.path.join(BASE_DIR, "app", "data", "paper_trading.db")

# Ensure folder exists (VERY IMPORTANT)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# -------------------- Models --------------------

class Order(BaseModel):
    # Kept for compatibility (not used by place flow)
    script: str
    order_type: str   # "BUY" or "SELL"
    qty: int
    price: float
    trigger_price: Optional[float] = None
    target: Optional[float] = None
    stoploss: Optional[float] = None

class OrderUpdate(BaseModel):
    # Kept for compatibility (not used by place flow)
    price: Optional[float]
    qty: Optional[int]
    trigger_price: Optional[float]
    target: Optional[float] = None
    stoploss: Optional[float] = None

class ExitOrder(BaseModel):
    # Kept for compatibility (not used directly)
    username: str
    script: str
    qty: int
    price: Optional[float] = None
    order_type: str

class OrderData(BaseModel):
    username: str
    script: str
    order_type: str              # BUY / SELL
    qty: int
    price: Optional[float] = None   # LIMIT trigger; 0/None => MARKET
    exchange: Optional[str] = "NSE"
    segment: Optional[str] = "intraday"   # intraday / delivery
    stoploss: Optional[float] = None
    target: Optional[float] = None
    allow_short: Optional[bool] = False   # allow short selling

class ModifyOrder(BaseModel):
    script: Optional[str] = None
    qty: Optional[int] = None
    price: Optional[float] = None       # for open orders, this is the LIMIT trigger
    stoploss: Optional[float] = None
    target: Optional[float] = None


class PositionModify(BaseModel):
    username: str
    script: str

    # ✅ qty cannot be changed in modify (Add/Exit only), so keep it optional
    new_qty: Optional[int] = None

    stoploss: Optional[float] = None
    target: Optional[float] = None

    # optional (kept for assurance / UI compatibility)
    price_type: Optional[str] = "MARKET"
    limit_price: Optional[float] = None

    # ✅ IMPORTANT
    segment: Optional[str] = None           # intraday/delivery
    short_first: Optional[bool] = None      # true if short position

    # ✅ REQUIRED: anchor of the exact position lot (from positions card)
    position_datetime: str

    class Config:
        extra = "ignore"
  # ignore any extra keys sent by frontend
       # anchor datetime of that position row
    
class CloseRequest(BaseModel):
    username: str
    script: str

class BrokerageSettings(BaseModel):
    brokerage_mode: str = "ABS"  # ABS or PCT
    brokerage_intraday_pct: str = "0.0005"
    brokerage_intraday_abs: str = "20"
    brokerage_delivery_pct: str = "0.005"
    brokerage_delivery_abs: str = "0"
    tax_intraday_pct: str = "0.00018"
    tax_delivery_pct: str = "0.0011"


# Treat prices within one paisa as equal to avoid float edge cases
PRICE_EPS = 0.01

def ge(a: float, b: float) -> bool:
    """a >= b with tolerance"""
    if a is None or b is None: return False
    return float(a) >= float(b) - PRICE_EPS

def le(a: float, b: float) -> bool:
    """a <= b with tolerance"""
    if a is None or b is None: return False
    return float(a) <= float(b) + PRICE_EPS

def _validate_buy_sl_target(sl: float | None, tgt: float | None, price: float):
    """
    BUY rules:
      - stoploss must be strictly BELOW price
      - target must be strictly ABOVE price
    """
    if price <= 0:
        raise HTTPException(
            status_code=400,
            detail="Live price unavailable for SL/Target validation"
        )

    if sl is not None and ge(sl, price):
        raise HTTPException(
            status_code=400,
            detail=f"❌ Invalid Stoploss: SL ({sl}) must be lower than price ({price}) for BUY order"
        )

    if tgt is not None and le(tgt, price):
        raise HTTPException(
            status_code=400,
            detail=f"❌ Invalid Target: Target ({tgt}) must be higher than price ({price}) for BUY order"
        )


def _validate_sell_sl_target(sl: float | None, tgt: float | None, price: float):
    """
    SELL / SHORT rules (SELL FIRST context):
      - stoploss must be strictly ABOVE current price
      - target must be strictly BELOW current price
    """
    if price <= 0:
        raise HTTPException(
            status_code=400,
            detail="Live price unavailable for SL/Target validation"
        )

    if sl is not None and le(sl, price):
        raise HTTPException(
            status_code=400,
            detail=f"❌ Invalid Stoploss: SL ({sl}) must be higher than price ({price}) for SELL order"
        )

    if tgt is not None and ge(tgt, price):
        raise HTTPException(
            status_code=400,
            detail=f"❌ Invalid Target: Target ({tgt}) must be lower than price ({price}) for SELL order"
        )



def _clean_level(x):
    if x in (None, "", 0, "0", "0.0"):
        return None
    return round(float(x), 4)

def _net_open_long_qty(c, username: str, script: str, segment: str) -> int:
    """
    Net open long qty = (Closed BUY, is_short=0) - (Closed SELL, is_short=0)
    This avoids date filters and avoids mixing SELL FIRST shorts.
    """
    c.execute("""
        SELECT
          COALESCE(SUM(CASE
            WHEN status='Closed' AND position_type='BUY'  AND is_short=0 THEN qty
            ELSE 0 END), 0)
          -
          COALESCE(SUM(CASE
            WHEN status='Closed' AND position_type='SELL' AND is_short=0 THEN qty
            ELSE 0 END), 0)
        FROM orders
        WHERE username=? AND script=? AND segment=?
    """, (username, script, segment))
    val = c.fetchone()[0]
    return int(val or 0)


def _lvl_key(x) -> str:
    """Stable key for lock tables / grouping where NULLs must match."""
    if x is None:
        return "0"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "0"


def _same_trade(sl1, tgt1, sl2, tgt2):
    return (
        _clean_level(sl1) == _clean_level(sl2)
        and _clean_level(tgt1) == _clean_level(tgt2)
    )

# -------------------- Time & constants --------------------
MARKET_TZ = dt_timezone.utc

from datetime import time

def _parse_hhmm_utc(
    env_val: str | None,
    default_h: int | None = None,
    default_m: int | None = None,
    *,
    env_name: str = "TIME"
) -> time:
    """
    Parse HH:MM into datetime.time (UTC).

    - If default_h/default_m are provided (ints): fallback to defaults (old behavior).
    - If default_h/default_m are None: STRICT mode -> missing/invalid env raises RuntimeError.

    Accepts '03:45' or '3:45' etc.
    """
    def _fallback_or_raise(msg: str) -> time:
        if default_h is None or default_m is None:
            raise RuntimeError(f"{msg} Set {env_name}=HH:MM (UTC), e.g. 03:45")
        return time(default_h, default_m)

    try:
        s = (env_val or "").strip()
        if not s:
            return _fallback_or_raise(f"Missing required env var {env_name}.")

        parts = s.split(":")
        if len(parts) != 2:
            return _fallback_or_raise(f"Invalid {env_name}='{s}' (expected HH:MM).")

        h = int(parts[0])
        m = int(parts[1])
        if h < 0 or h > 23 or m < 0 or m > 59:
            return _fallback_or_raise(f"Invalid {env_name}='{s}' (out of range).")

        return time(h, m)
    except Exception:
        return _fallback_or_raise(f"Invalid {env_name}='{(env_val or '').strip()}' (parse error).")
    
@router.get("/brokerage-settings/{username}")
def get_brokerage_settings(username: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    try:
        _ensure_tables(c)

        c.execute("SELECT * FROM brokerage_settings WHERE username=?", (username,))
        row = c.fetchone()

        if not row:
            defaults = BrokerageSettings().dict()

            # ✅ Insert defaults into DB for this user (first time)
            c.execute("""
                INSERT OR IGNORE INTO brokerage_settings
                  (username, brokerage_mode,
                   brokerage_intraday_pct, brokerage_intraday_abs,
                   brokerage_delivery_pct, brokerage_delivery_abs,
                   tax_intraday_pct, tax_delivery_pct, updated_at)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
            """, (
                username,
                defaults["brokerage_mode"],
                defaults["brokerage_intraday_pct"],
                defaults["brokerage_intraday_abs"],
                defaults["brokerage_delivery_pct"],
                defaults["brokerage_delivery_abs"],
                defaults["tax_intraday_pct"],
                defaults["tax_delivery_pct"],
            ))
            conn.commit()

            c.execute("SELECT * FROM brokerage_settings WHERE username=?", (username,))
            row = c.fetchone()

        d = dict(row) if row else BrokerageSettings().dict()
        d.pop("updated_at", None)
        return d
    finally:
        conn.close()



@router.post("/brokerage-settings/{username}")
def save_brokerage_settings(username: str, body: BrokerageSettings):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)
        c.execute("""
            INSERT INTO brokerage_settings
              (username, brokerage_mode,
               brokerage_intraday_pct, brokerage_intraday_abs,
               brokerage_delivery_pct, brokerage_delivery_abs,
               tax_intraday_pct, tax_delivery_pct, updated_at)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
            ON CONFLICT(username) DO UPDATE SET
              brokerage_mode=excluded.brokerage_mode,
              brokerage_intraday_pct=excluded.brokerage_intraday_pct,
              brokerage_intraday_abs=excluded.brokerage_intraday_abs,
              brokerage_delivery_pct=excluded.brokerage_delivery_pct,
              brokerage_delivery_abs=excluded.brokerage_delivery_abs,
              tax_intraday_pct=excluded.tax_intraday_pct,
              tax_delivery_pct=excluded.tax_delivery_pct,
              updated_at=datetime('now','localtime')
        """, (
            username,
            body.brokerage_mode,
            body.brokerage_intraday_pct,
            body.brokerage_intraday_abs,
            body.brokerage_delivery_pct,
            body.brokerage_delivery_abs,
            body.tax_intraday_pct,
            body.tax_delivery_pct,
        ))
        conn.commit()
        return {"success": True}
    finally:
        conn.close()



# ✅ Market hours from ENV (UTC)
# Defaults match your current hardcoded behavior so nothing breaks.

MARKET_OPEN = _parse_hhmm_utc(os.getenv("MARKET_OPEN_TIME_UTC"), 3, 45, env_name="MARKET_OPEN_TIME_UTC")  # 09:15 IST
MARKET_CLOSE = _parse_hhmm_utc(os.getenv("MARKET_CLOSE_TIME_UTC"), 10, 0, env_name="MARKET_CLOSE_TIME_UTC")  # 15:30 IST

# Cutoffs derived from market close (existing behavior)
EOD_CUTOFF = MARKET_CLOSE
DISPLAY_CUTOFF = MARKET_CLOSE

def _execute_sell_fill(
    c: sqlite3.Cursor,
    username: str,
    script: str,
    seg: str,
    exec_price: float,
    sell_qty: int,
    sell_sl: Optional[float],
    sell_tgt: Optional[float],
) -> Dict[str, Any]:
    """
    Execute a SELL fill in a way that:
      A) Closes existing BUY positions first (FIFO across BUY groups).
         - Closing rows inherit BUY's (segment, stoploss, target) so Positions collapses/greys out.
         - SELL's entered stoploss/target is NOT shown for this closing portion.
      B) If qty remains after closing all BUY positions, remainder becomes SELL FIRST:
         - remainder uses SELL's stoploss/target and shows as a new SELL FIRST position.
      C) If there are no open BUY positions, the entire fill becomes SELL FIRST.

    NOTE:
      - This function only manages how orders rows are inserted (and optional short portfolio upsert).
      - Funds credit/deductions and portfolio qty deductions should be handled by the caller (your existing logic).
    """
    script = (script or "").upper()
    seg = (seg or "intraday").lower()
    sell_qty = int(sell_qty or 0)
    exec_price = float(exec_price or 0.0)

    if sell_qty <= 0 or exec_price <= 0:
        return {"closed_qty": 0, "short_qty": 0}

    today = _today_db(c)

    # -------- Build "open BUY position qty" grouped by (sl,tgt) FIFO ----------
    # We use BUY rows (today + segment) and subtract NON-SHORT SELL rows for same (sl,tgt).
    c.execute("""
        SELECT stoploss, target, MIN(datetime) AS first_dt, COALESCE(SUM(qty),0) AS buy_qty
        FROM orders
        WHERE username=?
          AND script=?
          AND status='Closed'
          AND order_type='BUY'
          AND UPPER(COALESCE(position_type,'BUY'))!='COVER'
          AND lower(segment)=?
          AND substr(datetime,1,10)=?
        GROUP BY stoploss, target
        ORDER BY first_dt ASC
    """, (username, script, seg, today))
    buy_groups = c.fetchall()  # [(sl, tgt, first_dt, buy_qty), ...]

    # Total SELL used to close (non-short) by group
    c.execute("""
        SELECT stoploss, target, COALESCE(SUM(qty),0) AS sell_qty
        FROM orders
        WHERE username=?
          AND script=?
          AND status='Closed'
          AND order_type='SELL'
          AND COALESCE(is_short,0)=0
          AND lower(segment)=?
          AND substr(datetime,1,10)=?
        GROUP BY stoploss, target
    """, (username, script, seg, today))
    sold_map = {}
    for sl, tgt, sq in c.fetchall():
        sold_map[( _clean_level(sl), _clean_level(tgt) )] = int(sq or 0)

    # Prepare FIFO list of open-long groups (sl,tgt, open_qty)
    fifo_groups = []
    for sl, tgt, first_dt, bq in buy_groups:
        key = (_clean_level(sl), _clean_level(tgt))
        open_qty = int(bq or 0) - int(sold_map.get(key, 0))
        if open_qty > 0:
            fifo_groups.append((key[0], key[1], open_qty))

    # -------- Allocate SELL qty to close open BUY groups first ----------
    remaining = sell_qty
    closed_total = 0

    for buy_sl, buy_tgt, open_qty in fifo_groups:
        if remaining <= 0:
            break
        use = min(open_qty, remaining)
        if use <= 0:
            continue

        # Insert SELL that CLOSES this BUY group (inherits BUY sl/tgt)
        _insert_closed(
            c,
            username=username,
            script=script,
            side="SELL",
            qty=int(use),
            price=float(exec_price),
            segment=seg,
            stoploss=buy_sl,
            target=buy_tgt,
            is_short=0,
        )

        closed_total += use
        remaining -= use

    # -------- Remainder becomes TRUE SELL FIRST ----------
    # -------- Remainder becomes TRUE SELL FIRST ----------
    short_qty = int(remaining) if remaining > 0 else 0
    if short_qty > 0:
        # 🔒 TRY TO MERGE WITH EXISTING SELL FIRST (SAME DAY)
        c.execute("""
            SELECT id, qty, price
            FROM orders
            WHERE username=?
            AND script=?
            AND order_type='SELL'
            AND position_type='SELL_FIRST'
            AND status='Closed'
            AND lower(segment)=?
            AND COALESCE(stoploss,0)=COALESCE(?,0)
            AND COALESCE(target,0)=COALESCE(?,0)
            AND substr(datetime,1,10)=?
            ORDER BY datetime ASC, id ASC
            LIMIT 1
        """, (
            username,
            script,
            seg,
            _clean_level(sell_sl),
            _clean_level(sell_tgt),
            today,
        ))

        row = c.fetchone()

        if row:
            old_id, old_qty, old_price = row
            new_qty = old_qty + short_qty
            new_avg = (
                (old_qty * old_price) +
                (short_qty * exec_price)
            ) / new_qty

            # ✅ LOG THIS SELL FIRST EXECUTION (because UPDATE won't trigger activity)
            _log_activity(
                c,
                username=username,
                script=script,
                action="SELL",
                activity_type="SELL_FIRST",
                qty=short_qty,                 # log only THIS new executed qty
                price=exec_price,
                exchange="NSE",
                segment=seg,
                notes=f"merged into existing SELL_FIRST order id={old_id}"
            )
            
            c.execute("""
                UPDATE orders
                SET qty=?,
                    price=?,
                    datetime=datetime('now','localtime')
                WHERE id=?
            """, (
                new_qty,
                round(new_avg, 2),
                old_id
            ))
        else:
            _insert_closed(
                c,
                username=username,
                script=script,
                side="SELL",
                qty=short_qty,
                price=float(exec_price),
                segment=seg,
                stoploss=_clean_level(sell_sl),
                target=_clean_level(sell_tgt),
                is_short=1,
            )

        # portfolio tracking (same as before)
        #_upsert_portfolio(
          #  c,
           # username=username,
           # script=script,
            #add_qty=-int(short_qty),
            #add_avg_price=float(exec_price),
        #)
        # ✅ IMPORTANT: for DELIVERY SELL FIRST, write to portfolio immediately
        # so short positions show in Portfolio even if EOD pipeline skips.
        if seg == "delivery" and short_qty > 0 and is_after_market_close():
            print ("Is Market Close flag = {}".format(is_after_market_close()))
            _upsert_portfolio(
                c,
                username=username,
                script=script,
                add_qty=-int(short_qty),         # negative qty => SELL FIRST holding
                add_avg_price=float(exec_price), # avg sell price
            )

    return {
        "closed_qty": int(closed_total),
        "short_qty": int(short_qty),
    }



def _now_utc():
    """Return current datetime in US/Eastern timezone"""
    return datetime.now(dt_timezone.utc)


def _today_db(c: sqlite3.Cursor) -> str:
    """
    Returns today's date using the SAME clock used by your orders table timestamps.
    Because you write datetime as datetime('now','localtime'), we must read date from SQLite localtime.
    Works on local (IST) and Render (UTC).
    """
    c.execute("SELECT date('now','localtime')")
    return str(c.fetchone()[0])



def is_market_open() -> bool:
    """Check if current time is within US market hours"""
    now = _now_utc().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE

def is_after_market_close() -> bool:
    """
    Returns True if market is considered closed.
    Supports FORCE_MARKET_OPEN=true for testing.
    """

    # Market timezone aware current time
    now = datetime.now(MARKET_TZ).time()

    # After official end-of-day cutoff
    return now >= EOD_CUTOFF

# -------------------- DB helpers --------------------

def _ensure_tables(c: sqlite3.Cursor):
    c.execute("""
      CREATE TABLE IF NOT EXISTS funds (
        username TEXT PRIMARY KEY,
        available_amount REAL NOT NULL DEFAULT 0,
        total_amount REAL NOT NULL DEFAULT 0
      )
    """)


    c.execute("""
      CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        script TEXT NOT NULL,
        order_type TEXT NOT NULL,   -- BUY/SELL
        position_type TEXT,          -- ✅ ADD THIS
        qty INTEGER NOT NULL,
        price REAL NOT NULL,        -- LIMIT trigger (Open) or executed price (Closed)
        exchange TEXT,
        segment TEXT,               -- intraday/delivery
        status TEXT NOT NULL,       -- Open/Closed/Cancelled
        datetime TEXT NOT NULL,     -- localtime ISO
        pnl REAL,
        stoploss REAL,
        target REAL,
        is_short INTEGER DEFAULT 0  -- marks "SELL FIRST" rows
      )
    """)

    c.execute("""
      CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        script TEXT NOT NULL,
        qty INTEGER NOT NULL,
        avg_buy_price REAL NOT NULL,
        current_price REAL,
        datetime TEXT NOT NULL,
        updated_at TEXT,
        UNIQUE(username, script)
      )
    """)

    c.execute("""
      CREATE TABLE IF NOT EXISTS portfolio_exits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        script TEXT,
        qty INTEGER,
        price REAL,
        datetime TEXT,
        segment TEXT,               -- intraday/delivery
        exit_side TEXT              -- 'SELL' (long exit) or 'BUY' (short cover)
      )
    """)

    # --- NEW: carry-over store for DELIVERY "SELL FIRST" remainders ---
    # We keep this separate from 'portfolio' so it won't conflict with the existing UNIQUE(username, script)
    # and so you can represent carried short positions distinctly.
    c.execute("""
      CREATE TABLE IF NOT EXISTS portfolio_short (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT NOT NULL,
        script    TEXT NOT NULL,
        qty       INTEGER NOT NULL,
        avg_price REAL NOT NULL,
        datetime  TEXT NOT NULL,
        updated_at TEXT,
        UNIQUE(username, script)
      )
    """)


    # --- NEW: EOD run guard (prevents repeat EOD every scheduler tick) ---
    c.execute('''
      CREATE TABLE IF NOT EXISTS eod_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        day TEXT NOT NULL,                 -- localtime day (YYYY-MM-DD)
        datetime TEXT NOT NULL,            -- localtime
        UNIQUE(username, day)
      )
    ''')

    # --- NEW: SL/Target auto-exit lock (prevents duplicate auto exits) ---
    c.execute('''
      CREATE TABLE IF NOT EXISTS auto_exit_locks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        script   TEXT NOT NULL,
        segment  TEXT NOT NULL,
        day      TEXT NOT NULL,
        direction TEXT NOT NULL,           -- 'LONG'
        sl_key   TEXT NOT NULL,
        tgt_key  TEXT NOT NULL,
        datetime TEXT NOT NULL,
        UNIQUE(username, script, segment, day, direction, sl_key, tgt_key)
      )
    ''')

     # ✅ Activity Journal Table (Every action will be stored here forever)
    c.execute("""
    CREATE TABLE IF NOT EXISTS trade_activity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        datetime TEXT NOT NULL,          -- ISO time
        script TEXT NOT NULL,
        action TEXT NOT NULL,            -- BUY / SELL
        activity_type TEXT NOT NULL,     -- BUY / ADD / EXIT / SELL_FIRST / MODIFY / EOD / etc
        qty INTEGER NOT NULL,
        price REAL NOT NULL,
        exchange TEXT DEFAULT 'NSE',
        segment TEXT DEFAULT 'intraday', -- intraday/delivery
        notes TEXT DEFAULT ''
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS brokerage_settings (
        username TEXT PRIMARY KEY,
        brokerage_mode TEXT NOT NULL DEFAULT 'ABS',  -- ABS or PCT

        brokerage_intraday_pct TEXT NOT NULL DEFAULT '0.0005',
        brokerage_intraday_abs TEXT NOT NULL DEFAULT '20',

        brokerage_delivery_pct TEXT NOT NULL DEFAULT '0.005',
        brokerage_delivery_abs TEXT NOT NULL DEFAULT '0',

        tax_intraday_pct TEXT NOT NULL DEFAULT '0.00018',
        tax_delivery_pct TEXT NOT NULL DEFAULT '0.0011',

        updated_at TEXT
    )
    """)


    # ✅ Trigger: whenever a CLOSED row is inserted into orders -> add activity automatically
    c.execute("""
    CREATE TRIGGER IF NOT EXISTS trg_orders_closed_to_activity
    AFTER INSERT ON orders
    WHEN NEW.status = 'Closed'
    BEGIN
        INSERT INTO trade_activity
        (username, datetime, script, action, activity_type, qty, price, exchange, segment, notes)
        VALUES
        (
            NEW.username,
            NEW.datetime,
            NEW.script,
            NEW.order_type,
            COALESCE(NEW.position_type, NEW.order_type),
            NEW.qty,
            NEW.price,
            COALESCE(NEW.exchange, 'NSE'),
            COALESCE(NEW.segment, 'intraday'),
            ''
        );
    END;
    """)

    c.execute("""
    CREATE TRIGGER IF NOT EXISTS trg_orders_update_closed_to_activity
    AFTER UPDATE ON orders
    WHEN NEW.status='Closed' AND (OLD.status IS NULL OR OLD.status!='Closed')
    BEGIN
        INSERT INTO trade_activity
        (username, datetime, script, action, activity_type, qty, price, exchange, segment, notes)
        VALUES
        (
            NEW.username,
            NEW.datetime,
            NEW.script,
            NEW.order_type,
            COALESCE(NEW.position_type, NEW.order_type),
            NEW.qty,
            NEW.price,
            COALESCE(NEW.exchange, 'NSE'),
            COALESCE(NEW.segment, 'intraday'),
            'update-close'
        );
    END;
    """)

    # --- lightweight migrations for existing DBs ---
    
    # orders: add is_short if missing
    try:
        c.execute("PRAGMA table_info(orders)")
        ocols = [r[1].lower() for r in c.fetchall()]
        if "is_short" not in ocols:
            c.execute("ALTER TABLE orders ADD COLUMN is_short INTEGER DEFAULT 0")
    except Exception:
        pass

    # portfolio_exits: add segment / exit_side if missing
    try:
        c.execute("PRAGMA table_info(portfolio_exits)")
        pcols = [r[1].lower() for r in c.fetchall()]
        if "segment" not in pcols:
            c.execute("ALTER TABLE portfolio_exits ADD COLUMN segment TEXT")
        if "exit_side" not in pcols:
            c.execute("ALTER TABLE portfolio_exits ADD COLUMN exit_side TEXT")
    except Exception:
        pass

    # ✅ orders: add exit_price if missing
    try:
        c.execute("PRAGMA table_info(orders)")
        cols = [r[1].lower() for r in c.fetchall()]
        if "exit_price" not in cols:
            c.execute("ALTER TABLE orders ADD COLUMN exit_price REAL")
    except Exception:
        pass

def ensure_orders_schema():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        # 🛠 ensure columns exist
        cols = [row[1] for row in c.execute("PRAGMA table_info(orders);")]
        if "updated_at" not in cols:
            c.execute("ALTER TABLE orders ADD COLUMN updated_at TEXT;")
            conn.commit()

def _ensure_orders_position_type(c: sqlite3.Cursor):
    c.execute("PRAGMA table_info(orders)")
    cols = {row[1].lower() for row in c.fetchall()}
    if "position_type" not in cols:
        c.execute("ALTER TABLE orders ADD COLUMN position_type TEXT")

def _ensure_orders_updated_at(c: sqlite3.Cursor):
    c.execute("PRAGMA table_info(orders)")
    cols = {row[1].lower() for row in c.fetchall()}
    if "updated_at" not in cols:
        c.execute("ALTER TABLE orders ADD COLUMN updated_at TEXT")
def _ensure_eod_intraday_lock(c: sqlite3.Cursor):
    """
    Idempotency guard for intraday EOD square-off.
    Ensures we auto-square-off a (username, day, script, segment) only once.
    """
    c.execute("""
        CREATE TABLE IF NOT EXISTS eod_intraday_lock (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            day TEXT NOT NULL,        -- date('now','localtime') i.e. YYYY-MM-DD
            script TEXT NOT NULL,     -- normalized UPPER(TRIM(script))
            segment TEXT NOT NULL,    -- 'intraday'
            datetime TEXT NOT NULL,   -- localtime timestamp when lock was taken
            UNIQUE(username, day, script, segment)
        )
    """)

def _ensure_eod_delivery_lock(c: sqlite3.Cursor):
    """
    Idempotency guard for DELIVERY settlement.
    Prevents double-upsert into portfolio when scheduler runs again.
    """
    c.execute("""
        CREATE TABLE IF NOT EXISTS eod_delivery_lock (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            day TEXT NOT NULL,
            script TEXT NOT NULL,     -- normalized UPPER(TRIM(script))
            datetime TEXT NOT NULL,
            UNIQUE(username, day, script)
        )
    """)



def _ensure_position_settings_table(c):
    c.execute("""
        CREATE TABLE IF NOT EXISTS position_settings (
            username TEXT NOT NULL,
            script TEXT NOT NULL,
            segment TEXT NOT NULL,              -- intraday/delivery
            bucket TEXT NOT NULL,               -- LONG/SHORT
            anchor_datetime TEXT NOT NULL,      -- identifies the position row (first_buy_dt for long)
            stoploss REAL,
            target REAL,
            updated_at TEXT DEFAULT (datetime('now','localtime')),
            PRIMARY KEY (username, script, segment, bucket, anchor_datetime)
        )
    """)

# --- Short-first delivery carry table ----------------------------------------

def _ensure_portfolio_short_table(c: sqlite3.Cursor):
    """
    Holds DELIVERY 'SELL FIRST' carry-overs so we don't have to change the
    existing portfolio uniqueness. Safe to call many times.
    """
    c.execute("""
      CREATE TABLE IF NOT EXISTS portfolio_short (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        script   TEXT NOT NULL,
        qty      INTEGER NOT NULL,
        avg_price REAL NOT NULL,
        datetime  TEXT NOT NULL,
        updated_at TEXT,
        UNIQUE(username, script)
      )
    """)
        # orders: add exit_price if missing

def _distinct_usernames(c: sqlite3.Cursor) -> list[str]:
    # pick usernames from any table you trust exists
    c.execute("SELECT DISTINCT username FROM funds")
    users = [r[0] for r in c.fetchall() if r and r[0]]
    if users:
        return users

    # fallback (if funds empty): find from orders
    c.execute("SELECT DISTINCT username FROM orders")
    return [r[0] for r in c.fetchall() if r and r[0]]

@router.get("/activity/{username}")
def get_activity(username: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    try:
        _ensure_tables(c)
        c.execute("""
            SELECT id, datetime, script, action, activity_type, qty, price, exchange, segment, notes
            FROM trade_activity
            WHERE username=?
            ORDER BY datetime DESC, id DESC
        """, (username,))
        return [dict(r) for r in c.fetchall()]
    finally:
        conn.close()

@router.on_event("startup")
@repeat_every(seconds=10)
def auto_process_orders() -> None:
    # normal processing
    process_open_orders()

    if not is_after_market_close():
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)
        users = _distinct_usernames(c)
        for u in users:
            run_eod_pipeline(u)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("⚠️ auto EOD error:", e)
    finally:
        conn.close()


def _ensure_funds_row(c: sqlite3.Cursor, username: str) -> float:
    """
    ✅ IMPORTANT:
    Do NOT trust `funds.available_amount` as the source of truth anymore.

    We recompute available funds from:
      - funds.total_amount
      - executed orders (Closed/SETTLED)
      - additional_cost per trade leg
      - SELL_FIRST block rules (EQ = 1x, FNO = 3x)
    """
    # ensure funds row exists
    c.execute("SELECT 1 FROM funds WHERE username=?", (username,))
    if c.fetchone() is None:
        c.execute(
            "INSERT INTO funds (username, total_amount, available_amount) VALUES (?, 0.0, 0.0)",
            (username,),
        )

    try:
        from app.routers.funds import compute_funds_snapshot
        snap = compute_funds_snapshot(username=username, conn=None, sync_db=True)
        return float(snap.get("available_funds", 0.0) or 0.0)
    except Exception:
        # fallback: old behavior
        c.execute("SELECT available_amount FROM funds WHERE username = ?", (username,))
        row = c.fetchone()
        if not row:
            return 0.0
        return float(row[0] or 0.0)


# -------------------- Price helpers --------------------
# -------------------- Common insert helpers --------------------

def _insert_closed(
    c: sqlite3.Cursor,
    username: str,
    script: str,
    side: str,
    qty: int,
    price: float,
    segment: str,
    stoploss: Optional[float] = None,
    target: Optional[float] = None,
    is_short: int = 0,
    position_type_override: Optional[str] = None,   # ✅ NEW
    exchange: str = "NSE",
):
    side_u = side.upper()

    if position_type_override:
        position_type = position_type_override
    else:
        if side_u == "SELL":
            if is_short:
                position_type = "SELL_FIRST"
            else:
                position_type = "EXIT"
        else:
            position_type = "BUY"

    c.execute(
        """
        INSERT INTO orders
          (username, script, order_type, position_type,
           qty, price, exchange, segment,
           status, datetime, pnl, stoploss, target, is_short)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, 'Closed',
           datetime('now','localtime'), 0.0, ?, ?, ?)
        """,
        (
            username,
            script,
            side_u,
            position_type,
            qty,
            float(price),                       # ✅ price first
            (exchange or "NSE").upper(),        # ✅ then exchange
            (segment or "intraday").lower(),    # ✅ then segment
            stoploss,
            target,
            int(is_short),
        ),
    )

def _log_activity(
    c: sqlite3.Cursor,
    username: str,
    script: str,
    action: str,
    activity_type: str,
    qty: int,
    price: float,
    exchange: str = "NSE",
    segment: str = "intraday",
    notes: str = "",
):
    c.execute(
        """
        INSERT INTO trade_activity
          (username, datetime, script, action, activity_type, qty, price, exchange, segment, notes)
        VALUES
          (?, datetime('now','localtime'), ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            script,
            action,
            activity_type,
            int(qty),
            float(price),
            exchange,
            (segment or "intraday").lower(),
            notes or "",
        ),
    )


def _sum_closed(c: sqlite3.Cursor, username: str, script: str, side: str) -> int:
    c.execute(
        """
        SELECT COALESCE(SUM(qty),0) FROM orders
         WHERE username=? AND script=? AND order_type=? AND status='Closed'
        """,
        (username, script, side.upper()),
    )
    return int(c.fetchone()[0] or 0)

def _sum_closed_today_intraday(c: sqlite3.Cursor, username: str, script: str, side: str) -> int:
    #today = _now_utc().strftime("%Y-%m-%d")
    today = _today_db(c)
    c.execute(
        """
        SELECT COALESCE(SUM(qty),0) FROM orders
         WHERE username=? AND script=? AND order_type=? AND status='Closed'
           AND lower(segment)='intraday' AND substr(datetime,1,10)=?
        """,
        (username, script, side.upper(), today),
    )
    return int(c.fetchone()[0] or 0)

# -------------------- Utilities for portfolio --------------------

def _weighted_avg(qtys_prices):
    tot_q = sum(q for q, _ in qtys_prices)
    if tot_q <= 0:
        return 0.0
    return sum(q * p for q, p in qtys_prices) / tot_q

def _upsert_portfolio(c: sqlite3.Cursor, username: str, script: str, add_qty: int, add_avg_price: float):
    """
    Merge into portfolio with weighted average.
    Supports BOTH long (+qty) and short (-qty).

    Rules:
      - If position increases in same direction (both + or both -): weighted avg updates.
      - If position reduces (opposite signs): avg stays as-is (it's a partial exit/cover).
      - If qty becomes 0: delete row.
      - If position flips sign (e.g. short -> long): new avg becomes the incoming trade price.
    """
    now_iso = datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S")


    c.execute(
        "SELECT qty, avg_buy_price FROM portfolio WHERE username=? AND script=?",
        (username, script),
    )
    row = c.fetchone()

    if not row:
        # brand new position
        if add_qty == 0:
            return
        c.execute(
            """INSERT INTO portfolio
               (username, script, qty, avg_buy_price, current_price, datetime, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (username, script, int(add_qty), float(add_avg_price), float(add_avg_price), now_iso, now_iso),
        )
        return

    cur_qty = int(row[0] or 0)
    cur_avg = float(row[1] or 0.0)

    new_qty = cur_qty + int(add_qty)

    # If fully closed
    if new_qty == 0:
        c.execute("DELETE FROM portfolio WHERE username=? AND script=?", (username, script))
        return

    # Same direction increase (long add / short add)
    if (cur_qty > 0 and add_qty > 0) or (cur_qty < 0 and add_qty < 0):
        denom = abs(new_qty)
        new_avg = ((abs(cur_qty) * cur_avg) + (abs(add_qty) * float(add_avg_price))) / denom

    # Position reduced (cover / partial sell): keep avg as-is
    elif (cur_qty > 0 and add_qty < 0) or (cur_qty < 0 and add_qty > 0):
        # if it flips direction, reset avg to incoming trade price
        if (cur_qty < 0 and new_qty > 0) or (cur_qty > 0 and new_qty < 0):
            new_avg = float(add_avg_price)
        else:
            new_avg = cur_avg

    else:
        new_avg = cur_avg

    c.execute(
        """UPDATE portfolio
              SET qty=?, avg_buy_price=?, current_price=?, updated_at=?
            WHERE username=? AND script=?""",
        (int(new_qty), float(new_avg), float(new_avg), now_iso, username, script),
    )

# -------------------- EOD helpers & pipeline --------------------

def _cancel_open_limit_and_refund(c: sqlite3.Cursor, username: str, segment: Optional[str] = None):
    if segment:
        c.execute(
            """
            SELECT id, order_type, qty, price FROM orders
             WHERE username=? AND status='Open' AND lower(segment)=?
            """,
            (username, segment.lower()),
        )
    else:
        c.execute(
            """
            SELECT id, order_type, qty, price FROM orders
             WHERE username=? AND status='Open'
            """,
            (username,),
        )
    rows = c.fetchall()
    if not rows:
        return
    refund = 0.0
    for _oid, side, qty, trig in rows:
        if str(side).upper() == "BUY":
            refund += float(trig) * int(qty)
    if refund > 0:
        c.execute(
            "UPDATE funds SET available_amount = available_amount + ? WHERE username = ?",
            (refund, username),
        )
    if segment:
        c.execute(
            "UPDATE orders SET status='Cancelled' WHERE username=? AND status='Open' AND lower(segment)=?",
            (username, segment.lower()),
        )
    else:
        c.execute("UPDATE orders SET status='Cancelled' WHERE username=? AND status='Open'", (username,))

def run_eod_pipeline(username: str):
    """
    At/after EOD:
      INTRADAY:
        - Net long (BUY > SELL)  -> auto SELL, funds credit, history (exit_side='SELL')
        - Net short (SELL > BUY) -> auto BUY cover, funds debit, history (exit_side='BUY')

      DELIVERY:
        - Normal SELL (is_short=0) -> history
        - BUY remainder            -> portfolio (long carry)
        - SELL FIRST remainder     -> auto BUY at LIVE and add to portfolio (no history)

    Also cancels all still-open limit orders and refunds BUY blocks.
    Idempotent.
    """

    print("✅ EOD PIPELINE CALLED", datetime.now())

    # Only run after market close
    if not is_after_market_close():
        return {"status": "skipped", "message": "Market not closed yet"}

    conn = sqlite3.connect(DB_PATH, timeout=30)
    # ✅ Prevent concurrent EOD runs from double-settling (portfolio doubling etc.)
    conn.execute("BEGIN IMMEDIATE")
    c = conn.cursor()
    try:
        _ensure_tables(c)
        _ensure_funds_row(c, username)
        _ensure_orders_position_type(c)
        _ensure_eod_delivery_lock(c)
        
        today = _today_db(c)

        # ✅ IMPORTANT:
        # Previously, you inserted into eod_runs at the start and then refused to run again for the whole day.
        # That breaks the expected behavior when new trades are placed AFTER market close.
        # So we run whenever there are still CLOSED (unsettled) rows for today.

        c.execute(
            """
            SELECT 1
              FROM orders
             WHERE username=?
               AND status='Closed'
               AND substr(datetime,1,10)=?
             LIMIT 1
            """,
            (username, today),
        )
        if not c.fetchone():
            conn.commit()
            return {"status": "skipped", "message": "No pending CLOSED trades to settle", "day": today}

        print("✅ EOD PIPELINE RUNNING", datetime.now(), "user=", username, "day=", today)
        # 0) Cancel still-open limits (both segments) and refund BUY blocks
        _cancel_open_limit_and_refund(c, username, segment="intraday")
        _cancel_open_limit_and_refund(c, username, segment="delivery")

        # 1) INTRADAY: square-off both ways to history
        # 1) INTRADAY: square-off both LONG and SHORT correctly (no mixing)
        c.execute("""
            SELECT DISTINCT UPPER(TRIM(COALESCE(script,''))) AS sym
            FROM orders
            WHERE username=? AND lower(segment)='intraday'
            AND status='Closed' AND substr(datetime,1,10)=?
            AND TRIM(COALESCE(script,''))!=''
        """, (username, today))
        intraday_scripts = [r[0] for r in c.fetchall()]

        for script in intraday_scripts:
            # ---- LONG net (ignore COVER buys, ignore SELL_FIRST sells) ----
            c.execute("""
                SELECT
                COALESCE(SUM(CASE
                    WHEN order_type='BUY'
                    AND status='Closed'
                    AND lower(segment)='intraday'
                    AND substr(datetime,1,10)=?
                    AND UPPER(COALESCE(position_type,'BUY'))!='COVER'
                    THEN qty ELSE 0 END),0)
                -
                COALESCE(SUM(CASE
                    WHEN order_type='SELL'
                    AND status='Closed'
                    AND lower(segment)='intraday'
                    AND substr(datetime,1,10)=?
                    AND COALESCE(is_short,0)=0
                    THEN qty ELSE 0 END),0)
                FROM orders
                WHERE username=? AND UPPER(TRIM(COALESCE(script,'')))=?
            """, (today, today, username, script))
            long_net = int(c.fetchone()[0] or 0)

            # ---- SHORT net (SELL_FIRST - COVER/AUTO_BUY) ----
            c.execute("""
                SELECT
                COALESCE(SUM(CASE
                    WHEN order_type='SELL'
                    AND status='Closed'
                    AND lower(segment)='intraday'
                    AND substr(datetime,1,10)=?
                    AND (COALESCE(is_short,0)=1 OR UPPER(COALESCE(position_type,''))='SELL_FIRST')
                    THEN qty ELSE 0 END),0)
                -
                COALESCE(SUM(CASE
                    WHEN order_type='BUY'
                    AND status='Closed'
                    AND lower(segment)='intraday'
                    AND substr(datetime,1,10)=?
                    AND UPPER(COALESCE(position_type,'')) IN ('COVER','AUTO_BUY')
                    THEN qty ELSE 0 END),0)
                FROM orders
                WHERE username=? AND UPPER(TRIM(COALESCE(script,'')))=?
            """, (today, today, username, script))
            short_net = int(c.fetchone()[0] or 0)

            if long_net == 0 and short_net == 0:
                continue

            live = get_live_price(script)
            if live <= 0:
                # Fallback #1: last order price
                try:
                    c.execute("""
                        SELECT price FROM orders
                        WHERE username=? AND UPPER(TRIM(COALESCE(script,'')))=? AND COALESCE(price,0)>0
                        ORDER BY datetime DESC, id DESC
                        LIMIT 1
                    """, (username, script))
                    rp = c.fetchone()
                    if rp and rp[0]:
                        live = float(rp[0])
                except Exception:
                    pass

            if live <= 0:
                # Fallback #2: portfolio avg_buy_price (FIXED COLUMN)
                try:
                    c.execute("""
                        SELECT avg_buy_price FROM portfolio
                        WHERE username=? AND UPPER(TRIM(COALESCE(script,'')))=?
                        LIMIT 1
                    """, (username, script))
                    rp = c.fetchone()
                    if rp and rp[0]:
                        live = float(rp[0])
                except Exception:
                    pass

            if live <= 0:
                print(f"⚠️ EOD: live price unavailable for {script}; skipping intraday square-off.")
                continue

            if live <= 0:
                # Fallback #3: last non-zero executed price from ANY order row for this script
                try:
                    c.execute("""
                        SELECT price
                        FROM orders
                        WHERE UPPER(TRIM(COALESCE(script,'')))=?
                        AND COALESCE(price,0) > 0
                        ORDER BY datetime DESC, id DESC
                        LIMIT 1
                    """, (script,))
                    rp = c.fetchone()
                    if rp and rp[0]:
                        live = float(rp[0])
                except Exception:
                    pass



            # LONG square-off → AUTO_SELL
            if long_net > 0:
                qty = int(long_net)
                c.execute("UPDATE funds SET available_amount = available_amount + ? WHERE username=?",
                        (live * qty, username))
                _insert_closed(
                    c, username, script, "SELL", qty, float(live), "intraday",
                    is_short=0, position_type_override="AUTO_SELL",
                )
                c.execute("""
                    INSERT INTO portfolio_exits (username, script, qty, price, datetime, segment, exit_side)
                    VALUES (?, ?, ?, ?, datetime('now','localtime'), 'intraday', 'AUTO_SELL')
                """, (username, script, qty, float(live)))

            # SHORT square-off → AUTO_BUY (cover)
            if short_net > 0:
                qty = int(short_net)
                c.execute("UPDATE funds SET available_amount = available_amount - ? WHERE username=?",
                        (live * qty, username))
                _insert_closed(
                    c, username, script, "BUY", qty, float(live), "intraday",
                    is_short=1, position_type_override="AUTO_BUY",
                )
                c.execute("""
                    INSERT INTO portfolio_exits (username, script, qty, price, datetime, segment, exit_side)
                    VALUES (?, ?, ?, ?, datetime('now','localtime'), 'intraday', 'AUTO_BUY')
                """, (username, script, qty, float(live)))

        # 2) DELIVERY: normal sells -> history, long remainders -> portfolio,
        #              SELL FIRST remainder -> auto BUY at LIVE and add to portfolio.
        c.execute("""
            SELECT DISTINCT script
              FROM orders
             WHERE username=? AND lower(segment)='delivery'
               AND status='Closed' AND substr(datetime,1,10)=?
        """, (username, today))
        delivery_scripts = [r[0] for r in c.fetchall()]

        for script in delivery_scripts:

            sym = (script or "").strip().upper()

            # ✅ idempotency: settle each delivery script only once per day
            c.execute("""
                INSERT OR IGNORE INTO eod_delivery_lock (username, day, script, datetime)
                VALUES (?, ?, ?, datetime('now','localtime'))
            """, (username, today, sym))

            if c.rowcount == 0:
                # already settled today, skip to avoid portfolio qty doubling
                continue


            # today's BUY legs (delivery)
            c.execute("""
                SELECT qty, price
                  FROM orders
                 WHERE username=? AND script=? AND lower(segment)='delivery'
                   AND status='Closed' AND order_type='BUY' AND substr(datetime,1,10)=?
                       AND UPPER(COALESCE(position_type,'BUY'))!='COVER'
                 ORDER BY datetime ASC, id ASC
            """, (username, script, today))
            buys = [(int(q), float(p)) for q, p in c.fetchall()]
            total_buy_qty  = sum(q for q, _ in buys)
            total_buy_notl = sum(q * p for q, p in buys)

            # today's normal SELL legs (is_short=0)
            # today's normal SELL legs (NOT sell-first)
            c.execute("""
                SELECT qty, price
                FROM orders
                WHERE username=? AND script=? AND lower(segment)='delivery'
                AND status='Closed' AND order_type='SELL'
                AND COALESCE(is_short,0)=0
                AND UPPER(COALESCE(position_type,''))!='SELL_FIRST'
                AND substr(datetime,1,10)=?
                ORDER BY datetime ASC, id ASC
            """, (username, script, today))
            sells_normal = [(int(q), float(p)) for q, p in c.fetchall()]
            sell_normal_qty = sum(q for q, _ in sells_normal)

            # today's SELL FIRST legs (robust: is_short OR position_type)
            c.execute("""
                SELECT qty, price
                FROM orders
                WHERE username=? AND script=? AND lower(segment)='delivery'
                AND status='Closed' AND order_type='SELL'
                AND (
                        COALESCE(is_short,0)=1
                        OR UPPER(COALESCE(position_type,''))='SELL_FIRST'
                )
                AND substr(datetime,1,10)=?
                ORDER BY datetime ASC, id ASC
            """, (username, script, today))
            sells_shortfirst = [(int(q), float(p)) for q, p in c.fetchall()]
            sell_sf_qty = sum(q for q, _ in sells_shortfirst)

            # 2a) normal sells -> history (portfolio_exits)
            for q, p in sells_normal:
                c.execute("""
                    INSERT INTO portfolio_exits (username, script, qty, price, datetime, segment, exit_side)
                    VALUES (?, ?, ?, ?, datetime('now','localtime'), 'delivery', 'SELL')
                """, (username, script, q, p))

            # mark normal sells as SETTLED so they won't be re-processed
            c.execute("""
                UPDATE orders
                SET status='SETTLED'
                WHERE username=? AND script=?
                AND lower(segment)='delivery'
                AND order_type='SELL'
                AND COALESCE(is_short,0)=0
                AND UPPER(COALESCE(position_type,''))!='SELL_FIRST'
                AND status='Closed'
                AND substr(datetime,1,10)=?
            """, (username, script, today))

            # 2b) DELIVERY BUY → PORTFOLIO (FINAL SETTLEMENT)  (IDEMPOTENT)
            remaining_long = total_buy_qty - sell_normal_qty
            if remaining_long > 0:
                avg_buy = (total_buy_notl / total_buy_qty) if total_buy_qty else 0.0

                # reconcile portfolio qty to exactly remaining_long (prevents doubling on re-run)
                c.execute(
                    "SELECT COALESCE(qty,0) FROM portfolio WHERE username=? AND script=?",
                    (username, script),
                )
                cur_port_qty = int((c.fetchone() or [0])[0] or 0)

                # only reconcile if current portfolio is long/empty (don't overwrite an existing short)
                if cur_port_qty >= 0:
                    diff = int(remaining_long) - cur_port_qty
                    if diff != 0:
                        _upsert_portfolio(
                            c,
                            username=username,
                            script=script,
                            add_qty=int(diff),          # add only the difference
                            add_avg_price=float(avg_buy),
                        )

            # mark today's delivery BUY legs as SETTLED
            c.execute("""
                UPDATE orders
                SET status='SETTLED'
                WHERE username=? AND script=?
                AND lower(segment)='delivery'
                AND order_type='BUY'
                AND status='Closed'
                AND substr(datetime,1,10)=?
            """, (username, script, today))

            # compute net_today safely (prevents NameError and drives short carry)
            net_today = int(total_buy_qty) - int(sell_normal_qty) - int(sell_sf_qty)  # >0 long; <0 short

            # 2c) DELIVERY SELL FIRST → carry forward as SHORT position (negative qty)
            if net_today < 0 and sell_sf_qty > 0:
                target_short = int(net_today)  # negative
                avg_sell = (
                    sum(q * p for q, p in sells_shortfirst) / sell_sf_qty
                    if sell_sf_qty else 0.0
                )

                c.execute(
                    "SELECT COALESCE(qty,0) FROM portfolio WHERE username=? AND script=?",
                    (username, script),
                )
                cur_port_qty = int((c.fetchone() or [0])[0] or 0)

                # reconcile to exactly target_short
                diff = target_short - cur_port_qty
                if diff != 0:
                    _upsert_portfolio(
                        c,
                        username=username,
                        script=script,
                        add_qty=int(diff),
                        add_avg_price=float(avg_sell),
                    )
            # ✅ DO NOT DELETE sell-first rows (portfolio reconcile needs them).
            # Mark them SETTLED so they won't appear in open Positions, but still remain for portfolio reconciliation.
            c.execute("""
                UPDATE orders
                SET status='SETTLED'
                WHERE username=? AND script=? AND lower(segment)='delivery'
                AND status='Closed' AND order_type='SELL'
                AND (
                        COALESCE(is_short,0)=1
                        OR UPPER(COALESCE(position_type,''))='SELL_FIRST'
                )
                AND substr(datetime,1,10)=?
            """, (username, script, today))


        # ✅ After EOD: hide today's CLOSED rows from Positions/Open positions by marking SETTLED
        # ✅ After EOD: hide today's CLOSED rows from Positions/Open positions by marking SETTLED
        c.execute(
            """
            UPDATE orders
            SET status='SETTLED'
            WHERE username=?
            AND status='Closed'
            AND substr(datetime,1,10)=?
            AND (
                    lower(COALESCE(segment,'')) != 'intraday'
                    OR script NOT IN (
                        SELECT script
                        FROM orders
                        WHERE username=?
                        AND status='Closed'
                        AND lower(COALESCE(segment,''))='intraday'
                        AND substr(datetime,1,10)=?
                        GROUP BY script
                        HAVING COALESCE(SUM(CASE WHEN order_type='BUY' THEN qty ELSE 0 END),0) !=
                            COALESCE(SUM(CASE WHEN order_type='SELL' THEN qty ELSE 0 END),0)
                    )
            )
            """,
            (username, today, username, today),
        )



        ## Remove duplicates from Portfolio TABLE...
        c.execute("""
            DELETE FROM portfolio
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM portfolio
                GROUP BY username, script, qty, avg_buy_price
            )
        """)


        conn.commit()
    except Exception as e:
        conn.rollback()
        print("⚠️ run_eod_pipeline error:", e)
        raise
    finally:
        conn.close()

def _run_eod_if_due(username: str):
    """
    Run EOD only when NOT in forced test mode.
    """

    if is_after_market_close():
        run_eod_pipeline(username)

def _sum_closed_today_any(c: sqlite3.Cursor, username: str, script: str, side: str) -> int:
    """Sum closed BUY/SELL across all segments for today."""
    #today = _now_utc().strftime("%Y-%m-%d")
    today = _today_db(c)
    c.execute(
        """
        SELECT COALESCE(SUM(qty),0) FROM orders
         WHERE username=? AND script=? AND order_type=? AND status='Closed'
           AND substr(datetime,1,10)=?
        """,
        (username, script, side.upper(), today),
    )
    return int(c.fetchone()[0] or 0)

def _square_off_intraday_if_eod(username: str):
    """At EOD (after 3:45), auto square-off ONLY intraday positions.

    ✅ Long net (BUY > SELL)   → insert AUTO_SELL and credit funds.
    ✅ Short net (SELL > BUY)  → insert AUTO_BUY (cover) and debit funds.

    IMPORTANT FIXES:
      1) Idempotent per (username, day, script, segment) using eod_intraday_lock.
      2) Script normalization (UPPER(TRIM(script))) so SUZLON/suzlon/`SUZLON ` don't double-process.
      3) Long & Short are computed separately to avoid mixing SELL_FIRST with normal long exits.
      4) Auto legs are tagged via position_type = AUTO_SELL / AUTO_BUY so UI can label them.
    """
    if not is_after_market_close():
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)
        _ensure_funds_row(c, username)
        _ensure_orders_position_type(c)
        _ensure_eod_intraday_lock(c)

        today = _today_db(c)

        # Distinct normalized intraday scripts touched today OR still having intraday Open rows
        c.execute(
            """
            SELECT DISTINCT UPPER(TRIM(COALESCE(script,''))) AS sym
              FROM orders
             WHERE username=?
               AND lower(COALESCE(segment,'intraday'))='intraday'
               AND (
                    status='Open'
                 OR (status='Closed' AND substr(datetime,1,10)=?)
                 OR (status='SETTLED' AND substr(datetime,1,10)=?)
               )
               AND TRIM(COALESCE(script,''))!=''
            """,
            (username, today, today),
        )
        scripts = [r[0] for r in c.fetchall()]
        if not scripts:
            conn.commit()
            return

        def _sum_long(side: str, sym: str) -> int:
            if side.upper() == "BUY":
                c.execute(
                    """
                    SELECT COALESCE(SUM(qty),0)
                      FROM orders
                     WHERE username=?
                       AND UPPER(TRIM(COALESCE(script,'')))=?
                       AND status='Closed'
                       AND lower(COALESCE(segment,'intraday'))='intraday'
                       AND substr(datetime,1,10)=?
                       AND UPPER(COALESCE(order_type,''))='BUY'
                       AND COALESCE(is_short,0)=0
                       AND UPPER(COALESCE(position_type,'BUY'))!='COVER'
                    """,
                    (username, sym, today),
                )
            else:
                c.execute(
                    """
                    SELECT COALESCE(SUM(qty),0)
                      FROM orders
                     WHERE username=?
                       AND UPPER(TRIM(COALESCE(script,'')))=?
                       AND status='Closed'
                       AND lower(COALESCE(segment,'intraday'))='intraday'
                       AND substr(datetime,1,10)=?
                       AND UPPER(COALESCE(order_type,''))='SELL'
                       AND COALESCE(is_short,0)=0
                       AND UPPER(COALESCE(position_type,'')) IN ('SELL','EXIT','AUTO_SELL')
                    """,
                    (username, sym, today),
                )
            return int(c.fetchone()[0] or 0)

        def _sum_short(side: str, sym: str) -> int:
            if side.upper() == "SELL":
                c.execute(
                    """
                    SELECT COALESCE(SUM(qty),0)
                      FROM orders
                     WHERE username=?
                       AND UPPER(TRIM(COALESCE(script,'')))=?
                       AND status='Closed'
                       AND lower(COALESCE(segment,'intraday'))='intraday'
                       AND substr(datetime,1,10)=?
                       AND UPPER(COALESCE(order_type,''))='SELL'
                       AND (COALESCE(is_short,0)=1 OR UPPER(COALESCE(position_type,''))='SELL_FIRST')
                    """,
                    (username, sym, today),
                )
            else:
                c.execute(
                    """
                    SELECT COALESCE(SUM(qty),0)
                      FROM orders
                     WHERE username=?
                       AND UPPER(TRIM(COALESCE(script,'')))=?
                       AND status='Closed'
                       AND lower(COALESCE(segment,'intraday'))='intraday'
                       AND substr(datetime,1,10)=?
                       AND UPPER(COALESCE(order_type,''))='BUY'
                       AND UPPER(COALESCE(position_type,'')) IN ('COVER','AUTO_BUY')
                    """,
                    (username, sym, today),
                )
            return int(c.fetchone()[0] or 0)

        for sym in scripts:
            sym = (sym or "").strip().upper()
            if not sym:
                continue

            # ✅ idempotency guard: only one EOD square-off per script per day
            c.execute(
                """
                INSERT OR IGNORE INTO eod_intraday_lock (username, day, script, segment)
                VALUES (?, ?, ?, 'intraday')
                """,
                (username, today, sym),
            )
            if c.rowcount == 0:
                continue

            long_net = _sum_long("BUY", sym) - _sum_long("SELL", sym)
            short_net = _sum_short("SELL", sym) - _sum_short("BUY", sym)

            if long_net == 0 and short_net == 0:
                continue

            live = get_live_price(sym)
            if live <= 0:
                try:
                    c.execute(
                        """
                        SELECT price
                          FROM orders
                         WHERE username=? AND UPPER(TRIM(COALESCE(script,'')))=? AND COALESCE(price,0)>0
                         ORDER BY datetime DESC
                         LIMIT 1
                        """,
                        (username, sym),
                    )
                    rowp = c.fetchone()
                    if rowp and rowp[0]:
                        live = float(rowp[0])
                except Exception:
                    pass

            if live <= 0:
                try:
                    c.execute(
                        "SELECT avg_buy_price FROM portfolio WHERE username=? AND UPPER(TRIM(COALESCE(script,'')))=?",
                        (username, sym),
                    )
                    rowp = c.fetchone()
                    if rowp and rowp[0]:
                        live = float(rowp[0])
                except Exception:
                    pass

            if live <= 0:
                print(f"⚠️ EOD: live price unavailable for {sym}; skipping square-off so it remains open.")
                continue

            if long_net > 0:
                qty = int(long_net)
                c.execute(
                    "UPDATE funds SET available_amount = available_amount + ? WHERE username=?",
                    (live * qty, username),
                )
                _insert_closed(
                    c, username, sym, "SELL", qty, float(live), "intraday",
                    is_short=0, position_type_override="AUTO_SELL",
                )
                c.execute(
                    """
                    INSERT INTO portfolio_exits (username, script, qty, price, datetime, segment, exit_side)
                    VALUES (?, ?, ?, ?, datetime('now','localtime'), 'intraday', 'AUTO_SELL')
                    """,
                    (username, sym, qty, float(live)),
                )

            if short_net > 0:
                qty = int(short_net)
                c.execute(
                    "UPDATE funds SET available_amount = available_amount - ? WHERE username=?",
                    (live * qty, username),
                )
                _insert_closed(
                    c, username, sym, "BUY", qty, float(live), "intraday",
                    is_short=1, position_type_override="AUTO_BUY",
                )
                c.execute(
                    """
                    INSERT INTO portfolio_exits (username, script, qty, price, datetime, segment, exit_side)
                    VALUES (?, ?, ?, ?, datetime('now','localtime'), 'intraday', 'AUTO_BUY')
                    """,
                    (username, sym, qty, float(live)),
                )

        conn.commit()
    except Exception as e:
        conn.rollback()
        print("EOD square-off error:", e)
    finally:
        conn.close()



def _get_portfolio_qty(c: sqlite3.Cursor, username: str, script: str) -> int:
    c.execute("SELECT COALESCE(SUM(qty),0) FROM portfolio WHERE username=? AND script=?",
              (username, script))
    return int(c.fetchone()[0] or 0)


def _deduct_from_portfolio(c: sqlite3.Cursor, username: str, script: str, qty: int) -> int:
    """
    Deduct up to 'qty' from portfolio holdings. Returns actually deducted amount.
    """
    c.execute("SELECT qty, avg_buy_price FROM portfolio WHERE username=? AND script=?",
              (username, script))
    row = c.fetchone()
    if not row:
        return 0
    cur_qty, avg_price = int(row[0]), float(row[1])
    use = min(cur_qty, int(qty))
    if use <= 0:
        return 0
    new_qty = cur_qty - use
    if new_qty == 0:
        c.execute("DELETE FROM portfolio WHERE username=? AND script=?", (username, script))
    else:
        c.execute(
            "UPDATE portfolio SET qty=?, updated_at=datetime('now','localtime') WHERE username=? AND script=?",
            (new_qty, username, script),
        )
    return use


@router.put("/positions/modify")  # ✅ must be PUT
def modify_position(body: PositionModify):
    """
    Modify SL/Target for the *currently active* position for a script.

    ✅ Fixes:
      - Do NOT recreate / overwrite the orders table schema here (was corrupting DB).
      - Do NOT allow changing executed qty here (use Add / Exit for that).
      - Update SL/Target consistently for the active LONG or active SHORT (SELL FIRST) bucket.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    try:
        _ensure_tables(c)
        _ensure_orders_position_type(c)
        _ensure_orders_updated_at(c)
        _ensure_position_settings_table(c)

        username = (body.username or "").strip()
        script = (body.script or "").strip().upper()
        if not username or not script:
            raise HTTPException(status_code=400, detail="username and script are required")

        c.execute("SELECT date('now','localtime')")
        today = c.fetchone()[0]


        # Determine most recent segment for this script today (fallback to delivery)
        c.execute(
            """
            SELECT lower(COALESCE(segment,'delivery')) AS seg
            FROM orders
            WHERE username=? AND script=? AND substr(datetime,1,10)=?
            ORDER BY datetime DESC, id DESC
            LIMIT 1
            """,
            (username, script, today),
        )
        seg_row = c.fetchone()
        segment = (seg_row["seg"] if seg_row else "delivery")

        # ✅ if frontend sends segment, prefer it
        if body.segment and str(body.segment).lower() in ("intraday", "delivery"):
            segment = str(body.segment).lower()



        # Compute active LONG qty (ignoring COVER buys) and active SHORT qty
        c.execute(
            """
            SELECT
              COALESCE(SUM(CASE WHEN order_type='BUY'
                                AND status='Closed'
                                AND substr(datetime,1,10)=?
                                AND lower(COALESCE(segment,'delivery'))=?
                                AND UPPER(COALESCE(position_type,'BUY'))!='COVER'
                               THEN qty ELSE 0 END),0) AS buy_qty,
              COALESCE(SUM(CASE WHEN order_type='SELL'
                                AND status='Closed'
                                AND substr(datetime,1,10)=?
                                AND lower(COALESCE(segment,'delivery'))=?
                                AND COALESCE(is_short,0)=0
                               THEN qty ELSE 0 END),0) AS sell_qty,
              COALESCE(SUM(CASE WHEN order_type='SELL'
                                AND status='Closed'
                                AND substr(datetime,1,10)=?
                                AND lower(COALESCE(segment,'delivery'))=?
                                AND COALESCE(is_short,0)=1
                               THEN qty ELSE 0 END),0) AS sell_sf_qty,
              COALESCE(SUM(CASE WHEN order_type='BUY'
                                AND status='Closed'
                                AND substr(datetime,1,10)=?
                                AND lower(COALESCE(segment,'delivery'))=?
                                AND UPPER(COALESCE(position_type,''))='COVER'
                               THEN qty ELSE 0 END),0) AS cover_qty
            FROM orders
            WHERE username=? AND script=?
            """,
            (today, segment, today, segment, today, segment, today, segment, username, script),
        )
        r = c.fetchone()
        buy_qty = int(r["buy_qty"] or 0)
        sell_qty = int(r["sell_qty"] or 0)
        sell_sf_qty = int(r["sell_sf_qty"] or 0)
        cover_qty = int(r["cover_qty"] or 0)

        active_long = max(buy_qty - sell_qty, 0)
        active_short = max(sell_sf_qty - cover_qty, 0)
        # ✅ decide which position row user is modifying
        if body.short_first is True:
            bucket = "SHORT"
            active_qty = active_short
        elif body.short_first is False:
            bucket = "LONG"
            active_qty = active_long
        else:
            # fallback if frontend didn't send
            bucket = "SHORT" if active_short > 0 else "LONG"
            active_qty = active_short if active_short > 0 else active_long

        if active_qty <= 0:
            raise HTTPException(status_code=404, detail="No active position found to modify")


        desired_qty = int(body.new_qty or 0)
        if desired_qty <= 0:
            desired_qty = active_qty


        # Disallow changing executed qty here (prevents DB corruption / weird positions)
        # ✅ Disallow changing executed qty here (use Add/Exit)
        if desired_qty > 0 and desired_qty != active_qty:
            raise HTTPException(
                status_code=400,
                detail=f"Qty change is not allowed in Modify. Active qty={active_qty}. Use Add/Exit instead.",
            )

        # ✅ anchor datetime is REQUIRED so we modify correct active row
        anchor_dt = (body.position_datetime or "").strip()
        if not anchor_dt:
            raise HTTPException(status_code=400, detail="position_datetime is required for position modify")
        
        # ✅ Identify the exact anchor trade row (avoids mixing multiple SL/TGT groups)
        c.execute(
            """
            SELECT id, order_type, qty, stoploss, target, datetime, segment, is_short, position_type
            FROM orders
            WHERE username=? AND script=? AND status='Closed'
            AND datetime=? AND substr(datetime,1,10)=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (username, script, anchor_dt, today),
        )
        anchor_row = c.fetchone()
        if not anchor_row:
            raise HTTPException(status_code=404, detail="Position anchor not found for modify")

        anchor_id = int(anchor_row[0])  # ✅ NEW

        # segment (prefer frontend if sent, else anchor)
        anchor_segment = (anchor_row[6] or "delivery").lower()
        segment = anchor_segment
        if body.segment and str(body.segment).lower() in ("intraday", "delivery"):
            segment = str(body.segment).lower()

        # bucket from anchor (truth)
        anchor_is_short = int(anchor_row[7] or 0)
        anchor_pt = (anchor_row[8] or "").upper()
        bucket = "SHORT" if (anchor_is_short == 1 or anchor_pt == "SELL_FIRST") else "LONG"

        # group identity from anchor trade (IMPORTANT: separates SL/TGT=51/40 vs 0/0)
        key_sl = _clean_level(anchor_row[3])
        key_tgt = _clean_level(anchor_row[4])

        def _active_qty_for_group_and_anchor() -> int:
            c.execute(
                """
                SELECT script, order_type, qty, price, stoploss, target, datetime, segment, is_short, position_type
                FROM orders
                WHERE username=? AND script=? AND status='Closed'
                AND substr(datetime,1,10)=?
                ORDER BY datetime ASC, id ASC
                """,
                (username, script, today),
            )
            rows = c.fetchall()

            long_lots, short_lots = [], []
            episode = 0
            first_buy_dt = None

            def _is_flat(): return (not long_lots) and (not short_lots)
            def _reset_episode():
                nonlocal episode, first_buy_dt
                episode += 1
                first_buy_dt = None

            for (sc, side, q, px, sl, tgt, dt, seg, is_short, pt) in rows:
                sc = (sc or "").upper()
                seg = (seg or "delivery").lower()
                if sc != script or seg != segment:
                    continue

                short_bucket = (int(is_short or 0) == 1) or ((pt or "").upper() == "SELL_FIRST")
                row_bucket = "SHORT" if short_bucket else "LONG"

                if _clean_level(sl) != key_sl or _clean_level(tgt) != key_tgt or row_bucket != bucket:
                    continue

                side = (side or "").upper()
                qty = int(q or 0)

                if side == "BUY":
                    to_match = qty
                    while to_match > 0 and short_lots and short_lots[0]["episode"] == episode:
                        lot = short_lots[0]
                        use = min(lot["qty"], to_match)
                        lot["qty"] -= use
                        to_match -= use
                        if lot["qty"] == 0:
                            short_lots.pop(0)

                    if to_match > 0:
                        if _is_flat():
                            _reset_episode()
                        long_lots.append({"qty": to_match, "datetime": dt})
                        if first_buy_dt is None:
                            first_buy_dt = dt

                elif side == "SELL":
                    if _is_flat():
                        _reset_episode()

                    to_match = qty
                    while to_match > 0 and long_lots:
                        lot = long_lots[0]
                        use = min(lot["qty"], to_match)
                        lot["qty"] -= use
                        to_match -= use
                        if lot["qty"] == 0:
                            long_lots.pop(0)

                    if to_match > 0:
                        short_lots.append({"qty": to_match, "datetime": dt, "episode": episode})

            if bucket == "SHORT":
                for lot in short_lots:
                    if (lot.get("datetime") or "") == anchor_dt:
                        return int(lot.get("qty") or 0)
                return 0
            else:
                return int(sum(l.get("qty") or 0 for l in long_lots))

        active_qty = int(_active_qty_for_group_and_anchor() or 0)
        if active_qty <= 0:
            raise HTTPException(status_code=404, detail="No active position found to modify")

        desired_qty = int(body.new_qty or 0)
        if desired_qty <= 0:
            desired_qty = active_qty

        if desired_qty != active_qty:
            raise HTTPException(
                status_code=400,
                detail=f"Qty change is not allowed in Modify. Active qty={active_qty}. Use Add/Exit instead.",
            )

        new_sl = _clean_level(body.stoploss)
        new_tgt = _clean_level(body.target)


        # ✅ Validate SL/Target vs LIVE price (same behavior as Buy/Sell pages)
        live = float(get_live_price(script) or 0.0)
        if live <= 0:
            raise HTTPException(status_code=400, detail="Live price unavailable for SL/Target validation")

        # ✅ Validate SL/Target rules vs LIVE according to bucket
        if bucket == "SHORT":
            _validate_sell_sl_target(new_sl, new_tgt, live)
        else:
            _validate_buy_sl_target(new_sl, new_tgt, live)

        # ✅ Store SL/TGT for THIS active position row only (do NOT update orders table)
        c.execute(
            """
            INSERT INTO position_settings
                (username, script, segment, bucket, anchor_datetime, stoploss, target, updated_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
            ON CONFLICT(username, script, segment, bucket, anchor_datetime)
            DO UPDATE SET
                stoploss=excluded.stoploss,
                target=excluded.target,
                updated_at=datetime('now','localtime')
            """,
            (username, script, segment, bucket, anchor_dt, new_sl, new_tgt),
        )
        # ✅ Update ONLY the exact anchor row (prevents accidental multi-row updates)
        c.execute(
            "UPDATE orders SET stoploss=?, target=? WHERE id=?",
            (new_sl, new_tgt, anchor_id),
        )


        _log_activity(
            c,
            username=username,
            script=script,
            action="MODIFY",
            activity_type="MODIFY",
            qty=int(active_qty or 0),
            price=float(live),   # store live price at modify time
            exchange="NSE",
            segment=segment,
            notes=f"modified SL/Target ({bucket})"
        )

        conn.commit()
        return {
            "status": "ok",
            "message": "Position modified",
            "script": script,
            "segment": segment,
            "bucket": bucket,
            "position_datetime": anchor_dt,
            "active_long": active_long,
            "active_short": active_short,
            "stoploss": new_sl,
            "target": new_tgt,
        }

    finally:
        conn.close()


def _has_active_long(c, username, script, segment, sl, tgt):
    today = _today_db(c)

    c.execute("""
        SELECT
          COALESCE(SUM(
            CASE
              WHEN order_type='BUY'
                   AND UPPER(COALESCE(position_type,'BUY')) != 'COVER'
              THEN qty ELSE 0
            END
          ),0)
          -
          COALESCE(SUM(
            CASE
              WHEN order_type='SELL' AND COALESCE(is_short,0)=0
              THEN qty ELSE 0
            END
          ),0)
        FROM orders
        WHERE username=? AND script=?
          AND lower(segment)=?
          AND status='Closed'
          AND substr(datetime,1,10)=?
          AND COALESCE(stoploss,0)=COALESCE(?,0)
          AND COALESCE(target,0)=COALESCE(?,0)
    """, (
        username,
        script,
        segment,
        today,
        _clean_level(sl),
        _clean_level(tgt),
    ))

    return int(c.fetchone()[0] or 0) > 0


def _move_positions_to_portfolio_or_history(username: str):
    """
    After market close (3:45 PM IST):
      - Intraday:
          * Net long (remaining > 0)  → auto-sell at live price, goes to history (exit_side='SELL')
          * Net short (remaining < 0) → auto-buy cover at live price, goes to history (exit_side='BUY')
      - Delivery:
          * BUY remaining → goes to portfolio
          * SELL → goes to history (already captured as sells)
    """
    # ✅ Use IST (MARKET_TZ) for market-close logic (Render-safe)
    now_ist = datetime.now(MARKET_TZ).time()
    cutoff = time(15, 45)
    if now_ist < cutoff:
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)

        # ✅ Use IST date; keep _today_db(c) as your DB-authoritative date
        today = datetime.now(MARKET_TZ).strftime("%Y-%m-%d")
        today = _today_db(c)

        # fetch today's trades grouped by script/segment
        c.execute("""
          SELECT script, order_type, qty, price, datetime, segment
            FROM orders
           WHERE username=? AND status='Closed'
             AND substr(datetime,1,10)=?
           ORDER BY datetime ASC
        """, (username, today))
        rows = c.fetchall()

        grouped = {}
        for script, side, qty, price, dt, seg in rows:
            seg = (seg or "").lower()
            script = (script or "").upper().strip()
            if not script:
                continue
            grouped.setdefault((script, seg), {"buys": [], "sells": []})
            if (side or "").upper() == "BUY":
                grouped[(script, seg)]["buys"].append((qty, price, dt))
            else:
                grouped[(script, seg)]["sells"].append((qty, price, dt))

        for (script, seg), legs in grouped.items():
            total_buy = sum(q for q, _, _ in legs["buys"])
            total_sell = sum(q for q, _, _ in legs["sells"])
            remaining = total_buy - total_sell  # >0 long; <0 short

            # ✅ One consistent timestamp for ALL DB writes (IST)
            dt_ist = datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S")

            if seg == "intraday":
                live = get_live_price(script)
                if live <= 0:
                    continue

                if remaining > 0:
                    # Auto-sell longs
                    qty = remaining
                    _insert_closed(c, username, script, "SELL", qty, live, "intraday")
                    c.execute("""
                        INSERT INTO portfolio_exits (username, script, qty, price, datetime, segment, exit_side)
                        VALUES (?, ?, ?, ?, ?, 'intraday', 'SELL')
                    """, (username, script, qty, live, dt_ist))

                elif remaining < 0:
                    # Auto-buy cover shorts (SELL FIRST)
                    qty = abs(remaining)
                    _insert_closed(c, username, script, "BUY", qty, live, "intraday")
                    c.execute("""
                        INSERT INTO portfolio_exits (username, script, qty, price, datetime, segment, exit_side)
                        VALUES (?, ?, ?, ?, ?, 'intraday', 'BUY')
                    """, (username, script, qty, live, dt_ist))

            elif seg == "delivery":
                # ✅ Only remaining BUY goes to portfolio (SELL legs already in history)
                if remaining > 0:
                    total_invest = sum(q * p for q, p, _ in legs["buys"])
                    avg_price = (total_invest / total_buy) if total_buy else 0.0

                    c.execute("""
                        INSERT INTO portfolio (username, script, qty, avg_buy_price, current_price, datetime, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        username,
                        script,
                        int(remaining),
                        float(avg_price),
                        float(avg_price),
                        dt_ist,
                        dt_ist
                    ))

        # Clear all today's closed rows (migrated)
        c.execute("""
          DELETE FROM orders
           WHERE username=? AND status='Closed'
             AND substr(datetime,1,10)=?
        """, (username, today))

        conn.commit()
    except Exception as e:
        conn.rollback()
        print("⚠️ EOD move error:", e)
    finally:
        conn.close()


@router.post("/sell/preview")
def preview_sell(order: OrderData):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)

        script = order.script.upper()
        seg = (order.segment or "intraday").lower()
        #today = _now_utc().strftime("%Y-%m-%d")
        today = _today_db(c)
        req_qty = int(order.qty or 0)

        # ✅ segment-aware, non-short only
        c.execute("""
            SELECT COALESCE(SUM(CASE WHEN order_type='BUY' THEN qty ELSE 0 END),0) -
                   COALESCE(SUM(CASE WHEN order_type='SELL' AND COALESCE(is_short,0)=0 THEN qty ELSE 0 END),0)
            FROM orders
            WHERE username=? AND script=? AND status='Closed'
              AND lower(segment)=?
              AND substr(datetime,1,10)=?
        """, (order.username, script, seg, today))
        today_net = int(c.fetchone()[0] or 0)

        c.execute(
            "SELECT COALESCE(SUM(qty),0) FROM portfolio WHERE username=? AND script=?",
            (order.username, script),
        )
        portfolio_qty = int(c.fetchone()[0] or 0)

        owned = max(today_net + portfolio_qty, 0)
        allow_short = bool(order.allow_short)

        if req_qty <= 0:
            return {"can_sell": False, "message": "Quantity must be greater than zero"}

        if owned == 0 and not allow_short:
            return {
                "can_sell": False,
                "needs_confirmation": True,
                "owned_qty": 0,
                "message": f"You didn't buy {script}. Do you still want to sell first?"
            }
        
        # ✅ Oversell confirmation
        if owned > 0 and req_qty > owned and not allow_short:
            return {
                "can_sell": False,
                "needs_confirmation": True,
                "owned_qty": owned,
                "extra_short_qty": req_qty - owned,
                "message": f"You own {owned} qty. Selling {req_qty} will create SELL FIRST for extra {req_qty - owned}. Continue?"
            }


        return {
            "can_sell": True,
            "owned_qty": owned,
            "capped": owned > 0 and req_qty > owned and not allow_short
        }

    finally:
        conn.close()

# -------------------- Public EOD trigger --------------------

@router.post("/run_eod/{username}")
def run_eod(username: str):
    try:
        run_eod_pipeline(username)
        return {"success": True, "message": "EOD completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EOD failed: {e}")

# -------------------- Place order --------------------

def _get_open_position_qty(c: sqlite3.Cursor, username: str, script: str, segment: str) -> int:
    """
    Returns NET open quantity from FIFO position logic.
    BUY > 0, SELL FIRST < 0
    """
    #today = _now_utc().strftime("%Y-%m-%d")
    today = _today_db(c)
    c.execute("""
        SELECT order_type, qty
        FROM orders
        WHERE username=? AND script=? AND status='Closed'
          AND lower(segment)=?
          AND substr(datetime,1,10)=?
        ORDER BY datetime ASC, id ASC
    """, (username, script, segment.lower(), today))

    net = 0
    for side, qty in c.fetchall():
        if side == "BUY":
            net += int(qty)
        elif side == "SELL":
            net -= int(qty)

    return net

def _get_open_short_qty(c, username, script, segment):
    today = _now_utc().strftime("%Y-%m-%d")
    today = _today_db(c)

    c.execute("""
        SELECT COALESCE(SUM(qty),0)
        FROM orders
        WHERE username=? AND script=?
          AND status='Closed'
          AND order_type='SELL'
          AND is_short=1
          AND lower(segment)=?
          AND substr(datetime,1,10)=?
    """, (username, script, segment.lower(), today))
    return int(c.fetchone()[0] or 0)

def _handle_normal_buy(
    c,
    order,
    script,
    seg,
    buy_qty,
    buy_price,
    sl,
    tgt,
    today
):
    # 🔒 same logic you already use for BUY → BUY aggregation
    c.execute("""
        SELECT id, qty, price
        FROM orders
        WHERE username=?
          AND script=?
          AND order_type='BUY'
          AND status='Closed'
          AND lower(segment)=?
          AND substr(datetime,1,10)=?
          AND UPPER(COALESCE(position_type,'BUY'))='BUY'
          AND COALESCE(stoploss,0)=COALESCE(?,0)
          AND COALESCE(target,0)=COALESCE(?,0)
        ORDER BY datetime DESC, id DESC
        LIMIT 1
    """, (
        order.username,
        script,
        seg,
        today,
        sl,
        tgt,
    ))

    row = c.fetchone()

    if row:
        bid, bqty, bprice = row
        new_qty = bqty + buy_qty
        new_avg = ((bqty * bprice) + (buy_qty * buy_price)) / new_qty
        _log_activity(
            c,
            username=order.username,
            script=script,
            action="BUY",
            activity_type="ADD",     # ✅ because it’s merging extra qty
            qty=buy_qty,
            price=buy_price,
            exchange=(order.exchange or "NSE"),
            segment=seg,
            notes=f"merged into existing BUY order id={bid}"
        )

        c.execute("""
            UPDATE orders
            SET qty=?, price=?, datetime=datetime('now','localtime')
            WHERE id=?
        """, (new_qty, round(new_avg, 2), bid))
    else:
        _insert_closed(
            c,
            order.username,
            script,
            "BUY",
            buy_qty,
            buy_price,
            seg,
            stoploss=sl,
            target=tgt,
        )


@router.post("", response_model=Dict[str, Any])   # ✅ no trailing slash
def place_order(order: OrderData):

    print("🔥 PLACE_ORDER HIT 🔥", __file__)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)
        _ensure_orders_position_type(c)

        script = order.script.upper()
        seg = (order.segment or "intraday").lower()

        side_buy = order.order_type.upper() == "BUY"

        # ✅ Auto-correct segment ONLY for SELL (to avoid wrong holdings checks)
        # ❌ Do NOT auto-correct for BUY, because user can intentionally hold both intraday + delivery.
        if not side_buy:
            today = _today_db(c)

            def _open_long_qty(seg_name: str) -> int:
                c.execute("""
                    SELECT
                    COALESCE(SUM(CASE WHEN order_type='BUY'
                                        AND status='Closed'
                                        AND UPPER(COALESCE(position_type,'BUY'))!='COVER'
                                    THEN qty ELSE 0 END),0)
                    -
                    COALESCE(SUM(CASE WHEN order_type='SELL'
                                        AND status='Closed'
                                        AND COALESCE(is_short,0)=0
                                    THEN qty ELSE 0 END),0)
                    FROM orders
                    WHERE username=? AND script=?
                    AND lower(segment)=?
                    AND substr(datetime,1,10)=?
                """, (order.username, script, seg_name, today))
                return int(c.fetchone()[0] or 0)

            if _open_long_qty(seg) <= 0:
                other = "delivery" if seg == "intraday" else "intraday"
                if _open_long_qty(other) > 0:
                    seg = other

        trigger_price = float(order.price or 0.0)
        qty_req = int(order.qty)

        if qty_req <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be greater than zero.")

        live_price = get_live_price(script)
        available = _ensure_funds_row(c, order.username)

        # ===================== SELL FLOW =====================
        is_delivery = seg == "delivery"

        if not side_buy:
            today = _now_utc().strftime("%Y-%m-%d")
            today = _today_db(c)

            # Today's net BUYs
            # ✅ SEGMENT-AWARE today's net BUY
            # ✅ SEGMENT-AWARE today's net BUY (exclude COVER buys)
            c.execute(
                """
                SELECT
                COALESCE(SUM(CASE
                    WHEN order_type='BUY'
                    AND status='Closed'
                    AND lower(COALESCE(segment,'delivery'))=?
                    AND substr(datetime,1,10)=?
                    AND UPPER(COALESCE(position_type,''))!='COVER'
                    THEN qty ELSE 0 END),0) AS buy_qty,

                COALESCE(SUM(CASE
                    WHEN order_type='SELL'
                    AND status='Closed'
                    AND lower(COALESCE(segment,'delivery'))=?
                    AND substr(datetime,1,10)=?
                    AND COALESCE(is_short,0)=0
                    THEN qty ELSE 0 END),0) AS sell_qty
                FROM orders
                WHERE username=? AND script=?
                """,
                (seg, today, seg, today, order.username, script),
            )
            buy_qty, sell_qty = c.fetchone()
            today_net_buy = int(buy_qty or 0) - int(sell_qty or 0)

            # Portfolio holdings
            c.execute(
                "SELECT COALESCE(SUM(qty),0) FROM portfolio WHERE username=? AND script=?",
                (order.username, script),
            )
            portfolio_qty = int(c.fetchone()[0] or 0)

            allow_short = bool(order.allow_short)

            if is_delivery:
                owned_total = max(int(portfolio_qty), 0)
            else:
                owned_total = int(today_net_buy) + max(int(portfolio_qty), 0)

            # ✅ Never treat negative "owned" as owned
            owned_total = max(owned_total, 0)

            # ✅ active short qty today (SELL_FIRST - COVER)
            c.execute(
                """
                SELECT
                COALESCE(SUM(CASE WHEN order_type='SELL' AND status='Closed'
                    AND lower(COALESCE(segment,'delivery'))=?
                    AND substr(datetime,1,10)=?
                    AND COALESCE(is_short,0)=1
                    THEN qty ELSE 0 END),0) AS sf_qty,
                COALESCE(SUM(CASE WHEN order_type='BUY' AND status='Closed'
                    AND lower(COALESCE(segment,'delivery'))=?
                    AND substr(datetime,1,10)=?
                    AND UPPER(COALESCE(position_type,''))='COVER'
                    THEN qty ELSE 0 END),0) AS cover_qty
                FROM orders
                WHERE username=? AND script=?
                """,
                (seg, today, seg, today, order.username, script),
            )
            sf_qty, cover_qty = c.fetchone()
            active_short_qty = max(int(sf_qty or 0) - int(cover_qty or 0), 0)

            # ✅ If already short, this SELL is "ADD to short" → allow automatically
            if active_short_qty > 0:
                allow_short = True

            qty_to_sell = qty_req
            capped = False

            if owned_total == 0 and not allow_short:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "NEEDS_CONFIRM_SHORT",
                        "message": f"You didn't buy {script}. Do you still want to sell first?",
                        "script": script,
                        "requested_qty": qty_req,
                        "owned_qty": 0,
                    },
                )

            # ✅ If selling more than owned, require confirmation unless allow_short=true
            if owned_total > 0 and qty_req > owned_total and not allow_short:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "NEEDS_CONFIRM_OVERSHORT",
                        "message": f"You own {owned_total} qty of {script}. You are trying to sell {qty_req}. "
                                f"Please settle the owned {owned_total} qty FIRST and then place a fresh SELL FIRST order?",
                        "script": script,
                        "requested_qty": qty_req,
                        "owned_qty": owned_total,
                        "extra_short_qty": qty_req - owned_total,
                    },
                )

            # If allow_short=true, sell the full requested quantity
            qty_to_sell = qty_req
            capped = False

            will_short = 1 if (owned_total <= 0 and allow_short) else 0
            position_type = "SELL_FIRST" if will_short else "SELL"
                    # ===================== MARKET SELL =====================
            # ===================== MARKET SELL =====================
            if trigger_price == 0:
                if live_price <= 0:
                    raise HTTPException(status_code=400, detail="Live price unavailable")

                # credit funds
                c.execute(
                    "UPDATE funds SET available_amount = available_amount + ? WHERE username=?",
                    (live_price * qty_to_sell, order.username),
                )

                # ================= DELIVERY SELL =================
                # 🚨 ONLY deduct portfolio for NORMAL SELL
                if is_delivery and not will_short:
                    consume_portfolio = min(qty_to_sell, max(portfolio_qty, 0))
                    new_qty = portfolio_qty - consume_portfolio

                    if new_qty <= 0:
                        c.execute(
                            "DELETE FROM portfolio WHERE username=? AND script=?",
                            (order.username, script),
                        )
                    else:
                        c.execute(
                            """
                            UPDATE portfolio
                            SET qty=?, updated_at=datetime('now','localtime')
                            WHERE username=? AND script=?
                            """,
                            (new_qty, order.username, script),
                        )


                # ================= INTRADAY SELL =================
                else:
                    consume_today = max(0, min(qty_to_sell, today_net_buy))
                    remaining_after_today = qty_to_sell - consume_today
                    consume_portfolio = max(0, min(remaining_after_today, max(portfolio_qty, 0)))

                    if consume_portfolio > 0:
                        new_qty = portfolio_qty - consume_portfolio
                        if new_qty <= 0:
                            c.execute(
                                "DELETE FROM portfolio WHERE username=? AND script=?",
                                (order.username, script),
                            )
                        else:
                            c.execute(
                                """
                                UPDATE portfolio
                                SET qty=?, updated_at=datetime('now','localtime')
                                WHERE username=? AND script=?
                                """,
                                (new_qty, order.username, script),
                            )

                # 🔥 FINAL SAFETY CLEANUP
                c.execute(
                    "DELETE FROM portfolio WHERE username=? AND script=? AND qty=0",
                    (order.username, script),
                )

                # record order
                # ✅ record SELL fill correctly (close BUY first, remainder becomes SELL FIRST)
                split = _execute_sell_fill(
                    c=c,
                    username=order.username,
                    script=script,
                    seg=seg,
                    exec_price=live_price,
                    sell_qty=qty_to_sell,
                    sell_sl=order.stoploss,
                    sell_tgt=order.target,
                )

                conn.commit()

                # ✅ IMPORTANT: if market is closed + delivery, immediately run EOD so it moves to PORTFOLIO
                if seg == "delivery" and is_after_market_close():
                    run_eod_pipeline(order.username)
                
                return {
                    "success": True,
                    "message": "EXECUTED",
                    "triggered": True,
                    "segment": seg,
                    "capped_to_owned": capped,
                    "executed_qty": qty_to_sell,
                    "closed_against_buys_qty": split["closed_qty"],
                    "short_first_qty": split["short_qty"],
                    "short_first": bool(split["short_qty"] > 0),
                }

            # ===================== LIMIT SELL =====================
            if trigger_price > 0 and live_price > 0:

                # SELL FIRST (limit)
                if will_short and ge(live_price, trigger_price):
                    exec_price = float(trigger_price)

                    # credit funds (sell proceeds)
                    c.execute(
                        "UPDATE funds SET available_amount = available_amount + ? WHERE username=?",
                        (exec_price * qty_to_sell, order.username),
                    )

                    split = _execute_sell_fill(
                        c=c,
                        username=order.username,
                        script=script,
                        seg=seg,
                        exec_price=exec_price,
                        sell_qty=qty_to_sell,
                        sell_sl=order.stoploss,
                        sell_tgt=order.target,
                    )

                    conn.commit()
                    return {
                        "success": True,
                        "message": "EXECUTED",
                        "triggered": True,
                        "segment": seg,
                        "executed_qty": qty_to_sell,
                        "closed_against_buys_qty": split["closed_qty"],
                        "short_first_qty": split["short_qty"],
                        "short_first": bool(split["short_qty"] > 0),
                    }




                    # 🔥 cleanup
                    c.execute(
                        "DELETE FROM portfolio WHERE username=? AND script=? AND qty=0",
                        (order.username, script),
                    )

                    conn.commit()
                    return {
                        "success": True,
                        "message": "EXECUTED",
                        "triggered": True,
                        "segment": seg,
                        "capped_to_owned": capped,
                        "executed_qty": qty_to_sell,
                        "short_first": True,
                    }

                # NORMAL LIMIT SELL
                # NORMAL LIMIT SELL
                if ge(live_price, trigger_price):
                    exec_price = float(live_price)

                    c.execute(
                        "UPDATE funds SET available_amount = available_amount + ? WHERE username=?",
                        (exec_price * qty_to_sell, order.username),
                    )

                    # (keep your existing portfolio deduction code here)

                    split = _execute_sell_fill(
                        c=c,
                        username=order.username,
                        script=script,
                        seg=seg,
                        exec_price=exec_price,
                        sell_qty=qty_to_sell,
                        sell_sl=order.stoploss,
                        sell_tgt=order.target,
                    )

                    conn.commit()
                    if seg == "delivery" and is_after_market_close():
                        run_eod_pipeline(order.username)

                    return {
                        "success": True,
                        "message": "EXECUTED",
                        "triggered": True,
                        "segment": seg,
                        "capped_to_owned": capped,
                        "executed_qty": qty_to_sell,
                        "closed_against_buys_qty": split["closed_qty"],
                        "short_first_qty": split["short_qty"],
                        "short_first": bool(split["short_qty"] > 0),
                    }

                    conn.commit()
                    return {
                        "success": True,
                        "message": "EXECUTED",
                        "triggered": True,
                        "segment": seg,
                        "capped_to_owned": capped,
                        "executed_qty": qty_to_sell,
                        "short_first": False,
                    }

            # OPEN LIMIT SELL
            c.execute(
                """
                INSERT INTO orders
                  (username, script, order_type, position_type,
                   qty, price, exchange, segment,
                   status, datetime, pnl, stoploss, target, is_short)
                VALUES
                  (?, ?, 'SELL', ?, ?, ?, ?, ?, 'Open', ?, NULL, ?, ?, ?)
                """,
                (
                    order.username,
                    script,
                    position_type,
                    qty_to_sell,
                    trigger_price,
                    (order.exchange or "NSE").upper(),
                    seg,
                    _now_utc().strftime("%Y-%m-%d %H:%M:%S"),
                    order.stoploss,
                    order.target,
                    will_short,
                ),
            )
            conn.commit()
            return {
                "success": True,
                "message": "PLACED",
                "triggered": False,
                "segment": seg,
                "capped_to_owned": capped,
                "placed_qty": qty_to_sell,
                "short_first": bool(will_short),
            }

        # ===================== BUY FLOW =====================
        position_type = "BUY"
        #today = _now_utc().strftime("%Y-%m-%d")
        today = _today_db(c)
            
        # 🔒 HARD VALIDATION: BUY SL & Target sanity
        _validate_buy_sl_target(
            order.stoploss,
            order.target,
            live_price
        )

        # 🔑 STEP A: COVER ONLY THE EARLIEST SELL FIRST GROUP (FIFO)
        today = _now_utc().strftime("%Y-%m-%d")
        today = _today_db(c)

        c.execute("""
            WITH sf AS (
                SELECT
                    stoploss,
                    target,
                    MIN(datetime) AS first_dt,
                    COALESCE(SUM(qty),0) AS sell_qty
                FROM orders
                WHERE username=?
                AND script=?
                AND order_type='SELL'
                AND COALESCE(is_short,0)=1
                AND status='Closed'
                AND lower(segment)=?
                AND substr(datetime,1,10)=?
                GROUP BY stoploss, target
            ),
            cv AS (
                SELECT
                    stoploss,
                    target,
                    COALESCE(SUM(qty),0) AS cover_qty
                FROM orders
                WHERE username=?
                AND script=?
                AND order_type='BUY'
                AND UPPER(COALESCE(position_type,''))='COVER'
                AND status='Closed'
                AND lower(segment)=?
                AND substr(datetime,1,10)=?
                GROUP BY stoploss, target
            )
            SELECT
                sf.stoploss,
                sf.target,
                (sf.sell_qty - COALESCE(cv.cover_qty,0)) AS open_sf_qty,
                sf.first_dt
            FROM sf
            LEFT JOIN cv
            ON COALESCE(sf.stoploss,0)=COALESCE(cv.stoploss,0)
            AND COALESCE(sf.target,0)=COALESCE(cv.target,0)
            WHERE (sf.sell_qty - COALESCE(cv.cover_qty,0)) > 0
            ORDER BY sf.first_dt ASC
            LIMIT 1
        """, (
            order.username, script, seg, today,
            order.username, script, seg, today
        ))

        row = c.fetchone()

        if row:
            sf_sl, sf_tgt, open_sf_qty, _ = row
            open_sf_qty = int(open_sf_qty or 0)

            # 🔒 HARD BLOCK: don't allow BUY beyond remaining SELL FIRST qty
            if qty_req > open_sf_qty:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "SELL_FIRST_OPEN",
                        "message": (
                            f"Cannot place BUY order.\n\n"
                            f"There is an active SELL FIRST position of {open_sf_qty} quantity for {script}.\n"
                            f"Please BUY exactly {open_sf_qty} quantity to close the short position first.\n"
                            f"After closing it, place a new BUY order to open a fresh long position."
                        ),
                        "script": script,
                        "segment": seg,
                        "sell_first_open_qty": open_sf_qty,
                        "requested_qty": qty_req,
                        "required_buy_qty": open_sf_qty,
                    },
                )

            qty_to_cover = min(open_sf_qty, qty_req)
        else:
            qty_to_cover = 0


        # 🔒 HARD BLOCK: Do NOT allow excess BUY beyond SELL FIRST qty
        if row:
            sf_sl, sf_tgt, sf_qty, _ = row
            sf_qty = int(sf_qty or 0)

            if qty_req > sf_qty:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "SELL_FIRST_OPEN",
                        "message": (
                            f"Cannot place BUY order.\n\n"
                            f"An active SELL FIRST position of {sf_qty} quantity exists for {script}.\n"
                            f"Please BUY exactly {sf_qty} quantity to close the short position first.\n"
                            f"After closing it, place a new BUY order to open a fresh long position."
                        ),
                        "script": script,
                        "segment": seg,
                        "sell_first_qty": sf_qty,
                        "requested_qty": qty_req,
                        "required_buy_qty": sf_qty,
                    },
                )



        remaining_buy_qty = qty_req - qty_to_cover

        # 🔒 Determine if there is an ACTIVE long position
        #today = _now_utc().strftime("%Y-%m-%d")
        today = _today_db(c)

        c.execute("""
            SELECT COALESCE(SUM(CASE WHEN order_type='BUY' THEN qty ELSE 0 END),0) -
                COALESCE(SUM(CASE WHEN order_type='SELL' AND COALESCE(is_short,0)=0 THEN qty ELSE 0 END),0)
            FROM orders
            WHERE username=? AND script=? AND status='Closed'
            AND lower(segment)=?
            AND substr(datetime,1,10)=?
        """, (order.username, script, seg, today))

        net_open_long = int(c.fetchone()[0] or 0)


        if qty_to_cover > 0:
            # ✅ This BUY is NOT opening a long, it is covering an existing SELL FIRST group.
            # ✅ Keep the short-group sl/tgt so matching stays consistent.
            _insert_closed(
                c,
                username=order.username,
                script=script,
                side="BUY",
                qty=qty_to_cover,
                price=live_price,
                segment=seg,
                stoploss=_clean_level(sf_sl),      # ✅ keep group identity
                target=_clean_level(sf_tgt),        # ✅ keep group identity
                is_short=0,
                position_type_override="COVER",     # ✅ critical
            )


        # 🔒 STRICT MERGE KEY = script + segment + stoploss + target + today
        # 🔒 STRICT MERGE KEY = script + segment + stoploss + target + today
        # ✅ IMPORTANT: exclude COVER buys (they are NOT long openings, and must never be merged into)
        c.execute("""
            SELECT id, qty, price, stoploss, target, segment
            FROM orders
            WHERE username=?
            AND script=?
            AND order_type='BUY'
            AND status='Closed'
            AND lower(segment)=?
            AND substr(datetime,1,10)=?
            AND UPPER(COALESCE(position_type,'BUY')) = 'BUY'
            ORDER BY datetime ASC, id ASC
        """, (order.username, script, seg,today))


        #rows = c.fetchall()
        #existing_buy = None
        

        # 🔑 FIX: derive merge key from existing group if any
        if qty_to_cover > 0:
            merge_sl  = _clean_level(sf_sl)
            merge_tgt = _clean_level(sf_tgt)
        else:
            merge_sl  = _clean_level(order.stoploss)
            merge_tgt = _clean_level(order.target)



        # ✅ Find the most recent BUY row with same script+segment+SL+TGT for TODAY
        c.execute("""
            SELECT id, qty, price, stoploss, target
            FROM orders
            WHERE username=?
            AND script=?
            AND order_type='BUY'
            AND status='Closed'
            AND lower(segment)=?
            AND substr(datetime,1,10)=?
            AND UPPER(COALESCE(position_type,'BUY'))='BUY'
            AND COALESCE(stoploss,0)=COALESCE(?,0)
            AND COALESCE(target,0)=COALESCE(?,0)
            ORDER BY datetime DESC, id DESC
            LIMIT 1
        """, (order.username, script, seg, today, merge_sl, merge_tgt))

        row = c.fetchone()
        existing_buy = None

        # ✅ merge only if that group actually still has an ACTIVE LONG open
        if row:
            oid, oqty, oprice, osl, otgt = row
            if _has_active_long(c, order.username, script, seg, merge_sl, merge_tgt):
                existing_buy = (oid, oqty, oprice, osl, otgt)
                print ("_has_active_long sent TRUE and existing BUY function got executed...")



        # ================= MARKET BUY =================
        # ================= MARKET BUY =================
        if trigger_price == 0:

            # 🔒 Case 1: BUY fully consumed while covering SELL FIRST
            if remaining_buy_qty <= 0:
                conn.commit()
                return {
                    "success": True,
                    "message": "EXECUTED",
                    "triggered": True,
                    "segment": seg
                }

            # 🔒 Case 2: BUY has remaining qty → treat as TRUE BUY
            cost = live_price * remaining_buy_qty
            if available < cost:
                raise HTTPException(status_code=400, detail="❌ Insufficient funds")

            # Deduct funds ONLY for remaining BUY qty
            c.execute(
                "UPDATE funds SET available_amount = available_amount - ? WHERE username=?",
                (cost, order.username),
            )

            # =====================================================
            # 🔑 SINGLE SOURCE OF TRUTH FOR BUY POSITIONS
            # =====================================================
            if existing_buy:
                # ✅ Merge into existing BUY (BUY → BUY add)
                old_id, old_qty, old_price, old_sl, old_tgt = existing_buy

                new_qty = old_qty + remaining_buy_qty

                # Weighted average ONLY for same-direction add
                new_avg = (
                    (old_qty * old_price) +
                    (remaining_buy_qty * live_price)
                ) / new_qty
                _log_activity(
                    c,
                    username=order.username,
                    script=script,
                    action="BUY",
                    activity_type="ADD",
                    qty=int(remaining_buy_qty),
                    price=float(live_price),
                    exchange=(order.exchange or "NSE"),
                    segment=seg,
                    notes=f"merged into existing BUY order id={old_id}"
                )

                c.execute("""
                    UPDATE orders
                    SET qty=?,
                        price=?,
                        datetime=datetime('now','localtime')
                    WHERE id=?
                """, (
                    int(new_qty),
                    round(new_avg, 2),
                    old_id
                ))

            else:
                # ✅ Fresh BUY anchor (true BUY FIRST start)
                _insert_closed(
                    c,
                    order.username,
                    script,
                    "BUY",
                    int(remaining_buy_qty),
                    float(live_price),
                    seg,
                    stoploss=_clean_level(order.stoploss),
                    target=_clean_level(order.target),
                    exchange=(order.exchange or "NSE"),   # ✅ add
                )

            # 🔥 Safety: no zero-qty portfolio rows
            c.execute(
                "DELETE FROM portfolio WHERE username=? AND script=? AND qty=0",
                (order.username, script),
            )

            conn.commit()
            return {
                "success": True,
                "message": "EXECUTED",
                "triggered": True,
                "segment": seg
            }

        # ================= OPEN BUY (LIMIT) =================
        c.execute("""
            INSERT INTO orders
              (username, script, order_type, position_type,
               qty, price, exchange, segment,
               status, datetime, pnl, stoploss, target)
            VALUES
              (?, ?, 'BUY', 'BUY', ?, ?, ?, ?, 'Open', ?, NULL, ?, ?)
        """, (
            order.username,
            script,
            qty_req,
            trigger_price,
            (order.exchange or "NSE").upper(),
            seg,
            _now_utc().strftime("%Y-%m-%d %H:%M:%S"),
            order.stoploss,
            order.target,
        ))

        conn.commit()
        return {
            "success": True,
            "message": "PLACED",
            "triggered": False,
            "segment": seg
        }

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"❌ Order failed: {str(e)}")
    finally:
        conn.close()

def process_open_orders():
    
    
    """
    Background job (run every few seconds):

      PASS 1) Trigger OPEN limit orders strictly on their trigger price.
              • BUY executes when live <= trigger (fills at trigger).
              • If trigger_price=0 → treat as market BUY (fills at live).
              • SELL (normal) executes when live >= trigger (fills at trigger).
              • SELL FIRST (is_short=1) executes when live <= trigger (fills at trigger).
              Execution is IN-PLACE (row Open -> Closed) so Positions never duplicates rows.

      PASS 2) SL/Target watcher (today-only, idempotent):
              • If today's net is LONG  (>0): auto SELL qty at LIVE when live >= target OR live <= stoploss.
              • If today's net is SHORT (<0): auto BUY  qty at LIVE when live <= stoploss OR live >= target.
                (Matches your SELL FIRST convention: SL=lower bound, Target=upper bound.)
              Funds are adjusted, a Closed row is inserted, and a portfolio_exits row is recorded.
    """
    if not is_market_open():
        return  # ✅ do nothing if market is closed
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)
        _ensure_orders_position_type(c)


        # ---------------- PASS 1: trigger OPEN orders ----------------
        c.execute("""
            SELECT id, username, script, order_type, qty, price, segment, stoploss, target, is_short
              FROM orders
             WHERE status='Open'
             ORDER BY datetime ASC, id ASC
        """)
        rows = c.fetchall()

        for row in rows:
            order_id, username, script, side, qty, trigger_price, segment, stoploss, target, is_short = row

            # Atomically claim this row
            c.execute("UPDATE orders SET status='Processing' WHERE id=? AND status='Open'", (order_id,))
            if c.rowcount == 0:
                continue
            conn.commit()

            live_price = get_live_price(script)
            if not live_price or live_price <= 0:
                # couldn't price -> revert claim
                c.execute("UPDATE orders SET status='Open' WHERE id=? AND status='Processing'", (order_id,))
                conn.commit()
                continue

            trigger_price = float(trigger_price or 0.0)
            qty = int(qty or 0)

            try:
                if side == "BUY":
                    exec_price = None
                    # Case 1: limit BUY
                    if trigger_price > 0 and le(live_price, trigger_price):
                        exec_price = min(live_price, trigger_price)
                    # Case 2: market BUY (trigger not set)
                    elif trigger_price == 0 and live_price > 0:
                        exec_price = live_price

                    if exec_price is not None:
                        cost = exec_price * qty
                        available = _ensure_funds_row(c, username)
                        if available < cost:
                            # not enough funds -> revert
                            c.execute("UPDATE orders SET status='Open' WHERE id=? AND status='Processing'", (order_id,))
                            conn.commit()
                            continue

                        # deduct funds and close order
                        c.execute(
                            "UPDATE funds SET available_amount = available_amount - ? WHERE username=?",
                            (cost, username),
                        )
                        c.execute(
                            """
                            UPDATE orders
                               SET status   = 'Closed',
                                   price    = ?, 
                                   datetime = datetime('now','localtime')
                             WHERE id = ? AND status='Processing'
                            """,
                            (exec_price, order_id),
                        )
                        conn.commit()
                    else:
                        # not yet triggered
                        c.execute("UPDATE orders SET status='Open' WHERE id=? AND status='Processing'", (order_id,))
                        conn.commit()

                elif side == "SELL":
                    should_exec = False
                    if trigger_price > 0:
                        if is_short:  # SELL FIRST
                            should_exec = ge(live_price, trigger_price)
                        else:         # normal SELL
                            should_exec = ge(live_price, trigger_price)

                    if should_exec:
                        exec_price = max(live_price, trigger_price)
                        credit = exec_price * qty

                        # credit funds (same as your existing logic)
                        c.execute(
                            "UPDATE funds SET available_amount = available_amount + ? WHERE username=?",
                            (credit, username),
                        )

                        # ✅ We must DELETE the open row and insert correct closed rows via splitter
                        # because the fill may close BUY positions first and remainder becomes SELL FIRST.
                        c.execute("DELETE FROM orders WHERE id=? AND status='Processing'", (order_id,))

                        _execute_sell_fill(
                            c=c,
                            username=username,
                            script=script,
                            seg=(segment or "intraday").lower(),
                            exec_price=exec_price,
                            sell_qty=qty,
                            sell_sl=stoploss,
                            sell_tgt=target,
                        )

                        conn.commit()
                    else:
                        # not yet triggered
                        c.execute("UPDATE orders SET status='Open' WHERE id=? AND status='Processing'", (order_id,))
                        conn.commit()

            except Exception:
                # revert if error
                c.execute("UPDATE orders SET status='Open' WHERE id=? AND status='Processing'", (order_id,))
                conn.commit()
                raise

        # ---------------- PASS 2: SL/Target watcher ----------------
        # ---------------- PASS 2: SL/Target watcher ----------------
        # ✅ IMPORTANT: must use DB-local date because orders use datetime('now','localtime')
        today = _today_db(c)

        # We only need (username, script) pairs that have any CLOSED trades today
        c.execute(
            """
            SELECT DISTINCT username, UPPER(TRIM(COALESCE(script,''))) AS sym
              FROM orders
             WHERE status='Closed'
               AND substr(datetime,1,10)=?
               AND TRIM(COALESCE(script,''))!=''
            """,
            (today,),
        )
        pairs = c.fetchall()

        for username, script in pairs:
            script = (script or "").strip().upper()
            if not script:
                continue

            live = float(get_live_price(script) or 0.0)
            if live <= 0:
                continue

            # Ensure funds row exists (don’t fail on missing funds record)
            _ensure_funds_row(c, username)

            # =========================================================
            # (A) LONG buckets: BUY group (stoploss,target) minus normal SELL exits
            # =========================================================
            c.execute(
                """
                SELECT
                    lower(COALESCE(segment,'intraday')) AS seg,
                    stoploss,
                    target,
                    (
                      COALESCE(SUM(CASE
                        WHEN order_type='BUY'
                         AND status='Closed'
                         AND COALESCE(is_short,0)=0
                         AND UPPER(COALESCE(position_type,'BUY'))!='COVER'
                        THEN qty ELSE 0 END),0)
                      -
                      COALESCE(SUM(CASE
                        WHEN order_type='SELL'
                         AND status='Closed'
                         AND COALESCE(is_short,0)=0
                        THEN qty ELSE 0 END),0)
                    ) AS open_qty
                FROM orders
                WHERE username=?
                  AND UPPER(TRIM(COALESCE(script,'')))=?
                  AND status='Closed'
                  AND substr(datetime,1,10)=?
                GROUP BY seg, stoploss, target
                HAVING open_qty > 0
                """,
                (username, script, today),
            )
            long_groups = c.fetchall()

            for seg, sl, tgt, open_qty in long_groups:
                seg = (seg or "intraday").lower()
                sl = _clean_level(sl)
                tgt = _clean_level(tgt)
                qty_to_sell = int(open_qty or 0)
                if qty_to_sell <= 0:
                    continue

                hit = (tgt is not None and ge(live, tgt)) or (sl is not None and le(live, sl))
                if not hit:
                    continue

                # Idempotency lock per (username,script,segment,day,dir,sl,tgt)
                sl_key = _lvl_key(sl)
                tgt_key = _lvl_key(tgt)

                try:
                    c.execute(
                        """
                        INSERT INTO auto_exit_locks
                          (username, script, segment, day, direction, sl_key, tgt_key, datetime)
                        VALUES
                          (?, ?, ?, ?, 'LONG', ?, ?, datetime('now','localtime'))
                        """,
                        (username, script, seg, today, sl_key, tgt_key),
                    )
                except Exception:
                    # Already auto-exited this bucket today
                    continue

                # Credit funds for auto sell
                c.execute(
                    "UPDATE funds SET available_amount = available_amount + ? WHERE username=?",
                    (live * qty_to_sell, username),
                )

                _insert_closed(
                    c=c,
                    username=username,
                    script=script,
                    side="SELL",
                    qty=int(qty_to_sell),
                    price=float(live),
                    segment=seg,
                    stoploss=sl,
                    target=tgt,
                    is_short=0,
                    position_type_override="AUTO_SELL",
                )

                c.execute(
                    """
                    INSERT INTO portfolio_exits (username, script, qty, price, datetime, segment, exit_side)
                    VALUES (?, ?, ?, ?, datetime('now','localtime'), ?, 'AUTO_SELL')
                    """,
                    (username, script, int(qty_to_sell), float(live), seg),
                )

                conn.commit()

            # =========================================================
            # (B) SHORT buckets: SELL_FIRST group (stoploss,target) minus COVER/AUTO_BUY buys
            # Exit rules for SHORT:
            #   - Stoploss hit when live >= SL (SL is ABOVE entry)
            #   - Target hit when live <= TGT (TGT is BELOW entry)
            # =========================================================
            c.execute(
                """
                SELECT
                    lower(COALESCE(segment,'intraday')) AS seg,
                    stoploss,
                    target,
                    (
                      COALESCE(SUM(CASE
                        WHEN order_type='SELL'
                         AND status='Closed'
                         AND (COALESCE(is_short,0)=1 OR UPPER(COALESCE(position_type,''))='SELL_FIRST')
                        THEN qty ELSE 0 END),0)
                      -
                      COALESCE(SUM(CASE
                        WHEN order_type='BUY'
                         AND status='Closed'
                         AND UPPER(COALESCE(position_type,'')) IN ('COVER','AUTO_BUY')
                        THEN qty ELSE 0 END),0)
                    ) AS open_qty
                FROM orders
                WHERE username=?
                  AND UPPER(TRIM(COALESCE(script,'')))=?
                  AND status='Closed'
                  AND substr(datetime,1,10)=?
                GROUP BY seg, stoploss, target
                HAVING open_qty > 0
                """,
                (username, script, today),
            )
            short_groups = c.fetchall()

            for seg, sl, tgt, open_qty in short_groups:
                seg = (seg or "intraday").lower()
                sl = _clean_level(sl)
                tgt = _clean_level(tgt)
                qty_to_cover = int(open_qty or 0)
                if qty_to_cover <= 0:
                    continue

                hit = (sl is not None and ge(live, sl)) or (tgt is not None and le(live, tgt))
                if not hit:
                    continue

                # Idempotency lock for SHORT bucket
                sl_key = _lvl_key(sl)
                tgt_key = _lvl_key(tgt)

                try:
                    c.execute(
                        """
                        INSERT INTO auto_exit_locks
                          (username, script, segment, day, direction, sl_key, tgt_key, datetime)
                        VALUES
                          (?, ?, ?, ?, 'SHORT', ?, ?, datetime('now','localtime'))
                        """,
                        (username, script, seg, today, sl_key, tgt_key),
                    )
                except Exception:
                    continue

                # Debit funds for auto buy-to-cover (allow negative if needed)
                c.execute(
                    "UPDATE funds SET available_amount = available_amount - ? WHERE username=?",
                    (live * qty_to_cover, username),
                )

                _insert_closed(
                    c=c,
                    username=username,
                    script=script,
                    side="BUY",
                    qty=int(qty_to_cover),
                    price=float(live),
                    segment=seg,
                    stoploss=sl,
                    target=tgt,
                    is_short=1,
                    position_type_override="AUTO_BUY",
                )

                c.execute(
                    """
                    INSERT INTO portfolio_exits (username, script, qty, price, datetime, segment, exit_side)
                    VALUES (?, ?, ?, ?, datetime('now','localtime'), ?, 'AUTO_BUY')
                    """,
                    (username, script, int(qty_to_cover), float(live), seg),
                )

                conn.commit()
    
    except Exception as e:
        # optional: conn.rollback() if you want
        print("⚠️ Error in process_open_orders:", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass

# -------------------- Open orders --------------------

@router.get("/{username}")
def get_open_orders(username: str):
    # Auto-run EOD after cutoff so open limits get canceled/refunded as per rules.
    _run_eod_if_due(username)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)
        c.execute(
            """
            SELECT id, script, order_type, qty, price, datetime, segment, stoploss, target, is_short
              FROM orders
             WHERE username = ? AND status = 'Open'
             ORDER BY datetime DESC
            """,
            (username,),
        )
        rows = c.fetchall()
        out = []
        for oid, script, otype, qty, trig, dt, seg, sl, tgt, is_short in rows:
            script_u = (script or "").upper()
            live = get_live_price(script_u)
            away = abs(live - float(trig)) if (live and trig) else None
            out.append({
                "id": oid,
                "script": script_u,
                "type": otype,
                "qty": int(qty),
                "trigger_price": float(trig),
                "price": float(trig),
                "live_price": float(live or 0),
                "datetime": dt,
                "segment": seg,
                "stoploss": sl,
                "target": tgt,
                "status": "Open",
                "short_first": bool(is_short),   # 👈 expose to UI
                "status_msg": (f"Yet to trigger, ₹{away:.2f} away" if away is not None else "Awaiting market/quote"),
            })
        return out
    finally:
        conn.close()


# -------------------- Positions (EXECUTED ONLY) --------------------

@router.get("/positions/{username}")
def get_positions(username: str):
    """
    Positions tab:
      • FIFO pairing of BUY/SELL (long) and SELL FIRST/BUY (short)
      • Positions are UNIQUE by (script, segment, stoploss, target)
      • Preserves existing UI structure & calculations
    if is_after_market_close():
        return []

    """
 

    _run_eod_if_due(username)

    now = _now_utc().time()
    #today = _now_utc().strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = _today_db(c)

    try:
        _ensure_tables(c)
        _ensure_position_settings_table(c)

        # Fetch all executed trades for today
        c.execute(
            """
            SELECT script, order_type, qty, price, stoploss, target,
                   datetime, segment, is_short,  position_type
              FROM orders
             WHERE username = ?
               AND status   = 'Closed'
               AND substr(datetime,1,10) = ?
             ORDER BY datetime ASC, id ASC
            """,
            (username, today),
        )
        rows = c.fetchall()

        positions: List[Dict[str, Any]] = []

        # 🔑 KEY FIX: state grouped by (script, segment, stoploss, target)
        state: Dict[tuple, Dict[str, Any]] = {}
        def _overlay_active_settings(pos: Dict[str, Any]) -> Dict[str, Any]:
            """
            Apply SL/TGT only for inactive=False rows from position_settings.
            """
            try:
                if pos.get("inactive"):
                    return pos

                sym = (pos.get("symbol") or "").upper()
                seg = (pos.get("segment") or "intraday").lower()
                bucket = "SHORT" if pos.get("short_first") else "LONG"
                anchor = (pos.get("datetime") or "").strip()
                if not sym or not anchor:
                    return pos

                c.execute(
                    """
                    SELECT stoploss, target
                    FROM position_settings
                    WHERE username=? AND script=? AND segment=? AND bucket=? AND anchor_datetime=?
                    """,
                    (username, sym, seg, bucket, anchor),
                )
                r = c.fetchone()
                if r:
                    pos["stoploss"] = r[0]
                    pos["target"] = r[1]
                return pos
            except Exception:
                return pos


        def pos_key(script, segment, sl, tgt, is_short, position_type):
            # SHORT bucket if:
            # - SELL_FIRST rows (is_short=1), OR
            # - COVER buys, OR
            # - explicit position_type SELL_FIRST
            pt = (position_type or "").upper()
            short_bucket = (int(is_short or 0) == 1) or (pt in ("SELL_FIRST", "COVER"))

            return (
                script.upper(),
                (segment or "intraday").lower(),
                _clean_level(sl),
                _clean_level(tgt),
                "SHORT" if short_bucket else "LONG",   # ✅ KEY FIX
            )


        def _is_flat(st):
            return not st["long_lots"] and not st["short_lots"]

        def _reset_episode(st):
            st["episode"] += 1
            st["base_total_buy_qty"] = 0
            st["base_total_sell_qty"] = 0
            st["first_buy_dt"] = None

        # ---------------- BUILD STATE ----------------
        for script, side, qty, price, sl, tgt, dt, segment, is_short,position_type in rows:

            script  = (script or "").upper()
            side    = (side or "").upper()
            segment = (segment or "intraday").lower()
            qty     = int(qty or 0)
            price   = float(price or 0.0)

            key = pos_key(script, segment, sl, tgt,  is_short, position_type)

            st = state.setdefault(key, {
                "script": script,
                "segment": segment,
                "stoploss": _clean_level(sl),
                "target": _clean_level(tgt),
                "long_lots": [],
                "short_lots": [],
                "long_exits": [],
                "short_covers": [],
                "first_buy_dt": None,

                # existing
                "base_total_buy_qty": 0,

                # 🔑 NEW (minimal)
                "base_total_sell_qty": 0,
                "episode": 0,
            })

            # ---------------- BUY ----------------
            # ---------------- BUY ----------------
            if side == "BUY":

                to_match = qty

                # Cover shorts first (BUY used to cover must NOT inflate BUY denominator)
                while (
                    to_match > 0
                    and st["short_lots"]
                    and st["short_lots"][0]["episode"] == st["episode"]
                ):
                    lot = st["short_lots"][0]
                    use = min(lot["qty"], to_match)

                    entry = lot["price"]
                    exitp = price
                    per_share = entry - exitp
                    pnl_val = per_share * use
                    pct = ((per_share / entry) * 100.0) if entry else 0.0

                    st["short_covers"].append({
                        "qty": use,
                        "datetime": dt,
                        "entry_price": entry,
                        "exit_price": exitp,
                        "pnl_value": round(pnl_val, 2),
                        "pnl_percent": round(pct, 2),
                        "segment": segment,
                        "sl": st["stoploss"],
                        "tgt": st["target"],
                        "denom": st["base_total_sell_qty"],
                        "episode": st["episode"],
                    })

                    lot["qty"] -= use
                    to_match   -= use
                    if lot["qty"] == 0:
                        st["short_lots"].pop(0)

                # ✅ Only the leftover BUY becomes a LONG position and should affect denominator
                if to_match > 0:
                    # Start a fresh episode only when we are actually opening something new
                    if _is_flat(st):
                        _reset_episode(st)

                    # ✅ denom only counts qty that actually opens LONG
                    st["base_total_buy_qty"] += to_match

                    st["long_lots"].append({
                        "qty": to_match,
                        "price": price,
                        "datetime": dt,
                    })

                    if st["first_buy_dt"] is None:
                        st["first_buy_dt"] = dt




            # ---------------- SELL ----------------
            elif side == "SELL":

                if _is_flat(st):
                    _reset_episode(st)

                to_match = qty

                # Close long lots first
                while to_match > 0 and st["long_lots"]:
                    lot = st["long_lots"][0]
                    use = min(lot["qty"], to_match)

                    entry = lot["price"]
                    exitp = price
                    per_share = exitp - entry
                    pnl_val = per_share * use
                    pct = ((exitp / entry - 1.0) * 100.0) if entry else 0.0

                    st["long_exits"].append({
                        "qty": use,
                        "datetime": dt,
                        "entry_price": entry,
                        "exit_price": exitp,
                        "pnl_value": round(pnl_val, 2),
                        "pnl_percent": round(pct, 2),
                        "segment": segment,
                        "sl": st["stoploss"],
                        "tgt": st["target"],
                        "denom": st["base_total_buy_qty"],
                        "episode": st["episode"],
                    })

                    lot["qty"] -= use
                    to_match   -= use
                    if lot["qty"] == 0:
                        st["long_lots"].pop(0)

                # Remaining SELL becomes short (SELL FIRST)
                if to_match > 0:
                    st["base_total_sell_qty"] += to_match

                    st["short_lots"].append({
                        "qty": to_match,
                        "price": price,
                        "datetime": dt,
                        "episode": st["episode"],   # 🔒 CRITICAL
                    })

        # ---------------- BUILD RESPONSE ----------------
        for st in state.values():
            script  = st["script"]
            segment = st["segment"]
            sl      = st["stoploss"]
            tgt     = st["target"]

            # ✅ FIX: denominator for BUY positions/exits = original total BUY qty for this group
            denom_total = int(st.get("base_total_buy_qty") or 0)

            # Inactive long exits
            # -------- AGGREGATE INACTIVE LONG EXITS --------
            # Inactive long exits
            # -------- AGGREGATE INACTIVE LONG EXITS (LOCK PER EPISODE) --------
            if st["long_exits"]:
                by_ep = {}
                for e in st["long_exits"]:
                    by_ep.setdefault(e.get("episode", 0), []).append(e)

                for exits in by_ep.values():
                    total_qty = sum(x["qty"] for x in exits)
                    if total_qty <= 0:
                        continue

                    entry_notional = sum(x["qty"] * x["entry_price"] for x in exits)
                    avg_entry = entry_notional / total_qty if total_qty else 0.0

                    # last exit price of that episode
                    exit_price = exits[-1]["exit_price"]

                    pnl_value = sum(x["pnl_value"] for x in exits)
                    pnl_percent = (
                        ((exit_price / avg_entry - 1.0) * 100.0)
                        if avg_entry else 0.0
                    )

                    # ✅ denom must be locked to the SAME episode (take the max snapshot in that ep)
                    denom_snapshot = max(int(x.get("denom") or 0) for x in exits)
                    if denom_snapshot <= 0:
                        denom_snapshot = total_qty

                    positions.append({
                        "symbol": script,
                        "type": "BUY",
                        "qty": int(total_qty),

                        # ✅ NEVER allow total < qty
                        "total": int(max(denom_snapshot, total_qty)),

                        "price": round(avg_entry, 2),
                        "exit_price": exit_price,
                        "live_price": exit_price,
                        "pnl_value": round(pnl_value, 2),
                        "pnl_percent": round(pnl_percent, 2),
                        "abs_per_share": round(exit_price - avg_entry, 4),
                        "abs_pct": round(pnl_percent, 4),
                        "script_pnl": round(pnl_value, 2),
                        "stoploss": sl,
                        "target": tgt,
                        "inactive": True,

                        # keep episode’s last timestamp
                        "datetime": exits[-1]["datetime"],

                        "segment": segment,
                        "short_first": False,
                    })



            # Inactive short covers
            # -------- AGGREGATE INACTIVE SHORT COVERS (SELL FIRST) --------
            if st["short_covers"]:
                by_ep = {}
                for s in st["short_covers"]:
                    by_ep.setdefault(s["episode"], []).append(s)

                for covers in by_ep.values():
                    total_qty = sum(x["qty"] for x in covers)

                    entry_notional = sum(x["qty"] * x["entry_price"] for x in covers)
                    avg_entry = entry_notional / total_qty if total_qty else 0.0

                    exit_price = covers[-1]["exit_price"]

                    pnl_value = sum(x["pnl_value"] for x in covers)
                    pnl_percent = (
                        ((avg_entry - exit_price) / avg_entry * 100.0)
                        if avg_entry else 0.0
                    )

                    denom_snapshot = covers[-1].get("denom", total_qty)

                    positions.append({
                        "symbol": script,
                        "type": "SELL",
                        "qty": total_qty,
                        "total": denom_snapshot,
                        "price": round(avg_entry, 2),
                        "exit_price": exit_price,
                        "live_price": exit_price,
                        "pnl_value": round(pnl_value, 2),
                        "pnl_percent": round(pnl_percent, 2),
                        "abs_per_share": round(avg_entry - exit_price, 4),
                        "abs_pct": round(pnl_percent, 4),
                        "script_pnl": round(pnl_value, 2),
                        "stoploss": sl,
                        "target": tgt,
                        "inactive": True,
                        "datetime": covers[-1]["datetime"],
                        "segment": segment,
                        "short_first": True,
                    })

            # Active shorts
            # ---------------- ACTIVE SHORTS (SELL FIRST) - MERGED BY (segment, sl, tgt) ----------------
            if st["short_lots"]:
                lp = float(get_live_price(script) or 0.0)

                total_qty = sum(int(l["qty"] or 0) for l in st["short_lots"])
                if total_qty > 0:
                    wavg_entry = (
                        sum(int(l["qty"] or 0) * float(l["price"] or 0.0) for l in st["short_lots"])
                        / total_qty
                    )

                    per_share = wavg_entry - lp          # ✅ short profit when price falls
                    pnl_val = per_share * total_qty
                    pnl_percent = ((per_share / wavg_entry) * 100.0) if wavg_entry else 0.0

                    # ✅ choose earliest lot datetime as the "anchor" for modify UI
                    anchor_dt = min((l.get("datetime") or "") for l in st["short_lots"]) or today + " 00:00:00"

                    positions.append(_overlay_active_settings({
                        "symbol": script,
                        "type": "SELL",
                        "qty": int(total_qty),
                        "total": int(st["base_total_sell_qty"] or total_qty),
                        "price": round(wavg_entry, 2),
                        "live_price": lp,
                        "pnl_value": round(pnl_val, 2),
                        "pnl_percent": round(pnl_percent, 2),
                        "abs_per_share": round(per_share, 4),
                        "abs_pct": round(pnl_percent, 4),
                        "script_pnl": round(pnl_val, 2),

                        "stoploss": sl,
                        "target": tgt,

                        "inactive": False,
                        "datetime": anchor_dt,      # ✅ merged anchor
                        "segment": segment,
                        "short_first": True,
                    }))



            # Active longs
            if st["long_lots"]:
                wq = sum(l["qty"] for l in st["long_lots"])
                wavg = sum(l["qty"] * l["price"] for l in st["long_lots"]) / wq
                live_now = float(get_live_price(script) or 0.0)
                per_share = live_now - wavg
                pct = ((live_now / wavg - 1.0) * 100.0) if wavg else 0.0
                pnl_val = per_share * wq

                positions.append(_overlay_active_settings({
                    "symbol": script,
                    "type": "BUY",
                    "qty": wq,
                    "total": max(int(st["base_total_buy_qty"] or 0), int(wq)),
                    "price": round(wavg, 2),
                    "live_price": live_now,
                    "pnl_value": round(pnl_val, 2),
                    "pnl_percent": round(pct, 2),
                    "abs_per_share": round(per_share, 4),
                    "abs_pct": round(pct, 4),
                    "script_pnl": round(pnl_val, 2),

                    "stoploss": sl,
                    "target": tgt,

                    "inactive": False,
                    "segment": segment,
                    "short_first": False,
                    "datetime": st["first_buy_dt"] or today + " 00:00:00",  # ✅ anchor
                }))

        return positions

    except Exception as e:
        print("⚠️ Error in get_positions:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        conn.close()


@router.post("/exit")
def exit_order(order: OrderData):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)
        _ensure_orders_position_type(c)

        username = (order.username or "").strip()
        script = (order.script or "").strip().upper()
        if not username:
            raise HTTPException(status_code=400, detail="Invalid username")
        if not script:
            raise HTTPException(status_code=400, detail="Invalid script")

        _ensure_funds_row(c, username)

        seg = (order.segment or "intraday").strip().lower()
        if seg not in ("intraday", "delivery"):
            seg = "intraday"

        side = (order.order_type or "SELL").upper()
        today = _today_db(c)

        def owned_today_for(seg_name: str) -> int:
            c.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN order_type='BUY'
                                    AND UPPER(COALESCE(position_type,'BUY'))!='COVER'
                                   THEN qty ELSE 0 END),0)
                  -
                  COALESCE(SUM(CASE WHEN order_type='SELL'
                                    AND COALESCE(is_short,0)=0
                                   THEN qty ELSE 0 END),0)
                FROM orders
                WHERE username=? AND script=? AND status='Closed'
                  AND lower(COALESCE(segment,'intraday'))=?
                  AND substr(datetime,1,10)=?
                """,
                (username, script, seg_name, today),
            )
            return int(c.fetchone()[0] or 0)

        # portfolio qty (signed: long + / short -)
        # portfolio qty (signed)
        c.execute(
            "SELECT COALESCE(SUM(qty),0) FROM portfolio WHERE username=? AND script=?",
            (username, script),
        )
        portfolio_signed = int(c.fetchone()[0] or 0)

        portfolio_long_qty = max(portfolio_signed, 0)
        portfolio_short_qty = max(-portfolio_signed, 0)

        # ✅ backward compatible: keep old variable name used later in exit_order()
        portfolio_qty = portfolio_long_qty


        # If portfolio has ANY qty (long or short), EXIT is effectively delivery
        if (portfolio_long_qty > 0 or portfolio_short_qty > 0) and seg != "delivery":
            seg = "delivery"

        # -------------------- BUY EXIT = COVER SHORT --------------------
        if side == "BUY":
            c.execute(
                """
                SELECT
                  COALESCE(SUM(CASE
                      WHEN (position_type='SELL_FIRST' OR COALESCE(is_short,0)=1)
                           AND order_type='SELL'
                      THEN qty ELSE 0 END),0)
                  -
                  COALESCE(SUM(CASE
                      WHEN position_type='COVER' AND order_type='BUY'
                      THEN qty ELSE 0 END),0)
                FROM orders
                WHERE username=? AND script=? AND status='Closed'
                  AND lower(COALESCE(segment,'intraday'))=?
                  AND substr(datetime,1,10)=?
                """,
                (username, script, seg, today),
            )
            open_short = int(c.fetchone()[0] or 0)
            # ✅ If portfolio has short qty (delivery short), use that as open short
            if seg == "delivery" and portfolio_short_qty > 0:
                open_short = portfolio_short_qty

            if open_short <= 0:
                raise HTTPException(status_code=400, detail="No open short position to exit")

            req_qty = int(order.qty or open_short)

            # ✅ HARD BLOCK: don't allow cover qty > open short qty
            if order.qty is not None and req_qty > open_short:
                raise HTTPException(
                    status_code=400,
                    detail=f"❌ Exit qty ({req_qty}) cannot be greater than open short qty ({open_short}) for {script}."
                )
            
            cover_qty = req_qty  # now safe
            if cover_qty <= 0:
                raise HTTPException(status_code=400, detail="Invalid cover qty")


            live_price = float(get_live_price(script) or 0.0)
            if live_price <= 0:
                raise HTTPException(status_code=400, detail="Live price unavailable")

            c.execute("SELECT COALESCE(available_amount,0) FROM funds WHERE username=?", (username,))
            avail = float(c.fetchone()[0] or 0.0)
            need = live_price * cover_qty
            if avail < need:
                raise HTTPException(status_code=400, detail="❌ Insufficient funds to cover short")

            c.execute(
                "UPDATE funds SET available_amount = available_amount - ? WHERE username=?",
                (need, username),
            )

            _insert_closed(
                c=c,
                username=username,
                script=script,
                side="BUY",
                qty=int(cover_qty),
                price=float(live_price),
                segment=seg,
                stoploss=None,
                target=None,
                is_short=1,
                position_type_override="COVER",
            )

            conn.commit()
            if seg == "delivery" and is_after_market_close():
                run_eod_pipeline(username)

            return {"success": True, "message": f"Covered {cover_qty} {script} at {live_price}"}

        # -------------------- SELL EXIT = CLOSE LONG --------------------
        owned_today = max(owned_today_for(seg), 0)

        # If user sent wrong segment and there is NO portfolio, try the other segment before failing
        if owned_today == 0 and portfolio_qty == 0:
            other = "delivery" if seg == "intraday" else "intraday"
            other_owned = max(owned_today_for(other), 0)
            if other_owned > 0:
                seg = other
                owned_today = other_owned

        owned_total = owned_today + portfolio_long_qty
        if owned_total <= 0:
            raise HTTPException(status_code=400, detail="No open position to exit")


        # If qty not provided -> full exit
        req_qty = int(order.qty or owned_total)

        # ✅ HARD BLOCK: don't allow exit qty > open qty
        if order.qty is not None and req_qty > owned_total:
            raise HTTPException(
                status_code=400,
                detail=f"❌ Exit qty ({req_qty}) cannot be greater than open qty ({owned_total}) for {script}."
            )

        exit_qty = req_qty  # now safe
        if exit_qty <= 0:
            raise HTTPException(status_code=400, detail="Invalid exit qty")



        live_price = float(get_live_price(script) or 0.0)
        if live_price <= 0:
            raise HTTPException(status_code=400, detail="Live price unavailable")

        c.execute(
            "UPDATE funds SET available_amount = available_amount + ? WHERE username=?",
            (live_price * exit_qty, username),
        )

        # ✅ CRITICAL: cap today's part so _execute_sell_fill can NEVER oversell and create SELL_FIRST
        qty_from_today = min(exit_qty, owned_today)
        qty_from_portfolio = exit_qty - qty_from_today

        if qty_from_portfolio > 0:
            new_qty = portfolio_qty - qty_from_portfolio
            if new_qty <= 0:
                c.execute("DELETE FROM portfolio WHERE username=? AND script=?", (username, script))
            else:
                c.execute(
                    "UPDATE portfolio SET qty=?, updated_at=datetime('now','localtime') WHERE username=? AND script=?",
                    (int(new_qty), username, script),
                )

            _insert_closed(
                c=c,
                username=username,
                script=script,
                side="SELL",
                qty=int(qty_from_portfolio),
                price=float(live_price),
                segment=seg,
                stoploss=None,
                target=None,
                is_short=0,
                position_type_override="EXIT",
            )

        if qty_from_today > 0:
            _execute_sell_fill(
                c=c,
                username=username,
                script=script,
                seg=seg,
                exec_price=float(live_price),
                sell_qty=int(qty_from_today),  # ✅ capped
                sell_sl=None,
                sell_tgt=None,
            )

        conn.commit()
        if seg == "delivery" and is_after_market_close():
            run_eod_pipeline(username)

        return {"success": True, "message": f"Exited {exit_qty} {script} at {live_price}"}

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()

# -------------------- Modify & Cancel --------------------

@router.put("/{order_id}")
def modify_order(order_id: int, order: OrderData):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)
        _ensure_orders_position_type(c)

        # ✅ Fetch existing OPEN order (source of truth)
        c.execute("""
            SELECT username, script, order_type, COALESCE(is_short,0), COALESCE(segment,'intraday')
            FROM orders
            WHERE id=? AND status='Open'
        """, (order_id,))
        row = c.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Order not found")

        db_username, db_script, db_side, db_is_short, db_seg = row
        script = (db_script or "").upper()
        side = (db_side or "").upper()
        is_short = int(db_is_short or 0)

        # Normalize incoming fields
        new_qty = int(order.qty or 0)
        new_price = float(order.price or 0.0)   # LIMIT trigger
        # ✅ HARD BLOCK for SELL FIRST LIMIT modify:
        # Do not allow trigger/limit <= current live, otherwise it executes immediately.
        if side == "SELL" and is_short == 1 and new_price > 0:
            live = float(get_live_price(script) or 0.0)
            if live <= 0:
                raise HTTPException(status_code=400, detail="Live price unavailable for SELL FIRST validation")
            if new_price <= live:
                raise HTTPException(
                    status_code=400,
                    detail=f"❌ SELL FIRST limit price ({new_price}) must be GREATER than current price ({live})"
                )

        new_sl = _clean_level(order.stoploss)
        new_tgt = _clean_level(order.target)

        if new_qty <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be greater than zero.")

        # ✅ Validate SL/Target against LIVE price (same as placement pages)
        live = float(get_live_price(script) or 0.0)
        if live <= 0:
            raise HTTPException(status_code=400, detail="Live price unavailable for SL/Target validation")

        # BUY (long) validation
        if side == "BUY":
            _validate_buy_sl_target(new_sl, new_tgt, live)

        # SELL validation
        elif side == "SELL":
            # If user has provided SL/TGT, validate them.
            # For SELL_FIRST (is_short=1) this is absolutely required logic.
            if new_sl is not None or new_tgt is not None:
                _validate_sell_sl_target(new_sl, new_tgt, live)

        # ✅ Apply update
        c.execute("""
            UPDATE orders
            SET qty       = ?,
                price     = ?,
                stoploss  = ?,
                target    = ?,
                segment   = ?
            WHERE id = ? AND status='Open'
        """, (
            new_qty,
            new_price,
            new_sl,
            new_tgt,
            (order.segment or db_seg or "intraday").lower(),
            order_id
        ))

        conn.commit()
        return {"message": "Order modified successfully"}

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()


@router.post("/positions/close")
def close_position(data: dict):
    """
    Close a symbol across both tabs for the user:

      1) Cancels all OPEN orders (BUY & SELL) for this symbol.
         - Refunds blocked funds for OPEN BUY orders (qty * price).
      2) Removes today's executed rows (BUY & SELL) from Positions.
         - Refunds total BUY amount executed today (sum of qty * price for today's BUY fills).
      3) If there were any executed SELL rows with a valid exit_price, 
         keeps that in record for display/reference.

    Returns total refund summary and cleanup counts.
    """
    username = (data.get("username") or "").strip()
    script = (data.get("script") or "").upper().strip()

    if not username or not script:
        raise HTTPException(status_code=400, detail="Missing username or script")

    today = _now_utc().strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_tables(c)
        _ensure_funds_row(c, username)

        # ---- Step 1: refund blocked funds on OPEN BUY limits for this symbol
        c.execute(
            """
            SELECT COALESCE(SUM(qty * price), 0)
              FROM orders
             WHERE username=? AND script=? AND status='Open' AND order_type='BUY'
            """,
            (username, script),
        )
        refund_open_buys = float(c.fetchone()[0] or 0.0)
        if refund_open_buys > 0:
            c.execute(
                "UPDATE funds SET available_amount = available_amount + ? WHERE username=?",
                (refund_open_buys, username),
            )

        # ---- Step 2: cancel all OPEN orders (BUY & SELL)
        c.execute(
            "UPDATE orders SET status='Cancelled' WHERE username=? AND script=? AND status='Open'",
            (username, script),
        )
        cancelled_count = c.rowcount

        # ---- Step 3: calculate today's executed BUY refund
        c.execute(
            """
            SELECT COALESCE(SUM(qty * price), 0)
              FROM orders
             WHERE username=? AND script=? AND status='Closed'
               AND order_type='BUY' AND substr(datetime,1,10)=?
            """,
            (username, script, today),
        )
        refund_today_buys = float(c.fetchone()[0] or 0.0)
        if refund_today_buys > 0:
            c.execute(
                "UPDATE funds SET available_amount = available_amount + ? WHERE username=?",
                (refund_today_buys, username),
            )

        # ---- Step 4: get average of today's SELLs (for exit_price record)
        c.execute(
            """
            SELECT AVG(price)
              FROM orders
             WHERE username=? AND script=? AND order_type='SELL'
               AND status='Closed' AND substr(datetime,1,10)=?
            """,
            (username, script, today),
        )
        avg_sell_exit = float(c.fetchone()[0] or 0.0)

        # ---- Step 5: delete today's executed rows (BUY & SELL)
        c.execute(
            """
            DELETE FROM orders
             WHERE username=? AND script=? AND status='Closed'
               AND substr(datetime,1,10)=?
            """,
            (username, script, today),
        )
        deleted_today_count = c.rowcount

        # ---- Step 6: (optional) record exit info in portfolio_exits if SELLs occurred
        if avg_sell_exit > 0:
            try:
                c.execute(
                    """
                    INSERT INTO portfolio_exits (username, script, qty, price, datetime, segment, exit_side)
                    VALUES (?, ?, 0, ?, datetime('now','localtime'), 'intraday', 'SELL')
                    """,
                    (username, script, avg_sell_exit),
                )
            except Exception:
                pass

        conn.commit()

        # ---- Step 7: build response
        total_refund = refund_open_buys + refund_today_buys
        msg = (
            f"Closed {script}. Cancelled {cancelled_count} open order(s). "
            f"Removed {deleted_today_count} executed row(s) for today. "
            f"Refunded ₹{total_refund:.2f} "
            f"(open blocks ₹{refund_open_buys:.2f} + today buys ₹{refund_today_buys:.2f})."
        )

        if avg_sell_exit > 0:
            msg += f" Last exit price recorded: ₹{avg_sell_exit:.2f}."

        return {
            "success": True,
            "message": msg,
            "refund_open_buys": round(refund_open_buys, 2),
            "refund_today_buys": round(refund_today_buys, 2),
            "total_refund": round(total_refund, 2),
            "cancelled_open_orders": int(cancelled_count),
            "deleted_today_rows": int(deleted_today_count),
            "last_exit_price": round(avg_sell_exit, 2) if avg_sell_exit else None,
        }

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to close position: {str(e)}")
    finally:
        conn.close()


@router.post("/convert-to-market/{order_id}")
def convert_open_order_to_market(order_id: int):
    """
    Convert an OPEN LIMIT order into MARKET and execute immediately.

    ✅ SELL FIX:
      - Do NOT just UPDATE the open SELL row to Closed.
      - DELETE the open row and insert correct Closed rows via _execute_sell_fill()
        so it closes BUY positions first and only remainder becomes SELL FIRST.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        _ensure_tables(c)
        _ensure_orders_position_type(c)

        # 1️⃣ Fetch the open order
        c.execute("""
            SELECT username, script, order_type, qty, segment, is_short, stoploss, target
            FROM orders
            WHERE id=? AND status='Open'
        """, (order_id,))
        row = c.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Open order not found")

        username, script, side, qty, segment, is_short, stoploss, target = row
        script = (script or "").upper()
        qty = int(qty or 0)
        seg = (segment or "intraday").lower()
        is_short = int(is_short or 0)

        if qty <= 0:
            raise HTTPException(status_code=400, detail="Invalid quantity")

        # 2️⃣ Get live price
        live_price = get_live_price(script)
        if not live_price or live_price <= 0:
            raise HTTPException(status_code=400, detail="Live price unavailable")

        # 3️⃣ Handle BUY (keep your current logic, only delete open row if you want)
        if (side or "").upper() == "BUY":
            cost = live_price * qty

            # Optional safety: ensure funds row exists
            _ensure_funds_row(c, username)

            c.execute(
                "UPDATE funds SET available_amount = available_amount - ? WHERE username=?",
                (cost, username),
            )

            # Your existing approach is fine for BUY conversion:
            c.execute("""
                UPDATE orders
                SET status='Closed',
                    price=?,
                    datetime=datetime('now','localtime')
                WHERE id=? AND status='Open'
            """, (live_price, order_id))

            conn.commit()
            return {
                "success": True,
                "message": "Order converted to MARKET and executed",
                "triggered": True,
                "executed_price": live_price
            }

        # 4️⃣ Handle SELL (✅ FIXED)
        else:
            # Ensure funds row exists
            _ensure_funds_row(c, username)

            # Credit funds for the sell execution
            credit = live_price * qty
            c.execute(
                "UPDATE funds SET available_amount = available_amount + ? WHERE username=?",
                (credit, username),
            )

            # ---- Holdings / portfolio deduction (safe + minimal) ----
            # Use same date-base as DB timestamps (localtime)
            today = _today_db(c)

            # today's net BUYs (Closed) for intraday closing capacity
            c.execute(
                """
                SELECT COALESCE(SUM(CASE WHEN order_type='BUY' THEN qty ELSE 0 END),0) -
                       COALESCE(SUM(CASE WHEN order_type='SELL' THEN qty ELSE 0 END),0)
                  FROM orders
                 WHERE username=? AND script=? AND status='Closed'
                   AND substr(datetime,1,10)=?
                """,
                (username, script, today),
            )
            today_net_buy = int(c.fetchone()[0] or 0)

            # portfolio holdings
            c.execute(
                "SELECT COALESCE(SUM(qty),0) FROM portfolio WHERE username=? AND script=?",
                (username, script),
            )
            portfolio_qty = int(c.fetchone()[0] or 0)

            is_delivery = (seg == "delivery")

            # Max closable qty from holdings:
            # - delivery: only portfolio
            # - intraday: today's net + portfolio
            owned_total = max(portfolio_qty, 0) if is_delivery else (max(today_net_buy, 0) + max(portfolio_qty, 0))

            # If this open SELL is NOT short-enabled, don't allow exceeding holdings
            if is_short == 0 and qty > owned_total:
                raise HTTPException(
                    status_code=400,
                    detail="Not enough holdings to SELL. Enable SELL FIRST to short-sell."
                )

            # Close as much as possible from holdings; remainder (if any) becomes short
            close_qty = min(qty, owned_total)

            # Deduct portfolio only for the portion that consumes portfolio holdings
            # (Intraday closes today's net first, then portfolio)
            if close_qty > 0:
                if is_delivery:
                    consume_portfolio = min(close_qty, max(portfolio_qty, 0))
                else:
                    consume_today = min(close_qty, max(today_net_buy, 0))
                    remaining_after_today = close_qty - consume_today
                    consume_portfolio = min(remaining_after_today, max(portfolio_qty, 0))

                if consume_portfolio > 0:
                    new_qty = portfolio_qty - consume_portfolio
                    if new_qty <= 0:
                        c.execute(
                            "DELETE FROM portfolio WHERE username=? AND script=?",
                            (username, script),
                        )
                    else:
                        c.execute(
                            """
                            UPDATE portfolio
                            SET qty=?, updated_at=datetime('now','localtime')
                            WHERE username=? AND script=?
                            """,
                            (new_qty, username, script),
                        )

            # ✅ IMPORTANT:
            # Delete the OPEN LIMIT row, then insert proper Closed rows using _execute_sell_fill()
            c.execute("DELETE FROM orders WHERE id=? AND status='Open'", (order_id,))

            fill = _execute_sell_fill(
                c=c,
                username=username,
                script=script,
                seg=seg,
                exec_price=float(live_price),
                sell_qty=int(qty),
                sell_sl=stoploss,
                sell_tgt=target,
            )

            conn.commit()
            return {
                "success": True,
                "message": "Order converted to MARKET and executed",
                "triggered": True,
                "executed_price": live_price,
                "closed_qty": int(fill.get("closed_qty", 0)),
                "short_qty": int(fill.get("short_qty", 0)),
            }


    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# -------------------- History (LONG sells + SHORT covers) --------------------
# -------------------- History (LONG sells + SHORT covers) --------------------

def _to_f(v, fallback: float = 0.0) -> float:
    try:
        n = float(v)
        return n if n == n else fallback  # NaN check
    except Exception:
        return fallback


def _calc_additional_cost(rates: Dict[str, Any], segment: str, investment: float) -> float:
    """
    Same calculation as Orders.jsx gray-detail:
      additional_cost = brokerage + tax (per-side)
    """
    seg = (segment or "delivery").lower()
    is_intra = seg == "intraday"

    mode = (rates.get("brokerage_mode") or "ABS").upper()

    if mode == "PCT":
        pct = _to_f(
            rates.get("brokerage_intraday_pct") if is_intra else rates.get("brokerage_delivery_pct"),
            0.0
        )
        brokerage = float(investment or 0.0) * pct
    else:
        brokerage = _to_f(
            rates.get("brokerage_intraday_abs") if is_intra else rates.get("brokerage_delivery_abs"),
            0.0
        )

    tax_pct = _to_f(rates.get("tax_intraday_pct") if is_intra else rates.get("tax_delivery_pct"), 0.0)
    tax = float(investment or 0.0) * tax_pct

    addl = brokerage + tax
    return float(addl) if addl == addl else 0.0


def _augment_history_costs(username: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds the same "greyout" calculations to History rows:
      - additional_cost (entry + exit)
      - net_investment (entry net)
      - exit_price
      - exit_net_investment

    Also includes:
      - entry_additional_cost, exit_additional_cost, net_pnl (safe for frontend to ignore)
    """
    try:
        db_rates = get_brokerage_settings(username) or {}
    except Exception:
        db_rates = {}

    defaults = BrokerageSettings().dict()
    rates = {**defaults, **(db_rates or {})}

    for t in rows or []:
        try:
            seg = (t.get("segment") or "delivery").lower()

            qty = float(t.get("buy_qty") or t.get("qty") or 0.0)
            entry_px = float(t.get("buy_price") or 0.0)

            # Prefer explicit exit_price if backend provides it; else use sell_avg_price.
            exit_px_raw = t.get("exit_price", None)
            if exit_px_raw is None or exit_px_raw == "":
                exit_px_raw = t.get("sell_avg_price", None)
            exit_px = float(exit_px_raw or 0.0)

            if qty <= 0 or entry_px <= 0 or exit_px <= 0:
                continue

            entry_inv = entry_px * qty
            exit_inv = exit_px * qty

            # Shorts from _get_user_cover_history have short_first=True and entry_side='SELL_FIRST'
            is_short = bool(t.get("short_first")) or (
                str(t.get("entry_side") or "").upper() in ("SELL_FIRST", "SHORT")
            )

            entry_add = _calc_additional_cost(rates, seg, entry_inv)
            exit_add = _calc_additional_cost(rates, seg, exit_inv)

            if not is_short:
                # LONG: BUY then SELL
                entry_net = entry_inv + entry_add          # BUY net
                exit_net = exit_inv - exit_add             # SELL net
                net_pnl = exit_net - entry_net
            else:
                # SHORT: SELL_FIRST then COVER (BUY)
                entry_net = entry_inv - entry_add          # SELL_FIRST net (credit)
                exit_net = exit_inv + exit_add             # COVER net (debit)
                net_pnl = entry_net - exit_net

            total_add = entry_add + exit_add

            t["entry_additional_cost"] = round(entry_add, 2)
            t["exit_additional_cost"] = round(exit_add, 2)
            t["additional_cost"] = round(total_add, 2)
            t["net_investment"] = round(entry_net, 2)
            t["exit_price"] = round(exit_px, 2)
            t["exit_net_investment"] = round(exit_net, 2)
            t["net_pnl"] = round(net_pnl, 2)
        except Exception:
            continue

    return rows
    
@router.get("/history/{username}")
def history_api(username: str):
    """
    History tab (used by the main History page):

    - LONG trades: BUY -> SELL (from `orders`, FIFO matching) via `get_user_history()`
    - SHORT trades: SELL -> BUY (cover) built here via `_get_user_cover_history()`
    """
    _run_eod_if_due(username)

    long_history = get_user_history(username) or []
    cover_history = _get_user_cover_history(username) or []

    combined = list(long_history) + list(cover_history)

    def _sort_key(x: Dict[str, Any]):
        return x.get("sell_date") or x.get("datetime") or ""

    combined.sort(key=_sort_key)

    # ✅ Add same Orders grey-detail calculations into History payload
    combined = _augment_history_costs(username, combined)

    return combined

@router.post("/cancel/{order_id}")
def cancel_open_order(order_id: int):
    """
    Cancel an OPEN order and refund blocked funds
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        _ensure_tables(c)

        # 1️⃣ Fetch open order
        c.execute("""
            SELECT username, script, order_type, qty, price, segment
            FROM orders
            WHERE id=? AND status='Open'
        """, (order_id,))
        row = c.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Open order not found")

        username, script, order_type, qty, price, segment = row
        qty = int(qty)
        price = float(price or 0)

        # 2️⃣ Refund BUY limit order funds
        if order_type == "BUY" and price > 0:
            refund = qty * price
            c.execute("""
                UPDATE funds
                SET available_amount = available_amount + ?
                WHERE username=?
            """, (refund, username))

        # 3️⃣ Delete order
        c.execute("DELETE FROM orders WHERE id=?", (order_id,))
        conn.commit()

        return {
            "success": True,
            "message": f"{script} order cancelled and funds refunded"
        }

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
