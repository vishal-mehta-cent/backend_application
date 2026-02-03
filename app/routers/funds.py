# backend/app/routers/funds.py

from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import sqlite3
import os
import csv
import re
import time as _time
from contextlib import contextmanager
from pathlib import Path

router = APIRouter(prefix="/funds", tags=["funds"])

IS_RENDER = bool(os.getenv("RENDER"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if IS_RENDER:
    DB_PATH = "/data/paper_trading.db"
else:
    DB_PATH = os.path.join(BASE_DIR, "app", "data", "paper_trading.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Use the same lock file path across ALL routers (funds/orders/auth/etc.)
# Put it on the same disk as the DB so it's shared by all processes.
LOCK_PATH = os.getenv("NC_DB_LOCK_PATH") or (
    "/data/paper_trading.db.lock" if IS_RENDER else os.path.join(os.path.dirname(DB_PATH), "paper_trading.db.lock")
)

# ---------- Models ----------

class FundsChange(BaseModel):
    username: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)

class FundUpdate(BaseModel):
    amount: float = Field(..., gt=0)

class ResetReq(BaseModel):
    username: str = Field(..., min_length=1)

class BrokerageSettings(BaseModel):
    brokerage_mode: str = "ABS"  # "ABS" or "PCT"
    brokerage_intraday_pct: float = 0.0
    brokerage_delivery_pct: float = 0.0
    brokerage_intraday_abs: float = 0.0
    brokerage_delivery_abs: float = 0.0
    tax_intraday_pct: float = 0.0
    tax_delivery_pct: float = 0.0


# ---------- Cross-process DB write lock ----------

@contextmanager
def _db_write_lock(timeout_s: float = 30.0):
    """
    Cross-process file lock to serialize SQLite writers.
    This is the MOST reliable way to stop 'database is locked' on Render with background tasks + API requests.

    Important: Use the SAME LOCK_PATH across your whole backend (funds/orders/auth/historical/etc).
    """
    lock_file = Path(LOCK_PATH)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    f = open(lock_file, "a+", encoding="utf-8")
    start = _time.time()
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt  # type: ignore
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl  # type: ignore
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except Exception:
                if (_time.time() - start) >= timeout_s:
                    raise
                _time.sleep(0.10)

        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt  # type: ignore
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # type: ignore
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            f.close()
        except Exception:
            pass


# ---------- DB helpers ----------

def _apply_pragmas(conn: sqlite3.Connection, *, is_writer: bool) -> None:
    """
    Reduce lock errors and improve concurrency.
    NOTE: SQLite still allows ONE writer at a time; the file lock above enforces that.
    """
    try:
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    try:
        # give SQLite time to wait instead of throwing immediately
        conn.execute("PRAGMA busy_timeout=60000;")  # ms
    except Exception:
        pass

    # WAL only needs to be set by a writer connection (read-only connections cannot set it).
    if is_writer:
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass


def _db_uri_path(p: str) -> str:
    # SQLite URI requires forward slashes
    try:
        return Path(p).resolve().as_posix()
    except Exception:
        return p.replace("\\", "/")


def _conn(*, read_only: bool) -> sqlite3.Connection:
    """
    - read_only=True: prevents accidental writes on GET routes; also avoids write-lock escalation.
    - read_only=False: normal writer connection.
    """
    if read_only:
        try:
            uri = f"file:{_db_uri_path(DB_PATH)}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=60, check_same_thread=False)
            _apply_pragmas(conn, is_writer=False)
            return conn
        except Exception:
            # fallback (db may not exist yet)
            pass

    conn = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    _apply_pragmas(conn, is_writer=(not read_only))
    return conn


def _is_locked_err(e: Exception) -> bool:
    msg = str(e).lower()
    return isinstance(e, sqlite3.OperationalError) and ("database is locked" in msg or "database is busy" in msg)


def _retry_write(fn, tries: int = 10, base_sleep: float = 0.15):
    """
    Retry wrapper for transient SQLite write collisions.
    We also hold a cross-process lock, so retries are usually only needed when another router
    is still inside a long transaction.
    """
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            if _is_locked_err(e):
                _time.sleep(base_sleep * (i + 1))
                continue
            raise
    raise last  # noqa


def _ensure_tables(c: sqlite3.Cursor) -> None:
    # funds table (source of truth for deposits/withdrawals; available_amount is a cache)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS funds (
          username TEXT PRIMARY KEY,
          available_amount REAL NOT NULL DEFAULT 0,
          total_amount REAL NOT NULL DEFAULT 0
        )
        """
    )

    # migrations (best effort)
    try:
        c.execute("PRAGMA table_info(funds)")
        cols = {r[1].lower() for r in c.fetchall()}
        if "available_amount" not in cols:
            c.execute("ALTER TABLE funds ADD COLUMN available_amount REAL NOT NULL DEFAULT 0")
        if "total_amount" not in cols:
            c.execute("ALTER TABLE funds ADD COLUMN total_amount REAL NOT NULL DEFAULT 0")
    except Exception:
        pass

    # brokerage settings table
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS brokerage_settings (
          username TEXT PRIMARY KEY,
          brokerage_mode TEXT DEFAULT 'ABS',
          brokerage_intraday_pct REAL DEFAULT 0,
          brokerage_intraday_abs REAL DEFAULT 0,
          brokerage_delivery_pct REAL DEFAULT 0,
          brokerage_delivery_abs REAL DEFAULT 0,
          tax_intraday_pct REAL DEFAULT 0,
          tax_delivery_pct REAL DEFAULT 0,
          updated_at TEXT
        )
        """
    )


def init_funds_schema() -> None:
    """
    Called from main.py on startup.
    Safe to call multiple times.
    """
    with _db_write_lock(timeout_s=30):
        conn = _conn(read_only=False)
        try:
            c = conn.cursor()
            _ensure_tables(c)
            conn.commit()
        finally:
            conn.close()


# ---------- cost helpers ----------

def _to_f(v, fallback: float = 0.0) -> float:
    try:
        n = float(v)
        return n if (n == n and n not in (float("inf"), float("-inf"))) else fallback
    except Exception:
        return fallback


def _calc_additional_cost(rates: Dict[str, Any], segment: str, investment: float) -> float:
    seg = (segment or "delivery").lower()
    is_intra = (seg == "intraday")

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


def _get_brokerage_settings_readonly(conn: sqlite3.Connection, username: str) -> Dict[str, Any]:
    """
    READ ONLY: do not insert defaults here to avoid locks on GET.
    """
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM brokerage_settings WHERE username=?", (username,))
        row = c.fetchone()
        if not row:
            return BrokerageSettings().dict()
        d = dict(row)
        d.pop("updated_at", None)
        return d
    except Exception:
        return BrokerageSettings().dict()


def _upsert_brokerage_settings(username: str, data: BrokerageSettings) -> None:
    def _do():
        with _db_write_lock(timeout_s=30):
            conn = _conn(read_only=False)
            try:
                c = conn.cursor()
                _ensure_tables(c)
                c.execute(
                    """
                    INSERT INTO brokerage_settings
                      (username, brokerage_mode,
                       brokerage_intraday_pct, brokerage_intraday_abs,
                       brokerage_delivery_pct, brokerage_delivery_abs,
                       tax_intraday_pct, tax_delivery_pct, updated_at)
                    VALUES
                      (?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
                    ON CONFLICT(username) DO UPDATE SET
                      brokerage_mode         = excluded.brokerage_mode,
                      brokerage_intraday_pct = excluded.brokerage_intraday_pct,
                      brokerage_intraday_abs = excluded.brokerage_intraday_abs,
                      brokerage_delivery_pct = excluded.brokerage_delivery_pct,
                      brokerage_delivery_abs = excluded.brokerage_delivery_abs,
                      tax_intraday_pct       = excluded.tax_intraday_pct,
                      tax_delivery_pct       = excluded.tax_delivery_pct,
                      updated_at             = excluded.updated_at
                    """,
                    (
                        username,
                        data.brokerage_mode,
                        float(data.brokerage_intraday_pct),
                        float(data.brokerage_intraday_abs),
                        float(data.brokerage_delivery_pct),
                        float(data.brokerage_delivery_abs),
                        float(data.tax_intraday_pct),
                        float(data.tax_delivery_pct),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    _retry_write(_do)


# ---------- FNO detection helpers ----------

_INSTRUMENT_CACHE: Optional[Dict[str, str]] = None


def _load_instruments_map() -> Dict[str, str]:
    """
    Return map UPPER(tradingsymbol) -> instrument_type from instruments.csv (if present).
    Expected location: backend/app/instruments.csv
    """
    global _INSTRUMENT_CACHE
    if _INSTRUMENT_CACHE is not None:
        return _INSTRUMENT_CACHE

    try:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../app
        instruments_path = os.path.join(app_dir, "instruments.csv")

        mp: Dict[str, str] = {}
        if os.path.exists(instruments_path):
            with open(instruments_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = (row.get("tradingsymbol") or "").strip().upper()
                    it = (row.get("instrument_type") or "").strip().upper()
                    if ts:
                        mp[ts] = it

        _INSTRUMENT_CACHE = mp
        return mp
    except Exception:
        _INSTRUMENT_CACHE = {}
        return _INSTRUMENT_CACHE


def _is_fno_symbol(script: str) -> bool:
    s = (script or "").strip().upper()
    if not s:
        return False

    mp = _load_instruments_map()
    it = mp.get(s)
    if it and it != "EQ":
        return True

    # fallback heuristics
    if re.search(r"(\bCE\b|\bPE\b|\bFUT\b)", s):
        return True
    if re.search(r"(NIFTY|BANKNIFTY|FINNIFTY|SENSEX)", s) and re.search(r"(CE|PE|FUT)", s):
        return True
    return False


def _short_block_multiplier(script: str) -> float:
    # ✅ Your rule: FNO short blocks 3x; Equity short blocks 1x
    return 3.0 if _is_fno_symbol(script) else 1.0


# ---------- CORE: snapshot recompute ----------

def compute_funds_snapshot(
    username: str,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """
    Render-safe (READ ONLY): recomputes funds from DB as the source of truth.

    Includes additional costs on executed legs.

    Available funds = cash usable now, after:
      - executed trades cashflows (Closed/SETTLED)
      - additional_cost on each executed leg
      - SELL_FIRST margin blocks (EQ=1x, FNO=3x) for open executed shorts
      - OPEN order blocks (BUY blocks buy_value; SELL_FIRST blocks margin)
    """
    owns_conn = False
    if conn is None:
        conn = _conn(read_only=True)
        owns_conn = True

    try:
        c = conn.cursor()

        # base_total is what user has deposited minus withdrawals/spends (total_amount).
        base_total = 0.0
        try:
            c.execute("SELECT total_amount FROM funds WHERE username=?", (username,))
            row = c.fetchone()
            base_total = float((row[0] if row else 0.0) or 0.0)
        except Exception:
            base_total = 0.0

        rates = _get_brokerage_settings_readonly(conn, username)

        # detect orders schema columns defensively
        try:
            c.execute("PRAGMA table_info(orders)")
            cols = {r[1] for r in c.fetchall()}
        except Exception:
            cols = set()

        has_status = "status" in cols
        has_is_short = "is_short" in cols
        has_ptype = "position_type" in cols

        needed = {"id", "datetime", "script", "order_type", "qty", "price", "segment", "username"}
        if not needed.issubset(cols):
            return {
                "total_funds": round(float(base_total), 2),
                "available_funds": round(float(base_total), 2),
                "invested_long": 0.0,
                "blocked_short": 0.0,
                "open_blocked": 0.0,
                "total_cost_paid": 0.0,
            }

        select = "id, datetime, script, order_type, qty, price, segment"
        select += ", " + ("COALESCE(is_short, 0) AS is_short" if has_is_short else "0 AS is_short")
        select += ", " + ("COALESCE(position_type, '') AS position_type" if has_ptype else "'' AS position_type")
        select += ", " + ("COALESCE(status, '') AS status" if has_status else "'' AS status")

        # executed rows (cashflows + cost)
        q_exec = f"""
            SELECT {select}
            FROM orders
            WHERE username=?
              AND COALESCE(status,'') IN ('Closed','SETTLED')
            ORDER BY datetime ASC, id ASC
        """
        c.execute(q_exec, (username,))
        exec_rows = c.fetchall() or []

        # open orders (reserve)
        q_open = f"""
            SELECT {select}
            FROM orders
            WHERE username=?
              AND COALESCE(status,'')='Open'
            ORDER BY datetime ASC, id ASC
        """
        c.execute(q_open, (username,))
        open_rows = c.fetchall() or []

        cash = float(base_total)
        total_cost_paid = 0.0

        # FIFO books for remaining open lots
        long_lots: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
        short_lots: Dict[Tuple[str, str], List[Dict[str, float]]] = {}

        def _push_lot(book, key, qty, price):
            if qty <= 0:
                return
            book.setdefault(key, []).append({"qty": float(qty), "price": float(price)})

        def _consume_lot(book, key, qty_needed) -> List[Tuple[float, float]]:
            out: List[Tuple[float, float]] = []
            if qty_needed <= 0:
                return out
            lots = book.get(key) or []
            q = float(qty_needed)
            i = 0
            while q > 0 and i < len(lots):
                lot = lots[i]
                take = min(lot["qty"], q)
                if take > 0:
                    out.append((take, lot["price"]))
                    lot["qty"] -= take
                    q -= take
                if lot["qty"] <= 1e-9:
                    i += 1
                else:
                    break
            book[key] = [l for l in lots[i:] if l["qty"] > 1e-9]
            return out

        # 1) Apply executed cashflows (+ additional costs)
        for r in exec_rows:
            _id, dt, script, order_type, qty, price, segment, is_short, position_type, status = r
            script_u = (script or "").strip().upper()
            seg = (segment or "delivery").lower()
            key = (script_u, seg)

            try:
                q_qty = float(qty or 0)
                p_price = float(price or 0)
            except Exception:
                continue
            if q_qty <= 0 or p_price <= 0:
                continue

            trade_value = q_qty * p_price
            cost = float(_calc_additional_cost(rates, seg, trade_value) or 0.0)
            total_cost_paid += cost

            ot = (order_type or "").strip().upper()
            pt = (position_type or "").strip().upper()

            if ot == "BUY":
                # BUY to cover short
                if pt == "COVER":
                    matched = _consume_lot(short_lots, key, q_qty)
                    entry_value = sum(take * lot_price for take, lot_price in matched)
                    pnl = sum((lot_price - p_price) * take for take, lot_price in matched)  # entry - exit

                    mult = _short_block_multiplier(script_u)
                    # release blocked margin + realize P&L - cost
                    cash += mult * entry_value + pnl - cost
                else:
                    # normal BUY (long)
                    cash -= trade_value + cost
                    _push_lot(long_lots, key, q_qty, p_price)

            elif ot == "SELL":
                # short entry (SELL_FIRST)
                if int(is_short or 0) == 1 or pt in ("SELL_FIRST", "SHORT"):
                    mult = _short_block_multiplier(script_u)
                    cash -= mult * trade_value + cost
                    _push_lot(short_lots, key, q_qty, p_price)
                else:
                    # normal SELL (exit long)
                    cash += trade_value - cost
                    _consume_lot(long_lots, key, q_qty)

        # 2) Apply OPEN order blocks (reserve cash, no costs yet)
        open_blocked = 0.0
        for r in open_rows:
            _id, dt, script, order_type, qty, price, segment, is_short, position_type, status = r
            script_u = (script or "").strip().upper()

            try:
                q_qty = float(qty or 0)
                p_price = float(price or 0)
            except Exception:
                continue
            if q_qty <= 0 or p_price <= 0:
                continue

            ot = (order_type or "").strip().upper()
            pt = (position_type or "").strip().upper()
            trade_value = q_qty * p_price

            if ot == "BUY":
                open_blocked += trade_value
            elif ot == "SELL":
                if int(is_short or 0) == 1 or pt in ("SELL_FIRST", "SHORT"):
                    mult = _short_block_multiplier(script_u)
                    open_blocked += mult * trade_value

        cash -= float(open_blocked)

        invested_long = 0.0
        for _, lots in long_lots.items():
            for lot in lots:
                invested_long += float(lot["qty"] * lot["price"])

        blocked_short = 0.0
        for (s, _seg), lots in short_lots.items():
            mult = _short_block_multiplier(s)
            for lot in lots:
                blocked_short += float(mult * lot["qty"] * lot["price"])

        total_equity = cash + invested_long + blocked_short + open_blocked

        return {
            "total_funds": round(float(total_equity), 2),
            "available_funds": round(float(cash), 2),
            "invested_long": round(float(invested_long), 2),
            "blocked_short": round(float(blocked_short), 2),
            "open_blocked": round(float(open_blocked), 2),
            "total_cost_paid": round(float(total_cost_paid), 2),
        }

    finally:
        if owns_conn:
            conn.close()


def _sync_available_amount_cache(username: str) -> None:
    """
    OPTIONAL cache sync: updates funds.available_amount to current computed available_funds.
    This runs in writer paths only (add/spend/reset) to avoid locks on GET.
    """
    def _do():
        with _db_write_lock(timeout_s=30):
            conn = _conn(read_only=False)
            try:
                c = conn.cursor()
                _ensure_tables(c)

                # ensure user exists
                c.execute(
                    "INSERT INTO funds (username, total_amount, available_amount) VALUES (?, 0.0, 0.0) "
                    "ON CONFLICT(username) DO NOTHING",
                    (username,),
                )

                snap = compute_funds_snapshot(username=username, conn=None)
                c.execute("UPDATE funds SET available_amount=? WHERE username=?", (float(snap["available_funds"]), username))
                conn.commit()
            finally:
                conn.close()

    _retry_write(_do)


# ---------- Read endpoints ----------

@router.get("/available/{username}")
def get_available(username: str):
    """
    Render-safe: READ ONLY.
    """
    snap = compute_funds_snapshot(username=username, conn=None)
    return {
        "total_funds": float(snap.get("total_funds", 0.0)),
        "available_funds": float(snap.get("available_funds", 0.0)),
        "invested_long": float(snap.get("invested_long", 0.0)),
        "blocked_short": float(snap.get("blocked_short", 0.0)),
        "open_blocked": float(snap.get("open_blocked", 0.0)),
        "total_cost_paid": float(snap.get("total_cost_paid", 0.0)),
    }


@router.get("/brokerage/{username}")
def get_brokerage(username: str):
    conn = _conn(read_only=True)
    try:
        return _get_brokerage_settings_readonly(conn, username)
    finally:
        conn.close()


@router.post("/brokerage/{username}")
def set_brokerage(username: str, data: BrokerageSettings):
    _upsert_brokerage_settings(username, data)
    _sync_available_amount_cache(username)
    return {"success": True, "message": "Brokerage settings saved"}


# ---------- Write endpoints ----------

@router.post("/add")
def add_funds(body: FundsChange):
    """
    Deposit funds:
      total_amount += amount
      available_amount cache is recomputed safely (writer path).
    """
    def _do():
        with _db_write_lock(timeout_s=30):
            conn = _conn(read_only=False)
            try:
                c = conn.cursor()
                _ensure_tables(c)
                c.execute(
                    """
                    INSERT INTO funds (username, available_amount, total_amount)
                    VALUES (?, ?, ?)
                    ON CONFLICT(username) DO UPDATE SET
                      available_amount = available_amount + excluded.available_amount,
                      total_amount     = total_amount + excluded.total_amount
                    """,
                    (body.username, float(body.amount), float(body.amount)),
                )
                conn.commit()
            finally:
                conn.close()

        _sync_available_amount_cache(body.username)
        return {"success": True, "message": "Funds added"}

    try:
        return _retry_write(_do)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")


@router.post("/spend")
@router.post("/deduct")
def spend_funds(body: FundsChange):
    """
    Spend/withdraw funds:
      total_amount -= amount  (clamped at 0)
      available_amount cache is recomputed safely.

    Restores your "spend" functionality.
    """
    def _do():
        with _db_write_lock(timeout_s=30):
            conn = _conn(read_only=False)
            try:
                c = conn.cursor()
                _ensure_tables(c)

                c.execute(
                    "INSERT INTO funds (username, total_amount, available_amount) VALUES (?, 0.0, 0.0) "
                    "ON CONFLICT(username) DO NOTHING",
                    (body.username,),
                )

                c.execute("SELECT total_amount FROM funds WHERE username=?", (body.username,))
                row = c.fetchone()
                cur_total = float((row[0] if row else 0.0) or 0.0)
                new_total = max(0.0, cur_total - float(body.amount))

                c.execute("UPDATE funds SET total_amount=? WHERE username=?", (new_total, body.username))
                conn.commit()
            finally:
                conn.close()

        _sync_available_amount_cache(body.username)
        return {"success": True, "message": "Spend added"}

    try:
        return _retry_write(_do)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")


@router.post("/spend/{username}")
@router.post("/deduct/{username}")
def spend_funds_legacy(username: str, data: FundUpdate):
    body = FundsChange(username=username, amount=float(data.amount))
    return spend_funds(body)


@router.post("/reset/{username}")
def reset_funds_endpoint(username: str):
    """
    Endpoint used by your Reset button.
    """
    reset_user_funds(username)
    return {"success": True, "message": "Funds reset"}


@router.post("/reset")
def reset_funds_body(body: ResetReq):
    reset_user_funds(body.username)
    return {"success": True, "message": "Funds reset"}


# Back-compat: GET /funds/{username}
@router.get("/{username}")
def get_funds_legacy(username: str):
    snap = compute_funds_snapshot(username=username, conn=None)
    return {
        "total_funds": float(snap.get("total_funds", 0.0)),
        "available_funds": float(snap.get("available_funds", 0.0)),
    }


# Back-compat: POST /funds/{username} acts like add funds
@router.post("/{username}")
def add_funds_legacy(username: str, data: FundUpdate):
    body = FundsChange(username=username, amount=float(data.amount))
    return add_funds(body)


def reset_user_funds(username: str) -> None:
    """
    Reset deposits to 0 and cache to 0.
    """
    def _do():
        with _db_write_lock(timeout_s=30):
            conn = _conn(read_only=False)
            try:
                c = conn.cursor()
                _ensure_tables(c)
                c.execute(
                    """
                    INSERT INTO funds (username, total_amount, available_amount)
                    VALUES (?, 0.0, 0.0)
                    ON CONFLICT(username) DO UPDATE SET
                      total_amount = 0.0,
                      available_amount = 0.0
                    """,
                    (username,),
                )
                conn.commit()
            finally:
                conn.close()

    _retry_write(_do)
