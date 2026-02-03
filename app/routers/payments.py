"""
UPI QR Payments (NO payment gateway)

✅ Important:
UPI QR has NO webhook, so we cannot verify payment automatically.
To prevent fake "Paid" without payment:
- We confirm ONLY when user provides UTR/reference (manual proof).
- No timer, no focus/mouse heuristics.

✅ Queued Plans:
- Free trial runs first (if present)
- Buying Monthly during Free queues Monthly after Free
- Buying Quarterly after Monthly queues Quarterly after Monthly
- Same for Half-yearly and Annual
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import qrcode
import io
import base64
import time
import sqlite3
import os
from pathlib import Path
import re
from urllib.parse import urlencode, quote
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

try:
    from dateutil.relativedelta import relativedelta
    HAS_RELATIVEDELTA = True
except Exception:
    HAS_RELATIVEDELTA = False

router = APIRouter(prefix="/payments", tags=["payments"])


# ------------------- DB Path (Local + Render Safe) -------------------
def _resolve_db_path() -> str:
    # 1) explicit env always wins
    p = (os.getenv("DB_PATH") or "").strip()
    if p:
        return p

    # 2) if Render disk mounted at /data and writable, use it
    if os.path.isdir("/data") and os.access("/data", os.W_OK):
        return "/data/paper_trading.db"

    # 3) fallback (works even without disk; NOT persistent across deploys)
    # payments.py is /.../src/app/routers/payments.py -> parents[2] is /.../src
    return str(Path(__file__).resolve().parents[2] / "data" / "paper_trading.db")


DB_PATH = _resolve_db_path()

PAYMENT_ADMIN_KEY = (os.getenv("PAYMENT_ADMIN_KEY") or "").strip()
AUTO_CONFIRM_UPI = (os.getenv("AUTO_CONFIRM_UPI") or "false").strip().lower() == "true"

# Anti-fake knobs (tune as you like)
MIN_SECONDS_AFTER_INIT = int(os.getenv("UPI_MIN_SECONDS_AFTER_INIT", "10").strip() or "10")
MIN_SECONDS_AFTER_OPEN = int(os.getenv("UPI_MIN_SECONDS_AFTER_OPEN", "5").strip() or "5")


def get_db():
    # Ensure parent folder exists (prevents "unable to open database file")
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    # timeout helps when concurrent requests happen
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def norm_user(user_id: str) -> str:
    return (user_id or "").strip().lower()


def normalize_existing_user_ids():
    """One-time soft migration to avoid mixed-case user_ids causing duplicate trials."""
    try:
        with get_db() as conn:
            conn.execute("UPDATE subscription_periods SET user_id = LOWER(TRIM(user_id)) WHERE user_id IS NOT NULL")
            conn.execute("UPDATE payments SET user_id = LOWER(TRIM(user_id)) WHERE user_id IS NOT NULL")
            conn.commit()
    except Exception:
        pass


# ------------------- Plan durations -------------------
FREE_TRIAL_DAYS = int(os.getenv("FREE_TRIAL_DAYS", "3").strip() or "3")
FREE_TRIAL_MINUTES = int(os.getenv("FREE_TRIAL_MINUTES", "5").strip() or "5")


# ------------------- One-time migration: shorten legacy free trials -------------------
def shrink_existing_free_trials_to_days():
    """
    Caps ANY 'free' period duration to FREE_TRIAL_DAYS (from its start_at).
    This keeps old DB data consistent when changing trial duration.
    """
    try:
        max_dur = int(FREE_TRIAL_DAYS) * 86400
        with get_db() as conn:
            rows = conn.execute("""
                SELECT id, start_at, end_at
                FROM subscription_periods
                WHERE lower(trim(plan_id)) = 'free'
                  AND end_at > start_at
            """).fetchall()
            for r in rows or []:
                sid = int(r["id"])
                start_at = float(r["start_at"])
                end_at = float(r["end_at"])
                if (end_at - start_at) > max_dur:
                    new_end = start_at + max_dur
                    conn.execute(
                        "UPDATE subscription_periods SET end_at = ? WHERE id = ?",
                        (new_end, sid),
                    )
            conn.commit()
    except Exception:
        pass


# fallback days if relativedelta not available
MONTHLY_DAYS = int(os.getenv("MONTHLY_DAYS", "30").strip() or "30")
QUARTERLY_DAYS = int(os.getenv("QUARTERLY_DAYS", "90").strip() or "90")
HALFYEARLY_DAYS = int(os.getenv("HALFYEARLY_DAYS", "180").strip() or "180")
ANNUAL_DAYS = int(os.getenv("ANNUAL_DAYS", "365").strip() or "365")

# calendar scheduling preferred (matches your 9/1 -> 9/2 logic)
PLAN_SPEC: Dict[str, Dict[str, Any]] = {
    "free": {"type": "days", "value": FREE_TRIAL_DAYS},
    "monthly": {"type": "months", "value": 1},
    "quarterly": {"type": "months", "value": 3},
    "halfyearly": {"type": "months", "value": 6},
    "annual": {"type": "years", "value": 1},
}

PLAN_FALLBACK_DAYS: Dict[str, int] = {
    "free": FREE_TRIAL_DAYS,
    "monthly": MONTHLY_DAYS,
    "quarterly": QUARTERLY_DAYS,
    "halfyearly": HALFYEARLY_DAYS,
    "annual": ANNUAL_DAYS,
}


# ------------------- Request Models -------------------
class UpiRequest(BaseModel):
    user_id: str
    pa: str
    pn: str
    amount_inr: float
    tr: str
    tn: str
    plan_id: str


class ConfirmRequest(BaseModel):
    tr: str
    utr: str  # ✅ REQUIRED (prevents fake confirmations)


class AdminApproveRequest(BaseModel):
    tr: str


class OpenedRequest(BaseModel):
    tr: str


# ------------------- Init / migrations -------------------
def init_table_payments():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            tr TEXT PRIMARY KEY,
            user_id TEXT,
            upi_id TEXT,
            payer_name TEXT,
            amount REAL,
            status TEXT,
            plan_id TEXT,
            utr TEXT,
            note TEXT,
            created_at REAL,
            updated_at REAL,
            opened_at REAL
        )
        """)
        conn.commit()


def init_table_subscription_periods():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS subscription_periods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            plan_id TEXT NOT NULL,
            start_at REAL NOT NULL,
            end_at REAL NOT NULL,
            source_tr TEXT UNIQUE,
            created_at REAL NOT NULL
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sub_periods_user ON subscription_periods(user_id)")
        conn.commit()


def ensure_column(table: str, col_name: str, col_type: str):
    try:
        with get_db() as conn:
            cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if col_name not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                conn.commit()
    except Exception:
        pass


# ✅ IMPORTANT: run this from FastAPI startup (NOT at import time)
def init_payments_storage():
    init_table_payments()
    init_table_subscription_periods()
    normalize_existing_user_ids()

    ensure_column("payments", "user_id", "TEXT")
    ensure_column("payments", "plan_id", "TEXT")
    ensure_column("payments", "utr", "TEXT")
    ensure_column("payments", "note", "TEXT")
    ensure_column("payments", "updated_at", "REAL")
    ensure_column("payments", "opened_at", "REAL")

    shrink_existing_free_trials_to_days()


# ------------------- Time helpers -------------------
def _now() -> float:
    return time.time()


def _ts_to_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _dt_to_ts(dt: datetime) -> float:
    return dt.timestamp()


def compute_end(start_ts: float, plan_id: str) -> float:
    plan_id = (plan_id or "").strip().lower()
    if plan_id not in PLAN_SPEC:
        raise HTTPException(status_code=400, detail=f"Invalid plan_id: {plan_id}")

    spec = PLAN_SPEC[plan_id]

    # ✅ Minute-based plans (Free trial)
    if spec.get("type") == "minutes":
        return start_ts + (int(spec.get("value") or 0) * 60)
    # ✅ Day-based plans (Free trial - 3 days)
    if spec.get("type") == "days":
        return start_ts + (int(spec.get("value") or 0) * 86400)

    # ✅ Calendar scheduling preferred (matches month/year boundaries)
    if HAS_RELATIVEDELTA and spec["type"] in ("months", "years"):
        start_dt = _ts_to_dt(start_ts)
        if spec["type"] == "months":
            end_dt = start_dt + relativedelta(months=int(spec["value"]))
        else:
            end_dt = start_dt + relativedelta(years=int(spec["value"]))
        return _dt_to_ts(end_dt)

    # ✅ Fallback day-based durations
    days = int(PLAN_FALLBACK_DAYS.get(plan_id, 0))
    return start_ts + (days * 86400)


# ------------------- Payments functions -------------------
def save_transaction(tr, user_id, pa, pn, amount, note, plan_id):
    uid = norm_user(user_id)
    now = _now()
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO payments
            (tr, user_id, upi_id, payer_name, amount, status, plan_id, utr, note, created_at, updated_at, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tr, uid, pa, pn, amount, "pending", plan_id, None, note, now, now, None))
        conn.commit()


def update_status(tr, status, utr=None):
    now = _now()
    with get_db() as conn:
        result = conn.execute(
            "UPDATE payments SET status = ?, utr = ?, updated_at = ? WHERE tr = ?",
            (status, utr, now, tr),
        )
        conn.commit()
        return result.rowcount > 0


def set_opened_at(tr: str):
    now = _now()
    with get_db() as conn:
        result = conn.execute(
            "UPDATE payments SET opened_at = ?, updated_at = ? WHERE tr = ?",
            (now, now, tr),
        )
        conn.commit()
        return result.rowcount > 0


def get_transaction(tr):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM payments WHERE tr = ?", (tr,)).fetchone()
        return dict(row) if row else None


# ------------------- Subscription queue functions -------------------
def get_last_end(user_id: str) -> Optional[float]:
    uid = norm_user(user_id)
    with get_db() as conn:
        row = conn.execute(
            "SELECT MAX(end_at) AS last_end FROM subscription_periods WHERE lower(trim(user_id)) = ?",
            (uid,),
        ).fetchone()
        v = row["last_end"] if row else None
        return float(v) if v is not None else None


def get_active_period(user_id: str, now_ts: float) -> Optional[dict]:
    uid = norm_user(user_id)
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM subscription_periods
            WHERE lower(trim(user_id)) = ?
              AND start_at <= ?
              AND end_at > ?
            ORDER BY start_at DESC
            LIMIT 1
        """, (uid, now_ts, now_ts)).fetchone()
        return dict(row) if row else None


def get_next_period(user_id: str, now_ts: float) -> Optional[dict]:
    uid = norm_user(user_id)
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM subscription_periods
            WHERE lower(trim(user_id)) = ?
              AND start_at > ?
            ORDER BY start_at ASC
            LIMIT 1
        """, (uid, now_ts)).fetchone()
        return dict(row) if row else None


def get_future_periods(user_id: str, now_ts: float) -> List[dict]:
    uid = norm_user(user_id)
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM subscription_periods
            WHERE lower(trim(user_id)) = ?
              AND start_at > ?
            ORDER BY start_at ASC
        """, (uid, now_ts)).fetchall()
        return [dict(r) for r in rows] if rows else []


def ensure_free_trial_if_missing(user_id: str):
    """
    ✅ Creates free trial ONLY if user has ZERO rows in subscription_periods.
    That means:
    - First ever login -> free trial created
    - After trial expires -> it will NOT be created again (user must pay)
    """
    uid = norm_user(user_id)

    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM subscription_periods WHERE lower(trim(user_id)) = ?",
            (uid,),
        ).fetchone()
        c = int(row["c"] or 0) if row else 0

    if c > 0:
        return

    start_at = _now()
    end_at = compute_end(start_at, "free")

    with get_db() as conn:
        conn.execute("""
            INSERT INTO subscription_periods (user_id, plan_id, start_at, end_at, source_tr, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (uid, "free", start_at, end_at, None, _now()))
        conn.commit()


def queue_plan(user_id: str, plan_id: str, source_tr: Optional[str] = None) -> dict:
    uid = norm_user(user_id)
    plan_id = (plan_id or "").strip().lower()

    if plan_id not in PLAN_SPEC:
        raise HTTPException(status_code=400, detail=f"Invalid plan_id: {plan_id}")

    now_ts = _now()
    last_end = get_last_end(uid)
    start_at = max(now_ts, last_end) if last_end else now_ts
    end_at = compute_end(start_at, plan_id)

    with get_db() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO subscription_periods
            (user_id, plan_id, start_at, end_at, source_tr, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (uid, plan_id, start_at, end_at, source_tr, _now()))
        conn.commit()

    return {"user_id": uid, "plan_id": plan_id, "started_at": start_at, "expires_at": end_at}


def maybe_queue_from_tx(tx: dict):
    if not tx:
        return
    if (tx.get("status") or "").lower() != "success":
        return
    user_id = (tx.get("user_id") or "").strip()
    plan_id = (tx.get("plan_id") or "").strip().lower()
    tr = (tx.get("tr") or "").strip()
    if not user_id or not plan_id or not tr:
        return
    queue_plan(user_id, plan_id, source_tr=tr)


# ------------------- Free trial status helpers -------------------
def get_latest_free_trial(uid: str) -> Optional[dict]:
    uid = norm_user(uid)
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM subscription_periods
            WHERE lower(trim(user_id)) = ?
              AND lower(trim(plan_id)) = 'free'
            ORDER BY start_at DESC
            LIMIT 1
        """, (uid,)).fetchone()
        return dict(row) if row else None


def compute_free_trial_status(user_id: str, now_ts: float) -> str:
    """
    Returns: "active" | "expired" | "unavailable"
    - active: free trial exists and start_at <= now < end_at
    - expired: free trial exists and now >= end_at
    - unavailable: no free trial row exists
    """
    p = get_latest_free_trial(user_id)
    if not p:
        return "unavailable"
    start_at = float(p.get("start_at") or 0.0)
    end_at = float(p.get("end_at") or 0.0)
    if start_at <= now_ts < end_at:
        return "active"
    return "expired"


# ------------------- UPI URI safety -------------------
_TR_SAFE = re.compile(r"^[A-Za-z0-9]+$")
# ✅ UPI txn id (mostly digits, length varies by bank/app)
_UPI_TXN_ID = re.compile(r"^\d{10,18}$")                 # UPI transaction id (usually digits)
_APP_TXN_ID = re.compile(r"^(?=.*\d)[A-Za-z0-9]{8,25}$") # app id (GPay/PhonePe) must contain at least 1 digit


def validate_tr(tr: str):
    tr = (tr or "").strip()
    if len(tr) < 6 or len(tr) > 20:
        raise HTTPException(status_code=400, detail="Invalid tr length. Use 6-20 alphanumeric chars.")
    if not _TR_SAFE.match(tr):
        raise HTTPException(status_code=400, detail="Invalid tr. Use only A-Z a-z 0-9.")
    return tr


def validate_utr(utr: str):
    u = (utr or "").strip()
    if not u:
        raise HTTPException(status_code=400, detail="UTR is required to confirm payment.")

    if _UPI_TXN_ID.match(u) or _APP_TXN_ID.match(u):
        return u

    raise HTTPException(
        status_code=400,
        detail="Invalid UTR. Enter UPI Transaction ID (10-18 digits) or App Transaction ID (8-25 alphanumeric with at least 1 number).",
    )


def build_upi_uri(pa: str, pn: str, am: float, tr: str, tn: str):
    tn = (tn or "").strip()
    if len(tn) > 40:
        tn = tn[:40]
    params = {"pa": pa, "pn": pn, "am": f"{am:.2f}", "cu": "INR", "tr": tr, "tn": tn}
    return "upi://pay?" + urlencode(params, quote_via=quote)


# ------------------- API: Generate UPI QR -------------------
@router.post("/upi/init")
async def start_upi_transaction(body: UpiRequest):
    if not body.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    if not body.pa or "@" not in body.pa:
        raise HTTPException(status_code=400, detail="Invalid UPI ID")

    plan_id = (body.plan_id or "").strip().lower()
    if plan_id not in PLAN_SPEC:
        raise HTTPException(status_code=400, detail="Invalid plan_id")

    tr = validate_tr(body.tr)

    # ✅ Free trial auto created on first ever user (no rows)
    ensure_free_trial_if_missing(body.user_id.strip())

    upi_uri = build_upi_uri(body.pa, body.pn, body.amount_inr, tr, body.tn)

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(upi_uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    save_transaction(tr, body.user_id.strip(), body.pa, body.pn, body.amount_inr, body.tn, plan_id)

    return {"upi_uri": upi_uri, "qr_b64": qr_b64, "tr": tr, "plan_id": plan_id}


# ------------------- API: Mark that user opened UPI app -------------------
@router.post("/upi/opened")
async def mark_upi_opened(req: OpenedRequest):
    tx = get_transaction(req.tr)
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    ok = set_opened_at(req.tr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to mark opened")

    return {"ok": True}


# ------------------- API: Confirm Payment (UTR required) -------------------
@router.post("/upi/confirm")
async def confirm_payment(req: ConfirmRequest):
    tx = get_transaction(req.tr)
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    now_ts = _now()
    created_at = float(tx.get("created_at") or 0.0)
    opened_at = float(tx.get("opened_at") or 0.0)

    # ✅ must click "Open in UPI App" first
    # ✅ If opened_at is missing (often happens due to CORS/network on hosted),
    # auto-set it when user submits UTR instead of blocking them forever.
    if not opened_at:
        try:
            set_opened_at(req.tr)
            tx = get_transaction(req.tr)
            opened_at = float((tx or {}).get("opened_at") or 0.0)
        except Exception:
            opened_at = 0.0

    # ✅ basic timing protections (reduces instant fake submits)
    if created_at and (now_ts - created_at) < MIN_SECONDS_AFTER_INIT:
        raise HTTPException(status_code=400, detail=f"Please wait {MIN_SECONDS_AFTER_INIT} seconds, then submit UTR.")
    if opened_at and (now_ts - opened_at) < MIN_SECONDS_AFTER_OPEN:
        raise HTTPException(status_code=400, detail=f"Please wait {MIN_SECONDS_AFTER_OPEN} seconds after opening UPI app, then submit UTR.")

    utr = validate_utr(req.utr)
    utr_norm = utr.strip().upper()

    # ✅ block reusing same UTR across ANY transactions (all statuses)
    with get_db() as conn:
        row = conn.execute(
            "SELECT tr FROM payments WHERE UPPER(TRIM(utr)) = ? AND tr != ? LIMIT 1",
            (utr_norm, req.tr),
        ).fetchone()
        if row:
            raise HTTPException(status_code=409, detail="This UTR is already used for another transaction.")

    # If already success, return success
    if (tx.get("status") or "").lower() == "success":
        maybe_queue_from_tx(tx)
        return {"ok": True, "status": "success"}

    new_status = "success" if AUTO_CONFIRM_UPI else "submitted"

    ok = update_status(req.tr, new_status, utr=utr_norm)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update transaction")

    tx2 = get_transaction(req.tr)

    # queue plan only when success
    if new_status == "success":
        maybe_queue_from_tx(tx2)

    return {"ok": True, "status": new_status}


@router.post("/upi/admin-approve")
async def admin_approve_payment(body: AdminApproveRequest, key: str = ""):
    # Optional safety/testing route
    if not PAYMENT_ADMIN_KEY or key != PAYMENT_ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    tx = get_transaction(body.tr)
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    if (tx.get("status") or "").lower() == "success":
        maybe_queue_from_tx(tx)
        return {"ok": True, "status": "success"}

    ok = update_status(body.tr, "success", utr=tx.get("utr"))
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to approve transaction")

    tx2 = get_transaction(body.tr)
    maybe_queue_from_tx(tx2)
    return {"ok": True, "status": "success"}


# ------------------- API: Check Status -------------------
@router.get("/upi/status/{tr}")
async def check_upi_status(tr: str):
    tx = get_transaction(tr)
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    maybe_queue_from_tx(tx)

    return {"status": tx.get("status"), "utr": tx.get("utr"), "amount": tx.get("amount")}


# ------------------- API: Subscription status (current + all queued) -------------------
@router.get("/subscription/{user_id}")
async def get_subscription(user_id: str):
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id required")

    user_id = norm_user(user_id)

    # ✅ Auto-create free trial only on first ever user
    ensure_free_trial_if_missing(user_id)

    now_ts = _now()

    active = get_active_period(user_id, now_ts)
    nxt = get_next_period(user_id, now_ts)
    future = get_future_periods(user_id, now_ts)

    free_trial_status = compute_free_trial_status(user_id, now_ts)

    if active:
        expires_at = float(active["end_at"])
        seconds_left = max(0, int(expires_at - now_ts))
        minutes_left = max(0, int(seconds_left // 60))
        days_left = max(0, int(seconds_left // 86400))

        plan_id_active = (active.get("plan_id") or "").strip().lower()
        if plan_id_active == "free":
            time_left_label = f"{days_left} days"
        else:
            time_left_label = f"{days_left} days"

        out: Dict[str, Any] = {
            "user_id": user_id,
            "plan_id": active["plan_id"],
            "started_at": active["start_at"],
            "expires_at": active["end_at"],
            "active": True,
            "days_left": days_left,
            "minutes_left": minutes_left,
            "seconds_left": seconds_left,
            "time_left_label": time_left_label,
        }
    else:
        out = {
            "user_id": user_id,
            "plan_id": None,
            "started_at": None,
            "expires_at": None,
            "active": False,
            "days_left": 0,
            "minutes_left": 0,
            "seconds_left": 0,
        }

    # ✅ NEW: lets frontend hide free card if expired/unavailable
    out["free_trial_status"] = free_trial_status

    if nxt:
        starts_in_days = max(0, int((float(nxt["start_at"]) - now_ts) // 86400))
        out["upcoming"] = {
            "plan_id": nxt["plan_id"],
            "starts_at": nxt["start_at"],
            "expires_at": nxt["end_at"],
            "starts_in_days": starts_in_days,
        }
    else:
        out["upcoming"] = None

    queued_list = []
    for p in future:
        starts_in_days = max(0, int((float(p["start_at"]) - now_ts) // 86400))
        queued_list.append({
            "plan_id": p["plan_id"],
            "starts_at": p["start_at"],
            "expires_at": p["end_at"],
            "starts_in_days": starts_in_days,
        })
    out["queued"] = queued_list

    return out


@router.get("/health")
def health():
    return {
        "ok": True,
        "message": "Payments service active",
        "db_path": DB_PATH,
        "calendar_durations": HAS_RELATIVEDELTA,
        "auto_confirm_upi": AUTO_CONFIRM_UPI,
        "min_seconds_after_init": MIN_SECONDS_AFTER_INIT,
        "min_seconds_after_open": MIN_SECONDS_AFTER_OPEN,
    }
