# backend/app/routers/auth.py

from fastapi import APIRouter
from pydantic import BaseModel
import sqlite3  # ✅ Added import so ensure_allowed_users_exist() works
import uuid
from datetime import datetime, timedelta
import random
import time
import smtplib
import os
from email.message import EmailMessage
from typing import Optional


router = APIRouter(prefix="/auth", tags=["auth"])

# 🔒 Lockdown config
ALLOWED_USERS = {
    "Neurocrest_dev",
    "Neurocrest_test",
    "Neurocrest_masum",
    "Neurocrest_vishal",
    "Neurocrest_jinal",
    "Neurocrest_others",
    "Neurocrest_test_1",
}
FIXED_PASSWORD = "neurocrest123"
CONTACT_MSG = "Please contact on the WhatsApp number: 9426001601"

# ✅ Users allowed to login from multiple devices
MULTI_LOGIN_USERS = {
    "Neurocrest_masum",
    "Neurocrest_test",
    "Neurocrest_jinal",
    "Neurocrest_test_1",
}



IS_RENDER = bool(os.getenv("RENDER"))

def _resolve_db_path():
    if IS_RENDER:
        os.makedirs("/data", exist_ok=True)
        return "/data/paper_trading.db"

    # Local: backend/app/data/paper_trading.db
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../backend/app
    DATA_DIR = os.path.join(APP_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, "paper_trading.db")

DB_PATH = _resolve_db_path()


# ---------------- OTP store (in-memory) ----------------
OTP_STORE = {}  # email -> {"otp": "1234", "expiry": epoch_seconds}


def _clean_env(v):
    """Trim spaces and remove internal spaces (helps with Gmail app password pasted as 'xxxx xxxx xxxx xxxx')."""
    if not v:
        return ""
    return str(v).strip().replace(" ", "")


def _send_otp_email(to_email: str, otp: str):
    smtp_email = _clean_env(os.getenv("SMTP_EMAIL"))
    smtp_password = _clean_env(os.getenv("SMTP_PASSWORD"))

    if not smtp_email or not smtp_password:
        raise RuntimeError("SMTP_EMAIL / SMTP_PASSWORD not set in environment")

    msg = EmailMessage()
    msg["Subject"] = "NeuroCrest Email Verification OTP"
    msg["From"] = smtp_email
    msg["To"] = to_email
    msg.set_content(
        f"""
Hello,

Your NeuroCrest verification OTP is:

{otp}

This OTP is valid for 5 minutes.

— NeuroCrest Team
""".strip()
    )

    # ✅ Timeout helps avoid hanging
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
        smtp.login(smtp_email, smtp_password)
        smtp.send_message(msg)


def send_email(to_email: str, subject: str, body: str):
    smtp_email = _clean_env(os.getenv("SMTP_EMAIL"))
    smtp_password = _clean_env(os.getenv("SMTP_PASSWORD"))

    if not smtp_email or not smtp_password:
        raise RuntimeError("SMTP_EMAIL / SMTP_PASSWORD not set in environment")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_email
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
        smtp.login(smtp_email, smtp_password)
        smtp.send_message(msg)



def ensure_session_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            session_id TEXT NOT NULL,
            login_time TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _column_exists(conn, table_name: str, col_name: str) -> bool:
    try:
        c = conn.cursor()
        c.execute(f"PRAGMA table_info({table_name})")
        cols = [r[1] for r in c.fetchall()]  # (cid, name, type, notnull, dflt, pk)
        return col_name in cols
    except Exception:
        return False


def _ensure_users_extra_columns(conn):
    """
    Safe migration: add columns if DB already exists with old schema.
    """
    c = conn.cursor()

    # ✅ NEW: these two are required for OTP + duplicate check
    if not _column_exists(conn, "users", "email"):
        c.execute("ALTER TABLE users ADD COLUMN email TEXT DEFAULT ''")
    if not _column_exists(conn, "users", "phone"):
        c.execute("ALTER TABLE users ADD COLUMN phone TEXT DEFAULT ''")

    # Existing extra fields
    if not _column_exists(conn, "users", "first_name"):
        c.execute("ALTER TABLE users ADD COLUMN first_name TEXT DEFAULT ''")
    if not _column_exists(conn, "users", "last_name"):
        c.execute("ALTER TABLE users ADD COLUMN last_name TEXT DEFAULT ''")
    if not _column_exists(conn, "users", "city"):
        c.execute("ALTER TABLE users ADD COLUMN city TEXT DEFAULT ''")

    conn.commit()




# ✅ Make sure DB and allowed users exist
def ensure_allowed_users_exist():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT DEFAULT '',
            phone TEXT DEFAULT '',
            first_name TEXT DEFAULT '',
            last_name TEXT DEFAULT '',
            city TEXT DEFAULT '',
            funds REAL DEFAULT 0
        )
        """
    )

    # If DB existed with old schema, add new columns safely
    _ensure_users_extra_columns(conn)

    # ✅ NEW: make email & phone unique (ignore empty values so allowlisted '' won't break)
    try:
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(email) WHERE email != ''")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_phone_unique ON users(phone) WHERE phone != ''")
        conn.commit()
    except Exception:
        # If SQLite doesn't support partial indexes in some environment, we still block duplicates via SELECT checks.
        pass

    for user in ALLOWED_USERS:
        c.execute(
            "INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
            (user, FIXED_PASSWORD),
        )
    conn.commit()
    conn.close()


# ✅ Run once on import
ensure_allowed_users_exist()
ensure_session_table()

def ensure_forgot_otp_table(conn: sqlite3.Connection):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS forgot_password_otps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            email TEXT,
            phone TEXT,
            otp TEXT,
            created_at TEXT
        )
    """)

class ForgotRequest(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None

class ForgotVerify(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    otp: str
# ---------------- Pydantic models ----------------
class UserIn(BaseModel):
    username: str
    password: str


class UpdatePassword(BaseModel):
    username: str
    new_password: str
class ChangePassword(BaseModel):
    username: str
    current_password: str
    new_password: str


class UpdateEmail(BaseModel):
    username: str
    new_email: str


class GoogleToken(BaseModel):
    token: str


class SignupSendOTP(BaseModel):
    username: str
    password: str
    first_name: str
    last_name: str
    city: str
    email: str
    phone: str


class SignupVerifyOTP(SignupSendOTP):
    otp: str


# ---------------- Login route ----------------
@router.post("/login")
def login(user: UserIn):
    username = (user.username or "").strip()
    password = user.password or ""

    if not username or not password:
        return {"success": False, "message": "Username and password are required"}

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ✅ Allowlist users: fixed password + optional multi-login behavior (your existing logic)
    if username in ALLOWED_USERS:
        c.execute("SELECT password FROM users WHERE username=? LIMIT 1", (username,))
        r = c.fetchone()
        db_password = (r[0] if r else "") or FIXED_PASSWORD

        if password != db_password:
            conn.close()
            return {"success": False, "message": "Invalid credentials"}

        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # 🔐 SINGLE LOGIN USERS → delete old sessions
        if username not in MULTI_LOGIN_USERS:
            c.execute("DELETE FROM user_sessions WHERE username=?", (username,))

        # ✅ Insert new session
        c.execute(
            """
            INSERT INTO user_sessions (username, session_id, login_time)
            VALUES (?, ?, ?)
            """,
            (username, session_id, now),
        )
        conn.commit()
        conn.close()

        return {"success": True, "username": username, "session_id": session_id}

    # ✅ NEW SIGNUP USERS: check DB username/password
    c.execute(
        "SELECT password, email, phone FROM users WHERE username=? LIMIT 1",
        (username,),
    )
    row = c.fetchone()

    if not row:
        conn.close()
        return {"success": False, "message": "Invalid credentials"}

    db_password = row[0] or ""
    if password != db_password:
        conn.close()
        return {"success": False, "message": "Invalid credentials"}

    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    # Default: single-login for normal users (delete old sessions)
    c.execute("DELETE FROM user_sessions WHERE username=?", (username,))

    c.execute(
        """
        INSERT INTO user_sessions (username, session_id, login_time)
        VALUES (?, ?, ?)
        """,
        (username, session_id, now),
    )
    conn.commit()
    conn.close()

    return {
        "success": True,
        "username": username,
        "session_id": session_id,
        "email": row[1] or "",
        "phone": row[2] or "",
    }



# ---------------- Register route (kept disabled) ----------------
@router.post("/register")
def register(user: UserIn):
    print("🚫 Register blocked for:", user.username)
    return {"success": False, "message": CONTACT_MSG}
@router.post("/change-password")
def change_password(data: ChangePassword):
    """
    Frontend calls: POST /auth/change-password
    Body: { username, current_password, new_password }
    """
    username = (data.username or "").strip()
    current_password = data.current_password or ""
    new_password = data.new_password or ""

    if not username:
        return {"success": False, "message": "Username is required"}
    if not current_password or not new_password:
        return {"success": False, "message": "Current and new password are required"}
    if len(new_password) < 6:
        return {"success": False, "message": "New password must be at least 6 characters"}

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT password FROM users WHERE username=? LIMIT 1", (username,))
        row = c.fetchone()
        if not row:
            return {"success": False, "message": "User not found"}

        db_password = (row[0] or "")
        if username in ALLOWED_USERS and not db_password:
            db_password = FIXED_PASSWORD

        if current_password != db_password:
            return {"success": False, "message": "Current password is incorrect"}

        c.execute("UPDATE users SET password=? WHERE username=?", (new_password, username))
        conn.commit()
        return {"success": True, "message": "Password updated"}
    finally:
        conn.close()


# ---------------- Update password disabled (kept) ----------------
@router.post("/update-password")
def update_password(data: UpdatePassword):
    print("🚫 Update password blocked for:", data.username)
    return {
        "success": False,
        "message": f"Password is fixed and cannot be changed. {CONTACT_MSG}",
    }


# ---------------- Update email disabled (kept) ----------------
@router.post("/update-email")
def update_email(data: UpdateEmail):
    print("🚫 Update email blocked for:", data.username)
    return {
        "success": False,
        "message": f"Username/Email changes are disabled. {CONTACT_MSG}",
    }


# ---------------- Google Login disabled (kept) ----------------
@router.post("/google-login")
def google_login(data: GoogleToken):
    print("🚫 Google login blocked")
    return {"success": False, "message": CONTACT_MSG}


@router.get("/validate-session")
def validate_session(username: str, session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT 1 FROM user_sessions
        WHERE username=? AND session_id=?
        LIMIT 1
        """,
        (username, session_id),
    )
    row = c.fetchone()
    conn.close()
    return {"valid": bool(row)}


# ---------------- Signup OTP: send otp ----------------
@router.post("/signup/send-otp")
def signup_send_otp(data: SignupSendOTP):
    username = (data.username or "").strip()
    password = data.password or ""
    first_name = (data.first_name or "").strip()
    last_name = (data.last_name or "").strip()
    city = (data.city or "").strip()
    email = (str(data.email) or "").strip()
    phone = (data.phone or "").strip()

    if not username:
        return {"success": False, "message": "Username is required"}
    if not password:
        return {"success": False, "message": "Password is required"}

    # Mandatory details as per your requirement
    if not first_name:
        return {"success": False, "message": "First name is required"}
    if not last_name:
        return {"success": False, "message": "Last name is required"}
    if not city:
        return {"success": False, "message": "City is required"}
    if not email:
        return {"success": False, "message": "Email is required"}
    if not phone:
        return {"success": False, "message": "Mobile number is required"}

    # Prevent signup for allowlisted reserved usernames
    if username in ALLOWED_USERS:
        return {"success": False, "message": "This username is reserved. Please choose another."}

    # ✅ NEW: Block duplicates for username/email/phone
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT 1 FROM users WHERE username=? LIMIT 1", (username,))
    exists = c.fetchone()
    if exists:
        conn.close()
        return {"success": False, "message": "Username already exists"}

    c.execute("SELECT 1 FROM users WHERE email=? AND email!='' LIMIT 1", (email,))
    if c.fetchone():
        conn.close()
        return {"success": False, "message": "Email ID already exists. Please use another email."}

    c.execute("SELECT 1 FROM users WHERE phone=? AND phone!='' LIMIT 1", (phone,))
    if c.fetchone():
        conn.close()
        return {"success": False, "message": "Mobile number already exists. Please use another number."}

    conn.close()

    otp = str(random.randint(1000, 9999))
    OTP_STORE[email] = {"otp": otp, "expiry": time.time() + 300}

    try:
        _send_otp_email(email, otp)
    except Exception as e:
        return {"success": False, "message": f"Failed to send OTP: {e}"}

    return {"success": True, "message": "OTP sent to email"}


# ---------------- Signup OTP: verify + create user ----------------
@router.post("/signup/verify-otp")
def signup_verify_otp(data: SignupVerifyOTP):
    username = (data.username or "").strip()
    password = data.password or ""
    first_name = (data.first_name or "").strip()
    last_name = (data.last_name or "").strip()
    city = (data.city or "").strip()
    email = (str(data.email) or "").strip()
    phone = (data.phone or "").strip()
    otp = (data.otp or "").strip()

    if not username:
        return {"success": False, "message": "Username is required"}
    if not password:
        return {"success": False, "message": "Password is required"}

    # Mandatory details as per your requirement
    if not first_name:
        return {"success": False, "message": "First name is required"}
    if not last_name:
        return {"success": False, "message": "Last name is required"}
    if not city:
        return {"success": False, "message": "City is required"}
    if not email:
        return {"success": False, "message": "Email is required"}
    if not phone:
        return {"success": False, "message": "Mobile number is required"}
    if not otp:
        return {"success": False, "message": "OTP is required"}

    if username in ALLOWED_USERS:
        return {"success": False, "message": "This username is reserved. Please choose another."}

    record = OTP_STORE.get(email)
    if not record:
        return {"success": False, "message": "OTP not found. Please click Send OTP again."}

    if time.time() > record.get("expiry", 0):
        OTP_STORE.pop(email, None)
        return {"success": False, "message": "OTP expired. Please click Send OTP again."}

    if otp != record.get("otp"):
        return {"success": False, "message": "Invalid OTP"}

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ✅ NEW: extra safety duplicate checks before insert
    c.execute("SELECT 1 FROM users WHERE email=? AND email!='' LIMIT 1", (email,))
    if c.fetchone():
        conn.close()
        return {"success": False, "message": "Email ID already exists. Please use another email."}

    c.execute("SELECT 1 FROM users WHERE phone=? AND phone!='' LIMIT 1", (phone,))
    if c.fetchone():
        conn.close()
        return {"success": False, "message": "Mobile number already exists. Please use another number."}

    try:
        c.execute(
            """
            INSERT INTO users (username, password, email, phone, first_name, last_name, city)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (username, password, email, phone, first_name, last_name, city),
        )
        conn.commit()
    except sqlite3.IntegrityError as e:
        msg = str(e).lower()
        try:
            conn.close()
        except Exception:
            pass
        if "email" in msg:
            return {"success": False, "message": "Email ID already exists. Please use another email."}
        if "phone" in msg:
            return {"success": False, "message": "Mobile number already exists. Please use another number."}
        return {"success": False, "message": "Username already exists"}
    finally:
        try:
            conn.close()
        except Exception:
            pass

    OTP_STORE.pop(email, None)

    return {"success": True, "message": "Signup successful"}

@router.post("/forgot-password/request-otp")
def forgot_password_request_otp(req: ForgotRequest):
    """
    User enters either phone or email.
    Backend finds the user, then sends OTP to the user's stored email.
    """
    try:
        phone = (req.phone or "").strip()
        email_in = (req.email or "").strip().lower()

        if not phone and not email_in:
            return {"success": False, "message": "Enter either Mobile No. or Email ID."}

        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            ensure_forgot_otp_table(conn)

            # ✅ Find user by email OR phone
            user = None
            if email_in:
                c.execute("SELECT username, password, email, phone FROM users WHERE lower(email)=?", (email_in,))
                user = c.fetchone()

            if user is None and phone:
                c.execute("SELECT username, password, email, phone FROM users WHERE phone=?", (phone,))
                user = c.fetchone()

            if user is None:
                return {"success": False, "message": "User not found with provided email/mobile."}

            user_email = (user["email"] or "").strip().lower()
            if not user_email:
                return {"success": False, "message": "No email is registered for this user. Please contact support."}

            # ✅ If user typed an email, enforce match with registered email (optional but safer)
            if email_in and email_in != user_email:
                return {"success": False, "message": "Email does not match registered email for this user."}

            # ✅ Generate OTP
            otp = str(random.randint(1000, 9999))
            now_str = datetime.utcnow().isoformat()

            # Remove old OTPs for this user
            c.execute("DELETE FROM forgot_password_otps WHERE username=?", (user["username"],))

            # Insert new OTP
            c.execute(
                "INSERT INTO forgot_password_otps (username, email, phone, otp, created_at) VALUES (?, ?, ?, ?, ?)",
                (user["username"], user_email, user["phone"], otp, now_str),
            )
            conn.commit()

        # ✅ Send OTP email
        send_email(
            user_email,
            "NeuroCrest - Forgot Password OTP",
            f"""Hello,

Your OTP to reset NeuroCrest password is: {otp}

This OTP is valid for 10 minutes.

— NeuroCrest Team
""",
        )

        return {"success": True, "message": "OTP sent to your registered email.", "email": user_email}

    except Exception as e:
        return {"success": False, "message": f"Failed to send OTP email: {str(e)}"}


@router.post("/forgot-password/verify-otp")
def forgot_password_verify_otp(req: ForgotVerify):
    """
    Verify OTP. If correct, send stored password to user's email.
    """
    try:
        phone = (req.phone or "").strip()
        email_in = (req.email or "").strip().lower()
        otp_in = (req.otp or "").strip()

        if not otp_in or len(otp_in) != 4:
            return {"success": False, "message": "Please enter 4-digit OTP."}

        if not phone and not email_in:
            return {"success": False, "message": "Enter either Mobile No. or Email ID."}

        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            ensure_forgot_otp_table(conn)

            # ✅ Find user
            user = None
            if email_in:
                c.execute("SELECT username, password, email, phone FROM users WHERE lower(email)=?", (email_in,))
                user = c.fetchone()

            if user is None and phone:
                c.execute("SELECT username, password, email, phone FROM users WHERE phone=?", (phone,))
                user = c.fetchone()

            if user is None:
                return {"success": False, "message": "User not found."}

            user_email = (user["email"] or "").strip().lower()
            if not user_email:
                return {"success": False, "message": "No email is registered for this user. Please contact support."}

            # ✅ Fetch OTP row
            c.execute("SELECT otp, created_at FROM forgot_password_otps WHERE username=?", (user["username"],))
            row = c.fetchone()

            if not row:
                return {"success": False, "message": "OTP not found. Please request OTP again."}

            saved_otp = (row["otp"] or "").strip()
            created_at = row["created_at"]

            # ✅ Expiry check (10 minutes)
            created_dt = datetime.fromisoformat(created_at)
            if datetime.utcnow() - created_dt > timedelta(minutes=10):
                c.execute("DELETE FROM forgot_password_otps WHERE username=?", (user["username"],))
                conn.commit()
                return {"success": False, "message": "OTP expired. Please request OTP again."}

            if otp_in != saved_otp:
                return {"success": False, "message": "Incorrect OTP. Please enter correct OTP."}

            # ✅ OTP verified -> delete OTP
            c.execute("DELETE FROM forgot_password_otps WHERE username=?", (user["username"],))
            conn.commit()

        # ✅ Send password email
        pwd = user["password"]
        send_email(
            user_email,
            "NeuroCrest - Your Password",
            f"""Hello,

Your NeuroCrest password is:

{pwd}

Please keep it safe.

— NeuroCrest Team
""",
        )

        return {"success": True, "message": "Password sent successfully on email id."}

    except Exception as e:
        return {"success": False, "message": f"Failed to send password email: {str(e)}"}
