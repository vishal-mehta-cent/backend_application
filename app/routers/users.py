"""backend/app/routers/users.py

Profile read/update + Account Reset.

Account Reset ("Restore") requirement:
- Deletes all user-specific trading data from the DB
- Keeps the user row in `users` table (so login still exists)
- Also clears user-specific CSV/JSON/etc files (optional)
- IMPORTANT: clears data from BOTH possible DB locations (Render + Local),
  so you don't end up with multiple DB files showing old data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sqlite3
from datetime import datetime
import os
import glob

router = APIRouter(prefix="/users", tags=["users"])


# =====================================================
# DB PATH — LOCAL + RENDER SAFE (and handles mixed DBs)
# =====================================================
IS_RENDER = bool(os.getenv("RENDER")) or os.path.isdir("/data")

BACKEND_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

LOCAL_DB = os.path.join(BACKEND_ROOT, "app", "data", "paper_trading.db")
RENDER_DB = "/data/paper_trading.db"


def _primary_db_path() -> str:
    # Render should always use /data
    if IS_RENDER:
        return RENDER_DB
    # Local should use backend/app/data
    return LOCAL_DB


DB_PATH = _primary_db_path()


def _all_db_paths() -> List[str]:
    """
    Reset should wipe user-data from all possible DB files that exist.

    ✅ IMPORTANT:
    - Always touch PRIMARY first (Render => /data/paper_trading.db)
    - Also try legacy/extra locations if they exist
    - Also supports an env override PAPER_TRADING_DB
    """

    env_db = (os.getenv("PAPER_TRADING_DB") or "").strip() or None

    candidates: List[str] = []

    # 1) explicit env override (if you ever set it)
    if env_db:
        candidates.append(env_db)

    # 2) primary first
    candidates.append(_primary_db_path())

    # 3) common known locations
    candidates.extend([
        RENDER_DB,
        LOCAL_DB,
        os.path.join(BACKEND_ROOT, "paper_trading.db"),              # legacy
        os.path.join(os.getcwd(), "paper_trading.db"),              # cwd legacy
        os.path.join(os.getcwd(), "backend", "app", "data", "paper_trading.db"),
    ])

    # 4) if on Render disk, also scan /data for any sqlite dbs (in case of wrong filename)
    if os.path.isdir("/data"):
        try:
            for fp in glob.glob("/data/*.db"):
                candidates.append(fp)
        except Exception:
            pass

    out: List[str] = []
    seen = set()

    primary = _primary_db_path()

    for p in candidates:
        if not p or p in seen:
            continue
        seen.add(p)

        # include if exists OR if it's the primary path (even if not created yet)
        if os.path.exists(p) or p == primary:
            out.append(p)

    # final fallback
    if not out:
        out = [DB_PATH]

    return out


def _ensure_user_columns(c: sqlite3.Cursor) -> None:
    """Make sure users table has profile columns (safe add-only migrations)."""
    c.execute("PRAGMA table_info(users)")
    cols = {row[1].lower() for row in c.fetchall()}
    alters = []
    if "email" not in cols:
        alters.append("ALTER TABLE users ADD COLUMN email TEXT")
    if "phone" not in cols:
        alters.append("ALTER TABLE users ADD COLUMN phone TEXT")
    if "full_name" not in cols:
        alters.append("ALTER TABLE users ADD COLUMN full_name TEXT")
    if "created_at" not in cols:
        alters.append("ALTER TABLE users ADD COLUMN created_at TEXT")
    for stmt in alters:
        try:
            c.execute(stmt)
        except Exception:
            pass


class UpdateProfile(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    full_name: Optional[str] = None


class ResetAccountRequest(BaseModel):
    """
    Frontend should send:
      { "confirm": "RESET", "delete_files": true }
    """
    confirm: str = "RESET"
    delete_files: bool = True


def _table_exists(c: sqlite3.Cursor, table: str) -> bool:
    c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    )
    return c.fetchone() is not None


def _table_info(c: sqlite3.Cursor, table: str) -> Dict[str, Dict[str, str]]:
    """
    Returns dict:
      { "colname_lower": {"name": original_name, "type": type_upper} }
    """
    out: Dict[str, Dict[str, str]] = {}
    try:
        c.execute(f"PRAGMA table_info({table})")
        for r in c.fetchall():
            # r: (cid, name, type, notnull, dflt_value, pk)
            name = (r[1] or "").strip()
            ctype = (r[2] or "").strip().upper()
            if name:
                out[name.lower()] = {"name": name, "type": ctype}
    except Exception:
        pass
    return out


@router.get("/{username}")
def get_user(username: str) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_user_columns(c)
        c.execute(
            "SELECT username, email, phone, full_name, created_at, first_name, last_name, city FROM users WHERE username=?",
            (username,),
        )
        row = c.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        username, email, phone, full_name, created_at, first_name, last_name, city = row
        return {
            "username": username,
            "email": email,
            "phone": phone,
            "full_name": full_name,
            "created_at": created_at,
            "first_name": first_name,
            "last_name": last_name,
            "city": city,
        }

    finally:
        conn.commit()
        conn.close()


@router.patch("/{username}")
def update_user(username: str, data: UpdateProfile):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        _ensure_user_columns(c)

        c.execute("SELECT 1 FROM users WHERE username=?", (username,))
        if not c.fetchone():
            raise HTTPException(status_code=404, detail="User not found")

        fields = []
        params = []
        if data.email is not None:
            fields.append("email=?")
            params.append(data.email)
        if data.phone is not None:
            fields.append("phone=?")
            params.append(data.phone)
        if data.full_name is not None:
            fields.append("full_name=?")
            params.append(data.full_name)

        if not fields:
            return {"success": True, "message": "Nothing to update"}

        c.execute("SELECT created_at FROM users WHERE username=?", (username,))
        cr = c.fetchone()
        if cr and (cr[0] is None or cr[0] == ""):
            c.execute(
                "UPDATE users SET created_at=? WHERE username=?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username),
            )

        params.append(username)
        c.execute(f"UPDATE users SET {', '.join(fields)} WHERE username=?", params)
        conn.commit()
        return {"success": True}
    finally:
        conn.close()


@router.get("/funds/{username}")
def get_funds(username: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        if _table_exists(c, "users"):
            info = _table_info(c, "users")
            if "funds" in info:
                c.execute("SELECT funds FROM users WHERE username = ?", (username,))
                row = c.fetchone()
                if row:
                    return {"funds": row[0]}
        raise HTTPException(status_code=404, detail="User not found")
    finally:
        conn.close()


@router.post("/{username}/reset")
def reset_account(username: str, body: ResetAccountRequest):
    """
    ⚠️ DANGEROUS: wipes all user-specific data.
    - Keeps the `users` table row (so login/profile remains)
    - Deletes rows from tables that contain username/user_id-like columns
    - Runs on ALL DB locations found (Render + Local + legacy)
    """

    if (body.confirm or "").strip().upper() != "RESET":
        raise HTTPException(status_code=400, detail="Type RESET to confirm")

    deleted_total: Dict[str, int] = {}
    dbs_touched: List[str] = []
    dbs_checked: List[str] = _all_db_paths()

    def is_skip_table(t: str) -> bool:
        tl = (t or "").lower()

        # always keep users table + sqlite internal table
        if tl in ("users", "sqlite_sequence"):
            return True

        # ✅ DO NOT reset subscription/payment data (so plans never reset)
        protected_tables = {
            "payments",
            "subscription_periods",
        }
        if tl in protected_tables:
            return True

        # ✅ also protect any future payment/subscription tables you may add later
        protected_words = ("payment", "subscription", "stripe")
        if any(w in tl for w in protected_words):
            return True

        # ✅ never touch auth/session/device tables (prevents logout)
        bad_words = ("session", "token", "auth", "login", "device")
        return any(w in tl for w in bad_words)




    for path in dbs_checked:
        conn = None
        try:
            conn = sqlite3.connect(path, timeout=8)
            c = conn.cursor()

            # List all tables
            c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in c.fetchall() if r and r[0]]
            tables = [t for t in tables if not is_skip_table(t)]

            # Find user primary key if available (users.id)
            user_pk = None
            users_has_id = False
            if _table_exists(c, "users"):
                uinfo = _table_info(c, "users")
                users_has_id = "id" in uinfo
                if users_has_id:
                    try:
                        c.execute("SELECT id FROM users WHERE username=?", (username,))
                        rr = c.fetchone()
                        if rr:
                            user_pk = rr[0]
                    except Exception:
                        user_pk = None

            deleted_here: Dict[str, int] = {}

            for t in tables:
                info = _table_info(c, t)
                cols = set(info.keys())

                def _add_count(n: int):
                    return max(int(n or 0), 0)

                total_deleted = 0

                # Case A: username column
                if "username" in cols:
                    c.execute(f"DELETE FROM {t} WHERE {info['username']['name']}=?", (username,))
                    total_deleted += _add_count(c.rowcount)

                # Case B: user_id column (TEXT or INT)
                if "user_id" in cols:
                    colname = info["user_id"]["name"]
                    coltype = info["user_id"]["type"]

                    # Try username as text id
                    c.execute(f"DELETE FROM {t} WHERE {colname}=?", (username,))
                    total_deleted += _add_count(c.rowcount)

                    # If users.id is available, delete by integer id mapping
                    if users_has_id:
                        # safest: subquery (works even if we couldn't fetch user_pk)
                        c.execute(
                            f"DELETE FROM {t} WHERE {colname} IN (SELECT id FROM users WHERE username=?)",
                            (username,),
                        )
                        total_deleted += _add_count(c.rowcount)

                    # direct integer delete if we have user_pk and column is INT-ish
                    if user_pk is not None and ("INT" in coltype):
                        c.execute(f"DELETE FROM {t} WHERE {colname}=?", (user_pk,))
                        total_deleted += _add_count(c.rowcount)

                # Case C: some common legacy variations
                for legacy_col in ("user", "user_name", "userid", "uid"):
                    if legacy_col in cols:
                        cname = info[legacy_col]["name"]
                        c.execute(f"DELETE FROM {t} WHERE {cname}=?", (username,))
                        total_deleted += _add_count(c.rowcount)

                if total_deleted:
                    deleted_here[t] = total_deleted
                else:
                    # keep table in response to help debug “why 0”
                    deleted_here.setdefault(t, 0)

            # Reset funds columns if present (users table)
            if _table_exists(c, "users"):
                uinfo = _table_info(c, "users")
                updates = []
                if "funds" in uinfo:
                    updates.append(f"{uinfo['funds']['name']}=0.0")
                if "available_funds" in uinfo:
                    updates.append(f"{uinfo['available_funds']['name']}=0.0")
                if "total_funds" in uinfo:
                    updates.append(f"{uinfo['total_funds']['name']}=0.0")
                if updates:
                    c.execute(f"UPDATE users SET {', '.join(updates)} WHERE username=?", (username,))

            conn.commit()
            dbs_touched.append(path)

            # aggregate counts
            for k, v in deleted_here.items():
                deleted_total[k] = deleted_total.get(k, 0) + int(v or 0)

        except Exception:
            try:
                if conn:
                    conn.rollback()
            except Exception:
                pass
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

    # Optional: delete per-user files (CSV/JSON/etc.) for this user
    files_deleted: List[str] = []
    if body.delete_files:
        candidate_dirs = [
            "/data",
            os.path.join(BACKEND_ROOT, "app", "data"),
            os.path.join(BACKEND_ROOT, "app"),
            BACKEND_ROOT,
        ]
        candidate_dirs = [d for d in candidate_dirs if d and os.path.isdir(d)]

        patterns = [
            f"{username}_*.csv",
            f"{username}_*.json",
            f"{username}_*.xlsx",
            f"{username}_*.txt",
        ]

        for d in candidate_dirs:
            for pat in patterns:
                for fp in glob.glob(os.path.join(d, pat)):
                    if fp.lower().endswith(".db"):
                        continue
                    try:
                        os.remove(fp)
                        files_deleted.append(fp)
                    except Exception:
                        pass

    if not dbs_touched:
        raise HTTPException(status_code=404, detail="No DB was writable/available to reset")

    return {
        "success": True,
        "message": "Account data reset successfully",
        "dbs_checked": dbs_checked,      # 👈 helps you debug path issues on Render
        "dbs_touched": dbs_touched,
        "deleted_rows": deleted_total,
        "files_deleted": len(files_deleted),
    }
