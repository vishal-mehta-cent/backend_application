# backend/app/routers/kite.py
from fastapi import APIRouter
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import threading
import time

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore

from app.services import kite_ws_manager as manager
from kiteconnect import KiteConnect
from fastapi import Body, Header, HTTPException

router = APIRouter(prefix="/kite", tags=["kite"])


# -------------------------
# Render / Local detection
# -------------------------
def _is_render() -> bool:
    return bool(os.getenv("RENDER")) or Path("/data").exists()


IS_RENDER = _is_render()


# -------------------------
# .env loading (LOCAL ONLY)
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
DOTENV_PATH = BASE_DIR / ".env"

# On Render, rely on Render Environment vars (do NOT override them from a .env file)
if (not IS_RENDER) and DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)


def _get_api_key() -> str:
    return (os.getenv("KITE_API_KEY", "") or "").strip().strip('"')


def _get_access_token() -> str:
    return (os.getenv("KITE_ACCESS_TOKEN", "") or "").strip().strip('"')


def _norm_path(p: str) -> str:
    return (p or "").strip().strip('"').replace("\\", "/")


def _get_instruments_target_path() -> Path:
    """
    Source of truth for instruments file:
    - if INSTRUMENTS_CSV_PATH is set -> use it
    - else on Render -> /data/instruments.csv
    - else local -> backend/app/data/instruments.csv
    """
    p = _norm_path(os.getenv("INSTRUMENTS_CSV_PATH", ""))
    if p:
        return Path(p)

    if IS_RENDER:
        return Path("/data/instruments.csv")

    return BASE_DIR / "app" / "data" / "instruments.csv"


def _get_instruments_mirror_paths(target_path: Path) -> list[Path]:
    """
    Best-effort mirror locations (optional).
    We do NOT seed from these. We only WRITE to these using Zerodha download bytes.

    You can add your own paths via env:
      INSTRUMENTS_MIRROR_PATHS="/opt/render/project/src/app/instruments.csv,/opt/render/project/src/app/data/instruments.csv"
    """
    mirrors: list[Path] = []

    # user-defined mirrors
    raw = _norm_path(os.getenv("INSTRUMENTS_MIRROR_PATHS", ""))
    if raw:
        for part in raw.split(","):
            p = _norm_path(part)
            if p:
                mirrors.append(Path(p))

    # common repo/app paths (safe best-effort)
    # (On Render these may be ephemeral or read-only; we'll ignore failures.)
    mirrors += [
        BASE_DIR / "instruments.csv",
        BASE_DIR / "app" / "instruments.csv",
        BASE_DIR / "app" / "data" / "instruments.csv",
    ]

    # remove duplicates and exclude target
    uniq: list[Path] = []
    seen = set()
    for p in mirrors:
        s = str(p)
        if s == str(target_path):
            continue
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


kite = None


def init_kiteconnect():
    """
    Initializes and returns a global KiteConnect instance.
    Called once when backend starts.
    """
    global kite
    try:
        api_key = _get_api_key()
        access_token = _get_access_token()

        if not api_key or not access_token:
            raise Exception("KITE_API_KEY or KITE_ACCESS_TOKEN missing in env")

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        print("✅ KiteConnect initialized successfully")
        return kite

    except Exception as e:
        print(f"❌ Failed to initialize KiteConnect: {e}")
        kite = None
        return None


def get_kite_instance():
    """
    Returns initialized KiteConnect instance; auto-init if missing.
    """
    global kite
    if kite is None:
        kite = init_kiteconnect()

    if kite is None:
        raise Exception("⚠️ KiteConnect instance not available. Check credentials.")
    return kite


def _write_bytes_atomic(path: Path, content: bytes) -> tuple[bool, str]:
    """
    Atomic write: write to .tmp then replace.
    Returns (ok, message)
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(content)
        os.replace(str(tmp), str(path))
        return True, f"wrote {path}"
    except Exception as e:
        return False, f"failed {path}: {e}"


def _reload_manager_instruments(target_file: str) -> str:
    """
    Reload manager's in-memory DF after writing a new CSV.
    """
    try:
        if hasattr(manager, "reload_instruments"):
            manager.reload_instruments(path=target_file)  # type: ignore
            return "manager.reload_instruments() called"
    except Exception as e:
        return f"manager.reload_instruments() failed: {e}"

    # fallback (should not be needed once reload_instruments exists)
    try:
        if hasattr(manager, "_read_csv_safely"):
            df = manager._read_csv_safely(target_file)  # type: ignore
            if hasattr(manager, "INSTRUMENTS_DF"):
                manager.INSTRUMENTS_DF = df  # type: ignore
            if hasattr(manager, "EQUITY_DF"):
                manager.EQUITY_DF = df.copy()  # type: ignore
            return "manager INSTRUMENTS_DF refreshed"
    except Exception as e:
        return f"manager DF refresh failed: {e}"

    return "no reload method available"


# ===== Helper: Download instruments file =====
def download_instruments(force: bool = False) -> dict:
    """
    Downloads latest Zerodha instruments CSV.
    ✅ Writes to /data/instruments.csv on Render (persistent disk)
    ✅ Uses INSTRUMENTS_CSV_PATH if provided
    ✅ Atomic write
    ✅ Mirrors to other locations (best-effort) using the SAME downloaded bytes
    """
    target_path = _get_instruments_target_path()

    try:
        # Optional "freshness" skip (prevents repeated downloads)
        # You can tune via env: INSTRUMENTS_MIN_REFRESH_SECONDS (default 1800 = 30 min)
        min_refresh = int(os.getenv("INSTRUMENTS_MIN_REFRESH_SECONDS", "1800") or "1800")
        if (not force) and target_path.exists():
            age = (datetime.utcnow() - datetime.utcfromtimestamp(target_path.stat().st_mtime)).total_seconds()
            if age < min_refresh:
                return {
                    "status": "ok",
                    "message": f"Skipped download (fresh < {min_refresh}s)",
                    "latest_file": str(target_path),
                    "reload": "skipped",
                }

        instruments_url = "https://api.kite.trade/instruments"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
        resp = requests.get(instruments_url, headers=headers, timeout=60)

        if resp.status_code != 200:
            return {"status": "error", "message": f"Failed {resp.status_code}: {resp.text}"}

        content = resp.content

        ok, msg = _write_bytes_atomic(target_path, content)
        if not ok:
            return {"status": "error", "message": msg, "latest_file": str(target_path)}

        mirror_results = []
        for mp in _get_instruments_mirror_paths(target_path):
            ok2, msg2 = _write_bytes_atomic(mp, content)
            mirror_results.append({"path": str(mp), "ok": ok2, "message": msg2})

        # Reload manager DF
        reload_msg = _reload_manager_instruments(str(target_path))

        return {
            "status": "ok",
            "message": "Instrument list updated from Zerodha",
            "latest_file": str(target_path),
            "reload": reload_msg,
            "mirrors": mirror_results,
        }

    except Exception as e:
        return {"status": "error", "message": str(e), "latest_file": str(target_path)}


# -------------------------
# Daily auto-refresh @ 8:00 AM IST (Render)
# -------------------------
_DAILY_THREAD_STARTED = False


def _seconds_until_next_8am_ist() -> float:
    # 8:00 AM IST daily
    if ZoneInfo is None:
        # Fallback: IST = UTC+5:30
        ist_offset = timedelta(hours=5, minutes=30)
        now_utc = datetime.utcnow()
        now_ist = now_utc + ist_offset
        target_ist = now_ist.replace(hour=8, minute=0, second=0, microsecond=0)
        if now_ist >= target_ist:
            target_ist += timedelta(days=1)
        target_utc = target_ist - ist_offset
        return max(5.0, (target_utc - now_utc).total_seconds())

    tz = ZoneInfo("Asia/Kolkata")
    now_ist = datetime.now(tz)
    target_ist = now_ist.replace(hour=8, minute=0, second=0, microsecond=0)
    if now_ist >= target_ist:
        target_ist = target_ist + timedelta(days=1)
    return max(5.0, (target_ist - now_ist).total_seconds())


def _daily_refresh_loop():
    while True:
        try:
            sleep_s = _seconds_until_next_8am_ist()
            print(f"ℹ️ Daily instruments refresh scheduled in {int(sleep_s)} seconds (next 08:00 IST).")
            time.sleep(sleep_s)
            print("ℹ️ Running daily instruments refresh (08:00 IST) from Zerodha...")
            res = download_instruments(force=True)
            print(f"✅ Daily instruments refresh result: {res.get('status')} {res.get('message')}")
        except Exception as e:
            print(f"⚠️ Daily instruments refresh loop error: {e}")
            # wait a minute before retrying the loop
            time.sleep(60)


def _start_daily_thread_if_needed():
    global _DAILY_THREAD_STARTED

    # Enabled by default on Render; off by default locally unless you set it.
    enabled = os.getenv("ENABLE_DAILY_INSTRUMENTS_REFRESH")
    if enabled is None:
        enabled = "1" if IS_RENDER else "0"

    if enabled.strip() not in ("1", "true", "True", "YES", "yes"):
        return

    if _DAILY_THREAD_STARTED:
        return

    t = threading.Thread(target=_daily_refresh_loop, daemon=True)
    t.start()
    _DAILY_THREAD_STARTED = True
    print("✅ Started daily instruments refresh thread (08:00 IST).")


# ===== Routes =====
@router.get("/status")
def kite_status():
    """
    Show whether API key and access token are loaded from env.
    (Avoid exposing secrets in full.)
    """
    api_key = _get_api_key()
    access_token = _get_access_token()

    def _mask(s: str) -> str:
        if not s:
            return ""
        return s[:4] + "..." + s[-2:] if len(s) >= 8 else "****"

    return {
        "is_render": IS_RENDER,
        "api_key_loaded": bool(api_key),
        "api_key_preview": _mask(api_key) if api_key else None,
        "access_token_loaded": bool(access_token),
        "access_token_preview": _mask(access_token) if access_token else None,
        "instruments_target_path": str(_get_instruments_target_path()),
        "daily_refresh_enabled": os.getenv("ENABLE_DAILY_INSTRUMENTS_REFRESH", "1" if IS_RENDER else "0"),
    }


@router.post("/reload-access-token")
def reload_access_token():
    """
    For LOCAL:
      - if .env changed, it can be reloaded.
    For Render:
      - environment vars won't change at runtime; you must redeploy to pick new values.
    Also downloads the latest instruments file to persistent disk.
    """
    global kite

    # Reload .env only locally
    if (not IS_RENDER) and DOTENV_PATH.exists():
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)

    access_token = _get_access_token()
    api_key = _get_api_key()

    if not api_key or not access_token:
        return {"status": "error", "message": "KITE_API_KEY / KITE_ACCESS_TOKEN missing in env"}

    # Update runtime env
    os.environ["KITE_API_KEY"] = api_key
    os.environ["KITE_ACCESS_TOKEN"] = access_token

    # Refresh KiteConnect instance
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
    except Exception as e:
        return {"status": "error", "message": f"Failed to re-init KiteConnect: {e}"}

    # Download latest instruments to /data (or env path) — force it on token reload
    instruments_result = download_instruments(force=True)

    # If you have websocket restart logic elsewhere, keep it guarded:
    ws_restart = None
    try:
        if hasattr(manager, "_start_ws"):
            manager._start_ws()  # type: ignore
            ws_restart = "manager._start_ws() called"
    except Exception as e:
        ws_restart = f"manager._start_ws() failed: {e}"

    return {
        "status": "ok",
        "message": "Access token reloaded & instruments updated",
        "access_token_preview": access_token[:6] + "..." if access_token else None,
        "ws_restart": ws_restart,
        "instruments": instruments_result,
    }


@router.api_route("/refresh-instruments", methods=["GET", "POST"])
def refresh_instruments():
    """
    Manually download latest instruments file from Zerodha.
    Writes to /data/instruments.csv on Render.
    """
    result = download_instruments(force=True)
    return {
        "status": result.get("status"),
        "message": result.get("message", ""),
        "latest_file": result.get("latest_file"),
        "reload": result.get("reload"),
        "mirrors": result.get("mirrors"),
    }


# Initialize once at import
init_kiteconnect()

# Refresh instruments once at startup (will skip if fresh due to min_refresh)
download_instruments(force=False)

_start_daily_thread_if_needed()



@router.post("/set-access-token")
def set_access_token(
    access_token: str = Body(..., embed=True),
    admin_key: str = Header("", alias="X-Admin-Key"),
):
    # protect it
    if admin_key != (os.getenv("PAYMENT_ADMIN_KEY") or "").strip():
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not access_token.strip():
        return {"status": "error", "message": "Empty access_token"}

    # set in-process env (works immediately)
    os.environ["KITE_ACCESS_TOKEN"] = access_token.strip()

    # re-init KiteConnect
    global kite
    api_key = _get_api_key()
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token.strip())

    # refresh instruments + reload manager DF
    instruments_result = download_instruments(force=True)

    return {"status": "ok", "message": "Token updated & instruments refreshed", "instruments": instruments_result}

