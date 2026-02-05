# Backend/main.py

import os
import sys
import logging
import threading
import time
from pathlib import Path

# ---------------- Env ----------------
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

ENV_PATH = Path(__file__).resolve().parent / ".env"
if load_dotenv and ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # backend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                       # project root

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
routers_path = os.path.join(BASE_DIR, "app", "routers")
try:
    logging.info("Routers found: %s", os.listdir(routers_path))
except Exception as e:
    logging.error("Could not list routers: %s", e)

# ---------------- DB Init (safe) ----------------
def _safe_init_db():
    try:
        from init_db import init  # type: ignore
        init()
        logging.info("DB init completed.")
    except Exception as e:
        logging.warning(f"DB init skipped/failed: {e}")

_safe_init_db()

# ---------------- FastAPI ----------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import payments
from app.routers import recommendations
from app.routers.market import router as market_router
from app.routers.whatsapp import router as whatsapp_router
from app.routers.orders import router as orders_router
from app.routers.funds import router as funds_router
from app.routers.watchlist import router as watchlist_router
from app.routers import auth
# Optional scheduler deps (don’t crash if missing)
from app.routers.payments import init_payments_storage
try:
    from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
    from apscheduler.jobstores.base import JobLookupError  # type: ignore
except Exception:
    BackgroundScheduler = None  # type: ignore
    JobLookupError = Exception  # type: ignore

# ---------------- App ----------------
app = FastAPI(title="NeuroCrest Backend", version="1.0.0")

# ---------------- CORS ----------------
ALLOWED_ORIGINS = [
    # Local dev
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost",
    "https://localhost",

    # LAN dev (open on phone/emulator) — add/remove IPs you actually use
    "http://192.168.1.5:5173",
    "http://192.168.1.5:5174",
    "http://192.168.1.5:5175",
    "http://192.168.1.5:5176",
    "http://192.168.1.5:8000",
    "http://192.168.1.55:5173",
    "http://192.168.1.55:5176",
    "http://192.168.1.55:8000",

    # Android emulator loopback
    "http://10.0.2.2:5173",
    "http://10.0.2.2:5175",
    "http://10.0.2.2:8000",

    # Capacitor
    "capacitor://localhost",
    "ionic://localhost",

    # Deployed frontends / domains you mentioned
    "https://paper-trading-frontend.vercel.app",
    "https://frontend-app-ten-opal.vercel.app",
    "https://www.neurocrest.in",
    "https://neurocrest.in",
]

# Optional extra origins via env: ALLOWED_ORIGINS="https://x,https://y"
EXTRA = os.getenv("ALLOWED_ORIGINS", "")
if EXTRA:
    for o in [x.strip() for x in EXTRA.split(",") if x.strip()]:
        if o not in ALLOWED_ORIGINS:
            ALLOWED_ORIGINS.append(o)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://paper-trading-frontend.vercel.app",
        "https://www.neurocrest.in",
        "https://neurocrest.in",
        "capacitor://localhost",
        "ionic://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Health ----------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/", tags=["Health"])
async def root():
    return {"message": "✅ Backend is running!"}

# ---------------- Router include helper ----------------
def include_router_safely(module_path: str, router_attr: str = "router"):
    """
    Dynamically import and include a router. Logs and skips on error.
    """
    try:
        mod = __import__(module_path, fromlist=[router_attr])
        router = getattr(mod, router_attr)
        app.include_router(router)
        logging.info("Included router: %s", module_path)
    except Exception as e:
        logging.warning("Skipped router %s: %s", module_path, e)

# ---------------- Include Routers ----------------
# Use your module names here; any missing/broken router will be logged and skipped.
include_router_safely("app.routers.auth")
include_router_safely("app.routers.search")
app.include_router(watchlist_router)
include_router_safely("app.routers.quotes")
include_router_safely("app.routers.portfolio")
#include_router_safely("app.routers.orders")
app.include_router(orders_router)
include_router_safely("app.routers.auth_google")
app.include_router(funds_router)
include_router_safely("app.routers.feedback")
include_router_safely("app.routers.users")
include_router_safely("app.routers.features")

include_router_safely("app.routers.kite")
include_router_safely("app.routers.market") 
app.include_router(market_router)
app.include_router(payments.router)
#include_router_safely("app.routers.recommendations")
app.include_router(recommendations.router)
app.include_router(whatsapp_router)
app.include_router(auth.router)




logging.info("Routers found: %s", os.listdir(routers_path))

# ---------------- Instruments updater (optional) ----------------
def _call_update_instruments_if_available():
    """
    Try to run the synchronous 'update_instruments' you referenced.
    Non-fatal if not present.
    """
    try:
        from app.update_instruments import update_instruments  # type: ignore
        update_instruments()
        logging.info("update_instruments executed.")
    except Exception as e:
        logging.warning(f"Skipping instrument update: {e}")

# ---------------- .env Watcher (optional) ----------------
def env_watcher():
    """
    Background thread to detect .env changes and (optionally) refresh instruments.
    Safe: logs on error, never crashes app.
    """
    last_mtime = ENV_PATH.stat().st_mtime if ENV_PATH.exists() else None
    while True:
        try:
            if ENV_PATH.exists():
                current_mtime = ENV_PATH.stat().st_mtime
                if last_mtime is None or current_mtime != last_mtime:
                    logging.info("🔄 .env changed → reloading.")
                    if load_dotenv:
                        load_dotenv(ENV_PATH, override=True)
                    # If you have a cheap refresh function in kite router, call it guarded
                    try:
                        from app.routers.kite import download_instruments  # <-- fixed typo here
                        download_instruments()
                        logging.info("Instruments refreshed after .env reload.")
                    except Exception as ie:
                        logging.info(f"No download_instruments or failed to refresh: {ie}")
                    last_mtime = current_mtime
        except Exception as e:
            logging.error(f"env_watcher error: {e}")
        time.sleep(10)

# ---------------- Startup ----------------
@app.on_event("startup")
async def on_startup():
    init_payments_storage()
    logging.info("🚀 Startup initiated...")
    # Start .env watcher (daemon) if desired
    t = threading.Thread(target=env_watcher, name="env_watcher", daemon=True)
    t.start()

    # Try a gentle instruments refresh policy.
    # If you have an async refresh function, guard it.
    try:
        # Prefer cached CSV within last hour.
        csv_path = os.path.join(BASE_DIR, "app", "instruments.csv")
        if not os.path.exists(csv_path) or (time.time() - os.path.getmtime(csv_path) > 3600):
            # Try to call a known async or sync refresh function if present.
            refreshed = False
            try:
                from app.routers.kite import refresh_instruments  # type: ignore
                if callable(refresh_instruments):
                    await refresh_instruments()  # type: ignore
                    logging.info("Async instruments refresh done.")
                    refreshed = True
            except Exception:
                pass

            # 2) Fallback to sync updater you referenced
            if not refreshed:
                _call_update_instruments_if_available()
        else:
            logging.info("Using cached instruments file (fresh < 1h).")
    except Exception as e:
        logging.warning(f"Startup instruments step skipped/failed: {e}")

# ---------------- Optional: scheduler example ----------------
# If you want a timed job and apscheduler is installed, you can enable here.
# This is disabled by default to avoid extra threads in dev.
"""
if BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    def _daily_job():
        logging.info("Running daily job...")
        # put your cleanup or summary job here
    try:
        scheduler.add_job(
            _daily_job,
            trigger='cron',
            hour=15, minute=45,
            day_of_week='mon-fri',
            id='daily_order_cleanup',
            replace_existing=True
        )
        scheduler.start()
    except Exception as e:
        logging.warning(f"Apscheduler start failed: {e}")
"""

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    import uvicorn  # type: ignore
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
