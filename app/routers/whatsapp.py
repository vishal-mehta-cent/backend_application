from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os, json, platform
import pandas as pd

# ----------------------------------------------------
# CONDITIONAL IMPORT - SELENIUM FOR LOCAL ONLY
# ----------------------------------------------------
IS_RENDER = "RENDER" in os.environ  # Render auto-sets this env var

if not IS_RENDER:
    try:
        from Whatsapp_Push_Module import whatsapp_push_execution
        LOCAL_WHATSAPP_ENABLED = True
        print("📌 Local WhatsApp module loaded (Selenium enabled)")
    except Exception as e:
        print("⚠ Failed to import Whatsapp_Push_Module:", e)
        LOCAL_WHATSAPP_ENABLED = False
else:
    print("🚫 Render environment detected — WhatsApp automation disabled")
    LOCAL_WHATSAPP_ENABLED = False


router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])

# ----------------------------------------------------
# STORAGE LOCATION
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(BASE_DIR, "..", "data", "whatsapp_alerts.json")
RECO_CSV = os.path.join(BASE_DIR, "..", "data", "Recommendation_Data.csv")

# ----------------------------------------------------
# USER DETAILS CSV PATH (LOCAL vs RENDER)
# ----------------------------------------------------
if IS_RENDER:
    USER_DETAILS_CSV = "/data/whatsapp_user_details.csv"
else:
    USER_DETAILS_CSV = os.path.join(BASE_DIR, "..", "data", "whatsapp_user_details.csv")


os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)


# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def load_alerts():
    if not os.path.exists(STORE_PATH):
        return []
    try:
        with open(STORE_PATH, "r") as f:
            return json.load(f)
    except:
        return []


def save_alerts(alerts):
    with open(STORE_PATH, "w") as f:
        json.dump(alerts, f, indent=2)



class AlertRequest(BaseModel):
    script: str
    user_id: str = ""


class WhatsappNumberSave(BaseModel):
    user_id: str
    email_id: str
    whatsapp_number: str


# ----------------------------------------------------
# ADD ALERT
# ----------------------------------------------------
@router.post("/add-alert")
def add_alert(req: AlertRequest):
    script = req.script.strip().upper()
    if not script:
        raise HTTPException(status_code=400, detail="Invalid script name")

    alerts = load_alerts()

    # Duplicate check for both formats
    for item in alerts:
        if isinstance(item, str) and item == script:
            return {"status": "exists", "message": f"{script} already exists", "alerts": alerts}
        if isinstance(item, dict) and item.get("script") == script:
            return {"status": "exists", "message": f"{script} already exists", "alerts": alerts}

    # New entry
    new_entry = {
        "user_id": req.user_id or "GLOBAL",
        "script": script,
        "fast": False,
        "intraday": False,
        "btst": False,
        "shortterm": False
    }


    alerts.append(new_entry)
    save_alerts(alerts)

    return {"status": "ok", "message": f"{script} added", "alerts": alerts}


# ----------------------------------------------------
# REMOVE ALERT
# ----------------------------------------------------
@router.delete("/remove-alert/{script}")
def remove_alert(script: str):
    script = script.strip().upper()
    alerts = load_alerts()

    new_list = []
    for item in alerts:
        if isinstance(item, str) and item == script:
            continue
        if isinstance(item, dict) and item.get("script") == script:
            continue
        new_list.append(item)

    save_alerts(new_list)
    return {"status": "ok", "alerts": new_list}


# ----------------------------------------------------
# LIST ALERTS
# ----------------------------------------------------
# ----------------------------------------------------
# CLEAN LIST ALERTS (ONLY SCRIPT NAMES)
# ----------------------------------------------------
@router.get("/list")
def get_clean_alerts():
    alerts = load_alerts()

    clean_list = []

    for item in alerts:
        if isinstance(item, str):
            clean_list.append(item.strip().upper())
        elif isinstance(item, dict):
            script = str(item.get("script", "")).strip().upper()
            if script:
                clean_list.append(script)

    # Remove duplicates
    clean_list = list(dict.fromkeys(clean_list))

    return clean_list



# ----------------------------------------------------
# NORMALIZE FULL FORMAT
# ----------------------------------------------------
def normalize_alerts_full():
    alerts = load_alerts()
    normalized = []

    for item in alerts:
        if isinstance(item, str):
            normalized.append({
                "script": item,
                "fast": False,
                "intraday": False,
                "btst": False,
                "shortterm": False
            })
        else:
            # Ensure missing fields = False
            item.setdefault("fast", False)
            item.setdefault("intraday", False)
            item.setdefault("btst", False)
            item.setdefault("shortterm", False)
            normalized.append(item)

    save_alerts(normalized)
    return normalized


@router.get("/list-full")
def list_full():
    return normalize_alerts_full()


# ----------------------------------------------------
# SAVE SETTINGS
# ----------------------------------------------------
class SaveSettings(BaseModel):
    settings: list


@router.post("/save-settings")
def save_settings(req: SaveSettings):
    save_alerts(req.settings)
    return {"ok": True}


# ----------------------------------------------------
# PUSH ON SAVE — SEND WHATSAPP NOTIFICATION
# ----------------------------------------------------
@router.post("/push-on-save")
def push_on_save(req: SaveSettings):

    settings = req.settings

    csv_path = os.path.join(BASE_DIR, "..", "data", "Recommendation_Data.csv")
    if not os.path.exists(csv_path):
        return {"ok": False, "error": "Recommendation CSV not found"}


    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "script" not in df.columns:
        return {"ok": False, "error": "CSV missing 'script' column"}

    df["script"] = df["script"].astype(str).str.upper()

    final_messages = []

    # ----------------------------------------------------
    # STEP 1: BUILD ALL MESSAGES
    # ----------------------------------------------------
    for row in settings:
        script = row.get("script", "").upper()

        fast_flag = bool(row.get("fast", False))
        intraday_flag = bool(row.get("intraday", False))
        btst_flag = bool(row.get("btst", False))
        shortterm_flag = bool(row.get("shortterm", False))

        df_script = df[df["script"] == script]
        if df_script.empty:
            print(f"NO MATCH FOUND FOR {script} IN CSV")
            continue

        for _, srow in df_script.iterrows():

            strategy_text = str(srow.get("strategy", "")).strip()
            strategy = strategy_text.lower()

            is_fast = strategy.startswith("intraday - fast")
            is_intraday = strategy.startswith("intraday") and not is_fast
            is_btst = "btst" in strategy
            is_shortterm = "short" in strategy


            msg = f"""
SIGNAL : {srow.get('alert_details','')}
STRATEGY : {strategy_text}
Data Interval : {srow.get('data_interval','')}
Date : {srow.get('date','')}
Script : {script}
Price : {srow.get('close_price','')}
Price close to Res/Sup : {srow.get('screener_side','')}
Max Gain Potential : {srow.get('max_gain','')}

Resistance : {srow.get('resistance','')}
Support : {srow.get('support','')}
Pivot Upper Band : {srow.get('pivot_upper','')}
Pivot Lower Band : {srow.get('pivot_lower','')}
ALERTS : {srow.get('alert_details','')}

Screener : {srow.get('screener','')}
User Action : {srow.get('user_actions','')}
"""

            if is_fast and fast_flag:
                final_messages.append(msg)

            if is_intraday and intraday_flag:
                final_messages.append(msg)

            if is_btst and btst_flag:
                final_messages.append(msg)

            if is_shortterm and shortterm_flag:
                final_messages.append(msg)

    # ----------------------------------------------------
    # STEP 2: SEND WHATSAPP
    # ----------------------------------------------------
    sent_count = 0

    if LOCAL_WHATSAPP_ENABLED:
        for msg in final_messages:
            whatsapp_push_execution(msg)
            sent_count += 1
    else:
        # Render / cloud → simulate success
        sent_count = len(final_messages)

    # ----------------------------------------------------
    # STEP 3: RETURN RESULT
    # ----------------------------------------------------
    return {
        "status": "ok",
        "sent_count": sent_count
    }

# ============================================================
# SAVE WHATSAPP USER DETAILS (CSV BASED)
# ============================================================

class WhatsappUserSave(BaseModel):
    user_id: str
    email_id: str
    whatsapp_number: str
    settings: list   # coming from UI (scripts + checkboxes)


@router.post("/save-user-details")
def save_whatsapp_user_details(req: WhatsappUserSave):

    CSV_PATH = USER_DETAILS_CSV

    columns = [
        "user_id",
        "email_id",
        "whatsapp_number",
        "scripts",
        "strategies"
    ]

    # ------------------------------------------------
    # STEP 1: Load existing CSV (or create empty)
    # ------------------------------------------------
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=columns)

    # ------------------------------------------------
    # STEP 2: Keep rows of OTHER users
    # ------------------------------------------------
    df_other_users = df[df["user_id"] != req.user_id]

    # ------------------------------------------------
    # STEP 3: Build NEW rows from current table state
    # ------------------------------------------------
    new_rows = []

    for row in req.settings:
        script = str(row.get("script", "")).upper()

        if row.get("fast"):
            new_rows.append({
                "user_id": req.user_id,
                "email_id": req.email_id,
                "whatsapp_number": req.whatsapp_number,
                "scripts": script,
                "strategies": "Intraday Fast Alert"
            })

        if row.get("intraday"):
            new_rows.append({
                "user_id": req.user_id,
                "email_id": req.email_id,
                "whatsapp_number": req.whatsapp_number,
                "scripts": script,
                "strategies": "Intraday"
            })

        if row.get("btst"):
            new_rows.append({
                "user_id": req.user_id,
                "email_id": req.email_id,
                "whatsapp_number": req.whatsapp_number,
                "scripts": script,
                "strategies": "BTST"
            })

        if row.get("shortterm"):
            new_rows.append({
                "user_id": req.user_id,
                "email_id": req.email_id,
                "whatsapp_number": req.whatsapp_number,
                "scripts": script,
                "strategies": "Short-Term"
            })

    # ------------------------------------------------
    # STEP 4: Create dataframe from new rows
    # ------------------------------------------------
    df_user_new = pd.DataFrame(new_rows, columns=columns)

    # ------------------------------------------------
    # STEP 5: Merge back (overwrite user only)
    # ------------------------------------------------
    final_df = pd.concat([df_other_users, df_user_new], ignore_index=True)

    # ------------------------------------------------
    # STEP 6: Save
    # ------------------------------------------------
    final_df.to_csv(CSV_PATH, index=False)

    print(f"✅ WhatsApp preferences saved for {req.user_id}: {len(df_user_new)} rows")

    return {
        "status": "ok",
        "rows_saved": len(df_user_new)
    }


@router.post("/save-number")
def save_whatsapp_number(req: WhatsappNumberSave):

    CSV_PATH = USER_DETAILS_CSV  # already defined earlier

    cols = ["user_id", "email_id", "whatsapp_number", "scripts", "strategies"]

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=cols)

    # Remove old rows for this user
    df = df[df["user_id"] != req.user_id]

    # Insert placeholder row (scripts empty for now)
    new_row = {
        "user_id": req.user_id,
        "email_id": req.email_id,
        "whatsapp_number": req.whatsapp_number,
        "scripts": "",
        "strategies": ""
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    return {"status": "ok"}

@router.get("/get-number")
def get_whatsapp_number(user_id: str):

    CSV_PATH = USER_DETAILS_CSV

    if not os.path.exists(CSV_PATH):
        return {"whatsapp_number": ""}

    df = pd.read_csv(CSV_PATH)

    row = df[df["user_id"] == user_id]

    if row.empty:
        return {"whatsapp_number": ""}

    return {"whatsapp_number": str(row.iloc[0]["whatsapp_number"])}

@router.get("/user-settings")
def get_user_settings(user_id: str):

    # 1️⃣ Load ALL scripts from JSON (GLOBAL list)
    alerts = load_alerts()

    # 1️⃣ Extract scripts from JSON (FIXED)
    scripts = []

    for a in alerts:
        if isinstance(a, dict):
            script = str(a.get("script", "")).strip()
            if script:
                scripts.append(script.upper())
        elif isinstance(a, str):
            scripts.append(a.strip().upper())

    # Remove duplicates & empty
    scripts = list(dict.fromkeys([s for s in scripts if s]))


    # 2️⃣ Load USER-SPECIFIC CSV preferences
    if not os.path.exists(USER_DETAILS_CSV):
        df = pd.DataFrame(columns=["user_id", "scripts", "strategies"])
    else:
        df = pd.read_csv(USER_DETAILS_CSV)

    df_user = df[df["user_id"] == user_id]

    # 3️⃣ Build base settings for UI
    settings = {
        s: {
            "script": s,
            "fast": False,
            "intraday": False,
            "btst": False,
            "shortterm": False
        }
        for s in scripts
    }

    # 4️⃣ Apply user preferences
    for _, row in df_user.iterrows():
        script = str(row["scripts"]).upper()
        strategy = str(row["strategies"]).lower()

        if script not in settings:
            continue

        if "fast" in strategy:
            settings[script]["fast"] = True
        elif strategy == "intraday":
            settings[script]["intraday"] = True
        elif strategy == "btst":
            settings[script]["btst"] = True
        elif "short" in strategy:
            settings[script]["shortterm"] = True

    return { "settings": list(settings.values()) }


