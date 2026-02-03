import pandas as pd
from datetime import datetime
import pytz
import numpy as np

# -----------------------------
# IST Timezone
# -----------------------------
IST = pytz.timezone("Asia/Kolkata")

# ============================================================
# PARSE DATE → UNIX TIMESTAMP (VERY ROBUST)
# ============================================================
def parse_to_unix(dt_string, interval="2m"):
    """
    Convert CSV timestamp → Exact candle timestamp (Unix)
    Removes seconds, timezone, and rounds to nearest candle start.
    """

    if not dt_string:
        return int(datetime.now().timestamp())

    # Remove timezone if present
    dt_string = str(dt_string).replace("+05:30", "").strip()

    # Parse into datetime
    try:
        dt = datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S")
    except:
        try:
            dt = datetime.strptime(dt_string, "%Y-%m-%d %H:%M")
        except:
            return int(datetime.now().timestamp())

    # Remove seconds
    dt = dt.replace(second=0)

    # Determine frame size
    if interval == "2m":
        frame = 2
    elif interval == "15m":
        frame = 15
    else:
        frame = 1  # safe fallback

    # Round DOWN to nearest frame
    minute = (dt.minute // frame) * frame
    dt = dt.replace(minute=minute)

    # Localize to IST
    dt = IST.localize(dt)

    return int(dt.timestamp())



# ============================================================
# SAFE FLOAT
# ============================================================
def safe_float(x):
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except:
        return None


# ============================================================
# MAIN BACKEND SIGNAL GENERATOR (FINAL WORKING VERSION)
# ============================================================
def generate_signal_backend(symbol, df2, df15):

    symbol = symbol.upper()

    # ------------------------------------------------
    # FILTER ROWS FOR SYMBOL
    # ------------------------------------------------
    r2 = df2[
        (df2["script"].str.upper() == symbol)
        & (df2["Alert_details"].astype(str).str.strip() != "")
    ]

    r15 = df15[
        (df15["script"].str.upper() == symbol)
        & (df15["Alert_details"].astype(str).str.strip() != "")
    ]

    # ------------------------------------------------
    # PICK LATEST SIGNAL
    # ------------------------------------------------
    if not r2.empty:
        use = r2.iloc[-1]
    elif not r15.empty:
        use = r15.iloc[-1]
    else:
        return {"status": "error", "message": "No Signal Found"}

    # ------------------------------------------------
    # SAFE NUMERIC EXTRACTION
    # ------------------------------------------------
    close_price = safe_float(use.get("close_price"))
    pivot_upper = safe_float(use.get("pivot_upper"))
    pivot_lower = safe_float(use.get("pivot_lower"))
    resistance = safe_float(use.get("Resistance"))
    support = safe_float(use.get("Support"))

    if close_price is None : 
        return {"status": "error", "message": "Invalid numeric values in CSV"}

    # ------------------------------------------------
    # SIGNAL LOGIC (Based on pivot_upper)
    # ------------------------------------------------
    signal_type = signal_type = "BUY" if "Bullish" in str(use.get("Alert_details","")) else "SELL"

    # ------------------------------------------------
    # TIMESTAMP FIX
    # ------------------------------------------------
    if "timestamp" in use and not pd.isna(use["timestamp"]):
        raw = str(use["timestamp"])
    else:
        raw = str(use.get("Date", ""))

    interval = "2m" if "2" in str(use.get("data_interval", "")).lower() else "15m"
    ts = parse_to_unix(raw, interval)


    # ------------------------------------------------
    # PREPARE OUTPUT JSON
    # ------------------------------------------------
    result = {
        "status": "success",
        "symbol": symbol,
        "signal": signal_type,
        "price": close_price,
        "timestamp": ts,

        "resistance": resistance,
        "support": support,
        "pivot_upper": pivot_upper,
        "pivot_lower": pivot_lower,

        "interval": use.get("data_interval", "NA"),
        "screenerImplications": use.get("screener_implications", ""),
        "userAction": use.get("user_actions", ""),
        "alert": use.get("Alert_details", "")
    }

    # ------------------------------------------------
    # FIX JSON ⛔ NaN → VALID JSON
    # ------------------------------------------------
    for k, v in result.items():
        if isinstance(v, float) and (pd.isna(v) or v is None):
            result[k] = 0
        if v is None:
            result[k] = ""

    return result
