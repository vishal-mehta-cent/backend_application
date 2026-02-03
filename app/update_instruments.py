# =============================================================
# app/update_instruments.py — Auto-refresh instruments.csv daily
# =============================================================
import os
import requests
import pandas as pd

INSTRUMENTS_PATH = "app/instruments.csv"

def update_instruments():
    try:
        print("📦 Fetching latest Zerodha instruments list...")
        url = "https://api.kite.trade/instruments"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        content = response.content.decode("utf-8")
        rows = [line.split(",") for line in content.splitlines()]
        header = rows[0]
        df = pd.DataFrame(rows[1:], columns=header)

        df.to_csv(INSTRUMENTS_PATH, index=False)
        print(f"✅ Updated {INSTRUMENTS_PATH} with {len(df)} instruments.")
    except Exception as e:
        print(f"⚠️ Failed to update instruments file: {e}")
