# =============================================================
# app/fetch_full_instruments.py — fetch NSE + NFO (index + stock derivatives)
# =============================================================
import os
import pandas as pd
from kiteconnect import KiteConnect

INSTRUMENTS_PATH = "app/instruments.csv"

def fetch_full_instruments():
    try:
        print("📦 Fetching full instruments (NSE + NFO)...")
        kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
        kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))

        dfs = []
        for exch in ["NSE", "NFO"]:
            try:
                print(f"⏳ Downloading {exch} instruments...")
                data = kite.instruments(exch)
                df = pd.DataFrame(data)
                df["exchange"] = exch
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ Failed to load {exch}: {e}")

        if not dfs:
            print("❌ No data fetched!")
            return

        full = pd.concat(dfs, ignore_index=True)

        # ✅ Keep only relevant columns
        keep_cols = ["instrument_token", "exchange_token", "tradingsymbol", "name",
                     "last_price", "expiry", "strike", "instrument_type", "segment", "exchange"]
        full = full[keep_cols]

        # ✅ Save merged file
        full.to_csv(INSTRUMENTS_PATH, index=False)
        print(f"✅ Saved {len(full)} instruments to {INSTRUMENTS_PATH}")

        # ✅ Check if SUZLON exists
        has_suzlon = full[full["tradingsymbol"].str.contains("SUZLON", case=False, na=False)]
        if not has_suzlon.empty:
            print(f"✅ Found {len(has_suzlon)} SUZLON instruments (e.g. {has_suzlon.iloc[0]['tradingsymbol']})")
        else:
            print("⚠️ SUZLON not found — might not be active this expiry.")

    except Exception as e:
        print(f"⚠️ Error fetching instruments: {e}")

if __name__ == "__main__":
    fetch_full_instruments()
