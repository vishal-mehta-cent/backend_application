import sqlite3

conn = sqlite3.connect("paper_trading.db")
c = conn.cursor()

try:
    c.execute("ALTER TABLE orders ADD COLUMN stoploss REAL")
    c.execute("ALTER TABLE orders ADD COLUMN target REAL")
    print("✅ Columns added successfully.")
except Exception as e:
    print("⚠️ Migration skipped or already done:", e)

conn.commit()
conn.close()
