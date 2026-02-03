# check_portfolio.py
import sqlite3

conn = sqlite3.connect("paper_trading.db")
cur = conn.cursor()

# Optional: Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS portfolio (
    username TEXT,
    script TEXT,
    qty INTEGER,
    avg_buy_price REAL,
    current_price REAL
)
""")

# Add test row
cur.execute("""
INSERT INTO portfolio (username, script, qty, avg_buy_price, current_price)
VALUES (?, ?, ?, ?, ?)
""", ("Vishal Mehta", "RELIANCE", 10, 2500.0, 2580.0))

conn.commit()
conn.close()
print("✅ Dummy portfolio inserted.")
