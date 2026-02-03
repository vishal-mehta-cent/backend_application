# insert_dummy_portfolio.py

import sqlite3

conn = sqlite3.connect("paper_trading.db")
cur = conn.cursor()

# Ensure table exists
cur.execute("""
CREATE TABLE IF NOT EXISTS portfolio (
    username TEXT,
    script TEXT,
    qty INTEGER,
    avg_buy_price REAL,
    current_price REAL
)
""")

# Insert sample rows
rows = [
    ("Vishal Mehta", "RELIANCE", 10, 2450.0, 2520.0),
    ("Vishal Mehta", "TCS", 5, 3500.0, 3600.0),
    ("Vishal Mehta", "INFY", 8, 1450.0, 1420.0),
]

cur.executemany("INSERT INTO portfolio VALUES (?, ?, ?, ?, ?)", rows)

conn.commit()
conn.close()
print("✅ Dummy portfolio data inserted for Vishal Mehta.")
