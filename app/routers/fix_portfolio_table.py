import sqlite3

conn = sqlite3.connect("paper_trading.db")
c = conn.cursor()

# Drop old table if exists
c.execute("DROP TABLE IF EXISTS portfolio")

# Create correct table without current_price
c.execute("""
CREATE TABLE portfolio (
    username TEXT NOT NULL,
    script TEXT NOT NULL,
    qty INTEGER NOT NULL,
    avg_buy_price REAL NOT NULL
)
""")

conn.commit()
conn.close()

print("✅ portfolio table has been reset correctly.")
