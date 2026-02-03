# backend/reset_users_table.py

import sqlite3

conn = sqlite3.connect("paper_trading.db")
c = conn.cursor()

# Drop users table if exists
c.execute("DROP TABLE IF EXISTS users")

# Recreate users table with correct columns
c.execute("""
CREATE TABLE IF NOT EXISTS users (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
)
""")

conn.commit()
conn.close()
print("✅ users table reset with username and password columns.")
