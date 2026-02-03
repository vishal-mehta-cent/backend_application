# backend/seed_user.py

import sqlite3

conn = sqlite3.connect("paper_trading.db")
c = conn.cursor()

# Insert test user only if not already present
c.execute("SELECT * FROM users WHERE username = ?", ("testuser",))
if not c.fetchone():
    c.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        ("testuser", "testpass")
    )
    print("✅ Test user created: testuser / testpass")
else:
    print("ℹ️ Test user already exists.")

conn.commit()
conn.close()
