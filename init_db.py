# Backend/init_db.py
import sqlite3

def init():
    conn = sqlite3.connect("/data/paper_trading.db")
    c = conn.cursor()

    # ✅ Create orders table
    c.execute("""
    CREATE TABLE IF NOT EXISTS orders (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      username   TEXT NOT NULL,
      script     TEXT NOT NULL,
      order_type TEXT NOT NULL,     -- BUY or SELL
      qty        INTEGER NOT NULL,
      price      REAL NOT NULL,
      exchange   TEXT NOT NULL,
      segment    TEXT NOT NULL,
      status     TEXT DEFAULT 'OPEN',
      datetime   TEXT NOT NULL,
      pnl        REAL DEFAULT 0.0
    )
    """)

    # ✅ Create users table with funds column
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      password TEXT NOT NULL,
      funds REAL DEFAULT 0.0
    )
    """)

    # 🔁 If `funds` column doesn't exist yet, try adding (safe fallback)
    try:
        c.execute("ALTER TABLE users ADD COLUMN funds REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # ✅ Create portfolio table
    c.execute("""
    CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    script TEXT,
    qty INTEGER,
    avg_price REAL,
    datetime TEXT,
    stoploss REAL,
    target REAL
    )
    """)

    # ✅ Create watchlist table
    c.execute("""
    CREATE TABLE IF NOT EXISTS watchlist (
      username TEXT NOT NULL,
      script TEXT NOT NULL,
      PRIMARY KEY(username, script)
    )
    """)

    # ✅ Create separate funds table
    c.execute("""
    CREATE TABLE IF NOT EXISTS funds (
      username TEXT PRIMARY KEY,
      total_amount REAL DEFAULT 0,
      available_amount REAL DEFAULT 0
    )
    """)

    # ✅ Create feedback table
    c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      message TEXT NOT NULL,
      datetime TEXT
    )
    """)

    # ✅ Create contact table
    c.execute("""
    CREATE TABLE IF NOT EXISTS contact (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      email TEXT NOT NULL,
      phone TEXT,
      subject TEXT,
      message TEXT,
      datetime TEXT
    )
    """)

    # ✅ Create closed_trades table
    c.execute("""
    CREATE TABLE IF NOT EXISTS closed_trades (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT,
      script TEXT,
      qty INTEGER,
      buy_price REAL,
      sell_price REAL,
      buy_time TEXT,
      sell_time TEXT,
      pnl REAL
    )
    """)

    # ✅ Safe migration for stoploss and target
    existing_columns = [col[1] for col in c.execute("PRAGMA table_info(orders)").fetchall()]
    
    if "stoploss" not in existing_columns:
        c.execute("ALTER TABLE orders ADD COLUMN stoploss REAL")
        print("✅ stoploss column added")

    if "target" not in existing_columns:
        c.execute("ALTER TABLE orders ADD COLUMN target REAL")
        print("✅ target column added")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init()
