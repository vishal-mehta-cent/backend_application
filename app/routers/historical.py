# backend/app/routers/historical.py

import sqlite3
from collections import defaultdict
from typing import List, Dict, Any
import os

# Same DB path logic as orders.py
IS_RENDER = bool(os.getenv("RENDER"))

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if IS_RENDER:
    DB_PATH = "/data/paper_trading.db"
else:
    DB_PATH = os.path.join(BASE_DIR, "app", "data", "paper_trading.db")



def get_user_history(username: str) -> List[Dict[str, Any]]:
    """
    FIFO-based LONG trade history (BUY -> SELL):
    - Segment aware (intraday vs delivery)
    - Short aware: ignores SELL_FIRST / COVER legs so delivery carry doesn't leak into history
    """

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute(
        """
        SELECT id, datetime, script, order_type, qty, price,
               COALESCE(segment,'intraday') AS segment,
               COALESCE(is_short,0)        AS is_short,
               COALESCE(position_type,'')  AS position_type
          FROM orders
         WHERE username = ?
           AND status IN ('Closed','SETTLED')
         ORDER BY datetime ASC, id ASC
        """,
        (username,),
    )
    rows = c.fetchall()
    conn.close()

    inventory = defaultdict(list)  # key: (script, segment) -> FIFO lots
    history: List[Dict[str, Any]] = []

    for row in rows:
        script = (row["script"] or "").upper()
        side = (row["order_type"] or "").upper()
        qty = int(row["qty"] or 0)
        price = float(row["price"] or 0.0)
        dt = row["datetime"]
        segment = (row["segment"] or "intraday").lower()
        is_short = int(row["is_short"] or 0)
        ptype = (row["position_type"] or "").upper()

        if qty <= 0 or price <= 0:
            continue

        key = (script, segment)

        # Only LONG BUY lots feed inventory.
        # Ignore cover legs and short-related buys.
        if side == "BUY":
            if ptype == "COVER":
                continue
            if is_short == 1:
                continue
            inventory[key].append({"qty": qty, "price": price, "datetime": dt})
            continue

        # Only LONG SELL consumes inventory.
        if side == "SELL":
            if is_short == 1 or ptype == "SELL_FIRST":
                continue

            need = qty
            used = 0
            pnl_sum = 0.0
            buy_val_sum = 0.0
            buy_date = None

            while need > 0 and inventory[key]:
                lot = inventory[key][0]
                take = min(lot["qty"], need)

                if buy_date is None:
                    buy_date = lot["datetime"]

                used += take
                pnl_sum += (price - lot["price"]) * take
                buy_val_sum += lot["price"] * take

                lot["qty"] -= take
                need -= take
                if lot["qty"] == 0:
                    inventory[key].pop(0)

            if used == 0:
                continue

            avg_buy = round(buy_val_sum / used, 2)
            history.append({
                "symbol": script,
                "time": dt,
                "buy_qty": used,
                "buy_price": avg_buy,
                "buy_date": buy_date,
                "sell_qty": qty,
                "sell_avg_price": round(price, 2),
                "sell_date": dt,
                "invested_value": round(avg_buy * used, 2),
                "pnl": round(pnl_sum, 2),
                "segment": segment,
                "remaining_qty": 0,
                "is_closed": True,
            })

    return history
