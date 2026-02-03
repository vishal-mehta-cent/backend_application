# app/services/history.py
from typing import List, Dict
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel

class HistoryItem(BaseModel):
    time: str
    symbol: str
    buy_qty: int
    buy_price: float
    pnl: float
    remaining_qty: int
    is_closed: bool

def _fmt_time_ist(dt: datetime) -> str:
    # Treat naive as UTC; convert to IST (UTC+5:30)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ist = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
    return ist.strftime("%H:%M")

def build_history(username: str, orders: List) -> List[HistoryItem]:
    # Consider only filled orders
    filled = [o for o in orders if getattr(o, "status", "Filled") == "Filled"]
    filled.sort(key=lambda o: o.created_at)

    # Per-symbol FIFO lots
    symbol_lots: Dict[str, List[dict]] = {}

    for o in filled:
        side = o.side.upper()
        sym = o.symbol
        symbol_lots.setdefault(sym, [])

        if side == "BUY":
            symbol_lots[sym].append({
                "time": _fmt_time_ist(o.created_at),
                "symbol": sym,
                "buy_qty": int(o.qty),
                "buy_price": float(o.price),
                "remaining_qty": int(o.qty),
                "realized_pnl": 0.0,
            })
        elif side == "SELL":
            sell_qty_left = int(o.qty)
            sell_price = float(o.price)
            for lot in symbol_lots[sym]:
                if sell_qty_left <= 0:
                    break
                if lot["remaining_qty"] <= 0:
                    continue
                take = min(lot["remaining_qty"], sell_qty_left)
                lot["realized_pnl"] += (sell_price - lot["buy_price"]) * take
                lot["remaining_qty"] -= take
                sell_qty_left -= take
            # If sell_qty_left > 0, ignore (short not handled)

    items: List[HistoryItem] = []
    for sym, lots in symbol_lots.items():
        for lot in lots:
            items.append(HistoryItem(
                time=lot["time"],
                symbol=sym,
                buy_qty=lot["buy_qty"],
                buy_price=lot["buy_price"],
                pnl=round(lot["realized_pnl"], 2),
                remaining_qty=lot["remaining_qty"],
                is_closed=(lot["remaining_qty"] == 0),
            ))

    # newest first (optional)
    items.sort(key=lambda x: x.time, reverse=True)
    return items
