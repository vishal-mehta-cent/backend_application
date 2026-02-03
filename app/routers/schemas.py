# app/schemas.py
from pydantic import BaseModel

class HistoryItem(BaseModel):
    time: str
    symbol: str
    buy_qty: int
    buy_price: float
    pnl: float
    remaining_qty: int
    is_closed: bool