# backend/init_scripts.py
import csv
from sqlalchemy.orm import Session
from app.models import Script
from app.database import SessionLocal

def load_scripts_from_csv():
    db: Session = SessionLocal()
    with open("app/data/nse_equity_scripts.csv", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get("symbol") or row.get("Symbol")
            name = row.get("name") or row.get("Name")
            sector = row.get("sector") or row.get("Sector") or "Unknown"
            if symbol and name:
                script = Script(
                    symbol=symbol.strip().upper(),
                    name=name.strip(),
                    sector=sector.strip()
                )
                db.add(script)
        db.commit()
        db.close()

if __name__ == "__main__":
    load_scripts_from_csv()
