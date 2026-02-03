# app/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("DB_URL")
    or "sqlite:////data/paper_trading.db"
)

# Guard against accidental HTTP(S) URLs
if DATABASE_URL.startswith(("http://", "https://")):
    raise ValueError(
        "DATABASE_URL looks like an HTTP(S) URL. "
        "Use a proper DB URL, e.g.:\n"
        "  sqlite:///./app.db\n"
        "  postgresql+psycopg2://user:pass@host:5432/dbname\n"
        "  mysql+pymysql://user:pass@host:3306/dbname"
    )

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
