# Backend/app/routers/auth_google.py
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests
import sqlite3

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_CLIENT_ID = "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com"  # Replace

class GoogleToken(BaseModel):
    token: str

@router.post("/google-login")
async def google_login(data: GoogleToken):
    token = data.get("token")
    try:
        idinfo = id_token.verify_oauth2_token(
            data.token, requests.Request(), GOOGLE_CLIENT_ID
        )

        email = idinfo["email"]
        name = idinfo.get("name", "")
        sub = idinfo["sub"]  # Google unique user ID

        conn = sqlite3.connect("/data/paper_trading.db")
        c = conn.cursor()

        # Create table if not exists
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            email TEXT
        )""")

        # Register user if not exists
        c.execute("SELECT * FROM users WHERE username = ?", (email,))
        if not c.fetchone():
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                      (email, sub, email))
        conn.commit()
        conn.close()

        return {"status": "success", "username": email}

    except Exception as e:
        raise HTTPException(status_code=401, detail="Google login failed")
