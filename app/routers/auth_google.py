# Backend/app/routers/auth_google.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests
import sqlite3

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_CLIENT_ID = "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com"  # Replace this

class GoogleToken(BaseModel):
    token: str

@router.post("/google-login")
async def google_login(data: GoogleToken):
    try:
        # ✅ Verify token from frontend
        idinfo = id_token.verify_oauth2_token(
            data.token, requests.Request(), GOOGLE_CLIENT_ID
        )

        email = idinfo.get("email")
        name = idinfo.get("name", "")
        sub = idinfo.get("sub")  # Unique Google ID

        if not email:
            raise HTTPException(status_code=400, detail="Invalid token: email missing")

        conn = sqlite3.connect("paper_trading.db")
        c = conn.cursor()

        # ✅ Ensure users table exists
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            email TEXT
        )""")

        # ✅ Register user if not already present
        c.execute("SELECT * FROM users WHERE username = ?", (email,))
        if not c.fetchone():
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                      (email, sub, email))
        conn.commit()
        conn.close()

        return {
            "success": True,
            "message": f"Google login successful for {email}",
            "username": email
        }

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid or expired Google token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google login failed: {str(e)}")