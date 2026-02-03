# backend/app/routers/auth.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sqlite3
import jwt  # Decoding Google JWT (signature not verified here — dev-only)
import requests

router = APIRouter(prefix="/auth", tags=["auth"])

# ✅ Correct database path
DB_PATH = "paper_trading.db"

# 📦 Pydantic models
class UserIn(BaseModel):
    username: str
    password: str

class UpdatePassword(BaseModel):
    username: str
    new_password: str

class UpdateEmail(BaseModel):
    username: str
    new_email: str

class GoogleToken(BaseModel):
    token: str

# ✅ Login route
@router.post("/login")
def login(user: UserIn):
    print("🔑 Login attempt:", user.username, user.password)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ? AND password = ?", (user.username, user.password))
    row = cur.fetchone()
    conn.close()

    if row:
        print("✅ Login success for:", user.username)
        return {"success": True, "username": user.username}
    else:
        print("❌ Invalid credentials:", user.username)
        return {"success": False, "message": "Invalid credentials"}


# ✅ Register route
@router.post("/register")
def register(user: UserIn):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (user.username,))
    if cur.fetchone():
        conn.close()
        return {"success": False, "message": "Username already exists"}

    cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user.username, user.password))
    conn.commit()
    conn.close()
    return {"success": True, "message": "User registered successfully"}

# ✅ Update password
@router.post("/update-password")
def update_password(data: UpdatePassword):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE users SET password = ? WHERE username = ?", (data.new_password, data.username))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Password updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Update email (rename username)
@router.post("/update-email")
def update_email(data: UpdateEmail):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE users SET username = ? WHERE username = ?", (data.new_email, data.username))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Email updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Google Login route
@router.post("/google-login")
def google_login(data: GoogleToken):
    try:
        idinfo = jwt.decode(data.token, options={"verify_signature": False})
        email = idinfo.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Invalid token")

        # Auto-register if not present
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (email,))
        if not cur.fetchone():
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (email, "google"))
            conn.commit()
        conn.close()

        return {"success": True, "username": email}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Google login failed: {str(e)}")
