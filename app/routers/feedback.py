# backend/app/routers/feedback.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import sqlite3
from datetime import datetime
import resend
import os
from dotenv import load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

router = APIRouter(prefix="/feedback", tags=["Feedback"])

# ----------------------------
# ✅ Resend API configuration
# ----------------------------
resend.api_key = os.getenv("RESEND_API_KEY") or "re_UNhfhccS_KR7wEBEnizmdZ47J6FRKvMBt"

RECEIVER_EMAIL = "neurocrest.app@gmail.com"
DB_PATH = "app/data/paper_trading.db"

# ----------------------------
# Pydantic Models
# ----------------------------
class FeedbackForm(BaseModel):
    name: str = Field(..., example="Jinal")
    message: str = Field(..., example="Great App!")


class ContactForm(BaseModel):
    name: str = Field(..., example="Jinal")
    email: str = Field(..., example="gandhijinal1394@gmail.com")
    phone: str = Field(..., example="9054073750")
    subject: str = Field(..., example="Trade")
    message: str = Field(..., example="Nice App")


# ----------------------------
# Helper: Send email via Resend
# ----------------------------
def send_email(subject: str, body: str):
    """Send email using Resend"""
    try:
        print("📤 Sending email...")

        response = resend.Emails.send({
            # ✅ Use verified domain sender — onboarding@resend.dev
            "from": "Neurocrest <onboarding@resend.dev>",
            "to": ["jinal.neurocrest@gmail.com"],
            "subject": subject,
            "text": body,
        })

        print("✅ Email send response:", response)
        return response
    except Exception as e:
        print("❌ Email send failed:", e)
        raise HTTPException(status_code=500, detail=f"Email send failed: {e}")


# ----------------------------
# Feedback endpoint
# ----------------------------
@router.post("/submit")
def submit_feedback(data: FeedbackForm):
    """Handles Feedback form submission"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                message TEXT,
                datetime TEXT
            )
        """)
        c.execute(
            "INSERT INTO feedback (name, message, datetime) VALUES (?, ?, ?)",
            (data.name, data.message, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        conn.close()

        subject = f"New Feedback from {data.name}"
        body = f"Name: {data.name}\n\nFeedback:\n{data.message}"

        send_email(subject, body)
        return {"success": True, "message": "Feedback submitted successfully."}

    except Exception as e:
        print("❌ Feedback submission error:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Contact endpoint
# ----------------------------
@router.post("/contact")
def submit_contact(data: ContactForm):
    """Handles Contact form submission"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS contact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                phone TEXT,
                subject TEXT,
                message TEXT,
                datetime TEXT
            )
        """)
        c.execute(
            "INSERT INTO contact (name, email, phone, subject, message, datetime) VALUES (?, ?, ?, ?, ?, ?)",
            (
                data.name,
                data.email,
                str(data.phone),
                data.subject,
                data.message,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        conn.close()

        subject = f"New Contact Message from {data.name}"
        body = (
            f"Name: {data.name}\n"
            f"Email: {data.email}\n"
            f"Phone: {data.phone}\n"
            f"Subject: {data.subject}\n\n"
            f"Message:\n{data.message}"
        )

        send_email(subject, body)
        return {"success": True, "message": "Contact message sent successfully."}

    except Exception as e:
        print("❌ Contact submission error:", e)
        raise HTTPException(status_code=500, detail=str(e))
