from fastapi import FastAPI
import os
from pathlib import Path

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "message": "Dummy backend live ✅"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug/disk")
def debug_disk():
    data_dir = Path("/data")
    return {
        "PORT": os.getenv("PORT"),
        "/data_exists": data_dir.exists(),
        "files": [p.name for p in data_dir.glob("*")] if data_dir.exists() else [],
    }
