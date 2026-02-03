# backend/get_kite_token.py
import os
from urllib.parse import urlparse, parse_qs
from kiteconnect import KiteConnect

def dequote(s: str) -> str:
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    return s

API_KEY = dequote(os.getenv("KITE_API_KEY") or input("KITE_API_KEY: "))
API_SECRET = dequote(os.getenv("KITE_API_SECRET") or input("KITE_API_SECRET: "))

kite = KiteConnect(api_key=API_KEY)
print("Login URL:", kite.login_url())
print("\n1) Open the Login URL above, sign in.")
print("2) You'll be redirected to your app's redirect URL.")
print("3) COPY the FULL redirected URL from the address bar and paste it below.\n")

redirect_url = input("Paste FULL redirect URL here: ").strip()
# Example: https://your-redirect-url/?request_token=ABCD1234&action=login
qs = parse_qs(urlparse(redirect_url).query)
request_token = dequote((qs.get("request_token") or [""])[0])

if not request_token:
    raise SystemExit("Could not find request_token in the URL you pasted. Paste the FULL redirect URL.")

data = kite.generate_session(request_token, api_secret=API_SECRET)
access_token = data["access_token"]

print("\nACCESS_TOKEN:", access_token)
print("\nSet these in your terminal before starting Uvicorn (PowerShell):")
print(f'$env:KITE_API_KEY="{API_KEY}"')
print(f'$env:KITE_API_SECRET="{API_SECRET}"')
print(f'$env:KITE_ACCESS_TOKEN="{access_token}"')
