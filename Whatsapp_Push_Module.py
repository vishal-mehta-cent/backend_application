import pandas as pd
import time
import os
import platform
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from datetime import datetime

# ============================
# UNIVERSAL PATHS (LOCAL + RENDER)
# ============================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.abspath(os.path.join(BASE_PATH, ".."))
CONTROL_PATH = os.path.join(BACKEND_ROOT, "Control_files_2")
PUSH_FILE = os.path.join(CONTROL_PATH, "1_Intraday_anomaly_push_signals.csv")

os.makedirs(CONTROL_PATH, exist_ok=True)

print("📌 USING UNIVERSAL PATHS")
print("BASE_PATH:", BASE_PATH)
print("BACKEND_ROOT:", BACKEND_ROOT)
print("CONTROL_PATH:", CONTROL_PATH)
print("PUSH_FILE:", PUSH_FILE)


# ============================
# CSV UPDATE FUNCTION
# ============================

def update_push_csv(script):
    csv_path = PUSH_FILE

    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=["Date", "Script"])
        df.to_csv(csv_path, index=False)

    try:
        df = pd.read_csv(csv_path)
    except:
        df = pd.DataFrame(columns=["Date", "Script"])

    new_row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Script": script
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)

    print("✅ CSV updated successfully!")


# ============================
# SEND WHATSAPP MESSAGE (LOCAL + RENDER)
# ============================

def whatsapp_push_execution(message_list, number="6358801256"):

    if not message_list:
        print("No new messages to send.")
        return

    number = str(number).replace(" ", "").replace("+", "")
    full_number = "91" + number

    options = Options()

    # Detect OS for Profile Path
    if platform.system().lower() == "windows":
        USER_DATA_DIR = r"C:\Users\Neurocrest\selenium_whatsapp_profile"
    else:
        USER_DATA_DIR = "/opt/render/project/.chrome-profile"

    options.add_argument(f"--user-data-dir={USER_DATA_DIR}")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--remote-debugging-port=9222")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    driver.get(f"https://web.whatsapp.com/send?phone={full_number}")
    wait = WebDriverWait(driver, 60)

    try:
        inp = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//div[@contenteditable='true'][@data-tab='10']")
        ))
    except:
        print("❌ WhatsApp chat input not found — login issue?")
        return

    for msg in message_list:
        for line in msg.split("\n"):
            inp.send_keys(line)
            ActionChains(driver).key_down(Keys.SHIFT).key_down(Keys.ENTER)\
                               .key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()
        inp.send_keys(Keys.ENTER)
        time.sleep(1)

    print("✅ WhatsApp messages sent successfully!")


# ============================
# MAIN FUNCTION CALLED BY BACKEND
# ============================

def run_whatsapp_alert(symbol):

    msg = f"""
Manual WhatsApp Alert
Script: {symbol}
Triggered from Chart Page
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    whatsapp_push_execution([msg])
    update_push_csv(symbol)

    return {
        "status": "success",
        "message": "WhatsApp alert sent & CSV updated",
        "symbol": symbol
    }
