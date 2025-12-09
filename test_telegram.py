# test_telegram.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print(f"Token: {TOKEN[:5]}...   Chat ID: {CHAT_ID}")

if not TOKEN or not CHAT_ID:
    print("❌ Error: Missing credentials in .env file")
else:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "🚨 SafeRide Test Alert: Connection Successful!"
    }
    
    try:
        resp = requests.post(url, json=payload)
        if resp.status_code == 200:
            print("✅ Success! Check your Telegram app.")
        else:
            print(f"❌ Failed. API Error: {resp.text}")
    except Exception as e:
        print(f"❌ Error: {e}")