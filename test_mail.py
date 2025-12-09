# test_mail.py
from emailer import send_email
from dotenv import load_dotenv
import os

load_dotenv()

to = [os.getenv("ALERT_EMAIL_TO")]
print(f"Attempting to send to: {to}")

try:
    send_email(
        subject="Test Email from SafeRide",
        body="If you read this, credentials work!",
        to_emails=to
    )
    print("✅ Success! Email sent.")
except Exception as e:
    print(f"❌ Failed: {e}")