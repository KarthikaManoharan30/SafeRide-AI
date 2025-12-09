
# emailer.py
import os, smtplib, mimetypes
from email.message import EmailMessage
import boto3

def send_email_ses(subject, body, to_emails, attachments=None):
    # Needs AWS creds + SES verified sender
    ses = boto3.client("ses", region_name=os.getenv("AWS_REGION"))
    sender = os.getenv("SES_SENDER")  # e.g. "alerts@example.com"
    if not sender: raise RuntimeError("SES_SENDER not set")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)

    for path in attachments or []:
        ctype, _ = mimetypes.guess_type(path)
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        with open(path, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(path))

    ses.send_raw_email(RawMessage={"Data": msg.as_string()})
    return True

def send_email_smtp(subject, body, to_emails, attachments=None):
    host = os.getenv("SMTP_HOST"); port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER"); pwd = os.getenv("SMTP_PASS")
    sender = os.getenv("SMTP_SENDER", user)
    if not all([host, port, user, pwd, sender]):
        raise RuntimeError("SMTP env not set")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)

    for path in (attachments or []):
        if not path or not os.path.exists(path):
            continue

        
        # This code should run for every valid attachment
        ctype, _ = mimetypes.guess_type(path)
        # Use a default MIME type if guessing fails
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        
        with open(path, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(path))

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)

def send_email(subject, body, to_emails, attachments=None):
    # Choose SES if SES_SENDER is set, else SMTP
    if os.getenv("SES_SENDER"):
        return send_email_ses(subject, body, to_emails, attachments)
    return send_email_smtp(subject, body, to_emails, attachments)