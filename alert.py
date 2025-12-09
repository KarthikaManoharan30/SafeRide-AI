# alert.py

import os, time, requests
from datetime import datetime, timezone
from dotenv import load_dotenv
import datetime as dt

try:
    from zoneinfo import ZoneInfo  # py >= 3.9
except Exception:
    ZoneInfo = None

class AlertManager:
    def __init__(self, token: str | None = None, chat_id: str | None = None, cooldown_secs: int = 60):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.cooldown_secs = cooldown_secs
        self._last: dict[str, float] = {}

    def can_alert(self, key: str = "default") -> bool:
        now = time.time()
        last = self._last.get(key, 0.0)
        if now - last >= self.cooldown_secs:
            self._last[key] = now
            return True
        return False

    def send_text(self, text: str) -> bool:
        if not (self.token and self.chat_id):
            return False
        r = requests.post(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            data={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.ok

    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        if not (self.token and self.chat_id):
            return False
        with open(photo_path, "rb") as f:
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendPhoto",
                data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"},
                files={"photo": f},
                timeout=20,
            )
        return r.ok

def max_accident_conf(result, class_names: dict[int, str]):
    """Ultralytics Results -> return max confidence for 'accident' boxes, or None."""
    scores = [
        float(b.conf[0].item())
        for b in result.boxes
        if class_names.get(int(b.cls[0].item())) == "accident"
    ]
    return max(scores) if scores else None

def now_str():
    """Current time string in local timezone (LOCAL_TZ) or UTC."""
    tz = os.getenv("LOCAL_TZ")
    if tz and ZoneInfo:
        return dt.datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M:%S %Z")
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")