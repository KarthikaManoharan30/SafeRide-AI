# SafeRide: Accident & Helmet Detection
# - Image + Video pipelines
# - Accident -> Telegram (best frame) + Email (Critical Alert)
# - No-helmet -> Email (with Markdown + JSON reports)
# - Logs to RDS: detections + violations
# - Pre-signed S3 links in UI/alerts/emails

import requests  # <--- Essential for Direct Telegram
import os, uuid, json, tempfile
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
import datetime as dt

import streamlit as st
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from psycopg2.extras import Json
import cv2
import boto3
import uuid

from emailer import send_email
from alert import AlertManager, now_str, max_accident_conf
import rag_ingest
from reporter import build_incident_pdf
import base64
from pathlib import Path
import chat_agents_1 as CA

# -------------------- ENV --------------------
load_dotenv()  # read .env

# -------------------- CONFIGURATION --------------------
# Define what the model calls a "violation". 
# Add your model's specific class name here if it's different (e.g., "head", "person_no_helmet")
VIOLATION_CLASSES = ["nohelmet", "no-helmet", "no_helmet", "without_helmet"]

# -------------------- STREAMLIT PAGE CONFIG --------------------
st.set_page_config(page_title="SafeRide Detector", layout="wide")
st.title("🚦 SafeRide: Accident / Helmet / No-Helmet Detection")


# -------------------- DB SETUP --------------------

def ensure_database():
    host = os.getenv("RDS_HOST")
    port = os.getenv("RDS_PORT", "5432")
    user = os.getenv("RDS_USER")
    pwd  = os.getenv("RDS_PASSWORD")
    db   = os.getenv("RDS_DB", "postgresdb")
    if not all([host, port, user, pwd, db]):
        st.error("Database configuration is incomplete. Please check your .env file.")
        return

    admin_url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/postgres"
    engine_admin = create_engine(admin_url, isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    with engine_admin.connect() as conn:
        exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname=:d"), {"d": db}).scalar()
        if not exists:
            conn.execute(text(f"CREATE DATABASE {db}"))
            st.success(f"Database created: {db}")
        else:
            st.info(f"Database already exists: {db}")

with st.expander("🗄️ Database setup"):
    if st.button("Ensure database exists"):
        try:
            ensure_database()
        except Exception as e:
            st.error(f"DB create failed: {e}")

from dotenv import load_dotenv
load_dotenv() # Force load the .env file

print("DEBUG: User:", os.getenv('RDS_USER'))
print("DEBUG: Host:", os.getenv('RDS_HOST'))

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('RDS_USER')}:{os.getenv('RDS_PASSWORD')}@"
    f"{os.getenv('RDS_HOST')}:{os.getenv('RDS_PORT','5432')}/{os.getenv('RDS_DB')}"
)
engine = create_engine(DB_URL, pool_pre_ping=True)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
  id UUID PRIMARY KEY,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
  run_id TEXT,
  source_type TEXT CHECK (source_type IN ('image','video')),
  source_name TEXT,
  frame_index INT,
  class_id INT NOT NULL,
  class_name TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  bbox_xyxy JSONB NOT NULL,
  bbox_norm JSONB NOT NULL,
  s3_uri TEXT
);
"""

def create_table_if_needed():
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))

#CREATE violations table (if not exists)
CREATE_VIOLATIONS_SQL = """
CREATE TABLE IF NOT EXISTS violations (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_id TEXT,
    source_type TEXT CHECK (source_type IN ('image','video')),
    source_name TEXT,
    frame_index INT,
    class_name TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    bbox_xyxy JSONB NOT NULL,
    bbox_norm JSONB NOT NULL,
    s3_image_uri TEXT,
    s3_report_uri TEXT,
    email_sent BOOLEAN NOT NULL DEFAULT FALSE
);
"""
def create_violations_table_if_needed():
    with engine.begin() as conn:
        conn.execute(text(CREATE_VIOLATIONS_SQL))

col1, col2, col3 = st.columns(3)

if col1.button("Health check", key="btn_healthcheck"):
    try:
        with engine.begin() as conn:
            dbname, ts = conn.execute(text("SELECT current_database(), now()")).fetchone()
        st.success(f"DB OK ✅ {dbname} @ {ts}")
    except Exception as e:
        st.error(f"DB error: {e}")

if col2.button("Create detections table", key="btn_create_detections"):
    try:
        create_table_if_needed()
        st.success("detections table ensured ✅")
    except Exception as e:
        st.error(f"Failed to ensure detections: {e}")

if col3.button("Create violations table", key="btn_create_violations"):
    try:
        create_violations_table_if_needed()
        st.success("violations table ensured ✅")
    except Exception as e:
        st.error(f"Failed to ensure violations: {e}")


def _stringify_uuids(obj):
    """Recursively convert UUIDs (and datetimes) into JSON-safe strings."""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _stringify_uuids(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_uuids(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_stringify_uuids(v) for v in obj)
    return obj

with st.expander("Reports & RAG"):
    idx_col1, idx_col2 = st.columns([2,1])

    idx_dir_default = os.getenv("RAG_INDEX_DIR", "indexes/faiss_reports")
    rag_prefix_default = os.getenv("RAG_REPORT_PREFIX", "runs/")

    idx_dir = idx_col1.text_input("FAISS index dir", value=idx_dir_default, key="rag_idx_dir")
    rag_prefix = idx_col1.text_input("S3 report prefix", value=rag_prefix_default, key="rag_s3_prefix")

    if idx_col2.button("Build/Refresh Index", key="btn_build_index"):
        try:
            out_dir = rag_ingest.build_or_update_index(
                s3_prefix=rag_prefix,
                index_dir=idx_dir
            )
            st.success(f"RAG index refreshed ✅ → {out_dir}")
        except Exception as e:
            st.error(f"Failed to build index: {e}")        


# -------------------- S3 + HELPERS --------------------
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET  = os.getenv("S3_BUCKET")
s3 = boto3.client("s3", region_name=AWS_REGION)


def _parse_seconds(val: str, default: int = 3600) -> int:
    if not val:
        return default
    s = val.strip().split()[0]  # drop inline comments
    # allow suffixes like 24h, 15m, 30s
    try:
        if s.endswith(("h","H")):  return int(float(s[:-1]) * 3600)
        if s.endswith(("m","M")):  return int(float(s[:-1]) * 60)
        if s.endswith(("s","S")):  return int(float(s[:-1]))
        return int(float(s))
    except Exception:
        return default

PRESIGN_EXPIRES = _parse_seconds(os.getenv("PRESIGN_EXPIRES", "3600"))

def upload_to_s3(local_path: str, key_prefix: str = "runs") -> str:
    # tz-aware UTC timestamp in key path
    key = f"{key_prefix}/{dt.datetime.now(dt.timezone.utc):%Y/%m/%d}/{uuid.uuid4().hex}_{Path(local_path).name}"
    s3.upload_file(local_path, S3_BUCKET, key)
    return f"s3://{S3_BUCKET}/{key}"

def presign_s3_uri(s3_uri: str, expires: int = PRESIGN_EXPIRES) -> str:
    """s3://bucket/key -> temporary HTTPS link; falls back to original on error."""
    try:
        if not s3_uri or not s3_uri.startswith("s3://"):
            return s3_uri
        bucket, key = s3_uri[5:].split("/", 1)
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
    except Exception:
        return s3_uri

def write_md_report(path: str, title: str, fields: dict, image_s3=None, video_s3=None):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for k, v in fields.items():
        lines.append(f"**{k}:** {v}")
    if image_s3: lines += ["", f"**Image:** {image_s3}"]
    if video_s3: lines += ["", f"**Video:** {video_s3}"]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_json_report(path: str, payload: dict):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

# -------------------- INSERT HELPERS --------------------
def insert_detection_row(engine, row: dict):
    payload = {
        "id": row["id"],
        "run_id": row.get("run_id"),
        "source_type": row["source_type"],
        "source_name": row.get("source_name"),
        "frame_index": row.get("frame_index"),
        "class_id": row["class_id"],
        "class_name": row["class_name"],
        "confidence": row["confidence"],
        "s3_uri": row.get("s3_uri"),
        "bbox_xyxy": Json(row["bbox_xyxy"] if isinstance(row["bbox_xyxy"], dict) else json.loads(row["bbox_xyxy"])),
        "bbox_norm": Json(row["bbox_norm"] if isinstance(row["bbox_norm"], dict) else json.loads(row["bbox_norm"])),
    }
    sql = text("""
      INSERT INTO detections
        (id, run_id, source_type, source_name, frame_index,
         class_id, class_name, confidence, bbox_xyxy, bbox_norm, s3_uri)
      VALUES
        (:id, :run_id, :source_type, :source_name, :frame_index,
         :class_id, :class_name, :confidence, :bbox_xyxy, :bbox_norm, :s3_uri)
    """)
    with engine.begin() as conn:
        conn.execute(sql, payload)

def insert_violations_row(engine, row: dict):
    sql = text("""
      INSERT INTO violations
        (id, run_id, source_type, source_name, frame_index,
         class_name, confidence, bbox_xyxy, bbox_norm, s3_image_uri, s3_report_uri, email_sent)
      VALUES
        (:id, :run_id, :source_type, :source_name, :frame_index,
         :class_name, :confidence, :bbox_xyxy, :bbox_norm, :s3_image_uri, :s3_report_uri, :email_sent)
    """)
    payload = {
        "id": row["id"],
        "run_id": row.get("run_id"),
        "source_type": row["source_type"],
        "source_name": row.get("source_name"),
        "frame_index": row.get("frame_index"),
        "class_name": row["class_name"],
        "confidence": row["confidence"],
        "bbox_xyxy": Json(row["bbox_xyxy"]),
        "bbox_norm": Json(row["bbox_norm"]),
        "s3_image_uri": row.get("s3_image_uri"),
        "s3_report_uri": row.get("s3_report_uri"),
        "email_sent": bool(row.get("email_sent", False)),
    }
    try:
        with engine.begin() as conn:
            conn.execute(sql, payload)
    except Exception:
        pass

# -------------------- UI --------------------

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
if "_model_path" not in st.session_state or st.session_state["_model_path"] != MODEL_PATH:
    st.session_state["_model"] = YOLO(MODEL_PATH)
    st.session_state["_model_path"] = MODEL_PATH
model = st.session_state["_model"]
names = model.model.names
st.caption(f"Model loaded: {MODEL_PATH}")

# --- DEBUG: Show user exactly what the model classes are ---
with st.expander("🛠️ Debug: Check Model Classes"):
    st.write("These are the EXACT names your model knows:")
    st.json(names)
    st.info(f"The code matches these against 'nohelmet' violations: {VIOLATION_CLASSES}")


# -------------------- SIDEBAR CONTROLS --------------------
st.sidebar.header("⚙️ Controls")

pred_conf = st.sidebar.slider("Prediction confidence", 0.05, 0.75, 0.20, 0.01)
iou = st.sidebar.slider("NMS IoU", 0.10, 0.90, 0.60, 0.05)
acc_alert = st.sidebar.slider("Accident alert threshold", 0.05, 0.90, 0.20, 0.01)
acc_min_area = st.sidebar.slider("Accident min box area (%)", 0.1, 5.0, 1.0, 0.1) / 100.0
persist_k = st.sidebar.slider("Video alert persistence (frames)", 1, 10, 3, 1)
nohelmet_alert = st.sidebar.slider("Helmet violation email threshold", 0.05, 0.90, 0.20, 0.01)

alert_mgr = AlertManager(cooldown_secs=int(os.getenv("ALERT_COOLDOWN_SECS", "60")))


# -------------------- TABS --------------------
tab_detect, tab_chat = st.tabs(["🧪 Detect", "💬 Chat"])

with tab_detect:
    st.subheader("Run detection on image or video")
    mode = st.radio("Select input type", ["Image", "Video"], horizontal=True)

    def _norm(s: str) -> str:
        return s.replace(" ", "").replace("-", "").replace("_", "").lower()

# -------------------- IMAGE TAB --------------------
    if mode == "Image":
        f = st.file_uploader("Upload an image", type=["jpg","jpeg","png"],key= "image_uploader")
        if f:
            img = Image.open(f).convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img.save(tmp.name)

            # Primary inference (INCREASED IMG SIZE TO 1280 FOR BETTER DETECTION)
            res = model.predict(source=tmp.name, conf=pred_conf, iou=iou, imgsz=1280, save=False)[0]
            plotted_bgr = res.plot(line_width=3)
            st.image(plotted_bgr[:, :, ::-1], caption="Detections", width=900)

            # Save annotated still
            out_path = Path(tmp.name).with_suffix(".annot.jpg")
            Image.fromarray(plotted_bgr[:, :, ::-1]).save(out_path)

            # Upload
            # --- FIX: Prevent re-sending alerts on Chat refresh ---
            # Create a unique key for this specific file
            file_key = f"{f.name}_{f.size}"
            
            # If this is a new file, generate a new ID and reset the alert flag
            if "last_file_key" not in st.session_state or st.session_state["last_file_key"] != file_key:
                st.session_state["last_file_key"] = file_key
                st.session_state["fixed_run_id"] = uuid.uuid4().hex
                st.session_state["alert_already_sent"] = False

            run_id = st.session_state["fixed_run_id"]
            # ------------------------------------------------------

            s3_uri = upload_to_s3(str(out_path), key_prefix=f"runs/{run_id}/images")
            st.success(f"Snapshot saved to S3: {presign_s3_uri(s3_uri)}")

            # Logging + alerts
            H, W = res.orig_shape
            acc_scores = []
            email_to = [e.strip() for e in os.getenv("ALERT_EMAIL_TO","").split(",") if e.strip()]

            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                raw_name = names.get(cls_id, "unknown")
                cls_norm = _norm(raw_name)
                confv  = float(b.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                area_frac = ((x2 - x1) * (y2 - y1)) / (W * H)
                
                # --- CONSOLE DEBUGGING ---
                print(f"DEBUG: Found class '{raw_name}' (norm: {cls_norm}) with conf {confv:.2f}")

                if cls_norm == "accident" and area_frac >= acc_min_area:
                    acc_scores.append(confv)

                # log every detection
                insert_detection_row(engine, {
                    "id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "source_type": "image",
                    "source_name": f.name,
                    "frame_index": None,
                    "class_id": cls_id,
                    "class_name": raw_name,
                    "confidence": confv,
                    "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "bbox_norm": {"x1": x1/W, "y1": y1/H, "x2": x2/W, "y2": y2/H},
                    "s3_uri": s3_uri,
                })

                # Check if this class is in our violation list
                is_violation = (cls_norm in VIOLATION_CLASSES)

                if is_violation and confv >= nohelmet_alert:
                    md_path   = str(out_path.with_suffix(".md"))
                    json_path = str(out_path.with_suffix(".json"))

                    md_fields = {
                        "Time": now_str(),
                        "Run ID": run_id,
                        "Source": f.name,
                        "Class": raw_name,
                        "Confidence": f"{confv:.2f}",
                    }
                    write_md_report(md_path, "Helmet Violation", md_fields, image_s3=s3_uri)
                    write_json_report(json_path, {
                        "timestamp": now_str(),
                        "run_id": run_id,
                        "source_type": "image",
                        "source_name": f.name,
                        "frame_index": None,
                        "class_name": "nohelmet",
                        "confidence": confv,
                        "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "bbox_norm": {"x1": x1/W, "y1": y1/H, "x2": x2/W, "y2": y2/H},
                        "image_s3_uri": s3_uri,
                    })

                    s3_report_md   = upload_to_s3(md_path,   key_prefix=f"runs/{run_id}/reports")
                    s3_report_json = upload_to_s3(json_path, key_prefix=f"runs/{run_id}/reports")
                    # build PDF from the same fields + annotated image
                    pdf_path = str(out_path.with_suffix(".pdf"))
                    build_incident_pdf(pdf_path, "Helmet Violation", md_fields, image_path=str(out_path))
                    s3_report_pdf = upload_to_s3(pdf_path, key_prefix=f"runs/{run_id}/reports")

                    if email_to:
                        subject = f"[SafeRide] Helmet violation • {confv:.2f}"
                        body = (
                            f"Time: {now_str()}\n"
                            f"Run: {run_id}\n"
                            f"Source: {f.name}\n"
                            f"Confidence: {confv:.2f}\n"
                            f"S3 image: {presign_s3_uri(s3_uri)}\n"
                            f"Report (MD): {presign_s3_uri(s3_report_md)}\n"
                            f"Report (JSON): {presign_s3_uri(s3_report_json)}\n"
                            f"Report (PDF): {presign_s3_uri(s3_report_pdf)}\n"
                        )
                        send_email(subject, body, email_to, attachments=None)

                    insert_violations_row(engine, {
                        "id": str(uuid.uuid4()),
                        "run_id": run_id,
                        "source_type": "image",
                        "source_name": f.name,
                        "frame_index": None,
                        "class_name": "nohelmet",
                        "confidence": confv,
                        "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "bbox_norm": {"x1": x1/W, "y1": y1/H, "x2": x2/W, "y2": y2/H},
                        "s3_image_uri": s3_uri,
                        "s3_report_uri": s3_report_md,
                        "email_sent": True
                    })

            # ------------------------------------------------------------------
            # IMAGE ACCIDENT ALERT (Direct Telegram + Email)
            # ------------------------------------------------------------------
            acc_max = max(acc_scores) if acc_scores else None
            
            if acc_max is not None and acc_max >= acc_alert:
                st.error(f"🚨 Accident detected! (conf {acc_max:.2f} ≥ {acc_alert:.2f})")
                
                # --- FIX: Only send if we haven't sent it yet ---
                if not st.session_state.get("alert_already_sent", False):
                    # 1. SEND TELEGRAM (DIRECT - BYPASSING ALERTMANAGER)
                    try:
                        t_token = os.getenv("TELEGRAM_BOT_TOKEN")
                        t_chat = os.getenv("TELEGRAM_CHAT_ID")
                        if t_token and t_chat:
                            caption = (
                                f"🚨 Accident Detected\n"
                                f"Conf: {acc_max:.2f}\n"
                                f"Time: {now_str()}\n"
                                f"Run: {run_id}"
                            )
                            with open(str(out_path), "rb") as f_obj:
                                requests.post(
                                    f"https://api.telegram.org/bot{t_token}/sendPhoto",
                                    data={"chat_id": t_chat, "caption": caption},
                                    files={"photo": f_obj}
                                )
                            st.toast("✅ Telegram Alert Sent", icon="🚀")
                        else:
                            st.warning("⚠️ Telegram credentials missing")
                    except Exception as e:
                        st.error(f"Telegram Failed: {e}")

                    # 2. SEND EMAIL
                    if email_to:
                        subject = f"🚨 [SafeRide] Accident Detected • {acc_max:.2f}"
                        body = (
                            f"CRITICAL ALERT\n"
                            f"Accident detected with {acc_max:.2f} confidence.\n"
                            f"Time: {now_str()}\n"
                            f"Source: {f.name}\n"
                            f"S3 Image Link: {presign_s3_uri(s3_uri)}\n"
                        )
                        try:
                            send_email(subject, body, email_to, attachments=[str(out_path)])
                            st.success("📧 Accident Email Sent!")
                        except Exception as e:
                            st.error(f"Email Failed: {e}")
                    
                    # Mark as sent so it doesn't send again when you chat
                    st.session_state["alert_already_sent"] = True
                else:
                    st.info("ℹ️ Alert already sent for this image (skipping duplicate).")
                
                # 1. SEND TELEGRAM (DIRECT - BYPASSING ALERTMANAGER)
                try:
                    t_token = os.getenv("TELEGRAM_BOT_TOKEN")
                    t_chat = os.getenv("TELEGRAM_CHAT_ID")
                    if t_token and t_chat:
                        caption = (
                            f"🚨 Accident Detected\n"
                            f"Conf: {acc_max:.2f}\n"
                            f"Time: {now_str()}\n"
                            f"Run: {run_id}"
                        )
                        with open(str(out_path), "rb") as f_obj:
                            requests.post(
                                f"https://api.telegram.org/bot{t_token}/sendPhoto",
                                data={"chat_id": t_chat, "caption": caption},
                                files={"photo": f_obj}
                            )
                        st.toast("✅ Telegram Alert Sent", icon="🚀")
                    else:
                        st.warning("⚠️ Telegram credentials missing")
                except Exception as e:
                    st.error(f"Telegram Failed: {e}")

                # 2. SEND EMAIL
                if email_to:
                    subject = f"🚨 [SafeRide] Accident Detected • {acc_max:.2f}"
                    body = (
                        f"CRITICAL ALERT\n"
                        f"Accident detected with {acc_max:.2f} confidence.\n"
                        f"Time: {now_str()}\n"
                        f"Source: {f.name}\n"
                        f"S3 Image Link: {presign_s3_uri(s3_uri)}\n"
                    )
                    # Use the 'out_path' (annotated image) as attachment
                    try:
                        send_email(subject, body, email_to, attachments=[str(out_path)])
                        st.success("📧 Accident Email Sent!")
                    except Exception as e:
                        st.error(f"Email Failed: {e}")

# -------------------- VIDEO TAB --------------------
def process_video_and_log(video_path: str, source_name: str, run_id: str,
                        sample_fps: float = 2.0) -> tuple[str | None, int]:
    """
    - Writes an annotated MP4 via Ultralytics (save=True)
    - Uploads annotated MP4 to S3
    - Samples frames at ~sample_fps and logs detections to RDS (one row per box)
        - Tracks the best (max-confidence) 'accident' frame and sends THAT frame to Telegram
        - Sends one helmet-violation email per run with MD/JSON reports
        Returns: (s3_uri_video or None, rows_inserted)
    """
    #local helper
    def _norm(s: str) -> str:
        return s.replace(" ", "").replace("-", "").replace("_", "").lower()
    
    #Create annotated video on disk (INCREASED SIZE TO 1280)
    res = model.predict(
            source=video_path,
            conf=pred_conf, iou=iou, imgsz=1280,
            save=True, line_thickness=3,
            project="runs_saferride", name=f"predict_{run_id}"
        )
    save_dir = Path(res[0].save_dir)
    out_vid = next(save_dir.glob("*.mp4"), None)
    s3_uri_video = None
    if out_vid:
        s3_uri_video = upload_to_s3(str(out_vid), key_prefix=f"runs/{run_id}/videos")

    # 2) Sample frames and log detections
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps   = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    every_n = max(1, int(round(fps / max(sample_fps, 0.1))))

    progress = st.progress(0.0, text="Processing video frames…")
    rows_inserted, idx, sampled = 0, 0, 0
    

    # Track best accident frame across the run
    best_acc_conf = -1.0
    best_acc_frame_path: str | None = None

    # Accident persistence window & single alert per run
    window = deque(maxlen=persist_k)
    sent_accident_alert = False

    # Helmet violation: one email per run
    sent_nohelmet_email = False
    email_to = [e.strip() for e in os.getenv("ALERT_EMAIL_TO","").split(",") if e.strip()]
    
    try:
            while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if idx % every_n == 0:
                        # Save current frame to temp and predict (no save)
                        tmp_jpg_path = Path(tempfile.mkstemp(suffix=".jpg")[1])
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(tmp_jpg_path)

                        # Primary inference (INCREASED SIZE TO 1280)
                        fr = model.predict(source=str(tmp_jpg_path), conf=pred_conf, iou=iou, imgsz=1280, save=False)[0]
                        H, W = fr.orig_shape

                        # ACCIDENT: compute & track best
                        acc_conf = max_accident_conf(fr, names)  # None if no accident box
                        if acc_conf is not None and acc_conf > best_acc_conf:
                            best_acc_conf = acc_conf
                            ann_bgr = fr.plot(line_width=3)                     # BGR ndarray
                            best_path = Path(tempfile.mkstemp(suffix=".jpg")[1])
                            Image.fromarray(ann_bgr[:, :, ::-1]).save(best_path) # BGR->RGB
                            best_acc_frame_path = str(best_path)

                        # Insert rows to detections and track best no-helmet in frame
                        # (for email later)
                        nh_max, nh_box = None, None
                        for b in fr.boxes:
                            cls_id = int(b.cls[0].item())
                            raw_name = names.get(cls_id, "unknown")
                            cls_norm = _norm(raw_name)
                            confv  = float(b.conf[0].item())
                            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]

                            insert_detection_row(engine, {
                                "id": str(uuid.uuid4()),
                                "run_id": run_id,
                                "source_type": "video",
                                "source_name": source_name,
                                "frame_index": idx,
                                "class_id": cls_id,
                                "class_name": raw_name,
                                "confidence": confv,
                                "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "bbox_norm": {"x1": x1 / W, "y1": y1 / H, "x2": x2 / W, "y2": y2 / H},
                                "s3_uri": s3_uri_video,
                            })
                            rows_inserted += 1
                            
                            is_violation = (cls_norm in VIOLATION_CLASSES)
                            if is_violation and (nh_max is None or confv > nh_max):
                                nh_max,nh_box = confv,b

                        # ------------------------------------------------------
                        # VIDEO ACCIDENT ALERT (Direct Telegram + Email)
                        # ------------------------------------------------------
                        hit = (acc_conf is not None) and (acc_conf >= acc_alert)
                        window.append(1 if hit else 0)

                        if (not sent_accident_alert) and len(window) == persist_k and sum(window) == persist_k:
                            sent_accident_alert = True
                            frame_to_send = best_acc_frame_path or str(tmp_jpg_path)
                            conf_to_report = best_acc_conf if best_acc_conf >= 0 else (acc_conf or 0.0)

                            s3_img_uri = None
                            try:
                                s3_img_uri = upload_to_s3(frame_to_send, key_prefix=f"runs/{run_id}/frames")
                            except Exception:
                                pass
                            
                            st.error(f"🚨 Accident in video (≥{acc_alert:.2f}). Alerting...")

                            # 1. SEND TELEGRAM (DIRECT)
                            try:
                                t_token = os.getenv("TELEGRAM_BOT_TOKEN")
                                t_chat = os.getenv("TELEGRAM_CHAT_ID")
                                if t_token and t_chat:
                                    vid_caption = (
                                        f"🚨 Video Accident\n"
                                        f"Conf: {conf_to_report:.2f}\n"
                                        f"Time: {now_str()}\n"
                                        f"Run: {run_id}"
                                    )
                                    with open(frame_to_send, "rb") as f_obj:
                                        requests.post(
                                            f"https://api.telegram.org/bot{t_token}/sendPhoto",
                                            data={"chat_id": t_chat, "caption": vid_caption},
                                            files={"photo": f_obj}
                                        )
                                    st.toast("✅ Telegram Sent (Video)", icon="🚀")
                            except Exception as e:
                                print(f"Telegram Video Error: {e}")

                            # 2. SEND EMAIL
                            if email_to:
                                subject = f"🚨 [SafeRide] Video Accident Detected! ({conf_to_report:.2f})"
                                body = (
                                    f"CRITICAL VIDEO ALERT\n"
                                    f"Accident detected in video stream.\n\n"
                                    f"Time: {now_str()}\n"
                                    f"Confidence: {conf_to_report:.2f}\n"
                                    f"S3 Image Frame: {presign_s3_uri(s3_img_uri) if s3_img_uri else 'N/A'}\n"
                                    f"S3 Video Link: {presign_s3_uri(s3_uri_video) if s3_uri_video else 'Pending upload'}\n"
                                )
                                # Attach the best frame found
                                try:
                                    send_email(subject, body, email_to, attachments=[frame_to_send])
                                    st.success("📧 Accident Email Sent!")
                                except Exception:
                                    pass

                        # Helmet violation email (once per run) + MD/JSON reports
                        if (not sent_nohelmet_email) and email_to:
                            #use nh_max, nh_box from above (if any)
                                                       
                            if nh_max is not None and nh_max >= nohelmet_alert:
                                # Annotated still & S3
                                ann = fr.plot()
                                ann_path = tmp_jpg_path.with_suffix(".annot.jpg")
                                Image.fromarray(ann[:, :, ::-1]).save(ann_path)
                                s3_image_uri = upload_to_s3(str(ann_path), key_prefix=f"runs/{run_id}/images")

                                # Build/upload MD + JSON
                                x1, y1, x2, y2 = [float(v) for v in nh_box.xyxy[0].tolist()] if nh_box is not None else (0,0,0,0)
                                md_path   = str(ann_path.with_suffix(".md"))
                                json_path = str(ann_path.with_suffix(".json"))

                                md_fields = {
                                    "Time": now_str(),
                                    "Run ID": run_id,
                                    "Source": source_name,
                                    "Frame": idx,
                                    "Class": "nohelmet",
                                    "Confidence": f"{nh_max:.2f}",
                                }
                                write_md_report(md_path, "Helmet Violation (video)", md_fields,
                                                image_s3=s3_image_uri, video_s3=s3_uri_video)

                                write_json_report(json_path, {
                                    "timestamp": now_str(),
                                    "run_id": run_id,
                                    "source_type": "video",
                                    "source_name": source_name,
                                    "frame_index": idx,
                                    "class_name": "nohelmet",
                                    "confidence": float(nh_max),
                                    "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                    "bbox_norm": {"x1": x1 / W if W else 0, "y1": y1 / H if H else 0, "x2": x2 / W if W else 0, "y2": y2 / H if H else 0},
                                    "image_s3_uri": s3_image_uri,
                                    "video_s3_uri": s3_uri_video,
                                })

                                s3_report_md   = upload_to_s3(md_path,   key_prefix=f"runs/{run_id}/reports")
                                s3_report_json = upload_to_s3(json_path, key_prefix=f"runs/{run_id}/reports")
                                # build PDF from video violation snapshot
                                pdf_path = str(ann_path.with_suffix(".pdf"))
                                build_incident_pdf(pdf_path, "Helmet Violation (video)", md_fields, image_path=str(ann_path))
                                s3_report_pdf = upload_to_s3(pdf_path, key_prefix=f"runs/{run_id}/reports")

                                subject = f"[SafeRide] Helmet violation (video) • {nh_max:.2f}"
                                body = (
                                    f"Time: {now_str()}\n"
                                    f"Run: {run_id}\n"
                                    f"Source: {source_name} @ frame {idx}\n"
                                    f"Confidence: {nh_max:.2f}\n"
                                    f"S3 image: {presign_s3_uri(s3_image_uri)}\n"
                                    f"S3 video: {presign_s3_uri(s3_uri_video) if s3_uri_video else '-'}\n"
                                    f"Report (MD): {presign_s3_uri(s3_report_md)}\n"
                                    f"Report (JSON): {presign_s3_uri(s3_report_json)}\n"
                                    f"Report (PDF): {presign_s3_uri(s3_report_pdf)}\n"   
                                )
                                send_email(subject, body, email_to, attachments=None)
                                st.warning("📧 Helmet violation email sent (video)")

                                # violations row
                                insert_violations_row(engine, {
                                    "id": str(uuid.uuid4()),
                                    "run_id": run_id,
                                    "source_type": "video",
                                    "source_name": source_name,
                                    "frame_index": idx,
                                    "class_name": "nohelmet",
                                    "confidence": float(nh_max),
                                    "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                    "bbox_norm": {"x1": x1 / W if W else 0, "y1": y1 / H if H else 0, "x2": x2 / W if W else 0, "y2": y2 / H if H else 0},
                                    "s3_image_uri": s3_image_uri,
                                    "s3_report_uri": s3_report_md,
                                    "email_sent": True
                                })
                                sent_nohelmet_email = True

                    idx += 1
                    if total:
                        progress.progress(min(idx / total, 1.0))
    except Exception as e:
            st.error(f"Error processing video: {e}")
    finally:
            cap.release()
            progress.progress(1.0)
    
    return s3_uri_video, rows_inserted

if mode == "Video":
        v = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        sample_fps = st.slider("Sample FPS for logging (frames per second)", 0.5, 6.0, 2.0, 0.5)
        if v:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(v.read())
                local_video = tmp.name

            run_id = uuid.uuid4().hex

            s3_video, nrows = process_video_and_log(
                video_path=local_video,
                source_name=v.name,
                run_id=run_id,
                sample_fps=sample_fps
            )

            predict_dir = Path(f"runs_saferride/predict_{run_id}")
            out_vid = next(predict_dir.glob("*.mp4"), None)
            if out_vid:
                st.video(str(out_vid))
            if s3_video:
                st.success(f"Annotated video uploaded → {presign_s3_uri(s3_video)}")
            st.info(f"Logged {nrows} frame detections to RDS (≈ {sample_fps} FPS sampling).")

# -------------------- CHAT TAB --------------------

with tab_chat:
    st.subheader("Ask about detections, violations, or reports")

    st.markdown("""
    - Uses a custom ChatAgent with tools for RAG (FAISS), SQL (RDS), and S3 (presign)
    - RAG index must be built first (see sidebar)
    - Examples:
      - "Show me the latest 5 helmet violations"
      - "How many accidents were detected in the last 7 days?"    
      - "Get the presigned links for reports in run abc123"
      - "List all detections with confidence > 0.8"
      - "How many detections of class 'car' were there last month?"
      - "What reports are available for run xyz789?"
    """)
    if "_chat_history" not in st.session_state:
        st.session_state["_chat_history"] = []
    if "_clear_chat_q" not in st.session_state:
        st.session_state["_clear_chat_q"] = False
    
    # if previous action asked to clear, do it BEFORE rendering the widget
    if st.session_state["_clear_chat_q"]:
        st.session_state["_clear_chat_q"] = False
        st.session_state["chat_q"] = "" 
    
   # Chat input + Ask button
    if st.button("Clear chat", key="btn_clear_chat"):
        st.session_state["_chat_history"] = []
        st.session_state["_clear_chat_q"] = True
        st.rerun()

    user_q = st.text_input("Your question", key="chat_q")

    if st.button("Ask", use_container_width=True) and user_q.strip():
        try:
            resp = CA.answer(user_q.strip())
            st.session_state["_chat_history"].append(("user", user_q.strip()))
            st.session_state["_chat_history"].append(("assistant", resp))
        except Exception as e:
            st.error(f"Chat error: {e}")
        finally:
            st.session_state["_clear_chat_q"] = True  # request clearing on next run
            st.rerun()

    # Render history (last exchange only)
    msgs = st.session_state["_chat_history"][-2:] if st.session_state["_chat_history"] else []
    for role, content in msgs:
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            mode = content.get("mode")
            ans  = content.get("answer","")
            st.markdown(f"**Bot ({mode}):** {ans}")

            if content.get("used_agents"):
                st.caption("Agents used: " + ", ".join(content["used_agents"]))
            
            # NEW — show the exact SQL when the SQL agent ran
            if content.get("sql"):
                with st.expander("SQL used"):
                    st.code(content["sql"], language="sql")
            if content.get("sources"):
                with st.expander("Sources"):
                    for s in content["sources"]:
                        st.write(s)
            if content.get("rows"):
                with st.expander("Rows"):
                    rows = content["rows"]
                    for r in rows:
                        for k, v in list(r.items()):
                            if isinstance(v, str) and v.startswith("s3://"):
                                r[k] = presign_s3_uri(v)
                    st.write(rows)


st.image("D:\Helmet_detection\banner.jpg", use_container_width='3000')

st.subheader("\"The future of road safety is not just about smarter cars, but smarter oversight.\"")
st.caption("Enhancing road safety through AI by SafeRide")