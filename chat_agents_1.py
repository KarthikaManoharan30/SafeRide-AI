"""
Chat agent for SafeRide Streamlit chat tab.
- Single-file, simple router (VECTOR / SQL / PRESIGN / NONE)
- Matches saferide.py expectations: returns a dict with keys
  {"mode", "answer", "rows" (optional), "sources" (optional)}

Env vars used:
  RAG_MODEL=llama3.1
  RAG_EMBED_MODEL=nomic-embed-text
  RAG_INDEX_DIR=indexes/faiss_reports
  DB_URL=postgresql+psycopg2://user:pass@host:5432/db
"""

from __future__ import annotations
import os, json, uuid
import re
from typing import List, Dict, Any, Optional
import datetime as dt

# Load environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()

# LangChain (Ollama)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
# --- Report/Email agent (simple KPI + links email) ---
from datetime import datetime, timedelta, timezone
from sqlalchemy import text as sql_text
from emailer import send_email

# Optional SQL
try:
    from sqlalchemy import create_engine, text as sql_text
    HAVE_SQLALCHEMY = True
except Exception:
    HAVE_SQLALCHEMY = False

# -----------------------------
# Config
# -----------------------------
RAG_MODEL = os.getenv("RAG_MODEL", "llama3.1")
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
INDEX_DIR = os.getenv("RAG_INDEX_DIR", "indexes/faiss_reports")

# --- CONSTRUCT DB_URL FROM INDIVIDUAL RDS ENV VARS ---
RDS_HOST = os.getenv("RDS_HOST")
RDS_PORT = os.getenv("RDS_PORT", "5432")
RDS_USER = os.getenv("RDS_USER")
RDS_PASSWORD = os.getenv("RDS_PASSWORD")
RDS_DB = os.getenv("RDS_DB")

DB_URL = None
if all([RDS_HOST, RDS_PORT, RDS_USER, RDS_PASSWORD, RDS_DB]):
    DB_URL = f"postgresql+psycopg2://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB}"

# (Helper functions _intent, _as_text, _safe_json_default remain the same)
def _intent(query: str):
    q = query.lower()
    intents = {
        "sql": any(w in q for w in ["count", "how many", "latest", "top ", "last ", "yesterday", "today", "confidence", "violation", "highest"]),
        "rag": any(w in q for w in ["report", "reports", "summarize", "explain", "what do the report", "run "]),
        "report": any(w in q for w in ["generate", "make", "create", "weekly report", "monthly report", "report for", "email report"]),
        "email": "email" in q or "send mail" in q or "send email" in q
    }
    return intents

def _as_text(x: Any) -> str:
    try:
        return x.content
    except Exception:
        return str(x)

def _safe_json_default(obj: Any):
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    return obj

# -----------------------------
# LLM & Embeddings Setup
# -----------------------------
# UPDATED: Added base_url to point to host machine from Docker
llm = ChatOllama(
    model=RAG_MODEL, 
    temperature=0.2, 
    base_url="http://host.docker.internal:11434"
)

emb = OllamaEmbeddings(
    model=RAG_EMBED_MODEL, 
    base_url="http://host.docker.internal:11434"
)

_vector = None
_retriever = None
if os.path.isdir(INDEX_DIR):
    try:
        _vector = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
        _retriever = _vector.as_retriever(search_kwargs={"k": 4})
    except Exception:
        _vector, _retriever = None, None

def tool_vector_search(query: str) -> Dict[str, Any]:
    if not _retriever:
        return {"answer": "Vector index not available. Please build the RAG index first.", "sources": []}
    docs = _retriever.invoke(query)
    if not docs:
        brief = _as_text(llm.invoke(f"User: {query}\nAnswer briefly."))
        return {"answer": brief, "sources": []}
    blocks, sources = [], []
    for i, d in enumerate(docs, 1):
        src = (d.metadata or {}).get("s3_key", "unknown")
        sources.append(src)
        blocks.append(f"[Doc {i} | {src}]\n{d.page_content.strip()}")
    prompt = (
        "You are a helpful assistant. Use the provided context to answer concisely.\n"
        "Cite using the [Doc i | source] tags already present in the context when relevant.\n\n"
        f"Context:\n{os.linesep.join(blocks)}\n\nQuestion: {query}\n\nAnswer:"
    )
    ans = _as_text(llm.invoke(prompt))
    return {"answer": ans, "sources": sources}

# -----------------------------
# SQL Tool (read-only) - EDITED WITH CUSTOM QUERIES
# -----------------------------
_engine = None
if DB_URL and HAVE_SQLALCHEMY:
    try:
        _engine = create_engine(DB_URL, pool_pre_ping=True)
    except Exception:
        _engine = None

def tool_sql(query: str, limit: int = 50) -> Dict[str, Any]:
    if not _engine:
        return {"rows": [], "answer": "Database not available. Please check your .env file for RDS credentials."}

    q = query.strip()
    if not q.lower().lstrip().startswith("select"):
        ql = q.lower()
        if "latest" in ql and ("helmet" in ql or "no-helmet" in ql):
            q = (
                "SELECT id::text AS id, timestamp, run_id, source_name, confidence, "
                "s3_image_uri, s3_report_uri FROM violations "
                "WHERE class_name='nohelmet' ORDER BY timestamp DESC LIMIT 5"
            )
        
        # --- NEW CUSTOM BLOCK 1: HIGHEST CONFIDENCE ---
        elif "highest confidence" in ql or "most confident" in ql:
            q = ("SELECT class_name, confidence, source_name, timestamp FROM detections ORDER BY confidence DESC LIMIT 1")

        # --- NEW CUSTOM BLOCK 2: COUNT BY CLASS ---
        elif "count" in ql and "by class" in ql:
            q = ("SELECT class_name, COUNT(*) as detection_count FROM detections GROUP BY class_name ORDER BY class_name")

        # --- NEW CUSTOM BLOCK 3: SHOW THE SINGLE VIOLATION ---
        elif "show" in ql and ("the violation" in ql or "single violation" in ql):
            q = ("SELECT timestamp, source_name, confidence, s3_image_uri, s3_report_uri FROM violations ORDER BY timestamp DESC LIMIT 1")

        elif ("count" in ql or "how many" in ql) and "accident" in ql:
            q = "SELECT COUNT(*) AS num FROM detections WHERE class_name='accident'"
        else:
            q = (
                "SELECT id::text AS id, timestamp, run_id, source_name, class_name, confidence, s3_uri "
                "FROM detections ORDER BY timestamp DESC LIMIT 10"
            )

    if not q.lower().lstrip().startswith("select"):
        return {"rows": [], "answer": "Only SELECT queries are allowed."}

    if not re.search(r"\blimit\b", q, flags=re.IGNORECASE):
        q = f"SELECT * FROM ({q}) AS subq LIMIT {int(limit)}"

    rows = []
    with _engine.connect() as conn:
        res = conn.execute(sql_text(q))
        cols = list(res.keys())
        for r in res.fetchall():
            rows.append({c: _safe_json_default(v) for c, v in zip(cols, r)})

    summary_prompt = (
        "Summarize the following table for the user's question in 1-2 concise sentences.\n"
        f"Question: {query}\n"
        f"Rows (JSON):\n{json.dumps(rows, indent=2, default=_safe_json_default)}\n\n"
        "Answer:"
    )
    ans = _as_text(llm.invoke(summary_prompt))
    return {"rows": rows, "answer": ans, "sql": q}

# (Rest of the file: run_sql_agent, run_rag_agent, run_report_email_agent, etc., remains the same)
def run_sql_agent(query: str):
    out = tool_sql(query, limit=50)
    out.setdefault("mode", "SQL")
    out.setdefault("answer", "")
    out.setdefault("rows", [])
    out["used_agents"] = ["Planner", "SQL"]
    return out

def run_rag_agent(query: str):
    out = tool_vector_search(query)
    out.setdefault("mode", "VECTOR")
    out.setdefault("answer", "")
    out.setdefault("sources", [])
    out["used_agents"] = ["Planner", "RAG"]
    return out

def _window_from_query(q: str):
    ql = q.lower()
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)
    m = re.search(r"last\s+(\d+)\s*(day|days|week|weeks|month|months)", ql)
    if m:
        n = int(m.group(1)); unit = m.group(2)
        if unit.startswith("day"):   since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=n)
        elif unit.startswith("week"): since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7*n)
        elif unit.startswith("month"): since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30*n)
    return since

def run_report_email_agent(query: str):
    if not _engine:
        return {"mode":"REPORT","answer":"DB not configured; cannot compile report.","used_agents":["Planner","REPORT"]}
    since = _window_from_query(query)
    kpis = {}
    rows = []
    with _engine.connect() as conn:
        kpis["accidents"] = conn.execute(sql_text("SELECT COUNT(*) FROM detections WHERE class_name='accident' AND timestamp>=:s"), {"s": since}).scalar() or 0
        kpis["nohelmet"] = conn.execute(sql_text("SELECT COUNT(*) FROM violations WHERE class_name='nohelmet' AND timestamp>=:s"), {"s": since}).scalar() or 0
        res = conn.execute(sql_text("SELECT timestamp, source_name, confidence, s3_image_uri, s3_report_uri FROM violations WHERE timestamp>=:s ORDER BY timestamp DESC LIMIT 10"), {"s": since})
        cols = list(res.keys())
        for r in res.fetchall():
            rows.append({c:(r[i].isoformat() if hasattr(r[i],'isoformat') else r[i]) for i,c in enumerate(cols)})
    links = {"pdf": None, "violations": []}
    for r in rows:
        for k in ("s3_image_uri","s3_report_uri"):
            v = r.get(k)
            if isinstance(v, str) and v.startswith("s3://"):
                r[k] = v
    recipients = [e.strip() for e in os.getenv("ALERT_EMAIL_TO","").split(",") if e.strip()]
    emailed = False
    email_error = None
    if recipients:
        try:
            body = [f"Weekly Safety Summary since {since:%Y-%m-%d} (UTC)", f"- Accidents: {kpis['accidents']}", f"- No-helmet Violations: {kpis['nohelmet']}", "", "Latest violation links:"]
            for r in rows:
                body.append(f"• {r['timestamp']} | {r['source_name']} | conf {float(r['confidence']):.2f}")
                if r.get("s3_image_uri"):  body.append(f"  image: {r['s3_image_uri']}")
                if r.get("s3_report_uri"): body.append(f"  report: {r['s3_report_uri']}")
            send_email("[SafeRide] Weekly Safety Summary", "\n".join(body), recipients)
            emailed = True
        except Exception as e:
            email_error = f"{type(e).__name__}: {e}"
    ans = f"Accidents: {kpis['accidents']}, No-helmet: {kpis['nohelmet']} since {since:%Y-%m-%d} (UTC)."
    if emailed:
        ans += " Email sent to configured recipients."
    elif recipients:
        ans += f" Email failed: {email_error or 'unknown error'}"
    else:
        ans += " No ALERT_EMAIL_TO configured (set ALERT_EMAIL_TO in env to enable)."
    return {"mode":"REPORT", "answer": ans, "rows": rows, "links": links, "used_agents":["Planner","REPORT","EMAIL"] if emailed else ["Planner","REPORT"]}

def tool_presign(uri: str) -> str:
    return uri.strip()

def answer(user_msg: str) -> Dict[str, Any]:
    intents = _intent(user_msg)
    if intents["report"]:
        return run_report_email_agent(user_msg)
    if intents["rag"]:
        return run_rag_agent(user_msg)
    if intents["sql"]:
        return run_sql_agent(user_msg)
    msg = _as_text(llm.invoke(f"Answer concisely:\nUser: {user_msg}\nAssistant:"))
    return {"mode":"CHAT","answer":msg,"used_agents":["Planner","LLM"]}