# rag_ingest.py
# Pull reports (MD/JSON) from S3 -> chunk -> embed (Ollama) -> FAISS index

import os, json, io
from pathlib import Path
from datetime import datetime, timezone

import boto3
from dotenv import load_dotenv

#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET  = os.getenv("S3_BUCKET")
REPORT_PREFIX = os.getenv("RAG_REPORT_PREFIX", "runs/")  # we’ll scan runs/*/reports/*
INDEX_DIR = os.getenv("RAG_INDEX_DIR", "indexes/faiss_reports")


s3 = boto3.client("s3", region_name=AWS_REGION)

def list_report_keys(prefix: str):
    """Yield S3 keys under prefix (e.g. runs/*/reports/) ending with .md, .json, or .txt."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "/reports/" in key and (key.endswith(".md") or key.endswith(".json") or key.endswith(".txt")):
                yield key


def fetch_text(bucket: str, key: str) -> str:
    o = s3.get_object(Bucket=bucket, Key=key)
    data = o["Body"].read()
    if key.endswith(".json"):
        try:
            return json.dumps(json.loads(data), indent=2, default=str)
        except Exception:
            return data.decode("utf-8", errors="ignore")
    return data.decode("utf-8", errors="ignore")  # MD or fallback

def build_or_update_index(
    s3_prefix: str | None = None,
    index_dir: str | None = None,
    embed_model: str | None = None,
) -> str:
    """
    Scan S3 for report texts under s3_prefix, chunk, embed with Ollama, and save FAISS locally.
    Returns the path to the saved FAISS index directory.
    """
    if not S3_BUCKET:
        raise ValueError("S3_BUCKET is not set in environment")

    prefix = s3_prefix or REPORT_PREFIX           
    out_dir = index_dir or INDEX_DIR
    model_name = embed_model or os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")

    idx_path = Path(out_dir)
    idx_path.mkdir(parents=True, exist_ok=True)

    # UPDATED: Added base_url to point to host machine from Docker
    embeddings = OllamaEmbeddings(
        model=model_name,
        base_url="http://host.docker.internal:11434"
    )

    texts: list[str] = []
    metadatas: list[dict] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    count_files = 0
    for key in list_report_keys(prefix):
        try:
            txt = fetch_text(S3_BUCKET, key)
            if not isinstance(txt, str):
                txt = txt.decode("utf-8", errors="ignore")
            chunks = splitter.split_text(txt)
            if chunks:
                texts.extend(chunks)
                metadatas.extend([{"s3_key": key}] * len(chunks))
                count_files += 1
        except Exception as e:
            print("Skip", key, "->", e)

    if not texts:
        print("No report texts found under", prefix)
        # Create an empty-but-valid index so downstream chat doesn't fail
        empty = FAISS.from_texts(texts=[""], embedding=embeddings, metadatas=[{}])
        empty.save_local(out_dir)
        return out_dir

    vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    vs.save_local(out_dir)
    print(f"Indexed {len(texts)} chunks from {count_files} report files into {out_dir}")
    return out_dir

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    saved = build_or_update_index(
        s3_prefix=os.getenv("RAG_REPORT_PREFIX", "runs/"),
        index_dir=os.getenv("RAG_INDEX_DIR", "indexes/faiss_reports"),
        embed_model=os.getenv("RAG_EMBED_MODEL", "nomic-embed-text"),
    )
    print("FAISS index saved to", saved)