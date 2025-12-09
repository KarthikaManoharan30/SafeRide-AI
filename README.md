<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0d1117&height=300&section=header&text=SafeRide%20AI&fontSize=90&animation=fadeIn&fontAlignY=38&desc=Accident%20%26%20Helmet%20Detection%20System&descAlignY=51&descAlign=62" width="100%"/>
</div>

# 🛡️ SafeRide: AI-Powered Accident & Helmet Detection System

**SafeRide is an end-to-end AI monitoring solution designed to enhance road safety. It utilizes computer vision (YOLOv8) to detect accidents and helmet violations in real-time, automatically triggers multi-channel alerts, and provides an intelligent conversational interface using Local LLMs (Llama 3.2) to analyze incident data.**

This project demonstrates a robust integration of Edge AI, Cloud Services (AWS), and Agentic LLM workflows.

## 🌟 Key Features

- **👁️ Real-Time Computer Vision:**
    - Custom-trained **YOLOv8** model capable of detecting `accident`, `helmet`, and `nohelmet` classes with high precision.
    - Supports both **Image** snapshots and **Video** stream analysis.

- **☁️ Hybrid Cloud Architecture:**
    - **AWS S3:** Stores evidence (annotated images/videos) and generated incident reports (Markdown/JSON/PDF).
    - **AWS RDS (PostgreSQL):** Persistently logs detection metadata, confidence scores, and timestamps for audit trails.

- **🚨 Intelligent Alerting System:**
    - **Critical Accidents:** Bypasses standard queues to send an **Instant Telegram Alert** with photo evidence.
    - **Helmet Violations:** Generates a detailed PDF report and sends an **Email Notification** with secure S3 links.

- **🤖 Agentic AI Chat Assistant:**
    - Powered by **Llama 3.2 (1B)** running locally via Ollama.
    - **SQL Agent:** Translates natural language (e.g., *"How many accidents occurred last week?"*) into SQL queries to fetch exact statistics from the database.
    - **RAG (Retrieval-Augmented Generation) Agent:** Reads generated incident reports to answer contextual questions (e.g., *"Summarize the violation in Run ID 5a2b..."*).

- **🐳 Containerized Deployment:**
    - The core application runs inside a **Docker** container, communicating securely with the Host machine's GPU resources and Ollama instance.

---

## 🛠️ System Architecture

1. **Input:** User uploads media via **Streamlit** UI.
2. **Detection:** YOLOv8 processes frames.
3. **Logic:**
    - *Accident:* -> Telegram Bot -> AWS S3.
    - *No Helmet:* -> Emailer -> PDF Report -> AWS S3.
4. **Storage:** Metadata -> PostgreSQL (RDS); Files -> S3.
5. **Interaction:** User asks questions -> **LangChain Agents** route query to either Vector Store (FAISS) or Database (SQL) -> Llama 3.2 generates response.

---

## 🚀 How to Run This Project

### Prerequisites
1. **Docker Desktop** installed.
2. **Ollama** installed on the host machine (Windows/Mac/Linux).
3. **Git** installed.
4. **AWS Account** (S3 Bucket & RDS Database).

### Step 1: Clone the Repository
```bash
git clone https://github.com/KarthikaManoharan30/SafeRide-AI.git
cd SafeRide-AI
