<div align="center">
  <img src="https://images.unsplash.com/photo-1555255707-c07966088b7b?q=80&w=2000&auto=format&fit=crop" alt="SafeRide AI Banner" width="100%" style="border-radius: 10px;"/>
  <br/><br/>
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

git clone https://github.com/KarthikaManoharan30/SafeRide-AI.git
cd SafeRide-AI


### Step 2: Configure Environment Variables

### Database (AWS RDS)
RDS_HOST=your-db-endpoint.amazonaws.com
RDS_PORT=5432
RDS_USER=postgres
RDS_PASSWORD=your_password
RDS_DB=postgresdb

### Cloud Storage (AWS S3)
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name

### AI Models (Ollama)
RAG_MODEL=llama3.2:1b
RAG_EMBED_MODEL=llama3.2:1b

### Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
ALERT_EMAIL_TO=admin@example.com


### Step 3: Setup Ollama (Host Machine)
Since we are running the LLM on the host machine to save resources, we need to prepare the model and network permissions.

1. **Pull the Model:**
   Open PowerShell or Terminal and run:
   ```powershell
   ollama pull llama3.2:1b

2. **Expose Ollama to Docker:**
   Windows PowerShell (Run as Administrator):
   ```powershell
   $env:OLLAMA_HOST="0.0.0.0"; ollama serve

### Step 4: Build and Run with Docker Compose
Open a new terminal window in the project folder. Since this project uses Docker Compose, you can start everything with a single command.

     docker compose up

### Step 5: Access the App
Open your browser and navigate to:
     http://localhost:8501


### Step 6: First-Time Setup (Inside the App)
Once the app is running in your browser, you need to initialize the database and AI memory.

1. **Initialize Database:**
   - Expand the **"🗄️ Database setup"** section in the app.
   - Click **"Create detections table"**.
   - Click **"Create violations table"**.
   - *You only need to do this once.*

2. **Initialize AI Memory (RAG):**
   - Go to the **Detect** tab and run a test detection (upload an image) to generate some data.
   - Expand the **"Reports & RAG"** section.
   - Click **"Build/Refresh Index"**.
   - *Do this whenever you want the Chatbot to learn about new detections.*

**You are now ready to use the SafeRide AI system!**
