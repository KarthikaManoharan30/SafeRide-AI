st.image("D:\Helmet_detection\banner.jpg", use_container_width='3000')

🛡️ SafeRide: AI-Powered Accident & Helmet Detection System
SafeRide is an end-to-end AI monitoring solution designed to enhance road safety. It utilizes computer vision (YOLOv8) to detect accidents and helmet violations in real-time, automatically triggers multi-channel alerts, and provides an intelligent conversational interface using Local LLMs (Llama 3.2) to analyze incident data.
This project demonstrates a robust integration of Edge AI, Cloud Services (AWS), and Agentic LLM workflows.
🌟 Key Features
👁️ Real-Time Computer Vision:
Custom-trained YOLOv8 model capable of detecting accident, helmet, and nohelmet classes with high precision.
Supports both Image snapshots and Video stream analysis.
☁️ Hybrid Cloud Architecture:
AWS S3: Stores evidence (annotated images/videos) and generated incident reports (Markdown/JSON/PDF).
AWS RDS (PostgreSQL): Persistently logs detection metadata, confidence scores, and timestamps for audit trails.
🚨 Intelligent Alerting System:
Critical Accidents: Bypasses standard queues to send an Instant Telegram Alert with photo evidence.
Helmet Violations: Generates a detailed PDF report and sends an Email Notification with secure S3 links.
🤖 Agentic AI Chat Assistant:
Powered by Llama 3.2 (1B) running locally via Ollama.
SQL Agent: Translates natural language (e.g., "How many accidents occurred last week?") into SQL queries to fetch exact statistics from the database.
RAG (Retrieval-Augmented Generation) Agent: Reads generated incident reports to answer contextual questions (e.g., "Summarize the violation in Run ID 5a2b...").
🐳 Containerized Deployment:
The core application runs inside a Docker container, communicating securely with the Host machine's GPU resources and Ollama instance.
🛠️ System Architecture
Input: User uploads media via Streamlit UI.
Detection: YOLOv8 processes frames.
Logic:
Accident: -> Telegram Bot -> AWS S3.
No Helmet: -> Emailer -> PDF Report -> AWS S3.
Storage: Metadata -> PostgreSQL (RDS); Files -> S3.
Interaction: User asks questions -> LangChain Agents route query to either Vector Store (FAISS) or Database (SQL) -> Llama 3.2 generates response.
🚀 How to Run This Project
Prerequisites
Docker Desktop installed.
Ollama installed on the host machine (Windows/Mac/Linux).
Git installed.
AWS Account (S3 Bucket & RDS Database).
Step 1: Clone the Repository
code
Bash
git clone https://github.com/YOUR_USERNAME/saferide.git
cd saferide
Step 2: Configure Environment Variables
Create a file named .env in the root directory. Add your credentials (do not share this file):
code
Ini
# Database (AWS RDS)
RDS_HOST=your-db-endpoint.amazonaws.com
RDS_PORT=5432
RDS_USER=postgres
RDS_PASSWORD=your_password
RDS_DB=postgresdb

# Cloud Storage (AWS S3)
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name

# AI Models (Ollama)
# Using Llama 3.2 (1B) for optimized performance
RAG_MODEL=llama3.2:1b
RAG_EMBED_MODEL=llama3.2:1b

# Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
ALERT_EMAIL_TO=admin@example.com
Step 3: Setup Ollama (Host Machine)
Since we are running the LLM on the host machine to save resources, we need to prepare the model and network permissions.
Pull the Model:
Open PowerShell/Terminal and run:
code
Powershell
ollama pull llama3.2:1b
Expose Ollama to Docker:
Windows PowerShell (Run as Administrator):
code
Powershell
$env:OLLAMA_HOST="0.0.0.0"; ollama serve
Keep this terminal window open.
Step 4: Build and Run with Docker
Open a new terminal window in the project folder. We need to tell Docker how to talk to the Host's Ollama instance using --add-host.
Build the Image:
code
Bash
docker build -t saferide-app .
Run the Container:
code
Bash
docker run --add-host=host.docker.internal:host-gateway --env-file .env -p 8501:8501 -it saferide-app
Step 5: Access the App
Open your browser and navigate to:
http://localhost:8501
📚 Usage Guide
1. Database Initialization
On the first run, expand the "Database setup" section in the app and click:
Create detections table
Create violations table
2. Running Detections
Go to the Detect tab.
Upload an Image or Video.
The system will automatically log detections to the DB and send alerts if violations are found.
3. Using the AI Chat
To enable the Chatbot to read reports, expand "Reports & RAG" and click "Build/Refresh Index".
Go to the Chat tab and ask questions like:
"Show me the latest 5 helmet violations." (Uses SQL Agent)
"Summarize the accident report." (Uses RAG Agent)