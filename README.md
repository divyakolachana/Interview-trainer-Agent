# 🤖 AI Interview Trainer Agent
### Powered by IBM Watsonx.ai — Granite-13B

A full-stack AI-powered interview coaching platform built with **Python Flask** and **IBM Watsonx.ai**.  
Features mock interview sessions, real-time feedback, resume-based question generation, and progress tracking.

---

## 📁 Project Structure

```
interview_trainer/
├── app.py                  # Flask backend + IBM Watsonx integration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── session_history.json    # Auto-generated session history (gitignore this)
├── uploads/                # Auto-created upload folder
└── templates/
    └── index.html          # Complete frontend (single-page app)
```

---

## ⚡ Quick Start

### 1. Clone / Download the project
```bash
cd interview_trainer
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure IBM Watsonx credentials
```bash
cp .env.example .env
```
Edit `.env` and fill in your credentials:
```
IBM_API_KEY=your_ibm_cloud_api_key_here
IBM_PROJECT_ID=your_watsonx_project_id_here
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=some-random-secret-string
```

#### How to get IBM Watsonx credentials:
1. Go to [cloud.ibm.com](https://cloud.ibm.com) → Create a free account
2. Navigate to **Watsonx.ai** → Create a project
3. Go to **Manage** → **Access (IAM)** → Create an **API Key**
4. Copy your **Project ID** from the project settings URL
5. The URL is `https://us-south.ml.cloud.ibm.com` for US South region

### 5. Run the application
```bash
python app.py
```
Open your browser at: **http://localhost:5000**

---

## 🎛️ Customizing Agent Behavior

Edit the `AGENT_INSTRUCTIONS` dictionary at the top of `app.py`:

```python
AGENT_INSTRUCTIONS = {
    # Tone: "professional" | "friendly" | "strict" | "encouraging"
    "tone": "professional",

    # Interview type: "hr" | "technical" | "behavioral" | "mixed"
    "interview_type": "mixed",

    # Difficulty: "beginner" | "intermediate" | "advanced"
    "difficulty": "intermediate",

    # Role: "software_engineering" | "data_science" | "core_engineering"
    #        "finance" | "sales" | "generic"
    "target_role": "software_engineering",

    # Questions per session
    "questions_per_session": 5,

    # Max follow-up questions per answer
    "max_followups": 2,

    # Show sample improved answers for weak responses
    "suggest_improved_answers": True,
    "improvement_threshold": 6,  # scores below this get a sample answer
}
```

These defaults can also be changed at runtime via the **Agent Settings** page in the UI.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🎯 **Mock Interview Sessions** | Full interview flow — questions → answers → follow-ups → feedback |
| ⏱️ **Live Timer** | Per-question timer with color-coded warnings |
| 📄 **Resume / JD Upload** | Upload PDF, DOCX, or TXT to tailor questions |
| 🤖 **AI Follow-up Questions** | Dynamic follow-ups based on your answers |
| 📊 **Per-Answer Scoring** | Score 1–10 with strengths, weaknesses, clarity, confidence |
| 💡 **Sample Answers** | Model answers for weak responses |
| 📈 **Progress Tracker** | Score history, trend charts, session comparison |
| 💬 **AI Chat Coach** | Freeform prep chat with IBM Granite |
| 🌙 **Dark Mode** | Full dark/light theme toggle |
| 📱 **Mobile Responsive** | Works on all screen sizes |
| ⚙️ **Agent Settings UI** | Change type, difficulty, role, tone from the UI |

---

## 🚀 Deployment

### Option A: Local / On-Premise
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option B: IBM Code Engine
```bash
# Build and push Docker image
docker build -t interview-trainer .
docker tag interview-trainer us.icr.io/<your-namespace>/interview-trainer:latest
docker push us.icr.io/<your-namespace>/interview-trainer:latest

# Deploy on Code Engine
ibmcloud ce application create \
  --name interview-trainer \
  --image us.icr.io/<your-namespace>/interview-trainer:latest \
  --port 5000 \
  --env IBM_API_KEY=<your-key> \
  --env IBM_PROJECT_ID=<your-project-id>
```

### Option C: Docker (any cloud)
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t interview-trainer .
docker run -p 5000:5000 --env-file .env interview-trainer
```

---

## 🔒 Security Notes

- Never commit your `.env` file — it's listed in `.gitignore`
- Use `FLASK_DEBUG=False` in production
- Set a strong, random `FLASK_SECRET_KEY`
- For production, use a proper database instead of `session_history.json`

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET  /api/health` | GET | Health check + model info |
| `POST /api/upload` | POST | Upload resume/JD file |
| `POST /api/session/start` | POST | Start a new interview session |
| `POST /api/session/answer` | POST | Submit answer, get feedback |
| `GET  /api/session/:id` | GET | Get session state |
| `GET  /api/history` | GET | Session history + trends |
| `POST /api/questions/generate` | POST | Generate standalone questions |
| `POST /api/chat` | POST | Freeform AI coach chat |
| `GET  /api/config` | GET | Get current agent config |
| `POST /api/config` | POST | Update agent config |

---

## 🧰 Tech Stack

- **Backend**: Python 3.11+, Flask 3.0, Flask-CORS
- **AI Model**: IBM Watsonx.ai — `ibm/granite-13b-chat-v2`
- **File Parsing**: PyPDF2 (PDF), python-docx (DOCX)
- **Frontend**: HTML5, Bootstrap 5.3, Vanilla JS
- **Fonts**: Inter (Google Fonts)
- **Icons**: Bootstrap Icons

---

Made with ❤️ using IBM Watsonx.ai + Granite
