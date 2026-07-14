"""
AI-Powered Interview Trainer Agent
Built with Python Flask + IBM Watsonx.ai (Granite Models)
"""

import os
import json
import uuid
import re
from datetime import datetime
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  AGENT INSTRUCTIONS — Customize behavior here
# ─────────────────────────────────────────────────────────────────────────────
AGENT_INSTRUCTIONS = {
    # Tone of the interviewer agent
    # Options: "professional", "friendly", "strict", "encouraging"
    "tone": "professional",

    # Interview type
    # Options: "hr", "technical", "behavioral", "mixed"
    "interview_type": "mixed",

    # Difficulty level
    # Options: "beginner", "intermediate", "advanced"
    "difficulty": "intermediate",

    # Target industry / role
    # Options: "software_engineering", "data_science", "core_engineering",
    #          "finance", "sales", "generic"
    "target_role": "software_engineering",

    # Number of main questions per session (follow-ups are dynamic)
    "questions_per_session": 5,

    # Maximum follow-up questions per main question
    "max_followups": 2,

    # Whether to suggest improved answers for weak responses (score < threshold)
    "suggest_improved_answers": True,
    "improvement_threshold": 6,   # scores below this get a sample answer

    # System persona injected into every prompt
    "persona": (
        "You are an elite technical and HR interviewer with 15+ years of experience "
        "at top-tier companies. You conduct rigorous, insightful interviews and provide "
        "honest, constructive feedback that helps candidates grow."
    ),
}
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
CORS(app, supports_credentials=True)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH_MB", 5)) * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

# In-memory session store  {session_id: {...}}
interview_sessions: dict = {}

# ── IBM Watsonx.ai client ────────────────────────────────────────────────────
def get_watsonx_client():
    try:
        from ibm_watsonx_ai import APIClient, Credentials
        credentials = Credentials(
            url=os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            api_key=os.getenv("IBM_API_KEY"),
        )
        return APIClient(credentials)
    except Exception as e:
        print(f"[Watsonx] Client init error: {e}")
        return None


def call_watsonx(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """Send a prompt to IBM Watsonx Granite and return the response text."""
    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai import Credentials

        credentials = Credentials(
            url=os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            api_key=os.getenv("IBM_API_KEY"),
        )

        model = ModelInference(
            model_id="ibm/granite-4-h-small",
            credentials=credentials,
            project_id=os.getenv("IBM_PROJECT_ID"),
            params={
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "repetition_penalty": 1.1,
                "stop_sequences": ["<|endoftext|>"],
            },
        )
        response = model.generate_text(prompt=prompt) 
        return response.strip() if isinstance(response, str) else str(response)
    except Exception as e:
        print(f"[Watsonx] Inference error: {e}")
        return f"[ERROR] Could not reach IBM Watsonx: {str(e)}"


# ── Prompt builders ──────────────────────────────────────────────────────────
def _role_context(cfg: dict) -> str:
    role_map = {
        "software_engineering": "Software Engineering",
        "data_science": "Data Science / ML",
        "core_engineering": "Core / Mechanical / Electrical Engineering",
        "finance": "Finance / Investment Banking",
        "sales": "Sales / Business Development",
        "generic": "General / Cross-functional",
    }
    type_map = {
        "hr": "HR / Cultural Fit",
        "technical": "Technical / Domain Knowledge",
        "behavioral": "Behavioral (STAR method)",
        "mixed": "Mixed (HR + Technical + Behavioral)",
    }
    return (
        f"Role: {role_map.get(cfg.get('target_role','generic'), cfg.get('target_role','Generic'))}\n"
        f"Interview Type: {type_map.get(cfg.get('interview_type','mixed'), 'Mixed')}\n"
        f"Difficulty: {cfg.get('difficulty','intermediate').capitalize()}\n"
        f"Tone: {cfg.get('tone','professional').capitalize()}"
    )


def build_question_generation_prompt(cfg: dict, context: str = "", count: int = 5) -> str:
    instr = AGENT_INSTRUCTIONS.copy()
    instr.update(cfg)
    return f"""{instr['persona']}

{_role_context(instr)}
{'Context (resume / JD):' + chr(10) + context if context else ''}

Generate exactly {count} high-quality interview questions for the above profile.
Rules:
- Number each question (1. 2. 3. …)
- Match the difficulty and type specified
- Make questions specific, probing, and realistic
- Do NOT include answers
- Output ONLY the numbered questions, nothing else

Questions:"""


def build_followup_prompt(cfg: dict, question: str, answer: str) -> str:
    instr = AGENT_INSTRUCTIONS.copy()
    instr.update(cfg)
    return f"""{instr['persona']}

{_role_context(instr)}

You just asked the candidate: "{question}"
The candidate answered: "{answer}"

As a skilled interviewer, ask ONE sharp follow-up question that:
- Probes deeper into their answer
- Uncovers gaps or validates their claims
- Is natural and conversational

Output ONLY the follow-up question, no preamble."""


def build_evaluation_prompt(cfg: dict, question: str, answer: str) -> str:
    instr = AGENT_INSTRUCTIONS.copy()
    instr.update(cfg)
    return f"""{instr['persona']}

{_role_context(instr)}

Question asked: "{question}"
Candidate's answer: "{answer}"

Evaluate this answer and respond in the following JSON format ONLY:
{{
  "score": <integer 1-10>,
  "strengths": "<brief bullet points of what was good>",
  "weaknesses": "<brief bullet points of what was lacking>",
  "clarity": "<assessment of communication clarity>",
  "confidence": "<assessment of confidence level>",
  "technical_accuracy": "<assessment of technical correctness if applicable>",
  "summary": "<2-3 sentence overall assessment>",
  "improved_answer": "<a strong model answer for this question>"
}}

JSON response:"""


def build_session_summary_prompt(cfg: dict, qa_pairs: list) -> str:
    instr = AGENT_INSTRUCTIONS.copy()
    instr.update(cfg)
    qa_text = "\n".join(
        [f"Q{i+1}: {p['question']}\nA{i+1}: {p['answer']}\nScore: {p.get('score', 'N/A')}/10"
         for i, p in enumerate(qa_pairs)]
    )
    return f"""{instr['persona']}

{_role_context(instr)}

Here is the complete interview session:
{qa_text}

Provide a comprehensive session summary in JSON format:
{{
  "overall_score": <average score as float>,
  "grade": "<A/B/C/D/F>",
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "areas_for_improvement": ["<area 1>", "<area 2>", "<area 3>"],
  "hiring_recommendation": "<Strong Yes / Yes / Maybe / No>",
  "detailed_feedback": "<3-5 sentence holistic feedback>",
  "study_topics": ["<topic 1>", "<topic 2>", "<topic 3>"]
}}

JSON response:"""


# ── Utility helpers ──────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(filepath: str) -> str:
    ext = filepath.rsplit(".", 1)[1].lower()
    try:
        if ext == "txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == "pdf":
            import PyPDF2
            text = []
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        elif ext == "docx":
            from docx import Document
            doc = Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"[Could not extract text: {e}]"
    return ""


def parse_numbered_questions(raw: str) -> list:
    """Extract numbered questions from model output."""
    lines = raw.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if re.match(r"^\d+[\.\)]\s+.+", line):
            q = re.sub(r"^\d+[\.\)]\s+", "", line).strip()
            if q:
                questions.append(q)
    return questions


def safe_parse_json(raw: str) -> dict:
    """Try to parse JSON from model output, tolerating markdown fences."""
    raw = raw.strip()
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Find first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def new_session_store(config: dict) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "config": config,
        "questions": [],
        "current_q_index": 0,
        "followup_count": 0,
        "qa_pairs": [],          # [{question, answer, score, feedback, is_followup}]
        "context": "",           # resume / JD text
        "started_at": datetime.utcnow().isoformat(),
        "completed": False,
        "summary": None,
    }


# ── Session persistence helpers ──────────────────────────────────────────────
HISTORY_FILE = "session_history.json"


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_to_history(summary_record: dict):
    history = load_history()
    history.append(summary_record)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "agent": "Interview Trainer", "model": "ibm/granite-4-h-small"})


# ── Upload resume / JD ───────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF, DOCX, TXT files are allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    text = extract_text_from_file(filepath)
    preview = text[:500] + ("…" if len(text) > 500 else "")
    return jsonify({"success": True, "text": text, "preview": preview, "filename": filename})


# ── Start interview session ──────────────────────────────────────────────────
@app.route("/api/session/start", methods=["POST"])
def start_session():
    data = request.get_json() or {}
    config = {
        "interview_type": data.get("interview_type", AGENT_INSTRUCTIONS["interview_type"]),
        "difficulty":     data.get("difficulty",     AGENT_INSTRUCTIONS["difficulty"]),
        "target_role":    data.get("target_role",    AGENT_INSTRUCTIONS["target_role"]),
        "tone":           data.get("tone",           AGENT_INSTRUCTIONS["tone"]),
        "questions_per_session": int(data.get("questions_per_session",
                                              AGENT_INSTRUCTIONS["questions_per_session"])),
    }
    context = data.get("context", "")   # pasted / uploaded resume or JD text

    sess = new_session_store(config)
    sess["context"] = context

    # Generate questions
    prompt = build_question_generation_prompt(config, context, config["questions_per_session"])
    raw = call_watsonx(prompt, max_tokens=800, temperature=0.8)
    questions = parse_numbered_questions(raw)

    # Fallback if parsing failed
    if not questions:
        questions = [raw.strip()] if raw.strip() else ["Tell me about yourself."]

    sess["questions"] = questions
    interview_sessions[sess["id"]] = sess

    first_q = questions[0] if questions else "Tell me about yourself."
    return jsonify({
        "session_id": sess["id"],
        "message": (
            f"Welcome to your {config['interview_type'].upper()} interview for "
            f"{config['target_role'].replace('_',' ').title()}! "
            f"Difficulty: {config['difficulty'].capitalize()}. "
            f"I'll ask you {len(questions)} questions. Let's begin.\n\n"
            f"**Question 1:** {first_q}"
        ),
        "question": first_q,
        "question_number": 1,
        "total_questions": len(questions),
        "config": config,
    })


# ── Submit answer ────────────────────────────────────────────────────────────
@app.route("/api/session/answer", methods=["POST"])
def submit_answer():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    answer = (data.get("answer") or "").strip()

    if not session_id or session_id not in interview_sessions:
        return jsonify({"error": "Invalid session ID"}), 404
    if not answer:
        return jsonify({"error": "Answer cannot be empty"}), 400

    sess = interview_sessions[session_id]
    if sess["completed"]:
        return jsonify({"error": "Session already completed"}), 400

    cfg = sess["config"]
    current_idx = sess["current_q_index"]
    current_q = sess["questions"][current_idx]
    followup_count = sess["followup_count"]
    max_followups = AGENT_INSTRUCTIONS["max_followups"]

    # Evaluate the answer
    eval_prompt = build_evaluation_prompt(cfg, current_q, answer)
    eval_raw = call_watsonx(eval_prompt, max_tokens=700, temperature=0.3)
    feedback = safe_parse_json(eval_raw)
    score = int(feedback.get("score", 5))

    qa_record = {
        "question": current_q,
        "answer": answer,
        "score": score,
        "feedback": feedback,
        "is_followup": followup_count > 0,
        "timestamp": datetime.utcnow().isoformat(),
    }
    sess["qa_pairs"].append(qa_record)

    # Decide: follow-up or next main question or end
    should_followup = (
        followup_count < max_followups
        and not data.get("skip_followup", False)
    )

    if should_followup:
        fu_prompt = build_followup_prompt(cfg, current_q, answer)
        followup_q = call_watsonx(fu_prompt, max_tokens=200, temperature=0.7).strip()
        sess["followup_count"] += 1
        # Temporarily replace current question with follow-up for next answer evaluation
        sess["questions"].insert(current_idx + 1, followup_q)
        sess["current_q_index"] += 1

        response_msg = _format_feedback_message(feedback, score, cfg)
        response_msg += f"\n\n**Follow-up:** {followup_q}"

        return jsonify({
            "type": "followup",
            "feedback": feedback,
            "score": score,
            "message": response_msg,
            "next_question": followup_q,
            "question_number": sess["current_q_index"] + 1,
            "total_questions": len(sess["questions"]),
            "session_complete": False,
        })
    else:
        # Move to next main question
        sess["followup_count"] = 0
        # Find next non-followup question index
        main_questions = [q for q in sess["questions"] if not _is_followup_marker(q)]
        answered_main = sum(1 for p in sess["qa_pairs"] if not p["is_followup"])

        next_main_idx = None
        seen_main = 0
        for i, q in enumerate(sess["questions"]):
            if not _is_followup_marker(q):
                seen_main += 1
            if seen_main > answered_main:
                next_main_idx = i
                break

        if next_main_idx is None or answered_main >= cfg["questions_per_session"]:
            # End session
            sess["completed"] = True
            sess["completed_at"] = datetime.utcnow().isoformat()

            sum_prompt = build_session_summary_prompt(cfg, sess["qa_pairs"])
            sum_raw = call_watsonx(sum_prompt, max_tokens=800, temperature=0.3)
            summary = safe_parse_json(sum_raw)
            if not summary:
                scores = [p["score"] for p in sess["qa_pairs"] if p.get("score")]
                avg = round(sum(scores) / len(scores), 1) if scores else 5
                summary = {"overall_score": avg, "grade": _score_to_grade(avg)}
            sess["summary"] = summary

            # Persist to history
            history_record = {
                "session_id": sess["id"],
                "date": sess["started_at"],
                "config": cfg,
                "overall_score": summary.get("overall_score", 0),
                "grade": summary.get("grade", "N/A"),
                "total_questions": len(sess["qa_pairs"]),
                "summary": summary,
            }
            save_to_history(history_record)

            fb_msg = _format_feedback_message(feedback, score, cfg)
            return jsonify({
                "type": "session_complete",
                "feedback": feedback,
                "score": score,
                "message": fb_msg + "\n\n✅ **Interview Complete!** See your full report below.",
                "session_complete": True,
                "summary": summary,
                "all_qa": sess["qa_pairs"],
            })
        else:
            sess["current_q_index"] = next_main_idx
            next_q = sess["questions"][next_main_idx]
            q_num = answered_main + 1

            fb_msg = _format_feedback_message(feedback, score, cfg)
            fb_msg += f"\n\n**Question {q_num + 1}:** {next_q}"

            return jsonify({
                "type": "next_question",
                "feedback": feedback,
                "score": score,
                "message": fb_msg,
                "next_question": next_q,
                "question_number": q_num + 1,
                "total_questions": cfg["questions_per_session"],
                "session_complete": False,
            })


def _is_followup_marker(q: str) -> bool:
    return False   # All questions are stored flat; follow-up tracking is via qa_pairs


def _score_to_grade(score: float) -> str:
    if score >= 9: return "A+"
    if score >= 8: return "A"
    if score >= 7: return "B+"
    if score >= 6: return "B"
    if score >= 5: return "C"
    if score >= 4: return "D"
    return "F"


def _format_feedback_message(feedback: dict, score: int, cfg: dict) -> str:
    suggest = AGENT_INSTRUCTIONS["suggest_improved_answers"]
    threshold = AGENT_INSTRUCTIONS["improvement_threshold"]

    msg = f"**Score: {score}/10**\n\n"
    if feedback.get("strengths"):
        msg += f"✅ **Strengths:** {feedback['strengths']}\n\n"
    if feedback.get("weaknesses"):
        msg += f"⚠️ **Areas to Improve:** {feedback['weaknesses']}\n\n"
    if feedback.get("summary"):
        msg += f"💬 **Feedback:** {feedback['summary']}\n\n"
    if suggest and score < threshold and feedback.get("improved_answer"):
        msg += f"💡 **Sample Strong Answer:**\n_{feedback['improved_answer']}_\n\n"
    return msg.strip()


# ── Get session state ────────────────────────────────────────────────────────
@app.route("/api/session/<session_id>", methods=["GET"])
def get_session(session_id):
    sess = interview_sessions.get(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({
        "session_id": sess["id"],
        "config": sess["config"],
        "total_questions": len(sess["questions"]),
        "answered": len(sess["qa_pairs"]),
        "completed": sess["completed"],
        "summary": sess.get("summary"),
        "qa_pairs": sess["qa_pairs"],
        "started_at": sess["started_at"],
    })


# ── Session history / progress tracker ──────────────────────────────────────
@app.route("/api/history", methods=["GET"])
def get_history():
    history = load_history()
    # Compute trends
    if len(history) >= 2:
        recent = history[-5:]
        scores = [h["overall_score"] for h in recent if h.get("overall_score")]
        trend = "improving" if len(scores) >= 2 and scores[-1] > scores[0] else \
                "declining" if len(scores) >= 2 and scores[-1] < scores[0] else "stable"
    else:
        trend = "not enough data"
    return jsonify({"history": history, "trend": trend, "total_sessions": len(history)})


# ── Generate standalone questions (no session) ───────────────────────────────
@app.route("/api/questions/generate", methods=["POST"])
def generate_questions():
    data = request.get_json() or {}
    config = {
        "interview_type": data.get("interview_type", AGENT_INSTRUCTIONS["interview_type"]),
        "difficulty":     data.get("difficulty",     AGENT_INSTRUCTIONS["difficulty"]),
        "target_role":    data.get("target_role",    AGENT_INSTRUCTIONS["target_role"]),
        "tone":           data.get("tone",           AGENT_INSTRUCTIONS["tone"]),
    }
    context = data.get("context", "")
    count = int(data.get("count", 10))
    prompt = build_question_generation_prompt(config, context, count)
    raw = call_watsonx(prompt, max_tokens=1000, temperature=0.8)
    questions = parse_numbered_questions(raw)
    return jsonify({"questions": questions, "raw": raw, "config": config})


# ── Agent chat (freeform) ────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message is required"}), 400
    history = data.get("history", [])

    history_text = ""
    for h in history[-6:]:
        role = "User" if h.get("role") == "user" else "Interviewer"
        history_text += f"{role}: {h.get('content','')}\n"

    prompt = (
        f"{AGENT_INSTRUCTIONS['persona']}\n\n"
        f"You are helping a candidate prepare for a {AGENT_INSTRUCTIONS['target_role'].replace('_',' ')} interview.\n"
        f"Conversation so far:\n{history_text}\n"
        f"User: {message}\n"
        f"Interviewer:"
    )
    reply = call_watsonx(prompt, max_tokens=600, temperature=0.7)
    return jsonify({"reply": reply, "timestamp": datetime.utcnow().isoformat()})


# ── Agent configuration ──────────────────────────────────────────────────────
@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(AGENT_INSTRUCTIONS)


@app.route("/api/config", methods=["POST"])
def update_config():
    global AGENT_INSTRUCTIONS
    data = request.get_json() or {}
    allowed_keys = {
        "tone", "interview_type", "difficulty", "target_role",
        "questions_per_session", "max_followups",
        "suggest_improved_answers", "improvement_threshold",
    }
    for k, v in data.items():
        if k in allowed_keys:
            AGENT_INSTRUCTIONS[k] = v
    return jsonify({"success": True, "config": AGENT_INSTRUCTIONS})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    print(f"\n🚀 Interview Trainer Agent running on http://localhost:{port}")
    print(f"   Model : ibm/granite-4-h-small")
    print(f"   Config: {AGENT_INSTRUCTIONS['interview_type']} | "
          f"{AGENT_INSTRUCTIONS['difficulty']} | {AGENT_INSTRUCTIONS['target_role']}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
