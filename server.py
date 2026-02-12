from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sqlite3
import json
from datetime import datetime
import models

DB_NAME = "chatbot.db"

SYSTEM_PERSONA = (
    "You are a close friend, not an AI assistant. Your name is Empath. "
    "Rules for speaking:\n"
    "1. Speak casually and briefly (1-2 sentences max).\n"
    "2. lowercase is okay. slang is okay.\n"
    "3. NEVER give numbered lists of advice.\n"
    "4. NEVER say 'It is important to...' or 'I understand'. Just react naturally.\n"
    "5. If the user fights, take their side or ask what happened. Don't lecture them.\n"
    "6. Use the user's name if you know it."
)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, sender TEXT, message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS session_summaries
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, date_str TEXT, time_str TEXT, emotion_json TEXT, topic_summary TEXT)''')
    conn.commit()
    conn.close()

def save_message(user_id, sender, message):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (user_id, sender, message) VALUES (?, ?, ?)", (user_id, sender, message))
    conn.commit()
    conn.close()

def get_recent_history(user_id, limit=20):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT sender, message FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()[::-1]
    conn.close()
    return rows

def get_daily_summary(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT topic_summary FROM session_summaries WHERE user_id = ? ORDER BY id DESC LIMIT 5", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    stories = [row[0] for row in rows if row[0]]
    if stories:
        combined = "; ".join(stories)
        print(f"✅ FOUND MEMORY for {user_id}: {combined}")
        return combined 
    
    return ""

def save_session_summary(user_id, emotions, summary_text):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now()
    cursor.execute("INSERT INTO session_summaries (user_id, date_str, time_str, emotion_json, topic_summary) VALUES (?, ?, ?, ?, ?)",
                   (user_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), json.dumps(emotions), summary_text))
    conn.commit()
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 STARTING UP: Loading Database and Brain...")
    init_db()
    models.load_models()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    user_name: str
    message: str

class SessionEndRequest(BaseModel):
    user_id: str
    emotions: dict

@app.post("/chat")
async def chat(request: ChatRequest):
    user_id = request.user_id.lower()
    
    history = get_recent_history(user_id)
    daily_summary = get_daily_summary(user_id)
    label = models.detect_emotion(request.message)
    
    prompt_message = request.message
    trigger_words = ["remember", "who", "what", "when", "did i"]
    
    if daily_summary and any(w in request.message.lower() for w in trigger_words):
         print(f"⚡ INJECTING MEMORY: {daily_summary}")
         prompt_message = (
             f"User Question: {request.message}\n"
             f"History you know: {daily_summary}.\n"
             f"Instruction: Answer naturally. Don't say 'According to my records'. Just say it."
         )

    messages = [{"role": "system", "content": SYSTEM_PERSONA}]
    
    for row in history:
        role = "user" if row[0] == "user" else "assistant"
        messages.append({"role": role, "content": row[1]})
    
    messages.append({"role": "user", "content": prompt_message})
    
    bot_reply = models.generate_text(messages, temperature=0.65)
    
    save_message(user_id, "user", request.message)
    save_message(user_id, "bot", bot_reply)
    
    return {"reply": bot_reply, "emotion": label}

@app.post("/end_session")
async def end_session(request: SessionEndRequest):
    user_id = request.user_id.lower()
    recent_chat = get_recent_history(user_id, limit=15)
    if not recent_chat: return {"status": "no data"}
    
    chat_text = "\n".join([f"{r[0]}: {r[1]}" for r in recent_chat])
    
    summary_messages = [
        {"role": "user", "content": f"Summarize the events of this chat in 1 sentence:\n\n{chat_text}"}
    ]
    summary = models.generate_text(summary_messages, max_tokens=60, temperature=0.1)
    
    save_session_summary(user_id, request.emotions, summary)
    print(f"📝 Summary Saved for {user_id}: {summary}") 
    
    return {"status": "saved"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)