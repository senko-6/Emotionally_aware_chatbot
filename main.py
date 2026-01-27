# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uuid
import time

from models import predict_emotion, generate_response

app = FastAPI(title="Emotional Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_memory: Dict[str, List[Dict]] = {}

MAX_CONTEXT_TURNS = 5

class ChatRequest(BaseModel):
    user_id: str
    user_name: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    emotion: str
    conversation_id: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Controller flow:
    1. Fetch conversation memory
    2. Predict emotion (Feeler)
    3. Generate response (Generator)
    4. Update memory
    5. Return response
    """

    user_id = req.user_id
    user_message = req.message.strip()

    if not user_message:
        return ChatResponse(
            reply="I’m here whenever you’re ready to talk.",
            emotion="neutral",
            conversation_id=user_id
        )

    # Initialize memory if new user
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []

    history = conversation_memory[user_id]

    emotion = predict_emotion(user_message)

    response = generate_response(
        user_text=user_message,
        emotion=emotion,
        conversation_history=history[-MAX_CONTEXT_TURNS:]
    )

    history.append({
        "User": user_message,
        "Assistant": response,
        "emotion": emotion,
        "timestamp": time.time()
    })

    # Trim memory to avoid explosion
    if len(history) > 50:
        conversation_memory[user_id] = history[-50:]

    return ChatResponse(
        reply=response,
        emotion=emotion,
        conversation_id=user_id
    )