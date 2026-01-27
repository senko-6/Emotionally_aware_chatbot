# memory.py
from collections import defaultdict, deque

MAX_HISTORY = 5

memory_store = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

def get_history(user_id):
    return list(memory_store[user_id])

def add_turn(user_id, user_text, bot_text):
    memory_store[user_id].append({
        "User": user_text,
        "Assistant": bot_text
    })