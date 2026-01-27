import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)

# =========================
# EMOTION LABELS (UNCHANGED)
# =========================
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# =========================
# CHEAT SHEET (UNCHANGED)
# =========================
CHEAT_SHEET = {
    "admiration": ["praise", "acknowledge"],
    "amusement": ["joke", "lighten"],
    "anger": ["calm", "de_escalate"],
    "annoyance": ["empathize", "soothe"],
    "approval": ["agree", "validate"],
    "caring": ["support", "nurture"],
    "confusion": ["clarify", "explain"],
    "curiosity": ["inform", "explore"],
    "desire": ["encourage", "motivate"],
    "disappointment": ["empathize", "console"],
    "disapproval": ["caution", "advise"],
    "disgust": ["distance", "warn"],
    "embarrassment": ["reassure", "comfort"],
    "excitement": ["celebrate", "energize"],
    "fear": ["calm", "reassure"],
    "gratitude": ["acknowledge", "reciprocate"],
    "grief": ["comfort", "validate"],
    "joy": ["celebrate", "share"],
    "love": ["affirm", "support"],
    "nervousness": ["reassure", "encourage"],
    "optimism": ["encourage", "uplift"],
    "pride": ["acknowledge", "praise"],
    "realization": ["explain", "highlight"],
    "relief": ["affirm", "reassure"],
    "remorse": ["acknowledge", "console"],
    "sadness": ["empathize", "validate"],
    "surprise": ["explain", "react"],
    "neutral": ["acknowledge", "respond"]
}

# =========================
# LOAD FEELER MODEL
# =========================
FEELER_MODEL_PATH = "senko3485/feeler"

feeler_tokenizer = AutoTokenizer.from_pretrained(FEELER_MODEL_PATH)
if feeler_tokenizer.pad_token is None:
    feeler_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

feeler_model = AutoModelForSequenceClassification.from_pretrained(
    FEELER_MODEL_PATH
)
feeler_model.eval()

# =========================
# LOAD GENERATOR MODEL
# =========================
GENERATOR_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
generator_model = AutoModelForCausalLM.from_pretrained(
    GENERATOR_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# =========================
# FEELER: EMOTION PREDICTION
# =========================
def predict_emotion(text: str) -> str:
    tokens = feeler_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    tokens = {k: v.to(feeler_model.device) for k, v in tokens.items()}

    with torch.no_grad():
        logits = feeler_model(**tokens).logits

    probs = torch.sigmoid(logits)[0]

    emotion_scores = {
        label: probs[i].item()
        for i, label in enumerate(EMOTION_LABELS)
        if probs[i].item() > 0.1
    }

    if not emotion_scores:
        return "neutral"

    # highest probability emotion
    return max(emotion_scores, key=emotion_scores.get)

# =========================
# GENERATOR: RESPONSE
# =========================
def generate_response(
    user_text: str,
    emotion: str,
    conversation_history: list | None = None
) -> str:

    strategy_plan = " and ".join(
        CHEAT_SHEET.get(emotion, ["acknowledge"])
    )

    history_text = ""
    if conversation_history:
        last_messages = conversation_history[-5:]
        history_text = "\n".join(
            f"User: {m['User']}\nAssistant: {m['Assistant']}"
            for m in last_messages
        )

    prompt = f"""
You are an empathetic assistant.

Conversation so far:
{history_text}

Now respond to the latest user message only.

User: {user_text}
Emotion: {emotion}

You MUST follow this response strategy:
{strategy_plan}

Rules:
- First sentence must acknowledge emotion
- Do NOT mention emotion, strategy, or classification
- Do NOT give advice unless asked
- Use warm, human language
- Sound like a calm human, not a therapist
- 2–4 sentences only

Assistant:
""".strip()

    inputs = generator_tokenizer(
        prompt,
        return_tensors="pt"
    ).to(generator_model.device)

    output = generator_model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=generator_tokenizer.eos_token_id
    )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    response = generator_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()