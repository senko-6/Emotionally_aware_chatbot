import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

GEN_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" 
EMOTION_MODEL_NAME = "SamLowe/roberta-base-go_emotions"

HF_TOKEN = "" 

gen_model = None
gen_tokenizer = None
emotion_model = None
emotion_tokenizer = None

EMOTION_MAP = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
    19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
    23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}

def load_models():
    global gen_model, gen_tokenizer, emotion_model, emotion_tokenizer
    print("⏳ Loading Models... (This uses your GPU)")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=False, token=HF_TOKEN)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME, 
        device_map="auto", 
        quantization_config=bnb_config,
        token=HF_TOKEN
    )
    
    emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME).to("cuda")
    
    print("✅ System Ready!")

def detect_emotion(text):
    if not emotion_model: return "neutral"
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = EMOTION_MAP.get(probs.argmax().item(), "neutral")
    
    text_lower = text.lower()
    if label == "neutral" and any(w in text_lower for w in ["fight", "argue", "hate", "noise", "annoying"]):
        return "annoyance"
    return label

def generate_text(messages, max_tokens=150, temperature=0.7):
    if not gen_model: return "Loading..."
    
    formatted_messages = []
    system_instruction = ""
    
    for m in messages:
        if m['role'] == 'system':
            system_instruction = m['content']
            break
            
    first_user_found = False
    for m in messages:
        if m['role'] == 'user':
            content = m['content']
            if not first_user_found and system_instruction:
                content = f"Instruction: {system_instruction}\n\nUser Question: {content}"
                first_user_found = True
            formatted_messages.append({"role": "user", "content": content})
        elif m['role'] == 'assistant':
            formatted_messages.append({"role": "assistant", "content": m['content']})
            
    prompt = gen_tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
    inputs = gen_tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=True, 
            temperature=temperature,
            pad_token_id=gen_tokenizer.eos_token_id,
        )
        
    response = gen_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()