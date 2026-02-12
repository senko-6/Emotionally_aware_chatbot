import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "." 

def run_feeler():
    print("--- 1. Loading The Feeler (Emotion Model) ---")
    
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print("ERROR: I cannot find 'config.json' in this folder!")
        print("Make sure this script is in the SAME folder as your model files.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to("cuda")
        print("✅ Model loaded successfully on RTX 4060!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    test_sentences = [
        "I am so sad and heartbroken today.",
        "I am extremely happy and ecstatic!",
        "I love you so much, you are the best.",
        "I am furious, I hate this so much!",
        "I am terrified, there is a ghost!",
        "Wow! I can't believe that happened, amazing!",
        "The table is made of wood."
    ]

    print("\n--- 2. Decoding the Labels ---")
    print(f"{'TEST SENTENCE':<50} | {'PREDICTED LABEL'}")
    print("-" * 70)

    for text in test_sentences:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_id = logits.argmax().item()
        
        print(f"{text:<50} | LABEL_{predicted_id}")

    print("\n-------------------------------------------")
    print("Write down the mapping above (e.g. Joy = LABEL_1).")
    print("You will need this for the backend code!")

if __name__ == "__main__":
    run_feeler()