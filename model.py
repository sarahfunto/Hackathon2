import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .dataset import id2label
from .rules import rule_based_override

device = "cuda" if torch.cuda.is_available() else "cpu"


label_desc = {
    "no_counseling": "No genetic counseling recommended (in this prototype).",
    "counseling_recommended": "Genetic counseling recommended (in this prototype).",
    "high_risk_counseling": "High-risk case: genetic counseling strongly recommended (in this prototype).",
}

def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model

def predict_case(tokenizer, model, text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    raw_pred = int(np.argmax(probs))
    final_pred = rule_based_override(text, raw_pred)

    pred_label = id2label[final_pred]
    confidence = float(probs[final_pred])

    return {
        "text": text,
        "pred_id": final_pred,
        "pred_label": pred_label,
        "confidence": confidence,
        "explanation":label_desc[pred_label],
    }
