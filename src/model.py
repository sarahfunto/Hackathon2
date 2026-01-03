import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model(model_id: str | None = None, device: str | None = None):
    """
    Load tokenizer and model from Hugging Face or local directory.
    """
        if model_id is None:
        model_id = os.environ.get("MODEL_ID", "sarahfunto/genetic-lora-merged")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
   
    return tokenizer, model, device

def predict_case(tokenizer, model, text: str):
    """
    Predict genetic counseling priority for a given case text.
    """
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
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    confidence = float(probs[pred_id])

    return {
        "pred_id": pred_id,
        "confidence": confidence
    }


