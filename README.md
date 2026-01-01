# Genetic Counseling Triage Assistant (Educational Prototype)

## Description 

This project is an educational prototype of a genetic counseling triage assistant.

Given a short description of a patient's personal and family history, the system:

1. Predicts a priority level for genetic counseling:
   - `no_counseling`
   - `counseling_recommended`
   - `high_risk_counseling`
2. Applies a simple domain rule layer to override some obviously high-risk cases (e.g. multiple early-onset breast/ovarian cancers, BRCA mutations).
3. Uses a small RAG-like module (Retrieval-Augmented Generation) to retrieve synthetic genetic guidelines.
4. Returns the prediction and the most relevant guidelines.

IMPORTANT  
This project is based only on synthetic, simplified data.  
It must never be used for real medical decisions.  
It is for training / hackathon / educational purposes only.

---

##  Model Download (Hugging Face)

The fine-tuned DistilBERT model used by `main.py` is hosted on Hugging Face:

 https://huggingface.co/sarahfunto/genetic-lora-merged

To load it in your code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```
model_dir = "sarahfunto/genetic-lora-merged"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)



## Project structure

Hackathon2/
├── data/
│   └── genetic_cases.csv                # Synthetic genetic triage dataset
├── models/
│   └── genetic_lora_model/              # (Optional) Local model folder if downloaded
├── notebooks/
│   └── Hackathon2_synthetic_dataset_clean.ipynb   # Training & exploration notebook
├── src/
│   ├── dataset.py                       # Dataset loading helpers (id2label, etc.)
│   ├── model.py                         # Model loading + predict_case()
│   ├── rules.py                         # Business rules (high-risk overrides)
│   └── rag.py                           # Guidelines, embeddings, FAISS, explain_case()
├── main.py                              # Simple CLI demo
├── requirements.txt
└── README.md
