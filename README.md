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

The model is not stored in this repo (too large). It is loaded automatically from Hugging Face
The fine-tuned DistilBERT model used by `main.py` is hosted on Hugging Face:

 https://huggingface.co/sarahfunto/genetic-lora-merged

To load it in your code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = "sarahfunto/genetic-lora-merged"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
```
---

## 🚀 How to Run

Follow these steps to run the genetic counseling triage assistant locally.

### 1. Clone the repository
```python
```bash
git clone https://github.com/sarahfunto/Hackathon2.git
cd Hackathon2
### 2. Install dependencies
pip install -r requirements.txt
### 3. Run the assistant
python main.py
###4. Enter a case description
When the program starts, paste a short description (in English), e.g.:
Patient 33 years old. Mother breast cancer at 39.
Maternal aunt ovarian cancer at 52.
Then press ENTER on an empty line to run.
###Example output
=== MODEL PREDICTION ===
{'pred_label': 'high_risk_counseling', 'confidence': 0.91, ...}

=== RELEVANT GUIDELINES ===
- Multiple early-onset breast cancers in first-degree relatives...
- Known mutations such as BRCA1/2 justify genetic counseling...
WARNING : This tool is not a medical device and is built from synthetic data only
```

## Project structure

Hackathon2/
├── data/
│   └── genetic_cases.csv                # Synthetic genetic triage dataset
├── models/
│   └── model hosted on Hugging Face            
├── notebooks/
│   └── Hackathon2_synthetic_dataset.ipynb   # Training & exploration notebook
├── src/
│   ├── dataset.py                       # Dataset loading helpers (id2label, etc.)
│   ├── model.py                         # Model loading + predict_case()
│   ├── rules.py                         # Business rules (high-risk overrides)
│   └── rag.py                           # Guidelines, embeddings, FAISS, explain_case()
├── main.py                              # Simple CLI demo
├── requirements.txt
└── README.md
