from sentence_transformers import SentenceTransformer
import faiss

from .model import predict_case

guidelines = [
    "Multiple early-onset breast or ovarian cancers in first-degree relatives suggest hereditary cancer.",
    "Known mutations such as BRCA1, BRCA2, MLH1, MSH2 justify genetic counseling.",
    "Clusters of cancer across generations can indicate inherited risk.",
    "Early-onset colorectal cancer can suggest hereditary colorectal syndromes.",
    "This prototype is educational only and not a medical tool.",
]

def build_rag_index(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(guidelines, convert_to_numpy=True)

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    return embed_model, index

def retrieve_rules(embed_model, index, case_text: str, k: int = 3):
    emb = embed_model.encode([case_text], convert_to_numpy=True)
    dist, idx = index.search(emb, k)

    return [{"guideline": guidelines[i], "distance": float(d)}
            for d, i in zip(dist[0], idx[0])]

def explain_case(tokenizer, model, embed_model, index, case_text: str):
    pred = predict_case(tokenizer, model, case_text)
    rules = retrieve_rules(embed_model, index, case_text)

    return {
        "prediction": pred,
        "rules": rules,
    }
