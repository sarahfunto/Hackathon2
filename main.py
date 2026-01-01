
from src.model import load_model
from src.rag import build_rag_index, explain_case


def main():
    # 1) Load the fine-tuned model and tokenizer
    model_dir = "models/genetic_lora_model"
    tokenizer, model = load_model(model_dir)

    # 2) Build RAG components (SentenceTransformer + FAISS index)
    embed_model, index = build_rag_index()

    print("===============================================")
    print(" Genetic Counseling Triage Assistant (DEMO) ")
    print("===============================================")
    print()
    print("Paste a short case description in ENGLISH")
    print("(family history, personal history, etc.)")
    print("Then press ENTER on an empty line to run.")
    print()

    # Read multi-line input
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        # Empty line → stop reading
        if line.strip() == "":
            break

        lines.append(line)

    case_text = "\n".join(lines).strip()

    if not case_text:
        print("No input provided. Exiting.")
        return

    # 3) Run full explanation pipeline
    result = explain_case(tokenizer, model, embed_model, index, case_text)

    print("\n=== MODEL PREDICTION ===")
    print(result["prediction"])   # dict with pred_label, confidence, etc.

    print("\n=== RELEVANT GUIDELINES ===")
    for r in result["rules"]:
        print(f"- {r['guideline']}  (distance={r['distance']:.4f})")

    print("\n⚠️ This is an educational prototype only, based on synthetic data.")
    print("   It must NOT be used for real medical decisions.\n")


if __name__ == "__main__":
    main()
