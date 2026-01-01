def rule_based_override(text: str, model_pred: int) -> int:
    t = text.lower()

    if ("mother" in t and "breast cancer" in t and any(x in t for x in ["38", "39", "before 40"])) \
       and ("ovarian cancer" in t or "aunt" in t):
        return 2

    if any(gene in t for gene in ["brca1", "brca2", "tp53", "mlh1", "msh2", "pms2"]):
        return 2

    if "multiple" in t and "cancer" in t and ("family" in t or "relatives" in t):
        return 2

    return model_pred
