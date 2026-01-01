import pandas as pd
from datasets import Dataset, DatasetDict

id2label = {
    0: "no_counseling",
    1: "counseling_recommended",
    2: "high_risk_counseling",
}
label2id = {v: k for k, v in id2label.items()}
label_names = [id2label[i] for i in sorted(id2label.keys())]

def load_genetic_dataset(csv_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    df = pd.read_csv(csv_path)

    df_model = df[["case_text", "label"]].rename(
        columns={"case_text": "text", "label": "label_id"}
    )

    dataset = Dataset.from_pandas(df_model)
    dataset = dataset.shuffle(seed=seed)
    train_test = dataset.train_test_split(test_size=test_size, seed=seed)

    ds = DatasetDict(
        train=train_test["train"],
        validation=train_test["test"],
    )
    return ds
