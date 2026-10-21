"""Cross-check pretrained (or your own) fake-news classifiers yourself, on data you choose.

  python ml/evaluation/compare_pretrained.py \
      --model jy46604790/Fake-News-Bert-Detect:LABEL_0 \
      --model models/en:FAKE \
      --csv data/liar_test.csv --csv data/isot_test.csv

`--model ID[:FAKE_LABEL]` where FAKE_LABEL is the output label (or index) that means FAKE (read the model card!).
CSVs need columns text,label (1 = fake, 0 = real); build them with ml/training/build_english_eval.py.
Look at LIAR (short claims) first: ISOT is easy and in-domain numbers there say little about real news.
"""
import argparse
import json
import os
from metrics import compute_metrics, fake_index


def parse_spec(spec: str):
    """'org/name:LABEL_0' -> ('org/name', 'LABEL_0'); Windows paths like C:\\x\\model are left alone."""
    head, sep, tail = spec.rpartition(":")
    if sep and tail and "/" not in tail and "\\" not in tail and len(head) > 1:
        return head, tail
    return spec, None


def to_markdown(rows: list) -> str:
    cols = ["model", "data", "n", "macro_f1", "accuracy", "roc_auc"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def run(model_ref: str, fake_label, csv_path: str, max_len=256, bs=32, limit=None):
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForSequenceClassification.from_pretrained(model_ref).float().to(dev).eval()
    fi = fake_index(model.config.id2label, fake_label)
    df = pd.read_csv(csv_path)
    if limit:
        df = df.sample(min(limit, len(df)), random_state=42)
    probs = []
    with torch.no_grad():
        for i in range(0, len(df), bs):
            enc = tok(df["text"].iloc[i:i + bs].tolist(), return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(dev)
            probs += torch.softmax(model(**enc).logits, -1)[:, fi].cpu().tolist()
    m = compute_metrics(df["label"].values, [int(p >= 0.5) for p in probs], probs)
    return {"model": model_ref, "data": os.path.basename(csv_path), "n": len(df), "macro_f1": round(m["macro_f1"], 3),
            "accuracy": round(m["accuracy"], 3), "roc_auc": round(m.get("roc_auc", float("nan")), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="ID[:FAKE_LABEL], repeatable")
    ap.add_argument("--csv", action="append", required=True, help="repeatable")
    ap.add_argument("--limit", type=int, default=None, help="score only a random sample of each CSV (faster on CPU)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [run(*parse_spec(m), c, limit=a.limit) for m in a.model for c in a.csv]
    print("\n" + to_markdown(rows))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)
