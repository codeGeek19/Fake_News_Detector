"""Temperature-scale a trained classifier so its confidence is more honest, and store the value in its metadata.json
(the app divides logits by this temperature at inference).

  python ml/evaluation/calibrate.py --model_dir models/dev --csv data/dev_val.csv
Fit on VALIDATION data, never on the test set.
"""
import argparse
import json
import math
import os
import numpy as np


def _nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    z = logits / T
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


def fit_temperature(logits, labels, lo: float = 0.05, hi: float = 10.0, iters: int = 60) -> float:
    """1-D golden-section search for the temperature minimising negative log-likelihood (the NLL is unimodal in T)."""
    logits, y = np.asarray(logits, dtype=float), np.asarray(labels, dtype=int)
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(iters):
        if _nll(logits, y, c) < _nll(logits, y, d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    return round((a + b) / 2, 4)


if __name__ == "__main__":
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--csv", required=True, help="validation csv with text,label")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_dir).float().to(dev).eval()
    df = pd.read_csv(a.csv)
    out = []
    with torch.no_grad():
        for i in range(0, len(df), a.bs):
            enc = tok(df["text"].iloc[i:i + a.bs].tolist(), return_tensors="pt", truncation=True, max_length=a.max_len, padding=True).to(dev)
            out.append(model(**enc).logits.float().cpu().numpy())
    logits = np.concatenate(out)
    T = fit_temperature(logits, df["label"].values)
    print(f"NLL before: {_nll(logits, df['label'].values.astype(int), 1.0):.4f}   temperature: {T}   NLL after: {_nll(logits, df['label'].values.astype(int), T):.4f}")
    path = os.path.join(a.model_dir, "metadata.json")
    meta = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    meta["temperature"] = T
    json.dump(meta, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("saved temperature to", path)
