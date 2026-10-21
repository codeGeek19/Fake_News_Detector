"""Evaluate a model (local folder OR Hugging Face id) on ANY csv with columns text,label (label: 1 = fake, 0 = real).
  python evaluate.py --model models/en --csv data/en_test.csv
  python evaluate.py --model jy46604790/Fake-News-Bert-Detect --fake_label LABEL_0 --csv data/liar_test.csv --out r.json
Use it for in-domain, cross-domain and temporal tests, and to compare the pretrained model with one you trained."""
import argparse
import json
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from metrics import compute_metrics, fake_index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "--model_dir", dest="model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fake_label", default=None, help="name or index of the FAKE output (default: a label containing 'fake', else 1)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).float().to(dev).eval()
    fi = fake_index(model.config.id2label, a.fake_label)
    df = pd.read_csv(a.csv)
    probs = []
    with torch.no_grad():
        for i in range(0, len(df), a.bs):
            enc = tok(df["text"].iloc[i:i + a.bs].tolist(), return_tensors="pt", truncation=True,
                      max_length=a.max_len, padding=True).to(dev)
            probs += torch.softmax(model(**enc).logits / a.temperature, -1)[:, fi].cpu().tolist()
    res = compute_metrics(df["label"].values, [int(p >= 0.5) for p in probs], probs)
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
