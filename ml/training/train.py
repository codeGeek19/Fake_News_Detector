"""Fine-tune a transformer classifier (Colab-friendly: fp16, per-epoch checkpoints on Drive, auto-resume).
Saves {out_dir}/final with model + tokenizer + metadata.json (version, training cutoff, test metrics)."""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
from metrics import compute_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)          # google/muril-base-cased | xlm-roberta-base | roberta-base ...
ap.add_argument("--lang", required=True)           # en | hi
ap.add_argument("--train_csv", required=True)
ap.add_argument("--val_csv", required=True)
ap.add_argument("--test_csv", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--version", default="v1.0")
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max_train_samples", type=int, default=None)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)


def load(path, limit=None):
    df = pd.read_csv(path)
    if limit and len(df) > limit:
        df = df.sample(limit, random_state=42)
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    return df, ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len), batched=True, remove_columns=["text"])


train_df, train = load(a.train_csv, a.max_train_samples)
_, val = load(a.val_csv)
_, test = load(a.test_csv)

model = AutoModelForSequenceClassification.from_pretrained(
    a.model, num_labels=2, id2label={0: "REAL", 1: "FAKE"}, label2id={"REAL": 0, "FAKE": 1})


def hf_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"macro_f1": float(f1_score(p.label_ids, preds, average="macro")),
            "accuracy": float((preds == p.label_ids).mean())}


import math
warmup_steps = int(0.1 * math.ceil(len(train) / a.bs) * a.epochs)   # works on old and new transformers
args = TrainingArguments(
    output_dir=os.path.join(a.out_dir, "checkpoints"), num_train_epochs=a.epochs,
    per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs * 2, learning_rate=a.lr,
    weight_decay=0.01, warmup_steps=warmup_steps, fp16=torch.cuda.is_available(),
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="macro_f1", logging_steps=50, report_to="none")

trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=hf_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])

last = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
trainer.train(resume_from_checkpoint=last)          # survives Colab disconnects

out = trainer.predict(test)
probs = torch.softmax(torch.tensor(out.predictions), -1)[:, 1].numpy()
report = compute_metrics(out.label_ids, out.predictions.argmax(-1), probs)

final = os.path.join(a.out_dir, "final")
trainer.save_model(final)
tok.save_pretrained(final)
dates = pd.to_datetime(train_df["date"], errors="coerce", format="mixed") if "date" in train_df.columns else None
cutoff = dates.max().strftime("%Y-%m-%d") if dates is not None and dates.notna().any() else "unknown"
json.dump({"model_version": a.version, "base_model": a.model, "language": a.lang, "max_len": a.max_len,
           "labels": {"0": "REAL", "1": "FAKE"}, "training_data_cutoff": cutoff,
           "val_macro_f1": trainer.state.best_metric, "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "train_size": len(train),
           "test_metrics": report}, open(os.path.join(final, "metadata.json"), "w"), indent=2, ensure_ascii=False)
print(json.dumps(report, indent=2))
