import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p), "recall": float(r),
           "f1": float(f1), "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
           "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}
    if y_prob is not None and len(set(y_true.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def fake_index(id2label: dict, override=None) -> int:
    """Which output index means FAKE. Pretrained models disagree (LABEL_0 vs LABEL_1), so this is explicit."""
    labels = {int(k): str(v) for k, v in id2label.items()}
    if override is not None:
        o = str(override).strip()
        if o.isdigit() and int(o) in labels:
            return int(o)
        for k, v in labels.items():
            if v.lower() == o.lower():
                return k
        raise SystemExit(f"fake label {override!r} not among {list(labels.values())}")
    return next((k for k, v in labels.items() if "fake" in v.lower()), 1)
