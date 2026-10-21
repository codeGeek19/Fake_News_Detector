"""Evaluate the WHOLE system (classifier + evidence + verdict rules) against claims you have ground truth for.

Input CSV columns:  text,language,expected     (expected = fake | real, taken from a fact-checker's verdict; you collect these)
Run (backend must be running):  python eval_pipeline.py --csv claims.csv --api http://localhost:8000

Reports coverage (how often the system commits instead of answering Uncertain), accuracy among committed answers,
a confusion matrix, and per-language numbers. For a retrieval-based verifier these matter more than classifier accuracy alone.
"""
import argparse
import collections
import csv
import json
import time
import urllib.request

PRED = {"likely_fake": "fake", "likely_real": "real", "uncertain": "uncertain"}


def score(rows: list) -> dict:
    """rows: dicts with expected, language, prediction, model_prediction, classifier_available."""
    def summarize(rs, key):
        answered = [r for r in rs if PRED[r[key]] != "uncertain"]
        correct = sum(PRED[r[key]] == r["expected"] for r in answered)
        cm = collections.Counter((r["expected"], PRED[r[key]]) for r in rs)
        return {"n": len(rs), "coverage": len(answered) / len(rs) if rs else 0.0,
                "accuracy_when_answered": correct / len(answered) if answered else None,
                "confusion": {f"expected_{e}__pred_{p}": cm[(e, p)] for e in ("fake", "real") for p in ("fake", "real", "uncertain")}}
    out = {"final_system": summarize(rows, "prediction"), "by_language": {}}
    for lang in sorted({r["language"] for r in rows}):
        out["by_language"][lang] = summarize([r for r in rows if r["language"] == lang], "prediction")
    clf = [r for r in rows if r.get("classifier_available")]
    if clf:
        out["classifier_only"] = summarize(clf, "model_prediction")
    return out


def call(api: str, text: str, language: str) -> dict:
    req = urllib.request.Request(f"{api}/api/analyze", data=json.dumps({"text": text, "language": language}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--out", default=None)
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between requests (be kind to the news APIs and the rate limit)")
    a = ap.parse_args()
    rows = []
    for r in csv.DictReader(open(a.csv, encoding="utf-8")):
        res = call(a.api, r["text"], r.get("language") or "auto")
        rows.append({"expected": r["expected"].strip().lower(), "language": res["language"], "prediction": res["prediction"],
                     "model_prediction": res["model_prediction"], "classifier_available": res["classifier_available"]})
        time.sleep(a.delay)
    result = score(rows)
    print(json.dumps(result, indent=2))
    if a.out:
        json.dump(result, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
