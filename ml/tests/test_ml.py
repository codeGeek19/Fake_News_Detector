import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from metrics import compute_metrics  # noqa: E402


def test_metrics_perfect():
    m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], [.1, .9, .2, .8])
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["roc_auc"] == 1 and m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_metrics_imbalanced():
    m = compute_metrics([0] * 8 + [1] * 2, [0] * 8 + [0, 1])
    assert m["accuracy"] == .9 and m["macro_f1"] < m["accuracy"]


def test_prep_removes_near_duplicates(tmp_path):
    import pandas as pd, subprocess
    base = "The finance ministry announced a new policy on digital payments affecting millions of users across the country today."
    rows = [(base, "real", "2020-01-01"), (base + " Officials said more.", "real", "2020-01-02"),
            ("Completely different story about a cricket tournament final held in the city stadium on Sunday night.", "fake", "2020-01-03"),
            ("Another unrelated report on monsoon rainfall levels in the western ghats region this season again.", "real", "2020-01-04"),
            ("A fourth separate piece about a new metro line opening in the capital city after many years.", "fake", "2020-01-05"),
            ("Fifth article regarding a school curriculum revision announced by the state education board recently.", "real", "2020-01-06")]
    pd.DataFrame(rows, columns=["t", "l", "d"]).to_csv(tmp_path / "in.csv", index=False)
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "training", "prep_data.py"),
                        "--input", str(tmp_path / "in.csv"), "--text_col", "t", "--label_col", "l", "--fake_values", "fake",
                        "--date_col", "d", "--near_dup_threshold", "0.7", "--out_prefix", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    total = sum(len(pd.read_csv(tmp_path / f"o_{s}.csv")) for s in ("train", "val", "test"))
    assert total == 5                                    # near-duplicate removed
    assert pd.read_csv(tmp_path / "o_test.csv")["date"].min() >= pd.read_csv(tmp_path / "o_train.csv")["date"].max()  # temporal


def test_pipeline_scoring_separates_coverage_from_accuracy():
    from eval_pipeline import score
    rows = [{"expected": "fake", "language": "en", "prediction": "likely_fake", "model_prediction": "likely_fake", "classifier_available": True},
            {"expected": "fake", "language": "en", "prediction": "uncertain", "model_prediction": "likely_real", "classifier_available": True},
            {"expected": "real", "language": "hi", "prediction": "likely_fake", "model_prediction": "uncertain", "classifier_available": False},
            {"expected": "real", "language": "hi", "prediction": "likely_real", "model_prediction": "uncertain", "classifier_available": False}]
    r = score(rows)
    assert r["final_system"]["coverage"] == 0.75 and abs(r["final_system"]["accuracy_when_answered"] - 2 / 3) < 1e-9
    assert r["by_language"]["en"]["coverage"] == 0.5 and r["by_language"]["hi"]["accuracy_when_answered"] == 0.5
    assert r["classifier_only"]["n"] == 2 and r["classifier_only"]["accuracy_when_answered"] == 0.5
    assert r["final_system"]["confusion"]["expected_fake__pred_uncertain"] == 1


def _run_prep(tmp_path, rows, extra):
    import pandas as pd, subprocess
    pd.DataFrame(rows, columns=["t", "l", "d"]).to_csv(tmp_path / "in.csv", index=False)
    return subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "training", "prep_data.py"),
                           "--input", str(tmp_path / "in.csv"), "--text_col", "t", "--label_col", "l", "--out_prefix", str(tmp_path / "o")] + extra,
                          capture_output=True, text=True)


def test_prep_mixed_date_formats_still_temporal(tmp_path):
    base = "Distinct story number {} about the municipal budget and road repairs approved by the council after long debate today."
    dates = ["December 31, 2017", "Dec 30, 2017", "Nov 5, 2017", "March 3, 2017", "Feb 1, 2017", "January 9, 2017", "Oct 2, 2017", "June 7, 2017", "May 6, 2017", "April 4, 2017"]
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    rows = [(base.format(w) + " " + w * 3 + " unique " + str(i) * 5, "fake" if i % 2 else "real", d) for i, (w, d) in enumerate(zip(words, dates))]
    r = _run_prep(tmp_path, rows, ["--fake_values", "fake", "--date_col", "d"])
    assert "TEMPORAL" in r.stdout, r.stdout


def test_prep_real_values_drops_ambiguous_labels(tmp_path):
    rows = [(f"Article {i} " + "word " * 12 + str(i) * 7, lab, f"2020-01-{i + 1:02d}")
            for i, lab in enumerate(["False", "True", "Misleading", "Unproven", "False", "True", "Mixture", "False"])]
    r = _run_prep(tmp_path, rows, ["--fake_values", "false,misleading", "--real_values", "true"])
    assert "dropping 2 rows" in r.stdout, r.stdout + r.stderr


def test_explainability_output_shapes_tiny_model(tmp_path):
    """Needs torch+transformers+captum (run in Colab/Docker; skipped otherwise). Builds a tiny random BERT, no downloads."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    transformers = pytest.importorskip("transformers")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
    from app.explainability.attributions import integrated_gradients
    from app.explainability.tokens import token_table
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "india", "banned", "upi", "payments", "the", "##s"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = transformers.BertTokenizerFast(str(tmp_path / "vocab.txt"))
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=32, num_labels=2)
    model = transformers.BertForSequenceClassification(cfg).eval()

    class C:
        pass
    clf = C(); clf.tok, clf.model, clf.max_len = tok, model, 32
    text = "india banned upi payments"
    toks, attrs = token_table(clf, text), integrated_gradients(clf, text, n_steps=8)
    assert toks[0].special and toks[0].token == "[CLS]" and toks[1].word_start
    assert len(attrs) == 4 and all(-1.0001 <= a.normalized <= 1.0001 for a in attrs)
