import io
import os
import sys
import zipfile
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
for sub in ("training", "evaluation"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
import build_indic_dataset as bid          # noqa: E402
import build_english_eval as bee           # noqa: E402
import compare_pretrained as cp            # noqa: E402
from metrics import fake_index             # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', 'evaluation'))


def make_zip(path, files: dict):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data if isinstance(data, bytes) else data.encode("utf-8"))


def article(i, lab):                      # distinct, Devanagari, long enough to survive cleaning and near-duplicate removal
    return " ".join(f"शब्द{lab}{i}_{j}" for j in range(30))


def csv_bytes(n=40, labels=("Fake", "Real"), extra=True):
    rows = [{"id": i, "headline": f"शीर्षक {i}", "content": article(i, "x"), "label": labels[i % 2]} for i in range(n)]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def test_detects_label_column_and_joins_headline_with_body(tmp_path):
    make_zip(tmp_path / "a.zip", {"data/hindi.csv": csv_bytes()})
    root = tmp_path / "r"; zipfile.ZipFile(tmp_path / "a.zip").extractall(root)
    df = bid.load_records(root)
    assert df["label"].value_counts().to_dict() == {"fake": 20, "real": 20}
    assert df["text"].iloc[0].startswith("शीर्षक 0. ")             # headline + body


def test_folder_layout_fake_real(tmp_path):
    files = {f"{d}/{i}.txt": article(i, d) for d in ("Fake", "Real") for i in range(5)}
    make_zip(tmp_path / "b.zip", files)
    root = tmp_path / "r"; zipfile.ZipFile(tmp_path / "b.zip").extractall(root)
    assert bid.load_records(root)["label"].value_counts().to_dict() == {"fake": 5, "real": 5}


def test_label_from_file_name_when_no_label_column(tmp_path):
    one = lambda lab: pd.DataFrame({"text": [article(i, lab) for i in range(6)]}).to_csv(index=False)
    make_zip(tmp_path / "c.zip", {"Fake.csv": one("f"), "Real.csv": one("r")})
    root = tmp_path / "r"; zipfile.ZipFile(tmp_path / "c.zip").extractall(root)
    assert bid.load_records(root)["label"].value_counts().to_dict() == {"fake": 6, "real": 6}


def test_numeric_labels_are_never_guessed(tmp_path):
    make_zip(tmp_path / "d.zip", {"x.csv": csv_bytes(labels=(0, 1))})
    root = tmp_path / "r"; zipfile.ZipFile(tmp_path / "d.zip").extractall(root)
    # numeric 0/1 labels: the script must not decide which means fake
    with pytest.raises(bid.DatasetFormatError, match="--fake_values"):
        bid.load_records(root)
    df = bid.load_records(root, fake_values="1")
    assert df["label"].value_counts().to_dict() == {"fake": 20, "real": 20}


def test_ambiguous_folder_name_is_not_used_as_label(tmp_path):
    from pathlib import Path
    assert bid.path_label(Path("Hindi_F&R_News/articles.csv")) is None
    assert bid.path_label(Path("x/Fake_News/1.txt")) == "fake" and bid.path_label(Path("x/true.csv")) == "real"


def test_marathi_label_words():
    assert bid.label_of("खोटे") == "fake" and bid.label_of("खरे") == "real" and bid.label_of("Fake News") == "fake"
    assert bid.label_of("0") is None and bid.label_of("maybe") is None


def test_end_to_end_builds_hi_mr_and_joint_files(tmp_path):
    zdir = tmp_path / "zips"; zdir.mkdir()
    for lang in ("hi", "mr"):
        make_zip(zdir / bid.ZIPS[lang][0], {"d.csv": csv_bytes(n=80)})
    out = tmp_path / "data"
    bid.main(["--zip_dir", str(zdir), "--out", str(out)])
    for name in ("hi_train", "hi_val", "hi_test", "mr_train", "dev_train", "dev_val", "dev_test"):
        assert (out / f"{name}.csv").exists(), name
    dev = pd.read_csv(out / "dev_train.csv")
    assert set(dev["label"]) == {0, 1} and len(dev) == len(pd.read_csv(out / "hi_train.csv")) + len(pd.read_csv(out / "mr_train.csv"))


def test_inspect_mode_writes_nothing(tmp_path, capsys):
    zdir = tmp_path / "zips"; zdir.mkdir()
    make_zip(zdir / bid.ZIPS["hi"][0], {"d.csv": csv_bytes()})
    bid.main(["--zip_dir", str(zdir), "--out", str(tmp_path / "o"), "--langs", "hi", "--inspect"])
    assert not (tmp_path / "o" / "hi_train.csv").exists() and "label counts" in capsys.readouterr().out


def test_md5_check_rejects_corrupt_download(tmp_path, monkeypatch):
    (tmp_path / bid.ZIPS["hi"][0]).write_bytes(b"not the real file")
    with pytest.raises(bid.DatasetFormatError, match="md5"):
        bid.download("hi", tmp_path)


# ---------------- English benchmark helpers ----------------
def test_liar_frame_reads_label_and_statement_columns(tmp_path):
    def tsv(n, off):
        return "\n".join("\t".join([f"{i}.json", lab, f"Statement number {i + off} about taxes", "subj", "spk", "job", "st", "party", "0", "0", "0", "0", "0", "ctx"])
                         for i, lab in enumerate(["false", "true", "half-true", "pants-fire"][:n]))
    make_zip(tmp_path / "liar.zip", {"train.tsv": tsv(4, 0), "valid.tsv": tsv(2, 10), "test.tsv": tsv(2, 20)})
    df = bee.liar_frame(tmp_path / "liar.zip")
    assert list(df.columns) == ["label", "statement"] and len(df) == 8 and df["statement"].iloc[0].startswith("Statement number 0")


def test_compare_spec_parsing_and_table():
    assert cp.parse_spec("jy46604790/Fake-News-Bert-Detect:LABEL_0") == ("jy46604790/Fake-News-Bert-Detect", "LABEL_0")
    assert cp.parse_spec("models/en") == ("models/en", None)
    assert cp.parse_spec("C:\\models\\en") == ("C:\\models\\en", None)
    assert cp.parse_spec("C:\\models\\en:FAKE") == ("C:\\models\\en", "FAKE")
    md = cp.to_markdown([{"model": "m", "data": "liar_test.csv", "n": 10, "macro_f1": 0.6, "accuracy": 0.62, "roc_auc": 0.7}])
    assert md.splitlines()[0].startswith("| model") and "liar_test.csv" in md


def test_fake_index_shared_helper():
    assert fake_index({0: "LABEL_0", 1: "LABEL_1"}, "LABEL_0") == 0 and fake_index({0: "REAL", 1: "FAKE"}) == 1


def test_temperature_scaling_recovers_overconfidence():
    import numpy as np
    from calibrate import fit_temperature, _nll
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    true_logit = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1.5, 4000)         # genuinely noisy evidence
    logits = np.stack([-true_logit, true_logit], 1) * 4.0                         # logit gap = 8x the data's; analytic optimum T = 8 / (2*1/1.5**2) = 9
    T = fit_temperature(logits, y)
    assert 7.5 < T < 10.5 and _nll(logits, y, T) < 0.5 * _nll(logits, y, 1.0)
