"""Build English benchmark data: ISOT (long articles, Kaggle) and LIAR (short PolitiFact claims, UCSB).

  python ml/training/build_english_eval.py            ->  data/isot_{train,val,test}.csv and data/liar_{train,val,test}.csv

LIAR labels: pants-fire / false / barely-true = FAKE (1);  mostly-true / true = REAL (0);  half-true is dropped (ambiguous).
LIAR is research data (PolitiFact statements 2007-2016); check its terms. Both sets are old and mostly US politics.
"""
import argparse
import glob
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
ISOT_SLUG = "clmentbisaillon/fake-and-real-news-dataset"


def liar_frame(zip_path) -> pd.DataFrame:
    """train/valid/test .tsv (no header): column 1 = label, column 2 = statement."""
    tmp = Path(str(zip_path) + "_x")
    zipfile.ZipFile(zip_path).extractall(tmp)
    parts = []
    for split in ("train", "valid", "test"):
        f = glob.glob(str(tmp / "**" / f"{split}.tsv"), recursive=True)
        if not f:
            raise SystemExit(f"{split}.tsv not found inside {zip_path}")
        parts.append(pd.read_csv(f[0], sep="\t", header=None, usecols=[1, 2], names=["label", "statement"], quoting=3))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out); (out / "raw").mkdir(parents=True, exist_ok=True)
    prep = str(Path(__file__).with_name("prep_data.py"))

    # ---- LIAR ----
    zp = out / "raw" / "liar_dataset.zip"
    if not zp.exists():
        print("downloading", LIAR_URL); urllib.request.urlretrieve(LIAR_URL, zp)
    liar = liar_frame(zp); liar.to_csv(out / "raw" / "liar_all.csv", index=False)
    print("LIAR rows:", len(liar), liar["label"].value_counts().to_dict())
    subprocess.run([sys.executable, prep, "--input", str(out / "raw" / "liar_all.csv"), "--text_col", "statement", "--label_col", "label",
                    "--fake_values", "pants-fire,false,barely-true", "--real_values", "mostly-true,true", "--min_chars", "20",
                    "--out_prefix", str(out / "liar")], check=True)

    # ---- ISOT ----
    try:
        import kagglehub
        path = kagglehub.dataset_download(ISOT_SLUG)
    except Exception as e:
        raise SystemExit(f"ISOT download failed ({e!r}).\nDownload Fake.csv and True.csv from Kaggle ('Fake and Real News Dataset', "
                         f"{ISOT_SLUG}) into {out / 'raw' / 'isot'} and run again.") from e
    fake = pd.read_csv(glob.glob(f"{path}/**/Fake.csv", recursive=True)[0]); fake["label"] = "fake"
    true = pd.read_csv(glob.glob(f"{path}/**/True.csv", recursive=True)[0]); true["label"] = "real"
    isot = pd.concat([fake, true], ignore_index=True)
    isot["text_full"] = isot["title"].fillna("") + ". " + isot["text"].fillna("")    # 'subject' leaks the label, so it is not used
    isot[["text_full", "label", "date"]].to_csv(out / "raw" / "isot_all.csv", index=False)
    subprocess.run([sys.executable, prep, "--input", str(out / "raw" / "isot_all.csv"), "--text_col", "text_full", "--label_col", "label",
                    "--fake_values", "fake", "--date_col", "date", "--strip_wire_tags", "--out_prefix", str(out / "isot")], check=True)
    print("\nDone. Benchmark with:\n  python ml/evaluation/compare_pretrained.py --model jy46604790/Fake-News-Bert-Detect:LABEL_0 "
          f"--csv {out}/liar_test.csv --csv {out}/isot_test.csv")


if __name__ == "__main__":
    main()
