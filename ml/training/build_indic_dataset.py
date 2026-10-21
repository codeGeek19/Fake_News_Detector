"""Build Hindi + Marathi training data for a DevBERT / IndicBERT fake-news classifier.

Source: "Multilingual Fake News Detection Dataset: Gujarati, Hindi, Marathi, and Telugu"
        Zenodo record 11408513 (CC-BY-4.0), news articles labelled 'fake' or 'real'.
        Files used: Hindi_F&R_News.zip and Marathi_F&R_News.zip (md5 checked after download).

I could not open these files while writing this script, so their internal layout is detected automatically
(columns, label values, or fake/real folders). Run with --inspect first to see what was found. If labels are
numeric (0/1) the script REFUSES to guess which one means fake; pass --fake_values.

Output (default folder ./data):  hi_{train,val,test}.csv  mr_{train,val,test}.csv  dev_{train,val,test}.csv (Hindi+Marathi joint)
Columns: text,label,date   (label: 1 = fake, 0 = real).  The split is RANDOM (no dates in the data), so temporal drift is NOT tested,
and in-domain test scores will look better than real-world accuracy.

Usage:
  python ml/training/build_indic_dataset.py --inspect
  python ml/training/build_indic_dataset.py                       # download + build
  python ml/training/build_indic_dataset.py --zip_dir ./zips      # use zips you downloaded yourself
  python ml/training/build_indic_dataset.py --fake_values 1       # if labels are numeric and 1 = fake
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd

RECORD = "https://zenodo.org/records/11408513/files/{name}?download=1"
ZIPS = {  # md5 values are from the Zenodo record page
    "hi": ("Hindi_F&R_News.zip", "3be2e9e1f636472c99308a5eac1bb7e1"),
    "mr": ("Marathi_F&R_News.zip", "5aa6a5904c6812ead2a09448ffa51b38"),
}
FAKE_WORDS = {"fake", "false", "fake news", "fakenews", "hoax", "misinformation", "rumor", "rumour",
              "फेक", "फर्जी", "झूठ", "झूठा", "असत्य", "खोटे", "खोटी", "खोटा", "अफवा"}
REAL_WORDS = {"real", "true", "real news", "realnews", "genuine", "authentic", "credible",
              "असली", "सच", "सही", "सत्य", "खरे", "खरी", "खरा"}
TABULAR = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}


class DatasetFormatError(RuntimeError):
    pass


def label_of(value) -> str | None:
    """'fake' / 'real' / None for one raw label value (strings only; numbers are never guessed)."""
    v = str(value).strip().lower()
    if v in ("", "nan", "none"):
        return None
    if v in FAKE_WORDS:
        return "fake"
    if v in REAL_WORDS:
        return "real"
    if "fake" in v or "false" in v:
        return "fake"
    if "real" in v or "true" in v:
        return "real"
    return None


def path_label(path: Path) -> str | None:
    """Label from the file name or its parent folder (nearest wins). 'F&R' style names are ambiguous and ignored."""
    for part in (path.stem, path.parent.name):
        low = part.lower()
        f, r = ("fake" in low or "false" in low), any(k in low for k in ("real", "true", "genuine"))
        if f != r:
            return "fake" if f else "real"
    return None


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(lang: str, dest: Path) -> Path:
    name, expected = ZIPS[lang]
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / name
    if not out.exists():
        url = RECORD.format(name=quote(name))
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, out)
    got = md5(out)
    if got != expected:
        raise DatasetFormatError(f"{out} md5 {got} != expected {expected} (corrupt or changed download)")
    return out


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext in (".json", ".jsonl"):
        try:
            return pd.read_json(path, lines=(ext == ".jsonl"))
        except ValueError:
            return pd.read_json(path, lines=True)
    return pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", encoding="utf-8-sig", on_bad_lines="skip")


def infer_columns(df: pd.DataFrame, text_col=None, label_col=None):
    """Return (text_cols, label_col or None). Text = headline-like column (by name) + the longest text column."""
    label = label_col
    if label is None:
        best = None
        for c in df.columns:
            vals = df[c].dropna()
            if vals.empty or vals.nunique() > 8:
                continue
            mapped = vals.map(label_of)
            ok = mapped.notna().mean()
            named = bool(re.search(r"label|class|target|verdict|category|fake|real", str(c), re.I))
            if ok >= 0.95 and (best is None or named):
                best = c
            elif named and best is None and pd.api.types.is_numeric_dtype(vals):
                best = c                      # numeric label column: found, but values must come from --fake_values
        label = best
    if text_col:
        return [text_col], label
    strs = [c for c in df.columns if c != label and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))]   # pandas 2 (object) and 3 (str)
    if not strs:
        raise DatasetFormatError(f"no text column found; columns are {list(df.columns)}. Pass --text_col.")
    body = max(strs, key=lambda c: df[c].astype(str).str.len().mean())
    title = next((c for c in strs if c != body and re.search(r"title|headline|heading|शीर्षक|मथळा", str(c), re.I)), None)
    return ([title, body] if title else [body]), label


def load_records(root: Path, text_col=None, label_col=None, fake_values=None) -> pd.DataFrame:
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and "__MACOSX" not in p.parts and not p.name.startswith(".")]
    tables = [p for p in files if p.suffix.lower() in TABULAR]
    frames = []
    if tables:
        for p in tables:
            df = read_table(p)
            if df.empty:
                continue
            tcols, lcol = infer_columns(df, text_col, label_col)
            text = df[tcols].fillna("").astype(str).agg(". ".join, axis=1) if len(tcols) > 1 else df[tcols[0]].fillna("").astype(str)
            if lcol is not None:
                raw = df[lcol]
                if fake_values:
                    fv = {v.strip().lower() for v in fake_values.split(",")}
                    norm = raw.astype(str).str.strip().str.lower().str.replace(r"\.0$", "", regex=True)
                    lab = norm.map(lambda v: "fake" if v in fv else ("real" if v not in ("", "nan") else None))
                else:
                    lab = raw.map(label_of)
                    if lab.isna().mean() > 0.05:
                        sample = df.loc[lab.isna(), [lcol] + tcols[-1:]].head(3).to_dict("records")
                        raise DatasetFormatError(
                            f"{p.name}: label column '{lcol}' has values I will not guess: {raw.value_counts().head(6).to_dict()}.\n"
                            f"Look at a few rows {sample} and re-run with --fake_values (comma-separated raw values that mean FAKE).")
            else:
                pl = path_label(p)
                if pl is None:
                    raise DatasetFormatError(f"{p.name}: no label column ({list(df.columns)}) and the file name does not say fake/real. "
                                             "Pass --label_col.")
                lab = pd.Series(pl, index=df.index)
            frames.append(pd.DataFrame({"text": text.values, "label": lab.values, "src": p.name}))
    else:
        for p in files:
            if p.suffix.lower() != ".txt":
                continue
            pl = path_label(p)
            if pl:
                frames.append(pd.DataFrame({"text": [p.read_text(encoding="utf-8", errors="ignore")], "label": [pl], "src": [p.name]}))
    if not frames:
        raise DatasetFormatError(f"nothing usable found under {root}: {[p.name for p in files][:10]}")
    out = pd.concat(frames, ignore_index=True).dropna(subset=["label"])
    out["text"] = out["text"].astype(str).str.strip()
    return out[out["text"].str.len() > 0]


def describe(lang: str, df: pd.DataFrame) -> None:
    print(f"\n[{lang}] {len(df)} rows from {df['src'].nunique()} file(s); label counts: {df['label'].value_counts().to_dict()}")
    print("  mean text length:", int(df["text"].str.len().mean()), "chars")
    for lab in ("fake", "real"):
        s = df.loc[df["label"] == lab, "text"].head(2).tolist()
        for t in s:
            print(f"  e.g. {lab}: {t[:110]!r}")
    dev = sum("\u0900" <= ch <= "\u097F" for ch in "".join(df["text"].head(200))) / max(1, len("".join(df["text"].head(200))))
    print(f"  Devanagari share of a sample: {dev:.0%} (should be well above 50%)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hi,mr")
    ap.add_argument("--out", default="data")
    ap.add_argument("--zip_dir", default=None, help="folder that already contains the zips (skips the download)")
    ap.add_argument("--inspect", action="store_true", help="only show what the data looks like")
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--fake_values", default=None, help="raw label values meaning FAKE, e.g. '1' or 'fake,false'")
    a = ap.parse_args(argv)
    out = Path(a.out); (out / "raw").mkdir(parents=True, exist_ok=True)
    prep = Path(__file__).with_name("prep_data.py")

    for lang in [x.strip() for x in a.langs.split(",") if x.strip()]:
        zp = (Path(a.zip_dir) / ZIPS[lang][0]) if a.zip_dir else download(lang, out / "zips")
        if not zp.exists():
            raise SystemExit(f"{zp} not found")
        root = out / "raw" / f"{lang}_unzipped"
        zipfile.ZipFile(zp).extractall(root)
        try:
            df = load_records(root, a.text_col, a.label_col, a.fake_values)
        except DatasetFormatError as e:
            raise SystemExit(f"\n[{lang}] {e}")
        describe(lang, df)
        if a.inspect:
            continue
        raw_csv = out / "raw" / f"{lang}_all.csv"
        df[["text", "label"]].to_csv(raw_csv, index=False)
        subprocess.run([sys.executable, str(prep), "--input", str(raw_csv), "--text_col", "text", "--label_col", "label",
                        "--fake_values", "fake", "--real_values", "real", "--min_chars", "30",
                        "--out_prefix", str(out / lang)], check=True)

    if not a.inspect:
        langs = [x.strip() for x in a.langs.split(",") if x.strip()]
        if len(langs) > 1:
            for split in ("train", "val", "test"):
                parts = [pd.read_csv(out / f"{l}_{split}.csv") for l in langs]
                pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).to_csv(out / f"dev_{split}.csv", index=False)
            print("\nJoint Hindi+Marathi files written: dev_train.csv, dev_val.csv, dev_test.csv")
        print("\nNext:  python ml/training/train.py --model l3cube-pune/hindi-marathi-dev-bert --lang hi_mr "
              f"--train_csv {out}/dev_train.csv --val_csv {out}/dev_val.csv --test_csv {out}/dev_test.csv --out_dir out/dev --max_len 256")


if __name__ == "__main__":
    main()
