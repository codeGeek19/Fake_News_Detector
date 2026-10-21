"""Clean, label-normalise, de-duplicate and split a news dataset -> {prefix}_train/val/test.csv

Leakage controls:
  * exact duplicates removed (normalised text hash)
  * near-duplicates removed with MinHash-LSH BEFORE splitting, so copies can't straddle train/test
  * if a date column exists the split is TEMPORAL (oldest -> train, newest -> test)
Output columns: text,label,date   (label: 0 = REAL, 1 = FAKE)
"""
import argparse
import hashlib
import re
import unicodedata
import pandas as pd
from datasketch import MinHash, MinHashLSH

URL = re.compile(r"https?://\S+|www\.\S+", re.I)
WIRE = re.compile(r"^.{0,80}\((reuters|ap)\)\s*[-–—]\s*", re.I)   # ISOT leaks 'CITY (Reuters) -' for real news


def clean(t: str, strip_wire: bool) -> str:
    t = unicodedata.normalize("NFC", str(t))
    t = t.replace("\u200b", "").replace("\u200d", "").replace("\ufeff", "")
    t = URL.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return WIRE.sub("", t) if strip_wire else t


def minhash(text: str, num_perm=128) -> MinHash:
    w = text.lower().split()[:400]
    m = MinHash(num_perm=num_perm)
    for i in range(max(1, len(w) - 4)):
        m.update(" ".join(w[i:i + 5]).encode("utf-8"))
    return m


def near_dedup(texts, threshold):
    lsh, keep = MinHashLSH(threshold=threshold, num_perm=128), []
    for i, t in enumerate(texts):
        m = minhash(t)
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text_col", required=True)
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--fake_values", required=True, help="comma-separated raw label values meaning FAKE, e.g. fake,false,0")
    ap.add_argument("--real_values", default=None, help="optional comma-separated raw values meaning REAL; rows matching neither list are DROPPED")
    ap.add_argument("--date_col", default=None)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--min_chars", type=int, default=40)
    ap.add_argument("--near_dup_threshold", type=float, default=0.85)
    ap.add_argument("--strip_wire_tags", action="store_true", help="remove 'CITY (Reuters) -' prefixes (recommended for ISOT)")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    cols = {a.text_col: "text", a.label_col: "label"}
    if a.date_col:
        cols[a.date_col] = "date"
    df = df[list(cols)].rename(columns=cols)
    print("raw label values:", df["label"].value_counts(dropna=False).to_dict())

    fake = {v.strip().lower() for v in a.fake_values.split(",")}
    df = df.dropna(subset=["text", "label"])
    raw = df["label"].astype(str).str.strip().str.lower().str.replace(r"\.0$", "", regex=True)
    if a.real_values:
        real = {v.strip().lower() for v in a.real_values.split(",")}
        keep = raw.isin(fake | real)
        print(f"dropping {int((~keep).sum())} rows whose label is in neither list")
        df, raw = df[keep], raw[keep]
    df["label"] = raw.isin(fake).astype(int)
    df["text"] = df["text"].map(lambda t: clean(t, a.strip_wire_tags))
    df = df[df["text"].str.len() >= a.min_chars]
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")   # ISOT mixes "December 31, 2017" and "Dec 31, 2017"
    n0 = len(df)

    df["h"] = df["text"].map(lambda t: hashlib.md5(t.lower().encode("utf-8")).hexdigest())
    df = df.drop_duplicates("h").drop(columns="h").reset_index(drop=True)
    n1 = len(df)
    df = df.iloc[near_dedup(df["text"].tolist(), a.near_dup_threshold)].reset_index(drop=True)
    print(f"rows: {n0} -> exact-dedup {n1} -> near-dedup {len(df)}")

    if "date" in df and df["date"].notna().mean() > 0.9:
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        n = len(df)
        parts = {"train": df.iloc[: int(n * .70)], "val": df.iloc[int(n * .70): int(n * .85)], "test": df.iloc[int(n * .85):]}
        print("TEMPORAL split (train oldest -> test newest)")
        for k, d in parts.items():
            print(f"  {k}: {d['date'].min().date()} .. {d['date'].max().date()}")
    else:
        from sklearn.model_selection import train_test_split
        print("WARNING: no usable date column -> stratified RANDOM split. Temporal drift is NOT being tested.")
        tr, rest = train_test_split(df, test_size=.30, stratify=df["label"], random_state=42)
        va, te = train_test_split(rest, test_size=.50, stratify=rest["label"], random_state=42)
        parts = {"train": tr, "val": va, "test": te}

    for k, d in parts.items():
        d.to_csv(f"{a.out_prefix}_{k}.csv", index=False)
        print(f"{k}: {len(d)} rows, fake share {d['label'].mean():.2f}")


if __name__ == "__main__":
    main()
