"""SQLite log of analyses: timestamp, model versions, verdicts and the retrieved sources (with their publication dates).
The user's text is NOT stored unless STORE_TEXT=true; by default only its SHA-256 and length are kept."""
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

SCHEMA = """CREATE TABLE IF NOT EXISTS analyses (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, language TEXT, text_sha256 TEXT, text_chars INTEGER,
  text TEXT, prediction TEXT, model_prediction TEXT, classifier_ref TEXT, model_version TEXT, training_cutoff TEXT,
  nli_model TEXT, embedding_model TEXT, evidence_checked_at TEXT, flags TEXT, claims TEXT, sources TEXT)"""


def _connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    c = sqlite3.connect(path, timeout=5)
    c.execute(SCHEMA)
    return c


def save_analysis(path: str, resp, text: str, store_text: bool = False) -> int:
    d = resp.model_dump(mode="json")
    claims = [{"claim_type": c["claim_type"], "verdict": c["verdict"], **({"text": c["text"]} if store_text else {})}
              for c in d["claims"]]
    sources = [{k: s.get(k) for k in ("publisher", "title", "url", "published_at", "stance", "provenance", "rating")}
               for s in d["supporting_sources"] + d["contradicting_sources"] + d["related_articles"]]
    t = d["tech"]
    with closing(_connect(path)) as c, c:
        cur = c.execute(
            "INSERT INTO analyses (created_at, language, text_sha256, text_chars, text, prediction, model_prediction, classifier_ref,"
            " model_version, training_cutoff, nli_model, embedding_model, evidence_checked_at, flags, claims, sources)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), d["language"], hashlib.sha256(text.encode("utf-8")).hexdigest(), len(text),
             text if store_text else None, d["prediction"], d["model_prediction"], t["model_name"], t["model_version"],
             t["training_cutoff"], t["nli_model"], t["embedding_model"], t["evidence_checked_at"],
             json.dumps(d["flags"]), json.dumps(claims, ensure_ascii=False), json.dumps(sources, ensure_ascii=False)))
        return int(cur.lastrowid)


def recent(path: str, limit: int = 20) -> list:
    """Metadata only (never the user's text)."""
    with closing(_connect(path)) as c:
        rows = c.execute("SELECT id, created_at, language, prediction, model_prediction, model_version, evidence_checked_at,"
                         " flags, sources FROM analyses ORDER BY id DESC LIMIT ?", (max(1, min(limit, 100)),)).fetchall()
    return [{"id": r[0], "created_at": r[1], "language": r[2], "prediction": r[3], "model_prediction": r[4],
             "model_version": r[5], "evidence_checked_at": r[6], "flags": json.loads(r[7] or "[]"),
             "n_sources": len(json.loads(r[8] or "[]"))} for r in rows]
