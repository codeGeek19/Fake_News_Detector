import json
import logging
import os
import re
import time
from functools import lru_cache
from typing import Optional
from ..core.config import fake_label_override, get_settings, model_path
from ..core.diagnostics import load_cached, record_error
from ..schemas.schemas import Verdict

log = logging.getLogger("fnd.classifier")


class ModelNotInstalled(RuntimeError):
    pass


def take_from_prob(p_fake: float) -> Verdict:
    """The model's take: always a decision (fake or real). How far to trust it is reported separately (see `confident`)."""
    return Verdict.likely_fake if p_fake >= 0.5 else Verdict.likely_real


def resolve_fake_index(id2label: dict, override: Optional[str] = None) -> int:
    """Which output index means FAKE. Pretrained models disagree (LABEL_0 vs LABEL_1), so this is configurable."""
    labels = {int(k): str(v) for k, v in id2label.items()}
    if override is not None:
        o = str(override).strip()
        if o.isdigit() and int(o) in labels:
            return int(o)
        for k, v in labels.items():
            if v.lower() == o.lower():
                return k
        raise ValueError(f"fake label '{override}' not among {list(labels.values())}")
    for k, v in labels.items():
        if "fake" in v.lower():
            return k
    return 1


class Classifier:
    def __init__(self, ref: str, fake_override: Optional[str] = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        token = os.getenv("HF_TOKEN") or None
        self.ref = ref
        self.tok = AutoTokenizer.from_pretrained(ref, token=token)
        try:   # eager attention lets us read attention maps for the (optional) attention column
            model = AutoModelForSequenceClassification.from_pretrained(ref, token=token, attn_implementation="eager")
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(ref, token=token)
        self.model = model.float().eval()
        meta = os.path.join(ref, "metadata.json")
        self.meta = json.load(open(meta, encoding="utf-8")) if os.path.isfile(meta) else {}
        self.max_len = int(self.meta.get("max_len", get_settings().max_tokens))
        self.fake_idx = resolve_fake_index(self.model.config.id2label, fake_override)
        self.temperature = float(self.meta.get("temperature", 1.0))   # fitted on validation data by ml/evaluation/calibrate.py
        self.version = self.meta.get("model_version", "pretrained")
        self.cutoff = self.meta.get("training_data_cutoff", "unknown (pretrained model; see its model card)")
        import torch
        self._torch = torch

    def predict(self, text: str) -> dict:
        t0 = time.time()
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_len)
        with self._torch.no_grad():
            logits = self.model(**enc).logits / self.temperature
            p_fake = float(self._torch.softmax(logits, -1)[0, self.fake_idx])
        conf = max(p_fake, 1 - p_fake)
        return {"p_fake": p_fake, "verdict": take_from_prob(p_fake), "confidence": conf,
                "confident": conf >= get_settings().classifier_threshold,
                "n_tokens": int(enc["input_ids"].shape[1]), "ms": int((time.time() - t0) * 1000)}


@lru_cache(maxsize=None)
def get_classifier(lang: str) -> Optional[Classifier]:
    """None = no classifier configured for this language (evidence-only mode).
    ModelNotInstalled = configured but cannot be loaded (wrong path, offline, bad label name)."""
    ref = model_path(lang)
    if not ref:
        return None
    try:
        return Classifier(ref, fake_label_override(lang))
    except Exception as e:
        record_error("classifier", ref, e)
        raise ModelNotInstalled(f"Could not load '{ref}' for '{lang}': {e}") from e


class TokenizerHolder:
    """Tokenizer-only stand-in so the token table still works when a language has no classifier."""
    model = None

    def __init__(self, name: str, max_len: int):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, token=os.getenv("HF_TOKEN") or None)
        self.max_len = max_len


def get_tokenizer_holder(name: Optional[str], max_len: int):
    return load_cached("tokenizer", name, lambda: TokenizerHolder(name, max_len))


def read_metadata(lang: str) -> dict:
    ref = model_path(lang)
    info = {"language": lang, "configured": bool(ref), "ref": ref}
    if not ref:
        info["status"] = "not configured (evidence-only)"
    elif os.path.exists(ref):
        ok = os.path.isfile(os.path.join(ref, "config.json"))
        info["status"] = "local folder" if ok else "local folder missing model files"
        meta = os.path.join(ref, "metadata.json")
        if ok and os.path.isfile(meta):
            info.update(json.load(open(meta, encoding="utf-8")))
    else:
        info["status"] = "Hugging Face Hub (downloaded on first use)" if re.match(r"^[\w.\-]+/[\w.\-]+$", ref) else "path not found"
    return info
