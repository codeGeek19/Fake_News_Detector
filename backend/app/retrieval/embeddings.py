"""Multilingual sentence embeddings for semantic relevance between a claim and retrieved articles.
Default: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 50+ languages, mean pooling).
Used through plain transformers (no extra dependency). If it cannot load, ranking falls back to word overlap."""
import logging
from typing import Callable, Optional

from ..core.diagnostics import load_cached

log = logging.getLogger("fnd.embed")


class Embedder:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    def encode(self, texts: list):
        torch = self._torch
        enc = self.tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)          # mean pooling
        return torch.nn.functional.normalize(vec, dim=1)

    def similarity(self, claim: str, texts: list) -> list:
        if not texts:
            return []
        c, t = self.encode([claim]), self.encode(texts)
        return (t @ c.T).squeeze(1).tolist()                             # cosine (vectors are normalised)


def get_embedder(name: Optional[str]):
    return load_cached("embedding", name, lambda: Embedder(name))


def semantic_fn(name: Optional[str]) -> Optional[Callable]:
    e = get_embedder(name)
    return e.similarity if e else None
