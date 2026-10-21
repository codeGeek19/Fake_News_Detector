"""Translate-then-classify for Hindi/Marathi: the (English) classifier reads a machine translation of the text.
Default model: facebook/nllb-200-distilled-600M (supports hin_Deva, mar_Deva -> eng_Latn).
Caveats from its model card: CC-BY-NC-4.0 (non-commercial), described as a research model, trained on inputs up to 512 tokens,
not intended for document translation. We therefore translate only the first TRANSLATE_MAX_CHARS characters, in short chunks.
Loaded with AutoModelForSeq2SeqLM directly (the 'translation' pipeline was removed in transformers v5)."""
import logging
import re
from typing import Optional

from ..core.diagnostics import load_cached, record_error

log = logging.getLogger("fnd.translate")
LANG_CODE = {"hi": "hin_Deva", "mr": "mar_Deva"}
_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def split_chunks(text: str, max_chars: int = 300) -> list:
    """Pack whole sentences into chunks of at most max_chars (a single longer sentence is cut)."""
    chunks, cur = [], ""
    for sent in _SPLIT.split(text.strip()):
        while len(sent) > max_chars:
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(sent[:max_chars]); sent = sent[max_chars:]
        if cur and len(cur) + 1 + len(sent) > max_chars:
            chunks.append(cur); cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def supports(name: str, lang: str) -> bool:
    """NLLB is multilingual; Helsinki opus-mt-XX-en models serve one source language."""
    low = name.lower()
    if "nllb" in low:
        return lang in LANG_CODE
    m = re.search(r"opus-mt-([a-z]{2,3})-en", low)
    return (m.group(1) == lang) if m else True


class Translator:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._torch = torch
        self.nllb = "nllb" in name.lower()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
        self.eng = self.tok.convert_tokens_to_ids("eng_Latn") if self.nllb else None

    def to_english(self, text: str, lang: str) -> str:
        if self.nllb:
            self.tok.src_lang = LANG_CODE[lang]
        out, chunks = [], split_chunks(text)
        for i in range(0, len(chunks), 4):
            enc = self.tok(chunks[i:i + 4], return_tensors="pt", padding=True, truncation=True, max_length=256)
            kw = {"forced_bos_token_id": self.eng} if self.nllb else {}
            with self._torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=256, num_beams=2, **kw)
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
        return " ".join(out).strip()


def get_translator(name: Optional[str]):
    return load_cached("translator", name, lambda: Translator(name))


def translate_to_english(models: list, lang: str, text: str):
    """Try each candidate in order. Returns (english_text | None, status, model_used)
    status: 'ok' | 'no_translator' (none could be loaded; see /api/health) | 'failed' (loaded but translation failed)."""
    loaded = False
    for name in models:
        if not name or not supports(name, lang):
            continue
        t = get_translator(name)
        if t is None:
            continue
        loaded = True
        try:
            out = t.to_english(text, lang)
            if out:
                return out, "ok", name
        except Exception as e:                       # noqa: BLE001
            record_error("translate", name, e)
            log.exception("translation failed with %s", name)
    return None, ("failed" if loaded else "no_translator"), None
