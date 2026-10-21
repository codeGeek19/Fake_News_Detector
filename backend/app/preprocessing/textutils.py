import re
import unicodedata

# NOTE: do not use \w+ for Hindi. Python's \w does not match Devanagari combining vowel signs (matras),
# so it would cut Hindi words into pieces. Split on whitespace/punctuation instead.
_WORD = re.compile(r"[^\s.,;:!?\"'()\[\]{}|/\\\-–—।“”‘’]+")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\u200d", "").replace("\ufeff", "")
    text = URL.sub(" ", text)        # URLs in input are never fetched or followed
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())
