import re
from .textutils import normalize, words

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")
# Romanized-Hindi function words. A heuristic, not a trained detector: very short or
# English-heavy text can fool it. Swap for a proper classifier if Hinglish matters to you.
HINGLISH_HINTS = {"hai", "hain", "ne", "ka", "ki", "ke", "kiya", "ye", "yeh", "nahi", "kya",
                  "ko", "se", "mein", "aur", "bhi", "tha", "thi", "raha", "rahi", "ji", "hua",
                  "gaya", "kar", "karo", "liye", "wala", "wali", "sarkar"}


# Hindi and Marathi share the Devanagari script, so script alone cannot tell them apart.
# Standalone function words are compared instead (Marathi attaches many suffixes to nouns, so only
# separate words are counted). A heuristic: on ties the user's toggle decides.
HI_MARKERS = {"है", "हैं", "और", "में", "था", "थी", "थे", "नहीं", "किया", "की", "का", "के", "को", "से",
              "कि", "यह", "वह", "भी", "लिए", "हुआ", "गया", "रहा", "रही", "पर"}
MR_MARKERS = {"आहे", "आहेत", "आणि", "मध्ये", "होते", "होता", "होती", "नाही", "केले", "केली", "केला",
              "झाले", "झाली", "झाला", "त्यांनी", "म्हणाले", "सांगितले", "असे", "साठी", "कडून", "पण", "तर"}


def detect_language(text: str, user_choice: str = "auto") -> dict:
    text = normalize(text)
    dev = len(DEVANAGARI.findall(text))
    lat = len(LATIN.findall(text))
    total = (dev + lat) or 1
    if dev / total > 0.5:
        ws = words(text)
        hi_h, mr_h = sum(w in HI_MARKERS for w in ws), sum(w in MR_MARKERS for w in ws)
        lang = "mr" if mr_h > hi_h else "hi" if hi_h > mr_h else (user_choice if user_choice in ("hi", "mr") else "hi")
        script, hinglish = "devanagari", False
    else:
        ws = [w for w in words(text) if w.isascii()]
        hits = sum(w in HINGLISH_HINTS for w in ws)
        hinglish = hits >= 2 and hits / max(len(ws), 1) > 0.12
        lang, script = ("hi" if hinglish else "en"), "latin"
    return {"text": text, "language": lang, "script": script, "is_hinglish": hinglish,
            "mismatch": user_choice not in ("auto", lang)}
