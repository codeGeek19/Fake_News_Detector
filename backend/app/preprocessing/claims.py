import re
from dataclasses import dataclass
from .textutils import words

SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")
ATTRIB = [
    re.compile(r"^according to [^,]{1,80},\s*", re.I),
    re.compile(r"^as per [^,]{1,80},\s*", re.I),
    re.compile(r"^(social media )?users? (say|said|claim|claimed|are saying) (that )?", re.I),
    re.compile(r"^(it is|it's) being (claimed|said|reported) that\s*", re.I),
    re.compile(r"^(reports?|rumou?rs?) (say|claim|suggest)s? that\s*", re.I),
    re.compile(r"^[^,।]{1,60}\s+के अनुसार,?\s*"),
    re.compile(r"^(दावा किया जा रहा है|कहा जा रहा है|दावा है|खबर है) कि\s*"),
    re.compile(r"^[^,।]{1,60}च्या\s+मते,?\s*"),
    re.compile(r"^(असा दावा केला जात आहे|असे सांगितले जात आहे|दावा आहे) की\s*"),
]
OPINION = re.compile(
    r"\b(i think|i believe|in my opinion|should|must|best|worst|shameful|disgraceful|terrible|amazing)\b"
    r"|मेरे (विचार|हिसाब)|मुझे लगता|शर्मनाक|सबसे (अच्छा|बुरा)|माझ्या मते|मला वाटते|लाजिरवाणे|सर्वात (चांगले|वाईट)", re.I)
STRONG = re.compile(
    r"\b(ban|banned|announced|arrested|killed|dead|died|resign\w*|launched|approved|cancel\w*|shut\w*)\b"
    r"|प्रतिबंध|घोषणा|गिरफ्तार|मौत|इस्तीफ़ा|इस्तीफा|बंद", re.I)
STOP = {"the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are", "was", "were", "has", "have",
        "had", "been", "that", "this", "with", "for", "by", "from", "at", "as", "it", "its", "will",
        "का", "की", "के", "में", "से", "है", "हैं", "था", "थे", "और", "को", "ने", "पर", "कि", "यह", "वह", "भी",
        "आहे", "आहेत", "आणि", "मध्ये", "होते", "होता", "होती", "च्या", "ची", "चा", "चे", "ला", "हे", "हा", "ही", "या", "तो", "त्या", "पण", "तर"}


@dataclass
class Claim:
    text: str
    claim_type: str  # factual | opinion


def _strip_attribution(s: str) -> str:
    for p in ATTRIB:
        s2 = p.sub("", s, count=1)
        if s2 != s:
            return s2.strip()
    return s


def _score(s: str) -> int:
    ws = words(s)
    caps = sum(1 for w in re.findall(r"\S+", s)[1:] if w[:1].isupper())
    return (2 if re.search(r"\d", s) else 0) + min(3, caps) + (1 if STRONG.search(s) else 0) + (1 if len(ws) >= 6 else 0)


def extract_claims(text: str, max_claims: int = 3) -> list[Claim]:
    """Rule-based claim extraction: strips attributions, drops questions, flags opinions.
    It is a heuristic baseline; a trained claim-detection model would do better."""
    sents = [s.strip() for s in SPLIT.split(text) if s and s.strip()]
    factual, opinions = [], []
    for s in sents:
        s = _strip_attribution(s)
        if len(words(s)) < 3 or s.endswith("?"):
            continue
        (opinions if OPINION.search(s) else factual).append(s)
    if not factual and not opinions:           # very short headline: use it whole
        whole = _strip_attribution(text.strip())
        return [Claim(whole, "opinion" if OPINION.search(whole) else "factual")]
    ranked = sorted(range(len(factual)), key=lambda i: -_score(factual[i]))[:max_claims]
    out = [Claim(factual[i], "factual") for i in sorted(ranked)]
    out += [Claim(o, "opinion") for o in opinions[:1]]
    return out


def build_queries(claim: str) -> dict:
    ws = [w for w in words(claim) if w not in STOP and len(w) > 2]
    seen, kw = set(), []
    for w in ws:
        if w not in seen:
            seen.add(w)
            kw.append(w)
    return {"phrase": " ".join(claim.split()[:12]), "keywords": " ".join(kw[:5])}
