"""Transparent, rule-based evidence assessment. No LLM decides truth here.
The classifier alone NEVER yields a final verdict: without evidence the result is 'uncertain'."""
from datetime import datetime, timezone
from ..retrieval.common import domain_of
from ..schemas.schemas import Source, Stance, Verdict

FALSE_WORDS = ("false", "fake", "misleading", "incorrect", "satire", "pants on fire", "fabricated",
               "फर्जी", "फ़र्ज़ी", "झूठ", "गलत", "भ्रामक", "फेक",
               "खोटे", "खोटा", "खोटी", "चुकीचे", "दिशाभूल", "अफवा", "बनावट", "फसवे")
TRUE_WORDS = ("true", "correct", "accurate", "सच", "सही", "सत्य", "खरे", "खरा", "बरोबर")
MIN_INDEPENDENT = 2        # distinct publishers needed before news coverage can move the verdict
MIN_STANCE_CONF = 0.8


def factcheck_rating(s: Source) -> int:
    """+1 supports, -1 contradicts, 0 unknown. Ratings are free text per fact-checker; unmatched -> 0."""
    r = (s.rating or "").lower()
    if any(w in r for w in FALSE_WORDS):
        return -1
    if any(w in r for w in TRUE_WORDS):
        return 1
    return 0


def apply_factcheck_stances(sources: list) -> None:
    for s in sources:
        if s.provenance == "factcheck":
            r = factcheck_rating(s)
            s.stance = Stance.supports if r == 1 else Stance.contradicts if r == -1 else Stance.related
            s.stance_confidence = 1.0 if r else 0.0


def claim_verdict(sources: list):
    reasons, flags = [], []
    fc = [s for s in sources if s.provenance == "factcheck" and factcheck_rating(s) != 0]
    pro, con = sum(factcheck_rating(s) == 1 for s in fc), sum(factcheck_rating(s) == -1 for s in fc)
    if any("satire" in (s.rating or "").lower() for s in sources if s.provenance == "factcheck"):
        flags.append("satire_flagged")
    if pro and con:
        return Verdict.uncertain, ["factcheck_conflict"], flags
    if con:
        return Verdict.likely_fake, ["factcheck_false"], flags
    if pro:
        return Verdict.likely_real, ["factcheck_true"], flags

    news = [s for s in sources if s.provenance != "factcheck" and s.stance_confidence >= MIN_STANCE_CONF]
    sup = {domain_of(str(s.url)) for s in news if s.stance == Stance.supports}
    con_ = {domain_of(str(s.url)) for s in news if s.stance == Stance.contradicts}
    if len(con_) >= MIN_INDEPENDENT and not sup:
        return Verdict.likely_fake, ["news_contradict"], flags
    if len(sup) >= MIN_INDEPENDENT and not con_:
        return Verdict.likely_real, ["news_support"], flags

    if sources:
        reasons.append("insufficient_independent")
        dates = [s.published_at for s in sources if s.published_at]
        if dates and all((datetime.now(timezone.utc) - d).total_seconds() < 48 * 3600 for d in dates):
            flags.append("developing_story")
    else:
        reasons.append("no_evidence")
    flags.append("unverified")
    return Verdict.uncertain, reasons, flags


def overall(claims: list, model_verdict: Verdict) -> Verdict:
    factual = [c for c in claims if c.claim_type == "factual"]
    if not factual:
        return Verdict.uncertain
    if any(c.verdict == Verdict.likely_fake for c in factual):
        return Verdict.likely_fake
    if all(c.verdict == Verdict.likely_real for c in factual):
        return Verdict.likely_real
    return Verdict.uncertain
