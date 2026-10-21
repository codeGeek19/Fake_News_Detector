import asyncio
import logging
import time
from datetime import datetime, timezone
import httpx
from ..core.cache import cache_get, cache_set
from ..core.config import Settings, stance_model_name, translate_candidates
from ..explainability.attributions import integrated_gradients
from ..explainability.tokens import token_table
from ..preprocessing.claims import build_queries, extract_claims
from ..preprocessing.language import detect_language
from ..preprocessing.textutils import words
from ..retrieval.embeddings import semantic_fn
from ..retrieval.factcheck import search_factchecks
from ..retrieval.gdelt import search_gdelt
from ..retrieval.newsapi import search_newsapi
from ..retrieval.nli import get_stance_model
from ..retrieval.ranking import rank
from ..retrieval.rss import search_rss
from ..schemas.schemas import (AnalyzeRequest, AnalyzeResponse, ClaimResult, Stance, TechDetails, Verdict)
from . import explanation, llm_summary, verdict
from .classifier import ModelNotInstalled, get_classifier, get_tokenizer_holder
from .translate import translate_to_english

log = logging.getLogger("fnd.pipeline")


async def _cached(key, coro):
    hit = cache_get(key)
    if hit is not None:
        return hit
    sources, status = await coro
    if status.ok:                       # never cache failures
        cache_set(key, (sources, status))
    return sources, status


async def retrieve(claim: str, lang: str, client: httpx.AsyncClient, s: Settings):
    q = build_queries(claim)
    t = s.request_timeout
    results = await asyncio.gather(
        _cached(("fc", q["phrase"], lang), search_factchecks(q["phrase"], lang, s.factcheck_key, client, t)),
        _cached(("gd", q["keywords"], lang), search_gdelt(q["keywords"], lang, s.enable_gdelt, client, t)),
        _cached(("rss", q["keywords"], lang), search_rss(q["keywords"], lang, s.enable_google_news_rss, client, t)),
        _cached(("na", q["keywords"], lang), search_newsapi(q["keywords"], lang, s.newsapi_key, client, t)),
    )
    sources = [src for res, _ in results for src in res]
    return sources, [st for _, st in results]


def _stance_for_news(claim: str, sources: list, s: Settings, lang: str = "en") -> None:
    model = get_stance_model(stance_model_name(s, lang))
    if not model:
        return
    for src in [x for x in sources if x.provenance != "factcheck"][:8]:
        try:
            src.stance, src.stance_confidence = model.stance(claim, f"{src.title}. {src.snippet or ''}")
        except Exception:
            log.exception("stance failed")


async def analyze(req: AnalyzeRequest, client: httpx.AsyncClient, s: Settings) -> AnalyzeResponse:
    t0 = time.time()
    info = detect_language(req.text, req.language)
    lang, text = info["language"], info["text"]
    extra_limits, caveats = [], []
    clf, cls_text, translated, translation_model = None, text, None, None
    try:
        clf = get_classifier(lang)        # None -> no native model configured
    except ModelNotInstalled as e:
        if lang == "en":
            raise                         # the English classifier is required: misconfiguration -> 503
        log.warning("native %s model not usable (%s); falling back", lang, e)
        extra_limits.append(f"The configured {lang} model could not be loaded, so a fallback was used. Details: {str(e)[:160]}")

    if clf:
        status = "native"
    elif lang in ("hi", "mr"):
        cands = translate_candidates(s, lang)
        if not cands:
            status = "unavailable_no_model"
        else:
            try:
                en_clf = get_classifier("en")
            except ModelNotInstalled:
                en_clf = None
            if en_clf is None:
                status = "unavailable_english_model"
            else:
                # No native classifier: translate to English and let the English classifier read the translation.
                translated, tstat, translation_model = await asyncio.to_thread(
                    translate_to_english, cands, lang, text[:s.translate_max_chars])
                if translated:
                    clf, cls_text, status = en_clf, translated, "translated"
                else:
                    status = "unavailable_translator" if tstat == "no_translator" else "unavailable_translation_failed"
    else:
        status = "unavailable_no_model"   # English with MODEL_EN_PATH unset

    if clf:
        pred = await asyncio.to_thread(clf.predict, cls_text)
        holder = clf
        # The take is always given; these caveats say how far to trust it.
        if len(words(cls_text)) < s.classifier_min_words:
            caveats.append("short_text")  # classifiers learned from full articles; headlines are outside their training data
        if not pred["confident"]:
            caveats.append("low_confidence")
        if translated:
            caveats.append("translated")
    else:
        pred = {"p_fake": None, "verdict": Verdict.uncertain, "confidence": 0.0, "n_tokens": 0, "ms": 0}
        holder = await asyncio.to_thread(get_tokenizer_holder, stance_model_name(s, lang), s.max_tokens)

    tokens, attrs = [], []
    if holder:
        try:
            tokens = await asyncio.to_thread(token_table, holder, cls_text)
        except Exception:
            log.exception("token table failed")
    if clf:
        try:
            attrs = await asyncio.to_thread(integrated_gradients, clf, cls_text)
        except Exception:
            log.exception("integrated gradients failed")
            extra_limits.append("Token attribution could not be computed for this input.")
    if not clf:
        pred["n_tokens"] = len(tokens)

    sem = await asyncio.to_thread(semantic_fn, s.embed_model)     # None -> lexical relevance only
    claims, all_statuses, flags = [], {}, set()
    for c in extract_claims(text):
        if c.claim_type == "opinion":
            flags.add("opinion")
            claims.append(ClaimResult(text=c.text, claim_type="opinion", verdict=Verdict.uncertain, reasons=["opinion"]))
            continue
        sources, statuses = await retrieve(c.text, lang, client, s)
        for st in statuses:
            all_statuses.setdefault(st.provider, st)
        ranked = await asyncio.to_thread(rank, c.text, sources, 0.25, 12, sem)
        verdict.apply_factcheck_stances(ranked)
        await asyncio.to_thread(_stance_for_news, c.text, ranked, s, lang)
        v, reasons, f = verdict.claim_verdict(ranked)
        flags.update(f)
        claims.append(ClaimResult(
            text=c.text, claim_type="factual", verdict=v, reasons=reasons,
            supporting_sources=[x for x in ranked if x.stance == Stance.supports],
            contradicting_sources=[x for x in ranked if x.stance == Stance.contradicts],
            related_articles=[x for x in ranked if x.stance == Stance.related]))

    if not translated and clf and clf.meta.get("language") and lang not in str(clf.meta["language"]).replace(",", "_").split("_"):
        flags.add("cross_lingual")
        caveats.append("cross_lingual")      # classifier was trained on another language; unvalidated here (a 'hi_mr' model is fine for hi and mr)
    final = verdict.overall(claims, pred["verdict"])
    bucket = {"supporting": [x for c in claims for x in c.supporting_sources],
              "contradicting": [x for c in claims for x in c.contradicting_sources],
              "related": [x for c in claims for x in c.related_articles]}
    checked = datetime.now(timezone.utc)
    cutoff = clf.cutoff if clf else "n/a"
    expl, limits = explanation.build(lang, final, pred["confidence"], attrs, bucket, claims,
                                     list(all_statuses.values()), cutoff, checked.strftime("%Y-%m-%d %H:%M UTC"),
                                     has_clf=bool(clf), translated=bool(translated), short_text="short_text" in caveats)
    for c in claims:
        c.reasons = [explanation.reason_text(r, lang) for r in c.reasons]
    nli_name = stance_model_name(s, lang)
    nli_loaded = bool(nli_name) and get_stance_model(nli_name) is not None
    if s.enable_llm_summary and s.anthropic_key:
        top = sorted(bucket["supporting"] + bucket["contradicting"] + bucket["related"], key=lambda x: -x.relevance)
        expl.summary = await llm_summary.summarize(client, s, lang, expl.prediction, claims, top)
    return AnalyzeResponse(
        language=lang, script=info["script"], is_hinglish=info["is_hinglish"], language_mismatch=info["mismatch"],
        prediction=final, model_prediction=pred["verdict"], model_confidence=round(pred["confidence"], 4),
        classifier_available=bool(clf), classifier_status=status, model_caveats=caveats, translated_text=translated,
        claims=claims, tokens=tokens, attributions=attrs, explanation=expl,
        supporting_sources=bucket["supporting"], contradicting_sources=bucket["contradicting"],
        related_articles=bucket["related"], evidence_status=list(all_statuses.values()),
        flags=sorted(flags), limitations=limits + extra_limits,
        tech=TechDetails(model_name=clf.ref if clf else "none (evidence-only)",
                         nli_model=nli_name if nli_loaded else "none (stance detection off)",
                         embedding_model=s.embed_model if sem else "none (lexical ranking only)",
                         llm_model=s.llm_model if expl.summary else None,
                         translation_model=translation_model,
                         model_version=clf.version if clf else "n/a", training_cutoff=cutoff, language=lang,
                         n_tokens=pred["n_tokens"], inference_ms=pred["ms"],
                         total_ms=int((time.time() - t0) * 1000), evidence_checked_at=checked))
