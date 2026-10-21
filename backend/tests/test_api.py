import pytest
from fastapi.testclient import TestClient
from app.core import ratelimit
from app.core.config import get_settings
from app.main import app
from app.schemas.schemas import (Attribution, EvidenceStatus, TokenInfo, Verdict)
from app.services import pipeline


@pytest.fixture(autouse=True)
def reset(monkeypatch, tmp_path):
    from app.services import classifier
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.sqlite3"))
    monkeypatch.setenv("MIN_CLASSIFIER_WORDS", "1")      # most tests use short texts; abstention has its own tests
    monkeypatch.setenv("NLI_MODEL_EN", "")
    monkeypatch.setenv("TRANSLATE_MODEL", "")      # never download a translation model inside tests
    monkeypatch.setenv("EMBED_MODEL", "")          # never download an embedding model inside tests
    monkeypatch.delenv("ENABLE_LLM_SUMMARY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from app.core import diagnostics
    diagnostics.reset()
    monkeypatch.setenv("TRANSLATE_FALLBACK_HI", "")
    monkeypatch.setenv("TRANSLATE_FALLBACK_MR", "")
    ratelimit._hits.clear()
    classifier.get_classifier.cache_clear()
    for k in ("MODEL_EN_PATH", "MODEL_HI_PATH", "GOOGLE_FACTCHECK_API_KEY", "MODEL_EN_FAKE_LABEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NLI_MODEL", "")          # never download a model inside tests
    yield
    classifier.get_classifier.cache_clear()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and set(r["models"]) == {"en", "hi", "mr"}
    assert "evidence-only" in r["models"]["hi"]


def test_models_and_sources(client):
    assert len(client.get("/api/models").json()["models"]) == 3
    assert "publisher_types" in client.get("/api/sources").json()


def test_validation_too_short(client):
    assert client.post("/api/analyze", json={"text": "hi"}).status_code == 422


def test_validation_bad_language(client):
    assert client.post("/api/analyze", json={"text": "India launched a satellite.", "language": "fr"}).status_code == 422


def test_too_long(client, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_CHARS", "50")
    assert client.post("/api/analyze", json={"text": "x" * 60}).status_code == 413


def test_misconfigured_model_path_returns_503(client, monkeypatch):
    monkeypatch.setenv("MODEL_EN_PATH", "definitely/not-a-real-model-xyz")   # unreachable hub id / offline
    r = client.post("/api/analyze", json={"text": "India launched a new satellite today."})
    assert r.status_code == 503 and "could not load" in r.json()["detail"].lower()


def test_rate_limit(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    codes = [client.post("/api/analyze", json={"text": "India launched a new satellite."}).status_code for _ in range(3)]
    assert codes[-1] == 429


class FakeClf:
    ref, max_len, fake_idx = "fake-bert", 256, 0
    version, cutoff, meta = "v0", "2017-12-31", {}
    def predict(self, text):
        return {"p_fake": 0.9, "verdict": Verdict.likely_fake, "confidence": 0.9, "confident": True, "n_tokens": 5, "ms": 1}


def test_full_pipeline_with_fakes(client, monkeypatch):
    monkeypatch.setenv("MODEL_EN_PATH", "x")
    monkeypatch.setattr(pipeline, "get_classifier", lambda lang: FakeClf())
    monkeypatch.setattr(pipeline, "token_table", lambda c, t: [TokenInfo(token="UPI", raw="UPI", id=1, position=0, special=False, word_start=True)])
    monkeypatch.setattr(pipeline, "integrated_gradients", lambda c, t: [Attribution(token="banned", position=1, word_start=True, score=0.2, normalized=1.0)])

    async def no_evidence(claim, lang, http, s):
        return [], [EvidenceStatus(provider="google_factcheck", ok=False, error="API key not configured")]
    monkeypatch.setattr(pipeline, "retrieve", no_evidence)
    r = client.post("/api/analyze", json={"text": "The government has banned UPI payments from October.", "language": "en"})
    assert r.status_code == 200, r.text
    j = r.json()
    # classifier says fake at 90%, but with no evidence the FINAL verdict must stay uncertain
    assert j["classifier_available"] is True
    assert j["model_prediction"] == "likely_fake" and j["prediction"] == "uncertain"
    assert "No sufficiently reliable" in j["explanation"]["evidence"]
    assert j["tech"]["model_version"] == "v0" and j["tech"]["model_name"] == "fake-bert" and j["supporting_sources"] == []


def _no_evidence(monkeypatch):
    async def no_evidence(claim, lang, http, s):
        return [], [EvidenceStatus(provider="google_factcheck", ok=False, error="API key not configured")]
    monkeypatch.setattr(pipeline, "retrieve", no_evidence)


def test_hindi_without_classifier_is_evidence_only(client, monkeypatch):
    _no_evidence(monkeypatch)                      # no MODEL_HI_PATH configured
    r = client.post("/api/analyze", json={"text": "सरकार ने सभी UPI भुगतानों पर प्रतिबंध लगा दिया है।", "language": "hi"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["language"] == "hi" and j["classifier_available"] is False and j["classifier_status"] == "unavailable_no_model"
    assert j["prediction"] == "uncertain" and j["model_confidence"] == 0 and j["attributions"] == []
    assert "classifier" in j["explanation"]["why_model"]
    assert j["tech"]["model_name"].startswith("none")


def test_english_without_classifier_still_works(client, monkeypatch):
    _no_evidence(monkeypatch)
    r = client.post("/api/analyze", json={"text": "The government has banned UPI payments from October.", "language": "en"})
    assert r.status_code == 200 and r.json()["classifier_available"] is False


def test_marathi_without_classifier(client, monkeypatch):
    _no_evidence(monkeypatch)
    r = client.post("/api/analyze", json={"text": "सरकारने सर्व UPI पेमेंटवर बंदी घातली आहे आणि लोक नाराज आहेत.", "language": "mr"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["language"] == "mr" and j["classifier_available"] is False and j["language_mismatch"] is False
    assert "classifier" in j["explanation"]["why_model"] and "पुरावा" in j["explanation"]["evidence"]


def test_cross_lingual_flag_when_classifier_trained_on_english(client, monkeypatch):
    class EnglishTrained(FakeClf):
        meta = {"language": "en"}
    monkeypatch.setattr(pipeline, "get_classifier", lambda lang: EnglishTrained())
    monkeypatch.setattr(pipeline, "token_table", lambda c, t: [])
    monkeypatch.setattr(pipeline, "integrated_gradients", lambda c, t: [])
    _no_evidence(monkeypatch)
    j = client.post("/api/analyze", json={"text": "सरकारने सर्व UPI पेमेंटवर बंदी घातली आहे आणि लोक नाराज आहेत.", "language": "mr"}).json()
    assert "cross_lingual" in j["flags"]


def test_analysis_is_logged_without_user_text_and_history_endpoint(client, monkeypatch, tmp_path):
    import sqlite3
    _no_evidence(monkeypatch)
    text = "The government has banned UPI payments from October."
    assert client.post("/api/analyze", json={"text": text, "language": "en"}).status_code == 200
    items = client.get("/api/history").json()["items"]
    assert len(items) == 1 and items[0]["prediction"] == "uncertain" and "text" not in items[0]
    row = sqlite3.connect(tmp_path / "t.sqlite3").execute("select text, text_sha256, text_chars, embedding_model from analyses").fetchone()
    assert row[0] is None and len(row[1]) == 64 and row[2] == len(text) and row[3].startswith("none")


def test_store_text_opt_in(client, monkeypatch, tmp_path):
    import sqlite3
    monkeypatch.setenv("STORE_TEXT", "true")
    _no_evidence(monkeypatch)
    client.post("/api/analyze", json={"text": "The government has banned UPI payments from October.", "language": "en"})
    assert sqlite3.connect(tmp_path / "t.sqlite3").execute("select text from analyses").fetchone()[0].startswith("The government")


def test_history_can_be_disabled(client, monkeypatch):
    monkeypatch.setenv("ENABLE_HISTORY", "false")
    assert client.get("/api/history").status_code == 404


def test_overall_reasoning_explains_claim_contributions(client, monkeypatch):
    _no_evidence(monkeypatch)
    j = client.post("/api/analyze", json={"text": "The government has banned UPI payments. Fuel prices doubled in 2026.", "language": "en"}).json()
    assert "factual claim(s) analysed" in j["explanation"]["reasoning"] and "Rule:" in j["explanation"]["reasoning"]
    assert j["explanation"]["summary"] is None            # LLM summary is off by default


HI_TEXT = "सरकार ने सभी UPI भुगतानों पर प्रतिबंध लगा दिया है।"
MR_TEXT = "सरकारने सर्व UPI पेमेंटवर बंदी घातली आहे आणि लोक नाराज आहेत."


def _translated_setup(monkeypatch, seen):
    from app.schemas.schemas import Attribution, TokenInfo

    class Capturing(FakeClf):
        def predict(self, text):
            seen["predict"] = text
            return super().predict(text)
    monkeypatch.setenv("TRANSLATE_MODEL", "fake-translator")
    monkeypatch.setattr(pipeline, "get_classifier", lambda lang: Capturing() if lang == "en" else None)
    monkeypatch.setattr(pipeline, "token_table", lambda c, t: seen.__setitem__("tokens", t) or [TokenInfo(token="UPI", raw="UPI", id=1, position=0, special=False, word_start=True)])
    monkeypatch.setattr(pipeline, "integrated_gradients", lambda c, t: seen.__setitem__("ig", t) or [Attribution(token="banned", position=1, word_start=True, score=0.2, normalized=1.0)])
    _no_evidence(monkeypatch)


@pytest.mark.parametrize("text,lang", [(HI_TEXT, "hi"), (MR_TEXT, "mr")])
def test_hindi_and_marathi_are_classified_via_translation(client, monkeypatch, text, lang):
    seen = {}
    _translated_setup(monkeypatch, seen)
    monkeypatch.setattr(pipeline, "translate_to_english",
                        lambda models, l, t: seen.__setitem__("src", (models, l, t)) or ("The government has banned all UPI payments.", "ok", "fake-translator"))
    j = client.post("/api/analyze", json={"text": text, "language": lang}).json()
    assert j["language"] == lang and j["classifier_available"] is True
    assert j["translated_text"] == "The government has banned all UPI payments."
    # the English classifier, token table and attributions all read the TRANSLATION, not the Devanagari text
    assert seen["predict"] == seen["tokens"] == seen["ig"] == "The government has banned all UPI payments."
    assert seen["src"][0] == ["fake-translator"] and seen["src"][1] == lang
    assert j["classifier_status"] == "translated" and "translated" in j["model_caveats"] and "cross_lingual" not in j["flags"]
    assert j["tech"]["translation_model"] == "fake-translator" and j["model_prediction"] == "likely_fake"
    assert j["prediction"] == "uncertain"                       # still no evidence -> final stays Uncertain
    assert any("machine translation" in x or "मशीन" in x for x in j["limitations"])


@pytest.mark.parametrize("tstat,expected", [("no_translator", "unavailable_translator"), ("failed", "unavailable_translation_failed")])
def test_translation_failure_reports_why_and_stays_evidence_only(client, monkeypatch, tstat, expected):
    seen = {}
    _translated_setup(monkeypatch, seen)
    monkeypatch.setattr(pipeline, "translate_to_english", lambda models, l, t: (None, tstat, None))
    j = client.post("/api/analyze", json={"text": HI_TEXT, "language": "hi"}).json()
    assert j["classifier_available"] is False and j["translated_text"] is None and "predict" not in seen
    assert j["classifier_status"] == expected and j["model_prediction"] == "uncertain"


def test_english_path_never_translates(client, monkeypatch):
    seen = {}
    _translated_setup(monkeypatch, seen)
    def boom(*a):
        raise AssertionError("English must not be translated")
    monkeypatch.setattr(pipeline, "translate_to_english", boom)
    j = client.post("/api/analyze", json={"text": "The government has banned UPI payments from October.", "language": "en"}).json()
    assert j["classifier_available"] is True and j["translated_text"] is None
    assert seen["predict"] == "The government has banned UPI payments from October."


def _fake_en_pipeline(monkeypatch):
    from app.schemas.schemas import Attribution, TokenInfo
    monkeypatch.setattr(pipeline, "get_classifier", lambda lang: FakeClf())
    monkeypatch.setattr(pipeline, "token_table", lambda c, t: [TokenInfo(token="UPI", raw="UPI", id=1, position=0, special=False, word_start=True)])
    monkeypatch.setattr(pipeline, "integrated_gradients", lambda c, t: [Attribution(token="banned", position=1, word_start=True, score=0.2, normalized=1.0)])
    _no_evidence(monkeypatch)


def test_short_text_still_gives_a_take_with_a_caveat(client, monkeypatch):
    monkeypatch.setenv("MIN_CLASSIFIER_WORDS", "40")
    _fake_en_pipeline(monkeypatch)                     # FakeClf says "fake, 90%"
    j = client.post("/api/analyze", json={"text": "The government has banned UPI payments from October.", "language": "en"}).json()
    assert j["classifier_available"] is True and j["model_prediction"] == "likely_fake"       # a decision, not "uncertain"
    assert "short_text" in j["model_caveats"] and any("too short" in x for x in j["limitations"])
    assert j["prediction"] == "uncertain"                                                     # final result still needs evidence


def test_long_text_keeps_the_model_estimate(client, monkeypatch):
    monkeypatch.setenv("MIN_CLASSIFIER_WORDS", "40")
    _fake_en_pipeline(monkeypatch)
    text = " ".join(["The finance ministry announced a sweeping new policy on digital payments today."] * 6)
    j = client.post("/api/analyze", json={"text": text, "language": "en"}).json()
    assert j["model_prediction"] == "likely_fake" and "short_text" not in j["model_caveats"] and j["classifier_status"] == "native"


def test_safe_defaults_for_abstention(monkeypatch):
    monkeypatch.delenv("MIN_CLASSIFIER_WORDS", raising=False)
    monkeypatch.delenv("CLASSIFIER_THRESHOLD", raising=False)
    st = get_settings()
    assert st.classifier_min_words == 40 and st.classifier_threshold == 0.85


def test_english_uses_fever_nli_and_indic_uses_multilingual(monkeypatch):
    from app.core.config import stance_model_name
    monkeypatch.delenv("NLI_MODEL", raising=False)
    monkeypatch.delenv("NLI_MODEL_EN", raising=False)
    st = get_settings()
    assert "DeBERTa-v3-large-mnli-fever" in stance_model_name(st, "en")
    assert "mDeBERTa" in stance_model_name(st, "hi") and stance_model_name(st, "mr") == stance_model_name(st, "hi")
    monkeypatch.setenv("NLI_MODEL_EN", "")                     # disabled -> English falls back to the multilingual model
    assert stance_model_name(get_settings(), "en") == stance_model_name(get_settings(), "hi")


def test_stance_model_is_chosen_per_language(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline, "get_stance_model", lambda name: seen.append(name))      # returns None -> no-op
    monkeypatch.setenv("NLI_MODEL_EN", "en-model"); monkeypatch.setenv("NLI_MODEL", "multi-model")
    pipeline._stance_for_news("c", [], get_settings(), "en")
    pipeline._stance_for_news("c", [], get_settings(), "mr")
    assert seen == ["en-model", "multi-model"]


def test_missing_native_hindi_model_falls_back_instead_of_failing(client, monkeypatch):
    _no_evidence(monkeypatch)
    monkeypatch.setenv("MODEL_HI_PATH", "../models/dev-not-trained-yet")
    j = client.post("/api/analyze", json={"text": HI_TEXT, "language": "hi"})
    assert j.status_code == 200
    body = j.json()
    assert body["classifier_available"] is False
    assert any("could not be loaded" in x for x in body["limitations"])


def test_missing_english_model_is_still_a_hard_error(client, monkeypatch):
    monkeypatch.setenv("MODEL_EN_PATH", "definitely/not-a-real-model-xyz")
    assert client.post("/api/analyze", json={"text": "The government has banned UPI payments.", "language": "en"}).status_code == 503


def test_shared_hindi_marathi_model_is_not_flagged_cross_lingual(client, monkeypatch):
    class DevBert(FakeClf):
        meta = {"language": "hi_mr"}
    monkeypatch.setattr(pipeline, "get_classifier", lambda lang: DevBert())
    monkeypatch.setattr(pipeline, "token_table", lambda c, t: [])
    monkeypatch.setattr(pipeline, "integrated_gradients", lambda c, t: [])
    _no_evidence(monkeypatch)
    for lang, text in (("hi", HI_TEXT), ("mr", MR_TEXT)):
        j = client.post("/api/analyze", json={"text": text, "language": lang}).json()
        assert j["classifier_available"] is True and "cross_lingual" not in j["flags"] and j["translated_text"] is None


def test_low_confidence_is_a_caveat_not_an_abstention(client, monkeypatch):
    class Unsure(FakeClf):
        def predict(self, text):
            return {"p_fake": 0.58, "verdict": Verdict.likely_fake, "confidence": 0.58, "confident": False, "n_tokens": 5, "ms": 1}
    _fake_en_pipeline(monkeypatch)
    monkeypatch.setattr(pipeline, "get_classifier", lambda lang: Unsure())
    j = client.post("/api/analyze", json={"text": "The government has banned UPI payments from October.", "language": "en"}).json()
    assert j["model_prediction"] == "likely_fake" and "low_confidence" in j["model_caveats"]


def test_health_exposes_load_errors(client):
    from app.core import diagnostics
    diagnostics.record_error("translator", "x/y", RuntimeError("out of memory"))
    assert "out of memory" in client.get("/api/health").json()["load_errors"]["translator:x/y"]
