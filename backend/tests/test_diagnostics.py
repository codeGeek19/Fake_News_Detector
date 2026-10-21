from app.core import diagnostics
from app.core.config import get_settings, translate_candidates
from app.services.translate import supports, translate_to_english


def test_failed_load_is_retried_after_cooldown_and_error_is_recorded(monkeypatch):
    diagnostics.reset()
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise MemoryError("not enough RAM")
        return "model"
    t = [1000.0]
    monkeypatch.setattr(diagnostics.time, "time", lambda: t[0])
    assert diagnostics.load_cached("x", "m", flaky) is None and "not enough RAM" in diagnostics.errors()["x:m"]
    assert diagnostics.load_cached("x", "m", flaky) is None and calls["n"] == 1        # inside cool-down: not retried
    t[0] += 121
    assert diagnostics.load_cached("x", "m", flaky) == "model" and "x:m" not in diagnostics.errors()   # retried, recovered
    assert diagnostics.load_cached("x", "m", flaky) == "model" and calls["n"] == 2     # success is cached


def test_empty_name_means_disabled():
    assert diagnostics.load_cached("x", None, lambda: 1 / 0) is None


def test_translator_candidates_main_then_language_specific_backup(monkeypatch):
    monkeypatch.delenv("TRANSLATE_MODEL", raising=False)
    monkeypatch.delenv("TRANSLATE_FALLBACK_MR", raising=False)
    monkeypatch.delenv("TRANSLATE_FALLBACK_HI", raising=False)
    s = get_settings()
    assert translate_candidates(s, "mr") == ["facebook/nllb-200-distilled-600M", "Helsinki-NLP/opus-mt-mr-en"]
    assert translate_candidates(s, "hi")[-1] == "Helsinki-NLP/opus-mt-hi-en"
    monkeypatch.setenv("TRANSLATE_MODEL", ""); monkeypatch.setenv("TRANSLATE_FALLBACK_MR", "")
    assert translate_candidates(get_settings(), "mr") == []


def test_marian_models_serve_only_their_own_language():
    assert supports("Helsinki-NLP/opus-mt-mr-en", "mr") and not supports("Helsinki-NLP/opus-mt-mr-en", "hi")
    assert supports("facebook/nllb-200-distilled-600M", "hi") and supports("facebook/nllb-200-distilled-600M", "mr")


def test_unloadable_translators_report_no_translator(monkeypatch):
    diagnostics.reset()
    out = translate_to_english(["Helsinki-NLP/opus-mt-mr-en"], "mr", "सरकारने बंदी घातली आहे.")   # transformers/torch not installed here
    assert out == (None, "no_translator", None) and any("translator" in k for k in diagnostics.errors())
