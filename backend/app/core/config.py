import os
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TRANSLATE = "facebook/nllb-200-distilled-600M"
DEFAULT_NLI = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEFAULT_NLI_EN = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")


@dataclass
class Settings:
    factcheck_key: Optional[str]
    newsapi_key: Optional[str]
    enable_gdelt: bool
    enable_google_news_rss: bool
    nli_model: Optional[str]
    nli_model_en: Optional[str]
    max_tokens: int
    classifier_min_words: int
    classifier_threshold: float
    preload: bool
    embed_model: Optional[str]
    translate_model: Optional[str]
    translate_max_chars: int
    translate_fallback_hi: Optional[str]
    translate_fallback_mr: Optional[str]
    db_path: str
    store_text: bool
    enable_history: bool
    anthropic_key: Optional[str]
    llm_model: str
    enable_llm_summary: bool
    max_chars: int
    rate_limit_per_min: int
    request_timeout: float
    analyze_timeout: float
    cache_ttl: int
    cors_origins: list = field(default_factory=list)


def get_settings() -> Settings:
    """Read env on every call so tests can monkeypatch. Keys never leave the backend."""
    return Settings(
        factcheck_key=os.getenv("GOOGLE_FACTCHECK_API_KEY") or None,
        newsapi_key=os.getenv("NEWSAPI_KEY") or None,
        enable_gdelt=_bool("ENABLE_GDELT", True),
        enable_google_news_rss=_bool("ENABLE_GOOGLE_NEWS_RSS", True),
        nli_model=os.getenv("NLI_MODEL", DEFAULT_NLI) or None,        # Hindi/Marathi stance model; empty = off
        nli_model_en=os.getenv("NLI_MODEL_EN", DEFAULT_NLI_EN) or None,  # English stance model; empty = use NLI_MODEL
        max_tokens=int(os.getenv("MAX_TOKENS", "256")),
        classifier_min_words=int(os.getenv("MIN_CLASSIFIER_WORDS", "40")),     # below this the model estimate is shown as Uncertain
        classifier_threshold=float(os.getenv("CLASSIFIER_THRESHOLD", "0.85")),  # commit to fake/real only at >= this probability
        preload=_bool("PRELOAD_MODELS", False),
        embed_model=os.getenv("EMBED_MODEL", DEFAULT_EMBED) or None,    # empty = lexical ranking only
        translate_model=os.getenv("TRANSLATE_MODEL", DEFAULT_TRANSLATE) or None,   # empty = no translate-then-classify
        translate_max_chars=int(os.getenv("TRANSLATE_MAX_CHARS", "1200")),
        # small single-language backups, tried if the main translator cannot load (hi-en: existence not verified by me)
        translate_fallback_hi=os.getenv("TRANSLATE_FALLBACK_HI", "Helsinki-NLP/opus-mt-hi-en") or None,
        translate_fallback_mr=os.getenv("TRANSLATE_FALLBACK_MR", "Helsinki-NLP/opus-mt-mr-en") or None,
        db_path=os.getenv("DB_PATH", "data/fnd.sqlite3"),
        store_text=_bool("STORE_TEXT", False),                          # off: only a hash of the input is stored
        enable_history=_bool("ENABLE_HISTORY", True),
        anthropic_key=os.getenv("ANTHROPIC_API_KEY") or None,
        llm_model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
        enable_llm_summary=_bool("ENABLE_LLM_SUMMARY", False),
        max_chars=int(os.getenv("MAX_INPUT_CHARS", "5000")),
        rate_limit_per_min=int(os.getenv("RATE_LIMIT_PER_MIN", "20")),
        request_timeout=float(os.getenv("REQUEST_TIMEOUT_SEC", "8")),
        analyze_timeout=float(os.getenv("ANALYZE_TIMEOUT_SEC", "40")),
        cache_ttl=int(os.getenv("CACHE_TTL_SEC", "900")),
        cors_origins=[o.strip() for o in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()],
    )


def model_path(lang: str) -> Optional[str]:
    return os.getenv(f"MODEL_{lang.upper()}_PATH") or None


def fake_label_override(lang: str) -> Optional[str]:
    """Which output label of a pretrained classifier means FAKE (name or index), e.g. LABEL_0."""
    return os.getenv(f"MODEL_{lang.upper()}_FAKE_LABEL") or None


def stance_model_name(s: "Settings", lang: str) -> Optional[str]:
    """English uses the FEVER-trained large NLI model; Hindi/Marathi use the multilingual one."""
    return (s.nli_model_en or s.nli_model) if lang == "en" else s.nli_model


def translate_candidates(s: "Settings", lang: str) -> list:
    """Translators to try, in order, for Hindi/Marathi -> English."""
    backup = {"hi": s.translate_fallback_hi, "mr": s.translate_fallback_mr}.get(lang)
    out = []
    for name in (s.translate_model, backup):
        if name and name not in out:
            out.append(name)
    return out
