from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, HttpUrl


class Verdict(str, Enum):
    likely_real = "likely_real"
    likely_fake = "likely_fake"
    uncertain = "uncertain"


class Stance(str, Enum):
    supports = "supports"
    contradicts = "contradicts"
    related = "related"


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=10, max_length=20000)  # hard cap; env MAX_INPUT_CHARS is enforced in the route
    language: Literal["en", "hi", "mr", "auto"] = "auto"


class Source(BaseModel):
    title: str
    publisher: str
    url: HttpUrl                      # copied only from API responses, never generated
    published_at: Optional[datetime] = None   # None when the provider gave no date; never guessed
    snippet: Optional[str] = None
    stance: Stance = Stance.related
    stance_confidence: float = 0.0
    relevance: float = 0.0
    provenance: Literal["factcheck", "gdelt", "rss", "newsapi"]
    rating: Optional[str] = None      # a fact-checker's own textual rating
    source_type: Optional[str] = None  # wire / newspaper / factchecker ... (metadata, not a truth score)


class EvidenceStatus(BaseModel):
    provider: str
    ok: bool
    error: Optional[str] = None


class TokenInfo(BaseModel):
    token: str
    raw: str
    id: int
    position: int
    special: bool
    word_start: bool
    attention: Optional[float] = None


class Attribution(BaseModel):
    token: str
    position: int
    word_start: bool
    score: float          # raw Integrated Gradients contribution toward P(fake)
    normalized: float     # score / max|score|, in [-1, 1]; + = toward fake, - = toward real


class ClaimResult(BaseModel):
    text: str
    claim_type: Literal["factual", "opinion"]
    verdict: Verdict
    reasons: list[str] = []
    supporting_sources: list[Source] = []
    contradicting_sources: list[Source] = []
    related_articles: list[Source] = []


class Explanation(BaseModel):
    prediction: str
    confidence: str
    why_model: str
    evidence: str
    reasoning: str
    summary: Optional[str] = None     # optional LLM paraphrase of the evidence; never decides the verdict


class TechDetails(BaseModel):
    model_name: str
    nli_model: str
    embedding_model: str = "none (lexical ranking only)"
    llm_model: Optional[str] = None
    translation_model: Optional[str] = None
    model_version: str
    training_cutoff: str
    language: str
    n_tokens: int
    inference_ms: int
    total_ms: int
    evidence_checked_at: datetime


class AnalyzeResponse(BaseModel):
    language: str
    script: str
    is_hinglish: bool
    language_mismatch: bool
    prediction: Verdict               # FINAL, evidence-based
    model_prediction: Verdict         # classifier alone
    model_confidence: float
    classifier_available: bool
    classifier_status: str = "native"      # native | translated | unavailable_no_model | unavailable_english_model | unavailable_translator | unavailable_translation_failed
    model_caveats: list[str] = []          # short_text | low_confidence | translated | cross_lingual: why the model's take may be unreliable
    translated_text: Optional[str] = None     # set when the classifier read a machine translation (Hindi/Marathi)
    claims: list[ClaimResult]
    tokens: list[TokenInfo]
    attributions: list[Attribution]
    explanation: Explanation
    supporting_sources: list[Source]
    contradicting_sources: list[Source]
    related_articles: list[Source]
    evidence_status: list[EvidenceStatus]
    flags: list[str] = []             # opinion | satire_flagged | developing_story | unverified
    limitations: list[str]
    tech: TechDetails
