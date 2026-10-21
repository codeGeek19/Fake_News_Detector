import asyncio
import logging
import os
from fastapi import APIRouter, Depends, HTTPException, Request
from ..core import db, diagnostics
from ..core.config import get_settings, model_path
from ..core.ratelimit import rate_limit
from ..retrieval.common import source_types
from ..schemas.schemas import AnalyzeRequest, AnalyzeResponse
from ..services import pipeline
from ..services.classifier import ModelNotInstalled, read_metadata

router = APIRouter(prefix="/api")
log = logging.getLogger("fnd.api")


@router.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(rate_limit)])
async def analyze(req: AnalyzeRequest, request: Request):
    s = get_settings()
    if len(req.text) > s.max_chars:
        raise HTTPException(413, f"Text too long (max {s.max_chars} characters).")
    try:
        resp = await asyncio.wait_for(pipeline.analyze(req, request.app.state.http, s), timeout=s.analyze_timeout)
        try:                                   # logging must never break an analysis
            await asyncio.to_thread(db.save_analysis, s.db_path, resp, req.text, s.store_text)
        except Exception:
            log.exception("could not save analysis to SQLite")
        return resp
    except ModelNotInstalled as e:
        raise HTTPException(503, f"Model not installed: {e}")
    except asyncio.TimeoutError:
        raise HTTPException(504, "Analysis timed out. Please try again.")
    except HTTPException:
        raise
    except Exception:
        log.exception("analyze failed")
        raise HTTPException(500, "Internal error while analysing the text.")


@router.get("/health")
async def health():
    s = get_settings()
    return {"status": "ok", "load_errors": diagnostics.errors(),
            "models": {l: read_metadata(l)["status"] for l in ("en", "hi", "mr")},
            "providers": {"google_factcheck": bool(s.factcheck_key), "gdelt": s.enable_gdelt,
                          "google_news_rss": s.enable_google_news_rss, "newsapi": bool(s.newsapi_key),
                          "nli_stance_en": bool(s.nli_model_en or s.nli_model), "nli_stance_indic": bool(s.nli_model), "translate_then_classify": bool(s.translate_model)}}


@router.get("/models")
async def models():
    return {"models": [read_metadata(l) for l in ("en", "hi", "mr")]}


@router.get("/sources")
async def sources():
    return {"note": "Publisher type is metadata used as one weak ranking signal. It is not a truth score.",
            "publisher_types": source_types(),
            "providers": ["google_factcheck", "gdelt", "google_news_rss", "newsapi"]}


@router.get("/history")
async def history(limit: int = 20):
    """Recent analyses (metadata only: no user text). Disable with ENABLE_HISTORY=false."""
    s = get_settings()
    if not s.enable_history:
        raise HTTPException(404, "History is disabled.")
    return {"items": await asyncio.to_thread(db.recent, s.db_path, limit)}
