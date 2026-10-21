import asyncio
import logging
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router
from .core.config import get_settings

load_dotenv()   # reads .env from the repo root / backend folder if present
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _preload():
    """Load models at startup (in the background) so the first request isn't slow."""
    from .retrieval.nli import get_stance_model
    from .services.classifier import get_classifier, get_tokenizer_holder
    s = get_settings()
    for lang in ("en", "hi", "mr"):
        try:
            get_classifier(lang)
        except Exception:
            logging.getLogger("fnd").exception("preload failed for %s", lang)
    from .core.config import stance_model_name
    for lang in ("en", "hi"):
        get_stance_model(stance_model_name(s, lang))
    get_tokenizer_holder(s.nli_model, s.max_tokens)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(headers={"User-Agent": "fake-news-detector/1.0 (academic project)"})
    if get_settings().preload:
        asyncio.create_task(asyncio.to_thread(_preload))
    yield
    await app.state.http.aclose()


app = FastAPI(title="Bilingual Fake News Verification API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.include_router(router)
