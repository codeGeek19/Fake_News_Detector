import httpx
import pytest
from app.retrieval.factcheck import search_factchecks
from app.retrieval.gdelt import search_gdelt
from app.retrieval.ranking import rank
from app.schemas.schemas import Source, Stance, Verdict
from app.services import verdict


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_factcheck_missing_key():
    async with client(lambda r: httpx.Response(200)) as c:
        out, st = await search_factchecks("x", "en", None, c)
    assert out == [] and not st.ok and "key" in st.error.lower()


async def test_factcheck_timeout_is_reported_not_raised():
    def boom(request):
        raise httpx.ReadTimeout("slow", request=request)
    async with client(boom) as c:
        out, st = await search_factchecks("x", "en", "K", c)
    assert out == [] and not st.ok and st.error == "ReadTimeout"


async def test_factcheck_parses_and_skips_urlless():
    body = {"claims": [{"text": "UPI banned", "claimReview": [
        {"url": "https://example.org/fc1", "title": "No, UPI is not banned", "textualRating": "False",
         "publisher": {"name": "ExampleFacts"}, "reviewDate": "2026-10-01T00:00:00Z"},
        {"title": "no url", "textualRating": "False"}]}]}
    async with client(lambda r: httpx.Response(200, json=body)) as c:
        out, st = await search_factchecks("UPI", "en", "K", c)
    assert st.ok and len(out) == 1 and out[0].publisher == "ExampleFacts" and out[0].published_at.year == 2026


async def test_gdelt_non_json_handled():
    async with client(lambda r: httpx.Response(200, text="Query too short")) as c:
        out, st = await search_gdelt("a", "en", True, c)
    assert out == [] and not st.ok


async def test_gdelt_parses():
    body = {"articles": [{"url": "https://www.thehindu.com/a", "title": "UPI services normal", "domain": "thehindu.com",
                          "seendate": "20261001T101500Z"}]}
    async with client(lambda r: httpx.Response(200, json=body)) as c:
        out, st = await search_gdelt("UPI", "en", True, c)
    assert st.ok and out[0].source_type == "newspaper" and out[0].published_at.day == 1


def src(title, url="https://a.example/x", prov="gdelt", rating=None, stance=Stance.related, conf=0.0):
    return Source(title=title, publisher="P", url=url, provenance=prov, rating=rating, stance=stance, stance_confidence=conf)


def test_rank_drops_irrelevant_and_dedups():
    out = rank("government banned UPI payments", [src("UPI payments banned by government"), src("Cricket match report"),
                                                  src("UPI payments banned by government")])
    assert len(out) == 1


def test_verdict_no_evidence_is_uncertain():
    v, reasons, flags = verdict.claim_verdict([])
    assert v == Verdict.uncertain and reasons == ["no_evidence"]


def test_verdict_factcheck_false():
    v, r, _ = verdict.claim_verdict([src("x", prov="factcheck", rating="False")])
    assert v == Verdict.likely_fake and r == ["factcheck_false"]


def test_verdict_conflict():
    v, r, _ = verdict.claim_verdict([src("x", prov="factcheck", rating="False"), src("y", url="https://b.example/y", prov="factcheck", rating="True")])
    assert v == Verdict.uncertain and r == ["factcheck_conflict"]


def test_news_needs_independent_sources():
    one = [src("x", stance=Stance.contradicts, conf=0.95)]
    two = one + [src("y", url="https://b.example/y", stance=Stance.contradicts, conf=0.95)]
    assert verdict.claim_verdict(one)[0] == Verdict.uncertain
    assert verdict.claim_verdict(two)[0] == Verdict.likely_fake


def test_satire_flag():
    _, _, flags = verdict.claim_verdict([src("x", prov="factcheck", rating="Satire")])
    assert "satire_flagged" in flags


async def test_newsapi_skipped_for_non_english():
    from app.retrieval.newsapi import search_newsapi
    async with client(lambda r: (_ for _ in ()).throw(AssertionError("must not call the API"))) as c:
        out, st = await search_newsapi("x", "mr", "KEY", c)
    assert out == [] and st.error == "disabled"


def test_marathi_factcheck_rating_words():
    assert verdict.claim_verdict([src("x", prov="factcheck", rating="खोटे")])[0] == Verdict.likely_fake
    assert verdict.claim_verdict([src("x", prov="factcheck", rating="खरे")])[0] == Verdict.likely_real
