from app.preprocessing.language import detect_language
from app.preprocessing.claims import extract_claims, build_queries
from app.preprocessing.textutils import words, normalize


def test_english():
    r = detect_language("India launched a new satellite this week.")
    assert r["language"] == "en" and r["script"] == "latin" and not r["is_hinglish"]


def test_devanagari():
    r = detect_language("भारत ने नया उपग्रह लॉन्च किया है।")
    assert r["language"] == "hi" and r["script"] == "devanagari"


def test_hinglish_is_not_english():
    r = detect_language("Modi ji ne ye announcement kiya hai")
    assert r["language"] == "hi" and r["is_hinglish"] and r["script"] == "latin"


def test_toggle_mismatch_flag():
    assert detect_language("भारत ने नया उपग्रह लॉन्च किया है।", "en")["mismatch"] is True
    assert detect_language("India launched a satellite today.", "auto")["mismatch"] is False


def test_urls_stripped_and_not_followed():
    assert "http" not in normalize("see http://evil.example/x now")


def test_hindi_words_keep_matras():
    assert "प्रतिबंध" in words("सरकार ने UPI पर प्रतिबंध लगाया।")


def test_claim_extraction_strips_attribution():
    cs = extract_claims("According to social media users, the government has banned UPI. This happened yesterday and people are angry.")
    assert any("government has banned UPI" in c.text for c in cs)
    assert all(not c.text.lower().startswith("according to") for c in cs)


def test_opinion_not_factual():
    cs = extract_claims("In my opinion this policy is shameful.")
    assert cs[0].claim_type == "opinion"


def test_hindi_claims_split_on_danda():
    cs = extract_claims("सूत्रों के अनुसार, सरकार ने UPI पर प्रतिबंध लगा दिया है। यह कल हुआ।")
    assert cs and cs[0].claim_type == "factual"


def test_short_headline_kept():
    assert extract_claims("UPI banned today")[0].text == "UPI banned today"


def test_queries():
    q = build_queries("India has banned all UPI payments from October 2026.")
    assert "upi" in q["keywords"] and len(q["keywords"].split()) <= 5


def test_marathi_detected_by_function_words():
    r = detect_language("सरकारने सर्व UPI पेमेंटवर बंदी घातली आहे आणि लोक नाराज आहेत.")
    assert r["language"] == "mr" and r["script"] == "devanagari"


def test_hindi_not_confused_with_marathi():
    assert detect_language("सरकार ने सभी UPI भुगतानों पर प्रतिबंध लगा दिया है।")["language"] == "hi"


def test_devanagari_tie_uses_toggle():
    assert detect_language("भारत सरकार", "mr")["language"] == "mr"
    assert detect_language("भारत सरकार", "hi")["language"] == "hi"
    assert detect_language("भारत सरकार")["language"] == "hi"


def test_marathi_claim_attribution_stripped():
    cs = extract_claims("सूत्रांच्या मते, सरकारने UPI वर बंदी घातली आहे. ही बातमी काल आली.")
    assert cs and not cs[0].text.startswith("सूत्रांच्या")
