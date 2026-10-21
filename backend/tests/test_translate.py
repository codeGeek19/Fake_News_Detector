from app.services.translate import split_chunks


def test_chunks_respect_sentence_boundaries_and_limit():
    text = "सरकार ने नया नियम लागू किया है। लोग नाराज हैं। विपक्ष ने विरोध किया।"
    chunks = split_chunks(text, 40)
    assert all(len(c) <= 40 for c in chunks) and " ".join(chunks).replace("  ", " ") == text


def test_long_sentence_is_cut_not_dropped():
    chunks = split_chunks("a" * 700, 300)
    assert [len(c) for c in chunks] == [300, 300, 100]


def test_empty_and_whitespace():
    assert split_chunks("   ") == []
