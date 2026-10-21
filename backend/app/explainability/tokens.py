from ..schemas.schemas import TokenInfo


def display(t: str) -> str:
    return t.replace("▁", "").replace("Ġ", "").replace("##", "")


def word_starts(raw: list, special: list) -> list:
    prefix_style = any(t.startswith(("▁", "Ġ")) for t in raw)   # sentencepiece / byte-level BPE
    out, first = [], True
    for t, sp in zip(raw, special):
        if sp:
            out.append(False)
            continue
        ws = first or (t.startswith(("▁", "Ġ")) if prefix_style else not t.startswith("##"))
        out.append(ws)
        first = False
    return out


def token_table(clf, text: str) -> list:
    """Tokens, ids, positions and (if available) last-layer CLS attention. Attention is shown for
    education only: raw attention is NOT a faithful explanation of the prediction."""
    import torch
    enc = clf.tok(text, return_tensors="pt", truncation=True, max_length=clf.max_len)
    ids = enc["input_ids"][0].tolist()
    raw = clf.tok.convert_ids_to_tokens(ids)
    sp_ids = set(clf.tok.all_special_ids)
    special = [i in sp_ids for i in ids]
    att = None
    try:
        if getattr(clf, "model", None) is None:
            raise RuntimeError("tokenizer only")
        with torch.no_grad():
            out = clf.model(**enc, output_attentions=True)
        if out.attentions and out.attentions[-1] is not None:
            att = out.attentions[-1][0].mean(0)[0].tolist()
    except Exception:
        att = None
    ws = word_starts(raw, special)
    return [TokenInfo(token=display(r) or r, raw=r, id=i, position=p, special=s, word_start=w,
                      attention=round(att[p], 4) if att else None)
            for p, (r, i, s, w) in enumerate(zip(raw, ids, special, ws))]
