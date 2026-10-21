from ..schemas.schemas import Attribution
from .tokens import display, word_starts


def integrated_gradients(clf, text: str, n_steps: int = 24) -> list:
    """Token attribution toward P(fake) using Integrated Gradients (Captum) on the embedding layer.
    + = pushes toward FAKE, - = pushes toward REAL. It describes what the MODEL used, not what is TRUE."""
    import torch
    from captum.attr import LayerIntegratedGradients
    enc = clf.tok(text, return_tensors="pt", truncation=True, max_length=clf.max_len)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    sp_ids = set(clf.tok.all_special_ids)
    baseline = torch.full_like(ids, clf.tok.pad_token_id)
    for sid in sp_ids:                       # keep [CLS]/[SEP] in the baseline
        baseline[ids == sid] = sid

    def fwd(input_ids, attention_mask):
        logits = clf.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return torch.softmax(logits, -1)[:, clf.fake_idx]

    lig = LayerIntegratedGradients(fwd, clf.model.get_input_embeddings())
    attr = lig.attribute(ids, baselines=baseline, additional_forward_args=(mask,),
                         n_steps=n_steps, internal_batch_size=8)
    scores = attr.sum(-1)[0].detach()
    raw = clf.tok.convert_ids_to_tokens(ids[0].tolist())
    special = [i in sp_ids for i in ids[0].tolist()]
    ws = word_starts(raw, special)
    m = float(scores.abs().max()) or 1.0
    return [Attribution(token=display(r) or r, position=p, word_start=ws[p],
                        score=round(float(scores[p]), 5), normalized=round(float(scores[p]) / m, 4))
            for p, r in enumerate(raw) if not special[p]]
