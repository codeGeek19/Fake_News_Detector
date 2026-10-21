"""Template-based human-readable explanation. Everything stated here comes from model output or
retrieved evidence. (An LLM could paraphrase this later, but must never decide truth or add facts.)"""
from ..schemas.schemas import Explanation, Verdict

T = {
    "en": {
        "labels": {"likely_real": "Likely Real", "likely_fake": "Likely Fake", "uncertain": "Uncertain / insufficient evidence"},
        "conf": "{p}% (the model's own estimate; it is not calibrated and says nothing about whether the claim is true)",
        "why": "The classifier gave the most weight to words such as {words}. In its training data, patterns like these were associated with {side} articles. This reflects writing style, not proof about this specific claim.",
        "why_none": "No token attributions were available.",
        "overall": "Overall: {n} factual claim(s) analysed: {f} likely fake, {r} likely real, {u} uncertain. Rule: the overall result is Likely Fake if any claim is likely fake, Likely Real only if every factual claim is likely real, and Uncertain otherwise.",
        "no_factual": "No checkable factual claim was found in the text.",
        "why_nocl": "No writing-pattern classifier is configured for this language, so this result rests only on the retrieved evidence below.",
        "conf_none": "n/a (no classifier for this language)",
        "limit_noclf": "No writing-pattern classifier is configured for this language; the result rests only on retrieved evidence.",
        "limit_short": "The text is too short for a reliable writing-pattern estimate (the classifier was trained on full news articles), so treat the model's take with extra caution.",
        "limit_translated": "For Hindi/Marathi, the classifier reads a machine translation of your text (the first part only), so translation errors can affect it, and the highlighted words refer to the English translation.",
        "limit_clf_unknown": "The classifier is a pretrained public model; its training data is described on its model card, and it cannot know about recent events.",
        "side": {"fake": "fake", "real": "real"},
        "ev_none": "No sufficiently reliable supporting or contradicting evidence was found.",
        "ev": "Found {n} related articles: {s} supporting, {c} contradicting, {r} related or unclear.",
        "ev_err": " Some sources could not be reached: {p}.",
        "reasons": {
            "factcheck_false": "A published fact-check rates this claim false or misleading.",
            "factcheck_true": "A published fact-check rates this claim accurate.",
            "factcheck_conflict": "Fact-checkers disagree about this claim.",
            "news_contradict": "Several independent news sources report something that conflicts with this claim.",
            "news_support": "Several independent news sources report something consistent with this claim.",
            "insufficient_independent": "Related reports were found, but not enough independent sources to reach a conclusion.",
            "no_evidence": "No sufficiently reliable supporting or contradicting evidence was found.",
            "opinion": "This statement reads as an opinion, so it cannot be classified as true or false.",
        },
        "limits": [
            "The classifier learned patterns from training data up to {cutoff}; it cannot know about newer events.",
            "Token attribution shows what influenced the model, not whether a statement is true.",
            "Evidence was checked at {checked}. Developing stories can change quickly.",
            "News search covers only some publishers. For GDELT results the date shown is when GDELT first saw the article.",
        ],
    },
    "hi": {
        "labels": {"likely_real": "संभावित रूप से सही", "likely_fake": "संभावित रूप से गलत", "uncertain": "पर्याप्त प्रमाण उपलब्ध नहीं हैं"},
        "conf": "{p}% (यह Model का अपना अनुमान है; यह calibrated नहीं है और दावे के सच होने का प्रमाण नहीं है)",
        "why": "Classifier ने {words} जैसे शब्दों को सबसे अधिक महत्व दिया। प्रशिक्षण डेटा में ऐसे पैटर्न {side} समाचारों से जुड़े थे। यह लेखन शैली का संकेत है, इस विशेष दावे का प्रमाण नहीं।",
        "why_none": "Token attribution उपलब्ध नहीं था।",
        "overall": "कुल मिलाकर: {n} तथ्यात्मक दावों की जाँच हुई: {f} संभावित रूप से गलत, {r} संभावित रूप से सही, {u} अनिश्चित। नियम: कोई भी दावा संभावित रूप से गलत हो तो परिणाम 'संभावित रूप से गलत'; सभी दावे संभावित रूप से सही हों तभी 'संभावित रूप से सही'; अन्यथा 'अनिश्चित'।",
        "no_factual": "जाँचने योग्य कोई तथ्यात्मक दावा नहीं मिला।",
        "why_nocl": "इस भाषा के लिए कोई लेखन-पैटर्न classifier उपलब्ध नहीं है, इसलिए यह परिणाम केवल नीचे दिए गए प्रमाणों पर आधारित है।",
        "conf_none": "लागू नहीं (इस भाषा के लिए कोई classifier नहीं)",
        "limit_noclf": "इस भाषा के लिए कोई लेखन-पैटर्न classifier उपलब्ध नहीं है; परिणाम केवल प्राप्त प्रमाणों पर आधारित है।",
        "limit_short": "पाठ लेखन-पैटर्न के भरोसेमंद अनुमान के लिए बहुत छोटा है (classifier पूरे समाचार लेखों पर प्रशिक्षित है), इसलिए Model के आकलन पर अतिरिक्त सावधानी से भरोसा करें।",
        "limit_translated": "हिंदी/मराठी के लिए classifier आपके पाठ का मशीन अनुवाद (केवल शुरुआती भाग) पढ़ता है, इसलिए अनुवाद की गलतियाँ असर डाल सकती हैं और हाइलाइट किए शब्द अंग्रेज़ी अनुवाद के हैं।",
        "limit_clf_unknown": "Classifier एक पहले से प्रशिक्षित सार्वजनिक Model है; उसका प्रशिक्षण डेटा उसके model card में बताया गया है और वह हाल की घटनाओं के बारे में नहीं जान सकता।",
        "side": {"fake": "फ़र्ज़ी", "real": "असली"},
        "ev_none": "कोई पर्याप्त विश्वसनीय समर्थक या विरोधी प्रमाण नहीं मिला।",
        "ev": "{n} संबंधित लेख मिले: {s} समर्थन में, {c} विरोध में, {r} संबंधित या अस्पष्ट।",
        "ev_err": " कुछ स्रोतों तक पहुँचा नहीं जा सका: {p}।",
        "reasons": {
            "factcheck_false": "एक प्रकाशित फैक्ट-चेक इस दावे को गलत या भ्रामक बताता है।",
            "factcheck_true": "एक प्रकाशित फैक्ट-चेक इस दावे को सही बताता है।",
            "factcheck_conflict": "फैक्ट-चेक करने वाली संस्थाओं की राय इस दावे पर अलग-अलग है।",
            "news_contradict": "कई स्वतंत्र समाचार स्रोतों की रिपोर्ट इस दावे से मेल नहीं खाती।",
            "news_support": "कई स्वतंत्र समाचार स्रोत इस दावे से मेल खाती रिपोर्ट देते हैं।",
            "insufficient_independent": "कुछ संबंधित रिपोर्टें मिलीं, लेकिन निष्कर्ष के लिए पर्याप्त स्वतंत्र स्रोत नहीं हैं।",
            "no_evidence": "कोई पर्याप्त विश्वसनीय समर्थक या विरोधी प्रमाण नहीं मिला।",
            "opinion": "यह कथन राय जैसा लगता है, इसलिए इसे सही/गलत में नहीं बाँटा जा सकता।",
        },
        "limits": [
            "Classifier ने {cutoff} तक के प्रशिक्षण डेटा से पैटर्न सीखे हैं; यह नई घटनाओं के बारे में नहीं जान सकता।",
            "Token attribution यह दिखाता है कि Model किससे प्रभावित हुआ, यह नहीं कि कथन सच है या नहीं।",
            "प्रमाण {checked} पर जाँचे गए। विकसित हो रही खबरें तेज़ी से बदल सकती हैं।",
            "समाचार खोज केवल कुछ प्रकाशकों को कवर करती है। GDELT परिणामों में दिखाई गई तारीख वह है जब GDELT ने लेख पहली बार देखा।",
        ],
    },
}


def reason_text(code: str, lang: str) -> str:
    return T[lang]["reasons"].get(code, code)


def build(lang: str, final: Verdict, model_conf: float, attributions: list, sources: dict,
          claims: list, statuses: list, cutoff: str, checked: str, has_clf: bool = True, translated: bool = False, short_text: bool = False):
    t = T[lang]
    top = sorted([a for a in attributions if abs(a.normalized) > 0.05], key=lambda a: -abs(a.normalized))[:5]
    if not has_clf:
        why = t["why_nocl"]
    elif top:
        side = t["side"]["fake" if sum(a.score for a in top) > 0 else "real"]
        why = t["why"].format(words=", ".join(f"“{a.token}”" for a in top), side=side)
    else:
        why = t["why_none"]
    s, c, r = len(sources["supporting"]), len(sources["contradicting"]), len(sources["related"])
    n = s + c + r
    ev = t["ev"].format(n=n, s=s, c=c, r=r) if n else t["ev_none"]
    bad = [x.provider for x in statuses if not x.ok and x.error != "disabled"]
    if bad:
        ev += t["ev_err"].format(p=", ".join(bad))
    codes = [code for cl in claims for code in cl.reasons]
    reasoning = " ".join(dict.fromkeys(t["reasons"].get(x, x) for x in codes))
    fac = [c for c in claims if c.claim_type == "factual"]
    cnt = lambda v: sum(c.verdict.value == v for c in fac)   # noqa: E731
    reasoning = (reasoning + " " + (t["overall"].format(n=len(fac), f=cnt("likely_fake"), r=cnt("likely_real"), u=cnt("uncertain"))
                                    if fac else t["no_factual"])).strip()
    limits = [x.format(cutoff=cutoff, checked=checked) for x in t["limits"]]
    if translated:
        limits.insert(1, t["limit_translated"])
    if short_text:
        limits.insert(1, t["limit_short"])
    if not has_clf:
        limits[0] = t["limit_noclf"]
    elif cutoff.startswith("unknown"):
        limits[0] = t["limit_clf_unknown"]
    return Explanation(prediction=t["labels"][final.value],
                       confidence=t["conf"].format(p=round(model_conf * 100)) if has_clf else t["conf_none"],
                       why_model=why, evidence=ev, reasoning=reasoning), limits


T["mr"] = {
    "labels": {"likely_real": "बहुधा खरे", "likely_fake": "बहुधा खोटे", "uncertain": "पुरेसे पुरावे उपलब्ध नाहीत"},
    "conf": "{p}% (हा Model चा स्वतःचा अंदाज आहे; तो calibrated नाही आणि दावा खरा असल्याचा पुरावा नाही)",
    "why": "Classifier ने {words} सारख्या शब्दांना सर्वाधिक महत्त्व दिले. प्रशिक्षण डेटामध्ये असे नमुने {side} बातम्यांशी जोडलेले होते. हे लेखनशैलीचे संकेत आहे, या विशिष्ट दाव्याचा पुरावा नाही.",
    "why_none": "Token attribution उपलब्ध नव्हते.",
    "overall": "एकूण: {n} तथ्यात्मक दाव्यांची तपासणी झाली: {f} बहुधा खोटे, {r} बहुधा खरे, {u} अनिश्चित. नियम: कोणताही दावा बहुधा खोटा असल्यास निकाल 'बहुधा खोटे'; सर्व दावे बहुधा खरे असतील तरच 'बहुधा खरे'; अन्यथा 'अनिश्चित'.",
    "no_factual": "तपासण्यायोग्य तथ्यात्मक दावा सापडला नाही.",
    "why_nocl": "या भाषेसाठी लेखनशैली ओळखणारा classifier उपलब्ध नाही, त्यामुळे हा निकाल फक्त खाली दिलेल्या पुराव्यांवर आधारित आहे.",
    "conf_none": "लागू नाही (या भाषेसाठी classifier नाही)",
    "limit_noclf": "या भाषेसाठी लेखनशैली ओळखणारा classifier उपलब्ध नाही; निकाल फक्त मिळालेल्या पुराव्यांवर आधारित आहे.",
    "limit_short": "मजकूर लेखनशैलीच्या विश्वासार्ह अंदाजासाठी खूपच लहान आहे (classifier पूर्ण बातम्यांच्या लेखांवर प्रशिक्षित आहे), त्यामुळे Model च्या मताकडे अधिक सावधपणे पाहा.",
    "limit_translated": "हिंदी/मराठीसाठी classifier तुमच्या मजकुराचे मशीन भाषांतर (फक्त सुरुवातीचा भाग) वाचतो, त्यामुळे भाषांतरातील चुकांचा परिणाम होऊ शकतो आणि हायलाइट केलेले शब्द इंग्रजी भाषांतराचे आहेत.",
    "limit_clf_unknown": "Classifier हा आधीच प्रशिक्षित सार्वजनिक Model आहे; त्याचा प्रशिक्षण डेटा त्याच्या model card मध्ये दिलेला आहे आणि त्याला अलीकडील घटनांची माहिती नसते.",
    "side": {"fake": "खोट्या", "real": "खऱ्या"},
    "ev_none": "पुरेसा विश्वासार्ह समर्थक किंवा विरोधी पुरावा सापडला नाही.",
    "ev": "{n} संबंधित लेख सापडले: {s} समर्थनात, {c} विरोधात, {r} संबंधित किंवा अस्पष्ट.",
    "ev_err": " काही स्रोतांपर्यंत पोहोचता आले नाही: {p}.",
    "reasons": {
        "factcheck_false": "प्रकाशित फॅक्ट-चेकमध्ये हा दावा खोटा किंवा दिशाभूल करणारा ठरवला आहे.",
        "factcheck_true": "प्रकाशित फॅक्ट-चेकमध्ये हा दावा खरा ठरवला आहे.",
        "factcheck_conflict": "या दाव्याबाबत फॅक्ट-चेकर्समध्ये मतभेद आहेत.",
        "news_contradict": "अनेक स्वतंत्र बातमी स्रोतांचे वृत्त या दाव्याशी जुळत नाही.",
        "news_support": "अनेक स्वतंत्र बातमी स्रोतांचे वृत्त या दाव्याशी जुळते.",
        "insufficient_independent": "काही संबंधित वृत्ते सापडली, पण निष्कर्षासाठी पुरेसे स्वतंत्र स्रोत नाहीत.",
        "no_evidence": "पुरेसा विश्वासार्ह समर्थक किंवा विरोधी पुरावा सापडला नाही.",
        "opinion": "हे विधान मतासारखे वाटते, त्यामुळे ते खरे/खोटे असे वर्गीकृत करता येत नाही.",
    },
    "limits": [
        "Classifier ने {cutoff} पर्यंतच्या प्रशिक्षण डेटामधून नमुने शिकले आहेत; त्याला नवीन घटनांची माहिती नसते.",
        "Token attribution फक्त Model वर काय परिणाम झाला ते दाखवते, विधान खरे आहे की नाही हे नाही.",
        "पुरावे {checked} रोजी तपासले. घडत असलेल्या बातम्या वेगाने बदलू शकतात.",
        "बातम्यांचा शोध फक्त काही प्रकाशकांना व्यापतो. GDELT निकालांमध्ये दाखवलेली तारीख म्हणजे GDELT ने लेख पहिल्यांदा पाहिल्याची तारीख.",
    ],
}
