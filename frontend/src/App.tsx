import { useState } from "react";
import { analyze } from "./api";
import { T } from "./i18n";
import type { AnalyzeResponse, Lang } from "./types";
import LanguageToggle from "./components/LanguageToggle";
import VerdictBadge from "./components/VerdictBadge";
import Section from "./components/Section";
import TokenTable from "./components/TokenTable";
import AttributionText from "./components/AttributionText";
import EvidenceList, { SourceCard } from "./components/EvidenceList";
import HowItWorks from "./components/HowItWorks";

export default function App() {
  const [lang, setLang] = useState<Lang>("en");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [res, setRes] = useState<AnalyzeResponse | null>(null);
  const t = T[lang];

  async function submit() {
    if (text.trim().length < 10 || loading) return;
    setLoading(true); setError(null); setRes(null);
    try { setRes(await analyze(text, lang)); }
    catch (e) { setError(e instanceof Error && e.message !== "NETWORK" ? e.message : t.errNetwork); }
    finally { setLoading(false); }
  }

  const all = res ? [...res.supporting_sources, ...res.contradicting_sources, ...res.related_articles] : [];

  return (
    <main className="mx-auto max-w-4xl space-y-5 px-4 py-8" lang={lang}>
      <header className="space-y-3 text-center">
        <h1 className="text-3xl font-bold tracking-wide">{t.title}</h1>
        <p className="text-sm text-slate-600">{t.subtitle}</p>
        <LanguageToggle value={lang} onChange={setLang} />
      </header>

      <div className="space-y-3">
        <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder={t.placeholder} rows={7}
          maxLength={5000} className="w-full rounded-xl border border-slate-300 p-4 focus:outline-none focus:ring-2 focus:ring-indigo-500" />
        <div className="text-center">
          <button type="button" onClick={submit} disabled={loading || text.trim().length < 10}
            className="rounded-full bg-indigo-600 px-8 py-3 font-semibold text-white disabled:opacity-50">
            {loading ? t.analyzing : t.analyze}
          </button>
        </div>
      </div>

      {loading && <div role="status" className="text-center text-slate-500">{t.analyzing}</div>}
      {error && <div role="alert" className="rounded-xl border border-red-300 bg-red-50 p-4 text-red-800"><strong>{t.error}:</strong> {error}</div>}

      {res && (
        <div className="space-y-5">
          {res.language_mismatch && <p className="rounded-lg bg-amber-50 p-3 text-sm text-amber-800">{t.mismatch}</p>}
          {res.is_hinglish && <p className="rounded-lg bg-amber-50 p-3 text-sm text-amber-800">{t.hinglish}</p>}

          <Section title={t.result}>
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-3">
                <span className="text-sm text-slate-500">{t.finalAssess}:</span>
                <VerdictBadge verdict={res.prediction} label={t.verdict[res.prediction]} />
              </div>
              <div data-testid="model-take" className="flex flex-wrap items-center gap-3 text-sm">
                <span className="text-slate-500">{t.modelSays}:</span>
                {res.classifier_available ? (
                  <>
                    <VerdictBadge verdict={res.model_prediction} label={t.verdict[res.model_prediction]} size="sm" />
                    <span>{t.confidence}: {Math.round(res.model_confidence * 100)}%</span>
                    {res.model_caveats.map((c) => (
                      <span key={c} data-testid="caveat" className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-800">
                        {t.caveat[c as keyof typeof t.caveat] ?? c}
                      </span>
                    ))}
                  </>
                ) : (
                  <span data-testid="take-unavailable" className="text-slate-500">
                    {t.takeUnavailable}: {t.statusReason[res.classifier_status as keyof typeof t.statusReason] ?? res.classifier_status}
                  </span>
                )}
              </div>
              {res.flags.length > 0 && <div className="flex flex-wrap gap-2">{res.flags.map((f) =>
                <span key={f} className="rounded-full bg-slate-100 px-3 py-1 text-xs">{t.flags[f as keyof typeof t.flags] ?? f}</span>)}</div>}
            </div>
          </Section>

          {res.claims.length > 0 && (
            <Section title={t.claims}>
              <ol className="space-y-3">{res.claims.map((c, i) => (
                <li key={i} className="rounded-lg border border-slate-200 p-3">
                  <div className="mb-1 flex items-center gap-2"><strong>#{i + 1}</strong><VerdictBadge verdict={c.verdict} label={t.verdict[c.verdict]} size="sm" /></div>
                  <p className="text-sm">{c.text}</p>
                  {c.reasons.length > 0 && <p className="mt-1 text-xs text-slate-500">{c.reasons.join(" ")}</p>}
                </li>))}</ol>
            </Section>
          )}

          <Section title={t.why}>
            <div className="space-y-2 text-sm leading-6">
              <p>{res.explanation.why_model}</p>
              <p>{res.explanation.evidence}</p>
              {res.explanation.reasoning && <p>{res.explanation.reasoning}</p>}
              {res.explanation.summary && (
                <div data-testid="llm-summary" className="rounded-lg border border-indigo-200 bg-indigo-50 p-3">
                  <p className="font-medium">{t.summaryTitle}</p>
                  <p>{res.explanation.summary}</p>
                  <p className="mt-1 text-xs text-slate-500">{t.summaryNote}</p>
                </div>
              )}
              {res.classifier_available && <p className="text-slate-500">{t.confidence}: {res.explanation.confidence}</p>}
            </div>
            <HowItWorks t={t} />
          </Section>

          {res.translated_text && (
            <Section title={t.translationTitle}>
              <p data-testid="translation" className="text-sm leading-6">{res.translated_text}</p>
            </Section>
          )}
          {res.attributions.length > 0 && <Section title={t.importantWords}><AttributionText attributions={res.attributions} t={t} /></Section>}
          {res.tokens.length > 0 && <Section title={t.tokenAnalysis}><TokenTable tokens={res.tokens} t={t} /></Section>}

          <Section title={t.evidence}>
            <div className="space-y-5">
              <EvidenceList title={t.supporting} sources={res.supporting_sources} t={t} lang={lang} />
              <EvidenceList title={t.contradicting} sources={res.contradicting_sources} t={t} lang={lang} />
              {res.evidence_status.some((s) => !s.ok && s.error !== "disabled") && (
                <p className="text-xs text-slate-500">{t.unavailable}: {res.evidence_status.filter((s) => !s.ok && s.error !== "disabled").map((s) => `${s.provider} (${s.error})`).join(", ")}</p>)}
            </div>
          </Section>

          <Section title={t.related}>
            {all.length === 0 ? <p className="text-sm text-slate-500">{t.noEvidence}</p> :
              <div className="grid gap-3 md:grid-cols-2">{all.map((s) => <SourceCard key={`r-${s.url}`} s={s} t={t} lang={lang} />)}</div>}
          </Section>

          <Section title={t.tech}>
            <dl className="grid grid-cols-2 gap-2 text-sm md:grid-cols-3">
              {([[t.model, res.tech.model_name], [t.nli, res.tech.nli_model], [t.embedding, res.tech.embedding_model ?? "–"], ...(res.tech.translation_model ? [[t.translationModel, res.tech.translation_model] as [string, string]] : []), [t.version, res.tech.model_version], [t.cutoff, res.tech.training_cutoff],
                [t.language, res.tech.language], [t.tokens, String(res.tech.n_tokens)], [t.inference, `${res.tech.inference_ms} ms`],
                [t.total, `${res.tech.total_ms} ms`], [t.checked, new Date(res.tech.evidence_checked_at).toLocaleString()]] as [string, string][])
                .map(([k, v]) => <div key={k}><dt className="text-slate-500">{k}</dt><dd className="font-medium">{v}</dd></div>)}
            </dl>
            <h3 className="mt-4 font-semibold">{t.limitations}</h3>
            <ul className="list-disc pl-5 text-sm text-slate-600">{res.limitations.map((l, i) => <li key={i}>{l}</li>)}</ul>
          </Section>
        </div>
      )}
    </main>
  );
}
