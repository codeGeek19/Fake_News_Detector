import type { Source } from "../types";
import type { Dict } from "../i18n";

const STANCE_CLS = { supports: "bg-green-100 text-green-800", contradicts: "bg-red-100 text-red-800", related: "bg-slate-100 text-slate-700" };

export function SourceCard({ s, t, lang }: { s: Source; t: Dict; lang: string }) {
  const date = s.published_at
    ? new Date(s.published_at).toLocaleDateString(({ hi: "hi-IN", mr: "mr-IN" } as Record<string, string>)[lang] ?? "en-IN", { year: "numeric", month: "short", day: "numeric" })
    : t.dateUnknown;
  return (
    <article data-testid="source-card" className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between text-sm text-slate-500">
        <span className="font-semibold text-slate-700">{s.publisher}</span><span>{date}</span>
      </div>
      <h4 className="mt-2 font-medium">{s.title}</h4>
      {s.snippet && s.snippet.trim() !== s.title.trim() && <p data-testid="snippet" className="mt-1 text-sm text-slate-600">{s.snippet.slice(0, 240)}</p>}
      {s.rating && <p className="mt-1 text-sm text-slate-600">Rating: {s.rating}</p>}
      <div className="mt-3 flex items-center justify-between">
        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${STANCE_CLS[s.stance]}`}>{t.stance[s.stance]}</span>
        <a href={s.url} target="_blank" rel="noopener noreferrer" className="text-sm font-medium text-indigo-600 hover:underline">{t.readOriginal}</a>
      </div>
    </article>
  );
}

export default function EvidenceList({ title, sources, t, lang }: { title: string; sources: Source[]; t: Dict; lang: string }) {
  return (
    <div>
      <h3 className="mb-2 font-semibold">{title}</h3>
      {sources.length === 0 ? <p className="text-sm text-slate-500">{t.noEvidence}</p> :
        <div className="grid gap-3 md:grid-cols-2">{sources.map((s) => <SourceCard key={s.url} s={s} t={t} lang={lang} />)}</div>}
    </div>
  );
}
