import type { Dict } from "../i18n";

export default function HowItWorks({ t }: { t: Dict }) {
  return (
    <details className="mt-4 rounded-lg bg-slate-50 p-3 text-sm text-slate-700">
      <summary className="cursor-pointer font-medium">ℹ️ {t.howTitle}</summary>
      <ul className="mt-2 list-disc space-y-1 pl-5">{t.how.map((h, i) => <li key={i}>{h}</li>)}</ul>
    </details>
  );
}
