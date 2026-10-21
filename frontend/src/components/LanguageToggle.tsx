import type { Lang } from "../types";

export default function LanguageToggle({ value, onChange }: { value: Lang; onChange: (l: Lang) => void }) {
  const base = "px-4 py-2 text-sm font-medium transition";
  const on = "bg-indigo-600 text-white", off = "bg-white text-slate-700 hover:bg-slate-100";
  const opts: [Lang, string][] = [["en", "English 🇬🇧"], ["hi", "हिंदी 🇮🇳"], ["mr", "मराठी 🇮🇳"]];
  return (
    <div role="group" aria-label="Language" className="inline-flex overflow-hidden rounded-full border border-slate-300">
      {opts.map(([code, label]) => (
        <button key={code} type="button" aria-pressed={value === code} onClick={() => onChange(code)}
          className={`${base} ${value === code ? on : off}`}>{label}</button>
      ))}
    </div>
  );
}
