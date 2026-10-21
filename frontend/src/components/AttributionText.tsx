import type { Attribution } from "../types";
import type { Dict } from "../i18n";

export function colorFor(n: number): string {
  const a = Math.min(1, Math.abs(n)) * 0.75;
  if (Math.abs(n) < 0.08) return "transparent";
  return n > 0 ? `rgba(239,68,68,${a})` : `rgba(34,197,94,${a})`;
}

export default function AttributionText({ attributions, t }: { attributions: Attribution[]; t: Dict }) {
  return (
    <div>
      <p className="leading-9 text-base" data-testid="attribution-text">
        {attributions.map((a) => (
          <span key={a.position}>
            {a.word_start && a.position > 1 ? " " : ""}
            <span
              data-testid="attr-token"
              title={`${t.contribution}: ${a.score >= 0 ? "+" : ""}${a.score.toFixed(3)} (${a.score >= 0 ? t.towardFake : t.towardReal})`}
              style={{ backgroundColor: colorFor(a.normalized) }}
              className="rounded px-0.5"
            >{a.token}</span>
          </span>
        ))}
      </p>
      <div className="mt-3 flex flex-wrap gap-4 text-xs text-slate-600">
        <span>🔴 {t.legendFake}</span><span>🟢 {t.legendReal}</span><span>⚪ {t.legendLow}</span>
      </div>
    </div>
  );
}
