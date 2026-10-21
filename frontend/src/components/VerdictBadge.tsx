import type { Verdict } from "../types";

const STYLE: Record<Verdict, { icon: string; cls: string }> = {
  likely_real: { icon: "🟢", cls: "bg-green-50 text-green-800 border-green-300" },
  likely_fake: { icon: "🔴", cls: "bg-red-50 text-red-800 border-red-300" },
  uncertain: { icon: "🟡", cls: "bg-amber-50 text-amber-800 border-amber-300" },
};

export default function VerdictBadge({ verdict, label, size = "md" }: { verdict: Verdict; label: string; size?: "sm" | "md" }) {
  const s = STYLE[verdict];
  return (
    <span data-testid={`verdict-${verdict}`} className={`inline-flex items-center gap-2 rounded-full border font-semibold ${s.cls} ${size === "md" ? "px-4 py-2 text-lg" : "px-2 py-0.5 text-xs"}`}>
      <span aria-hidden>{s.icon}</span>{label}
    </span>
  );
}
