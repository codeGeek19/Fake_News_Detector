import { useState } from "react";
import type { TokenInfo } from "../types";
import type { Dict } from "../i18n";

/** Words are shown as groups of sub-word pieces; click a piece to inspect it. */
export default function TokenTable({ tokens, t }: { tokens: TokenInfo[]; t: Dict }) {
  const [sel, setSel] = useState<number | null>(null);
  const groups: TokenInfo[][] = [];
  tokens.forEach((tk) => {
    if (tk.special || tk.word_start || groups.length === 0 || groups[groups.length - 1][0].special) groups.push([tk]);
    else groups[groups.length - 1].push(tk);
  });
  const hasAtt = tokens.some((x) => x.attention !== null);
  return (
    <div>
      <div className="flex flex-wrap gap-2" data-testid="token-chips">
        {groups.map((g, gi) => (
          <span key={gi} className="inline-flex overflow-hidden rounded-md border border-slate-300">
            {g.map((tk) => (
              <button key={tk.position} type="button" onClick={() => setSel(tk.position)}
                className={`px-2 py-1 text-sm ${tk.special ? "bg-slate-200 text-slate-500" : "bg-white"} ${sel === tk.position ? "ring-2 ring-indigo-500" : ""} hover:bg-indigo-50`}>
                {tk.token}
              </button>
            ))}
          </span>
        ))}
      </div>
      <div className="mt-4 max-h-72 overflow-auto rounded-lg border border-slate-200">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-slate-100"><tr>
            <th className="px-3 py-2">{t.token}</th><th className="px-3 py-2">{t.tokenId}</th><th className="px-3 py-2">{t.position}</th>
            {hasAtt && <th className="px-3 py-2">{t.attention}</th>}
          </tr></thead>
          <tbody>
            {tokens.map((tk) => (
              <tr key={tk.position} onClick={() => setSel(tk.position)} className={`cursor-pointer border-t ${sel === tk.position ? "bg-indigo-50" : ""}`}>
                <td className="px-3 py-1 font-mono">{tk.raw}</td><td className="px-3 py-1">{tk.id}</td><td className="px-3 py-1">{tk.position}</td>
                {hasAtt && <td className="px-3 py-1">{tk.attention ?? "–"}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
