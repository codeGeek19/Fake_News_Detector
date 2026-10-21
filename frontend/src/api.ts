import type { AnalyzeResponse, Lang } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export async function analyze(text: string, language: Lang): Promise<AnalyzeResponse> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/analyze`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, language }),
    });
  } catch {
    throw new Error("NETWORK");
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { const j = await res.json(); if (typeof j.detail === "string") detail = j.detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}
