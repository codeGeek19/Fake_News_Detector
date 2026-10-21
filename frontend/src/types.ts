export type Lang = "en" | "hi" | "mr";
export type Verdict = "likely_real" | "likely_fake" | "uncertain";
export type Stance = "supports" | "contradicts" | "related";

export interface Source {
  title: string; publisher: string; url: string; published_at: string | null; snippet: string | null;
  stance: Stance; stance_confidence: number; relevance: number; provenance: string; rating: string | null;
  source_type: string | null;
}
export interface TokenInfo { token: string; raw: string; id: number; position: number; special: boolean; word_start: boolean; attention: number | null }
export interface Attribution { token: string; position: number; word_start: boolean; score: number; normalized: number }
export interface ClaimResult {
  text: string; claim_type: "factual" | "opinion"; verdict: Verdict; reasons: string[];
  supporting_sources: Source[]; contradicting_sources: Source[]; related_articles: Source[];
}
export interface AnalyzeResponse {
  language: string; script: string; is_hinglish: boolean; language_mismatch: boolean;
  prediction: Verdict; model_prediction: Verdict; model_confidence: number; classifier_available: boolean; classifier_status: string; model_caveats: string[]; translated_text?: string | null;
  claims: ClaimResult[]; tokens: TokenInfo[]; attributions: Attribution[];
  explanation: { prediction: string; confidence: string; why_model: string; evidence: string; reasoning: string; summary?: string | null };
  supporting_sources: Source[]; contradicting_sources: Source[]; related_articles: Source[];
  evidence_status: { provider: string; ok: boolean; error: string | null }[];
  flags: string[]; limitations: string[];
  tech: { model_name: string; nli_model: string; embedding_model?: string; llm_model?: string | null; translation_model?: string | null; model_version: string; training_cutoff: string; language: string; n_tokens: number;
          inference_ms: number; total_ms: number; evidence_checked_at: string };
}
