import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, vi } from "vitest";
import App from "./App";
import type { AnalyzeResponse } from "./types";

const src = (o: object) => ({ title: "UPI services working normally", publisher: "The Hindu", url: "https://example.org/a",
  published_at: "2026-10-01T00:00:00Z", snippet: "Officials said UPI services are running normally across the country.", stance: "contradicts", stance_confidence: 0.9, relevance: 0.8,
  provenance: "gdelt", rating: null, source_type: "newspaper", ...o });

const RES: AnalyzeResponse = {
  language: "en", script: "latin", is_hinglish: false, language_mismatch: false, prediction: "uncertain",
  model_prediction: "likely_fake", model_confidence: 0.78, classifier_available: true, classifier_status: "native", model_caveats: [],
  claims: [{ text: "UPI is banned.", claim_type: "factual", verdict: "uncertain", reasons: ["Not enough sources."], supporting_sources: [], contradicting_sources: [src({}) as never], related_articles: [] }],
  tokens: [
    { token: "[CLS]", raw: "[CLS]", id: 101, position: 0, special: true, word_start: false, attention: null },
    { token: "mis", raw: "mis", id: 5, position: 1, special: false, word_start: true, attention: null },
    { token: "information", raw: "##information", id: 6, position: 2, special: false, word_start: false, attention: null }],
  attributions: [
    { token: "mis", position: 1, word_start: true, score: 0.17, normalized: 1 },
    { token: "information", position: 2, word_start: false, score: -0.05, normalized: -0.3 }],
  explanation: { prediction: "Uncertain", confidence: "78%", why_model: "The classifier weighted “mis”.", evidence: "Found 1 related article.", reasoning: "" },
  supporting_sources: [], contradicting_sources: [src({}) as never], related_articles: [],
  evidence_status: [{ provider: "google_factcheck", ok: false, error: "API key not configured" }],
  flags: ["unverified"], limitations: ["Limit one."],
  tech: { model_name: "roberta-base", nli_model: "mdeberta-nli", model_version: "v1.0", training_cutoff: "2017-12-31", language: "en", n_tokens: 3,
          inference_ms: 12, total_ms: 900, evidence_checked_at: "2026-10-02T10:00:00Z" },
};

const mockFetch = (ok: boolean, body: unknown, status = 200) =>
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok, status, json: async () => body }));

afterEach(() => vi.unstubAllGlobals());

async function submit(text = "UPI has been banned from October.") {
  await userEvent.type(screen.getByRole("textbox"), text);
  await userEvent.click(screen.getByRole("button", { name: /analyze news|समाचार का विश्लेषण करें/i }));
}

test("language toggle localises the UI", async () => {
  render(<App />);
  expect(screen.getByRole("button", { name: "Analyze News" })).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /हिंदी/ }));
  expect(screen.getByRole("button", { name: "समाचार का विश्लेषण करें" })).toBeInTheDocument();
});

test("analyze button disabled until enough text", async () => {
  render(<App />);
  expect(screen.getByRole("button", { name: "Analyze News" })).toBeDisabled();
});

test("shows loading then result, with evidence card linking to the source", async () => {
  mockFetch(true, RES);
  render(<App />);
  await submit();
  await waitFor(() => expect(screen.getAllByTestId("verdict-uncertain")[0]).toBeInTheDocument());
  expect(screen.getByTestId("verdict-likely_fake")).toBeInTheDocument();      // model estimate kept separate
  const link = screen.getAllByRole("link", { name: /Read Original/ })[0];
  expect(link).toHaveAttribute("href", "https://example.org/a");
  expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
  expect(screen.getAllByTestId("source-card").length).toBeGreaterThan(0);
});

test("token visualisation shows sub-word pieces and highlights", async () => {
  mockFetch(true, RES);
  render(<App />);
  await submit();
  await waitFor(() => screen.getByTestId("token-chips"));
  expect(screen.getByRole("button", { name: "[CLS]" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "mis" })).toBeInTheDocument();
  const t = screen.getAllByTestId("attr-token")[0];
  expect(t.getAttribute("title")).toContain("+0.170");
  expect(t.style.backgroundColor).toContain("239");                           // red = toward fake
});

test("shows error state when the backend reports a missing model", async () => {
  mockFetch(false, { detail: "Model not installed: MODEL_EN_PATH is not set" }, 503);
  render(<App />);
  await submit();
  expect(await screen.findByRole("alert")).toHaveTextContent(/Model not installed/);
});

test("shows network error state", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fail")));
  render(<App />);
  await submit();
  expect(await screen.findByRole("alert")).toHaveTextContent(/Cannot reach the server/);
});

test("evidence-only mode: no big banner, a compact reason, and no fake model take", async () => {
  mockFetch(true, { ...RES, classifier_available: false, classifier_status: "unavailable_translator", model_confidence: 0, attributions: [], model_prediction: "uncertain" });
  render(<App />);
  await submit();
  const el = await screen.findByTestId("take-unavailable");
  expect(el).toHaveTextContent(/translation model could not be loaded/);
  expect(screen.queryByText(/No writing-pattern classifier/)).not.toBeInTheDocument();
  expect(screen.queryByTestId("attribution-text")).not.toBeInTheDocument();
});

test("the model's take is shown even when the final result is Uncertain", async () => {
  mockFetch(true, { ...RES, prediction: "uncertain", model_prediction: "likely_real", model_confidence: 0.72, model_caveats: ["short_text", "translated"] });
  render(<App />);
  await submit();
  const take = await screen.findByTestId("model-take");
  expect(take).toHaveTextContent("Model's take");
  expect(take).toHaveTextContent("Likely Real");
  expect(take).toHaveTextContent("72%");
  expect(screen.getAllByTestId("caveat").map((c) => c.textContent)).toEqual(["Short text: less reliable", "Read a machine translation"]);
  expect(screen.getAllByTestId("verdict-uncertain")[0]).toBeInTheDocument();       // final result stays Uncertain
});

test("Marathi is available and localises the UI", async () => {
  render(<App />);
  await userEvent.click(screen.getByRole("button", { name: /मराठी/ }));
  expect(screen.getByRole("button", { name: "बातमीचे विश्लेषण करा" })).toBeInTheDocument();
});

test("evidence card shows the snippet", async () => {
  mockFetch(true, RES);
  render(<App />);
  await submit();
  expect((await screen.findAllByTestId("snippet"))[0]).toHaveTextContent(/running normally/);
});

test("shows the optional LLM summary with its caveat only when present", async () => {
  mockFetch(true, { ...RES, explanation: { ...RES.explanation, summary: "No source confirms the claim." } });
  render(<App />);
  await submit();
  expect(await screen.findByTestId("llm-summary")).toHaveTextContent(/No source confirms/);
  expect(screen.getByTestId("llm-summary")).toHaveTextContent(/never decides the result/);
});

test("shows the English translation the classifier read (Hindi/Marathi)", async () => {
  mockFetch(true, { ...RES, language: "hi", translated_text: "The government has banned all UPI payments.", flags: ["classified_on_translation"],
                    tech: { ...RES.tech, translation_model: "nllb-test" } });
  render(<App />);
  await submit();
  expect(await screen.findByTestId("translation")).toHaveTextContent("banned all UPI payments");
  expect(screen.getByText(/Classifier read a machine translation/)).toBeInTheDocument();
  expect(screen.getByText("nllb-test")).toBeInTheDocument();
});
