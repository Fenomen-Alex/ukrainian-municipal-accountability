import { StructuredComplaint } from "../lib/schema.js";
import type { ComplaintInput } from "../lib/schema.js";
import { normalizeComplaintInput } from "../lib/normalize.js";
import { spawnSync } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..", "..");

export interface InferenceResult {
  structured: StructuredComplaint;
  raw: string;
}

export class InferenceError extends Error {
  constructor(message: string, public readonly raw?: string) {
    super(message);
    this.name = "InferenceError";
  }
}

export async function analyzeComplaint(input: ComplaintInput): Promise<InferenceResult> {
  const normalized = normalizeComplaintInput(input);
  const modelPath = process.env.V2_MODEL_PATH || "ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused";
  const python = process.env.PYTHON_PATH || ".venv/bin/python";

  const result = spawnSync(
    python,
    [
      "-m",
      "ml.tune.serve_v2",
      "--model",
      modelPath,
      "--text",
      normalized.text,
    ],
    {
      cwd: ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 120000,
    },
  );

  if (result.error) {
    throw new InferenceError(`Inference failed: ${result.error.message}`);
  }

  if (result.status !== 0) {
    throw new InferenceError(
      `Inference process exited with code ${result.status}: ${result.stderr}`,
    );
  }

  const stdout = result.stdout || "";
  const lines = stdout.split("\n");
  let raw = "";
  let inJson = false;
  for (const line of lines) {
    if (line.trim().startsWith("{") && !inJson) {
      inJson = true;
      raw = line;
      continue;
    }
    if (inJson) {
      raw += "\n" + line;
    }
  }
  if (!raw.trim()) {
    throw new InferenceError("No JSON found in model output", stdout);
  }
  try {
    const parsed = JSON.parse(raw.trim());
    const structured = StructuredComplaint.parse(parsed);
    return { structured, raw: raw.trim() };
  } catch (err) {
    throw new InferenceError(
      `Failed to parse model response: ${err instanceof Error ? err.message : String(err)}`,
      stdout,
    );
  }
}
