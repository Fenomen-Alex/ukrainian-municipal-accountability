import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  analyzeComplaint,
  clearIdentityTokenCache,
  fetchIdentityToken,
  InferenceError,
} from "../../src/services/inference.js";

const INFER_URL = "https://uma-inference.example.run.app";
const METADATA_PREFIX = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity";
const CONTENT =
  '[{"domain":"roads","issue":"яма","object":"вул. Перемоги","requested_action":"ремонт","attributes":{}}]';
const COMPLETION_RESPONSE = { choices: [{ message: { content: CONTENT } }] };

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function headersOf(call: unknown[] | undefined): Record<string, string> {
  if (!call) return {};
  return ((call[1] as RequestInit | undefined)?.headers ?? {}) as Record<string, string>;
}

describe("inference auth (INFERENCE_AUTH=metadata)", () => {
  const originalEnv: Record<string, string | undefined> = {};

  beforeEach(() => {
    originalEnv.INFERENCE_SERVICE_URL = process.env.INFERENCE_SERVICE_URL;
    originalEnv.INFERENCE_AUTH = process.env.INFERENCE_AUTH;
    process.env.INFERENCE_SERVICE_URL = INFER_URL;
    clearIdentityTokenCache();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    for (const [k, v] of Object.entries(originalEnv)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
    clearIdentityTokenCache();
  });

  it("sends Authorization: Bearer <identity token> to the completion endpoint", async () => {
    process.env.INFERENCE_AUTH = "metadata";
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith(METADATA_PREFIX)) {
        return new Response("token-abc123", { status: 200 });
      }
      if (url === `${INFER_URL}/v1/chat/completions`) {
        return jsonResponse(COMPLETION_RESPONSE);
      }
      return new Response("unexpected", { status: 500 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const out = await analyzeComplaint({ text: "На вулиці Перемоги яма, прошу відремонтувати" });

    expect(out.structured.topics[0]!.domain).toBe("roads");
    const completionCall = fetchMock.mock.calls.find(
      (c) => String(c[0]) === `${INFER_URL}/v1/chat/completions`,
    );
    expect(completionCall).toBeTruthy();
    expect(headersOf(completionCall).Authorization).toBe("Bearer token-abc123");
    const metadataCall = fetchMock.mock.calls.find((c) => String(c[0]).startsWith(METADATA_PREFIX));
    expect(metadataCall).toBeTruthy();
    expect(headersOf(metadataCall)).toMatchObject({ "Metadata-Flavor": "Google" });
  });

  it("caches the identity token across analyze calls", async () => {
    process.env.INFERENCE_AUTH = "metadata";
    let metadataFetches = 0;
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith(METADATA_PREFIX)) {
        metadataFetches++;
        return new Response(`token-${metadataFetches}`, { status: 200 });
      }
      return jsonResponse(COMPLETION_RESPONSE);
    });
    vi.stubGlobal("fetch", fetchMock);

    await analyzeComplaint({ text: "перше звернення про ямy на дорозі" });
    await analyzeComplaint({ text: "друге звернення про витік води" });

    expect(metadataFetches).toBe(1);
    expect(fetchMock.mock.calls.filter((c) => String(c[0]).startsWith(METADATA_PREFIX))).toHaveLength(1);
  });

  it("fails closed with InferenceError when the metadata server is unreachable", async () => {
    process.env.INFERENCE_AUTH = "metadata";
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith(METADATA_PREFIX)) {
        throw new TypeError("fetch failed");
      }
      return jsonResponse(COMPLETION_RESPONSE);
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(analyzeComplaint({ text: "яма на дорозі" })).rejects.toBeInstanceOf(InferenceError);
    expect(fetchMock.mock.calls.some((c) => String(c[0]).endsWith("/v1/chat/completions"))).toBe(false);
  });

  it("fails closed when the metadata server returns a non-OK status", async () => {
    process.env.INFERENCE_AUTH = "metadata";
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith(METADATA_PREFIX)) {
        return new Response("nope", { status: 403 });
      }
      return jsonResponse(COMPLETION_RESPONSE);
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(analyzeComplaint({ text: "яма на дорозі" })).rejects.toBeInstanceOf(InferenceError);
  });

  it("does not send Authorization when auth is not configured", async () => {
    delete process.env.INFERENCE_AUTH;
    const fetchMock = vi.fn(async () => jsonResponse(COMPLETION_RESPONSE));
    vi.stubGlobal("fetch", fetchMock);

    await analyzeComplaint({ text: "яма на дорозі" });

    const completionCall = fetchMock.mock.calls[0];
    expect(headersOf(completionCall).Authorization).toBeUndefined();
  });

  it("fetchIdentityToken rejects on an empty token body", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("   ", { status: 200 })));
    await expect(fetchIdentityToken(INFER_URL)).rejects.toBeInstanceOf(InferenceError);
  });
});
