import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { analyzeComplaint, InferenceError } from "../../src/services/inference.js";

const INFER_URL = "https://uma-inference.example.run.app";
// marker present but no "Прошу ..." sentence, so the deterministic fallback cannot fill
const TEXT_WITH_REQUEST =
  "На вулиці Соборній не горять ліхтарі. Треба відновити освітлення.";
const TEXT_WITHOUT_REQUEST = "На вулиці Соборній не горять ліхтарі вже тиждень.";

const EMPTY_ACTIONS = {
  choices: [
    {
      message: {
        content:
          '[{"domain":"electricity","issue":"На вулиці Соборній не горять ліхтарі","object":"не вказано","requested_action":"","attributes":{}}]',
      },
    },
  ],
};
const FILLED_ACTION = {
  choices: [
    {
      message: {
        content:
          '[{"domain":"electricity","issue":"На вулиці Соборній не горять ліхтарі","object":"не вказано","requested_action":"відновити освітлення","attributes":{}}]',
      },
    },
  ],
};

function completionBodies(fetchMock: ReturnType<typeof vi.fn>): string[] {
  return fetchMock.mock.calls
    .filter((c) => String(c[0]).endsWith("/v1/chat/completions"))
    .map((c) => String((c[1] as RequestInit).body));
}

describe("requested_action retry", () => {
  const originalUrl = process.env.INFERENCE_SERVICE_URL;

  beforeEach(() => {
    process.env.INFERENCE_SERVICE_URL = INFER_URL;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    if (originalUrl === undefined) delete process.env.INFERENCE_SERVICE_URL;
    else process.env.INFERENCE_SERVICE_URL = originalUrl;
  });

  it("re-asks once with a reinforced prompt when no action was filled but the text has a request", async () => {
    let call = 0;
    const fetchMock = vi.fn(async () => {
      call++;
      return new Response(JSON.stringify(call === 1 ? EMPTY_ACTIONS : FILLED_ACTION), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const out = await analyzeComplaint({ text: TEXT_WITH_REQUEST });

    expect(out.structured.topics[0]!.requested_action).toBe("відновити освітлення");
    const bodies = completionBodies(fetchMock);
    expect(bodies).toHaveLength(2);
    expect(bodies[0]).not.toContain("НАГАДУВАННЯ");
    expect(bodies[1]).toContain("НАГАДУВАННЯ");
  });

  it("does not re-ask when the first response already fills requested_action", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify(FILLED_ACTION), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await analyzeComplaint({ text: TEXT_WITH_REQUEST });

    expect(completionBodies(fetchMock)).toHaveLength(1);
  });

  it("does not re-ask when the user text contains no request markers", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify(EMPTY_ACTIONS), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const out = await analyzeComplaint({ text: TEXT_WITHOUT_REQUEST });

    expect(out.structured.topics[0]!.requested_action).toBe("");
    expect(completionBodies(fetchMock)).toHaveLength(1);
  });

  it("still succeeds when the re-ask also fails", async () => {
    let call = 0;
    const fetchMock = vi.fn(async () => {
      call++;
      if (call === 2) throw new TypeError("fetch failed");
      return new Response(JSON.stringify(EMPTY_ACTIONS), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const out = await analyzeComplaint({ text: TEXT_WITH_REQUEST });

    expect(out.structured.topics).toHaveLength(1);
    expect(completionBodies(fetchMock)).toHaveLength(2);
  });

  it("propagates InferenceError when the first call itself fails", async () => {
    const fetchMock = vi.fn(
      async () => new Response("boom", { status: 500 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(analyzeComplaint({ text: TEXT_WITH_REQUEST })).rejects.toBeInstanceOf(
      InferenceError,
    );
    expect(completionBodies(fetchMock)).toHaveLength(1);
  });
});
