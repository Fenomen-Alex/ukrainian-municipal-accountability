import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import type { FastifyInstance } from "fastify";
import { buildApp } from "../src/app.js";

let app: FastifyInstance;

beforeEach(async () => {
  app = await buildApp({ rateLimit: false });
  delete process.env.INFERENCE_SERVICE_URL;
});

afterEach(async () => {
  await app.close();
  vi.unstubAllGlobals();
  delete process.env.INFERENCE_SERVICE_URL;
});

const SECURITY_HEADERS = [
  "content-security-policy",
  "x-content-type-options",
  "x-frame-options",
  "referrer-policy",
  "permissions-policy",
];

describe("security headers", () => {
  it("sets headers on HTML responses", async () => {
    const res = await app.inject({ method: "GET", url: "/" });
    expect(res.statusCode).toBe(200);
    for (const h of SECURITY_HEADERS) expect(res.headers[h]).toBeTruthy();
    expect(res.headers["content-security-policy"]).toContain("default-src 'self'");
    expect(res.headers["content-security-policy"]).toContain("frame-ancestors 'none'");
  });

  it("sets headers on API responses", async () => {
    const res = await app.inject({ method: "GET", url: "/health" });
    expect(res.headers["x-content-type-options"]).toBe("nosniff");
  });
});

describe("static assets", () => {
  it("serves /app.js", async () => {
    const res = await app.inject({ method: "GET", url: "/app.js" });
    expect(res.statusCode).toBe(200);
    expect(res.headers["content-type"]).toContain("javascript");
  });

  it("serves /styles.css", async () => {
    const res = await app.inject({ method: "GET", url: "/styles.css" });
    expect(res.statusCode).toBe(200);
    expect(res.headers["content-type"]).toContain("css");
  });

  it("404s unknown paths as JSON without stack", async () => {
    const res = await app.inject({ method: "GET", url: "/nope" });
    expect(res.statusCode).toBe(404);
    expect(res.headers["content-type"]).toContain("application/json");
    expect(res.body).not.toContain("stack");
    expect(res.body).not.toContain("    at ");
  });

  it("serves /app page", async () => {
    const res = await app.inject({ method: "GET", url: "/app" });
    expect(res.statusCode).toBe(200);
    expect(res.body).toContain("Формування звернення");
  });
});

describe("POST /api/analyze", () => {
  it("400 on invalid body with mapped details, no echo of input", async () => {
    const res = await app.inject({ method: "POST", url: "/api/analyze", payload: { text: "" } });
    expect(res.statusCode).toBe(400);
    const body = res.json();
    expect(body.error).toBe("Помилка валідації");
    expect(Array.isArray(body.details)).toBe(true);
    expect(body.details[0]).toHaveProperty("path");
    expect(body.details[0]).toHaveProperty("message");
    expect(res.body).not.toContain("stack");
  });

  it("413 on oversized body with generic message", async () => {
    const res = await app.inject({
      method: "POST",
      url: "/api/analyze",
      payload: { text: "x".repeat(70000) },
    });
    expect(res.statusCode).toBe(413);
    expect(res.json()).toHaveProperty("error");
    expect(res.body).not.toContain("    at ");
  });

  it("400 on malformed JSON without stack", async () => {
    const res = await app.inject({
      method: "POST",
      url: "/api/analyze",
      headers: { "content-type": "application/json" },
      payload: "{invalid",
    });
    expect(res.statusCode).toBe(400);
    expect(res.json()).toHaveProperty("error");
    expect(res.body).not.toContain("    at ");
  });

  it("returns structured only on success (no raw)", async () => {
    process.env.INFERENCE_SERVICE_URL = "http://inference.test";
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              choices: [
                {
                  message: {
                    content: JSON.stringify([
                      {
                        domain: "roads",
                        issue: "Не горять ліхтарі",
                        object: "вул. Шевченка",
                        requested_action: "Відновити освітлення",
                        attributes: {},
                      },
                    ]),
                  },
                },
              ],
            }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
      ),
    );
    const res = await app.inject({
      method: "POST",
      url: "/api/analyze",
      payload: { text: "На вулиці Шевченка не горять ліхтарі" },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.structured.topics).toHaveLength(1);
    expect(body.structured.topics[0].domain).toBe("roads");
    expect(body).not.toHaveProperty("raw");
  });

  it("502 on upstream failure without leaking internals", async () => {
    process.env.INFERENCE_SERVICE_URL = "http://inference.test";
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response("upstream internal secret http://10.0.0.5:8080", { status: 500 }),
      ),
    );
    const res = await app.inject({
      method: "POST",
      url: "/api/analyze",
      payload: { text: "Проблема" },
    });
    expect(res.statusCode).toBe(502);
    expect(res.json().error).toBe("Не вдалося обробити запит. Спробуйте пізніше.");
    expect(res.body).not.toContain("10.0.0.5");
    expect(res.body).not.toContain("raw");
  });

  it("502 on unreachable inference without leaking internals", async () => {
    process.env.INFERENCE_SERVICE_URL = "http://127.0.0.1:1";
    const res = await app.inject({
      method: "POST",
      url: "/api/analyze",
      payload: { text: "Проблема" },
    });
    expect(res.statusCode).toBe(502);
    const body = res.json();
    expect(body.error).toBe("Не вдалося обробити запит. Спробуйте пізніше.");
    expect(res.body).not.toContain("ECONNREFUSED");
    expect(res.body).not.toContain("fetch failed");
  });
});

describe("rate limiting", () => {
  it("429 after max requests per IP", async () => {
    const limited = await buildApp({ rateLimit: { windowMs: 60000, max: 3 } });
    let last;
    for (let i = 0; i < 4; i++) {
      last = await limited.inject({
        method: "POST",
        url: "/api/analyze",
        payload: { text: "" },
        remoteAddress: "9.9.9.9",
      });
    }
    expect(last!.statusCode).toBe(429);
    expect(last!.json().error).toContain("Забагато запитів");
    expect(last!.headers["retry-after"]).toBeTruthy();
    await limited.close();
  });

  it("does not rate-limit other routes", async () => {
    const limited = await buildApp({ rateLimit: { windowMs: 60000, max: 1 } });
    for (let i = 0; i < 5; i++) {
      const res = await limited.inject({ method: "GET", url: "/health" });
      expect(res.statusCode).toBe(200);
    }
    await limited.close();
  });
});
