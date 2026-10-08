import type { FastifyInstance, FastifyRequest } from "fastify";
import { ComplaintInput } from "../lib/schema.js";
import { analyzeComplaint, InferenceError } from "../services/inference.js";
import { SlidingWindowLimiter } from "../lib/rate-limit.js";
import type { BuildOptions } from "../app.js";

export async function registerRoutes(app: FastifyInstance, opts: BuildOptions = {}) {
  const rlOpts = opts.rateLimit === undefined ? { windowMs: 60000, max: 10 } : opts.rateLimit;
  const limiter = rlOpts === false ? null : new SlidingWindowLimiter(rlOpts.windowMs, rlOpts.max);

  app.addHook("onRequest", async (req, reply) => {
    if (!limiter || req.method !== "POST" || req.url !== "/api/analyze") return;
    const r = limiter.check(req.ip);
    if (!r.allowed) {
      reply.header("Retry-After", String(r.retryAfterSec));
      return reply
        .status(429)
        .send({ error: "Забагато запитів. Зачекайте хвилину та спробуйте ще раз." });
    }
  });

  app.post("/api/analyze", async (req: FastifyRequest, reply) => {
    const result = ComplaintInput.safeParse(req.body);
    if (!result.success) {
      return reply.status(400).send({
        error: "Помилка валідації",
        details: result.error.issues.map((i) => ({
          path: i.path.join("."),
          message: i.message,
        })),
      });
    }
    try {
      const analysis = await analyzeComplaint(result.data);
      return reply.send({ structured: analysis.structured });
    } catch (err) {
      const isInference = err instanceof InferenceError;
      console.error(`[analyze] ${err instanceof Error ? err.name : "Error"}`);
      return reply.status(isInference ? 502 : 500).send({
        error: isInference
          ? "Не вдалося обробити запит. Спробуйте пізніше."
          : "Внутрішня помилка. Спробуйте пізніше.",
      });
    }
  });

  app.get("/health", async () => ({ status: "ok" }));
}
