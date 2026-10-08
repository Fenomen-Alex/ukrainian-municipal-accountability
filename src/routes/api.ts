import type { FastifyInstance, FastifyRequest } from "fastify";
import { ComplaintInput } from "../lib/schema.js";
import { analyzeComplaint, InferenceError } from "../services/inference.js";
import type { BuildOptions } from "../app.js";

export async function registerRoutes(app: FastifyInstance, opts: BuildOptions = {}) {
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

  // `opts` is consumed by Task 3 (rate limiting); see registerRoutes signature.
  void opts;
}
