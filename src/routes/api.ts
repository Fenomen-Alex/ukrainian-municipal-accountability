import type { FastifyInstance } from "fastify";
import { ComplaintInput } from "../lib/schema.js";
import { analyzeComplaint, InferenceError } from "../services/inference.js";

export async function registerRoutes(app: FastifyInstance) {
  app.post("/api/analyze", async (req, reply) => {
    const result = ComplaintInput.safeParse(req.body);
    if (!result.success) {
      return reply.status(400).send({
        error: "Помилка валідації",
        details: result.error.issues,
      });
    }
    try {
      const analysis = await analyzeComplaint(result.data);
      return reply.send({
        structured: analysis.structured,
        raw: analysis.raw,
      });
    } catch (err) {
      const message = err instanceof InferenceError ? err.message : String(err);
      const raw = err instanceof InferenceError ? err.raw : undefined;
      return reply.status(500).send({
        error: message,
        raw,
      });
    }
  });

  app.get("/health", async () => {
    return { status: "ok" };
  });
}
