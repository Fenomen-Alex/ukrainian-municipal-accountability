import Fastify from "fastify";
import type { FastifyInstance } from "fastify";
import fastifyStatic from "@fastify/static";
import { readFile } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { registerRoutes } from "./routes/api.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const SECURITY_HEADERS: Record<string, string> = {
  "Content-Security-Policy":
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Resource-Policy": "same-origin",
};

export interface BuildOptions {
  rateLimit?: false | { windowMs: number; max: number };
}

export async function buildApp(opts: BuildOptions = {}): Promise<FastifyInstance> {
  const app = Fastify({
    logger: false,
    trustProxy: true,
    bodyLimit: 65536,
  });

  app.addHook("onSend", async (req, reply, payload) => {
    for (const [k, v] of Object.entries(SECURITY_HEADERS)) reply.header(k, v);
    if (req.protocol === "https") {
      reply.header("Strict-Transport-Security", "max-age=31536000");
    }
    return payload;
  });

  app.setErrorHandler((err: unknown, _req, reply) => {
    const e = err as { statusCode?: number; name?: string; message?: string };
    const status = e.statusCode && e.statusCode >= 400 ? e.statusCode : 500;
    if (status >= 500) {
      console.error(`[error] ${e.name ?? "Error"}: ${e.message ?? "unknown"}`);
      return reply.status(status).send({ error: "Внутрішня помилка. Спробуйте пізніше." });
    }
    return reply.status(status).send({ error: e.message ?? "Помилка запиту" });
  });

  await app.register(fastifyStatic, {
    root: path.resolve(__dirname, "..", "public"),
    prefix: "/",
    index: false,
  });

  await registerRoutes(app, opts);

  app.get("/", async (_req, reply) => {
    const html = await readFile(path.resolve(__dirname, "..", "public", "index.html"), "utf8");
    return reply.type("text/html; charset=utf-8").send(html);
  });

  app.get("/app", async (_req, reply) => {
    const html = await readFile(path.resolve(__dirname, "..", "public", "app.html"), "utf8");
    return reply.type("text/html; charset=utf-8").send(html);
  });

  return app;
}
