import Fastify from "fastify";
import cors from "@fastify/cors";
import multipart from "@fastify/multipart";
import { readFile } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { registerRoutes } from "./routes/api.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = Fastify({
  logger: false,
});

await app.register(cors, { origin: true });
await app.register(multipart);

await registerRoutes(app);

app.get("/", async (req, reply) => {
  const publicPath = path.resolve(__dirname, "..", "public", "index.html");
  const html = await readFile(publicPath, "utf8");
  return reply.type("text/html").send(html);
});

app.get("/app", async (req, reply) => {
  const publicPath = path.resolve(__dirname, "..", "public", "app.html");
  const html = await readFile(publicPath, "utf8");
  return reply.type("text/html").send(html);
});

const port = Number(process.env.PORT) || 3000;
const host = process.env.HOST || "localhost";

app.listen({ port, host }, (err, address) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log(`Server listening at ${address}`);
});
