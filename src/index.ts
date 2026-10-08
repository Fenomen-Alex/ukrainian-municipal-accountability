import { buildApp } from "./app.js";

const app = await buildApp();
const port = Number(process.env.PORT) || 3000;
const host = process.env.HOST || "localhost";

app.listen({ port, host }, (err, address) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log(`Server listening at ${address}`);
});
