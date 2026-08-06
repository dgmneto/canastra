/** The production entry: one table, one port, a save file under server/data/. */

import { fileURLToPath } from "node:url";
import { startServer } from "./server.js";

const port = Number(process.env.PORT ?? 3001);
const saveFile = fileURLToPath(new URL("../data/game.json", import.meta.url));

startServer({ port, saveFile }).then((server) => {
  console.log(`canastra server on http://localhost:${server.port()}`);
});
