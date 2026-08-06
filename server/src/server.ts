/**
 * HTTP + WebSocket serving, kept apart from the table so the smoke run can
 * boot one in-process on an ephemeral port.
 *
 * One port carries everything: static files from web/dist when built, and the
 * game WebSocket at /ws. In development the Vite dev server proxies /ws here.
 */

import { createServer, type Server as HttpServer } from "node:http";
import type { AddressInfo } from "node:net";
import { existsSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { WebSocketServer, type WebSocket } from "ws";
import { loadEngine } from "@canastra/harness/node";
import { parseClientMessage } from "@canastra/protocol";
import { Table, type TableOptions } from "./table.js";
import { clearGame, loadGame, saveGame } from "./persistence.js";

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".map": "application/json",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

export interface ServerOptions extends TableOptions {
  port: number;
  /** Where the save file lives; null keeps the table in memory (the smoke run). */
  saveFile: string | null;
  /** The built web client; defaults to web/dist relative to this package. */
  distDir?: string;
}

export interface RunningServer {
  port(): number;
  close(): Promise<void>;
}

export async function startServer(options: ServerOptions): Promise<RunningServer> {
  await loadEngine();

  const distDir =
    options.distDir ?? fileURLToPath(new URL("../../web/dist", import.meta.url));

  const persist = (save: import("./persistence.js").SaveGame | null): void => {
    if (!options.saveFile) return;
    try {
      if (save) saveGame(options.saveFile, save);
      else clearGame(options.saveFile);
    } catch (error) {
      // The save is explicitly safe to lose — a disk failure must never take
      // the single server process down with it.
      console.error("save failed; continuing without it", error);
    }
  };

  // A save that no longer parses or fails the engine's invariant check costs
  // the match in progress, not the server.
  let table: Table;
  let resumed = false;
  const restored = options.saveFile ? loadGame(options.saveFile) : null;
  try {
    if (restored) {
      table = Table.restore(restored, { ...options, onChange: persist });
      resumed = true;
    } else {
      table = new Table({ ...options, onChange: persist });
    }
  } catch {
    table = new Table({ ...options, onChange: persist });
    resumed = false;
  }
  // Resume a restored match so its bots keep playing; a fresh lobby table is a no-op.
  if (resumed) table.resume();

  const http: HttpServer = createServer((req, res) => {
    if (!existsSync(distDir)) {
      res.writeHead(200, { "content-type": "text/plain; charset=utf-8" });
      res.end(
        "canastra server is up. The web client is not built — run `npm run build --prefix web`, or use the Vite dev server on :5173.\n",
      );
      return;
    }
    const pathname = (req.url ?? "/").split("?")[0];
    const file = normalize(join(distDir, pathname === "/" ? "index.html" : pathname));
    if (file !== distDir && !file.startsWith(distDir + sep)) {
      res.writeHead(403);
      res.end();
      return;
    }
    if (!existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404);
      res.end("not found");
      return;
    }
    res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
    res.end(readFileSync(file));
  });

  const wss = new WebSocketServer({ server: http, path: "/ws" });
  /** Dead-connection detection: miss two pongs (~20 s) and the close path runs. */
  const alive = new Map<WebSocket, boolean>();

  wss.on("connection", (ws) => {
    alive.set(ws, true);
    table.connect(ws);
    ws.on("pong", () => alive.set(ws, true));
    ws.on("message", (data) => {
      alive.set(ws, true);
      const message = parseClientMessage(String(data));
      if (message) table.handle(ws, message);
    });
    ws.on("close", () => {
      alive.delete(ws);
      table.disconnect(ws);
    });
  });

  const keepalive = setInterval(() => {
    for (const ws of wss.clients) {
      if (alive.get(ws) === false) {
        ws.terminate(); // 'close' fires, the seat gets covered
        continue;
      }
      alive.set(ws, false);
      ws.ping();
    }
  }, 10_000);

  await new Promise<void>((resolve) => http.listen(options.port, resolve));

  return {
    port: () => (http.address() as AddressInfo).port,
    close: () =>
      new Promise((resolve) => {
        clearInterval(keepalive);
        table.dispose(); // stop the bot/settle timers before the sockets die
        for (const ws of wss.clients) ws.terminate();
        wss.close();
        http.close(() => resolve());
      }),
  };
}
