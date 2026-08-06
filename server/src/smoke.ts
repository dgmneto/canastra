/**
 * The end-to-end smoke run — on demand, not a test framework:
 *
 *   npm run smoke
 *
 * Boots a real server in-process (ephemeral port, temp save file), connects
 * fake players over real WebSockets, and checks the promises the spec makes:
 * the lobby works, a match with bots runs to §14, a dropped player is covered
 * and can reclaim, and a restarted server resumes the table.
 *
 * The scripted humans reuse a real bot's candidate list against their own
 * `view`, exactly the way the harness driver does — so the check plays legal
 * Canastra without re-implementing any rules.
 */

import { mkdtempSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import WebSocket from "ws";
import { botById, makeRng } from "@canastra/bots";
import type { Action, PlayerView, Seat } from "@canastra/bots";
import type { ClientMessage, ServerMessage, TableState } from "@canastra/protocol";
import { startServer, type RunningServer } from "./server.js";

let failures = 0;

function ok(condition: boolean, what: string): void {
  console.log(`${condition ? "ok" : "NOT OK"} - ${what}`);
  if (!condition) failures++;
}

async function waitFor(
  condition: () => boolean,
  what: string,
  timeoutMs = 15_000,
): Promise<boolean> {
  const started = Date.now();
  while (!condition()) {
    if (Date.now() - started > timeoutMs) {
      ok(false, `${what} (timed out)`);
      return false;
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  ok(true, what);
  return true;
}

/** A scripted player: connects, and whenever its view says it must act, plays like the bot. */
class Player {
  view: PlayerView | null = null;
  table: TableState | null = null;
  seat: Seat | null = null;
  token: string | null = null;
  handOvers = 0;

  private ws: WebSocket | null = null;
  private rng = makeRng(42);
  private safeMode = false;
  private driving = false;
  private pending: ((accepted: boolean) => void) | null = null;
  private settledHand = -1;

  constructor(
    private name: string,
    private port: number,
    token?: string,
  ) {
    this.token = token ?? null;
  }

  connect(): Promise<void> {
    return new Promise((resolve) => {
      const ws = new WebSocket(`ws://localhost:${this.port}/ws`);
      this.ws = ws;
      ws.on("open", () => {
        this.send({ type: "hello", name: this.name, token: this.token ?? undefined });
        resolve();
      });
      ws.on("message", (data) => this.receive(JSON.parse(String(data)) as ServerMessage));
    });
  }

  terminate(): void {
    this.ws?.terminate(); // an unclean drop, like a dead network
  }

  send(message: ClientMessage): void {
    this.ws?.send(JSON.stringify(message));
  }

  sit(seat: Seat): void {
    this.send({ type: "sit", seat });
  }

  start(): void {
    this.send({ type: "start" });
  }

  private receive(message: ServerMessage): void {
    switch (message.type) {
      case "welcome":
        this.token = message.token;
        this.seat = message.seat;
        this.table = message.table;
        break;
      case "table":
        this.table = message.table;
        break;
      case "view": {
        this.view = message.view;
        this.pending?.(true);
        this.pending = null;
        void this.drive();
        break;
      }
      case "refused": {
        this.pending?.(false);
        this.pending = null;
        break;
      }
      case "handOver":
        this.handOvers++;
        break;
      case "event":
        break;
    }
  }

  /** Play our turn by proposing the bot's candidates until one is accepted. */
  private async drive(): Promise<void> {
    if (this.driving || !this.view || this.seat === null) return;
    this.driving = true;
    try {
      while (
        this.view &&
        this.view.turn === this.seat &&
        (this.view.phase === "AwaitingDraw" ||
          this.view.phase === "Melding" ||
          this.view.phase === "AwaitingRefusalChoice")
      ) {
        const bot = botById("random-plus");
        const candidates = bot.candidates(this.view, { rng: this.rng, safeMode: this.safeMode });
        let accepted: Action | null = null;
        for (const candidate of candidates) {
          if (await this.act(candidate)) {
            accepted = candidate;
            break;
          }
        }
        if (!accepted) {
          // Out of ideas — the same dead-end the driver's restart exists for.
          this.safeMode = true;
          this.send({ type: "restartTurn" });
          await new Promise((resolve) => setTimeout(resolve, 50));
        } else if (accepted.type === "Discard" || accepted.type === "EndTurnWithoutDiscard") {
          this.safeMode = false;
        }
      }
      if (this.view && this.view.phase === "HandOver" && this.settledHand !== this.view.hand_number) {
        this.settledHand = this.view.hand_number;
        this.send({ type: "settle" });
      }
    } finally {
      this.driving = false;
    }
  }

  /** The next message addressed to us — a view or a refusal — settles the attempt. */
  private act(action: Action): Promise<boolean> {
    return new Promise((resolve) => {
      this.pending = resolve;
      this.send({ type: "action", action });
    });
  }
}

async function main(): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), "canastra-smoke-"));
  const saveFile = join(dir, "game.json");
  let server: RunningServer | null = await startServer({
    port: 0,
    saveFile,
    botDelayMs: 0,
    settleDelayMs: 50,
  });

  try {
    const port = server.port();

    // --- lobby ---
    const ana = new Player("Ana", port);
    await ana.connect();
    ana.sit(0);
    await waitFor(() => ana.seat === 0 && ana.table?.seats[0].kind === "human", "Ana sits at seat 0");

    const bruno = new Player("Bruno", port);
    await bruno.connect();
    bruno.sit(1);
    await waitFor(() => bruno.seat === 1, "Bruno sits at seat 1");

    // --- the match runs ---
    ana.start();
    await waitFor(() => ana.view !== null && ana.view.hand.length === 15, "the deal reaches a seated player");

    // --- a dropped player is covered and reclaims ---
    await waitFor(() => ana.table !== null && ana.table.phase !== "lobby", "the match is running");
    bruno.terminate();
    await waitFor(
      () =>
        ana.table?.seats[1].kind === "human" &&
        ana.table.seats[1].connected === false,
      "a dropped seat is covered by a bot",
    );
    const brunoReturns = new Player("Bruno", port, bruno.token ?? undefined);
    await brunoReturns.connect();
    const reclaimed = await waitFor(
      () => brunoReturns.seat === 1 && brunoReturns.view !== null,
      "the token reclaims the seat mid-match",
    );
    let activeBruno: Player = reclaimed ? brunoReturns : bruno;
    void activeBruno;

    // --- a restarted server resumes the table ---
    await waitFor(() => existsSync(saveFile), "the save file exists");
    const scoresBefore = ana.table?.scores ?? null;
    await server.close();
    server = await startServer({ port: 0, saveFile, botDelayMs: 0, settleDelayMs: 50 });
    const port2 = server.port();
    const carla = new Player("Carla", port2);
    await carla.connect();
    await waitFor(
      () => carla.table !== null && carla.table.phase !== "lobby" && carla.table.phase !== undefined,
      "a restarted server resumes the match",
    );
    ok(
      JSON.stringify(carla.table?.scores) === JSON.stringify(scoresBefore),
      "the resumed table kept the scores",
    );
    const anaReturns = new Player("Ana", port2, ana.token ?? undefined);
    await anaReturns.connect();
    await waitFor(() => anaReturns.seat === 0 && anaReturns.view !== null, "Ana reclaims after the restart");

    // --- the match runs to §14 ---
    await waitFor(
      () => anaReturns.table?.phase === "MatchOver" || carla.table?.phase === "MatchOver",
      "the match runs to MatchOver",
      300_000,
    );
    const scores = anaReturns.table?.scores ?? carla.table?.scores;
    console.log(`final score: ${scores?.[0]} vs ${scores?.[1]}`);
    ok((anaReturns.handOvers + carla.handOvers) > 0, "at least one hand settlement was broadcast");
  } finally {
    await server?.close();
    rmSync(dir, { recursive: true, force: true });
  }

  if (failures) {
    console.error(`${failures} check(s) failed`);
    process.exit(1);
  }
  console.log("smoke: all good");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
