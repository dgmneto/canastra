/**
 * The one global table.
 *
 * Everything session-like lives here: who is connected, who sits where, the
 * match in progress, and the bots covering empty or abandoned seats. The
 * engine stays behind `Match` — this class never adjudicates a rule, it only
 * routes messages and drives bots.
 *
 * One process, one table: players find the game by connecting, not by picking
 * a room (the "single global table" decision in the spec).
 */

import { randomUUID } from "node:crypto";
import type { WebSocket } from "ws";
import { Match } from "@canastra/harness";
import type { LogLine } from "@canastra/harness";
import { makeRng } from "@canastra/bots";
import type { Action, Rng, Seat } from "@canastra/bots";
import type { ClientMessage, SeatOccupant, ServerMessage, TableState } from "@canastra/protocol";
import type { SaveGame, SaveSeat } from "./persistence.js";

/** The policy that covers a seat with no human deciding for it. */
const COVER_BOT = "random-plus";

/**
 * A human seat has a live socket; a covered seat is a human's seat being
 * played by a bot until they reclaim it with their token (the name and token
 * survive; the socket does not).
 */
type SeatState =
  | { kind: "empty" }
  | { kind: "human"; name: string; token: string; ws: WebSocket }
  | { kind: "covered"; name: string; token: string }
  | { kind: "bot"; botId: string };

interface Client {
  ws: WebSocket;
  name: string;
  token: string;
  seat: Seat | null;
}

export interface TableOptions {
  /** Pacing between bot actions, so humans can follow. 0 in the smoke run. */
  botDelayMs?: number;
  /** The pause at HandOver before the hand is banked. */
  settleDelayMs?: number;
  /** Persistence hook, called after every change worth saving. */
  onChange?: (save: SaveGame | null) => void;
}

export class Table {
  private clients = new Map<WebSocket, Client>();
  private seats: SeatState[] = [
    { kind: "empty" },
    { kind: "empty" },
    { kind: "empty" },
    { kind: "empty" },
  ];
  private match: Match | null = null;
  private rng: Rng = makeRng(1);
  /** §6 backstop bookkeeping — the same rule the harness CLI follows. */
  private safeMode = false;
  private botTimer: ReturnType<typeof setTimeout> | null = null;
  private settleTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private options: TableOptions = {}) {
    // Task 4's pump() consumes these; name them here so noUnusedLocals stays
    // quiet while the scaffold is lobby-only.
    void this.rng;
    void this.safeMode;
    void this.botTimer;
    void this.settleTimer;
  }

  /** Rebuild a table from a save. Human seats come back covered: their owners reclaim by token. */
  static restore(save: SaveGame, options: TableOptions = {}): Table {
    const table = new Table(options);
    table.seats = save.seats.map((seat): SeatState => {
      if (seat.kind === "human") return { kind: "covered", name: seat.name, token: seat.token };
      if (seat.kind === "bot") return { kind: "bot", botId: seat.botId };
      return { kind: "empty" };
    });
    if (save.match) {
      table.match = Match.restore(save.match.snapshot, {
        seed: BigInt(save.match.seed),
        bots: save.match.lineup,
        startedAt: save.match.startedAt,
        log: save.match.log as LogLine[],
      });
      table.rng = makeRng(Number(BigInt(save.match.seed) % 2147483647n) || 1);
    }
    return table;
  }

  /** What gets written to disk after every change. Null match → the lobby. */
  serialize(): SaveGame {
    const seats: SaveSeat[] = this.seats.map((seat): SaveSeat => {
      if (seat.kind === "human" || seat.kind === "covered")
        return { kind: "human", name: seat.name, token: seat.token };
      if (seat.kind === "bot") return { kind: "bot", botId: seat.botId };
      return { kind: "empty" };
    });
    if (!this.match) return { seats, match: null };
    const header = this.match.log[0] as { startedAt: string; bots: string[] };
    return {
      seats,
      match: {
        seed: this.match.seed.toString(),
        startedAt: header.startedAt,
        lineup: header.bots,
        snapshot: this.match.snapshot(),
        log: this.match.log,
      },
    };
  }

  // --- connection management (called by server.ts) ---

  connect(ws: WebSocket): void {
    this.clients.set(ws, { ws, name: "", token: "", seat: null });
  }

  disconnect(ws: WebSocket): void {
    const client = this.clients.get(ws);
    this.clients.delete(ws);
    if (client?.seat != null) {
      this.coverSeat(client.seat);
      this.afterChange();
    }
  }

  handle(ws: WebSocket, message: ClientMessage): void {
    const client = this.clients.get(ws);
    if (!client) return;
    switch (message.type) {
      case "hello":
        return this.hello(client, message.name, message.token);
      case "sit":
        return this.sit(client, message.seat);
      case "stand":
        return this.stand(client);
      case "start":
        return this.start(client);
      case "action":
        return this.act(client, message.action);
      case "restartTurn":
        return this.restart(client);
      case "settle":
        return this.settleMessage(client);
    }
  }

  // --- lobby ---

  private hello(client: Client, name: string, token?: string): void {
    client.name = name.trim() || "Jogador";
    // Reclaim: a seat remembering this token? A token that matches nothing is
    // stale (the seat was taken by someone else) and this is a new arrival.
    const at = token
      ? this.seats.findIndex(
          (seat) =>
            (seat.kind === "human" || seat.kind === "covered") && seat.token === token,
        )
      : -1;
    if (at >= 0) {
      const seat = this.seats[at] as { name: string; token: string };
      // Replacing one live tab with another: the displaced tab loses the seat,
      // and the reclaiming client vacates any seat it already held.
      for (const other of this.clients.values()) {
        if (other !== client && other.seat === at) {
          other.seat = null;
          this.send(other.ws, {
            type: "welcome",
            token: other.token,
            seat: null,
            table: this.tableState(),
          });
        }
      }
      if (client.seat !== null && client.seat !== at) this.vacate(client);
      this.seats[at] = { kind: "human", name: seat.name, token: seat.token, ws: client.ws };
      client.name = seat.name;
      client.token = seat.token;
      client.seat = at;
    } else {
      client.token = token || randomUUID();
      client.seat = null;
    }
    this.send(client.ws, {
      type: "welcome",
      token: client.token,
      seat: client.seat,
      table: this.tableState(),
    });
    if (client.seat !== null) this.sendView(client.seat);
    this.broadcastTable();
  }

  private sit(client: Client, seat: Seat): void {
    if (seat < 0 || seat > 3) return;
    // A client that sits without greeting would otherwise persist an empty
    // token and an anonymous name — assign both on the spot.
    if (!client.token) client.token = randomUUID();
    if (!client.name) client.name = "Jogador";
    const target = this.seats[seat];
    if (target.kind === "human") {
      return this.refuse(client, "SeatTaken", `seat ${seat} is taken`);
    }
    if (client.seat !== null) this.vacate(client); // stand from wherever we were
    this.seats[seat] = { kind: "human", name: client.name, token: client.token, ws: client.ws };
    client.seat = seat;
    // The token is now bound to the seat — say so, and show the hand if there is one.
    this.send(client.ws, {
      type: "welcome",
      token: client.token,
      seat,
      table: this.tableState(),
    });
    if (this.match) this.sendView(seat);
    this.afterChange();
  }

  private stand(client: Client): void {
    if (client.seat === null) return;
    this.vacate(client);
    this.afterChange();
  }

  /** Leave a seat: a bot covers mid-match; the seat empties in the lobby. Token binding cleared. */
  private vacate(client: Client): void {
    const seat = client.seat;
    if (seat === null) return;
    this.seats[seat] = this.match
      ? { kind: "bot", botId: COVER_BOT }
      : { kind: "empty" };
    client.seat = null;
  }

  /** A dropped connection: the seat stays the human's, played by a bot, reclaimable by token. */
  private coverSeat(seat: Seat): void {
    const current = this.seats[seat];
    if (current.kind === "human") {
      this.seats[seat] = { kind: "covered", name: current.name, token: current.token };
    }
  }

  // --- match (implemented in Task 4) ---

  private start(client: Client): void {
    this.refuse(client, "NotRunning", "match flow lands in Task 4");
  }

  private act(client: Client, action: Action): void {
    void action;
    this.refuse(client, "NotRunning", "no match in progress");
  }

  private restart(client: Client): void {
    this.refuse(client, "NotRunning", "no match in progress");
  }

  private settleMessage(client: Client): void {
    this.refuse(client, "NotRunning", "no hand to settle");
  }

  // --- plumbing ---

  private refuse(client: Client, error: string, detail: string): void {
    this.send(client.ws, { type: "refused", violation: { error, detail } });
  }

  private send(ws: WebSocket, message: ServerMessage): void {
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(message));
  }

  private broadcast(message: ServerMessage): void {
    for (const client of this.clients.keys()) this.send(client, message);
  }

  private broadcastTable(): void {
    this.broadcast({ type: "table", table: this.tableState() });
  }

  private sendView(seat: Seat): void {
    const occupant = this.seats[seat];
    if (!this.match || occupant.kind !== "human") return;
    this.send(occupant.ws, { type: "view", view: this.match.views()[seat] });
  }

  private occupantOf(seat: SeatState): SeatOccupant {
    switch (seat.kind) {
      case "human":
        return { kind: "human", name: seat.name, connected: true };
      case "covered":
        return { kind: "human", name: seat.name, connected: false };
      case "bot":
        return { kind: "bot", botId: seat.botId };
      case "empty":
        return { kind: "empty" };
    }
  }

  private tableState(): TableState {
    const view = this.match?.views()[0] ?? null;
    return {
      seats: this.seats.map((seat) => this.occupantOf(seat)) as TableState["seats"],
      phase: view ? view.phase : "lobby",
      turn: view && view.phase !== "MatchOver" ? view.turn : null,
      scores: view ? view.scores : null,
      handNumber: view ? view.hand_number : null,
    };
  }

  /** Every state change routes through here: tell everyone, save, then let bots play. */
  private afterChange(): void {
    this.broadcastTable();
    for (let seat = 0; seat < 4; seat++) this.sendView(seat as Seat);
    this.options.onChange?.(this.serialize());
  }
}
