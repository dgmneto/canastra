/**
 * The game client's connection: one WebSocket, one immutable state blob for
 * React, reconnect with backoff. It holds no rules — it renders what the
 * server sends and sends what the player clicks.
 *
 * The token in localStorage is what makes reclaim work: close the tab
 * mid-match, reopen, and the server hands the seat back.
 */

import type { ClientMessage, ServerMessage, TableState } from "@canastra/protocol";
import type { Card, HandScore, PlayerView, RuleViolation, Seat } from "@canastra/bots";

const TOKEN_KEY = "canastra:token";
const NAME_KEY = "canastra:name";

export interface ClientState {
  connected: boolean;
  seat: Seat | null;
  table: TableState | null;
  view: PlayerView | null;
  /** Newest first, capped — the move feed. */
  events: string[];
  /** The last refusal addressed to us, until the next accepted action clears it. */
  refusal: RuleViolation | null;
  /** §13 settlement, between HandOver and the settle. */
  handOver: [HandScore, HandScore] | null;
  /** The card the last draw added to our hand, so the UI can highlight it. */
  drawn: Card | null;
}

const INITIAL: ClientState = {
  connected: false,
  seat: null,
  table: null,
  view: null,
  events: [],
  refusal: null,
  handOver: null,
  drawn: null,
};

/** The multiset `a − b`: what shows up in `a` beyond what `b` already held. */
function multisetDiff(a: Card[], b: Card[]): Card[] {
  const rest = [...b];
  const extra: Card[] = [];
  for (const card of a) {
    const at = rest.indexOf(card);
    if (at >= 0) rest.splice(at, 1);
    else extra.push(card);
  }
  return extra;
}

export class GameClient {
  private state: ClientState = INITIAL;
  private listeners = new Set<() => void>();
  private ws: WebSocket | null = null;
  private token: string | undefined = localStorage.getItem(TOKEN_KEY) ?? undefined;
  private retryMs = 1_000;
  name: string = localStorage.getItem(NAME_KEY) ?? "";

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getState = (): ClientState => this.state;

  private emit(patch: Partial<ClientState>): void {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener();
  }

  connect(): void {
    const url = new URL("/ws", window.location.href);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(url);
    this.ws = ws;
    ws.onopen = () => {
      this.retryMs = 1_000;
      this.send({ type: "hello", name: this.name || "Jogador", token: this.token });
    };
    ws.onmessage = (event) => {
      try {
        this.receive(JSON.parse(String(event.data)) as ServerMessage);
      } catch {
        // Drop a malformed frame silently; only the well-formed server sends here.
      }
    };
    ws.onclose = () => {
      this.emit({ connected: false });
      setTimeout(() => this.connect(), this.retryMs);
      this.retryMs = Math.min(this.retryMs * 2, 10_000);
    };
  }

  setName(name: string): void {
    this.name = name;
    localStorage.setItem(NAME_KEY, name);
    this.send({ type: "hello", name: name || "Jogador", token: this.token });
  }

  send(message: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(message));
  }

  private receive(message: ServerMessage): void {
    switch (message.type) {
      case "welcome":
        this.token = message.token;
        localStorage.setItem(TOKEN_KEY, message.token);
        // seat-null means reclaim failed: this is a fresh session, so the dead
        // session's private state must not linger.
        this.emit(
          message.seat === null
            ? { connected: true, seat: null, table: message.table, view: null, handOver: null, refusal: null, drawn: null }
            : { connected: true, seat: message.seat, table: message.table }
        );
        break;
      case "table":
        // A settle (or a new match) closes the settlement panel.
        this.emit({
          table: message.table,
          handOver: message.table.phase === "HandOver" ? this.state.handOver : null,
        });
        break;
      case "view": {
        // Spot the card a draw just added, so the hand can highlight it. Only
        // a draw adds exactly one card and removes none — a pile take adds many,
        // a meld or a discard removes — so anything else clears the highlight.
        const previous = this.state.view;
        let drawn = this.state.drawn;
        if (!previous || previous.seat !== message.view.seat) {
          drawn = null;
        } else {
          const added = multisetDiff(message.view.hand, previous.hand);
          const removed = multisetDiff(previous.hand, message.view.hand);
          if (added.length === 1 && removed.length === 0) drawn = added[0];
          else if (added.length + removed.length > 0) drawn = null;
        }
        this.emit({ view: message.view, refusal: null, drawn });
        break;
      }
      case "event":
        this.emit({ events: [message.text, ...this.state.events].slice(0, 200) });
        break;
      case "refused":
        this.emit({ refusal: message.violation });
        break;
      case "handOver":
        this.emit({ handOver: message.scores });
        break;
    }
  }
}
