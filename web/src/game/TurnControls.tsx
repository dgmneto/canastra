import { useState } from "react";
import type { GameClient } from "./client";
import type { Action, Card, MeldTarget, PlayerView, Seat } from "@canastra/bots";
import { CardChip, compareCards } from "../ui/Cards";

const RANKS = "23456789TJQKA";

/**
 * A cheap shape check for enabling the "pegar o lixo" affordances (the button
 * and the pile cell) — the engine still judges the real attempt. §5: the top
 * card must be natural and not a 3, and the hand must hold two natural cards
 * that close a contiguous three with it — or two aces, for the Aces meld.
 */
export function canTakePile(view: PlayerView): boolean {
  if (view.phase !== "AwaitingDraw") return false;
  const top = view.discard[view.discard.length - 1];
  if (!top || top === "JOKER") return false;
  const topIndex = RANKS.indexOf(top[0]);
  if (topIndex < 2) return false; // a 2 is wild and a 3 blocks the pile

  // Natural 4..A of the top's suit, by rank index — wilds and 3s cannot join
  // the core. Two decks mean duplicates, but the core needs distinct ranks.
  const held = new Set(
    view.hand
      .filter((card) => card !== "JOKER" && card[1] === top[1])
      .map((card) => RANKS.indexOf(card[0]))
      .filter((rank) => rank >= 2),
  );
  const pairs: [number, number][] = [
    [topIndex - 2, topIndex - 1],
    [topIndex - 1, topIndex + 1],
    [topIndex + 1, topIndex + 2],
  ];
  if (pairs.some(([a, b]) => a >= 2 && b <= 12 && held.has(a) && held.has(b))) return true;

  // §7.2: two natural aces from hand plus an ace on top also take the pile.
  return top[0] === "A" && view.hand.filter((card) => card[0] === "A").length >= 2;
}

/**
 * The phase-driven action bar. The client offers; the engine judges — buttons
 * are enabled by cheap shape checks (3+ cards for a meld, 1 for a discard),
 * never by re-implemented rules, and a wrong guess comes back as a `refused`
 * banner that costs nothing.
 *
 * `pileMode` and `meldPick` live in the parent, because their targets do too:
 * the pile is taken by clicking the discard cell, and a lay-off by clicking
 * the meld itself on the table.
 */
export function TurnControls({
  client,
  view,
  seat,
  drawn,
  pileMode,
  setPileMode,
  meldPick,
  setMeldPick,
}: {
  client: GameClient;
  view: PlayerView;
  seat: Seat;
  drawn: Card | null;
  pileMode: boolean;
  setPileMode: (on: boolean) => void;
  meldPick: ((target: MeldTarget) => void) | null;
  setMeldPick: (pick: ((target: MeldTarget) => void) | null) => void;
}) {
  /** Indices into the sorted hand. */
  const [selected, setSelected] = useState<number[]>([]);

  const sorted = [...view.hand].sort(compareCards);
  const cards = selected.map((at) => sorted[at]);
  const myTurn = view.turn === seat;

  const send = (action: Action) => {
    client.send({ type: "action", action });
    setSelected([]);
    setPileMode(false);
    setMeldPick(null);
  };

  const toggle = (at: number) =>
    setSelected((previous) =>
      previous.includes(at) ? previous.filter((each) => each !== at) : [...previous, at],
    );

  if (!myTurn) {
    return (
      <div className="controls">
        <YourHand view={view} sorted={sorted} selected={selected} toggle={toggle} drawn={drawn} />
        <p className="dim">aguardando os outros…</p>
      </div>
    );
  }

  return (
    <div className="controls">
      <YourHand view={view} sorted={sorted} selected={selected} toggle={toggle} drawn={drawn} />

      {view.frozen.length > 0 && (
        <p className="dim">cartas esmaecidas vieram do lixo e estão congeladas neste turno (§5)</p>
      )}

      <div className="actions">
        {view.phase === "AwaitingDraw" && !pileMode && (
          <>
            <button onClick={() => send({ type: "Draw" })}>comprar do monte</button>
            <button disabled={!canTakePile(view)} onClick={() => setPileMode(true)}>
              pegar o lixo
            </button>
          </>
        )}

        {view.phase === "AwaitingDraw" && pileMode && !meldPick && (
          <>
            <span>base: escolha exatamente 2 cartas naturais</span>
            <button
              disabled={cards.length !== 2}
              onClick={() =>
                send({
                  type: "TakeDiscardPile",
                  core: [cards[0], cards[1]] as [Card, Card],
                  target: { kind: "NewMeld" },
                })
              }
            >
              como jogo novo
            </button>
            <button
              disabled={cards.length !== 2}
              onClick={() =>
                setMeldPick(
                  () => (target: MeldTarget) =>
                    send({
                      type: "TakeDiscardPile",
                      core: [cards[0], cards[1]] as [Card, Card],
                      target,
                    }),
                )
              }
            >
              em um jogo já baixado…
            </button>
            <button onClick={() => { setPileMode(false); setSelected([]); }}>cancelar</button>
          </>
        )}

        {view.phase === "AwaitingRefusalChoice" && view.pending_refusal && (
          <>
            <span>oferecida: <CardChip card={view.pending_refusal} /></span>
            <button onClick={() => send({ type: "KeepDrawnCard" })}>ficar com ela</button>
            <button onClick={() => send({ type: "RefuseDrawnCard" })}>recusar</button>
          </>
        )}

        {view.phase === "Melding" && !meldPick && (
          <>
            <button disabled={cards.length < 3} onClick={() => send({ type: "LayMeld", cards })}>
              baixar novo jogo
            </button>
            <button
              disabled={cards.length === 0}
              onClick={() =>
                setMeldPick(
                  () => (target: MeldTarget) =>
                    send({ type: "AddToMeld", meld: (target as { meld: number }).meld, cards }),
                )
              }
            >
              encaixar em um jogo…
            </button>
            <button disabled={cards.length !== 1} onClick={() => send({ type: "Discard", card: cards[0] })}>
              descartar
            </button>
            {view.stock_count === 0 && view.hand.length === 1 && (
              <button onClick={() => send({ type: "EndTurnWithoutDiscard" })}>
                encerrar sem descartar
              </button>
            )}
            <button
              onClick={() => {
                client.send({ type: "restartTurn" });
                setSelected([]);
                setPileMode(false);
                setMeldPick(null);
              }}
            >
              recomeçar o turno
            </button>
          </>
        )}

        {meldPick && (
          <>
            <span>clique em um dos jogos da sua dupla na mesa</span>
            <button onClick={() => setMeldPick(null)}>cancelar</button>
          </>
        )}
      </div>
    </div>
  );
}

function YourHand({
  view,
  sorted,
  selected,
  toggle,
  drawn,
}: {
  view: PlayerView;
  sorted: Card[];
  selected: number[];
  toggle: (at: number) => void;
  drawn: Card | null;
}) {
  const frozen = [...view.frozen];
  let fresh = drawn;
  return (
    <div className="hand yours">
      {sorted.map((card, at) => {
        // §5: frozen is a multiset — dim only as many copies as were swept up.
        const frozenAt = frozen.indexOf(card);
        if (frozenAt >= 0) frozen.splice(frozenAt, 1);
        // The drawn card likewise: only the copy that just arrived is "new".
        const isNew = frozenAt < 0 && fresh === card;
        if (isNew) fresh = null;
        return (
          <button
            key={`${card}-${at}`}
            className={`card-button${selected.includes(at) ? " selected" : ""}${isNew ? " new" : ""}`}
            onClick={() => toggle(at)}
          >
            <CardChip
              card={card}
              dim={frozenAt >= 0}
              note={
                frozenAt >= 0
                  ? "congelada neste turno (§5)"
                  : isNew
                    ? "comprada agora"
                    : undefined
              }
            />
          </button>
        );
      })}
    </div>
  );
}
