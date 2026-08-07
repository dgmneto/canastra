import { useState } from "react";
import type { GameClient } from "./client";
import type { Action, Card, MeldTarget, PlayerView, Seat } from "@canastra/bots";
import { CardChip, compareCards } from "../ui/Cards";

/**
 * The phase-driven action bar. The client offers; the engine judges — buttons
 * are enabled by cheap shape checks (3+ cards for a meld, 1 for a discard),
 * never by re-implemented rules, and a wrong guess comes back as a `refused`
 * banner that costs nothing.
 */
export function TurnControls({
  client,
  view,
  seat,
}: {
  client: GameClient;
  view: PlayerView;
  seat: Seat;
}) {
  /** Indices into the sorted hand. */
  const [selected, setSelected] = useState<number[]>([]);
  /** Taking the pile: pick exactly two core cards, then a target. */
  const [pileMode, setPileMode] = useState(false);
  /** Add-to-meld armed: the next click on one of our melds is the target. */
  const [addingToMeld, setAddingToMeld] = useState(false);

  const sorted = [...view.hand].sort(compareCards);
  const cards = selected.map((at) => sorted[at]);
  const myTurn = view.turn === seat;

  const send = (action: Action) => {
    client.send({ type: "action", action });
    setSelected([]);
    setPileMode(false);
    setAddingToMeld(false);
  };

  const toggle = (at: number) =>
    setSelected((previous) =>
      previous.includes(at) ? previous.filter((each) => each !== at) : [...previous, at],
    );

  if (!myTurn) {
    return (
      <div className="controls">
        <YourHand view={view} sorted={sorted} selected={selected} toggle={toggle} />
        <p className="dim">waiting for the others…</p>
      </div>
    );
  }

  return (
    <div className="controls">
      <YourHand view={view} sorted={sorted} selected={selected} toggle={toggle} />

      {view.frozen.length > 0 && (
        <p className="dim">dimmed cards came from the pile and are frozen this turn (§5)</p>
      )}

      <div className="actions">
        {view.phase === "AwaitingDraw" && !pileMode && (
          <>
            <button onClick={() => send({ type: "Draw" })}>draw from stock</button>
            <button disabled={view.discard.length === 0} onClick={() => setPileMode(true)}>
              take the pile
            </button>
          </>
        )}

        {view.phase === "AwaitingDraw" && pileMode && (
          <>
            <span>core: pick exactly 2 natural cards</span>
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
              as a new meld
            </button>
            <button disabled={cards.length !== 2} onClick={() => setAddingToMeld(true)}>
              onto an existing meld…
            </button>
            <button onClick={() => { setPileMode(false); setSelected([]); }}>cancel</button>
          </>
        )}

        {view.phase === "AwaitingRefusalChoice" && view.pending_refusal && (
          <>
            <span>offered: <CardChip card={view.pending_refusal} /></span>
            <button onClick={() => send({ type: "KeepDrawnCard" })}>keep it</button>
            <button onClick={() => send({ type: "RefuseDrawnCard" })}>refuse it</button>
          </>
        )}

        {view.phase === "Melding" && (
          <>
            <button disabled={cards.length < 3} onClick={() => send({ type: "LayMeld", cards })}>
              lay as new meld
            </button>
            <button disabled={cards.length === 0} onClick={() => setAddingToMeld(true)}>
              add to a meld…
            </button>
            <button disabled={cards.length !== 1} onClick={() => send({ type: "Discard", card: cards[0] })}>
              discard
            </button>
            {view.stock_count === 0 && view.hand.length === 1 && (
              <button onClick={() => send({ type: "EndTurnWithoutDiscard" })}>
                end without discarding
              </button>
            )}
            <button
              onClick={() => {
                client.send({ type: "restartTurn" });
                setSelected([]);
                setPileMode(false);
                setAddingToMeld(false);
              }}
            >
              restart turn
            </button>
          </>
        )}
      </div>

      {addingToMeld && (
        <MeldPicker
          view={view}
          seat={seat}
          onPick={(target) => {
            if (pileMode)
              send({
                type: "TakeDiscardPile",
                core: [cards[0], cards[1]] as [Card, Card],
                target,
              });
            else send({ type: "AddToMeld", meld: (target as { meld: number }).meld, cards });
          }}
          onCancel={() => setAddingToMeld(false)}
        />
      )}
    </div>
  );
}

function YourHand({
  view,
  sorted,
  selected,
  toggle,
}: {
  view: PlayerView;
  sorted: Card[];
  selected: number[];
  toggle: (at: number) => void;
}) {
  const frozen = [...view.frozen];
  return (
    <div className="hand yours">
      {sorted.map((card, at) => {
        // §5: frozen is a multiset — dim only as many copies as were swept up.
        const frozenAt = frozen.indexOf(card);
        if (frozenAt >= 0) frozen.splice(frozenAt, 1);
        return (
          <button
            key={`${card}-${at}`}
            className={`card-button${selected.includes(at) ? " selected" : ""}`}
            onClick={() => toggle(at)}
          >
            <CardChip
              card={card}
              dim={frozenAt >= 0}
              note={frozenAt >= 0 ? "frozen this turn (§5)" : undefined}
            />
          </button>
        );
      })}
    </div>
  );
}

function MeldPicker({
  view,
  seat,
  onPick,
  onCancel,
}: {
  view: PlayerView;
  seat: Seat;
  onPick: (target: MeldTarget) => void;
  onCancel: () => void;
}) {
  const team = seat % 2;
  return (
    <div className="meld-picker">
      {view.tables[team].melds.map((_, index) => (
        <button key={index} onClick={() => onPick({ kind: "Existing", meld: index })}>
          meld #{index}
        </button>
      ))}
      <button onClick={onCancel}>cancel</button>
    </div>
  );
}
