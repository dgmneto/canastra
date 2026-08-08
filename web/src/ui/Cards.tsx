import type { Card, Meld, Suit } from "../types";

const SUIT_SYMBOL: Record<string, string> = { C: "♣", D: "♦", H: "♥", S: "♠" };
const SUIT_NAME: Record<Suit, string> = {
  Clubs: "♣",
  Diamonds: "♦",
  Hearts: "♥",
  Spades: "♠",
};

/** A rank the way it is read at the table: the wire's "T" is a 10. */
export function rankText(rank: string): string {
  return rank === "T" ? "10" : rank;
}

export function CardChip({ card, dim, note }: { card: Card; dim?: boolean; note?: string }) {
  if (card === "JOKER") {
    return (
      <span className={`card joker${dim ? " dim" : ""}`} title={note}>
        ★
      </span>
    );
  }
  const [rank, suit] = [card[0], card[1]];
  const red = suit === "D" || suit === "H";
  return (
    <span className={`card${red ? " red" : ""}${dim ? " dim" : ""}`} title={note}>
      {rankText(rank)}
      {SUIT_SYMBOL[suit]}
    </span>
  );
}

export function Hand({ cards, frozen }: { cards: Card[]; frozen: Card[] }) {
  const remaining = [...frozen];
  return (
    <div className="hand">
      {[...cards]
        .sort(compareCards)
        .map((card, index) => {
          // §5: frozen is a multiset, so only as many copies as were swept up.
          const at = remaining.indexOf(card);
          if (at >= 0) remaining.splice(at, 1);
          return (
            <CardChip
              key={`${card}-${index}`}
              card={card}
              dim={at >= 0}
              note={at >= 0 ? "congelada neste turno (§5)" : undefined}
            />
          );
        })}
    </div>
  );
}

/**
 * One meld on a partnership's table. When `onPick` is given the meld is a
 * button — the game client arms it while the player is choosing where to lay
 * off, and hovering shows the green border.
 */
export function MeldView({
  meld,
  index,
  onPick,
}: {
  meld: Meld;
  index: number;
  onPick?: () => void;
}) {
  const Tag = onPick ? "button" : "div";
  const props = onPick ? { type: "button" as const, onClick: onPick } : {};

  if (meld.kind === "Aces") {
    const canastra = meld.aces.length + (meld.wild ? 1 : 0) >= 7;
    return (
      <Tag className={`meld${canastra ? " canastra" : ""}${onPick ? " pickable" : ""}`} {...props}>
        <span className="meld-tag">#{index} A</span>
        {meld.aces.map((card, at) => (
          <CardChip key={at} card={card} />
        ))}
        {meld.wild && <CardChip card={meld.wild} note="curinga" />}
      </Tag>
    );
  }

  const canastra = meld.cards.length >= 7;
  const dirty = meld.cards.some((slot) => slot.card.startsWith("2"));
  return (
    <Tag
      className={`meld${canastra ? (dirty ? " canastra dirty" : " canastra") : ""}${onPick ? " pickable" : ""}`}
      {...props}
    >
      <span className="meld-tag">
        #{index} {SUIT_NAME[meld.suit]}
      </span>
      {meld.cards.map((slot, at) => (
        <CardChip
          key={at}
          card={slot.card}
          note={
            slot.standingInRank
              ? `no lugar de ${rankText(slot.standingInRank)}${slot.locked ? " — travada (§9)" : " — ainda móvel"}`
              : undefined
          }
        />
      ))}
      {canastra && <span className="badge">{dirty ? "suja" : "limpa"}</span>}
    </Tag>
  );
}

const RANK_ORDER = "23456789TJQKA";

export function compareCards(a: Card, b: Card): number {
  if (a === "JOKER") return b === "JOKER" ? 0 : 1;
  if (b === "JOKER") return -1;
  if (a[1] !== b[1]) return a[1].localeCompare(b[1]);
  return RANK_ORDER.indexOf(a[0]) - RANK_ORDER.indexOf(b[0]);
}
