import type { Card, Meld, Suit } from "../types";

const SUIT_SYMBOL: Record<string, string> = { C: "♣", D: "♦", H: "♥", S: "♠" };
const SUIT_NAME: Record<Suit, string> = {
  Clubs: "♣",
  Diamonds: "♦",
  Hearts: "♥",
  Spades: "♠",
};

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
      {rank}
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
              note={at >= 0 ? "frozen this turn (§5)" : undefined}
            />
          );
        })}
    </div>
  );
}

export function MeldView({ meld, index }: { meld: Meld; index: number }) {
  if (meld.kind === "Aces") {
    const canastra = meld.aces.length + (meld.wild ? 1 : 0) >= 7;
    return (
      <div className={`meld${canastra ? " canastra" : ""}`}>
        <span className="meld-tag">#{index} A</span>
        {meld.aces.map((card, at) => (
          <CardChip key={at} card={card} />
        ))}
        {meld.wild && <CardChip card={meld.wild} note="wild" />}
      </div>
    );
  }

  const canastra = meld.cards.length >= 7;
  const dirty = meld.cards.some((slot) => slot.card.startsWith("2"));
  return (
    <div className={`meld${canastra ? (dirty ? " canastra dirty" : " canastra") : ""}`}>
      <span className="meld-tag">
        #{index} {SUIT_NAME[meld.suit]}
      </span>
      {meld.cards.map((slot, at) => (
        <CardChip
          key={at}
          card={slot.card}
          note={
            slot.standingInRank
              ? `standing in for ${slot.standingInRank}${slot.locked ? " — locked (§9)" : " — still movable"}`
              : undefined
          }
        />
      ))}
      {canastra && <span className="badge">{dirty ? "suja" : "limpa"}</span>}
    </div>
  );
}

const RANK_ORDER = "23456789TJQKA";

export function compareCards(a: Card, b: Card): number {
  if (a === "JOKER") return b === "JOKER" ? 0 : 1;
  if (b === "JOKER") return -1;
  if (a[1] !== b[1]) return a[1].localeCompare(b[1]);
  return RANK_ORDER.indexOf(a[0]) - RANK_ORDER.indexOf(b[0]);
}
