import { useEffect, useState } from "react";
import type { GameClient, ClientState } from "./client";
import type { HandScore, MeldTarget, RuleViolation, Seat } from "@canastra/bots";
import { handScoreTotal } from "@canastra/bots";
import { CardChip, MeldView } from "../ui/Cards";
import { SEAT_NAMES } from "./Lobby";
import { TurnControls, canTakePile } from "./TurnControls";

const PHASE_PT: Record<string, string> = {
  lobby: "saguão",
  AwaitingDraw: "comprando",
  AwaitingRefusalChoice: "decidindo",
  Melding: "baixando",
  HandOver: "fim da mão",
  MatchOver: "fim da partida",
};

/**
 * The game screen. Everything rendered here comes from the player's own
 * `view` (their hand, the tables, the pile) plus the public `table` state —
 * no other seat's cards ever reach the browser.
 *
 * Team labels follow the viewer: your partnership is "nós", the other "eles"
 * (seats 0+2 vs 1+3). Spectators get neutral labels.
 */
export function Table({ client, state }: { client: GameClient; state: ClientState }) {
  const { table, view, seat, events } = state;
  /** Taking the pile: armed from the button or by clicking the discard cell. */
  const [pileMode, setPileMode] = useState(false);
  /** Lay-off armed: the next click on one of our melds is the target. */
  const [meldPick, setMeldPick] = useState<((target: MeldTarget) => void) | null>(null);

  // A new turn (or a new hand) drops any half-armed interaction.
  const turn = view?.turn;
  const handNumber = view?.hand_number;
  useEffect(() => {
    setPileMode(false);
    setMeldPick(null);
  }, [seat, turn, handNumber]);

  if (!table) return null;

  const myTeam = seat !== null ? seat % 2 : null;
  const teamName = (team: number) =>
    myTeam === null ? (team === 0 ? "Sul·Norte" : "Oeste·Leste") : team === myTeam ? "nós" : "eles";

  const occupantName = (at: Seat) => {
    const occupant = table.seats[at];
    if (occupant.kind === "human") return occupant.name;
    if (occupant.kind === "bot") return `bot (${occupant.botId})`;
    return "—";
  };

  const over = table.phase === "MatchOver";
  const handOver = table.phase === "HandOver";

  const myTurn = view !== null && seat !== null && view.turn === seat;
  const canDraw = myTurn && view.phase === "AwaitingDraw" && !pileMode;
  const takeable = myTurn && canTakePile(view);
  const discardTop = view?.discard[view.discard.length - 1] ?? null;

  return (
    <div className="app">
      <header>
        <h1>Canastra</h1>
        <span className="sub">
          {seat !== null ? `você — ${SEAT_NAMES[seat]}` : "assistindo"}
        </span>
        {seat !== null && <button onClick={() => client.send({ type: "stand" })}>levantar</button>}
        <a href="/sandbox.html" className="dim">sandbox</a>
      </header>

      <section className="status">
        <Stat label="mão" value={String(table.handNumber ?? "—")} />
        <Stat label="fase" value={PHASE_PT[table.phase] ?? table.phase} />
        <Stat
          label="vez"
          value={table.turn !== null ? `${occupantName(table.turn)} (${SEAT_NAMES[table.turn]})` : "—"}
        />
        {table.scores && <Stat label={teamName(0)} value={String(table.scores[0])} />}
        {table.scores && <Stat label={teamName(1)} value={String(table.scores[1])} />}
        {view?.went_out != null && <Stat label="bateu" value={SEAT_NAMES[view.went_out]} />}
      </section>

      {state.refusal && <div className="refusal-banner">{describeRefusal(state.refusal)}</div>}

      {handOver && state.handOver && (
        <HandOverPanel scores={state.handOver} teamName={teamName} client={client} seated={seat !== null} />
      )}

      {over && table.scores && (
        <div className="panel">
          <h2>fim de partida</h2>
          <p>
            {teamName(0)} {table.scores[0]} — {table.scores[1]} {teamName(1)}
          </p>
          {seat !== null && (
            <button onClick={() => client.send({ type: "start" })}>nova partida</button>
          )}
        </div>
      )}

      <main className="table-grid">
        <div className="middle">
          <div className="top-row">
            <button
              className="cell"
              disabled={!canDraw}
              onClick={() => client.send({ type: "action", action: { type: "Draw" } })}
              title={canDraw ? "comprar do monte" : undefined}
            >
              <span className="cell-title">monte</span>
              <span className="stock-card">{view?.stock_count ?? "—"}</span>
            </button>

            <button
              className={`cell${pileMode ? " armed" : ""}`}
              disabled={!takeable && !pileMode}
              onClick={() => setPileMode(!pileMode)}
              title={takeable ? "pegar o lixo" : undefined}
            >
              <span className="cell-title">lixo ({view?.discard.length ?? 0})</span>
              {/* Only the top card shows; the rest of the pile stays hidden. */}
              {discardTop ? <CardChip card={discardTop} /> : <span className="dim">vazio</span>}
            </button>
          </div>

          {[0, 1].map((team) => (
            <div key={team} className="team-table">
              <h2>
                {teamName(team)}
                {view && !view.tables[team].opened && ` — não abriu (mínimo ${view.opening_minimum})`}
              </h2>
              {view && view.tables[team].red_threes.length > 0 && (
                <div className="reds">
                  3 vermelhos: {view.tables[team].red_threes.map((card, at) => <CardChip key={at} card={card} />)}
                </div>
              )}
              {view?.tables[team].melds.map((meld, index) => (
                <MeldView
                  key={index}
                  meld={meld}
                  index={index}
                  onPick={
                    meldPick && team === myTeam
                      ? () => meldPick({ kind: "Existing", meld: index })
                      : undefined
                  }
                />
              ))}
            </div>
          ))}
        </div>

        <aside className="log">
          <h2>jogadas</h2>
          {events.map((text, index) => (
            <div key={index} className="event">{text}</div>
          ))}
        </aside>
      </main>

      <footer>
        {seat === null ? (
          <div className="spectate">
            {table.seats.map((occupant, at) =>
              occupant.kind === "human" && occupant.connected ? null : (
                <button key={at} onClick={() => client.send({ type: "sit", seat: at })}>
                  assumir o lugar de {SEAT_NAMES[at]} (
                  {occupant.kind === "human" ? "ausente" : occupant.kind})
                </button>
              ),
            )}
          </div>
        ) : (
          view && (
            <TurnControls
              client={client}
              view={view}
              seat={seat}
              drawn={state.drawn}
              pileMode={pileMode}
              setPileMode={setPileMode}
              meldPick={meldPick}
              setMeldPick={setMeldPick}
            />
          )
        )}
      </footer>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

/** A card as read at the table, inside a sentence: `"TS"` reads as `10♠`. */
function fmtCard(card: unknown): string {
  if (typeof card !== "string") return String(card);
  if (card === "JOKER") return "o Coringa";
  const suits: Record<string, string> = { C: "♣", D: "♦", H: "♥", S: "♠" };
  return `${card[0] === "T" ? "10" : card[0]}${suits[card[1]] ?? card[1]}`;
}

const MELD_REASON_PT: Record<string, string> = {
  TooFewCards: "um jogo precisa de pelo menos três cartas",
  TooManyWilds: "um jogo aceita no máximo um curinga",
  MixedSuits: "uma sequência é de um naipe só",
  NotASequence: "essas cartas não formam uma sequência",
  DuplicateRank: "essa carta já está na sequência",
  TooLong: "uma sequência vai no máximo do 4 ao Ás",
  WildLocked: "o curinga está travado e não pode mais se mover",
  WrongSuitForTwo: "um 2 só entra em sequência do próprio naipe",
  NotAnAce: "só entram ases e curingas num jogo de ases",
};

/** A refusal, spelled out in Portuguese with the numbers the violation carries. */
function describeRefusal(refusal: RuleViolation): string {
  switch (refusal.error) {
    case "NotYourTurn":
      return "não é a sua vez";
    case "WrongPhase":
      return `essa jogada não vale nesta fase (${PHASE_PT[String(refusal.phase)] ?? String(refusal.phase)})`;
    case "HandIsOver":
      return "a mão já acabou";
    case "StockEmpty":
      return "o monte acabou";
    case "CardNotInHand":
      return `você não tem ${fmtCard(refusal.card)}`;
    case "NoSuchMeld":
      return "a sua dupla não tem esse jogo";
    case "InvalidMeld": {
      const reason = (refusal.reason as { reason?: string } | undefined)?.reason;
      return MELD_REASON_PT[reason ?? ""] ?? "jogo inválido";
    }
    case "OpeningMinimumNotMet":
      return `você baixou ${String(refusal.laid)} dos ${String(refusal.required)} necessários para abrir`;
    case "CannotReachOpeningMinimum":
      return `mesmo baixando tudo você chega no máximo a ${String(refusal.best_possible)} ` +
        `(${String(refusal.laid)} na mesa), abaixo dos ${String(refusal.required)} para abrir`;
    case "DiscardPileEmpty":
      return "o lixo está vazio";
    case "DiscardPileBlocked":
      return `${fmtCard(refusal.card)} no topo trava o lixo`;
    case "WildInDiscardCore":
      return "as 3 cartas que pegam o lixo precisam ser naturais";
    case "WildInPileCoreMeld":
      return "esse jogo não aceita curinga neste turno (§5)";
    case "CardFrozen":
      return `${fmtCard(refusal.card)} veio do lixo e está congelada neste turno`;
    case "NoCleanCanastra":
      return "bater exige uma canastra limpa na mesa";
    case "HandNotOver":
      return "a mão ainda está em andamento";
    case "MustDiscard":
      return "é preciso descartar para encerrar o turno";
    case "NoCardsGiven":
      return "nenhuma carta foi escolhida";
    case "WouldStrandLastCard":
      return "isso deixaria uma carta que você não poderia descartar, sem canastra limpa para bater";
    // The server's own lobby refusals.
    case "SeatTaken":
      return "esse lugar está ocupado";
    case "NotSeated":
      return "você não está sentado à mesa";
    case "MatchRunning":
      return "já há uma partida em andamento";
    default: {
      const extras = Object.entries(refusal)
        .filter(([key]) => key !== "error" && key !== "detail")
        .map(([key, value]) => `${key}: ${String(value)}`)
        .join(", ");
      return extras ? `${refusal.error} (${extras})` : refusal.error;
    }
  }
}

function HandOverPanel({
  scores,
  teamName,
  client,
  seated,
}: {
  scores: [HandScore, HandScore];
  teamName: (team: number) => string;
  client: GameClient;
  seated: boolean;
}) {
  return (
    <div className="panel">
      <h2>fim da mão</h2>
      {[0, 1].map((team) => (
        <p key={team}>
          {teamName(team)}: {handScoreTotal(scores[team])} (
          {scores[team].table_cards} da mesa, −{scores[team].hand_cards} da mão
          {scores[team].canastra_bonus ? `, ${scores[team].canastra_bonus} de canastras` : ""}
          {scores[team].red_three_bonus ? `, ${scores[team].red_three_bonus} dos 3 vermelhos` : ""}
          {scores[team].going_out_bonus ? `, ${scores[team].going_out_bonus} por bater` : ""}
          {scores[team].unopened_penalty ? `, ${scores[team].unopened_penalty} por não abrir` : ""})
        </p>
      ))}
      {seated && <button onClick={() => client.send({ type: "settle" })}>próxima mão</button>}
    </div>
  );
}
