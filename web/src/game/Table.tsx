import type { GameClient, ClientState } from "./client";
import type { HandScore } from "@canastra/bots";
import { handScoreTotal } from "@canastra/bots";
import { CardChip, MeldView } from "../ui/Cards";
import { SEAT_NAMES } from "./Lobby";
import { TurnControls } from "./TurnControls";

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
  if (!table) return null;

  const myTeam = seat !== null ? seat % 2 : null;
  const teamName = (team: number) =>
    myTeam === null ? (team === 0 ? "Sul·Norte" : "Oeste·Leste") : team === myTeam ? "nós" : "eles";

  const over = table.phase === "MatchOver";
  const handOver = table.phase === "HandOver";

  return (
    <div className="app">
      <header>
        <h1>Canastra</h1>
        <span className="sub">
          {seat !== null ? `${SEAT_NAMES[seat]} — you` : "spectating"}
        </span>
        {seat !== null && <button onClick={() => client.send({ type: "stand" })}>stand</button>}
        <a href="/sandbox.html" className="dim">sandbox</a>
      </header>

      <section className="status">
        <Stat label="hand" value={String(table.handNumber ?? "—")} />
        <Stat label="phase" value={table.phase} />
        <Stat label="turn" value={table.turn !== null ? SEAT_NAMES[table.turn] : "—"} />
        <Stat label="stock" value={view ? String(view.stock_count) : "—"} />
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
          <h2>match over</h2>
          <p>
            {teamName(0)} {table.scores[0]} — {table.scores[1]} {teamName(1)}
          </p>
          {seat !== null && (
            <button onClick={() => client.send({ type: "start" })}>new match</button>
          )}
        </div>
      )}

      <main className="table-grid">
        <div className="middle">
          <div className="pile">
            <h2>discard ({view?.discard.length ?? 0})</h2>
            <div className="hand">
              {(view?.discard ?? [])
                .slice(-14)
                .map((card, index, shown) => (
                  <CardChip key={index} card={card} dim={index !== shown.length - 1} />
                ))}
            </div>
          </div>

          {[0, 1].map((team) => (
            <div key={team} className="team-table">
              <h2>
                {teamName(team)}
                {view && !view.tables[team].opened && ` — not open (needs ${view.opening_minimum})`}
              </h2>
              {view && view.tables[team].red_threes.length > 0 && (
                <div className="reds">
                  red 3s: {view.tables[team].red_threes.map((card, at) => <CardChip key={at} card={card} />)}
                </div>
              )}
              {view?.tables[team].melds.map((meld, index) => (
                <MeldView key={index} meld={meld} index={index} />
              ))}
            </div>
          ))}
        </div>

        <aside className="log">
          <h2>moves</h2>
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
                  take {SEAT_NAMES[at]}'s seat (
                  {occupant.kind === "human" ? "away" : occupant.kind})
                </button>
              ),
            )}
          </div>
        ) : (
          view && <TurnControls client={client} view={view} seat={seat} />
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

/** A refusal, spelled out with the numbers the violation carries. */
function describeRefusal(refusal: { error: string; [field: string]: unknown }): string {
  const extras = Object.entries(refusal)
    .filter(([key]) => key !== "error" && key !== "detail")
    .map(([key, value]) => `${key}: ${String(value)}`)
    .join(", ");
  return extras ? `${refusal.error} (${extras})` : refusal.error;
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
      <h2>hand over</h2>
      {[0, 1].map((team) => (
        <p key={team}>
          {teamName(team)}: {handScoreTotal(scores[team])} (
          {scores[team].table_cards} table, −{scores[team].hand_cards} hand
          {scores[team].canastra_bonus ? `, ${scores[team].canastra_bonus} canastras` : ""}
          {scores[team].red_three_bonus ? `, ${scores[team].red_three_bonus} red 3s` : ""}
          {scores[team].going_out_bonus ? `, ${scores[team].going_out_bonus} bateu` : ""}
          {scores[team].unopened_penalty ? `, ${scores[team].unopened_penalty} never opened` : ""})
        </p>
      ))}
      {seated && <button onClick={() => client.send({ type: "settle" })}>next hand</button>}
    </div>
  );
}
