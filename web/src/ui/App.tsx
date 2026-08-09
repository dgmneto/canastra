import { useCallback, useEffect, useRef, useState } from "react";
import { Match, loadEngine, logToText } from "../match";
import { label, penaltyLabel, step } from "../driver";
import { makeRng, type Rng } from "../rng";
import { BOTS, DEFAULT_BOT, botById } from "@canastra/bots";
import type { HandScore, PlayerView } from "../types";
import { handScoreTotal } from "../types";
import { Hand, CardChip, MeldView } from "./Cards";

const SEAT_NAMES = ["Sul", "Oeste", "Norte", "Leste"];
/**
 * §2: partners sit facing each other, so seats 0 and 2 are one partnership.
 * The sandbox watches from seat 0, which is what makes "nós" and "eles" mean
 * anything — team 0 is always the one seat 0 belongs to.
 */
const TEAM_NAMES = ["nós", "eles"];

interface Event {
  text: string;
  refusals: string[];
}

export function App() {
  const [ready, setReady] = useState(false);
  const [seedText, setSeedText] = useState("7");
  const [views, setViews] = useState<PlayerView[] | null>(null);
  const [handScores, setHandScores] = useState<[HandScore, HandScore] | null>(null);
  const [events, setEvents] = useState<Event[]>([]);
  const [playing, setPlaying] = useState(false);
  const [delay, setDelay] = useState(250);
  // A bot reaches a legal move by guessing, so most turns refuse a dozen
  // candidates first. Interesting when you are chasing a rule, noise otherwise.
  const [showRefusals, setShowRefusals] = useState(false);
  // Pause at `HandOver`, before the hand is settled, so the final position is
  // still on the table to look at.
  const [stopOnHandEnd, setStopOnHandEnd] = useState(false);
  // Which bot sits in each seat. Changing one takes effect on the next match,
  // since swapping a policy mid-hand would make the log describe two matches.
  const [lineup, setLineup] = useState<string[]>(() => SEAT_NAMES.map(() => DEFAULT_BOT.id));

  const match = useRef<Match | null>(null);
  // The lineup as it was when this match was dealt. Held apart from `lineup`
  // so editing the pickers mid-match cannot change who is playing it — the log
  // header names these, and it has to stay true for the whole match.
  const seated = useRef<string[]>(SEAT_NAMES.map(() => DEFAULT_BOT.id));
  const rng = useRef<Rng>(makeRng(1));
  // Set for the rest of a turn once a turn has dead-ended, so the retry does
  // not walk into the same wall: draw and discard only.
  const safeMode = useRef(false);

  useEffect(() => {
    loadEngine().then(() => setReady(true));
  }, []);

  const start = useCallback(() => {
    let seed: bigint;
    try {
      seed = BigInt(seedText);
    } catch {
      seed = 7n;
    }
    seated.current = [...lineup];
    match.current = new Match(seed, seated.current);
    // Poking at a position from the console is most of what a sandbox is for.
    (globalThis as { canastra?: Match }).canastra = match.current;
    rng.current = makeRng(Number(seed % 2147483647n) || 1);
    safeMode.current = false;
    setViews(match.current.views());
    setHandScores(match.current.handScores());
    setEvents([]);
    setPlaying(false);
  }, [seedText, lineup]);

  useEffect(() => {
    if (ready && !match.current) start();
  }, [ready, start]);

  const advance = useCallback(() => {
    const current = match.current;
    if (!current) return false;

    const view = current.views()[0];
    if (view.phase === "MatchOver") return false;

    const acting = view.turn;
    const bot = botById(seated.current[acting]);
    const result = step(current, current.views()[acting], bot, {
      rng: rng.current,
      safeMode: safeMode.current,
      encode: () => current.encodeState(acting),
    });
    if (!result) return false;

    // The retry has to stay in safe mode for the whole turn. Clearing it on the
    // next action instead would replay the same dead end — the deal is
    // deterministic, so an unmodified retry reproduces the position exactly.
    if (result.action === "restartTurn") {
      safeMode.current = true;
    } else if (
      result.action !== "settleHand" &&
      (result.action.type === "Discard" || result.action.type === "EndTurnWithoutDiscard")
    ) {
      safeMode.current = false;
    }

    const after = current.views();
    setViews(after);
    setHandScores(current.handScores());
    setEvents((previous) =>
      [
        {
          text:
            label(result.action, acting, `${SEAT_NAMES[acting]} (${bot.name})`) +
            (result.penalized ? penaltyLabel(after[acting]) : ""),
          refusals: result.refusals,
        },
        ...previous,
      ].slice(0, 200),
    );

    // §11: the hand has stopped but is not yet banked, so everything that
    // decided the score is still on the table. `step` settles it on the next
    // call, which is where the position would disappear.
    if (stopOnHandEnd && after[0].phase === "HandOver") setPlaying(false);
    return true;
  }, [stopOnHandEnd]);

  useEffect(() => {
    if (!playing) return;
    const timer = setTimeout(() => {
      if (!advance()) setPlaying(false);
    }, delay);
    return () => clearTimeout(timer);
  }, [playing, delay, advance, events]);

  const download = () => {
    if (!match.current) return;
    const blob = new Blob([logToText(match.current.log)], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `canastra-${match.current.seed}.jsonl`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  if (!ready) return <div className="loading">loading engine…</div>;
  if (!views) return <div className="loading">dealing…</div>;

  const table = views[0];
  const over = table.phase === "MatchOver";

  return (
    <div className="app">
      <header>
        <h1>Canastra</h1>
        <span className="sub">engine sandbox</span>
        <label>
          seed
          <input value={seedText} onChange={(event) => setSeedText(event.target.value)} size={8} />
        </label>
        <button onClick={start}>new match</button>
        <button onClick={advance} disabled={over}>
          step
        </button>
        <button onClick={() => setPlaying(!playing)} disabled={over}>
          {playing ? "pause" : "play"}
        </button>
        <label>
          speed
          <input
            type="range"
            min={0}
            max={600}
            step={50}
            value={600 - delay}
            onChange={(event) => setDelay(600 - Number(event.target.value))}
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={showRefusals}
            onChange={(event) => setShowRefusals(event.target.checked)}
          />
          refusals
        </label>
        <label>
          <input
            type="checkbox"
            checked={stopOnHandEnd}
            onChange={(event) => setStopOnHandEnd(event.target.checked)}
          />
          stop on hand end
        </label>
        <button onClick={download}>download log</button>
      </header>

      <section className="status">
        <Stat label="hand" value={String(table.hand_number)} />
        <Stat label="phase" value={table.phase} />
        <Stat label="turn" value={SEAT_NAMES[table.turn]} />
        <Stat label="stock" value={String(table.stock_count)} />
        <Stat label="nós" value={String(table.scores[0])} />
        <Stat label="eles" value={String(table.scores[1])} />
        {table.went_out !== null && <Stat label="bateu" value={SEAT_NAMES[table.went_out]} />}
      </section>

      <main>
        <div className="seats">
          {views.map((view, seat) => (
            <div key={seat} className={`seat${view.turn === seat ? " active" : ""}`}>
              <div className="seat-head">
                <strong>{SEAT_NAMES[seat]}</strong>
                <span className="team">{TEAM_NAMES[seat % 2]}</span>
                <select
                  className="bot-pick"
                  value={lineup[seat]}
                  title={`${botById(lineup[seat]).blurb} — takes effect on the next match`}
                  onChange={(event) => {
                    // Read the value now, not inside the updater: the updater
                    // runs later, by which point `event.target` is whatever the
                    // select currently holds rather than what was just chosen.
                    const chosen = event.target.value;
                    setLineup((previous) =>
                      previous.map((id, at) => (at === seat ? chosen : id)),
                    );
                  }}
                >
                  {BOTS.map((bot) => (
                    <option key={bot.id} value={bot.id}>
                      {bot.name}
                    </option>
                  ))}
                </select>
                <span className="count">{view.hand.length} cards</span>
                {view.pending_refusal && (
                  <span className="badge">offered {view.pending_refusal}</span>
                )}
              </div>
              <Hand cards={view.hand} frozen={view.frozen} />
            </div>
          ))}
        </div>

        <div className="middle">
          <div className="pile">
            <h2>discard ({table.discard.length})</h2>
            <div className="hand">
              {table.discard.slice(-14).map((card, index) => (
                <CardChip key={index} card={card} dim={index !== table.discard.length - 1 && table.discard.length <= 14} />
              ))}
            </div>
          </div>

          {[0, 1].map((team) => (
            <div key={team} className="team-table">
              <h2>
                {TEAM_NAMES[team]}
                {table.tables[team].opened ? "" : ` — not open (needs ${views[team].opening_minimum})`}
                {handScores && <HandTotal score={handScores[team]} />}
              </h2>
              {table.tables[team].red_threes.length > 0 && (
                <div className="reds">
                  red 3s:{" "}
                  {table.tables[team].red_threes.map((card, index) => (
                    <CardChip key={index} card={card} />
                  ))}
                </div>
              )}
              {table.tables[team].melds.map((meld, index) => (
                <MeldView key={index} meld={meld} index={index} />
              ))}
            </div>
          ))}
        </div>

        <aside className="log">
          <h2>moves</h2>
          {events.map((event, index) => (
            <div key={index} className="event">
              <div>{event.text}</div>
              {showRefusals && event.refusals.length > 0 && (
                <details>
                  <summary>{event.refusals.length} refused first</summary>
                  {event.refusals.map((refusal, at) => (
                    <div key={at} className="refusal">
                      {refusal}
                    </div>
                  ))}
                </details>
              )}
            </div>
          ))}
        </aside>
      </main>
    </div>
  );
}

/**
 * §13 as it stands right now, with the arithmetic behind it on hover. Mid-hand
 * this moves — cards leave hand for the table, and a seventh card on a meld
 * swings it by 200 or 500 at once.
 */
function HandTotal({ score }: { score: HandScore }) {
  const total = handScoreTotal(score);
  const parts = score.unopened_penalty
    ? [`${signed(score.unopened_penalty)} never opened — nothing else counts`]
    : [
        `${score.table_cards} on the table`,
        `${signed(-score.hand_cards)} in hand`,
        score.canastra_bonus ? `${score.canastra_bonus} canastras` : null,
        score.red_three_bonus ? `${signed(score.red_three_bonus)} red 3s` : null,
        score.going_out_bonus ? `${score.going_out_bonus} for going out` : null,
      ].filter(Boolean);

  return (
    <span className={`hand-total${total < 0 ? " negative" : ""}`} title={parts.join(" · ")}>
      {signed(total)}
    </span>
  );
}

/** A true minus sign rather than a hyphen, so the columns line up. */
function signed(value: number): string {
  return value < 0 ? `−${Math.abs(value)}` : `+${value}`;
}

function Stat({ label: name, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{name}</span>
      <strong>{value}</strong>
    </div>
  );
}
