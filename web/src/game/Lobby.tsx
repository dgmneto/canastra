import type { GameClient, ClientState } from "./client";
import type { SeatOccupant } from "@canastra/protocol";

export const SEAT_NAMES = ["Sul", "Oeste", "Norte", "Leste"];

function OccupantLabel({ occupant }: { occupant: SeatOccupant }) {
  if (occupant.kind === "human")
    return (
      <span>
        {occupant.name}
        {!occupant.connected && <em> (away — bot playing)</em>}
      </span>
    );
  if (occupant.kind === "bot") return <span>bot ({occupant.botId})</span>;
  return <span className="dim">empty</span>;
}

export function Lobby({ client, state }: { client: GameClient; state: ClientState }) {
  const { table, seat } = state;
  if (!table) return null;
  const seated = seat !== null;

  return (
    <div className="app">
      <header>
        <h1>Canastra</h1>
        <span className="sub">mesa única — pick a seat, partnerships are 0+2 vs 1+3</span>
      </header>

      <section className="lobby">
        <label>
          your name
          <input
            defaultValue={client.name}
            size={12}
            onBlur={(event) => client.setName(event.target.value)}
          />
        </label>

        <div className="lobby-seats">
          {table.seats.map((occupant, at) => (
            <div key={at} className={`lobby-seat${seat === at ? " mine" : ""}`}>
              <strong>{SEAT_NAMES[at]}</strong>
              <span className="team">{at % 2 === 0 ? "nós (0+2)" : "eles (1+3)"}</span>
              <OccupantLabel occupant={occupant} />
              {seat === at ? (
                <button onClick={() => client.send({ type: "stand" })}>stand</button>
              ) : (
                occupant.kind !== "human" && (
                  <button onClick={() => client.send({ type: "sit", seat: at })}>sit</button>
                )
              )}
            </div>
          ))}
        </div>

        <button disabled={!seated} onClick={() => client.send({ type: "start" })}>
          start the match
        </button>
        <p className="dim">Empty seats are played by bots. Anyone seated can start.</p>
      </section>
    </div>
  );
}
