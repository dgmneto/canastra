import type { GameClient, ClientState } from "./client";
import type { SeatOccupant } from "@canastra/protocol";

export const SEAT_NAMES = ["Sul", "Oeste", "Norte", "Leste"];

function OccupantLabel({ occupant }: { occupant: SeatOccupant }) {
  if (occupant.kind === "human")
    return (
      <span>
        {occupant.name}
        {!occupant.connected && <em> (ausente — bot jogando)</em>}
      </span>
    );
  if (occupant.kind === "bot") return <span>bot ({occupant.botId})</span>;
  return <span className="dim">vazio</span>;
}

export function Lobby({ client, state }: { client: GameClient; state: ClientState }) {
  const { table, seat } = state;
  if (!table) return null;
  const seated = seat !== null;

  return (
    <div className="app">
      <header>
        <h1>Canastra</h1>
        <span className="sub">mesa única — escolha um lugar; as duplas são 0+2 contra 1+3</span>
      </header>

      <section className="lobby">
        <label>
          seu nome
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
                <button onClick={() => client.send({ type: "stand" })}>levantar</button>
              ) : occupant.kind !== "human" || !occupant.connected ? (
                <button onClick={() => client.send({ type: "sit", seat: at })}>
                  {occupant.kind === "human" ? "assumir" : "sentar"}
                </button>
              ) : null}
            </div>
          ))}
        </div>

        <button disabled={!seated} onClick={() => client.send({ type: "start" })}>
          começar a partida
        </button>
        <p className="dim">Lugares vazios são jogados por bots. Qualquer pessoa sentada pode começar.</p>
      </section>
    </div>
  );
}
