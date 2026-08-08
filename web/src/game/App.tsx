import { useEffect, useSyncExternalStore } from "react";
import { GameClient } from "./client";
import { Lobby } from "./Lobby";
import { Table } from "./Table";

const client = new GameClient();

export function App() {
  useEffect(() => client.connect(), []);
  const state = useSyncExternalStore(client.subscribe, client.getState);

  if (!state.connected || !state.table) {
    return <div className="loading">conectando…</div>;
  }
  if (state.table.phase === "lobby") {
    return <Lobby client={client} state={state} />;
  }
  return <Table client={client} state={state} />;
}
