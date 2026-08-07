# protocol/ — `@canastra/protocol`

The client↔server wire messages for the multiplayer table (`ClientMessage`, `ServerMessage`,
`TableState`), shared by `server/` and `web/`. No game logic and no runtime dependencies — the
engine wire types come from `@canastra/bots`. One design rule: nothing here may be able to carry a
`GameState`, a snapshot, or another seat's hand.
