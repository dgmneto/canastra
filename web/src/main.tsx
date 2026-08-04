import { createRoot } from "react-dom/client";
import { App } from "./ui/App";
import * as lab from "./lab";
import "./styles.css";

// Bot-vs-bot runs from the console: `lab.headToHead("random", "random-plus")`.
(globalThis as { lab?: typeof lab }).lab = lab;

// No StrictMode: it double-invokes effects, which would build two wasm games
// per match and leave one of them leaked.
createRoot(document.getElementById("root")!).render(<App />);
