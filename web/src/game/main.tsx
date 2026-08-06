import { createRoot } from "react-dom/client";
import { App } from "./App";
import "../styles.css";

// No StrictMode, matching the sandbox: double-invoked effects would open two
// WebSockets per mount.
createRoot(document.getElementById("root")!).render(<App />);
