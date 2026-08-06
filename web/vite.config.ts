import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The game client talks to @canastra/server; the sandbox page needs nothing.
    proxy: { "/ws": { target: "ws://localhost:3001", ws: true } },
  },
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("index.html", import.meta.url)),
        sandbox: fileURLToPath(new URL("sandbox.html", import.meta.url)),
      },
    },
  },
});
