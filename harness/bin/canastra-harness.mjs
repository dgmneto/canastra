#!/usr/bin/env node
/**
 * Bin wrapper so `canastra-harness` works as a plain node executable.
 *
 * The CLI itself is TypeScript; running it needs a TS loader, so this spawns
 * tsx on the real entry point. `tsx` is a dependency of this package, resolved
 * here by the same node_modules walk.
 *
 * The real path matters: invoked through npm's `.bin` symlink, `import.meta.url`
 * (and anything derived from it) points at `node_modules/@canastra/harness`,
 * which would break the engine's relative import into `web/src/engine`. Do not
 * let it; the entry is resolved against the actual harness directory.
 */
import { spawn } from "node:child_process";
import { realpathSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const harnessRoot = dirname(dirname(realpathSync(fileURLToPath(import.meta.url))));

const child = spawn(
  process.execPath,
  [require.resolve("tsx/cli"), "src/cli.ts", ...process.argv.slice(2)],
  { stdio: "inherit", cwd: harnessRoot },
);
child.on("exit", (code) => process.exit(code ?? 1));
