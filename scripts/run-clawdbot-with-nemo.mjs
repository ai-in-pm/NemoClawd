import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const command = [
  "node",
  path.join(rootDir, "apps", "clawdbot-main", "dist", "index.js"),
  ...process.argv.slice(2),
];

const pythonPath = process.platform === "win32"
  ? path.join(rootDir, "apps", "NeMo-Agent-Toolkit-develop", ".venv", "Scripts", "python.exe")
  : path.join(rootDir, "apps", "NeMo-Agent-Toolkit-develop", ".venv", "bin", "python");

const child = spawn(command[0], command.slice(1), {
  cwd: rootDir,
  env: {
    ...process.env,
    NEMOCLAWD_ROOT_DIR: rootDir,
    NEMOCLAWD_NAT_PYTHON: process.env.NEMOCLAWD_NAT_PYTHON ?? pythonPath,
    NEMOCLAWD_NAT_WORKDIR: process.env.NEMOCLAWD_NAT_WORKDIR ?? path.join(rootDir, "apps", "NeMo-Agent-Toolkit-develop"),
  },
  stdio: "inherit",
});

child.on("close", (code) => {
  process.exit(code ?? 1);
});
