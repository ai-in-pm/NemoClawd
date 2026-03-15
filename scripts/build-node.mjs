import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const clawdbotDir = path.join(rootDir, "apps", "clawdbot-main");

await run("pnpm", ["build"], { cwd: clawdbotDir });

function quoteWindowsArg(value) {
  return /[\s"]/u.test(value) ? `"${value.replaceAll('"', '\\"')}"` : value;
}

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const invocation = process.platform === "win32"
      ? {
          command: process.env.ComSpec ?? "cmd.exe",
          args: ["/d", "/s", "/c", [command, ...args].map(quoteWindowsArg).join(" ")],
        }
      : { command, args };

    const child = spawn(invocation.command, invocation.args, {
      cwd: options.cwd ?? rootDir,
      env: process.env,
      stdio: "inherit",
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }

      reject(new Error(`${command} ${args.join(" ")} failed with exit code ${code ?? "null"}`));
    });
  });
}
