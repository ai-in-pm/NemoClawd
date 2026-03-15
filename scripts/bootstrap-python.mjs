import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const nemoDir = path.join(rootDir, "apps", "NeMo-Agent-Toolkit-develop");
const pythonInVenv = process.platform === "win32"
  ? path.join(nemoDir, ".venv", "Scripts", "python.exe")
  : path.join(nemoDir, ".venv", "bin", "python");
const uvBinary = process.platform === "win32" ? "uv.exe" : "uv";

if (!fs.existsSync(pythonInVenv)) {
  if (await commandExists(uvBinary)) {
    await run(uvBinary, ["venv", "--python", "3.12", ".venv"], { cwd: nemoDir });
  } else {
    await run(resolveSystemPython(), ["-m", "venv", ".venv"], { cwd: nemoDir });
  }
}

await run(pythonInVenv, ["-m", "ensurepip", "--upgrade"], { cwd: nemoDir });
await run(pythonInVenv, ["-m", "pip", "install", "--upgrade", "pip", "build", "setuptools", "setuptools-scm", "setuptools_dynamic_dependencies"], { cwd: nemoDir });
await run(pythonInVenv, ["-m", "pip", "install", "nvidia-nat"], { cwd: nemoDir });

function resolveSystemPython() {
  return process.platform === "win32" ? "python" : "python3";
}

function commandExists(command) {
  return new Promise((resolve) => {
    const checker = spawn(command, ["--version"], {
      cwd: rootDir,
      env: process.env,
      stdio: "ignore",
    });

    checker.on("error", () => resolve(false));
    checker.on("close", (code) => resolve(code === 0));
  });
}

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
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
