import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const nemoDir = path.join(rootDir, "apps", "NeMo-Agent-Toolkit-develop");
const nemoArtifactOutDir = path.join(rootDir, "artifacts", "python-nemo-source");
const pythonInVenv = process.platform === "win32"
  ? path.join(nemoDir, ".venv", "Scripts", "python.exe")
  : path.join(nemoDir, ".venv", "bin", "python");

await run(
  pythonInVenv,
  [
    "-m",
    "build",
    "--no-isolation",
    "--wheel",
    "--sdist",
    "--outdir",
    path.join(rootDir, "artifacts", "python-bridge"),
  ],
  { cwd: rootDir },
);

await run(
  pythonInVenv,
  ["-m", "build", "--no-isolation", "--wheel", "--sdist", "--outdir", nemoArtifactOutDir],
  {
  cwd: nemoDir,
  env: {
    ...process.env,
    SETUPTOOLS_SCM_PRETEND_VERSION: process.env.SETUPTOOLS_SCM_PRETEND_VERSION ?? "0.0.0",
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_NAT:
      process.env.SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_NAT ?? "0.0.0",
  },
  },
);

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd ?? rootDir,
      env: options.env ?? process.env,
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
