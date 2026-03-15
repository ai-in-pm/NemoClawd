import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

export type BridgeCommandResult<T = unknown> = {
  ok: boolean;
  code: number | null;
  stdout: string;
  stderr: string;
  payload: T;
};

export type NemoBridgeHealth = {
  ok: boolean;
  natAvailable: boolean;
  natExecutable: string | null;
  python: string;
  pythonVersion: string;
  platform: string;
  cwd: string;
};

export type NemoWorkflowRunResult = {
  ok: boolean;
  code: number;
  command: string[];
  cwd: string;
  stdout: string;
  stderr: string;
};

export type NemoBridgeOptions = {
  rootDir?: string;
  pythonBinary?: string;
  natWorkdir?: string;
  timeoutMs?: number;
  env?: NodeJS.ProcessEnv;
};

export type RunWorkflowOptions = {
  configFile: string;
  input: string;
  natWorkdir?: string;
  timeoutMs?: number;
  extraArgs?: string[];
};

const defaultTimeoutMs = 120_000;

function currentModuleDir(): string {
  return path.dirname(fileURLToPath(import.meta.url));
}

function defaultRootDir(): string {
  return path.resolve(currentModuleDir(), "..");
}

function defaultPythonBinary(rootDir: string): string {
  if (process.env.NEMOCLAWD_NAT_PYTHON) {
    return process.env.NEMOCLAWD_NAT_PYTHON;
  }

  const venvRoot = path.join(rootDir, "apps", "NeMo-Agent-Toolkit-develop", ".venv");
  return process.platform === "win32"
    ? path.join(venvRoot, "Scripts", "python.exe")
    : path.join(venvRoot, "bin", "python");
}

function defaultNatWorkdir(rootDir: string): string {
  return process.env.NEMOCLAWD_NAT_WORKDIR ?? path.join(rootDir, "apps", "NeMo-Agent-Toolkit-develop");
}

function parseJson<T>(stdout: string): T {
  const trimmed = stdout.trim();
  if (!trimmed) {
    throw new Error("Expected JSON output from the Python bridge, but stdout was empty.");
  }

  return JSON.parse(trimmed) as T;
}

export class NemoBridge {
  readonly rootDir: string;
  readonly pythonBinary: string;
  readonly natWorkdir: string;
  readonly timeoutMs: number;
  readonly env: NodeJS.ProcessEnv;

  constructor(options: NemoBridgeOptions = {}) {
    this.rootDir = options.rootDir ?? defaultRootDir();
    this.pythonBinary = options.pythonBinary ?? defaultPythonBinary(this.rootDir);
    this.natWorkdir = options.natWorkdir ?? defaultNatWorkdir(this.rootDir);
    this.timeoutMs = options.timeoutMs ?? defaultTimeoutMs;
    this.env = options.env ?? {};
  }

  async health(): Promise<BridgeCommandResult<NemoBridgeHealth>> {
    return this.runBridgeCommand<NemoBridgeHealth>(["health"], this.timeoutMs);
  }

  async runWorkflow(options: RunWorkflowOptions): Promise<BridgeCommandResult<NemoWorkflowRunResult>> {
    const args = [
      "run",
      "--config-file",
      path.resolve(this.rootDir, options.configFile),
      "--input",
      options.input,
      "--nat-workdir",
      options.natWorkdir ?? this.natWorkdir,
    ];

    for (const extraArg of options.extraArgs ?? []) {
      args.push("--nat-arg", extraArg);
    }

    return this.runBridgeCommand<NemoWorkflowRunResult>(args, options.timeoutMs ?? this.timeoutMs);
  }

  private async runBridgeCommand<T>(args: string[], timeoutMs: number): Promise<BridgeCommandResult<T>> {
    const command = [this.pythonBinary, "-m", "nemoclawd_bridge", ...args];

    return new Promise((resolve, reject) => {
      const child = spawn(command[0], command.slice(1), {
        cwd: this.rootDir,
        env: {
          ...process.env,
          ...this.env,
          PYTHONPATH: path.join(this.rootDir, "python_src"),
          NEMOCLAWD_ROOT_DIR: this.rootDir,
        },
        stdio: ["ignore", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";
      let settled = false;

      const timer = setTimeout(() => {
        child.kill("SIGKILL");
      }, timeoutMs);

      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });

      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });

      child.on("error", (error) => {
        if (settled) {
          return;
        }

        settled = true;
        clearTimeout(timer);
        reject(error);
      });

      child.on("close", (code) => {
        if (settled) {
          return;
        }

        settled = true;
        clearTimeout(timer);

        try {
          const payload = parseJson<T>(stdout);
          resolve({
            ok: code === 0,
            code,
            stdout,
            stderr,
            payload,
          });
        } catch (error) {
          reject(
            new Error(
              `Failed to parse Python bridge output. Exit code: ${code ?? "null"}. Stdout: ${stdout}. Stderr: ${stderr}. Error: ${String(error)}`,
            ),
          );
        }
      });
    });
  }
}
