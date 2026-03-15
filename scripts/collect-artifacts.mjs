import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const artifactsDir = path.join(rootDir, "artifacts");
const pythonMirrorDir = path.join(artifactsDir, "python-nemo-source");

await fs.mkdir(pythonMirrorDir, { recursive: true });

const pythonArtifacts = [];
for (const entry of await fs.readdir(pythonMirrorDir)) {
  pythonArtifacts.push(path.relative(rootDir, path.join(pythonMirrorDir, entry)));
}

const manifest = {
  generatedAt: new Date().toISOString(),
  typescriptBridgeDist: "dist",
  clawdbotDist: "apps/clawdbot-main/dist",
  pythonBridgeBuilds: "artifacts/python-bridge",
  nemoPythonBuilds: pythonArtifacts,
};

await fs.writeFile(path.join(artifactsDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
