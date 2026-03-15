import path from "node:path";
import { fileURLToPath } from "node:url";
import { pathToFileURL } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const distEntryUrl = pathToFileURL(path.join(rootDir, "dist", "index.js")).href;
const { NemoBridge } = await import(distEntryUrl);

const bridge = new NemoBridge({ rootDir });
const result = await bridge.health();

console.log(JSON.stringify(result.payload, null, 2));

if (!result.ok || !result.payload.natAvailable) {
  process.exitCode = 1;
}
