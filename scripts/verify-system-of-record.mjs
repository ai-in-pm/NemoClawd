import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sorPath = path.join(rootDir, "system-of-record.json");
const artifactsDir = path.join(rootDir, "artifacts");
const statusPath = path.join(artifactsDir, "system-of-record.status.json");

function readText(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function checkIncludes(checks, text, needle, message) {
  if (!text.includes(needle)) {
    checks.errors.push(message);
  }
}

function main() {
  const checks = {
    ok: true,
    checkedAt: new Date().toISOString(),
    errors: [],
    warnings: [],
    summary: [],
  };

  if (!fs.existsSync(sorPath)) {
    checks.ok = false;
    checks.errors.push("Missing system-of-record.json");
    return finalize(checks);
  }

  const sor = JSON.parse(readText(sorPath));
  const readmePath = path.join(rootDir, sor.sources.readme);
  if (!fs.existsSync(readmePath)) {
    checks.ok = false;
    checks.errors.push(`Missing README at ${readmePath}`);
  } else {
    const readme = readText(readmePath);
    checkIncludes(
      checks,
      readme,
      sor.identity.assistantName,
      `README does not contain assistant name ${sor.identity.assistantName}`,
    );
    checks.summary.push(`README validated at ${readmePath}`);
  }

  const optionalSources = [
    {
      key: "canvas",
      checks: (text) => {
        checkIncludes(
          checks,
          text,
          sor.identity.assistantName,
          "Canvas does not contain the canonical assistant name",
        );
        checkIncludes(
          checks,
          text,
          sor.identity.sessionKey,
          "Canvas does not contain canonical session key",
        );
      },
    },
    {
      key: "clawdbotConfig",
      checks: (text) => {
        checkIncludes(
          checks,
          text,
          `\"name\": \"${sor.identity.assistantName}\"`,
          "clawdbot config does not contain canonical assistant identity name",
        );
      },
    },
  ];

  for (const source of optionalSources) {
    const sourcePath = sor.sources[source.key];
    if (!sourcePath) continue;
    const resolved = path.isAbsolute(sourcePath) ? sourcePath : path.join(rootDir, sourcePath);
    if (!fs.existsSync(resolved)) {
      checks.warnings.push(`Optional source not found: ${resolved}`);
      continue;
    }
    const text = readText(resolved);
    source.checks(text);
    checks.summary.push(`${source.key} validated at ${resolved}`);
  }

  checks.ok = checks.errors.length === 0;
  return finalize(checks);
}

function finalize(checks) {
  fs.mkdirSync(artifactsDir, { recursive: true });
  fs.writeFileSync(statusPath, `${JSON.stringify(checks, null, 2)}\n`, "utf8");

  if (checks.ok) {
    console.log("System of Record verification passed");
    for (const line of checks.summary) {
      console.log(`- ${line}`);
    }
    for (const warning of checks.warnings) {
      console.log(`- warning: ${warning}`);
    }
    process.exitCode = 0;
    return;
  }

  console.error("System of Record verification failed");
  for (const err of checks.errors) {
    console.error(`- ${err}`);
  }
  for (const warning of checks.warnings) {
    console.error(`- warning: ${warning}`);
  }
  process.exitCode = 1;
}

main();
