import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const appName = "EasyMLX";
const bundleId = "dev.easymlx.app";
const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const electronDist = path.join(appDir, "node_modules", "electron", "dist");
const sourceApp = path.join(electronDist, "Electron.app");
const cacheDir = path.join(appDir, ".desktop");
const brandedApp = path.join(cacheDir, `${appName}.app`);
const plistPath = path.join(brandedApp, "Contents", "Info.plist");
const executablePath = path.join(brandedApp, "Contents", "MacOS", "Electron");
const versionPath = path.join(electronDist, "version");
const stampPath = path.join(cacheDir, ".electron-version");

function readText(filePath: string): string {
  return fs.readFileSync(filePath, "utf8").trim();
}

function runPlistBuddy(command: string): void {
  const result = spawnSync("/usr/libexec/PlistBuddy", ["-c", command, plistPath], { stdio: "pipe" });
  if (result.status !== 0) {
    throw new Error(result.stderr.toString() || `PlistBuddy failed: ${command}`);
  }
}

function setPlistString(key: string, value: string): void {
  const setResult = spawnSync("/usr/libexec/PlistBuddy", ["-c", `Set :${key} ${value}`, plistPath], {
    stdio: "pipe"
  });
  if (setResult.status === 0) {
    return;
  }
  runPlistBuddy(`Add :${key} string ${value}`);
}

function ensureBrandedElectronApp(): void {
  const electronVersion = readText(versionPath);
  const previousVersion = fs.existsSync(stampPath) ? readText(stampPath) : "";
  if (!fs.existsSync(brandedApp) || previousVersion !== electronVersion) {
    fs.rmSync(brandedApp, { force: true, recursive: true });
    fs.mkdirSync(cacheDir, { recursive: true });
    fs.cpSync(sourceApp, brandedApp, { recursive: true });
    fs.writeFileSync(stampPath, electronVersion);
  }

  setPlistString("CFBundleDisplayName", appName);
  setPlistString("CFBundleName", appName);
  setPlistString("CFBundleIdentifier", bundleId);
}

ensureBrandedElectronApp();

const electronArgs = process.argv.slice(2);
if (electronArgs.length === 0 || electronArgs[0].startsWith("-")) {
  electronArgs.unshift(appDir);
}

const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const child = spawn(executablePath, electronArgs, {
  cwd: appDir,
  env,
  stdio: ["inherit", "inherit", "pipe"]
});

const STDERR_FILTERS = [/representedObject is not a WeakPtrToElectronMenuModelAsNSObject/];

let stderrBuffer = "";
child.stderr?.on("data", (chunk: Buffer) => {
  stderrBuffer += chunk.toString();
  const lines = stderrBuffer.split("\n");
  stderrBuffer = lines.pop() ?? "";
  for (const line of lines) {
    if (!STDERR_FILTERS.some((pattern) => pattern.test(line))) {
      process.stderr.write(`${line}\n`);
    }
  }
});
child.stderr?.on("end", () => {
  if (stderrBuffer && !STDERR_FILTERS.some((pattern) => pattern.test(stderrBuffer))) {
    process.stderr.write(stderrBuffer);
  }
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});

child.on("error", (error) => {
  console.error(error);
  process.exit(1);
});
