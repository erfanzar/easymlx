import { app, BrowserWindow, dialog, Menu, nativeImage, type MenuItemConstructorOptions } from "electron";
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import path from "node:path";

const isDev = process.argv.includes("--dev");
const smokeMode = process.env.EASYMLX_DESKTOP_SMOKE === "1";
const screenshotPath = process.env.EASYMLX_DESKTOP_SCREENSHOT;
const screenshotScrollY = Number.parseInt(process.env.EASYMLX_DESKTOP_SCREENSHOT_SCROLL_Y ?? "0", 10);
const screenshotTab = process.env.EASYMLX_DESKTOP_SCREENSHOT_TAB;
const screenshotFixture = process.env.EASYMLX_DESKTOP_SCREENSHOT_FIXTURE;
const appDir = findAppDir();
const repoRoot = path.resolve(appDir, "../../..");
const appName = "EasyMLX";
const appIconPath = path.join(repoRoot, "images", "easymlx-logo.png");
const backendHost = process.env.EASYMLX_DESKTOP_HOST ?? "127.0.0.1";
const backendPort = Number.parseInt(process.env.EASYMLX_DESKTOP_PORT ?? "8719", 10);
const viteHost = process.env.EASYMLX_DESKTOP_VITE_HOST ?? "127.0.0.1";
const vitePort = Number.parseInt(process.env.EASYMLX_DESKTOP_VITE_PORT ?? "5174", 10);
const windowWidth = Number.parseInt(process.env.EASYMLX_DESKTOP_WIDTH ?? "1440", 10);
const windowHeight = Number.parseInt(process.env.EASYMLX_DESKTOP_HEIGHT ?? "940", 10);

let backendProcess: ChildProcess | null = null;
let viteProcess: ChildProcess | null = null;
let shuttingDown = false;

configureAppIdentity();

function findAppDir(): string {
  let current = __dirname;
  for (let depth = 0; depth < 5; depth += 1) {
    if (fileExists(path.join(current, "package.json"))) {
      return current;
    }
    current = path.dirname(current);
  }
  throw new Error("Could not locate easymlx-app package.json.");
}

function fileExists(filePath: string): boolean {
  try {
    return fs.existsSync(filePath);
  } catch {
    return false;
  }
}

function configureAppIdentity(): void {
  process.title = appName;
  app.name = appName;
  app.setName(appName);
  app.setAppUserModelId("dev.easymlx.app");
  app.setAboutPanelOptions({
    applicationName: appName,
    applicationVersion: app.getVersion(),
    iconPath: appIconPath
  });
}

function installApplicationMenu(): void {
  const appMenu: MenuItemConstructorOptions[] =
    process.platform === "darwin"
      ? [
          {
            label: appName,
            submenu: [
              { role: "about", label: `About ${appName}` },
              { type: "separator" },
              { role: "services" },
              { type: "separator" },
              { role: "hide", label: `Hide ${appName}` },
              { role: "hideOthers" },
              { role: "unhide" },
              { type: "separator" },
              { role: "quit", label: `Quit ${appName}` }
            ]
          },
          { role: "fileMenu" },
          { role: "editMenu" },
          { role: "viewMenu" },
          { role: "windowMenu" }
        ]
      : [
          { role: "fileMenu" },
          { role: "editMenu" },
          { role: "viewMenu" },
          { role: "windowMenu" }
        ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(appMenu));
}

function setDockIdentity(): void {
  if (process.platform !== "darwin" || !app.dock || !fileExists(appIconPath)) {
    return;
  }
  app.dock.setIcon(nativeImage.createFromPath(appIconPath));
}

function findPython(): string {
  const candidates = [
    process.env.EASYMLX_PYTHON,
    path.join(repoRoot, ".venv", "bin", "python3"),
    path.join(repoRoot, ".venv", "bin", "python"),
    path.join(repoRoot, ".venv", "Scripts", "python.exe"),
    "python3",
    "python"
  ].filter(Boolean) as string[];
  return candidates.find((candidate) => candidate.includes(path.sep) ? fileExists(candidate) : true) ?? "python3";
}

function prefixPythonPath(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const pythonPath = path.join(repoRoot, "lib", "python");
  return {
    ...env,
    PYTHONPATH: [pythonPath, env.PYTHONPATH].filter(Boolean).join(path.delimiter),
    EASYMLX_DESKTOP: "1"
  };
}

function writePrefixed(stream: NodeJS.WriteStream, prefix: string, chunk: Buffer): void {
  const lines = chunk.toString().split(/\r?\n/);
  for (const line of lines) {
    if (line) {
      stream.write(`${prefix} ${line}\n`);
    }
  }
}

function startBackend(): string {
  const externalUrl = process.env.EASYMLX_DESKTOP_BACKEND_URL?.replace(/\/$/, "");
  if (externalUrl) {
    return externalUrl;
  }

  const python = findPython();
  const backendUrl = `http://${backendHost}:${backendPort}`;
  const child = spawn(
    python,
    ["-m", "easymlx.cli", "app", "--host", backendHost, "--port", String(backendPort)],
    {
      cwd: repoRoot,
      env: prefixPythonPath(process.env),
      stdio: ["ignore", "pipe", "pipe"]
    }
  );
  backendProcess = child;

  child.stdout?.on("data", (chunk: Buffer) => writePrefixed(process.stdout, "[easymlx]", chunk));
  child.stderr?.on("data", (chunk: Buffer) => writePrefixed(process.stderr, "[easymlx]", chunk));
  child.on("error", (error) => {
    if (!shuttingDown) {
      dialog.showErrorBox("EasyMLX server failed", error.message);
      app.quit();
    }
  });
  child.on("exit", (code, signal) => {
    backendProcess = null;
    if (!shuttingDown && code !== 0) {
      dialog.showErrorBox("EasyMLX server stopped", `Backend exited with ${signal ?? `code ${code}`}.`);
      app.quit();
    }
  });

  return backendUrl;
}

function startVite(backendUrl: string): string {
  const viteUrl = `http://${viteHost}:${vitePort}`;
  const bun = process.env.EASYMLX_BUN ?? "bun";
  const child = spawn(
    bun,
    ["run", "vite", "--host", viteHost, "--port", String(vitePort), "--strictPort"],
    {
      cwd: appDir,
      env: {
        ...process.env,
        EASYMLX_API_TARGET: backendUrl
      },
      stdio: ["ignore", "pipe", "pipe"]
    }
  );
  viteProcess = child;

  child.stdout?.on("data", (chunk: Buffer) => writePrefixed(process.stdout, "[vite]", chunk));
  child.stderr?.on("data", (chunk: Buffer) => writePrefixed(process.stderr, "[vite]", chunk));
  child.on("error", (error) => {
    if (!shuttingDown) {
      dialog.showErrorBox("EasyMLX frontend failed", error.message);
      app.quit();
    }
  });
  child.on("exit", (code, signal) => {
    viteProcess = null;
    if (!shuttingDown && code !== 0) {
      dialog.showErrorBox("EasyMLX frontend stopped", `Vite exited with ${signal ?? `code ${code}`}.`);
      app.quit();
    }
  });

  return viteUrl;
}

function requestOnce(url: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const client = parsed.protocol === "https:" ? https : http;
    const request = client.get(parsed, { timeout: 2000 }, (response) => {
      response.resume();
      resolve(response.statusCode ?? 0);
    });
    request.on("timeout", () => request.destroy(new Error("request timed out")));
    request.on("error", reject);
  });
}

async function waitForHttp(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastError: unknown = null;

  while (Date.now() < deadline) {
    try {
      const status = await requestOnce(url);
      if (status >= 200 && status < 500) {
        return;
      }
      lastError = new Error(`HTTP ${status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }

  const reason = lastError instanceof Error ? lastError.message : String(lastError);
  throw new Error(`Timed out waiting for ${url}: ${reason}`);
}

function ensureStaticBuild(): void {
  const indexPath = path.join(repoRoot, "lib", "python", "easymlx", "app", "static", "index.html");
  if (!fileExists(indexPath)) {
    throw new Error("Built frontend was not found. Run `bun run build` before starting the desktop app.");
  }
}

async function createWindow(url: string): Promise<void> {
  const win = new BrowserWindow({
    title: appName,
    width: Number.isFinite(windowWidth) ? windowWidth : 1440,
    height: Number.isFinite(windowHeight) ? windowHeight : 940,
    minWidth: 1100,
    minHeight: 720,
    show: false,
    movable: true,
    icon: appIconPath,
    backgroundColor: "#00000000",
    transparent: true,
    hasShadow: true,
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  win.once("ready-to-show", () => {
    if (!smokeMode && !screenshotPath) {
      win.show();
    }
  });

  await win.loadURL(url);
  if (isDev && process.env.EASYMLX_DESKTOP_DEVTOOLS === "1") {
    win.webContents.openDevTools({ mode: "detach" });
  }

  if (screenshotPath) {
    await new Promise((resolve) => setTimeout(resolve, 700));
    await win.webContents.executeJavaScript(`
      const tab = ${JSON.stringify(screenshotTab ?? "")};
      if (tab) {
        const buttons = [...document.querySelectorAll("button")];
        buttons.find((button) => button.textContent?.trim().toLowerCase() === tab.toLowerCase())?.click();
      }
      const y = ${Number.isFinite(screenshotScrollY) ? screenshotScrollY : 0};
      document.scrollingElement?.scrollTo(0, y);
      document.querySelector(".workspace")?.scrollTo(0, y);
    `);
    if (screenshotFixture === "long-chat") {
      const metrics = await win.webContents.executeJavaScript(`
        const messages = document.querySelector(".messages");
        const workspace = document.querySelector(".workspace");
        const conversation = document.querySelector(".conversation-pane");
        if (messages) {
          const thinking = Array.from({ length: 18 }, (_, index) =>
            String(index + 1) + ". Thinking stream line for scroll validation. This should stay inside the thinking block and not grow the whole page."
          ).join("\\n\\n");
          const content = Array.from({ length: 8 }, (_, index) =>
            "Assistant answer line " + String(index + 1) + " for the long chat layout fixture."
          ).join("\\n");
          messages.innerHTML = [
            '<article class="message user"><div class="message-role">user</div><div class="message-content">Can you code?</div></article>',
            '<article class="message assistant"><div class="message-role">assistant</div><section class="thinking-block"><div class="thinking-title"><span>Thinking</span></div><div class="thinking-content">' + thinking + '</div></section><div class="message-content">' + content + '</div></article>',
            '<article class="message user"><div class="message-role">user</div><div class="message-content">Now keep going with a longer response.</div></article>',
            '<article class="message assistant"><div class="message-role">assistant</div><section class="thinking-block"><div class="thinking-title"><span>Thinking</span></div><div class="thinking-content">' + thinking + '</div></section><div class="message-content">' + content + "\\n" + content + '</div></article>'
          ].join("");
          for (const block of [...messages.querySelectorAll(".thinking-content")]) {
            block.scrollTop = block.scrollHeight;
          }
          messages.scrollTop = messages.scrollHeight;
        }
        ({
          messagesClientHeight: messages?.clientHeight ?? 0,
          messagesScrollHeight: messages?.scrollHeight ?? 0,
          messagesScrollTop: Math.round(messages?.scrollTop ?? 0),
          workspaceClientHeight: workspace?.clientHeight ?? 0,
          workspaceScrollHeight: workspace?.scrollHeight ?? 0,
          conversationClientHeight: conversation?.clientHeight ?? 0,
          conversationScrollHeight: conversation?.scrollHeight ?? 0,
          bodyClientHeight: document.body.clientHeight,
          bodyScrollHeight: document.body.scrollHeight
        });
      `);
      process.stdout.write(`[screenshot] ${JSON.stringify(metrics)}\n`);
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
    const image = await win.webContents.capturePage();
    fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
    fs.writeFileSync(screenshotPath, image.toPNG());
  }

  if (smokeMode || screenshotPath) {
    setTimeout(() => app.quit(), 500);
  }
}

function stopProcess(child: ChildProcess | null): void {
  if (!child || child.killed) {
    return;
  }
  child.kill("SIGTERM");
  setTimeout(() => {
    if (!child.killed) {
      child.kill("SIGKILL");
    }
  }, 2500).unref();
}

function stopChildren(): void {
  shuttingDown = true;
  stopProcess(viteProcess);
  stopProcess(backendProcess);
}

async function boot(): Promise<void> {
  configureAppIdentity();
  installApplicationMenu();
  setDockIdentity();

  if (!isDev) {
    ensureStaticBuild();
  }

  const backendUrl = startBackend();
  await waitForHttp(`${backendUrl}/health`, 45_000);

  const appUrl = isDev ? startVite(backendUrl) : backendUrl;
  if (isDev) {
    await waitForHttp(appUrl, 45_000);
  }

  await createWindow(appUrl);
}

app.on("before-quit", stopChildren);
app.on("window-all-closed", () => app.quit());

process.on("SIGINT", () => {
  stopChildren();
  app.quit();
});

process.on("SIGTERM", () => {
  stopChildren();
  app.quit();
});

app.whenReady().then(() => {
  void boot().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    dialog.showErrorBox("EasyMLX failed to start", message);
    stopChildren();
    app.quit();
  });
});
