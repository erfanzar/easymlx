import {
  Activity,
  AlertTriangle,
  Bot,
  Boxes,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  Command,
  Copy,
  Cpu,
  Gauge,
  MessageSquarePlus,
  Monitor,
  Moon,
  Package,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Search,
  Send,
  Settings,
  SlidersHorizontal,
  Square,
  Sun,
  Trash2,
  X,
  Zap
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from "react";
import type {
  Dispatch,
  KeyboardEvent,
  ReactNode,
  SetStateAction
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  authHeaders,
  getConfigSchema,
  getHealth,
  getInfo,
  getMetrics,
  getModels,
  loadModel,
  responseMessage,
  unloadModel
} from "./api";
import type {
  AppInfo,
  AppModel,
  ChatMessage,
  ChatRequest,
  ChatStreamChunk,
  ChatUsage,
  ConfigSchema,
  EngineConfigField,
  Health,
  LoadModelRequest,
  Metrics,
  ModelStatus,
  SamplingState
} from "./types";
import {
  coerceEngineValue,
  compactObject,
  fieldDefaultToString,
  formatNumber,
  groupBy,
  isPlainObject,
  nullableText,
  parseJsonObject,
  parseJsonValue
} from "./utils";

type SectionId = "chat" | "models" | "engine" | "telemetry" | "settings";
type ThemeMode = "system" | "light" | "dark";
type Density = "comfortable" | "compact";

type Conversation = {
  id: string;
  title: string;
  model: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
};

type LoadFormState = {
  modelId: string;
  servedName: string;
  tokenizer: string;
  revision: string;
  convertedCache: string;
  modelClass: string;
  configPath: string;
  weightsName: string;
  subfolder: string;
  device: string;
  modelDtype: string;
  quantization: string;
  quantBits: string;
  quantGroupSize: string;
  cacheDtype: string;
  cacheBits: string;
  attnMechanism: string;
  draftModelDtype: string;
  draftQuantization: string;
  draftQuantBits: string;
  draftQuantGroupSize: string;
  localFilesOnly: boolean;
  forceConversion: boolean;
  replace: boolean;
  strict: boolean;
  lazy: boolean;
  autoConvertHf: string;
  copySupportFiles: boolean;
  modelKwargsJson: string;
  engineKwargsJson: string;
};

type LoadProfile = {
  id: string;
  name: string;
  updatedAt: number;
  form: LoadFormState;
  runtimeValues: Record<string, string | boolean>;
};

type EngineEvent = {
  id: string;
  at: number;
  tone: "good" | "warn" | "bad" | "neutral";
  text: string;
  detail?: string;
};

type SparkPoint = { t: number; v: number };

const STORAGE = {
  apiKey: "easymlx-api-key",
  theme: "easymlx-theme",
  density: "easymlx-density",
  conversations: "easymlx-conversations",
  activeConversation: "easymlx-active-conversation",
  profiles: "easymlx-load-profiles"
};

const MAX_PROFILES = 24;
const MAX_CONVERSATIONS = 100;
const MAX_EVENTS = 80;
const SPARK_LENGTH = 40;
const POLL_INTERVAL_MS = 1000;
const MAIN_MODEL_HINT = "Qwen/Qwen3.5-9B";
const DRAFT_MODEL_HINT = "z-lab/Qwen3.5-9B-DFlash";
const DRAFT_DEFAULT_TOKENS = "3";

const ESSENTIAL_RUNTIME_FIELDS = new Set([
  "max_model_len",
  "max_num_seqs",
  "max_num_batched_tokens",
  "page_size",
  "hbm_utilization",
  "dtype",
  "tool_parser",
  "reasoning_parser"
]);

const QUANT_OPTIONS: Array<{ value: string; label: string; hint: string }> = [
  { value: "", label: "None", hint: "Full precision" },
  { value: "affine", label: "Affine", hint: "Bits + group" },
  { value: "mxfp4", label: "MXFP4", hint: "4-bit FP" },
  { value: "mxfp8", label: "MXFP8", hint: "8-bit FP" },
  { value: "nvfp4", label: "NVFP4", hint: "NVIDIA FP4" }
];

const DTYPE_OPTIONS = ["", "float16", "bfloat16", "float32"];

const STEPS = [
  { id: 0, num: "01", title: "Source" },
  { id: 1, num: "02", title: "Identity" },
  { id: 2, num: "03", title: "Weights" },
  { id: 3, num: "04", title: "Engine" },
  { id: 4, num: "05", title: "Review" }
];

const SLASH_COMMANDS: Array<{ cmd: string; desc: string }> = [
  { cmd: "/model", desc: "Switch active model" },
  { cmd: "/clear", desc: "Clear current chat" },
  { cmd: "/new", desc: "Start a new chat" },
  { cmd: "/temp", desc: "Set temperature" },
  { cmd: "/max", desc: "Set max_tokens" },
  { cmd: "/stop", desc: "Set stop strings (comma)" },
  { cmd: "/system", desc: "Add a system message" }
];

const PRESETS = [
  { title: "Explain a concept", body: "Explain transformer attention to a senior engineer in 4 sentences." },
  { title: "Code review", body: "Review this Python function for clarity and edge cases:\n\n" },
  { title: "Brainstorm", body: "Give me 5 angles for a blog post about MLX vs CUDA." },
  { title: "Debug help", body: "I'm hitting this error, walk me through likely causes:\n\n" }
];

const defaultSampling: SamplingState = {
  max_tokens: 4096,
  temperature: 0,
  top_p: 1,
  top_k: 0,
  presence_penalty: 0,
  repetition_penalty: 1
};

const defaultLoadForm: LoadFormState = {
  modelId: "",
  servedName: "",
  tokenizer: "",
  revision: "",
  convertedCache: "",
  modelClass: "",
  configPath: "",
  weightsName: "",
  subfolder: "",
  device: "",
  modelDtype: "",
  quantization: "",
  quantBits: "4",
  quantGroupSize: "64",
  cacheDtype: "",
  cacheBits: "3",
  attnMechanism: "unified",
  draftModelDtype: "",
  draftQuantization: "mxfp4",
  draftQuantBits: "4",
  draftQuantGroupSize: "64",
  localFilesOnly: false,
  forceConversion: false,
  replace: true,
  strict: true,
  lazy: false,
  autoConvertHf: "",
  copySupportFiles: true,
  modelKwargsJson: "",
  engineKwargsJson: ""
};

function uid(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function writeJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota / serialization */
  }
}

function applyThemeToDocument(theme: ThemeMode): void {
  const root = document.documentElement;
  const resolved =
    theme === "system"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light"
      : theme;
  root.dataset.theme = resolved;
}

function applyDensityToDocument(density: Density): void {
  document.documentElement.dataset.density = density;
}

function applyPlatformToDocument(): void {
  const ua = navigator.userAgent || "";
  const platform = ua.includes("Mac") ? "darwin" : ua.includes("Win") ? "win32" : "linux";
  document.documentElement.dataset.platform = platform;
}

function runtimeDefaultsFromSchema(schema: ConfigSchema): Record<string, string | boolean> {
  const next: Record<string, string | boolean> = {};
  for (const field of schema.engine) {
    next[field.name] = field.type === "bool" ? Boolean(field.default) : fieldDefaultToString(field);
  }
  return next;
}

function isFieldDirty(field: EngineConfigField, value: string | boolean | undefined): boolean {
  if (field.type === "bool") {
    return Boolean(value) !== Boolean(field.default);
  }
  const current = String(value ?? "");
  const def = fieldDefaultToString(field);
  return current !== def;
}

function profileIdFor(form: LoadFormState): string {
  const source = `${form.servedName.trim() || form.modelId.trim()}|${form.modelId.trim()}|${form.tokenizer.trim()}`;
  return source.toLowerCase().replace(/[^a-z0-9_.-]+/g, "-").replace(/^-+|-+$/g, "") || `profile-${Date.now()}`;
}

function profileNameFor(form: LoadFormState): string {
  return (
    form.servedName.trim() ||
    form.modelId.trim().split(/[\\/]/).filter(Boolean).at(-1) ||
    "model"
  );
}

function quantizationConfig(mode: string, bits: string, groupSize: string): unknown {
  if (!mode) return undefined;
  if (mode !== "affine") return mode;
  return {
    mode: "affine",
    bits: Number.parseInt(bits || "4", 10),
    group_size: Number.parseInt(groupSize || "64", 10)
  };
}

function quantizationSummary(mode: string, bits: string, groupSize: string): string {
  if (!mode) return "none";
  if (mode === "affine") return `affine (${bits || "4"}b · g${groupSize || "64"})`;
  return mode;
}

function buildDraftModelKwargs(form: LoadFormState): Record<string, unknown> {
  const kwargs: Record<string, unknown> = {};
  if (form.draftModelDtype) kwargs.dtype = form.draftModelDtype;
  const quantization = quantizationConfig(form.draftQuantization, form.draftQuantBits, form.draftQuantGroupSize);
  if (quantization !== undefined) kwargs.quantization = quantization;
  return kwargs;
}

function hydrateDraftFormFromEngineKwargs(form: LoadFormState, engineKwargs: Record<string, unknown>): LoadFormState {
  const draftKwargs = engineKwargs.speculative_model_kwargs;
  if (!isPlainObject(draftKwargs)) return form;

  const next = { ...form };
  if (typeof draftKwargs.dtype === "string") {
    next.draftModelDtype = draftKwargs.dtype;
  }

  const quantization = draftKwargs.quantization;
  if (typeof quantization === "string") {
    next.draftQuantization = quantization;
  } else if (isPlainObject(quantization)) {
    const mode = typeof quantization.mode === "string" ? quantization.mode : "";
    next.draftQuantization = mode || "affine";
    if (typeof quantization.bits === "number") next.draftQuantBits = String(quantization.bits);
    if (typeof quantization.group_size === "number") next.draftQuantGroupSize = String(quantization.group_size);
  }
  return next;
}

function createProfile(form: LoadFormState, runtimeValues: Record<string, string | boolean>): LoadProfile {
  return {
    id: profileIdFor(form),
    name: profileNameFor(form),
    updatedAt: Date.now(),
    form: { ...form },
    runtimeValues: { ...runtimeValues }
  };
}

function profileFromModel(model: AppModel): LoadProfile {
  let form: LoadFormState = {
    ...defaultLoadForm,
    modelId: model.model_id,
    servedName: model.served_name,
    tokenizer: model.tokenizer ?? "",
    revision: model.revision ?? "",
    convertedCache: model.converted_cache_dir ?? "",
    modelClass: model.model_class ?? "",
    localFilesOnly: model.local_files_only,
    forceConversion: model.force_conversion,
    modelKwargsJson: Object.keys(model.model_kwargs).length ? JSON.stringify(model.model_kwargs, null, 2) : ""
  };
  form = hydrateDraftFormFromEngineKwargs(form, model.engine_kwargs);
  const runtimeValues: Record<string, string | boolean> = {};
  for (const [key, value] of Object.entries(model.engine_kwargs)) {
    if (typeof value === "boolean") runtimeValues[key] = value;
    else if (value != null) {
      runtimeValues[key] =
        Array.isArray(value)
          ? value.join(",")
          : typeof value === "object"
            ? JSON.stringify(value, null, 2)
            : String(value);
    }
  }
  return {
    id: profileIdFor(form),
    name: profileNameFor(form),
    updatedAt: Math.round((model.loaded_at ?? model.created_at) * 1000),
    form,
    runtimeValues
  };
}

function mergeProfiles(profiles: LoadProfile[], models: AppModel[]): LoadProfile[] {
  let next = profiles;
  for (const model of models) {
    if (model.status !== "ready") continue;
    const profile = profileFromModel(model);
    if (!next.some((item) => item.id === profile.id)) {
      next = [profile, ...next].slice(0, MAX_PROFILES);
    }
  }
  return next;
}

function upsertProfile(profiles: LoadProfile[], profile: LoadProfile): LoadProfile[] {
  return [profile, ...profiles.filter((item) => item.id !== profile.id)].slice(0, MAX_PROFILES);
}

function scrollToBottom(element: HTMLElement | null): void {
  if (!element) return;
  element.scrollTop = element.scrollHeight;
  window.requestAnimationFrame(() => {
    element.scrollTop = element.scrollHeight;
  });
}

function timeAgo(timestamp: number): string {
  const seconds = Math.max(1, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function compactNumber(value: number): string {
  return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function buildSparkPath(points: SparkPoint[], width: number, height: number): string {
  if (points.length < 2) return "";
  const values = points.map((p) => p.v);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const stepX = width / (points.length - 1);
  return points
    .map((p, i) => {
      const x = i * stepX;
      const y = height - ((p.v - min) / range) * height;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function App() {
  const [section, setSection] = useState<SectionId>("chat");
  const [theme, setTheme] = useState<ThemeMode>(() => (readJson<ThemeMode>(STORAGE.theme, "system")));
  const [density, setDensity] = useState<Density>(() => (readJson<Density>(STORAGE.density, "comfortable")));
  const [apiKey, setApiKey] = useState(() => localStorage.getItem(STORAGE.apiKey) ?? "");
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [schema, setSchema] = useState<ConfigSchema | null>(null);
  const [models, setModels] = useState<AppModel[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [runtimeValues, setRuntimeValues] = useState<Record<string, string | boolean>>({});
  const [sampling, setSampling] = useState<SamplingState>(defaultSampling);
  const [enableThinking, setEnableThinking] = useState(true);
  const [stopStrings, setStopStrings] = useState("");
  const [toolsJson, setToolsJson] = useState("");
  const [responseFormatJson, setResponseFormatJson] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>(() =>
    readJson<Conversation[]>(STORAGE.conversations, [])
  );
  const [lastUsage, setLastUsage] = useState<ChatUsage | null>(null);
  const [lastTtft, setLastTtft] = useState<number | null>(null);
  const [streamTimings, setStreamTimings] = useState<{ firstTokenAt: number; finishedAt: number | null } | null>(null);
  const [activeConversationId, setActiveConversationId] = useState<string>(
    () => localStorage.getItem(STORAGE.activeConversation) ?? ""
  );
  const [selectedModel, setSelectedModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [showInspector, setShowInspector] = useState(false);
  const [showHistory, setShowHistory] = useState(true);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [loadForm, setLoadForm] = useState<LoadFormState>(defaultLoadForm);
  const [loadProfiles, setLoadProfiles] = useState<LoadProfile[]>(() =>
    readJson<LoadProfile[]>(STORAGE.profiles, [])
  );
  const [engineCategory, setEngineCategory] = useState("");
  const [engineQuery, setEngineQuery] = useState("");
  const [engineDirtyOnly, setEngineDirtyOnly] = useState(false);
  const [tpsHistory, setTpsHistory] = useState<SparkPoint[]>([]);
  const [reqHistory, setReqHistory] = useState<SparkPoint[]>([]);
  const [tokensHistory, setTokensHistory] = useState<SparkPoint[]>([]);
  const [errorRateHistory, setErrorRateHistory] = useState<SparkPoint[]>([]);
  const [events, setEvents] = useState<EngineEvent[]>([]);
  const [toast, setToast] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const toastTimer = useRef<number | null>(null);
  const previousModelStatuses = useRef<Map<string, ModelStatus>>(new Map());

  const readyModels = useMemo(() => models.filter((m) => m.status === "ready"), [models]);
  const loadingCount = useMemo(() => models.filter((m) => m.status === "loading").length, [models]);
  const errorCount = useMemo(() => models.filter((m) => m.status === "error").length, [models]);
  const activeConversation = conversations.find((c) => c.id === activeConversationId) ?? null;
  const messages = activeConversation?.messages ?? [];

  const showToast = useCallback((message: string) => {
    setToast(message);
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(""), 3800);
  }, []);

  const pushEvent = useCallback((event: Omit<EngineEvent, "id" | "at">) => {
    setEvents((current) => [{ id: uid(), at: Date.now(), ...event }, ...current].slice(0, MAX_EVENTS));
  }, []);

  /* ---------- Platform detection ---------- */
  useEffect(() => {
    applyPlatformToDocument();
  }, []);

  /* ---------- Theme + storage sync ---------- */
  useEffect(() => {
    applyThemeToDocument(theme);
    writeJson(STORAGE.theme, theme);
    if (theme !== "system") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = () => applyThemeToDocument("system");
    media.addEventListener("change", listener);
    return () => media.removeEventListener("change", listener);
  }, [theme]);

  useEffect(() => {
    applyDensityToDocument(density);
    writeJson(STORAGE.density, density);
  }, [density]);

  useEffect(() => {
    localStorage.setItem(STORAGE.apiKey, apiKey.trim());
  }, [apiKey]);

  useEffect(() => {
    writeJson(STORAGE.conversations, conversations);
  }, [conversations]);

  useEffect(() => {
    if (activeConversationId) localStorage.setItem(STORAGE.activeConversation, activeConversationId);
    else localStorage.removeItem(STORAGE.activeConversation);
  }, [activeConversationId]);

  useEffect(() => {
    writeJson(STORAGE.profiles, loadProfiles);
  }, [loadProfiles]);

  /* ---------- Bootstrap (info + schema) ---------- */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [nextInfo, nextSchema] = await Promise.all([getInfo(apiKey), getConfigSchema(apiKey)]);
        if (cancelled) return;
        setInfo(nextInfo);
        setSchema(nextSchema);
        setRuntimeValues(runtimeDefaultsFromSchema(nextSchema));
        if (!engineCategory && nextSchema.engine.length) {
          const firstCategory = nextSchema.engine[0].category;
          setEngineCategory(firstCategory);
        }
      } catch (error) {
        showToast((error as Error).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [apiKey, showToast, engineCategory]);

  /* ---------- Polling ---------- */
  const refreshAll = useCallback(async () => {
    const [modelsResult, healthResult, metricsResult] = await Promise.allSettled([
      getModels(apiKey),
      getHealth(apiKey),
      getMetrics(apiKey)
    ]);
    if (modelsResult.status === "fulfilled") {
      setModels(modelsResult.value.data);
      setLoadProfiles((current) => mergeProfiles(current, modelsResult.value.data));
    }
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    else setHealth(null);
    if (metricsResult.status === "fulfilled") setMetrics(metricsResult.value);
    else setMetrics(null);
  }, [apiKey]);

  useEffect(() => {
    const tick = () => {
      if (!document.hidden) void refreshAll();
    };
    tick();
    const timer = window.setInterval(tick, POLL_INTERVAL_MS);
    document.addEventListener("visibilitychange", tick);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", tick);
    };
  }, [refreshAll]);

  /* ---------- Telemetry sparklines ---------- */
  useEffect(() => {
    if (!metrics) return;
    const now = Date.now();
    const total = metrics.total_requests || 0;
    const failed = metrics.failed_requests || 0;
    const errRate = total ? (failed / total) * 100 : 0;
    setTpsHistory((current) => [...current, { t: now, v: metrics.average_tokens_per_second || 0 }].slice(-SPARK_LENGTH));
    setReqHistory((current) => [...current, { t: now, v: metrics.active_requests || 0 }].slice(-SPARK_LENGTH));
    setTokensHistory((current) => [...current, { t: now, v: metrics.total_tokens_generated || 0 }].slice(-SPARK_LENGTH));
    setErrorRateHistory((current) => [...current, { t: now, v: errRate }].slice(-SPARK_LENGTH));
  }, [metrics]);

  /* ---------- Track model state changes for events ---------- */
  useEffect(() => {
    const previous = previousModelStatuses.current;
    for (const model of models) {
      const prior = previous.get(model.served_name);
      if (prior === model.status) continue;
      if (prior === undefined && model.status === "loading") {
        pushEvent({ tone: "neutral", text: `Loading `, detail: model.served_name });
      } else if (model.status === "ready") {
        pushEvent({ tone: "good", text: `Ready: ${model.served_name}`, detail: model.load_seconds ? `${model.load_seconds}s` : undefined });
      } else if (model.status === "error") {
        pushEvent({ tone: "bad", text: `Error loading ${model.served_name}`, detail: model.error ?? undefined });
      }
      previous.set(model.served_name, model.status);
    }
    for (const name of Array.from(previous.keys())) {
      if (!models.some((m) => m.served_name === name)) {
        previous.delete(name);
        pushEvent({ tone: "warn", text: `Unloaded: ${name}` });
      }
    }
  }, [models, pushEvent]);

  /* ---------- Default selected model ---------- */
  useEffect(() => {
    if (selectedModel && readyModels.some((m) => m.served_name === selectedModel)) return;
    setSelectedModel(readyModels[0]?.served_name ?? "");
  }, [readyModels, selectedModel]);

  /* ---------- Cmd+K ---------- */
  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((open) => !open);
      } else if (event.key === "Escape") {
        setPaletteOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  /* ---------- Engine helpers ---------- */
  const collectRuntimeConfig = useCallback(() => {
    if (!schema) return {};
    const values: Record<string, unknown> = {};
    for (const field of schema.engine) {
      const current = runtimeValues[field.name];
      const value = coerceEngineValue(field, String(current ?? ""), Boolean(current));
      if (value !== undefined) values[field.name] = value;
    }
    return compactObject(values);
  }, [runtimeValues, schema]);

  const resetRuntimeDefaults = useCallback(() => {
    if (!schema) return;
    setRuntimeValues(runtimeDefaultsFromSchema(schema));
    showToast("Runtime defaults restored");
  }, [schema, showToast]);

  /* ---------- Conversation helpers ---------- */
  const ensureConversation = useCallback(
    (model: string): Conversation => {
      if (activeConversation) return activeConversation;
      const fresh: Conversation = {
        id: uid(),
        title: "New chat",
        model: model || "",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: []
      };
      setConversations((current) => [fresh, ...current].slice(0, MAX_CONVERSATIONS));
      setActiveConversationId(fresh.id);
      return fresh;
    },
    [activeConversation]
  );

  const updateConversation = useCallback((id: string, updater: (c: Conversation) => Conversation) => {
    setConversations((current) =>
      current.map((c) => (c.id === id ? { ...updater(c), updatedAt: Date.now() } : c))
    );
  }, []);

  const startNewConversation = useCallback(() => {
    setActiveConversationId("");
    setPrompt("");
  }, []);

  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((current) => current.filter((c) => c.id !== id));
      if (activeConversationId === id) setActiveConversationId("");
    },
    [activeConversationId]
  );

  /* ---------- Load model action ---------- */
  const handleLoadModel = useCallback(async () => {
    if (!loadForm.modelId.trim()) {
      showToast("Model ID or path is required");
      return;
    }
    try {
      const modelKwargs = parseJsonObject(loadForm.modelKwargsJson, "Model kwargs JSON");
      const extraEngineKwargs = parseJsonObject(loadForm.engineKwargsJson, "Extra eSurge kwargs JSON");
      if (loadForm.modelDtype) modelKwargs.dtype = loadForm.modelDtype;
      if (loadForm.configPath.trim()) modelKwargs.config = loadForm.configPath.trim();
      if (loadForm.weightsName.trim()) modelKwargs.weights_name = loadForm.weightsName.trim();
      if (loadForm.subfolder.trim()) modelKwargs.subfolder = loadForm.subfolder.trim();
      if (loadForm.device) modelKwargs.device = loadForm.device;
      modelKwargs.strict = loadForm.strict;
      modelKwargs.lazy = loadForm.lazy;
      modelKwargs.copy_support_files = loadForm.copySupportFiles;
      if (loadForm.autoConvertHf) modelKwargs.auto_convert_hf = loadForm.autoConvertHf === "true";
      if (loadForm.quantization) {
        modelKwargs.quantization = quantizationConfig(loadForm.quantization, loadForm.quantBits, loadForm.quantGroupSize);
      }
      const configOverrides: Record<string, unknown> = {};
      if (loadForm.cacheDtype) {
        configOverrides.cache_dtype = loadForm.cacheDtype;
        if (loadForm.cacheDtype === "turboquant" || loadForm.cacheDtype === "tq") {
          configOverrides.cache_bits = Number.parseInt(loadForm.cacheBits || "3", 10);
        }
      }
      if (loadForm.attnMechanism) {
        configOverrides.attn_mechanism = loadForm.attnMechanism;
      }
      if (Object.keys(configOverrides).length > 0) {
        const existingConfig = modelKwargs.config;
        if (existingConfig && typeof existingConfig === "object" && !Array.isArray(existingConfig)) {
          modelKwargs.config = { ...(existingConfig as Record<string, unknown>), ...configOverrides };
        } else if (typeof existingConfig === "string") {
          showToast("Config path ignored — using attention/cache overrides");
          modelKwargs.config = configOverrides;
        } else {
          modelKwargs.config = configOverrides;
        }
      }
      const engineKwargs = collectRuntimeConfig();
      const draftModel = String(engineKwargs.speculative_model ?? "").trim();
      if (draftModel) {
        const draftModelKwargs = buildDraftModelKwargs(loadForm);
        if (Object.keys(draftModelKwargs).length > 0) {
          const existingDraftKwargs = engineKwargs.speculative_model_kwargs;
          engineKwargs.speculative_model_kwargs = {
            ...(isPlainObject(existingDraftKwargs) ? existingDraftKwargs : {}),
            ...draftModelKwargs
          };
        }
      } else {
        delete engineKwargs.num_speculative_tokens;
        delete engineKwargs.speculative_method;
        delete engineKwargs.speculative_model_kwargs;
      }
      const payload: LoadModelRequest = {
        model_id: loadForm.modelId.trim(),
        served_name: nullableText(loadForm.servedName),
        tokenizer: nullableText(loadForm.tokenizer),
        revision: nullableText(loadForm.revision),
        converted_cache_dir: nullableText(loadForm.convertedCache),
        local_files_only: loadForm.localFilesOnly,
        force_conversion: loadForm.forceConversion,
        model_class: nullableText(loadForm.modelClass),
        replace: loadForm.replace,
        model_kwargs: modelKwargs,
        engine_kwargs: { ...engineKwargs, ...extraEngineKwargs }
      };
      await loadModel(apiKey, payload);
      setLoadProfiles((current) => upsertProfile(current, createProfile(loadForm, runtimeValues)));
      showToast(`Loading ${profileNameFor(loadForm)}…`);
      setDrawerOpen(false);
      setStepIndex(0);
      setLoadForm(defaultLoadForm);
      setSection("models");
      await refreshAll();
    } catch (error) {
      showToast((error as Error).message);
    }
  }, [apiKey, collectRuntimeConfig, loadForm, refreshAll, runtimeValues, showToast]);

  const handleUnloadModel = useCallback(
    async (servedName: string) => {
      try {
        await unloadModel(apiKey, servedName);
        showToast(`${servedName} unloaded`);
        await refreshAll();
      } catch (error) {
        showToast((error as Error).message);
      }
    },
    [apiKey, refreshAll, showToast]
  );

  /* ---------- Chat send ---------- */
  const handleSend = useCallback(async () => {
    let text = prompt.trim();
    if (!text) return;
    if (isSending) return;

    if (text.startsWith("/")) {
      const handled = await handleSlash(text);
      if (handled) {
        setPrompt("");
        return;
      }
    }

    if (!selectedModel) {
      showToast("Load a ready model first");
      return;
    }
    text = prompt.trim();

    const conversation = ensureConversation(selectedModel);
    const userMessage: ChatMessage = { role: "user", content: text };
    const draftMessages: ChatMessage[] = [...conversation.messages, userMessage, { role: "assistant", content: "", reasoning: "" }];
    const conversationId = conversation.id;

    updateConversation(conversationId, (c) => ({
      ...c,
      title: c.title === "New chat" || c.messages.length === 0 ? text.slice(0, 60) : c.title,
      model: selectedModel,
      messages: draftMessages
    }));

    let request: ChatRequest;
    try {
      request = buildChatRequest(
        selectedModel,
        draftMessages.slice(0, -1),
        sampling,
        stopStrings,
        toolsJson,
        responseFormatJson,
        enableThinking
      );
    } catch (error) {
      showToast((error as Error).message);
      return;
    }

    setPrompt("");
    setIsSending(true);
    setLastUsage(null);
    setLastTtft(null);
    setStreamTimings(null);
    const requestStartedAt = Date.now();
    let firstTokenAt: number | null = null;
    const controller = new AbortController();
    abortRef.current = controller;

    const appendDelta = (content: string, reasoning: string) => {
      if ((content || reasoning) && firstTokenAt === null) {
        firstTokenAt = Date.now();
        setLastTtft((firstTokenAt - requestStartedAt) / 1000);
        setStreamTimings({ firstTokenAt, finishedAt: null });
      }
      updateConversation(conversationId, (c) => {
        const next = [...c.messages];
        const last = next[next.length - 1];
        if (!last || last.role !== "assistant") return c;
        next[next.length - 1] = {
          ...last,
          content: `${last.content}${content}`,
          reasoning: `${last.reasoning ?? ""}${reasoning}`
        };
        return { ...c, messages: next };
      });
    };

    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders(apiKey) },
        body: JSON.stringify(request),
        signal: controller.signal
      });
      if (!response.ok || !response.body) throw new Error(await responseMessage(response));
      await readChatStream(response, appendDelta, (usage) => setLastUsage(usage));
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        appendDelta(`\n\n_${(error as Error).message}_`, "");
      }
    } finally {
      abortRef.current = null;
      setIsSending(false);
      setStreamTimings((current) =>
        current ? { ...current, finishedAt: Date.now() } : null
      );
      void refreshAll();
    }
  }, [
    apiKey,
    ensureConversation,
    isSending,
    prompt,
    refreshAll,
    responseFormatJson,
    sampling,
    enableThinking,
    selectedModel,
    showToast,
    stopStrings,
    toolsJson,
    updateConversation
  ]);

  const handleSlash = useCallback(
    async (text: string): Promise<boolean> => {
      const [head, ...rest] = text.trim().split(/\s+/);
      const arg = rest.join(" ").trim();
      switch (head) {
        case "/clear":
          if (activeConversation) {
            updateConversation(activeConversation.id, (c) => ({ ...c, messages: [] }));
          }
          showToast("Chat cleared");
          return true;
        case "/new":
          startNewConversation();
          showToast("New chat");
          return true;
        case "/model":
          if (arg) {
            const matched = readyModels.find((m) => m.served_name.toLowerCase().startsWith(arg.toLowerCase()));
            if (matched) {
              setSelectedModel(matched.served_name);
              showToast(`Switched to ${matched.served_name}`);
              return true;
            }
            showToast(`No ready model matching "${arg}"`);
            return true;
          }
          return false;
        case "/temp": {
          const value = Number.parseFloat(arg);
          if (Number.isFinite(value)) {
            setSampling((s) => ({ ...s, temperature: value }));
            showToast(`Temperature: ${value}`);
            return true;
          }
          showToast("/temp expects a number");
          return true;
        }
        case "/max": {
          const value = Number.parseInt(arg, 10);
          if (Number.isFinite(value) && value > 0) {
            setSampling((s) => ({ ...s, max_tokens: value }));
            showToast(`max_tokens: ${value}`);
            return true;
          }
          showToast("/max expects a positive integer");
          return true;
        }
        case "/stop":
          setStopStrings(arg);
          showToast(arg ? `Stops: ${arg}` : "Stops cleared");
          return true;
        case "/system": {
          if (!arg) {
            showToast("/system needs a message");
            return true;
          }
          const conversation = ensureConversation(selectedModel);
          updateConversation(conversation.id, (c) => ({
            ...c,
            messages: [...c.messages, { role: "system", content: arg }]
          }));
          showToast("System message added");
          return true;
        }
        default:
          return false;
      }
    },
    [activeConversation, ensureConversation, readyModels, selectedModel, showToast, startNewConversation, updateConversation]
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const onSelectPreset = useCallback((body: string) => {
    setPrompt(body);
  }, []);

  /* ---------- Palette items ---------- */
  const paletteItems = useMemo(() => {
    const sections: SectionId[] = ["chat", "models", "engine", "telemetry", "settings"];
    const items: Array<{ id: string; group: string; label: string; hint?: string; icon: ReactNode; run: () => void }> = [];
    for (const id of sections) {
      items.push({
        id: `nav-${id}`,
        group: "Navigate",
        label: `Go to ${id[0].toUpperCase() + id.slice(1)}`,
        icon: sectionIcon(id),
        run: () => setSection(id)
      });
    }
    items.push({
      id: "action-new-chat",
      group: "Actions",
      label: "New chat",
      icon: <MessageSquarePlus size={14} />,
      run: () => {
        startNewConversation();
        setSection("chat");
      }
    });
    items.push({
      id: "action-add-model",
      group: "Actions",
      label: "Add model",
      icon: <Plus size={14} />,
      run: () => {
        setSection("models");
        setStepIndex(0);
        setDrawerOpen(true);
      }
    });
    items.push({
      id: "action-reset-runtime",
      group: "Actions",
      label: "Reset engine to defaults",
      icon: <RotateCcw size={14} />,
      run: resetRuntimeDefaults
    });
    items.push({
      id: "action-toggle-theme",
      group: "Actions",
      label: theme === "dark" ? "Switch to light" : theme === "light" ? "Switch to system" : "Switch to dark",
      icon: theme === "dark" ? <Sun size={14} /> : theme === "light" ? <Monitor size={14} /> : <Moon size={14} />,
      run: () => setTheme(theme === "dark" ? "light" : theme === "light" ? "system" : "dark")
    });
    for (const model of readyModels) {
      items.push({
        id: `model-${model.served_name}`,
        group: "Switch model",
        label: model.served_name,
        hint: model.model_id,
        icon: <Bot size={14} />,
        run: () => {
          setSelectedModel(model.served_name);
          setSection("chat");
        }
      });
    }
    for (const profile of loadProfiles.slice(0, 6)) {
      items.push({
        id: `profile-${profile.id}`,
        group: "Load profile",
        label: profile.name,
        hint: profile.form.modelId,
        icon: <Package size={14} />,
        run: () => {
          setLoadForm({ ...defaultLoadForm, ...profile.form });
          setRuntimeValues((current) => ({
            ...(schema ? runtimeDefaultsFromSchema(schema) : current),
            ...profile.runtimeValues
          }));
          setStepIndex(4);
          setSection("models");
          setDrawerOpen(true);
        }
      });
    }
    return items;
  }, [loadProfiles, readyModels, resetRuntimeDefaults, schema, startNewConversation, theme]);

  /* ---------- Render ---------- */
  const activeReady = readyModels.find((m) => m.served_name === selectedModel) ?? readyModels[0] ?? null;
  const railBadgeForModels =
    loadingCount > 0
      ? loadingCount.toString()
      : errorCount > 0
        ? errorCount.toString()
        : undefined;

  return (
    <div className="app-shell">
      <div className="titlebar-drag" />
      <div className="shell-body">
      <aside className="rail">
        <RailButton id="chat" label="Chat" active={section === "chat"} onClick={() => setSection("chat")} />
        <RailButton
          id="models"
          label="Models"
          active={section === "models"}
          onClick={() => setSection("models")}
          badge={railBadgeForModels}
        />
        <RailButton id="engine" label="Engine" active={section === "engine"} onClick={() => setSection("engine")} />
        <RailButton
          id="telemetry"
          label="Telemetry"
          active={section === "telemetry"}
          onClick={() => setSection("telemetry")}
        />
        <div className="rail-spacer" />
        <RailButton id="settings" label="Settings" active={section === "settings"} onClick={() => setSection("settings")} />
      </aside>

      <main className="canvas">
        {section === "chat" && (
          <ChatSection
            conversations={conversations}
            activeId={activeConversationId}
            onActiveId={setActiveConversationId}
            onDelete={deleteConversation}
            onNew={startNewConversation}
            showHistory={showHistory}
            onToggleHistory={() => setShowHistory((s) => !s)}
            messages={messages}
            readyModels={readyModels}
            selectedModel={selectedModel}
            onSelectedModel={setSelectedModel}
            activeReady={activeReady}
            metrics={metrics}
            usage={lastUsage}
            ttft={lastTtft}
            streamTimings={streamTimings}
            prompt={prompt}
            onPrompt={setPrompt}
            isSending={isSending}
            onSend={handleSend}
            onStop={stopStreaming}
            onPreset={onSelectPreset}
            sampling={sampling}
            onSampling={setSampling}
            enableThinking={enableThinking}
            onEnableThinking={setEnableThinking}
            stopStringsValue={stopStrings}
            onStopStrings={setStopStrings}
            toolsJson={toolsJson}
            onToolsJson={setToolsJson}
            responseFormatJson={responseFormatJson}
            onResponseFormatJson={setResponseFormatJson}
            showInspector={showInspector}
            onToggleInspector={() => setShowInspector((s) => !s)}
            onCopyMessage={(text) => {
              void navigator.clipboard?.writeText(text);
              showToast("Copied");
            }}
          />
        )}

        {section === "models" && (
          <ModelsSection
            models={models}
            metrics={metrics}
            tpsHistory={tpsHistory}
            profiles={loadProfiles}
            onUnload={handleUnloadModel}
            onAdd={() => {
              setLoadForm(defaultLoadForm);
              setStepIndex(0);
              setDrawerOpen(true);
            }}
            onApplyProfile={(profile) => {
              setLoadForm({ ...defaultLoadForm, ...profile.form });
              setRuntimeValues((current) => ({
                ...(schema ? runtimeDefaultsFromSchema(schema) : current),
                ...profile.runtimeValues
              }));
              setStepIndex(4);
              setDrawerOpen(true);
            }}
            onForgetProfile={(profileId) => {
              setLoadProfiles((current) => current.filter((p) => p.id !== profileId));
              showToast("Profile removed");
            }}
            onRefresh={refreshAll}
          />
        )}

        {section === "engine" && schema && (
          <EngineSection
            schema={schema}
            values={runtimeValues}
            onValues={setRuntimeValues}
            onReset={resetRuntimeDefaults}
            category={engineCategory}
            onCategory={setEngineCategory}
            query={engineQuery}
            onQuery={setEngineQuery}
            dirtyOnly={engineDirtyOnly}
            onDirtyOnly={setEngineDirtyOnly}
          />
        )}

        {section === "telemetry" && (
          <TelemetrySection
            health={health}
            metrics={metrics}
            models={models}
            events={events}
            tpsHistory={tpsHistory}
            reqHistory={reqHistory}
            tokensHistory={tokensHistory}
            errorRateHistory={errorRateHistory}
          />
        )}

        {section === "settings" && (
          <SettingsSection
            theme={theme}
            onTheme={setTheme}
            density={density}
            onDensity={setDensity}
            apiKey={apiKey}
            onApiKey={setApiKey}
            info={info}
            onResetRuntime={resetRuntimeDefaults}
            onClearConversations={() => {
              setConversations([]);
              setActiveConversationId("");
              showToast("Conversations cleared");
            }}
            conversationsCount={conversations.length}
            profilesCount={loadProfiles.length}
            onClearProfiles={() => {
              setLoadProfiles([]);
              showToast("Profiles cleared");
            }}
          />
        )}
      </main>
      </div>

      <footer className="status-bar">
        <span>
          <span className={`dot ${health ? "online" : "offline"}`} />
          {health ? "connected" : "offline"}
        </span>
        <span className="sep" />
        <span>
          <strong>{activeReady?.served_name ?? "no model"}</strong>
          {" "}· {readyModels.length} ready
        </span>
        <span className="sep" />
        <span>{formatNumber(metrics?.average_tokens_per_second)} tps</span>
        <span className="sep" />
        <span>{metrics?.active_requests ?? 0} active</span>
        <span className="sep" />
        <span>uptime {formatDuration(metrics?.uptime_seconds ?? health?.uptime_seconds ?? 0)}</span>
        <span className="k-hint">
          press <kbd>⌘K</kbd> for commands
        </span>
      </footer>

      <LoadDrawer
        open={drawerOpen}
        stepIndex={stepIndex}
        onStepIndex={setStepIndex}
        onClose={() => setDrawerOpen(false)}
        form={loadForm}
        onForm={setLoadForm}
        engineFields={schema?.engine ?? []}
        runtimeValues={runtimeValues}
        onRuntimeValues={setRuntimeValues}
        onSubmit={handleLoadModel}
      />

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        items={paletteItems}
      />

      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}

/* ============================================================ */
/* Rail                                                         */
/* ============================================================ */

function sectionIcon(id: SectionId) {
  switch (id) {
    case "chat":
      return <Bot size={14} />;
    case "models":
      return <Package size={14} />;
    case "engine":
      return <Cpu size={14} />;
    case "telemetry":
      return <Gauge size={14} />;
    case "settings":
      return <Settings size={14} />;
  }
}

function RailButton({
  id,
  label,
  active,
  onClick,
  badge
}: {
  id: SectionId;
  label: string;
  active: boolean;
  onClick: () => void;
  badge?: string;
}) {
  const props: { className: string; onClick: () => void; type: "button"; "data-badge"?: string } = {
    className: active ? "rail-button active" : "rail-button",
    onClick,
    type: "button"
  };
  if (badge) props["data-badge"] = badge;
  return (
    <button {...props}>
      {sectionIconLarge(id)}
      <span className="rail-tooltip">{label}</span>
    </button>
  );
}

function sectionIconLarge(id: SectionId) {
  const size = 18;
  switch (id) {
    case "chat":
      return <Bot size={size} />;
    case "models":
      return <Package size={size} />;
    case "engine":
      return <Cpu size={size} />;
    case "telemetry":
      return <Gauge size={size} />;
    case "settings":
      return <Settings size={size} />;
  }
}

/* ============================================================ */
/* Chat                                                         */
/* ============================================================ */

function numberFromEngineValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function speculativeChatStatus(model: AppModel | null, sampling: SamplingState): { label: string; title: string; tone: "good" | "warn" } | null {
  if (!model) return null;
  const speculativeModel = String(model.engine_kwargs.speculative_model ?? "").trim();
  const tokens = numberFromEngineValue(model.engine_kwargs.num_speculative_tokens) ?? 0;
  if (!speculativeModel || tokens <= 0) return null;

  const method =
    typeof model.engine_kwargs.speculative_method === "string"
      ? model.engine_kwargs.speculative_method
      : speculativeModel.toLowerCase().includes("dflash")
        ? "dflash"
        : "draft";

  if (sampling.temperature > 0) {
    return {
      label: `${method} off`,
      title: `Speculative decoding is disabled because Temp is ${sampling.temperature}. Set Temp to 0 for exact greedy speculation.`,
      tone: "warn"
    };
  }

  return {
    label: `${method}:${Math.trunc(tokens)}`,
    title: `${method} speculative decoding is eligible for streamed greedy chat.`,
    tone: "good"
  };
}

function ChatSection(props: {
  conversations: Conversation[];
  activeId: string;
  onActiveId: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
  showHistory: boolean;
  onToggleHistory: () => void;
  messages: ChatMessage[];
  readyModels: AppModel[];
  selectedModel: string;
  onSelectedModel: (value: string) => void;
  activeReady: AppModel | null;
  metrics: Metrics | null;
  usage: ChatUsage | null;
  ttft: number | null;
  streamTimings: { firstTokenAt: number; finishedAt: number | null } | null;
  prompt: string;
  onPrompt: (value: string) => void;
  isSending: boolean;
  onSend: () => void;
  onStop: () => void;
  onPreset: (body: string) => void;
  sampling: SamplingState;
  onSampling: Dispatch<SetStateAction<SamplingState>>;
  enableThinking: boolean;
  onEnableThinking: (value: boolean) => void;
  stopStringsValue: string;
  onStopStrings: (value: string) => void;
  toolsJson: string;
  onToolsJson: (value: string) => void;
  responseFormatJson: string;
  onResponseFormatJson: (value: string) => void;
  showInspector: boolean;
  onToggleInspector: () => void;
  onCopyMessage: (text: string) => void;
}) {
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const stickToBottomRef = useRef(true);
  const lastMessage = props.messages.at(-1);
  const [tickNow, setTickNow] = useState(() => Date.now());

  useEffect(() => {
    if (!props.isSending) return;
    const timer = window.setInterval(() => setTickNow(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, [props.isSending]);

  useEffect(() => {
    const el = messagesRef.current;
    if (!el) return;
    const onScroll = () => {
      const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      stickToBottomRef.current = distanceFromBottom < 80;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useLayoutEffect(() => {
    if (stickToBottomRef.current) {
      scrollToBottom(messagesRef.current);
    }
  }, [props.messages.length, lastMessage?.content, lastMessage?.reasoning]);

  useEffect(() => {
    if (props.isSending) {
      stickToBottomRef.current = true;
      scrollToBottom(messagesRef.current);
    }
  }, [props.isSending]);

  const completionTokens = props.usage?.completion_tokens ?? null;
  const promptTokens = props.usage?.prompt_tokens ?? null;
  const totalTokens =
    props.usage?.total_tokens ??
    ((promptTokens ?? 0) + (completionTokens ?? 0) || null);

  let liveTps: number | null = null;
  if (props.streamTimings) {
    const endAt = props.streamTimings.finishedAt ?? tickNow;
    const elapsed = (endAt - props.streamTimings.firstTokenAt) / 1000;
    if (elapsed > 0.2) {
      if (completionTokens) {
        liveTps = completionTokens / elapsed;
      } else {
        const generated =
          (lastMessage?.content?.length ?? 0) + (lastMessage?.reasoning?.length ?? 0);
        liveTps = generated / 4 / elapsed;
      }
    }
  }

  const tpsDisplay =
    liveTps !== null
      ? `${liveTps.toFixed(1)} tps`
      : props.metrics?.average_tokens_per_second
        ? `${props.metrics.average_tokens_per_second.toFixed(1)} tps`
        : null;

  const ttftDisplay =
    props.ttft !== null
      ? props.ttft < 1
        ? `${Math.round(props.ttft * 1000)} ms TTFT`
        : `${props.ttft.toFixed(2)} s TTFT`
      : null;

  const maxLen =
    (props.activeReady?.engine_kwargs.max_model_len as number | undefined) ?? null;

  let usedTokens: number;
  let isExactTokens = false;
  if (totalTokens) {
    usedTokens = totalTokens;
    isExactTokens = true;
  } else {
    const totalChars = props.messages.reduce(
      (sum, m) => sum + m.content.length + (m.reasoning?.length ?? 0),
      0
    );
    usedTokens = Math.round(totalChars / 4);
  }
  const contextDisplay = maxLen
    ? `${compactNumber(usedTokens)} / ${compactNumber(maxLen)} ctx`
    : null;
  const contextTitle = maxLen
    ? `${isExactTokens ? "" : "~"}${usedTokens.toLocaleString()} of ${maxLen.toLocaleString()} tokens${
        promptTokens !== null && completionTokens !== null
          ? ` (prompt ${promptTokens.toLocaleString()}, completion ${completionTokens.toLocaleString()})`
          : ""
      }`
    : "";
  const contextRatio = maxLen ? usedTokens / maxLen : 0;
  const contextTone =
    contextRatio > 0.9 ? "danger" : contextRatio > 0.7 ? "warn" : "neutral";
  const speculativeStatus = speculativeChatStatus(props.activeReady, props.sampling);

  const trimmedPrompt = props.prompt.trimStart();
  const showSlashMenu = trimmedPrompt.startsWith("/") && !trimmedPrompt.includes("\n");
  const slashFilter = trimmedPrompt.split(/\s/)[0];
  const filteredSlash = SLASH_COMMANDS.filter((cmd) => cmd.cmd.startsWith(slashFilter));

  const onTextareaKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      props.onSend();
    }
  };

  return (
    <div className={`chat-layout ${props.showHistory ? "" : "history-hidden"}`}>
      {props.showHistory && (
        <aside className="chat-history">
          <div className="chat-history-head">
            <strong>Conversations</strong>
            <button className="btn-icon" onClick={props.onNew} title="New chat" type="button">
              <MessageSquarePlus size={15} />
            </button>
          </div>
          <div className="chat-history-list">
            {props.conversations.length === 0 && (
              <div style={{ padding: "20px 12px", color: "var(--text-quaternary)", fontSize: 12, textAlign: "center" }}>
                No conversations yet
              </div>
            )}
            {props.conversations.map((conversation) => (
              <button
                key={conversation.id}
                className={`chat-history-item ${conversation.id === props.activeId ? "active" : ""}`}
                onClick={() => props.onActiveId(conversation.id)}
                type="button"
              >
                <span className="chat-history-title">{conversation.title || "Untitled"}</span>
                <span className="chat-history-meta">
                  {conversation.model || "—"} · {timeAgo(conversation.updatedAt)}
                </span>
                <span
                  className="chat-history-delete"
                  role="button"
                  tabIndex={-1}
                  onClick={(event) => {
                    event.stopPropagation();
                    props.onDelete(conversation.id);
                  }}
                  title="Delete conversation"
                >
                  <Trash2 size={13} />
                </span>
              </button>
            ))}
          </div>
        </aside>
      )}

      <section className="chat-main">
        <div className="chat-topbar">
          <button className="btn-icon" onClick={props.onToggleHistory} type="button" title={props.showHistory ? "Hide history" : "Show history"}>
            {props.showHistory ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
          </button>
          <select
            value={props.selectedModel}
            onChange={(event) => props.onSelectedModel(event.target.value)}
          >
            {props.readyModels.length ? (
              props.readyModels.map((model) => (
                <option key={model.served_name} value={model.served_name}>
                  {model.served_name}
                </option>
              ))
            ) : (
              <option value="">No ready model</option>
            )}
          </select>
          <span className="spacer" />
          {ttftDisplay && (
            <span className="metric-pill" title="Time to first token (last response)">
              {ttftDisplay}
            </span>
          )}
          {contextDisplay && (
            <span className={`metric-pill tone-${contextTone}`} title={contextTitle}>
              {contextDisplay}
            </span>
          )}
          {speculativeStatus && (
            <span className={`metric-pill tone-${speculativeStatus.tone}`} title={speculativeStatus.title}>
              <Zap size={12} />
              {speculativeStatus.label}
            </span>
          )}
          {tpsDisplay && (
            <span className={`metric-pill tps ${props.isSending ? "live" : ""}`}>
              <span className="tps-dot" />
              {tpsDisplay}
            </span>
          )}
          <button className="btn btn-ghost" onClick={props.onToggleInspector} type="button" title="Sampling & tools">
            <SlidersHorizontal size={14} /> Tune
          </button>
        </div>

        {props.activeReady?.has_chat_template === false && (
          <div className="chat-notice">
            <AlertTriangle size={14} />
            <span>
              No chat template detected. Set Tokenizer to the original HF repo so the model gets formatted prompts.
            </span>
          </div>
        )}
        {speculativeStatus && props.enableThinking && (
          <div className="chat-notice">
            <AlertTriangle size={14} />
            <span>
              Speculative decoding is fastest with Think off. Turn off Think in Tune before benchmarking draft speed.
            </span>
          </div>
        )}

        <div className="messages" ref={messagesRef}>
          {props.messages.length === 0 ? (
            <div className="chat-empty">
              <Bot size={32} />
              <h2>Start a conversation</h2>
              <span style={{ fontSize: 13 }}>
                {props.activeReady ? `Chat with ${props.activeReady.served_name}` : "Load a model to begin"}
              </span>
              <div className="empty-presets">
                {PRESETS.map((preset) => (
                  <button className="empty-preset" key={preset.title} onClick={() => props.onPreset(preset.body)} type="button">
                    <strong>{preset.title}</strong>
                    {preset.body.split("\n")[0].slice(0, 80)}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages-inner">
              {props.messages.map((message, index) => (
                <MessageRow
                  key={`${message.role}-${index}`}
                  message={message}
                  onCopy={() => props.onCopyMessage(message.content)}
                />
              ))}
            </div>
          )}
        </div>

        <div className="composer-wrap">
          <div className="composer">
            <textarea
              ref={composerRef}
              value={props.prompt}
              onChange={(event) => props.onPrompt(event.target.value)}
              onKeyDown={onTextareaKey}
              placeholder={props.activeReady ? "Message EasyMLX  ·  / for commands  ·  ⏎ to send" : "Load a model first"}
              rows={1}
            />
            <div className="composer-bar">
              <span className="hint">
                {props.activeReady
                  ? `${props.activeReady.served_name} · t=${props.sampling.temperature}${tpsDisplay ? ` · ${tpsDisplay}` : ""}`
                  : "no model"}
              </span>
              <span className="spacer" />
              {props.isSending ? (
                <button className="btn btn-secondary" onClick={props.onStop} type="button">
                  <Square size={13} /> Stop
                </button>
              ) : (
                <button
                  className="btn btn-primary"
                  onClick={props.onSend}
                  type="button"
                  disabled={!props.prompt.trim()}
                >
                  <Send size={13} /> Send
                </button>
              )}
            </div>
            {showSlashMenu && filteredSlash.length > 0 && (
              <div className="slash-menu">
                {filteredSlash.map((cmd) => (
                  <button
                    className="slash-item"
                    key={cmd.cmd}
                    type="button"
                    onClick={() => props.onPrompt(`${cmd.cmd} `)}
                  >
                    <span className="cmd">{cmd.cmd}</span>
                    <span className="desc">{cmd.desc}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className={props.showInspector ? "inspector-panel open" : "inspector-panel"}>
          <header>
            <h3>Tune</h3>
            <button className="btn-icon" onClick={props.onToggleInspector} type="button">
              <X size={15} />
            </button>
          </header>
          <div className="inspector-panel-body">
            <div className="inspector-section">
              <h4>Sampling</h4>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={props.enableThinking}
                  onChange={(event) => props.onEnableThinking(event.target.checked)}
                />
                <span className="toggle-switch" />
                <span>Think</span>
              </label>
              <div className="sampling-grid">
                <NumberField label="Max" value={props.sampling.max_tokens} min={1} step={1} onChange={(v) => props.onSampling((s) => ({ ...s, max_tokens: v }))} />
                <NumberField label="Temp" value={props.sampling.temperature} min={0} max={2} step={0.05} onChange={(v) => props.onSampling((s) => ({ ...s, temperature: v }))} />
                <NumberField label="Top P" value={props.sampling.top_p} min={0} max={1} step={0.01} onChange={(v) => props.onSampling((s) => ({ ...s, top_p: v }))} />
                <NumberField label="Top K" value={props.sampling.top_k} min={0} step={1} onChange={(v) => props.onSampling((s) => ({ ...s, top_k: v }))} />
                <NumberField label="Presence" value={props.sampling.presence_penalty} step={0.05} onChange={(v) => props.onSampling((s) => ({ ...s, presence_penalty: v }))} />
                <NumberField label="Repeat" value={props.sampling.repetition_penalty} step={0.05} onChange={(v) => props.onSampling((s) => ({ ...s, repetition_penalty: v }))} />
              </div>
            </div>
            <div className="inspector-section">
              <h4>Stops</h4>
              <label className="field">
                <span>Comma separated</span>
                <input value={props.stopStringsValue} onChange={(event) => props.onStopStrings(event.target.value)} placeholder="###,</s>" />
              </label>
            </div>
            <div className="inspector-section">
              <h4>Tools JSON</h4>
              <label className="field">
                <span>OpenAI tools</span>
                <textarea
                  value={props.toolsJson}
                  onChange={(event) => props.onToolsJson(event.target.value)}
                  spellCheck={false}
                  placeholder='[{"type":"function","function":{"name":"lookup"}}]'
                />
              </label>
            </div>
            <div className="inspector-section">
              <h4>Response format</h4>
              <label className="field">
                <span>JSON</span>
                <textarea
                  value={props.responseFormatJson}
                  onChange={(event) => props.onResponseFormatJson(event.target.value)}
                  spellCheck={false}
                  placeholder='{"type":"json_object"}'
                />
              </label>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function MessageRow({ message, onCopy }: { message: ChatMessage; onCopy: () => void }) {
  const [showReasoning, setShowReasoning] = useState(false);
  const reasoningRef = useRef<HTMLDivElement | null>(null);
  useLayoutEffect(() => {
    if (showReasoning) scrollToBottom(reasoningRef.current);
  }, [message.reasoning, showReasoning]);

  return (
    <article className={`message ${message.role}`}>
      {message.reasoning && (
        <div className="reasoning-block">
          <button className="reasoning-toggle" type="button" onClick={() => setShowReasoning((s) => !s)}>
            <BrainCircuit size={12} />
            <span>Thinking</span>
            <ChevronRight size={11} style={{ marginLeft: "auto", transform: showReasoning ? "rotate(90deg)" : "none", transition: "transform 0.15s" }} />
          </button>
          {showReasoning && (
            <div className="reasoning-content" ref={reasoningRef}>
              {message.reasoning}
            </div>
          )}
        </div>
      )}
      {message.content && (
        <div className="message-bubble">
          {message.role === "assistant" ? (
            <div className="markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>
          ) : (
            message.content
          )}
        </div>
      )}
      {message.role === "assistant" && message.content && (
        <div className="message-actions">
          <button onClick={onCopy} type="button" title="Copy">
            <Copy size={12} /> Copy
          </button>
        </div>
      )}
    </article>
  );
}

/* ============================================================ */
/* Models                                                       */
/* ============================================================ */

function ModelsSection(props: {
  models: AppModel[];
  metrics: Metrics | null;
  tpsHistory: SparkPoint[];
  profiles: LoadProfile[];
  onUnload: (servedName: string) => void;
  onAdd: () => void;
  onApplyProfile: (profile: LoadProfile) => void;
  onForgetProfile: (profileId: string) => void;
  onRefresh: () => void;
}) {
  const ready = props.models.filter((m) => m.status === "ready");
  const loading = props.models.filter((m) => m.status === "loading");
  const errored = props.models.filter((m) => m.status === "error");

  return (
    <>
      <header className="panel-top">
        <div>
          <h1>Models</h1>
          <p>{ready.length} ready · {loading.length} loading · {errored.length} errored</p>
        </div>
        <div className="toolbar">
          <button className="btn btn-icon" onClick={props.onRefresh} title="Refresh" type="button">
            <RefreshCw size={14} />
          </button>
          <button className="btn btn-primary" onClick={props.onAdd} type="button">
            <Plus size={14} /> Add model
          </button>
        </div>
      </header>
      <div className="panel-body">
        {props.profiles.length > 0 && (
          <div className="profiles-strip">
            <span style={{ fontSize: 11, color: "var(--text-tertiary)", textTransform: "uppercase", letterSpacing: "0.05em", marginRight: 4 }}>
              Profiles
            </span>
            {props.profiles.slice(0, 10).map((profile) => (
              <span className="profile-chip" key={profile.id}>
                <button
                  type="button"
                  onClick={() => props.onApplyProfile(profile)}
                  style={{ color: "inherit", fontSize: "inherit", fontFamily: "inherit", padding: 0, display: "inline-flex", alignItems: "center", gap: 6 }}
                  title={profile.form.modelId}
                >
                  <Package size={11} /> {profile.name}
                </button>
                <button className="profile-chip-remove" type="button" onClick={() => props.onForgetProfile(profile.id)} title="Forget">
                  <X size={10} />
                </button>
              </span>
            ))}
          </div>
        )}

        <div className="models-grid">
          {props.models.map((model) => (
            <ModelCard
              key={model.served_name}
              model={model}
              tps={props.metrics?.average_tokens_per_second ?? 0}
              onUnload={() => props.onUnload(model.served_name)}
            />
          ))}
          <button className="add-model-card" onClick={props.onAdd} type="button">
            <Plus size={20} />
            <strong>Add model</strong>
            <span>5-step setup</span>
          </button>
        </div>
      </div>
    </>
  );
}

function ModelCard({ model, tps, onUnload }: { model: AppModel; tps: number; onUnload: () => void }) {
  const tags: string[] = [];
  const dtype = model.model_kwargs.dtype;
  if (typeof dtype === "string") tags.push(dtype);
  const quant = model.model_kwargs.quantization;
  if (typeof quant === "string") tags.push(quant);
  else if (quant && typeof quant === "object" && "mode" in quant) tags.push(String((quant as Record<string, unknown>).mode));
  if (model.has_chat_template) tags.push("chat-template");
  const config = model.model_kwargs.config;
  if (config && typeof config === "object" && "attn_mechanism" in config) {
    tags.push(String((config as Record<string, unknown>).attn_mechanism));
  }
  if (model.engine_kwargs.speculative_model) {
    const speculativeModel = String(model.engine_kwargs.speculative_model).toLowerCase();
    const method =
      typeof model.engine_kwargs.speculative_method === "string"
        ? model.engine_kwargs.speculative_method
        : speculativeModel.includes("dflash")
          ? "dflash"
          : "draft";
    const tokens = model.engine_kwargs.num_speculative_tokens;
    tags.push(`${method}${typeof tokens === "number" && tokens > 0 ? `:${tokens}` : ""}`);
  }

  const memTarget = (model.engine_kwargs.hbm_utilization as number | undefined) ?? 0;
  const ctx = (model.engine_kwargs.max_model_len as number | undefined) ?? 0;

  return (
    <article className="model-card-lg">
      <div className="model-card-head">
        <div style={{ minWidth: 0 }}>
          <strong>{model.served_name}</strong>
          <p title={model.model_id}>{model.model_id}</p>
        </div>
        <span className={`pill ${model.status}`}>{model.status}</span>
      </div>
      <div className="model-card-stats">
        <div className="stat-block">
          <span>TPS</span>
          <strong>{model.status === "ready" ? formatNumber(tps, 1) : "—"}</strong>
        </div>
        <div className="stat-block">
          <span>Context</span>
          <strong>{ctx ? compactNumber(ctx) : "—"}</strong>
        </div>
        <div className="stat-block">
          <span>HBM</span>
          <strong>{memTarget ? `${Math.round(memTarget * 100)}%` : "—"}</strong>
        </div>
      </div>
      {model.error && (
        <div style={{ fontSize: 12, color: "var(--danger)", padding: "6px 8px", background: "var(--danger-faint)", borderRadius: 8 }}>
          {model.error}
        </div>
      )}
      <div className="model-card-footer">
        <div className="tags">
          {tags.map((tag) => (
            <span className="tag" key={tag}>{tag}</span>
          ))}
          {model.load_seconds != null && model.status === "ready" && <span className="tag">{model.load_seconds}s load</span>}
        </div>
        <button className="btn btn-icon" onClick={onUnload} title="Unload" type="button">
          <Trash2 size={14} />
        </button>
      </div>
    </article>
  );
}

/* ============================================================ */
/* Load Drawer (5-step stepper)                                 */
/* ============================================================ */

function LoadDrawer(props: {
  open: boolean;
  stepIndex: number;
  onStepIndex: (n: number) => void;
  onClose: () => void;
  form: LoadFormState;
  onForm: Dispatch<SetStateAction<LoadFormState>>;
  engineFields: EngineConfigField[];
  runtimeValues: Record<string, string | boolean>;
  onRuntimeValues: Dispatch<SetStateAction<Record<string, string | boolean>>>;
  onSubmit: () => void;
}) {
  const setField = <K extends keyof LoadFormState>(key: K, value: LoadFormState[K]) => {
    props.onForm((current) => ({ ...current, [key]: value }));
  };
  const setRuntimeField = (name: string, value: string | boolean) => {
    props.onRuntimeValues((current) => ({ ...current, [name]: value }));
  };
  const setDraftModel = (value: string) => {
    props.onRuntimeValues((current) => {
      const currentTokens = String(current.num_speculative_tokens ?? "").trim();
      return {
        ...current,
        speculative_model: value,
        speculative_method: current.speculative_method || "dflash",
        num_speculative_tokens:
          value.trim() && (!currentTokens || currentTokens === "0")
            ? DRAFT_DEFAULT_TOKENS
            : current.num_speculative_tokens ?? ""
      };
    });
  };

  const essentialFields = props.engineFields.filter((field) => ESSENTIAL_RUNTIME_FIELDS.has(field.name));
  const speculativeMethodField = props.engineFields.find((field) => field.name === "speculative_method");
  const speculativeMethodChoices = speculativeMethodField?.choices.length ? speculativeMethodField.choices : ["dflash", "draft", "eagle3"];
  const draftModelValue = String(props.runtimeValues.speculative_model ?? "");
  const draftTokensValue = String(props.runtimeValues.num_speculative_tokens ?? "0");
  const speculativeMethodValue = String(props.runtimeValues.speculative_method ?? "dflash");

  const next = () => {
    if (props.stepIndex < STEPS.length - 1) props.onStepIndex(props.stepIndex + 1);
    else props.onSubmit();
  };
  const back = () => {
    if (props.stepIndex > 0) props.onStepIndex(props.stepIndex - 1);
  };

  return (
    <>
      <div className={`drawer-backdrop ${props.open ? "open" : ""}`} onClick={props.onClose} />
      <div className={`drawer ${props.open ? "open" : ""}`} role="dialog" aria-label="Load model">
        <div className="drawer-head">
          <div>
            <h2>Add model</h2>
            <p>Step {props.stepIndex + 1} of {STEPS.length} · {STEPS[props.stepIndex].title}</p>
          </div>
          <button className="btn-icon" onClick={props.onClose} type="button">
            <X size={16} />
          </button>
        </div>

        <div className="stepper-rail">
          {STEPS.map((step, idx) => {
            const stateClass = idx < props.stepIndex ? "completed" : idx === props.stepIndex ? "active" : "";
            return (
              <button
                key={step.id}
                className={`stepper-step ${stateClass}`}
                type="button"
                onClick={() => props.onStepIndex(idx)}
              >
                <div className="step-line" />
                <span className="step-num">{step.num}</span>
                <span className="step-title">{step.title}</span>
              </button>
            );
          })}
        </div>

        <div className="drawer-body">
          {props.stepIndex === 0 && (
            <div className="stepper-fields">
              <div className="stepper-subhead">Where does this model come from?</div>
              <TextField
                label="Model ID or path"
                value={props.form.modelId}
                onChange={(v) => setField("modelId", v)}
                placeholder={`${MAIN_MODEL_HINT} or /path/to/model`}
                wide
              />
              <TextField
                label="Revision"
                value={props.form.revision}
                onChange={(v) => setField("revision", v)}
                placeholder="main"
              />
              <TextField
                label="Subfolder"
                value={props.form.subfolder}
                onChange={(v) => setField("subfolder", v)}
              />
              <CheckField label="Local files only" checked={props.form.localFilesOnly} onChange={(v) => setField("localFilesOnly", v)} />
              <CheckField label="Force conversion" checked={props.form.forceConversion} onChange={(v) => setField("forceConversion", v)} />
            </div>
          )}

          {props.stepIndex === 1 && (
            <div className="stepper-fields">
              <div className="stepper-subhead">Naming & tokenizer</div>
              <TextField
                label="Served name"
                value={props.form.servedName}
                onChange={(v) => setField("servedName", v)}
                placeholder="auto"
                wide
              />
              <TextField
                label="Tokenizer"
                value={props.form.tokenizer}
                onChange={(v) => setField("tokenizer", v)}
                placeholder="defaults to model"
                wide
              />
              <TextField
                label="Model class override"
                value={props.form.modelClass}
                onChange={(v) => setField("modelClass", v)}
                placeholder="pkg.module:ClassName"
                wide
              />
              <CheckField label="Replace existing" checked={props.form.replace} onChange={(v) => setField("replace", v)} />
            </div>
          )}

          {props.stepIndex === 2 && (
            <div className="stepper-fields">
              <div className="stepper-subhead">Precision</div>
              <label className="field">
                <span>Model dtype</span>
                <select value={props.form.modelDtype} onChange={(event) => setField("modelDtype", event.target.value)}>
                  {DTYPE_OPTIONS.map((opt) => (
                    <option key={opt} value={opt}>{opt || "auto"}</option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Auto convert HF</span>
                <select value={props.form.autoConvertHf} onChange={(event) => setField("autoConvertHf", event.target.value)}>
                  <option value="">auto</option>
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              </label>

              <div className="stepper-subhead">Quantization</div>
              <div className="quant-modes">
                {QUANT_OPTIONS.map((option) => (
                  <button
                    key={option.value || "none"}
                    className={`quant-mode ${props.form.quantization === option.value ? "active" : ""}`}
                    onClick={() => setField("quantization", option.value)}
                    type="button"
                  >
                    <strong>{option.label}</strong>
                    <span>{option.hint}</span>
                  </button>
                ))}
              </div>
              {props.form.quantization === "affine" && (
                <>
                  <TextField label="Bits" value={props.form.quantBits} onChange={(v) => setField("quantBits", v)} type="number" />
                  <TextField label="Group size" value={props.form.quantGroupSize} onChange={(v) => setField("quantGroupSize", v)} type="number" />
                </>
              )}

              <div className="stepper-subhead">Attention</div>
              <label className="field span-2">
                <span>Attention mechanism</span>
                <select value={props.form.attnMechanism} onChange={(event) => setField("attnMechanism", event.target.value)}>
                  <option value="">default (unified)</option>
                  <option value="auto">auto</option>
                  <option value="vanilla">vanilla</option>
                  <option value="sdpa">sdpa</option>
                  <option value="unified">unified</option>
                  <option value="page_attention">page_attention</option>
                </select>
              </label>

              <div className="stepper-subhead">KV cache</div>
              <label className="field">
                <span>Cache dtype</span>
                <select value={props.form.cacheDtype} onChange={(event) => setField("cacheDtype", event.target.value)}>
                  <option value="">default (fp16)</option>
                  <option value="fp8">fp8</option>
                  <option value="turboquant">TurboQuant</option>
                  <option value="turboquant2">TurboQuant 2-bit</option>
                  <option value="turboquant3">TurboQuant 3-bit</option>
                  <option value="turboquant4">TurboQuant 4-bit</option>
                </select>
              </label>
              {(props.form.cacheDtype === "turboquant" || props.form.cacheDtype === "tq") && (
                <label className="field">
                  <span>Cache bits</span>
                  <select value={props.form.cacheBits} onChange={(event) => setField("cacheBits", event.target.value)}>
                    <option value="2">2-bit</option>
                    <option value="3">3-bit</option>
                    <option value="4">4-bit</option>
                  </select>
                </label>
              )}

              <div className="stepper-subhead">Checkpoint files</div>
              <TextField label="Config path or ID" value={props.form.configPath} onChange={(v) => setField("configPath", v)} placeholder="optional" />
              <TextField label="Weights file" value={props.form.weightsName} onChange={(v) => setField("weightsName", v)} placeholder="model.safetensors" />
              <TextField label="Converted cache" value={props.form.convertedCache} onChange={(v) => setField("convertedCache", v)} wide />
              <CheckField label="Strict weights" checked={props.form.strict} onChange={(v) => setField("strict", v)} />
              <CheckField label="Lazy load" checked={props.form.lazy} onChange={(v) => setField("lazy", v)} />
              <CheckField label="Copy support files" checked={props.form.copySupportFiles} onChange={(v) => setField("copySupportFiles", v)} />
            </div>
          )}

          {props.stepIndex === 3 && (
            <div className="stepper-fields">
              <div className="stepper-subhead">Engine essentials</div>
              {essentialFields.length === 0 && (
                <div style={{ gridColumn: "1 / -1", color: "var(--text-tertiary)", fontSize: 12 }}>
                  Engine schema not yet loaded. You can still edit advanced fields below or in the Engine tab.
                </div>
              )}
              {essentialFields.map((field) => (
                <RuntimeField
                  field={field}
                  key={field.name}
                  value={props.runtimeValues[field.name]}
                  onChange={(value) =>
                    props.onRuntimeValues((current) => ({ ...current, [field.name]: value }))
                  }
                />
              ))}
              <div className="stepper-subhead">Draft model</div>
              <TextField
                label="Draft model ID or path"
                value={draftModelValue}
                onChange={setDraftModel}
                placeholder={`${DRAFT_MODEL_HINT} or /path/to/dflash-draft`}
                wide
              />
              <label className="field">
                <span>Speculative method</span>
                <select value={speculativeMethodValue} onChange={(event) => setRuntimeField("speculative_method", event.target.value)}>
                  {speculativeMethodChoices.map((choice) => (
                    <option key={choice} value={choice}>{choice}</option>
                  ))}
                </select>
              </label>
              <TextField
                label="Draft tokens"
                value={draftTokensValue}
                onChange={(value) => setRuntimeField("num_speculative_tokens", value)}
                type="number"
              />
              <label className="field span-2">
                <span>Draft dtype</span>
                <select value={props.form.draftModelDtype} onChange={(event) => setField("draftModelDtype", event.target.value)}>
                  {DTYPE_OPTIONS.map((opt) => (
                    <option key={opt} value={opt}>{opt || "auto"}</option>
                  ))}
                </select>
              </label>

              <div className="stepper-subhead">Draft quantization</div>
              <div className="quant-modes">
                {QUANT_OPTIONS.map((option) => (
                  <button
                    key={option.value || "none"}
                    className={`quant-mode ${props.form.draftQuantization === option.value ? "active" : ""}`}
                    onClick={() => setField("draftQuantization", option.value)}
                    type="button"
                  >
                    <strong>{option.label}</strong>
                    <span>{option.hint}</span>
                  </button>
                ))}
              </div>
              {props.form.draftQuantization === "affine" && (
                <>
                  <TextField label="Draft bits" value={props.form.draftQuantBits} onChange={(value) => setField("draftQuantBits", value)} type="number" />
                  <TextField label="Draft group size" value={props.form.draftQuantGroupSize} onChange={(value) => setField("draftQuantGroupSize", value)} type="number" />
                </>
              )}
              <div className="stepper-subhead">Raw overrides</div>
              <label className="field span-2">
                <span>Additional model kwargs JSON</span>
                <textarea
                  value={props.form.modelKwargsJson}
                  onChange={(event) => setField("modelKwargsJson", event.target.value)}
                  spellCheck={false}
                  placeholder='{"trust_remote_code": false}'
                />
              </label>
              <label className="field span-2">
                <span>Additional eSurge kwargs JSON</span>
                <textarea
                  value={props.form.engineKwargsJson}
                  onChange={(event) => setField("engineKwargsJson", event.target.value)}
                  spellCheck={false}
                  placeholder='{"memory_utilization": 0.85}'
                />
              </label>
            </div>
          )}

          {props.stepIndex === 4 && (
            <div className="stepper-fields">
              <div className="stepper-subhead">Review & launch</div>
              <dl className="review-kv">
                <dt>Model ID</dt>
                <dd>{props.form.modelId || "—"}</dd>
                <dt>Served as</dt>
                <dd>{profileNameFor(props.form)}</dd>
                <dt>Tokenizer</dt>
                <dd>{props.form.tokenizer || "(model default)"}</dd>
                <dt>dtype</dt>
                <dd>{props.form.modelDtype || "auto"}</dd>
                <dt>Quantization</dt>
                <dd>
                  {props.form.quantization || "none"}
                  {props.form.quantization === "affine" && ` (${props.form.quantBits}b · g${props.form.quantGroupSize})`}
                </dd>
                <dt>KV cache</dt>
                <dd>
                  {props.form.cacheDtype || "fp16 (default)"}
                  {(props.form.cacheDtype === "turboquant" || props.form.cacheDtype === "tq") && ` (${props.form.cacheBits}-bit)`}
                </dd>
                <dt>Attention</dt>
                <dd>{props.form.attnMechanism || "unified (default)"}</dd>
                <dt>Draft model</dt>
                <dd>{draftModelValue || "none"}</dd>
                <dt>Draft method</dt>
                <dd>{draftModelValue ? speculativeMethodValue : "—"}</dd>
                <dt>Draft tokens</dt>
                <dd>{draftModelValue ? draftTokensValue : "—"}</dd>
                <dt>Draft quantization</dt>
                <dd>
                  {draftModelValue
                    ? quantizationSummary(props.form.draftQuantization, props.form.draftQuantBits, props.form.draftQuantGroupSize)
                    : "—"}
                </dd>
                <dt>Replace existing</dt>
                <dd>{props.form.replace ? "yes" : "no"}</dd>
                <dt>Local files only</dt>
                <dd>{props.form.localFilesOnly ? "yes" : "no"}</dd>
              </dl>
              <div className="stepper-subhead">Engine overrides (vs default)</div>
              <pre className="review-block">
                {previewEngineDiff(props.engineFields, props.runtimeValues)}
              </pre>
            </div>
          )}
        </div>

        <div className="drawer-foot">
          <button className="btn btn-ghost" onClick={back} disabled={props.stepIndex === 0} type="button">
            <ChevronLeft size={14} /> Back
          </button>
          <div style={{ flex: 1 }} />
          <button className="btn btn-secondary" onClick={props.onClose} type="button">
            Cancel
          </button>
          <button className="btn btn-primary" onClick={next} type="button">
            {props.stepIndex < STEPS.length - 1 ? (
              <>Next <ChevronRight size={14} /></>
            ) : (
              <><Play size={14} /> Load model</>
            )}
          </button>
        </div>
      </div>
    </>
  );
}

function previewEngineDiff(fields: EngineConfigField[], values: Record<string, string | boolean>): string {
  const diff: Record<string, unknown> = {};
  for (const field of fields) {
    if (!isFieldDirty(field, values[field.name])) continue;
    if (field.type === "bool") diff[field.name] = Boolean(values[field.name]);
    else diff[field.name] = values[field.name] ?? "";
  }
  if (Object.keys(diff).length === 0) return "// using engine defaults";
  return JSON.stringify(diff, null, 2);
}

/* ============================================================ */
/* Engine                                                       */
/* ============================================================ */

function EngineSection(props: {
  schema: ConfigSchema;
  values: Record<string, string | boolean>;
  onValues: Dispatch<SetStateAction<Record<string, string | boolean>>>;
  onReset: () => void;
  category: string;
  onCategory: (value: string) => void;
  query: string;
  onQuery: (value: string) => void;
  dirtyOnly: boolean;
  onDirtyOnly: (value: boolean) => void;
}) {
  const groups = useMemo(() => groupBy(props.schema.engine, (f) => f.category), [props.schema]);
  const categories = Object.keys(groups);

  useEffect(() => {
    if (!props.category && categories.length) props.onCategory(categories[0]);
  }, [categories, props]);

  const dirtyByCategory = useMemo(() => {
    const map: Record<string, number> = {};
    for (const field of props.schema.engine) {
      if (isFieldDirty(field, props.values[field.name])) {
        map[field.category] = (map[field.category] ?? 0) + 1;
      }
    }
    return map;
  }, [props.schema, props.values]);

  const totalDirty = Object.values(dirtyByCategory).reduce((sum, n) => sum + n, 0);
  const queryLower = props.query.trim().toLowerCase();

  let visibleFields: EngineConfigField[];
  if (queryLower) {
    visibleFields = props.schema.engine.filter(
      (field) =>
        field.name.toLowerCase().includes(queryLower) ||
        field.label.toLowerCase().includes(queryLower) ||
        (field.hint ?? "").toLowerCase().includes(queryLower)
    );
  } else {
    visibleFields = groups[props.category] ?? [];
  }
  if (props.dirtyOnly) {
    visibleFields = visibleFields.filter((field) => isFieldDirty(field, props.values[field.name]));
  }
  const visibleByCategory = groupBy(visibleFields, (f) => f.category);

  return (
    <>
      <header className="panel-top">
        <div>
          <h1>Engine</h1>
          <p>{props.schema.engine.length} fields · {totalDirty} modified</p>
        </div>
        <div className="toolbar">
          <button className="btn btn-secondary" onClick={props.onReset} type="button">
            <RotateCcw size={14} /> Defaults
          </button>
        </div>
      </header>

      <div className="engine-layout">
        <aside className="engine-categories">
          {categories.map((category) => (
            <button
              key={category}
              className={`engine-category-btn ${props.category === category && !queryLower ? "active" : ""}`}
              onClick={() => {
                props.onCategory(category);
                props.onQuery("");
              }}
              type="button"
            >
              <span>{category}</span>
              {dirtyByCategory[category] && <span className="dirty-dot" title={`${dirtyByCategory[category]} modified`} />}
              <span className="engine-category-count">{groups[category].length}</span>
            </button>
          ))}
        </aside>
        <div className="engine-main">
          <div className="engine-toolbar">
            <div className="search-input">
              <Search size={14} />
              <input
                value={props.query}
                onChange={(event) => props.onQuery(event.target.value)}
                placeholder="Search across all fields…"
              />
            </div>
            <label className="toggle">
              <input type="checkbox" checked={props.dirtyOnly} onChange={(event) => props.onDirtyOnly(event.target.checked)} />
              <span className="toggle-switch" />
              <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>Modified only</span>
            </label>
          </div>
          <div className="engine-fields">
            {visibleFields.length === 0 ? (
              <div style={{ padding: 32, textAlign: "center", color: "var(--text-tertiary)" }}>
                {props.dirtyOnly ? "No modified fields" : queryLower ? "No matches" : "Empty category"}
              </div>
            ) : queryLower ? (
              Object.entries(visibleByCategory).map(([category, fields]) => (
                <div className="engine-field-group" key={category}>
                  <h4>{category}</h4>
                  <div className="engine-field-row">
                    {fields.map((field) => (
                      <RuntimeField
                        field={field}
                        key={field.name}
                        value={props.values[field.name]}
                        dirty={isFieldDirty(field, props.values[field.name])}
                        onChange={(value) => props.onValues((current) => ({ ...current, [field.name]: value }))}
                      />
                    ))}
                  </div>
                </div>
              ))
            ) : (
              <div className="engine-field-row">
                {visibleFields.map((field) => (
                  <RuntimeField
                    field={field}
                    key={field.name}
                    value={props.values[field.name]}
                    dirty={isFieldDirty(field, props.values[field.name])}
                    onChange={(value) => props.onValues((current) => ({ ...current, [field.name]: value }))}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

/* ============================================================ */
/* Telemetry                                                    */
/* ============================================================ */

function TelemetrySection(props: {
  health: Health | null;
  metrics: Metrics | null;
  models: AppModel[];
  events: EngineEvent[];
  tpsHistory: SparkPoint[];
  reqHistory: SparkPoint[];
  tokensHistory: SparkPoint[];
  errorRateHistory: SparkPoint[];
}) {
  const ready = props.models.filter((m) => m.status === "ready");
  const total = props.metrics?.total_requests ?? 0;
  const failed = props.metrics?.failed_requests ?? 0;
  const errorRate = total ? (failed / total) * 100 : 0;
  return (
    <>
      <header className="panel-top">
        <div>
          <h1>Telemetry</h1>
          <p>Server state: {props.health?.status ?? "offline"} · uptime {formatDuration(props.metrics?.uptime_seconds ?? props.health?.uptime_seconds ?? 0)}</p>
        </div>
      </header>
      <div className="panel-body no-padding">
        <div className="telemetry-body">
          <div className="hero-strip">
            <SparkCard
              icon={<Zap size={12} />}
              label="Tokens / sec"
              value={formatNumber(props.metrics?.average_tokens_per_second, 1)}
              foot={`avg over ${props.tpsHistory.length} samples`}
              points={props.tpsHistory}
            />
            <SparkCard
              icon={<Activity size={12} />}
              label="Active requests"
              value={String(props.metrics?.active_requests ?? 0)}
              foot={`peak ${peakOf(props.reqHistory)}`}
              points={props.reqHistory}
            />
            <SparkCard
              icon={<Boxes size={12} />}
              label="Tokens generated"
              value={compactNumber(props.metrics?.total_tokens_generated ?? 0)}
              foot={`${total} total requests`}
              points={props.tokensHistory}
            />
            <SparkCard
              icon={<AlertTriangle size={12} />}
              label="Error rate"
              value={`${errorRate.toFixed(1)}%`}
              foot={`${failed} failed`}
              points={props.errorRateHistory}
              tone="danger"
            />
          </div>

          <section className="telemetry-section">
            <div className="telemetry-section-head">
              <h3>Loaded models ({ready.length})</h3>
              <span style={{ fontSize: 11, color: "var(--text-tertiary)" }}>aggregate TPS shared across server</span>
            </div>
            {ready.length === 0 ? (
              <div style={{ padding: 24, textAlign: "center", color: "var(--text-tertiary)", fontSize: 12 }}>
                No models loaded
              </div>
            ) : (
              <table className="telemetry-table">
                <thead>
                  <tr>
                    <th>Served name</th>
                    <th>Source</th>
                    <th>Tokenizer</th>
                    <th>Load time</th>
                    <th>Template</th>
                  </tr>
                </thead>
                <tbody>
                  {ready.map((model) => (
                    <tr key={model.served_name}>
                      <td><strong>{model.served_name}</strong></td>
                      <td className="num">{model.model_id}</td>
                      <td className="num">{model.tokenizer ?? "—"}</td>
                      <td className="num">{model.load_seconds != null ? `${model.load_seconds}s` : "—"}</td>
                      <td>
                        <span className={`pill ${model.has_chat_template ? "ready" : "neutral"}`}>
                          {model.has_chat_template ? "yes" : "fallback"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          <section className="telemetry-section">
            <div className="telemetry-section-head">
              <h3>Health</h3>
              <span style={{ fontSize: 11, color: "var(--text-tertiary)" }}>last check {props.health ? "now" : "—"}</span>
            </div>
            <div className="health-grid">
              <HealthRow label="Server" ok={Boolean(props.health)} value={props.health?.status ?? "offline"} />
              <HealthRow label="Models endpoint" ok={ready.length > 0} value={`${ready.length} ready`} />
              <HealthRow label="Active requests" ok={(props.metrics?.active_requests ?? 0) >= 0} value={String(props.metrics?.active_requests ?? 0)} />
              <HealthRow label="Tool executions" ok={(props.metrics?.failed_tool_executions ?? 0) === 0} value={`${props.metrics?.tool_executions ?? 0} ok / ${props.metrics?.failed_tool_executions ?? 0} failed`} />
              <HealthRow label="Admin actions" ok={true} value={String(props.metrics?.admin_actions ?? 0)} />
              <HealthRow label="Error rate" ok={errorRate < 5} value={`${errorRate.toFixed(1)}%`} />
            </div>
          </section>

          <section className="telemetry-section">
            <div className="telemetry-section-head">
              <h3>Events</h3>
              <span style={{ fontSize: 11, color: "var(--text-tertiary)" }}>{props.events.length} recent</span>
            </div>
            <div className="event-timeline">
              {props.events.length === 0 ? (
                <div style={{ padding: 24, textAlign: "center", color: "var(--text-tertiary)", fontSize: 12 }}>
                  No events yet — server activity will appear here
                </div>
              ) : (
                props.events.map((event) => (
                  <div className="event-item" key={event.id}>
                    <span className="event-time">{timeAgo(event.at)}</span>
                    <span className={`event-dot ${event.tone}`} />
                    <span className="event-text">
                      <strong>{event.text}</strong>
                      {event.detail && <> · {event.detail}</>}
                    </span>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </div>
    </>
  );
}

function peakOf(points: SparkPoint[]): number {
  if (!points.length) return 0;
  return Math.max(...points.map((p) => p.v));
}

function HealthRow({ label, ok, value }: { label: string; ok: boolean; value: string }) {
  return (
    <div className="health-item">
      <span className="dot" style={{ background: ok ? "var(--success)" : "var(--danger)" }} />
      <span style={{ flex: 1 }}>{label}</span>
      <span style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-mono)", fontSize: 11 }}>{value}</span>
    </div>
  );
}

function SparkCard({
  icon,
  label,
  value,
  foot,
  points,
  tone = "default"
}: {
  icon: ReactNode;
  label: string;
  value: string;
  foot: string;
  points: SparkPoint[];
  tone?: "default" | "danger";
}) {
  const W = 220;
  const H = 40;
  const path = buildSparkPath(points, W, H);
  const stroke = tone === "danger" ? "var(--danger)" : "var(--accent)";
  const fillStroke = tone === "danger" ? "var(--danger-faint)" : "var(--accent-faint)";
  return (
    <div className="spark-card">
      <span className="spark-card-head">
        {icon} {label}
      </span>
      <strong>{value}</strong>
      <svg className="spark-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {path && (
          <>
            <path d={`${path} L${W},${H} L0,${H} Z`} fill={fillStroke} />
            <path d={path} fill="none" stroke={stroke} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </>
        )}
      </svg>
      <em>{foot}</em>
    </div>
  );
}

/* ============================================================ */
/* Settings                                                     */
/* ============================================================ */

function SettingsSection(props: {
  theme: ThemeMode;
  onTheme: (t: ThemeMode) => void;
  density: Density;
  onDensity: (d: Density) => void;
  apiKey: string;
  onApiKey: (value: string) => void;
  info: AppInfo | null;
  onResetRuntime: () => void;
  onClearConversations: () => void;
  conversationsCount: number;
  profilesCount: number;
  onClearProfiles: () => void;
}) {
  return (
    <>
      <header className="panel-top">
        <div>
          <h1>Settings</h1>
          <p>EasyMLX {props.info?.easymlx_version ?? "local"} · {props.info?.openai_base_url ?? "/v1"}</p>
        </div>
      </header>
      <div className="panel-body no-padding">
        <div className="settings-body">
          <section className="setting-group">
            <div className="setting-group-head">
              <h3>Appearance</h3>
              <p>Looks the same as macOS. System follows your OS.</p>
            </div>
            <div className="setting-row">
              <div className="label">
                <strong>Theme</strong>
                <span>Light, dark, or follow system</span>
              </div>
              <div className="control">
                <div className="theme-picker">
                  <button className={props.theme === "system" ? "active" : ""} type="button" onClick={() => props.onTheme("system")}>
                    <Monitor size={12} /> System
                  </button>
                  <button className={props.theme === "light" ? "active" : ""} type="button" onClick={() => props.onTheme("light")}>
                    <Sun size={12} /> Light
                  </button>
                  <button className={props.theme === "dark" ? "active" : ""} type="button" onClick={() => props.onTheme("dark")}>
                    <Moon size={12} /> Dark
                  </button>
                </div>
              </div>
            </div>
            <div className="setting-row">
              <div className="label">
                <strong>Density</strong>
                <span>Tighter spacing for dense workflows</span>
              </div>
              <div className="control">
                <div className="density-picker">
                  <button className={props.density === "comfortable" ? "active" : ""} type="button" onClick={() => props.onDensity("comfortable")}>
                    Comfortable
                  </button>
                  <button className={props.density === "compact" ? "active" : ""} type="button" onClick={() => props.onDensity("compact")}>
                    Compact
                  </button>
                </div>
              </div>
            </div>
          </section>

          <section className="setting-group">
            <div className="setting-group-head">
              <h3>API</h3>
              <p>Bearer token sent with every request</p>
            </div>
            <div className="setting-row">
              <div className="label">
                <strong>API key</strong>
                <span>Stored in your browser only</span>
              </div>
              <div className="control" style={{ minWidth: 240 }}>
                <input
                  value={props.apiKey}
                  onChange={(event) => props.onApiKey(event.target.value)}
                  placeholder="Bearer token"
                  type="password"
                  autoComplete="off"
                  style={{
                    background: "var(--bg-input)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: 8,
                    padding: "7px 10px",
                    fontSize: 13,
                    width: "100%",
                    outline: "none"
                  }}
                />
              </div>
            </div>
          </section>

          <section className="setting-group">
            <div className="setting-group-head">
              <h3>Data</h3>
              <p>Local browser storage</p>
            </div>
            <div className="setting-row">
              <div className="label">
                <strong>Conversations</strong>
                <span>{props.conversationsCount} saved locally</span>
              </div>
              <div className="control">
                <button className="btn btn-danger" onClick={props.onClearConversations} type="button" disabled={!props.conversationsCount}>
                  <Trash2 size={13} /> Clear
                </button>
              </div>
            </div>
            <div className="setting-row">
              <div className="label">
                <strong>Load profiles</strong>
                <span>{props.profilesCount} saved</span>
              </div>
              <div className="control">
                <button className="btn btn-danger" onClick={props.onClearProfiles} type="button" disabled={!props.profilesCount}>
                  <Trash2 size={13} /> Clear
                </button>
              </div>
            </div>
            <div className="setting-row">
              <div className="label">
                <strong>Engine config</strong>
                <span>Reset all fields to schema defaults</span>
              </div>
              <div className="control">
                <button className="btn btn-secondary" onClick={props.onResetRuntime} type="button">
                  <RotateCcw size={13} /> Reset
                </button>
              </div>
            </div>
          </section>

          <section className="setting-group">
            <div className="setting-group-head">
              <h3>About</h3>
              <p>Server endpoints reported by the host</p>
            </div>
            <div className="setting-row">
              <div className="label">
                <strong>Models endpoint</strong>
                <span style={{ fontFamily: "var(--font-mono)" }}>{props.info?.models_endpoint ?? "—"}</span>
              </div>
            </div>
            <div className="setting-row">
              <div className="label">
                <strong>Chat endpoint</strong>
                <span style={{ fontFamily: "var(--font-mono)" }}>{props.info?.chat_endpoint ?? "—"}</span>
              </div>
            </div>
          </section>
        </div>
      </div>
    </>
  );
}

/* ============================================================ */
/* Command palette                                              */
/* ============================================================ */

function CommandPalette({
  open,
  onClose,
  items
}: {
  open: boolean;
  onClose: () => void;
  items: Array<{ id: string; group: string; label: string; hint?: string; icon: ReactNode; run: () => void }>;
}) {
  const [query, setQuery] = useState("");
  const [activeIdx, setActiveIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIdx(0);
      window.requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter((item) =>
      item.label.toLowerCase().includes(q) ||
      item.group.toLowerCase().includes(q) ||
      (item.hint ?? "").toLowerCase().includes(q)
    );
  }, [items, query]);

  const grouped = useMemo(() => {
    const groups: Record<string, typeof filtered> = {};
    for (const item of filtered) {
      groups[item.group] = groups[item.group] ?? [];
      groups[item.group].push(item);
    }
    return groups;
  }, [filtered]);

  useEffect(() => {
    if (activeIdx >= filtered.length) setActiveIdx(0);
  }, [filtered.length, activeIdx]);

  const onKey = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIdx((idx) => Math.min(idx + 1, filtered.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIdx((idx) => Math.max(idx - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      const item = filtered[activeIdx];
      if (item) {
        item.run();
        onClose();
      }
    }
  };

  return (
    <div className={`palette-backdrop ${open ? "open" : ""}`} onClick={onClose}>
      <div className="palette" onClick={(event) => event.stopPropagation()}>
        <div className="palette-search">
          <Search size={15} style={{ color: "var(--text-tertiary)" }} />
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={onKey}
            placeholder="Type a command or search…"
          />
          <Command size={14} style={{ color: "var(--text-quaternary)" }} />
        </div>
        <div className="palette-list">
          {filtered.length === 0 ? (
            <div className="palette-empty">No matches</div>
          ) : (
            (() => {
              let runningIdx = 0;
              return Object.entries(grouped).map(([group, list]) => (
                <div key={group}>
                  <div className="palette-group-title">{group}</div>
                  {list.map((item) => {
                    const myIdx = runningIdx++;
                    return (
                      <button
                        key={item.id}
                        className={`palette-item ${myIdx === activeIdx ? "active" : ""}`}
                        onMouseEnter={() => setActiveIdx(myIdx)}
                        onClick={() => {
                          item.run();
                          onClose();
                        }}
                        type="button"
                      >
                        <span className="icon">{item.icon}</span>
                        <span className="label">{item.label}</span>
                        {item.hint && <span className="hint">{item.hint}</span>}
                      </button>
                    );
                  })}
                </div>
              ));
            })()
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================ */
/* Form primitives                                              */
/* ============================================================ */

function RuntimeField({
  field,
  value,
  onChange,
  dirty
}: {
  field: EngineConfigField;
  value: string | boolean | undefined;
  onChange: (value: string | boolean) => void;
  dirty?: boolean;
}) {
  const wide = new Set(["text", "password", "int-list", "string-list", "pair-list", "json"]).has(field.type);

  if (field.type === "bool") {
    return (
      <div className={`engine-field engine-field-bool ${wide ? "wide" : ""}`}>
        <label className="toggle-row">
          <input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />
          <span className="toggle-switch" />
          <span className="bool-label">{field.label}</span>
          {dirty && <span className="dirty-indicator" />}
          <span className="default-hint">def: {String(field.default ?? "false")}</span>
        </label>
        {field.hint && <div className="hint">{field.hint}</div>}
      </div>
    );
  }

  if (field.type === "select") {
    return (
      <div className={`engine-field ${wide ? "wide" : ""}`}>
        <div className="field-label-row">
          <span>{field.label}</span>
          {dirty && <span className="dirty-indicator" />}
          <span className="default-hint">def: {fieldDefaultToString(field) || "auto"}</span>
        </div>
        <select value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}>
          {(field.choices.length ? field.choices : [""]).map((choice) => (
            <option key={choice || "auto"} value={choice}>{choice || "auto"}</option>
          ))}
        </select>
        {field.hint && <div className="hint">{field.hint}</div>}
      </div>
    );
  }

  if (field.type === "json") {
    return (
      <div className={`engine-field ${wide ? "wide" : ""}`}>
        <div className="field-label-row">
          <span>{field.label}</span>
          {dirty && <span className="dirty-indicator" />}
          <span className="default-hint">def: {fieldDefaultToString(field) || "—"}</span>
        </div>
        <textarea
          value={String(value ?? "")}
          onChange={(event) => onChange(event.target.value)}
          spellCheck={false}
          placeholder={field.placeholder ?? undefined}
        />
        {field.hint && <div className="hint">{field.hint}</div>}
      </div>
    );
  }

  return (
    <div className={`engine-field ${wide ? "wide" : ""}`}>
      <div className="field-label-row">
        <span>{field.label}</span>
        {dirty && <span className="dirty-indicator" />}
        <span className="default-hint">def: {fieldDefaultToString(field) || "—"}</span>
      </div>
      <input
        value={String(value ?? "")}
        onChange={(event) => onChange(event.target.value)}
        type={field.type === "password" ? "password" : field.type === "int" || field.type === "float" ? "number" : "text"}
        min={field.min ?? undefined}
        max={field.max ?? undefined}
        step={field.step ?? undefined}
        placeholder={field.placeholder ?? undefined}
        style={{
          background: "var(--bg-input)",
          border: "1px solid var(--border-subtle)",
          borderRadius: 8,
          padding: "7px 10px",
          fontSize: 13,
          width: "100%",
          outline: "none"
        }}
      />
      {field.hint && <div className="hint">{field.hint}</div>}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  const [display, setDisplay] = useState(() => String(value));
  useEffect(() => {
    const parsed = Number.parseFloat(display);
    if (!Number.isFinite(parsed) || parsed !== value) {
      setDisplay(String(value));
    }
  }, [value]);
  return (
    <label className="field">
      <span>{label}</span>
      <input
        value={display}
        min={min}
        max={max}
        step={step}
        type="number"
        onChange={(event) => {
          const raw = event.target.value;
          setDisplay(raw);
          if (raw === "" || raw === "-" || raw === ".") return;
          const parsed = Number.parseFloat(raw);
          if (Number.isFinite(parsed)) onChange(parsed);
        }}
        onBlur={() => {
          const parsed = Number.parseFloat(display);
          if (!Number.isFinite(parsed)) {
            setDisplay(String(value));
          } else if (parsed !== value) {
            onChange(parsed);
            setDisplay(String(parsed));
          }
        }}
      />
    </label>
  );
}

function TextField({
  label,
  value,
  onChange,
  placeholder,
  type = "text",
  wide = false
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: string;
  wide?: boolean;
}) {
  return (
    <label className={wide ? "field span-2" : "field"}>
      <span>{label}</span>
      <input
        value={value}
        type={type}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
      />
    </label>
  );
}

function CheckField({
  label,
  checked,
  onChange
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="check-field">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

/* ============================================================ */
/* Chat networking                                              */
/* ============================================================ */

function buildChatRequest(
  model: string,
  messages: ChatMessage[],
  sampling: SamplingState,
  stopStrings: string,
  toolsJson: string,
  responseFormatJson: string,
  enableThinking: boolean
): ChatRequest {
  const request: ChatRequest = {
    model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    chat_template_kwargs: { enable_thinking: enableThinking },
    ...sampling
  };
  const stops = stopStrings
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (stops.length) request.stop = stops;
  const tools = parseJsonValue(toolsJson, "Tools JSON");
  if (tools !== undefined) request.tools = tools;
  const responseFormat = parseJsonValue(responseFormatJson, "Response format JSON");
  if (responseFormat !== undefined) request.response_format = responseFormat;
  return request;
}

async function readChatStream(
  response: Response,
  onDelta: (content: string, reasoning: string) => void,
  onUsage: (usage: ChatUsage) => void
) {
  const reader = response.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const event of events) {
      const lines = event.split("\n").filter((line) => line.startsWith("data: "));
      for (const line of lines) {
        const data = line.slice(6).trim();
        if (!data || data === "[DONE]") continue;
        const payload = JSON.parse(data) as ChatStreamChunk;
        const delta = payload.choices?.[0]?.delta;
        if (delta) {
          onDelta(delta.content ?? "", delta.delta_reasoning_content ?? delta.reasoning_content ?? "");
        }
        if (payload.usage) {
          onUsage(payload.usage);
        }
      }
    }
  }
}

export default App;
