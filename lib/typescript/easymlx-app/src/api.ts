import type {
  AppInfo,
  AppModelsResponse,
  ConfigSchema,
  Health,
  LoadModelRequest,
  Metrics
} from "./types";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const jsonHeaders = (apiKey: string): HeadersInit => ({
  "Content-Type": "application/json",
  ...authHeaders(apiKey)
});

export const authHeaders = (apiKey: string): HeadersInit => {
  const token = apiKey.trim();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

export async function getInfo(apiKey: string): Promise<AppInfo> {
  return getJson("/app/api/info", apiKey);
}

export async function getConfigSchema(apiKey: string): Promise<ConfigSchema> {
  return getJson("/app/api/config-schema", apiKey);
}

export async function getModels(apiKey: string): Promise<AppModelsResponse> {
  return getJson("/app/api/models", apiKey);
}

export async function getHealth(apiKey: string): Promise<Health> {
  return getJson("/health", apiKey);
}

export async function getMetrics(apiKey: string): Promise<Metrics> {
  return getJson("/metrics", apiKey);
}

export async function loadModel(apiKey: string, payload: LoadModelRequest): Promise<unknown> {
  return postJson("/app/api/models/load", apiKey, payload);
}

export async function unloadModel(apiKey: string, servedName: string): Promise<unknown> {
  return postJson(`/app/api/models/${encodeURIComponent(servedName)}/unload`, apiKey, {});
}

export async function getJson<T>(path: string, apiKey: string): Promise<T> {
  const response = await fetch(path, { headers: authHeaders(apiKey) });
  if (!response.ok) {
    throw new ApiError(await responseMessage(response), response.status);
  }
  return response.json() as Promise<T>;
}

export async function postJson<T>(path: string, apiKey: string, payload: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: jsonHeaders(apiKey),
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new ApiError(await responseMessage(response), response.status);
  }
  return response.json() as Promise<T>;
}

export async function responseMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as {
      detail?: unknown;
      error?: { message?: string };
    };
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (payload.detail) {
      return JSON.stringify(payload.detail);
    }
    if (payload.error?.message) {
      return payload.error.message;
    }
  } catch {
    return `${response.status} ${response.statusText}`;
  }
  return `${response.status} ${response.statusText}`;
}
