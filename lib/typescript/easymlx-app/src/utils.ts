import type { EngineConfigField } from "./types";

export function nullableText(value: string): string | null {
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

export function parseJsonObject(value: string, label: string): Record<string, unknown> {
  const text = value.trim();
  if (!text) {
    return {};
  }
  const parsed = JSON.parse(text) as unknown;
  if (!isPlainObject(parsed)) {
    throw new Error(`${label} must be a JSON object`);
  }
  return parsed;
}

export function parseJsonValue(value: string, label: string): unknown | undefined {
  const text = value.trim();
  if (!text) {
    return undefined;
  }
  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    throw new Error(`${label}: ${(error as Error).message}`);
  }
}

export function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function groupBy<T>(items: T[], keyFn: (item: T) => string): Record<string, T[]> {
  return items.reduce<Record<string, T[]>>((groups, item) => {
    const key = keyFn(item);
    groups[key] = groups[key] ?? [];
    groups[key].push(item);
    return groups;
  }, {});
}

export function fieldDefaultToString(field: EngineConfigField): string {
  if (field.default == null) {
    return "";
  }
  if (Array.isArray(field.default)) {
    return field.default.join(",");
  }
  return String(field.default);
}

export function coerceEngineValue(field: EngineConfigField, rawValue: string, checked: boolean): unknown {
  if (field.type === "bool") {
    return checked;
  }

  const raw = rawValue.trim();
  if (!raw) {
    return undefined;
  }
  if ((field.name === "tool_parser" || field.name === "reasoning_parser") && raw.toLowerCase() === "auto") {
    return undefined;
  }

  if (field.type === "int") {
    return Number.parseInt(raw, 10);
  }
  if (field.type === "float") {
    return Number.parseFloat(raw);
  }
  if (field.type === "int-list") {
    return raw
      .split(/[,\s]+/)
      .map((item) => item.trim())
      .filter(Boolean)
      .map((item) => Number.parseInt(item, 10));
  }
  if (field.type === "string-list") {
    return raw
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  if (field.type === "json") {
    return parseJsonValue(raw, field.label);
  }
  if (field.type === "pair-list") {
    return raw
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
      .map((item) => item.split(/[x:]/i).map((part) => Number.parseInt(part.trim(), 10)));
  }
  return raw;
}

export function compactObject(values: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined || value === null || value === "") {
      continue;
    }
    result[key] = value;
  }
  return result;
}

export function formatNumber(value: number | undefined, digits = 2): string {
  return Number(value ?? 0).toFixed(digits);
}
