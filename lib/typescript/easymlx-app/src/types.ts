export type ModelStatus = "loading" | "ready" | "error";

export type ChatRole = "system" | "user" | "assistant" | "tool";

export type ChatMessage = {
  role: ChatRole;
  content: string;
  reasoning?: string;
};

export type AppModel = {
  served_name: string;
  model_id: string;
  status: ModelStatus;
  error: string | null;
  created_at: number;
  loaded_at: number | null;
  load_seconds: number | null;
  tokenizer: string | null;
  tokenizer_source: string | null;
  has_chat_template: boolean | null;
  revision: string | null;
  local_files_only: boolean;
  converted_cache_dir: string | null;
  force_conversion: boolean;
  model_class: string | null;
  model_kwargs: Record<string, unknown>;
  engine_kwargs: Record<string, unknown>;
};

export type AppModelsResponse = {
  data: AppModel[];
};

export type AppInfo = {
  name: string;
  easymlx_version: string;
  openai_base_url: string;
  models_endpoint: string;
  chat_endpoint: string;
};

export type FieldType =
  | "bool"
  | "float"
  | "int"
  | "int-list"
  | "json"
  | "pair-list"
  | "password"
  | "select"
  | "string-list"
  | "text";

export type EngineConfigField = {
  name: string;
  label: string;
  type: FieldType;
  category: string;
  default: unknown;
  choices: string[];
  hint: string | null;
  min: number | null;
  max: number | null;
  step: number | null;
  placeholder: string | null;
};

export type SamplingConfigField = {
  name: string;
  label: string;
  type: "float" | "int";
  default: number;
  min?: number;
  max?: number;
  step?: number;
};

export type ConfigSchema = {
  engine: EngineConfigField[];
  sampling: SamplingConfigField[];
  model: {
    quantization_modes: string[];
    dtype_choices: string[];
  };
};

export type Metrics = {
  uptime_seconds: number;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  total_tokens_generated: number;
  average_tokens_per_second: number;
  active_requests: number;
  models_loaded: number;
  status: string;
  auth_stats: Record<string, unknown>;
  tool_executions: number;
  failed_tool_executions: number;
  admin_actions: number;
};

export type Health = {
  status: string;
  models: string[];
  timestamp: number;
  uptime_seconds: number;
  active_requests: number;
};

export type LoadModelRequest = {
  model_id: string;
  served_name: string | null;
  tokenizer: string | null;
  revision: string | null;
  local_files_only: boolean;
  converted_cache_dir: string | null;
  force_conversion: boolean;
  model_class: string | null;
  replace: boolean;
  model_kwargs: Record<string, unknown>;
  engine_kwargs: Record<string, unknown>;
};

export type ChatRequest = {
  model: string;
  messages: ChatMessage[];
  stream: true;
  max_tokens: number;
  temperature: number;
  top_p: number;
  top_k: number;
  presence_penalty: number;
  repetition_penalty: number;
  stop?: string[];
  tools?: unknown;
  response_format?: unknown;
  stream_options?: { include_usage?: boolean };
  chat_template_kwargs?: Record<string, unknown>;
};

export type ChatUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

export type ChatStreamChunk = {
  choices?: Array<{
    delta?: {
      role?: string;
      content?: string;
      reasoning_content?: string;
      delta_reasoning_content?: string;
      tool_calls?: unknown;
    };
    finish_reason?: string | null;
  }>;
  usage?: ChatUsage | null;
};

export type SamplingState = {
  max_tokens: number;
  temperature: number;
  top_p: number;
  top_k: number;
  presence_penalty: number;
  repetition_penalty: number;
};
