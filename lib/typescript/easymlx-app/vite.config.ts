import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const apiTarget = process.env.EASYMLX_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../../python/easymlx/app/static",
    emptyOutDir: true,
    sourcemap: true
  },
  server: {
    proxy: {
      "/app/api": apiTarget,
      "/v1": apiTarget,
      "/health": apiTarget,
      "/metrics": apiTarget
    }
  }
});
