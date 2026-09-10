import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build output is served by demo_service's FastAPI app (StaticFiles on dist/).
// The dev server proxies API paths to it, so `npm run dev` works against a
// running `uvicorn demo_service.api:app --port 8100`.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: {
    proxy: Object.fromEntries(
      ["/health", "/source", "/run", "/suite"].map((p) => [
        p,
        "http://127.0.0.1:8100",
      ])
    ),
  },
});
