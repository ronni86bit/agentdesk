import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      // Dev convenience: forward /api/* to the FastAPI backend (same
      // shape as the production nginx proxy) so the app can run with an
      // empty VITE_API_BASE_URL. SSE needs buffering off in production;
      // the Vite dev proxy streams by default.
      "/api": {
        target: process.env.DEV_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./vitest.setup.js",
    globals: true,
    css: false,
  },
});
