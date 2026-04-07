import { defineConfig } from "vite";

/** 로컬 백엔드(uvicorn :8000) — VITE_API_BASE 비우면 브라우저는 5173 출처로 요청 후 여기로 전달 */
const backend = "http://127.0.0.1:8000";
const toBackend = { target: backend, changeOrigin: true };

const apiProxy: Record<string, { target: string; changeOrigin: boolean }> = {
  "/api": toBackend,
  "/docs": toBackend,
  "/redoc": toBackend,
  "/openapi.json": toBackend,
};

export default defineConfig({
  build: {
    target: "es2020",
    cssMinify: true,
    cssCodeSplit: true,
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          if (id.includes("node_modules")) return "vendor";
          return undefined;
        },
      },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: apiProxy,
  },
  preview: {
    port: 4173,
    strictPort: true,
    proxy: apiProxy,
  },
});
