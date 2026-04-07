import { defineConfig } from "vite";

/**
 * 프론트(5173) ↔ 백엔드(8000) 연결 방식
 * - frontend/.env.development 에서 VITE_API_BASE 를 비움 → 브라우저는 /api 로만 요청(같은 출처)
 * - 아래 proxy 가 /api·/docs 등을 http://127.0.0.1:8000 으로 넘김
 * - VITE_API_BASE=http://127.0.0.1:8000 으로 두면 브라우저가 백엔드에 직접 호출(CORS 필요)
 */
const backend = "http://127.0.0.1:8000";
const toBackend = { target: backend, changeOrigin: true, secure: false, ws: true };

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
    /** dev.ps1에서 포트를 비워 항상 5173으로 고정 실행 */
    strictPort: true,
    /** true: 같은 Wi‑Fi에서 PC IP로 접속 가능 (터미널에 Network URL 표시) */
    host: true,
    proxy: apiProxy,
  },
  preview: {
    port: 4173,
    strictPort: false,
    proxy: apiProxy,
  },
});
