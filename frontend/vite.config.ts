import { defineConfig, loadEnv } from "vite";

/**
 * 프론트(5173) ↔ 백엔드 연결
 * - frontend/.env.development 에서 VITE_API_BASE 를 비움 → 브라우저는 /api 로만 요청(같은 출처)
 * - 프록시 대상은 VITE_DEV_BACKEND_URL (기본 http://127.0.0.1:8000)
 * - Windows 등에서 8000 바인딩 실패 시 .env.development.local 에 VITE_DEV_BACKEND_URL=http://127.0.0.1:8010
 * - VITE_API_BASE=http://127.0.0.1:8000 으로 두면 브라우저가 백엔드에 직접 호출(CORS 필요)
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backend = env.VITE_DEV_BACKEND_URL || "http://127.0.0.1:8000";

  const longProxy = {
    target: backend,
    changeOrigin: true,
    secure: false,
    ws: true,
    configure: (proxy: { proxyTimeout?: number; timeout?: number }) => {
      proxy.proxyTimeout = 600_000;
      proxy.timeout = 600_000;
    },
  };

  const apiProxy: Record<string, Record<string, unknown>> = {
    "/api": longProxy,
    "/docs": longProxy,
    "/redoc": longProxy,
    "/openapi.json": longProxy,
  };

  return {
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
      strictPort: false,
      host: true,
      proxy: apiProxy,
    },
    preview: {
      port: 4173,
      strictPort: false,
      proxy: apiProxy,
    },
  };
});
