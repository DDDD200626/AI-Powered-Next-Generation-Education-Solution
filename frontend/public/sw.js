/* ClassPulse — 최소 PWA (설치 가능 조건 충족용). 오프라인 캐시 없음. */
self.addEventListener("install", (event) => {
  self.skipWaiting();
});
self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});
