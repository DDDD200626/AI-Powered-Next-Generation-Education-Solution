import "./style.css";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

function errDetail(data: unknown): string {
  if (!data || typeof data !== "object") return "요청 실패";
  const d = data as { detail?: unknown };
  if (d.detail == null) return "요청 실패";
  if (typeof d.detail === "string") return d.detail;
  if (Array.isArray(d.detail)) return JSON.stringify(d.detail);
  return String(d.detail);
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = (await r.json().catch(() => ({}))) as T & { detail?: unknown };
  if (!r.ok) throw new Error(errDetail(data));
  return data;
}

function $(sel: string): HTMLElement {
  const el = document.querySelector(sel);
  if (!el) throw new Error(`Missing element: ${sel}`);
  return el as HTMLElement;
}

function button(sel: string): HTMLButtonElement {
  const el = document.querySelector(sel);
  if (!(el instanceof HTMLButtonElement)) throw new Error(`Not a button: ${sel}`);
  return el;
}

/** 중복 클릭 방지 + 로딩 표시 */
async function withBusy(btnEl: HTMLButtonElement, task: () => Promise<void>): Promise<void> {
  if (btnEl.disabled) return;
  const label = btnEl.textContent;
  btnEl.disabled = true;
  btnEl.textContent = "처리 중…";
  try {
    await task();
  } finally {
    btnEl.disabled = false;
    btnEl.textContent = label;
  }
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderTable(container: HTMLElement, columns: string[], rows: Record<string, unknown>[]) {
  if (!rows.length) {
    container.innerHTML = "<p class='note'>데이터가 없습니다.</p>";
    return;
  }
  const th = columns.map((c) => `<th>${escapeHtml(String(c))}</th>`).join("");
  const tr = rows
    .map(
      (row) =>
        "<tr>" +
        columns.map((c) => `<td>${escapeHtml(String(row[c] ?? ""))}</td>`).join("") +
        "</tr>"
    )
    .join("");
  container.innerHTML = `<table class="data"><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>`;
}

interface Hit {
  chunk_id: number;
  source: string;
  text: string;
  score?: number;
}

function renderHits(container: HTMLElement, hits: Hit[] | undefined) {
  container.innerHTML = "";
  if (!hits?.length) {
    container.innerHTML = "<p class='note'>근거가 없습니다.</p>";
    return;
  }
  for (const h of hits) {
    const det = document.createElement("details");
    det.className = "hit";
    det.innerHTML =
      `<summary>[${h.chunk_id}] ${escapeHtml(h.source)} · 유사도 ` +
      (h.score != null ? Number(h.score).toFixed(3) : "") +
      `</summary><pre>${escapeHtml(h.text)}</pre>`;
    container.appendChild(det);
  }
}

function showTab(id: string) {
  document.querySelectorAll(".tab").forEach((t) => {
    const b = t as HTMLButtonElement;
    const on = b.dataset.tab === id;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.querySelectorAll(".panel").forEach((p) => {
    const on = p.id === `panel-${id}`;
    p.classList.toggle("active", on);
    p.setAttribute("aria-hidden", on ? "false" : "true");
  });
}

async function refreshPill() {
  const pill = $("#pill-status");
  try {
    const r = await fetch(apiUrl("/api/health"));
    const j = (await r.json()) as { openai_configured?: boolean };
    pill.className = j.openai_configured ? "pill pill-ok" : "pill pill-warn";
    pill.textContent = j.openai_configured
      ? "OpenAI 서버 키 설정됨"
      : "OPENAI_API_KEY 미설정 — 일부 AI 기능 비활성";
  } catch {
    pill.className = "pill pill-warn";
    pill.textContent = "백엔드에 연결할 수 없습니다 (포트 8000 확인)";
  }
}

async function dashPreview() {
  const csv_text = (document.getElementById("dash-csv") as HTMLTextAreaElement).value;
  const data = await postJSON<{
    note?: string;
    columns: string[];
    rows: Record<string, unknown>[];
    stats: unknown;
  }>("/api/dashboard/preview", { csv_text });
  $("#dash-note").textContent = data.note ?? "";
  renderTable($("#dash-table-wrap"), data.columns, data.rows);
  $("#dash-stats").textContent = JSON.stringify(data.stats, null, 2);
}

async function opsPreview() {
  const csv_text = (document.getElementById("ops-csv") as HTMLTextAreaElement).value;
  const data = await postJSON<{
    note?: string;
    columns: string[];
    rows: Record<string, unknown>[];
    stats: unknown;
  }>("/api/ops/preview", { csv_text });
  $("#ops-note").textContent = data.note ?? "";
  renderTable($("#ops-table-wrap"), data.columns, data.rows);
  $("#ops-stats").textContent = JSON.stringify(data.stats, null, 2);
}

function init() {
  document.querySelectorAll(".tab").forEach((b) => {
    b.addEventListener("click", () => showTab((b as HTMLElement).dataset.tab ?? "dash"));
  });

  void refreshPill();

  button("#btn-learn").addEventListener("click", async () => {
    const q = (document.getElementById("learn-q") as HTMLInputElement).value.trim();
    const k = Math.min(8, Math.max(2, parseInt((document.getElementById("learn-k") as HTMLInputElement).value, 10) || 4));
    $("#learn-hits").innerHTML = "";
    ($("#learn-ans") as HTMLElement).textContent = "";
    if (!q) {
      alert("질문을 입력하세요.");
      return;
    }
    await withBusy(button("#btn-learn"), async () => {
      try {
        const data = await postJSON<{ hits: Hit[]; answer?: string | null; answer_error?: string | null }>(
          "/api/learn",
          { query: q, top_k: k }
        );
        renderHits($("#learn-hits") as HTMLElement, data.hits);
        ($("#learn-ans") as HTMLElement).textContent = data.answer_error ?? data.answer ?? "";
      } catch (e) {
        alert(e instanceof Error ? e.message : String(e));
      }
    });
  });

  button("#btn-comp").addEventListener("click", async () => {
    const question = (document.getElementById("comp-q") as HTMLInputElement).value.trim();
    const answer = (document.getElementById("comp-a") as HTMLTextAreaElement).value.trim();
    const top_k = Math.min(8, Math.max(2, parseInt((document.getElementById("comp-k") as HTMLInputElement).value, 10) || 4));
    $("#comp-hits").innerHTML = "";
    ($("#comp-report") as HTMLElement).textContent = "";
    if (!question || !answer) {
      alert("질문과 답변을 입력하세요.");
      return;
    }
    await withBusy(button("#btn-comp"), async () => {
      try {
        const data = await postJSON<{ hits: Hit[]; report?: string | null; report_error?: string | null }>(
          "/api/comprehension",
          { question, answer, top_k }
        );
        renderHits($("#comp-hits") as HTMLElement, data.hits);
        ($("#comp-report") as HTMLElement).textContent = data.report_error ?? data.report ?? "";
      } catch (e) {
        alert(e instanceof Error ? e.message : String(e));
      }
    });
  });

  button("#btn-integ").addEventListener("click", async () => {
    const submission = (document.getElementById("integ-sub") as HTMLTextAreaElement).value.trim();
    const assignment_prompt = (document.getElementById("integ-prompt") as HTMLTextAreaElement).value;
    $("#integ-heur").textContent = "";
    $("#integ-sim").textContent = "";
    ($("#integ-narr") as HTMLElement).textContent = "";
    if (!submission) {
      alert("제출 본문을 입력하세요.");
      return;
    }
    await withBusy(button("#btn-integ"), async () => {
      try {
        const data = await postJSON<{
          heuristic: unknown;
          max_similarity: number;
          max_similarity_source: string | null;
          narrative?: string | null;
          narrative_error?: string | null;
        }>("/api/integrity", { assignment_prompt, submission });
        $("#integ-heur").textContent = JSON.stringify(data.heuristic, null, 2);
        $("#integ-sim").textContent =
          "자료 대비 최대 유사도(참고): " +
          data.max_similarity +
          (data.max_similarity_source ? " · 출처: " + data.max_similarity_source : "");
        ($("#integ-narr") as HTMLElement).textContent = data.narrative_error ?? data.narrative ?? "";
      } catch (e) {
        alert(e instanceof Error ? e.message : String(e));
      }
    });
  });

  button("#btn-fb").addEventListener("click", async () => {
    const rubric = (document.getElementById("rub") as HTMLTextAreaElement).value;
    const submission = (document.getElementById("sub") as HTMLTextAreaElement).value.trim();
    ($("#fb-out") as HTMLElement).textContent = "";
    if (!submission) {
      alert("학생 제출을 입력하세요.");
      return;
    }
    await withBusy(button("#btn-fb"), async () => {
      try {
        const data = await postJSON<{ draft: string }>("/api/teacher-feedback", { rubric, submission });
        ($("#fb-out") as HTMLElement).textContent = data.draft ?? "";
      } catch (e) {
        alert(e instanceof Error ? e.message : String(e));
      }
    });
  });

  button("#btn-dash-preview").addEventListener("click", async () => {
    await withBusy(button("#btn-dash-preview"), async () => {
      try {
        await dashPreview();
      } catch (e) {
        alert(e instanceof Error ? e.message : String(e));
      }
    });
  });

  button("#btn-dash-sum").addEventListener("click", async () => {
    ($("#dash-summary") as HTMLElement).textContent = "";
    await withBusy(button("#btn-dash-sum"), async () => {
      try {
        await dashPreview();
        const csv_text = (document.getElementById("dash-csv") as HTMLTextAreaElement).value;
        const data = await postJSON<{ summary: string }>("/api/dashboard/summarize", { csv_text });
        ($("#dash-summary") as HTMLElement).textContent = data.summary ?? "";
      } catch (e) {
        alert(e instanceof Error ? e.message : String(e));
      }
    });
  });

  button("#btn-ops-preview").addEventListener("click", async () => {
    await withBusy(button("#btn-ops-preview"), async () => {
      try {
        await opsPreview();
      } catch (e) {
        alert(e instanceof Error ? e.message : String(e));
      }
    });
  });

  button("#btn-ops-sum").addEventListener("click", async () => {
    ($("#ops-summary") as HTMLElement).textContent = "";
    await withBusy(button("#btn-ops-sum"), async () => {
      try {
        await opsPreview();
        const csv_text = (document.getElementById("ops-csv") as HTMLTextAreaElement).value;
        const data = await postJSON<{ summary: string }>("/api/ops/summarize", { csv_text });
        ($("#ops-summary") as HTMLElement).textContent = data.summary ?? "";
      } catch (e) {
        alert(e instanceof Error ? e.message : String(e));
      }
    });
  });

  showTab("dash");
}

init();
