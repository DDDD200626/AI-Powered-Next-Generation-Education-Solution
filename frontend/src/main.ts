import "./style.css";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

interface MemberForm {
  name: string;
  role: string;
  commits: string;
  pull_requests: string;
  lines_changed: string;
  tasks_completed: string;
  meetings_attended: string;
  self_report: string;
  peer_notes: string;
}

interface HealthJson {
  status?: string;
  openai_configured?: boolean;
}

interface DimensionScores {
  technical: number;
  collaboration: number;
  initiative: number;
}

interface MemberEvaluation {
  name: string;
  role: string;
  contribution_index: number;
  dimensions: DimensionScores;
  evidence_summary: string;
  caveats: string;
}

interface EvaluateResponse {
  mode: string;
  project_summary: string;
  fairness_notes: string;
  members: MemberEvaluation[];
  disclaimer: string;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function emptyMember(): MemberForm {
  return {
    name: "",
    role: "",
    commits: "",
    pull_requests: "",
    lines_changed: "",
    tasks_completed: "",
    meetings_attended: "",
    self_report: "",
    peer_notes: "",
  };
}

const state: {
  project_name: string;
  project_description: string;
  evaluation_criteria: string;
  members: MemberForm[];
  result: EvaluateResponse | null;
  loading: boolean;
  error: string | null;
  health: HealthJson | null;
} = {
  project_name: "",
  project_description: "",
  evaluation_criteria: "",
  members: [emptyMember(), emptyMember()],
  result: null,
  loading: false,
  error: null,
  health: null,
};

function parseOptInt(s: string): number | null {
  const t = s.trim();
  if (!t) return null;
  const n = parseInt(t, 10);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

async function refreshHealth(): Promise<void> {
  try {
    const r = await fetch(apiUrl("/api/health"));
    state.health = (await r.json()) as HealthJson;
  } catch {
    state.health = null;
  }
}

async function submitEvaluate(): Promise<void> {
  state.error = null;
  state.result = null;
  const nameOk = state.members.some((m) => m.name.trim());
  if (!state.project_name.trim() || !nameOk) {
    state.error = "프로젝트 이름과 최소 한 명의 이름을 입력하세요.";
    render();
    return;
  }

  const body = {
    project_name: state.project_name.trim(),
    project_description: state.project_description.trim(),
    evaluation_criteria: state.evaluation_criteria.trim(),
    members: state.members
      .filter((m) => m.name.trim())
      .map((m) => ({
        name: m.name.trim(),
        role: m.role.trim(),
        commits: parseOptInt(m.commits),
        pull_requests: parseOptInt(m.pull_requests),
        lines_changed: parseOptInt(m.lines_changed),
        tasks_completed: parseOptInt(m.tasks_completed),
        meetings_attended: parseOptInt(m.meetings_attended),
        self_report: m.self_report,
        peer_notes: m.peer_notes,
      })),
  };

  state.loading = true;
  render();

  try {
    const r = await fetch(apiUrl("/api/evaluate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as EvaluateResponse & { detail?: unknown };
    if (!r.ok) {
      const detail = data.detail;
      state.error =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? JSON.stringify(detail)
            : "요청에 실패했습니다.";
      state.loading = false;
      render();
      return;
    }
    state.result = data;
  } catch (e) {
    state.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.loading = false;
  }
  render();
}

function render(): void {
  const app = document.getElementById("app");
  if (!app) return;

  const h = state.health;
  const pill =
    h?.openai_configured === true
      ? '<span class="pill ok">OpenAI 연결됨 — AI 평가 모드</span>'
      : '<span class="pill warn">OPENAI_API_KEY 없음 — 휴리스틱 평가</span>';

  const membersHtml = state.members
    .map(
      (m, i) => `
    <div class="member-card" data-idx="${i}">
      <div class="member-card-head">
        <h3 class="member-card-title">멤버 ${i + 1}</h3>
        ${
          state.members.length > 1
            ? `<button type="button" class="btn btn-danger btn-remove" data-idx="${i}">삭제</button>`
            : ""
        }
      </div>
      <div class="grid-2">
        <div>
          <label class="lbl" for="name-${i}">이름</label>
          <input class="txt" id="name-${i}" data-f="name" data-i="${i}" value="${escapeHtml(m.name)}" placeholder="홍길동" />
        </div>
        <div>
          <label class="lbl" for="role-${i}">역할</label>
          <input class="txt" id="role-${i}" data-f="role" data-i="${i}" value="${escapeHtml(m.role)}" placeholder="백엔드, UI, 기획…" />
        </div>
      </div>
      <div class="mini-grid" style="margin-top:0.65rem">
        <div>
          <label class="lbl">커밋 수</label>
          <input class="txt" type="number" min="0" data-f="commits" data-i="${i}" value="${escapeHtml(m.commits)}" placeholder="선택" />
        </div>
        <div>
          <label class="lbl">PR 수</label>
          <input class="txt" type="number" min="0" data-f="pull_requests" data-i="${i}" value="${escapeHtml(m.pull_requests)}" placeholder="선택" />
        </div>
        <div>
          <label class="lbl">변경 라인(근사)</label>
          <input class="txt" type="number" min="0" data-f="lines_changed" data-i="${i}" value="${escapeHtml(m.lines_changed)}" placeholder="선택" />
        </div>
        <div>
          <label class="lbl">완료 태스크</label>
          <input class="txt" type="number" min="0" data-f="tasks_completed" data-i="${i}" value="${escapeHtml(m.tasks_completed)}" placeholder="선택" />
        </div>
        <div>
          <label class="lbl">회의 참여</label>
          <input class="txt" type="number" min="0" data-f="meetings_attended" data-i="${i}" value="${escapeHtml(m.meetings_attended)}" placeholder="선택" />
        </div>
      </div>
      <div style="margin-top:0.65rem">
        <label class="lbl">본인 기여 서술</label>
        <textarea class="txt" data-f="self_report" data-i="${i}" rows="3" placeholder="맡은 일, 구현 내용, 협업 방식 등">${escapeHtml(m.self_report)}</textarea>
      </div>
      <div style="margin-top:0.65rem">
        <label class="lbl">동료/팀 메모 (선택)</label>
        <textarea class="txt" data-f="peer_notes" data-i="${i}" rows="2" placeholder="익명 피드백 요약 등">${escapeHtml(m.peer_notes)}</textarea>
      </div>
    </div>`
    )
    .join("");

  let resultHtml = "";
  if (state.result) {
    const res = state.result;
    const modeBadge =
      res.mode === "ai"
        ? '<span class="badge badge-ai">AI 평가</span>'
        : '<span class="badge badge-heuristic">휴리스틱</span>';

    const cards = res.members
      .map((mem) => {
        const d = mem.dimensions;
        return `
      <div class="result-card">
        <h3>${escapeHtml(mem.name)}</h3>
        <p class="result-role">${escapeHtml(mem.role || "역할 미입력")}</p>
        <div class="score-label">종합 기여 지수</div>
        <div class="score-big">${mem.contribution_index.toFixed(1)}</div>
        <div class="dim-row">
          <span>기술·구현</span><span>${d.technical.toFixed(0)}</span>
          <div class="dim-bar"><span style="width:${Math.min(100, d.technical)}%"></span></div>
          <span>협업</span><span>${d.collaboration.toFixed(0)}</span>
          <div class="dim-bar"><span style="width:${Math.min(100, d.collaboration)}%"></span></div>
          <span>주도성</span><span>${d.initiative.toFixed(0)}</span>
          <div class="dim-bar"><span style="width:${Math.min(100, d.initiative)}%"></span></div>
        </div>
        <p class="muted" style="margin-top:0.75rem">${escapeHtml(mem.evidence_summary)}</p>
        ${mem.caveats ? `<p class="muted">${escapeHtml(mem.caveats)}</p>` : ""}
      </div>`;
      })
      .join("");

    resultHtml = `
    <section class="panel">
      <div class="results-header">
        <h2 style="margin:0">평가 결과</h2>
        ${modeBadge}
      </div>
      <p class="prose">${escapeHtml(res.project_summary || "—")}</p>
      <p class="prose"><strong>공정성·한계:</strong> ${escapeHtml(res.fairness_notes || "—")}</p>
      <div class="result-grid">${cards}</div>
      <p class="footer-note">${escapeHtml(res.disclaimer)}</p>
    </section>`;
  }

  app.innerHTML = `
    <header class="site-header">
      <h1 class="site-title">팀 프로젝트 기여도 자동 평가</h1>
      <p class="site-lead">
        Git 수치·태스크·자기·동료 서술을 묶어 기여도를 추정합니다. OpenAI 키가 있으면 루브릭에 가깝게 서술을 반영하고, 없으면 정량 휴리스틱만 사용합니다.
      </p>
      <div class="pill-row">${pill}</div>
    </header>

    <section class="panel">
      <h2>프로젝트</h2>
      <div class="grid-2">
        <div>
          <label class="lbl" for="project_name">프로젝트 이름</label>
          <input class="txt" id="project_name" value="${escapeHtml(state.project_name)}" placeholder="예: 캡스톤 A팀" />
        </div>
        <div>
          <label class="lbl" for="project_description">목표·범위 (선택)</label>
          <textarea class="txt" id="project_description" rows="2" placeholder="과제 요구사항 요약">${escapeHtml(state.project_description)}</textarea>
        </div>
      </div>
      <div style="margin-top:1rem">
        <label class="lbl" for="evaluation_criteria">추가 평가 기준 (선택)</label>
        <textarea class="txt" id="evaluation_criteria" rows="2" placeholder="교수 루브릭, 배점 비율 등">${escapeHtml(state.evaluation_criteria)}</textarea>
      </div>
    </section>

    <section class="panel">
      <h2>팀원</h2>
      ${membersHtml}
      <button type="button" class="btn btn-ghost" id="btn-add-member">+ 멤버 추가</button>
      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-eval" ${state.loading ? "disabled" : ""}>
          ${state.loading ? "평가 중…" : "평가 실행"}
        </button>
      </div>
      ${state.error ? `<p class="err">${escapeHtml(state.error)}</p>` : ""}
    </section>
    ${resultHtml}
  `;

  wire();
}

function readFormFromDom(): void {
  const pn = document.getElementById("project_name") as HTMLInputElement | null;
  const pd = document.getElementById("project_description") as HTMLTextAreaElement | null;
  const ec = document.getElementById("evaluation_criteria") as HTMLTextAreaElement | null;
  if (pn) state.project_name = pn.value;
  if (pd) state.project_description = pd.value;
  if (ec) state.evaluation_criteria = ec.value;

  document.querySelectorAll("[data-f][data-i]").forEach((el) => {
    const f = el.getAttribute("data-f") as keyof MemberForm;
    const i = parseInt(el.getAttribute("data-i") || "-1", 10);
    if (i < 0 || i >= state.members.length) return;
    const row = state.members[i];
    if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
      row[f] = el.value as never;
    }
  });
}

function wire(): void {
  document.getElementById("btn-add-member")?.addEventListener("click", () => {
    readFormFromDom();
    state.members.push(emptyMember());
    render();
  });

  document.querySelectorAll(".btn-remove").forEach((btn) => {
    btn.addEventListener("click", () => {
      readFormFromDom();
      const idx = parseInt((btn as HTMLElement).dataset.idx || "-1", 10);
      if (idx >= 0 && state.members.length > 1) {
        state.members.splice(idx, 1);
        render();
      }
    });
  });

  document.getElementById("btn-eval")?.addEventListener("click", () => {
    readFormFromDom();
    void submitEvaluate();
  });
}

void refreshHealth().then(() => render());
