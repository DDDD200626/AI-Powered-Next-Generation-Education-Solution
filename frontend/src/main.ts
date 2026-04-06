import "./style.css";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

const REPO_URL =
  "https://github.com/DDDD200626/AI-Powered-Next-Generation-Education-Solution";

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

type SiteView = "hub" | "analyze" | "team" | "at-risk" | "feedback" | "about";

interface ProviderKeys {
  gemini?: boolean;
  openai?: boolean;
  claude?: boolean;
  grok?: boolean;
}

interface ModelJudgment {
  provider: string;
  model_label: string;
  ok: boolean;
  error?: string | null;
  cheating_likelihood?: number | null;
  learning_state_summary: string;
  mismatch_analysis: string;
  future_prediction: string;
  confidence_note: string;
}

interface AnalyzeResponse {
  providers_used: string[];
  providers_skipped: string[];
  judgments: ModelJudgment[];
  consensus_cheating_avg: number | null;
  consensus_summary: string;
  disclaimer: string;
}

interface TeamMemberRow {
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

interface TeamDimensionScores {
  technical: number;
  collaboration: number;
  initiative: number;
}

interface TeamMemberOut {
  name: string;
  role?: string;
  contribution_index: number;
  dimensions: TeamDimensionScores;
  evidence_summary?: string;
  caveats?: string;
}

interface TeamEvaluateResponse {
  mode: string;
  members: TeamMemberOut[];
  fairness_notes?: string;
  disclaimer?: string;
}

interface WeekRow {
  week_label: string;
  engagement: string;
  assessment_score: string;
}

interface AtRiskResponse {
  mode: string;
  dropout_risk: number;
  trend_summary: string;
  signals: string[];
  intervention_suggestions: string;
  disclaimer?: string;
}

interface FeedbackResponse {
  mode: string;
  draft_feedback: string;
  strengths: string[];
  improvements: string[];
  disclaimer?: string;
}

function emptyTeamMember(): TeamMemberRow {
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

function emptyWeekRow(): WeekRow {
  return { week_label: "", engagement: "", assessment_score: "" };
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const state: {
  view: SiteView;
  course_name: string;
  student_or_group_label: string;
  weekly_study_hours: string;
  lms_video_watch_ratio: string;
  quiz_average: string;
  assignment_on_time_ratio: string;
  discussion_count: string;
  attendance_ratio: string;
  learning_notes: string;
  midterm_score: string;
  final_score: string;
  exam_time_anomaly_note: string;
  exam_notes: string;
  context_for_educator: string;
  result: AnalyzeResponse | null;
  loading: boolean;
  error: string | null;
  health: { providers?: ProviderKeys } | null;
  team: {
    project_name: string;
    project_description: string;
    evaluation_criteria: string;
    members: TeamMemberRow[];
    result: TeamEvaluateResponse | null;
    loading: boolean;
    error: string | null;
  };
  atRisk: {
    course_name: string;
    student_label: string;
    weeks: WeekRow[];
    notes: string;
    result: AtRiskResponse | null;
    loading: boolean;
    error: string | null;
  };
  feedback: {
    rubric: string;
    assignment_prompt: string;
    submission: string;
    result: FeedbackResponse | null;
    loading: boolean;
    error: string | null;
  };
} = {
  view: "hub",
  course_name: "",
  student_or_group_label: "",
  weekly_study_hours: "",
  lms_video_watch_ratio: "",
  quiz_average: "",
  assignment_on_time_ratio: "",
  discussion_count: "",
  attendance_ratio: "",
  learning_notes: "",
  midterm_score: "",
  final_score: "",
  exam_time_anomaly_note: "",
  exam_notes: "",
  context_for_educator: "",
  result: null,
  loading: false,
  error: null,
  health: null,
  team: {
    project_name: "",
    project_description: "",
    evaluation_criteria: "",
    members: [emptyTeamMember(), emptyTeamMember()],
    result: null,
    loading: false,
    error: null,
  },
  atRisk: {
    course_name: "",
    student_label: "",
    weeks: [emptyWeekRow(), emptyWeekRow(), emptyWeekRow()],
    notes: "",
    result: null,
    loading: false,
    error: null,
  },
  feedback: {
    rubric: "",
    assignment_prompt: "",
    submission: "",
    result: null,
    loading: false,
    error: null,
  },
};

function parseOptFloat(s: string): number | null {
  const t = s.trim();
  if (!t) return null;
  const n = parseFloat(t);
  return Number.isFinite(n) ? n : null;
}

function parseOptInt(s: string): number | null {
  const t = s.trim();
  if (!t) return null;
  const n = parseInt(t, 10);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

async function refreshHealth(): Promise<void> {
  try {
    const r = await fetch(apiUrl("/api/health"));
    state.health = (await r.json()) as { providers?: ProviderKeys };
  } catch {
    state.health = null;
  }
}

async function submitAnalyze(): Promise<void> {
  readForm();
  state.error = null;
  state.result = null;
  state.loading = true;
  render();

  const body = {
    course_name: state.course_name.trim(),
    student_or_group_label: state.student_or_group_label.trim(),
    learning: {
      weekly_study_hours_self_report: parseOptFloat(state.weekly_study_hours),
      lms_video_watch_ratio: parseOptFloat(state.lms_video_watch_ratio),
      quiz_average: parseOptFloat(state.quiz_average),
      assignment_on_time_ratio: parseOptFloat(state.assignment_on_time_ratio),
      discussion_or_forum_count: parseOptInt(state.discussion_count),
      attendance_or_checkin_ratio: parseOptFloat(state.attendance_ratio),
      notes: state.learning_notes,
    },
    exam: {
      midterm_score: parseOptFloat(state.midterm_score),
      final_or_recent_exam_score: parseOptFloat(state.final_score),
      exam_time_anomaly_note: state.exam_time_anomaly_note,
      notes: state.exam_notes,
    },
    context_for_educator: state.context_for_educator.trim(),
  };

  try {
    const r = await fetch(apiUrl("/api/analyze"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as AnalyzeResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.loading = false;
      render();
      return;
    }
    state.result = data;
    state.view = "analyze";
  } catch (e) {
    state.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.loading = false;
  }
  render();
}

function readTeamForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | null;
  state.team.project_name = g("team_project_name")?.value ?? "";
  state.team.project_description = g("team_project_description")?.value ?? "";
  state.team.evaluation_criteria = g("team_evaluation_criteria")?.value ?? "";
  state.team.members = state.team.members.map((_, i) => ({
    name: g(`tm_name_${i}`)?.value ?? "",
    role: g(`tm_role_${i}`)?.value ?? "",
    commits: g(`tm_commits_${i}`)?.value ?? "",
    pull_requests: g(`tm_pr_${i}`)?.value ?? "",
    lines_changed: g(`tm_lines_${i}`)?.value ?? "",
    tasks_completed: g(`tm_tasks_${i}`)?.value ?? "",
    meetings_attended: g(`tm_meet_${i}`)?.value ?? "",
    self_report: g(`tm_self_${i}`)?.value ?? "",
    peer_notes: g(`tm_peer_${i}`)?.value ?? "",
  }));
}

async function submitTeam(): Promise<void> {
  readTeamForm();
  state.team.error = null;
  state.team.result = null;
  state.team.loading = true;
  render();

  const members = state.team.members
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
    }));

  if (!state.team.project_name.trim() || members.length === 0) {
    state.team.error = "프로젝트명과 최소 한 명의 이름을 입력하세요.";
    state.team.loading = false;
    render();
    return;
  }

  const body = {
    project_name: state.team.project_name.trim(),
    project_description: state.team.project_description,
    evaluation_criteria: state.team.evaluation_criteria,
    members,
  };

  try {
    const r = await fetch(apiUrl("/api/team/evaluate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as TeamEvaluateResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.team.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.team.loading = false;
      render();
      return;
    }
    state.team.result = data;
  } catch (e) {
    state.team.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.team.loading = false;
  }
  render();
}

function readAtRiskForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | null;
  state.atRisk.course_name = g("ar_course")?.value ?? "";
  state.atRisk.student_label = g("ar_student")?.value ?? "";
  state.atRisk.notes = g("ar_notes")?.value ?? "";
  state.atRisk.weeks = state.atRisk.weeks.map((_, i) => ({
    week_label: g(`ar_week_${i}`)?.value ?? "",
    engagement: g(`ar_eng_${i}`)?.value ?? "",
    assessment_score: g(`ar_as_${i}`)?.value ?? "",
  }));
}

async function submitAtRisk(): Promise<void> {
  readAtRiskForm();
  state.atRisk.error = null;
  state.atRisk.result = null;
  state.atRisk.loading = true;
  render();

  const weeks = state.atRisk.weeks
    .filter((w) => w.week_label.trim())
    .map((w) => {
      const eng = parseOptFloat(w.engagement);
      if (eng === null) return null;
      const row: { week_label: string; engagement: number; assessment_score?: number } = {
        week_label: w.week_label.trim(),
        engagement: eng,
      };
      const as = parseOptFloat(w.assessment_score);
      if (as !== null) row.assessment_score = as;
      return row;
    })
    .filter((x): x is NonNullable<typeof x> => x !== null);

  if (weeks.length === 0) {
    state.atRisk.error = "주차 라벨과 참여 점수(0–100)를 한 줄 이상 입력하세요.";
    state.atRisk.loading = false;
    render();
    return;
  }

  const body = {
    course_name: state.atRisk.course_name.trim(),
    student_label: state.atRisk.student_label.trim(),
    weeks,
    notes: state.atRisk.notes,
  };

  try {
    const r = await fetch(apiUrl("/api/at-risk/evaluate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as AtRiskResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.atRisk.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.atRisk.loading = false;
      render();
      return;
    }
    state.atRisk.result = data;
  } catch (e) {
    state.atRisk.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.atRisk.loading = false;
  }
  render();
}

function readFeedbackForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLTextAreaElement | null;
  state.feedback.rubric = g("fb_rubric")?.value ?? "";
  state.feedback.assignment_prompt = g("fb_prompt")?.value ?? "";
  state.feedback.submission = g("fb_submission")?.value ?? "";
}

async function submitFeedback(): Promise<void> {
  readFeedbackForm();
  state.feedback.error = null;
  state.feedback.result = null;
  state.feedback.loading = true;
  render();

  if (!state.feedback.rubric.trim() || !state.feedback.submission.trim()) {
    state.feedback.error = "루브릭과 제출물을 입력하세요.";
    state.feedback.loading = false;
    render();
    return;
  }

  const body = {
    rubric: state.feedback.rubric.trim(),
    assignment_prompt: state.feedback.assignment_prompt,
    submission: state.feedback.submission,
  };

  try {
    const r = await fetch(apiUrl("/api/feedback/draft"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as FeedbackResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.feedback.error =
        typeof d === "string"
          ? d
          : d && typeof d === "object" && "detail" in d
            ? String((d as { detail: string }).detail)
            : "요청 실패";
      state.feedback.loading = false;
      render();
      return;
    }
    state.feedback.result = data;
  } catch (e) {
    state.feedback.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.feedback.loading = false;
  }
  render();
}

function providerPills(): string {
  const p = state.health?.providers;
  if (!p) return '<span class="pill pill-muted">백엔드 연결 확인 중…</span>';
  const mk = (name: string, on: boolean | undefined) =>
    `<span class="pill ${on ? "pill-on" : "pill-off"}">${name}</span>`;
  return [mk("Gemini", p.gemini), mk("ChatGPT", p.openai), mk("Claude", p.claude), mk("Grok", p.grok)].join("");
}

function navHtml(): string {
  const cur = (v: SiteView) => (state.view === v ? "nav-link active" : "nav-link");
  return `
  <header class="site-header">
    <div class="nav-inner">
      <a href="#" class="brand brand-mark" data-view="hub" aria-label="홈">
        <span class="brand-title">EduSignal</span>
        <span class="brand-tag">ED · AI SOLUTION LAB</span>
      </a>
      <nav class="nav" aria-label="주요 메뉴">
        <button type="button" class="${cur("hub")}" data-view="hub">허브</button>
        <button type="button" class="${cur("analyze")}" data-view="analyze">과정·시험</button>
        <button type="button" class="${cur("team")}" data-view="team">팀</button>
        <button type="button" class="${cur("at-risk")}" data-view="at-risk">이탈</button>
        <button type="button" class="${cur("feedback")}" data-view="feedback">피드백</button>
        <button type="button" class="${cur("about")}" data-view="about">안내</button>
        <a class="nav-link nav-gh" href="${REPO_URL}" target="_blank" rel="noopener noreferrer">GitHub</a>
      </nav>
    </div>
  </header>`;
}

function footerHtml(): string {
  return `
  <footer class="site-footer">
    <div class="footer-inner">
      <p class="footer-line">
        <a href="${REPO_URL}" target="_blank" rel="noopener noreferrer">저장소</a>
        · 교육 보조 도구이며 징계·단정에 사용할 수 없습니다.
      </p>
      <p class="footer-muted">EduSignal — 교육 현장 페인 포인트를 AI로 보조하는 도구 모음</p>
    </div>
  </footer>`;
}

function hubHtml(): string {
  return `
  <div class="page page-animate home-page">
    <section class="hero-block home-hero-main">
      <p class="eyebrow">Education × AI</p>
      <h1 class="home-headline">도구 허브</h1>
      <p class="hero-text home-lead">
        아래 네 가지는 <strong>실제 API</strong>와 연결되어 있습니다. 과정–시험 분석은 다중 LLM, 팀·이탈은 OpenAI 또는 휴리스틱, 피드백 초안은 OpenAI가 필요합니다.
      </p>
      <div class="pill-row hero-pills">${providerPills()}</div>
    </section>

    <section class="section-block hud-section home-section">
      <h2 class="section-title">기능 선택</h2>
      <div class="solution-grid">
        <article class="solution-tile solution-tile--live hud-card">
          <span class="solution-badge solution-badge--on">다중 LLM</span>
          <h3>과정 vs 시험 불일치</h3>
          <p>LMS·과제 지표와 시험 점수를 넣어 부정행위 <em>의심도</em>·학습 상태·위험을 참고용으로 제시합니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="analyze">열기</button>
        </article>
        <article class="solution-tile solution-tile--live hud-card">
          <span class="solution-badge solution-badge--on">팀</span>
          <h3>팀 기여도 초안</h3>
          <p>커밋·과제·자기·동료 서술을 바탕으로 기여 지수·차원 점수 초안을 만듭니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="team">열기</button>
        </article>
        <article class="solution-tile solution-tile--live hud-card">
          <span class="solution-badge solution-badge--on">조기 경보</span>
          <h3>학습 이탈 신호</h3>
          <p>주차별 참여 점수를 넣어 위험 지수·개입 제안 초안을 봅니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="at-risk">열기</button>
        </article>
        <article class="solution-tile solution-tile--live hud-card">
          <span class="solution-badge solution-badge--on">OpenAI</span>
          <h3>과제 피드백 초안</h3>
          <p>루브릭·제출물로 피드백 문단·강점·개선점 초안을 생성합니다. 교사 검수 후 전달하세요.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="feedback">열기</button>
        </article>
      </div>
    </section>
  </div>`;
}

function aboutHtml(): string {
  return `
  <div class="page page-animate about-page">
    <h1 class="page-title">이용 안내</h1>
    <div class="prose-block">
      <p><strong>목적.</strong> 교수·조교·기관의 <strong>의사결정 보조</strong>입니다. 부정행위 여부는 인간의 심사·증거·절차에 따릅니다.</p>
      <p><strong>API 키.</strong> <code>backend/.env</code>에 Google(Gemini), OpenAI, Anthropic, xAI(Grok) 키를 넣으면 해당 모델이 응답합니다. 키가 없으면 과정–시험·팀·이탈은 휴리스틱을 씁니다. <strong>피드백 초안</strong>은 OpenAI 키가 필요합니다.</p>
      <p><strong>개인정보.</strong> 실명 대신 익명 라벨만 쓰는 것을 권장합니다.</p>
      <p><strong>소스 코드.</strong> <a href="${REPO_URL}" target="_blank" rel="noopener noreferrer">GitHub 저장소</a></p>
    </div>
  </div>`;
}

function teamMemberBlock(i: number, m: TeamMemberRow): string {
  return `
  <div class="member-block hud-panel" data-member-idx="${i}">
    <h4 class="subh">멤버 ${i + 1}</h4>
    <div class="grid-2">
      <div><label class="lbl">이름</label><input class="txt" id="tm_name_${i}" value="${escapeHtml(m.name)}" /></div>
      <div><label class="lbl">역할</label><input class="txt" id="tm_role_${i}" value="${escapeHtml(m.role)}" /></div>
    </div>
    <div class="grid-3">
      <div><label class="lbl">커밋 수</label><input class="txt" id="tm_commits_${i}" type="number" min="0" value="${escapeHtml(m.commits)}" /></div>
      <div><label class="lbl">PR 수</label><input class="txt" id="tm_pr_${i}" type="number" min="0" value="${escapeHtml(m.pull_requests)}" /></div>
      <div><label class="lbl">변경 라인</label><input class="txt" id="tm_lines_${i}" type="number" min="0" value="${escapeHtml(m.lines_changed)}" /></div>
      <div><label class="lbl">완료 태스크</label><input class="txt" id="tm_tasks_${i}" type="number" min="0" value="${escapeHtml(m.tasks_completed)}" /></div>
      <div><label class="lbl">회의 출석</label><input class="txt" id="tm_meet_${i}" type="number" min="0" value="${escapeHtml(m.meetings_attended)}" /></div>
    </div>
    <label class="lbl">자기 서술</label>
    <textarea class="txt" id="tm_self_${i}" rows="2">${escapeHtml(m.self_report)}</textarea>
    <label class="lbl">동료 메모</label>
    <textarea class="txt" id="tm_peer_${i}" rows="2">${escapeHtml(m.peer_notes)}</textarea>
  </div>`;
}

function teamResultHtml(): string {
  const res = state.team.result;
  if (!res) return "";
  const rows = res.members
    .map(
      (m) => `
    <tr>
      <td>${escapeHtml(m.name)}</td>
      <td>${m.contribution_index.toFixed(1)}</td>
      <td>${m.dimensions.technical.toFixed(0)} / ${m.dimensions.collaboration.toFixed(0)} / ${m.dimensions.initiative.toFixed(0)}</td>
      <td class="muted small">${escapeHtml(m.evidence_summary || "")}</td>
    </tr>`
    )
    .join("");
  return `
  <section class="panel panel-result hud-panel">
    <h2>결과 <span class="pill ${res.mode === "ai" ? "pill-on" : "pill-muted"}">${escapeHtml(res.mode)}</span></h2>
    ${res.fairness_notes ? `<p class="prose">${escapeHtml(res.fairness_notes)}</p>` : ""}
    <table class="data-table">
      <thead><tr><th>이름</th><th>기여 지수</th><th>기술·협업·주도</th><th>근거 요약</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="footer-note muted small">${escapeHtml(res.disclaimer || "")}</p>
  </section>`;
}

function teamHtml(): string {
  const blocks = state.team.members.map((m, i) => teamMemberBlock(i, m)).join("");
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">팀 프로젝트</p>
    <h1 class="page-title">기여도 초안</h1>
    <p class="lead analyze-lead">OpenAI 키가 있으면 서술을 반영한 JSON 분석, 없으면 정량 휴리스틱만 사용합니다.</p>

    <section class="panel hud-panel">
      <div class="grid-2">
        <div>
          <label class="lbl">프로젝트명</label>
          <input class="txt" id="team_project_name" value="${escapeHtml(state.team.project_name)}" />
        </div>
      </div>
      <label class="lbl">프로젝트 설명</label>
      <textarea class="txt" id="team_project_description" rows="2">${escapeHtml(state.team.project_description)}</textarea>
      <label class="lbl">평가 기준 (선택)</label>
      <textarea class="txt" id="team_evaluation_criteria" rows="2">${escapeHtml(state.team.evaluation_criteria)}</textarea>
      ${blocks}
      <div class="row-actions">
        <button type="button" class="btn btn-ghost" id="btn-team-add">멤버 추가</button>
        <button type="button" class="btn btn-ghost" id="btn-team-remove" ${state.team.members.length <= 1 ? "disabled" : ""}>마지막 멤버 제거</button>
        <button type="button" class="btn btn-primary" id="btn-team-run" ${state.team.loading ? "disabled" : ""}>
          ${state.team.loading ? "처리 중…" : "평가 실행"}
        </button>
      </div>
      ${state.team.error ? `<p class="err">${escapeHtml(state.team.error)}</p>` : ""}
    </section>
    ${teamResultHtml()}
  </div>`;
}

function weekRowHtml(i: number, w: WeekRow): string {
  return `
  <div class="grid-3 member-block hud-panel" style="padding:1rem;margin-bottom:0.5rem;">
    <div><label class="lbl">주차</label><input class="txt" id="ar_week_${i}" placeholder="예: 3주차" value="${escapeHtml(w.week_label)}" /></div>
    <div><label class="lbl">참여 0–100</label><input class="txt" id="ar_eng_${i}" type="number" min="0" max="100" step="0.1" value="${escapeHtml(w.engagement)}" /></div>
    <div><label class="lbl">소평가·퀴즈 (선택)</label><input class="txt" id="ar_as_${i}" type="number" min="0" max="100" step="0.1" value="${escapeHtml(w.assessment_score)}" /></div>
  </div>`;
}

function atRiskResultHtml(): string {
  const r = state.atRisk.result;
  if (!r) return "";
  const sig = r.signals.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
  return `
  <section class="panel panel-result hud-panel">
    <h2>이탈·위험 지수 <span class="score-num">${r.dropout_risk.toFixed(1)}</span> / 100 <span class="pill ${r.mode === "ai" ? "pill-on" : "pill-muted"}">${escapeHtml(r.mode)}</span></h2>
    <p class="prose">${escapeHtml(r.trend_summary)}</p>
    <h4>신호</h4>
    <ul class="prose">${sig}</ul>
    <h4>개입 제안</h4>
    <p class="prose">${escapeHtml(r.intervention_suggestions)}</p>
    <p class="footer-note muted small">${escapeHtml(r.disclaimer || "")}</p>
  </section>`;
}

function atRiskHtml(): string {
  const weeks = state.atRisk.weeks.map((w, i) => weekRowHtml(i, w)).join("");
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">조기 경보</p>
    <h1 class="page-title">학습 이탈 신호</h1>
    <p class="lead analyze-lead">주차별 참여 지표를 입력하세요. OpenAI 키가 있으면 서술 요약이 붙습니다.</p>
    <section class="panel hud-panel">
      <div class="grid-2">
        <div><label class="lbl">과목 (선택)</label><input class="txt" id="ar_course" value="${escapeHtml(state.atRisk.course_name)}" /></div>
        <div><label class="lbl">학습자 라벨 (선택)</label><input class="txt" id="ar_student" value="${escapeHtml(state.atRisk.student_label)}" /></div>
      </div>
      ${weeks}
      <div class="row-actions">
        <button type="button" class="btn btn-ghost" id="btn-ar-add">주차 행 추가</button>
        <button type="button" class="btn btn-ghost" id="btn-ar-remove" ${state.atRisk.weeks.length <= 1 ? "disabled" : ""}>마지막 행 제거</button>
      </div>
      <label class="lbl">추가 메모</label>
      <textarea class="txt" id="ar_notes" rows="2">${escapeHtml(state.atRisk.notes)}</textarea>
      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-ar-run" ${state.atRisk.loading ? "disabled" : ""}>
          ${state.atRisk.loading ? "분석 중…" : "평가 실행"}
        </button>
      </div>
      ${state.atRisk.error ? `<p class="err">${escapeHtml(state.atRisk.error)}</p>` : ""}
    </section>
    ${atRiskResultHtml()}
  </div>`;
}

function feedbackHtml(): string {
  const r = state.feedback.result;
  const resultBlock = r
    ? `
  <section class="panel panel-result hud-panel">
    <h2>피드백 초안</h2>
    <p class="prose">${escapeHtml(r.draft_feedback)}</p>
    <h4>강점</h4>
    <ul>${r.strengths.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    <h4>개선</h4>
    <ul>${r.improvements.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    <p class="footer-note muted small">${escapeHtml(r.disclaimer || "")}</p>
  </section>`
    : "";
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">과제</p>
    <h1 class="page-title">피드백 초안 생성</h1>
    <p class="lead analyze-lead"><code>OPENAI_API_KEY</code>가 필요합니다.</p>
    <section class="panel hud-panel">
      <label class="lbl">루브릭·채점 기준</label>
      <textarea class="txt" id="fb_rubric" rows="4">${escapeHtml(state.feedback.rubric)}</textarea>
      <label class="lbl">과제 설명 (선택)</label>
      <textarea class="txt" id="fb_prompt" rows="3">${escapeHtml(state.feedback.assignment_prompt)}</textarea>
      <label class="lbl">학생 제출물</label>
      <textarea class="txt" id="fb_submission" rows="8">${escapeHtml(state.feedback.submission)}</textarea>
      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-fb-run" ${state.feedback.loading ? "disabled" : ""}>
          ${state.feedback.loading ? "생성 중…" : "초안 생성"}
        </button>
      </div>
      ${state.feedback.error ? `<p class="err">${escapeHtml(state.feedback.error)}</p>` : ""}
    </section>
    ${resultBlock}
  </div>`;
}

function resultSectionHtml(): string {
  if (!state.result) return "";
  const res = state.result;
  const avg =
    res.consensus_cheating_avg != null
      ? `<p class="consensus"><strong>합의 평균 부정행위 의심도:</strong> ${res.consensus_cheating_avg.toFixed(1)} / 100</p>`
      : "";
  const cards = res.judgments
    .map((j) => {
      if (!j.ok) {
        return `<div class="card card-err"><h3>${escapeHtml(j.provider)}</h3><p class="err">${escapeHtml(j.error || "오류")}</p></div>`;
      }
      const score =
        j.cheating_likelihood != null
          ? `<div class="score-line">부정행위 의심도 <span class="score-num">${j.cheating_likelihood.toFixed(1)}</span> / 100</div>`
          : "";
      return `<div class="card">
        <div class="card-head">
          <h3>${escapeHtml(j.provider)}</h3>
          <span class="model-tag">${escapeHtml(j.model_label || "")}</span>
        </div>
        ${score}
        <h4>학습 상태</h4>
        <p class="prose">${escapeHtml(j.learning_state_summary)}</p>
        <h4>과정–시험 불일치</h4>
        <p class="prose">${escapeHtml(j.mismatch_analysis)}</p>
        <h4>미래 예측·위험</h4>
        <p class="prose future">${escapeHtml(j.future_prediction)}</p>
        <p class="muted small">${escapeHtml(j.confidence_note)}</p>
      </div>`;
    })
    .join("");
  const skippedHtml =
    res.providers_skipped.length > 0
      ? `<p class="skipped muted small">건너뜀: ${escapeHtml(res.providers_skipped.join(" · "))}</p>`
      : "";
  return `
  <section class="panel panel-result hud-panel" id="results">
    <h2>분석 결과</h2>
    <p class="prose">${escapeHtml(res.consensus_summary)}</p>
    ${avg}
    ${skippedHtml}
    <div class="card-grid">${cards}</div>
    <p class="footer-note">${escapeHtml(res.disclaimer)}</p>
  </section>`;
}

function analyzeHtml(): string {
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">입력</p>
    <h1 class="page-title">과정 지표와 시험 점수</h1>
    <p class="lead analyze-lead">아래 항목을 채운 뒤 실행하세요. 상단 알약은 백엔드에 연결된 API 키 상태입니다.</p>
    <div class="pill-row">${providerPills()}</div>

    <section class="panel hud-panel">
      <div class="grid-2">
        <div>
          <label class="lbl">과목명 (선택)</label>
          <input class="txt" id="course_name" value="${escapeHtml(state.course_name)}" />
        </div>
        <div>
          <label class="lbl">식별 라벨 (익명, 선택)</label>
          <input class="txt" id="student_or_group_label" value="${escapeHtml(state.student_or_group_label)}" placeholder="예: 학습자-042" />
        </div>
      </div>
      <h3 class="subh">학습 과정</h3>
      <div class="grid-3">
        <div>
          <label class="lbl">주간 학습시간 (자기보고)</label>
          <input class="txt" id="weekly_study_hours" type="number" step="0.1" min="0" value="${escapeHtml(state.weekly_study_hours)}" />
        </div>
        <div>
          <label class="lbl">강의 시청률 %</label>
          <input class="txt" id="lms_video_watch_ratio" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.lms_video_watch_ratio)}" />
        </div>
        <div>
          <label class="lbl">캐퀴즈·소평균 %</label>
          <input class="txt" id="quiz_average" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.quiz_average)}" />
        </div>
        <div>
          <label class="lbl">과제 기한 내 제출 비율 %</label>
          <input class="txt" id="assignment_on_time_ratio" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.assignment_on_time_ratio)}" />
        </div>
        <div>
          <label class="lbl">토론·포럼 참여 횟수</label>
          <input class="txt" id="discussion_count" type="number" min="0" value="${escapeHtml(state.discussion_count)}" />
        </div>
        <div>
          <label class="lbl">출석·출첵 비율 %</label>
          <input class="txt" id="attendance_ratio" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.attendance_ratio)}" />
        </div>
      </div>
      <label class="lbl">학습 과정 메모</label>
      <textarea class="txt" id="learning_notes" rows="2">${escapeHtml(state.learning_notes)}</textarea>

      <h3 class="subh">시험·평가</h3>
      <div class="grid-2">
        <div>
          <label class="lbl">중간고사 %</label>
          <input class="txt" id="midterm_score" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.midterm_score)}" />
        </div>
        <div>
          <label class="lbl">기말/최근 시험 %</label>
          <input class="txt" id="final_score" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.final_score)}" />
        </div>
      </div>
      <label class="lbl">시험 시간·제출 이상 (선택)</label>
      <input class="txt" id="exam_time_anomaly_note" value="${escapeHtml(state.exam_time_anomaly_note)}" placeholder="예: 매우 짧은 시간에 고득점" />
      <label class="lbl">시험 메모</label>
      <textarea class="txt" id="exam_notes" rows="2">${escapeHtml(state.exam_notes)}</textarea>

      <label class="lbl">교수·조교 맥락 (선택)</label>
      <textarea class="txt" id="context_for_educator" rows="2" placeholder="난이도, 오픈북 여부 등">${escapeHtml(state.context_for_educator)}</textarea>

      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-analyze" ${state.loading ? "disabled" : ""}>
          ${state.loading ? "분석 중…" : "다중 AI 분석 실행"}
        </button>
      </div>
      ${state.error ? `<p class="err">${escapeHtml(state.error)}</p>` : ""}
    </section>
    ${resultSectionHtml()}
  </div>`;
}

function mainContentHtml(): string {
  switch (state.view) {
    case "hub":
      return hubHtml();
    case "about":
      return aboutHtml();
    case "team":
      return teamHtml();
    case "at-risk":
      return atRiskHtml();
    case "feedback":
      return feedbackHtml();
    default:
      return analyzeHtml();
  }
}

function render(): void {
  const app = document.getElementById("app");
  if (!app) return;
  app.innerHTML = `
  <div class="site game-studio">
    <div class="ambient-bg" aria-hidden="true"></div>
    <div class="scanlines" aria-hidden="true"></div>
    ${navHtml()}
    <main class="site-main">${mainContentHtml()}</main>
    ${footerHtml()}
  </div>`;
  wire();
}

function readForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | null;
  state.course_name = g("course_name")?.value ?? "";
  state.student_or_group_label = g("student_or_group_label")?.value ?? "";
  state.weekly_study_hours = g("weekly_study_hours")?.value ?? "";
  state.lms_video_watch_ratio = g("lms_video_watch_ratio")?.value ?? "";
  state.quiz_average = g("quiz_average")?.value ?? "";
  state.assignment_on_time_ratio = g("assignment_on_time_ratio")?.value ?? "";
  state.discussion_count = g("discussion_count")?.value ?? "";
  state.attendance_ratio = g("attendance_ratio")?.value ?? "";
  state.learning_notes = g("learning_notes")?.value ?? "";
  state.midterm_score = g("midterm_score")?.value ?? "";
  state.final_score = g("final_score")?.value ?? "";
  state.exam_time_anomaly_note = g("exam_time_anomaly_note")?.value ?? "";
  state.exam_notes = g("exam_notes")?.value ?? "";
  state.context_for_educator = g("context_for_educator")?.value ?? "";
}

function setView(v: SiteView): void {
  if (state.view === "analyze") {
    readForm();
  }
  if (state.view === "team") {
    readTeamForm();
  }
  if (state.view === "at-risk") {
    readAtRiskForm();
  }
  if (state.view === "feedback") {
    readFeedbackForm();
  }
  state.view = v;
  void refreshHealth().then(() => {
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function wire(): void {
  document.querySelectorAll("[data-view]").forEach((el) => {
    el.addEventListener("click", (e) => {
      const v = (el as HTMLElement).dataset.view as SiteView | undefined;
      if (!v || v === state.view) {
        if ((el as HTMLElement).classList.contains("brand")) e.preventDefault();
        return;
      }
      e.preventDefault();
      setView(v);
    });
  });

  document.getElementById("btn-analyze")?.addEventListener("click", () => {
    readForm();
    void submitAnalyze();
  });

  document.getElementById("btn-team-add")?.addEventListener("click", () => {
    readTeamForm();
    state.team.members.push(emptyTeamMember());
    state.view = "team";
    render();
  });

  document.getElementById("btn-team-remove")?.addEventListener("click", () => {
    readTeamForm();
    if (state.team.members.length > 1) state.team.members.pop();
    state.view = "team";
    render();
  });

  document.getElementById("btn-team-run")?.addEventListener("click", () => {
    void submitTeam();
  });

  document.getElementById("btn-ar-add")?.addEventListener("click", () => {
    readAtRiskForm();
    state.atRisk.weeks.push(emptyWeekRow());
    state.view = "at-risk";
    render();
  });

  document.getElementById("btn-ar-remove")?.addEventListener("click", () => {
    readAtRiskForm();
    if (state.atRisk.weeks.length > 1) state.atRisk.weeks.pop();
    state.view = "at-risk";
    render();
  });

  document.getElementById("btn-ar-run")?.addEventListener("click", () => {
    void submitAtRisk();
  });

  document.getElementById("btn-fb-run")?.addEventListener("click", () => {
    void submitFeedback();
  });
}

void refreshHealth().then(() => render());
