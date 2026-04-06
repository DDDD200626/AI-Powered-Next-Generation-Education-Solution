import "./style.css";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

const REPO_URL =
  "https://github.com/DDDD200626/AI-Powered-Next-Generation-Education-Solution";

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

type SiteView = "home" | "analyze" | "about";

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
} = {
  view: "home",
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
      <a href="#" class="brand" data-view="home" aria-label="홈">EduSignal</a>
      <nav class="nav" aria-label="주요 메뉴">
        <button type="button" class="${cur("home")}" data-view="home">홈</button>
        <button type="button" class="${cur("analyze")}" data-view="analyze">분석</button>
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
      <p class="footer-muted">EduSignal — 학습 과정과 시험 불일치 다중 AI 분석</p>
    </div>
  </footer>`;
}

function homeHtml(): string {
  return `
  <div class="page home-page">
    <section class="hero-block">
      <p class="eyebrow">Next-Gen Education · Multi-LLM</p>
      <h1>학습은 낮은데 시험만 높을 때,<br />데이터와 AI가 함께 짚어봅니다.</h1>
      <p class="hero-text">
        LMS·과제·출석 등 <strong>과정 지표</strong>와 <strong>시험 결과</strong>를 넣으면
        Gemini · ChatGPT · Claude · Grok이 병렬로 불일치를 해석하고, 부정행위 <em>의심 수준</em>·학습 상태·미래 위험을 제안합니다.
      </p>
      <div class="hero-actions">
        <button type="button" class="btn btn-primary" data-view="analyze">분석 시작</button>
        <button type="button" class="btn btn-ghost" data-view="about">이용 안내</button>
      </div>
      <div class="pill-row hero-pills">${providerPills()}</div>
    </section>

    <section class="section-block">
      <h2 class="section-title">왜 여러 AI인가요</h2>
      <div class="feature-grid">
        <article class="feature-card">
          <span class="feature-ico" aria-hidden="true">◇</span>
          <h3>불일치 진단</h3>
          <p>과정 대비 시험 편차가 클 때, 단순 수치만으로 놓치기 쉬운 패턴을 서술과 함께 봅니다.</p>
        </article>
        <article class="feature-card">
          <span class="feature-ico" aria-hidden="true">◆</span>
          <h3>학습 상태</h3>
          <p>참여·이해·습관 측면에서 교육적으로 의미 있는 요약을 시도합니다.</p>
        </article>
        <article class="feature-card">
          <span class="feature-ico" aria-hidden="true">○</span>
          <h3>미래 예측</h3>
          <p>이 패턴이 이어질 때 이후 시험·학습에 대한 위험·개선 방향을 짧게 예측합니다.</p>
        </article>
      </div>
    </section>

    <section class="section-block cta-block">
      <h2 class="section-title">백엔드를 켜고 분석하세요</h2>
      <p class="muted">로컬에서는 <code>uvicorn learning_analysis.main:app --port 8000</code> 후 이 페이지를 열면 API 키 상태가 위에 표시됩니다.</p>
      <button type="button" class="btn btn-primary" data-view="analyze">입력 화면으로</button>
    </section>
  </div>`;
}

function aboutHtml(): string {
  return `
  <div class="page about-page">
    <h1 class="page-title">이용 안내</h1>
    <div class="prose-block">
      <p><strong>목적.</strong> 교수·조교·기관의 <strong>의사결정 보조</strong>입니다. 부정행위 여부는 인간의 심사·증거·절차에 따릅니다.</p>
      <p><strong>API 키.</strong> <code>backend/.env</code>에 Google(Gemini), OpenAI, Anthropic, xAI(Grok) 키를 넣으면 해당 모델이 함께 응답합니다. 키가 없으면 규칙 기반 휴리스틱만 사용됩니다.</p>
      <p><strong>개인정보.</strong> 실명 대신 익명 라벨만 쓰는 것을 권장합니다.</p>
      <p><strong>소스 코드.</strong> <a href="${REPO_URL}" target="_blank" rel="noopener noreferrer">GitHub 저장소</a>에서 확인·이슈·PR을 남길 수 있습니다.</p>
    </div>
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
  <section class="panel panel-result" id="results">
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
  <div class="page analyze-page">
    <p class="eyebrow">입력</p>
    <h1 class="page-title">과정 지표와 시험 점수</h1>
    <p class="lead analyze-lead">아래 항목을 채운 뒤 실행하세요. 상단 알약은 백엔드에 연결된 API 키 상태입니다.</p>
    <div class="pill-row">${providerPills()}</div>

    <section class="panel">
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
    case "home":
      return homeHtml();
    case "about":
      return aboutHtml();
    default:
      return analyzeHtml();
  }
}

function render(): void {
  const app = document.getElementById("app");
  if (!app) return;
  app.innerHTML = `
  <div class="site">
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
}

void refreshHealth().then(() => render());
