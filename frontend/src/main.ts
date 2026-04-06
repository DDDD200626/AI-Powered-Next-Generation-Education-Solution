import "./style.css";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

const REPO_URL =
  "https://github.com/DDDD200626/AI-Powered-Next-Generation-Education-Solution";

/** 심사기준 — 평가 요청 시 AI·문서에 반영되도록 기본 문구 */
const DEFAULT_TEAM_EVALUATION_CRITERIA = `심사기준(평가 시 참고):
■ 기술적 완성도 — 시스템 구조·API·시각화·예외 처리
■ AI 활용 능력 및 효율성 — 다중 모델·생성형 보강(선택)·휴리스틱 폴백
■ 기획력 및 실무 접합성 — 팀 과제·교육 운영(지표·동료·결과 점수·협업)과의 연결
■ 창의성 — 기여-결과 불일치·협업 네트워크·역할 유형·고급 이상 탐지·규칙 기반 설명 카드·팀 역할 밸런스·면담 질문 키트·가상 시뮬레이터`;

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

type SiteView =
  | "hub"
  | "analyze"
  | "team"
  | "at-risk"
  | "feedback"
  | "syllabus"
  | "discussion"
  | "rubric"
  | "llm";

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
  /** 프로젝트·동료 평가 등 결과 점수(선택, 기여 추정과 비교) */
  outcome_score: string;
  self_report: string;
  peer_notes: string;
  /** 주차별 활동(선택): 입력 시 타임라인에 반영 */
  timeline: { period_label: string; activity_score: string }[];
}

interface TeamDimensionScores {
  technical: number;
  collaboration: number;
  initiative: number;
}

interface TimelinePointOut {
  period_label: string;
  share_percent: number;
  activity_score?: number | null;
}

interface TeamMemberOut {
  name: string;
  role?: string;
  contribution_index: number;
  dimensions: TeamDimensionScores;
  evidence_summary?: string;
  caveats?: string;
  free_rider_suspected?: boolean;
  free_rider_risk?: number;
  free_rider_signals?: string[];
  ai_feedback?: string;
  timeline?: TimelinePointOut[];
  contribution_type_label?: string;
  role_scores?: Record<string, number>;
}

interface NetworkNode {
  id: string;
  label: string;
  x: number;
  y: number;
  contribution_index: number;
}

interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
}

interface NetworkGraph {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

interface MismatchItem {
  member_name: string;
  contribution_index: number;
  outcome_score?: number | null;
  gap?: number | null;
  note?: string;
  severity?: string;
}

interface AnomalyAlert {
  member_name: string;
  code: string;
  severity?: string;
  message?: string;
}

interface MemberExplainFact {
  member_name: string;
  facts: string[];
}

interface TeamRoleBalance {
  dev: number;
  doc: number;
  leader: number;
  supporter: number;
  balance_hint: string;
}

interface ReflectionKit {
  team_storyline: string;
  teacher_questions: string[];
  encouragement_line: string;
}

interface CreativeInsights {
  explain_facts: MemberExplainFact[];
  team_role_balance: TeamRoleBalance;
  reflection_kit: ReflectionKit;
}

interface TeamEvaluateResponse {
  mode: string;
  members: TeamMemberOut[];
  fairness_notes?: string;
  free_rider_summary?: string;
  collaboration_network?: NetworkGraph;
  contribution_outcome_summary?: string;
  mismatches?: MismatchItem[];
  anomaly_alerts?: AnomalyAlert[];
  advanced_mode?: string;
  creative_insights?: CreativeInsights;
  request_id?: string;
  generated_at?: string;
  processing_ms?: number;
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

interface CourseAskResponse {
  mode: string;
  answer_draft: string;
  citations: string[];
  caveats?: string;
}

interface DiscussionSynthesizeResponse {
  mode: string;
  summary: string;
  themes: string[];
  participation_notes: string;
  suggested_followups: string[];
  disclaimer?: string;
}

interface RubricAlignResponse {
  mode: string;
  alignment_score: number;
  matched_rubric_points: string[];
  gaps: string[];
  suggestions: string;
  disclaimer?: string;
}

interface LLMTextResult {
  provider: string;
  model_label: string;
  ok: boolean;
  text: string;
  error?: string | null;
}

interface LLMCompareResponse {
  providers_used: string[];
  providers_skipped: string[];
  results: LLMTextResult[];
  disclaimer: string;
}

function emptyTimelineRow(): { period_label: string; activity_score: string } {
  return { period_label: "", activity_score: "" };
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
    outcome_score: "",
    self_report: "",
    peer_notes: "",
    timeline: [
      emptyTimelineRow(),
      emptyTimelineRow(),
      emptyTimelineRow(),
      emptyTimelineRow(),
    ],
  };
}

function emptyWeekRow(): WeekRow {
  return { week_label: "", engagement: "", assessment_score: "" };
}

function emptyDiscussionPost(): { author: string; text: string } {
  return { author: "", text: "" };
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
    /** JSON 배열: [{"source":"이름","target":"이름","weight":0-100}] */
    collaboration_edges_json: string;
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
  syllabus: {
    course_name: string;
    syllabus_text: string;
    question: string;
    result: CourseAskResponse | null;
    loading: boolean;
    error: string | null;
  };
  discussion: {
    thread_title: string;
    posts: { author: string; text: string }[];
    result: DiscussionSynthesizeResponse | null;
    loading: boolean;
    error: string | null;
  };
  rubricAlign: {
    rubric: string;
    grader_rationale: string;
    student_work: string;
    result: RubricAlignResponse | null;
    loading: boolean;
    error: string | null;
  };
  llmCompare: {
    task_title: string;
    system_hint: string;
    prompt: string;
    result: LLMCompareResponse | null;
    loading: boolean;
    error: string | null;
  };
} = {
  view: "team",
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
    evaluation_criteria: DEFAULT_TEAM_EVALUATION_CRITERIA,
    members: [emptyTeamMember(), emptyTeamMember()],
    collaboration_edges_json: "",
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
  syllabus: {
    course_name: "",
    syllabus_text: "",
    question: "",
    result: null,
    loading: false,
    error: null,
  },
  discussion: {
    thread_title: "",
    posts: [emptyDiscussionPost(), emptyDiscussionPost()],
    result: null,
    loading: false,
    error: null,
  },
  rubricAlign: {
    rubric: "",
    grader_rationale: "",
    student_work: "",
    result: null,
    loading: false,
    error: null,
  },
  llmCompare: {
    task_title: "",
    system_hint: "",
    prompt: "",
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

const TEAM_DRAFT_KEY = "team_eval_draft_v2";
let teamDraftTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleTeamDraftSave(): void {
  if (state.view !== "team") return;
  if (teamDraftTimer) clearTimeout(teamDraftTimer);
  teamDraftTimer = setTimeout(() => {
    teamDraftTimer = null;
    readTeamForm();
    try {
      const payload = {
        project_name: state.team.project_name,
        project_description: state.team.project_description,
        evaluation_criteria: state.team.evaluation_criteria,
        collaboration_edges_json: state.team.collaboration_edges_json,
        members: state.team.members,
      };
      localStorage.setItem(TEAM_DRAFT_KEY, JSON.stringify(payload));
    } catch {
      /* 저장소 불가 시 무시 */
    }
  }, 750);
}

function clearTeamDraft(): void {
  try {
    localStorage.removeItem(TEAM_DRAFT_KEY);
  } catch {
    /* ignore */
  }
}

function hydrateTeamDraftIfEmpty(): void {
  const hasInput =
    state.team.project_name.trim() !== "" ||
    state.team.project_description.trim() !== "" ||
    state.team.members.some((m) => m.name.trim() !== "");
  if (hasInput) return;
  try {
    const raw = localStorage.getItem(TEAM_DRAFT_KEY);
    if (!raw) return;
    const d = JSON.parse(raw) as {
      project_name?: string;
      project_description?: string;
      evaluation_criteria?: string;
      collaboration_edges_json?: string;
      members?: TeamMemberRow[];
    };
    if (typeof d.project_name === "string") state.team.project_name = d.project_name;
    if (typeof d.project_description === "string") state.team.project_description = d.project_description;
    if (typeof d.evaluation_criteria === "string") state.team.evaluation_criteria = d.evaluation_criteria;
    if (typeof d.collaboration_edges_json === "string") state.team.collaboration_edges_json = d.collaboration_edges_json;
    if (Array.isArray(d.members) && d.members.length > 0) {
      state.team.members = d.members.map((m) => ({
        ...emptyTeamMember(),
        ...m,
        timeline: m.timeline?.length ? m.timeline : emptyTeamMember().timeline,
      }));
    }
  } catch {
    /* ignore */
  }
}

function teamExportSummaryText(res: TeamEvaluateResponse): string {
  const lines: string[] = [];
  lines.push(`프로젝트: ${state.team.project_name || "(이름 없음)"}`);
  lines.push(`모드: ${res.mode} · 고급: ${res.advanced_mode ?? ""}`);
  if (res.request_id) lines.push(`요청 ID: ${res.request_id}`);
  if (res.generated_at) lines.push(`생성 시각: ${res.generated_at}`);
  if (res.processing_ms != null) lines.push(`서버 처리: ${res.processing_ms}ms`);
  lines.push("");
  res.members.forEach((m) => {
    lines.push(
      `- ${m.name}: 기여 ${m.contribution_index.toFixed(1)} · 의심도 ${m.free_rider_risk ?? "—"} · ${m.contribution_type_label || "유형 미분류"}`
    );
  });
  const ci = res.creative_insights;
  if (ci?.reflection_kit?.team_storyline) {
    lines.push("");
    lines.push("[창의 인사이트 · 팀 스토리라인]");
    lines.push(ci.reflection_kit.team_storyline);
  }
  if (ci?.reflection_kit?.teacher_questions?.length) {
    lines.push("");
    lines.push("[교육자용 질문]");
    ci.reflection_kit.teacher_questions.forEach((q, i) => lines.push(`${i + 1}. ${q}`));
  }
  if (res.contribution_outcome_summary) {
    lines.push("");
    lines.push("[기여·결과]");
    lines.push(res.contribution_outcome_summary);
  }
  return lines.join("\n");
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
    outcome_score: g(`tm_outcome_${i}`)?.value ?? "",
    self_report: g(`tm_self_${i}`)?.value ?? "",
    peer_notes: g(`tm_peer_${i}`)?.value ?? "",
    timeline: [0, 1, 2, 3].map((j) => ({
      period_label: g(`tm_tl_p_${i}_${j}`)?.value ?? "",
      activity_score: g(`tm_tl_s_${i}_${j}`)?.value ?? "",
    })),
  }));
  state.team.collaboration_edges_json = g("team_collaboration_edges")?.value ?? "";
}

async function submitTeam(): Promise<void> {
  readTeamForm();
  state.team.error = null;
  state.team.result = null;
  state.team.loading = true;
  render();

  const members = state.team.members
    .filter((m) => m.name.trim())
    .map((m) => {
      const timeline = (m.timeline || [])
        .filter((row) => row.period_label.trim())
        .map((row) => {
          const n = parseFloat(row.activity_score.trim());
          return Number.isFinite(n)
            ? { period_label: row.period_label.trim(), activity_score: Math.min(100, Math.max(0, n)) }
            : null;
        })
        .filter((x): x is { period_label: string; activity_score: number } => x !== null);
      const oc = parseOptFloat(m.outcome_score);
      return {
        name: m.name.trim(),
        role: m.role.trim(),
        commits: parseOptInt(m.commits),
        pull_requests: parseOptInt(m.pull_requests),
        lines_changed: parseOptInt(m.lines_changed),
        tasks_completed: parseOptInt(m.tasks_completed),
        meetings_attended: parseOptInt(m.meetings_attended),
        self_report: m.self_report,
        peer_notes: m.peer_notes,
        timeline,
        ...(oc !== null ? { outcome_score: Math.min(100, Math.max(0, oc)) } : {}),
      };
    });

  if (!state.team.project_name.trim() || members.length === 0) {
    state.team.error = "프로젝트명과 최소 한 명의 이름을 입력하세요.";
    state.team.loading = false;
    render();
    return;
  }

  let collaboration_edges: { source: string; target: string; weight: number }[] = [];
  const rawEdges = state.team.collaboration_edges_json.trim();
  if (rawEdges) {
    try {
      const parsed = JSON.parse(rawEdges) as unknown;
      if (!Array.isArray(parsed)) throw new Error("not array");
      collaboration_edges = parsed.map((e: unknown) => {
        if (!e || typeof e !== "object") throw new Error("bad edge");
        const o = e as Record<string, unknown>;
        const w = typeof o.weight === "number" ? o.weight : parseFloat(String(o.weight ?? 5));
        return {
          source: String(o.source ?? "").trim(),
          target: String(o.target ?? "").trim(),
          weight: Number.isFinite(w) ? Math.min(100, Math.max(0, w)) : 5,
        };
      });
    } catch {
      state.team.error = "협업 네트워크 JSON 형식이 올바르지 않습니다. 예: [{\"source\":\"A\",\"target\":\"B\",\"weight\":40}]";
      state.team.loading = false;
      render();
      return;
    }
  }

  const body = {
    project_name: state.team.project_name.trim(),
    project_description: state.team.project_description,
    evaluation_criteria: state.team.evaluation_criteria,
    members,
    collaboration_edges,
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
    clearTeamDraft();
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

function readSyllabusForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | null;
  state.syllabus.course_name = g("sy_course")?.value ?? "";
  state.syllabus.syllabus_text = g("sy_text")?.value ?? "";
  state.syllabus.question = g("sy_q")?.value ?? "";
}

async function submitSyllabus(): Promise<void> {
  readSyllabusForm();
  state.syllabus.error = null;
  state.syllabus.result = null;
  state.syllabus.loading = true;
  render();

  const text = state.syllabus.syllabus_text.trim();
  const q = state.syllabus.question.trim();
  if (text.length < 20 || q.length < 2) {
    state.syllabus.error = "안내 문구는 20자 이상, 질문은 2자 이상 입력하세요.";
    state.syllabus.loading = false;
    render();
    return;
  }

  const body = {
    course_name: state.syllabus.course_name.trim(),
    syllabus_text: text,
    question: q,
  };

  try {
    const r = await fetch(apiUrl("/api/course/ask"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as CourseAskResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.syllabus.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.syllabus.loading = false;
      render();
      return;
    }
    state.syllabus.result = data;
  } catch (e) {
    state.syllabus.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.syllabus.loading = false;
  }
  render();
}

function readDiscussionForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | null;
  state.discussion.thread_title = g("disc_title")?.value ?? "";
  state.discussion.posts = state.discussion.posts.map((_, i) => ({
    author: g(`disc_author_${i}`)?.value ?? "",
    text: g(`disc_text_${i}`)?.value ?? "",
  }));
}

async function submitDiscussion(): Promise<void> {
  readDiscussionForm();
  state.discussion.error = null;
  state.discussion.result = null;
  state.discussion.loading = true;
  render();

  const posts = state.discussion.posts
    .filter((p) => p.text.trim())
    .map((p) => ({
      author_label: p.author.trim() || "익명",
      text: p.text.trim(),
    }));

  if (posts.length === 0) {
    state.discussion.error = "최소 한 게시글 본문을 입력하세요.";
    state.discussion.loading = false;
    render();
    return;
  }

  const body = {
    thread_title: state.discussion.thread_title.trim(),
    posts,
  };

  try {
    const r = await fetch(apiUrl("/api/discussion/synthesize"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as DiscussionSynthesizeResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.discussion.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.discussion.loading = false;
      render();
      return;
    }
    state.discussion.result = data;
  } catch (e) {
    state.discussion.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.discussion.loading = false;
  }
  render();
}

function readRubricAlignForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLTextAreaElement | null;
  state.rubricAlign.rubric = g("ra_rubric")?.value ?? "";
  state.rubricAlign.grader_rationale = g("ra_rationale")?.value ?? "";
  state.rubricAlign.student_work = g("ra_student")?.value ?? "";
}

async function submitRubricAlign(): Promise<void> {
  readRubricAlignForm();
  state.rubricAlign.error = null;
  state.rubricAlign.result = null;
  state.rubricAlign.loading = true;
  render();

  if (!state.rubricAlign.rubric.trim() || !state.rubricAlign.grader_rationale.trim()) {
    state.rubricAlign.error = "루브릭과 채점 근거를 입력하세요.";
    state.rubricAlign.loading = false;
    render();
    return;
  }

  const body = {
    rubric: state.rubricAlign.rubric.trim(),
    grader_rationale: state.rubricAlign.grader_rationale.trim(),
    student_work_excerpt: state.rubricAlign.student_work.trim(),
  };

  try {
    const r = await fetch(apiUrl("/api/rubric/check"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as RubricAlignResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.rubricAlign.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.rubricAlign.loading = false;
      render();
      return;
    }
    state.rubricAlign.result = data;
  } catch (e) {
    state.rubricAlign.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.rubricAlign.loading = false;
  }
  render();
}

function readLlmCompareForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | null;
  state.llmCompare.task_title = g("llm_task_title")?.value ?? "";
  state.llmCompare.system_hint = g("llm_system_hint")?.value ?? "";
  state.llmCompare.prompt = g("llm_prompt")?.value ?? "";
}

async function submitLlmCompare(): Promise<void> {
  readLlmCompareForm();
  state.llmCompare.error = null;
  state.llmCompare.result = null;
  state.llmCompare.loading = true;
  render();

  const p = state.llmCompare.prompt.trim();
  if (!p) {
    state.llmCompare.error = "분석할 내용을 입력하세요.";
    state.llmCompare.loading = false;
    render();
    return;
  }

  const body = {
    task_title: state.llmCompare.task_title.trim(),
    system_hint: state.llmCompare.system_hint,
    prompt: p,
  };

  try {
    const r = await fetch(apiUrl("/api/llm/compare"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as LLMCompareResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.llmCompare.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.llmCompare.loading = false;
      render();
      return;
    }
    state.llmCompare.result = data;
  } catch (e) {
    state.llmCompare.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.llmCompare.loading = false;
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
      <a href="#" class="brand brand-mark" data-view="team" aria-label="팀 기여도 자동 평가">
        <span class="brand-title">팀 기여도</span>
        <span class="brand-tag">자동 평가 시스템 · AI</span>
      </a>
      <nav class="nav" aria-label="주요 메뉴">
        <button type="button" class="${cur("team")}" data-view="team">평가</button>
        <button type="button" class="${cur("hub")}" data-view="hub">부가 도구</button>
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
      <p class="footer-muted">팀 프로젝트 기여도 자동 평가 시스템 — 교육 보조·참고용</p>
    </div>
  </footer>`;
}

function hubHtml(): string {
  return `
  <div class="page page-animate home-page">
    <section class="hero-block home-hero-main">
      <p class="eyebrow">부가 모듈</p>
      <h1 class="home-headline">교육 현장 보조 AI</h1>
      <p class="hero-text home-lead">
        본 서비스의 <strong>핵심은 상단 「평가」의 팀 기여도 자동 평가</strong>입니다. 아래는 같은 백엔드에 연결된 <strong>선택</strong> 도구입니다.
      </p>
      <p class="muted small home-lead" style="margin-top:0.5rem;">
        로컬: 루트 <code>npm run dev</code> → API <code>8000</code> + 웹 <code>5173</code> · <code>/api</code> 프록시
      </p>
      <div class="pill-row hero-pills">${providerPills()}</div>
      <p class="row-actions" style="margin-top:1rem;">
        <button type="button" class="btn btn-primary" data-view="team">팀 기여도 평가로 돌아가기</button>
      </p>
    </section>

    <section class="section-block hud-section home-section">
      <h2 class="section-title">부가 도구</h2>
      <div class="solution-grid">
        <article class="solution-tile solution-tile--live hud-card">
          <span class="solution-badge solution-badge--on">다중 LLM</span>
          <h3>과정 vs 시험 불일치</h3>
          <p>LMS·과제 지표와 시험 점수를 넣어 부정행위 <em>의심도</em>·학습 상태·위험을 참고용으로 제시합니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="analyze">열기</button>
        </article>
        <article class="solution-tile solution-tile--live hud-card">
          <span class="solution-badge solution-badge--on">4모델</span>
          <h3>Gemini · ChatGPT · Claude · Grok</h3>
          <p>같은 프롬프트로 네 AI에 동시에 질문하고 응답을 나란히 비교합니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="llm">열기</button>
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
        <article class="solution-tile solution-tile--live hud-card">
          <span class="solution-badge solution-badge--on">반복 질문</span>
          <h3>강의 안내 Q&amp;A 초안</h3>
          <p>실라버스·공지 일부와 학생 질문을 넣으면 안내 근거에 기반한 답변 초안·인용을 봅니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="syllabus">열기</button>
        </article>
        <article class="solution-tile solution-tile--live hud-card">
          <span class="solution-badge solution-badge--on">대규모 토론</span>
          <h3>토론 스레드 요약</h3>
          <p>게시글을 붙여 넣으면 주제·참여 양상·후속 질문 초안을 정리합니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="discussion">열기</button>
        </article>
        <article class="solution-tile solution-tile--live hud-card">
          <span class="solution-badge solution-badge--on">공정성</span>
          <h3>루브릭·채점 근거 일치</h3>
          <p>루브릭과 채점 코멘트를 비교해 정합성 점수와 개선 힌트를 봅니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="rubric">열기</button>
        </article>
      </div>
    </section>
  </div>`;
}

function teamMemberBlock(i: number, m: TeamMemberRow): string {
  const tl = m.timeline?.length ? m.timeline : [0, 1, 2, 3].map(() => emptyTimelineRow());
  const timelineRows = tl.slice(0, 4).map((row, j) => {
    const p = row.period_label ?? "";
    const s = row.activity_score ?? "";
    return `
    <div class="grid-2">
      <div><label class="lbl">기간 ${j + 1}</label><input class="txt" id="tm_tl_p_${i}_${j}" placeholder="예: 3주차" value="${escapeHtml(p)}" /></div>
      <div><label class="lbl">활동 0–100</label><input class="txt" id="tm_tl_s_${i}_${j}" type="number" min="0" max="100" step="0.1" value="${escapeHtml(s)}" /></div>
    </div>`;
  });
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
      <div><label class="lbl">결과 점수 (선택, 0–100)</label><input class="txt" id="tm_outcome_${i}" type="number" min="0" max="100" step="0.1" placeholder="발표·동료평가 등" value="${escapeHtml(m.outcome_score)}" /></div>
    </div>
    <details class="timeline-details">
      <summary>주차별 활동 점수 (선택, 무임승차·타임라인에 반영)</summary>
      <p class="muted small">같은 주차 라벨을 팀원 간에 맞추면 비교가 쉽습니다. 비우면 서버가 가상 시계열을 생성합니다.</p>
      ${timelineRows.join("")}
    </details>
    <label class="lbl">자기 서술</label>
    <textarea class="txt" id="tm_self_${i}" rows="2">${escapeHtml(m.self_report)}</textarea>
    <label class="lbl">동료 메모</label>
    <textarea class="txt" id="tm_peer_${i}" rows="2">${escapeHtml(m.peer_notes)}</textarea>
  </div>`;
}

function teamNetworkSvg(net: NetworkGraph | undefined): string {
  if (!net?.nodes?.length) return "";
  const pos = new Map(net.nodes.map((n) => [n.id, { x: n.x, y: n.y }]));
  const W = 520;
  const H = 480;
  const edgeLines = (net.edges || [])
    .map((e) => {
      const a = pos.get(e.source);
      const b = pos.get(e.target);
      if (!a || !b) return "";
      const sw = Math.max(0.6, Math.min(5, (e.weight / 100) * 5));
      return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="rgba(37,99,235,0.35)" stroke-width="${sw}" />`;
    })
    .join("");
  const circles = net.nodes
    .map((n) => {
      const lab = n.label || n.id;
      return `<g>
      <circle cx="${n.x}" cy="${n.y}" r="20" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
      <text x="${n.x}" y="${n.y + 4}" text-anchor="middle" fill="#0f172a" font-size="10" font-family="system-ui,sans-serif">${escapeHtml(lab)}</text>
      <text x="${n.x}" y="${n.y + 28}" text-anchor="middle" fill="#64748b" font-size="8" font-family="system-ui,sans-serif">${n.contribution_index.toFixed(0)}</text>
    </g>`;
    })
    .join("");
  return `
  <div class="network-chart-wrap hud-panel">
    <h3 class="subh">협업 네트워크</h3>
    <p class="muted small">노드는 팀원, 선은 상호작용 가중치입니다. 엣지를 입력하지 않으면 기여 지수로 추정한 완전 그래프입니다.</p>
    <svg class="network-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" aria-label="협업 네트워크 그래프">
      ${edgeLines}
      ${circles}
    </svg>
  </div>`;
}

function teamTimelineSvg(members: TeamMemberOut[]): string {
  const sample = members.find((m) => m.timeline && m.timeline.length);
  if (!sample?.timeline?.length) return "";
  const labels = sample.timeline.map((t) => t.period_label);
  const n = members.length;
  const k = labels.length;
  const W = 560;
  const H = 240;
  const padL = 44;
  const padR = 12;
  const padT = 14;
  const padB = 40;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const colors = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#0891b2", "#dc2626"];
  let maxY = 40;
  for (const m of members) {
    for (const p of m.timeline || []) {
      maxY = Math.max(maxY, p.share_percent);
    }
  }
  maxY = Math.min(100, Math.ceil(maxY / 10) * 10 + 5);

  const lines: string[] = [];
  for (let mi = 0; mi < n; mi++) {
    const tl = members[mi].timeline || [];
    if (tl.length === 0) continue;
    const color = colors[mi % colors.length];
    const coords = tl.map((p, t) => {
      const x = k === 1 ? padL + chartW / 2 : padL + (t / Math.max(1, k - 1)) * chartW;
      const y = padT + chartH - (p.share_percent / maxY) * chartH;
      return `${x},${y}`;
    });
    if (coords.length === 1) {
      const [cx, cy] = coords[0].split(",").map(Number);
      lines.push(`<circle cx="${cx}" cy="${cy}" r="5" fill="${color}" />`);
    } else {
      lines.push(
        `<polyline fill="none" stroke="${color}" stroke-width="2.5" points="${coords.join(" ")}" />`
      );
    }
  }

  const xLabels = labels
    .map((lab, t) => {
      const x = k === 1 ? padL + chartW / 2 : padL + (t / Math.max(1, k - 1)) * chartW;
      return `<text x="${x}" y="${H - 12}" text-anchor="middle" fill="#64748b" font-size="10" font-family="system-ui,sans-serif">${escapeHtml(lab)}</text>`;
    })
    .join("");

  const legend = members
    .map(
      (m, i) =>
        `<span class="timeline-legend-item" style="color:${colors[i % colors.length]}">● ${escapeHtml(m.name)}</span>`
    )
    .join(" ");

  return `
  <div class="timeline-chart-wrap hud-panel">
    <h3 class="subh">기여도 변화 (시간에 따른 팀 내 상대 비중 %)</h3>
    <p class="muted small legend-line">${legend}</p>
    <svg class="timeline-chart" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" aria-label="기여도 타임라인">
      <rect x="${padL}" y="${padT}" width="${chartW}" height="${chartH}" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" rx="2"/>
      ${lines.join("")}
      ${xLabels}
    </svg>
  </div>`;
}

function normNums(vals: number[]): number[] {
  const s = vals.reduce((a, b) => a + b, 0) || 1;
  return vals.map((v) => v / s);
}

/** 백엔드 `_heuristic`과 동일한 가중(커밋·태스크·라인·PR·자기서술 분량). */
function heuristicContributionIndicesFromRows(members: TeamMemberRow[]): number[] {
  const n = members.length;
  if (n === 0) return [];
  const parseN = (s: string) => {
    const x = parseFloat(String(s).trim());
    return Number.isFinite(x) && x >= 0 ? x : 0;
  };
  const commits = members.map((m) => parseN(m.commits));
  const tasks = members.map((m) => parseN(m.tasks_completed));
  const lines = members.map((m) => parseN(m.lines_changed));
  const prs = members.map((m) => parseN(m.pull_requests));
  const words = members.map((m) =>
    Math.max(0, (m.self_report || "").trim().split(/\s+/).filter(Boolean).length)
  );
  const nc = normNums(commits);
  const nt = normNums(tasks);
  const nl = normNums(lines);
  const npr = normNums(prs);
  const nw = normNums(words.map((w) => w));
  const raw = members.map(
    (_, i) => 0.22 * nc[i] + 0.18 * nt[i] + 0.18 * nl[i] + 0.12 * npr[i] + 0.3 * nw[i]
  );
  const rawSum = raw.reduce((a, b) => a + b, 0) || 1;
  return raw.map((x) => Math.round((10000 * x) / rawSum) / 100);
}

function simulateExtraCommits(members: TeamMemberRow[], memberIdx: number, extra: number): number[] {
  const parseN = (s: string) => {
    const x = parseFloat(String(s).trim());
    return Number.isFinite(x) && x >= 0 ? x : 0;
  };
  const copy = members.map((m, i) =>
    i === memberIdx ? { ...m, commits: String(parseN(m.commits) + extra) } : m
  );
  return heuristicContributionIndicesFromRows(copy);
}

function teamRoleRadarSvg(b: TeamRoleBalance): string {
  const vals = [b.dev, b.doc, b.leader, b.supporter];
  const labels = ["개발", "문서", "리더", "서포터"];
  const cx = 100;
  const cy = 100;
  const rmax = 78;
  const n = 4;
  const grid = [0.25, 0.5, 0.75, 1].map(
    (t) =>
      `<circle cx="${cx}" cy="${cy}" r="${rmax * t}" fill="none" stroke="#e2e8f0" stroke-width="1" />`
  );
  const axes = labels
    .map((lab, i) => {
      const ang = -Math.PI / 2 + (2 * Math.PI * i) / n;
      const x2 = cx + rmax * Math.cos(ang);
      const y2 = cy + rmax * Math.sin(ang);
      const lx = cx + (rmax + 14) * Math.cos(ang);
      const ly = cy + (rmax + 14) * Math.sin(ang);
      return `<line x1="${cx}" y1="${cy}" x2="${x2}" y2="${y2}" stroke="#cbd5e1" stroke-width="1" /><text x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="middle" fill="#475569" font-size="10" font-family="system-ui,sans-serif">${lab}</text>`;
    })
    .join("");
  const pts = vals.map((v, i) => {
    const ang = -Math.PI / 2 + (2 * Math.PI * i) / n;
    const rr = (Math.min(100, Math.max(0, v)) / 100) * rmax;
    return [cx + rr * Math.cos(ang), cy + rr * Math.sin(ang)];
  });
  const pathD =
    pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ") + " Z";
  return `<svg class="radar-svg" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" aria-label="팀 평균 역할 밸런스">
    ${grid.join("")}
    ${axes}
    <path d="${pathD}" fill="rgba(37,99,235,0.12)" stroke="#2563eb" stroke-width="2" />
  </svg>`;
}

function teamCreativePanelHtml(res: TeamEvaluateResponse): string {
  const c = res.creative_insights;
  if (!c) return "";
  const rk = c.reflection_kit;
  const factsHtml = (c.explain_facts || [])
    .map(
      (x) => `
    <article class="fact-card">
      <h5 class="fact-card-title">${escapeHtml(x.member_name)}</h5>
      <ul class="fact-card-ul">${(x.facts || []).map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul>
    </article>`
    )
    .join("");
  return `
  <section class="creative-panel hud-panel" aria-labelledby="creative-heading">
    <h3 class="subh" id="creative-heading">창의 인사이트 · 설명 가능성 · 면담 키트</h3>
    <p class="muted small creative-lead">규칙 기반 자동 생성입니다. 징계·단정용이 아니라 <strong>교육·성찰·면담 준비</strong>용입니다.</p>
    <div class="creative-grid">
      <div class="creative-col creative-col--story">
        <h4 class="creative-h4">팀 스토리라인</h4>
        <p class="prose creative-story">${escapeHtml(rk.team_storyline)}</p>
        <p class="creative-encourage">${escapeHtml(rk.encouragement_line)}</p>
      </div>
      <div class="creative-col creative-col--radar">
        <h4 class="creative-h4">팀 역할 밸런스 (평균)</h4>
        ${teamRoleRadarSvg(c.team_role_balance)}
        <p class="muted small">${escapeHtml(c.team_role_balance.balance_hint)}</p>
      </div>
      <div class="creative-col creative-col--questions">
        <h4 class="creative-h4">교육자용 질문</h4>
        <ol class="reflection-ol">${(rk.teacher_questions || []).map((q) => `<li>${escapeHtml(q)}</li>`).join("")}</ol>
      </div>
    </div>
    <details class="creative-details">
      <summary>팀원별 규칙 기반 설명 카드</summary>
      <div class="fact-card-grid">${factsHtml}</div>
    </details>
  </section>`;
}

function teamSimulatorHtml(): string {
  const members = state.team.members.filter((m) => m.name.trim());
  const opts = members
    .map((m, i) => `<option value="${i}">${escapeHtml(m.name)}</option>`)
    .join("");
  return `
  <section class="creative-sim hud-panel no-print" aria-labelledby="sim-heading">
    <h3 class="subh" id="sim-heading">가상 시뮬레이터 (교육용)</h3>
    <p class="muted small">휴리스틱 평가와 동일한 가중으로, 선택한 팀원의 <strong>커밋 수만</strong> 가상 증가시켰을 때의 기여 지수 변화를 보여 줍니다. 실제 서버 재요청이 아닙니다.</p>
    <div class="grid-2 sim-controls">
      <div>
        <label class="lbl" for="team-sim-member">팀원</label>
        <select class="txt" id="team-sim-member">${opts || '<option value="0">—</option>'}</select>
      </div>
      <div>
        <label class="lbl" for="team-sim-extra">가상 추가 커밋: <span id="team-sim-extra-lbl">0</span></label>
        <input type="range" class="sim-range" id="team-sim-extra" min="0" max="40" value="0" />
      </div>
    </div>
    <div id="team-sim-output" class="sim-output" aria-live="polite"></div>
  </section>`;
}

function updateTeamSimDisplay(): void {
  readTeamForm();
  const sel = document.getElementById("team-sim-member") as HTMLSelectElement | null;
  const range = document.getElementById("team-sim-extra") as HTMLInputElement | null;
  const lbl = document.getElementById("team-sim-extra-lbl");
  const out = document.getElementById("team-sim-output");
  if (!sel || !range || !out) return;
  const extra = parseInt(range.value, 10) || 0;
  if (lbl) lbl.textContent = String(extra);
  const members = state.team.members.filter((m) => m.name.trim());
  if (members.length === 0) {
    out.innerHTML = '<p class="muted small">이름이 입력된 팀원이 없습니다.</p>';
    return;
  }
  let idx = parseInt(sel.value, 10);
  if (!Number.isFinite(idx) || idx < 0 || idx >= members.length) idx = 0;
  const base = heuristicContributionIndicesFromRows(members);
  const sim = simulateExtraCommits(members, idx, extra);
  const rows = members
    .map((m, i) => {
      const d = sim[i] - base[i];
      const delta = d === 0 ? "±0.0" : d > 0 ? `+${d.toFixed(1)}` : d.toFixed(1);
      return `<tr><td>${escapeHtml(m.name)}</td><td>${base[i].toFixed(1)}</td><td>${sim[i].toFixed(1)}</td><td class="sim-delta">${delta}</td></tr>`;
    })
    .join("");
  out.innerHTML = `<div class="table-scroll-wrap"><table class="data-table data-table--sim"><thead><tr><th>이름</th><th>현재(추정)</th><th>시뮬</th><th>Δ</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function teamResultHtml(): string {
  const res = state.team.result;
  if (!res) return "";
  const rows = res.members
    .map((m) => {
      const sus = m.free_rider_suspected
        ? `<span class="free-rider-badge" title="자동 의심">무임승차 의심</span>`
        : `<span class="pill pill-muted">—</span>`;
      const risk = m.free_rider_risk != null ? m.free_rider_risk.toFixed(0) : "—";
      const role = m.contribution_type_label
        ? `<span class="role-badge">${escapeHtml(m.contribution_type_label)}</span>`
        : "—";
      const rs = m.role_scores;
      const rsTxt =
        rs && Object.keys(rs).length
          ? `<span class="muted small role-scores-mini">개발 ${(rs.dev ?? 0).toFixed(0)} · 문서 ${(rs.doc ?? 0).toFixed(0)} · 리더 ${(rs.leader ?? 0).toFixed(0)} · 서포터 ${(rs.supporter ?? 0).toFixed(0)}</span>`
          : "";
      return `
    <tr class="${m.free_rider_suspected ? "row-suspected" : ""}">
      <td>${escapeHtml(m.name)} ${sus}</td>
      <td>${m.contribution_index.toFixed(1)}</td>
      <td>${risk}</td>
      <td>${role} ${rsTxt}</td>
      <td>${m.dimensions.technical.toFixed(0)} / ${m.dimensions.collaboration.toFixed(0)} / ${m.dimensions.initiative.toFixed(0)}</td>
      <td class="muted small">${escapeHtml(m.evidence_summary || "")}</td>
    </tr>`;
    })
    .join("");
  const sigBlock = res.members
    .filter((m) => m.free_rider_signals?.length)
    .map(
      (m) =>
        `<p class="signal-line"><strong>${escapeHtml(m.name)}</strong>: ${(m.free_rider_signals || []).map((s) => escapeHtml(s)).join(" · ")}</p>`
    )
    .join("");
  const feedbackBlocks = res.members
    .map(
      (m) => `
    <article class="member-feedback-card hud-panel">
      <h4>${escapeHtml(m.name)}${m.free_rider_suspected ? ` <span class="free-rider-badge">의심</span>` : ""}</h4>
      <p class="prose feedback-text">${escapeHtml(m.ai_feedback || "")}</p>
    </article>`
    )
    .join("");
  const chart = teamTimelineSvg(res.members);
  const net = teamNetworkSvg(res.collaboration_network);
  const adv = res.advanced_mode
    ? `<span class="pill ${res.advanced_mode === "openai_enriched" ? "pill-on" : "pill-muted"}">${escapeHtml(res.advanced_mode)}</span>`
    : "";
  const coSummary = res.contribution_outcome_summary
    ? `<div class="advanced-summary-box hud-panel"><h3 class="subh">기여 vs 결과 불일치 분석</h3><p class="prose whitespace-pre-wrap">${escapeHtml(res.contribution_outcome_summary)}</p></div>`
    : "";
  const mmRows = (res.mismatches || [])
    .map(
      (x) => `
    <tr>
      <td>${escapeHtml(x.member_name)}</td>
      <td>${x.contribution_index?.toFixed(1) ?? "—"}</td>
      <td>${x.outcome_score != null ? x.outcome_score.toFixed(1) : "—"}</td>
      <td>${x.gap != null ? x.gap.toFixed(1) : "—"}</td>
      <td><span class="sev sev-${escapeHtml(x.severity || "low")}">${escapeHtml(x.severity || "")}</span></td>
      <td class="muted small">${escapeHtml(x.note || "")}</td>
    </tr>`
    )
    .join("");
  const mismatchBlock =
    res.mismatches && res.mismatches.length
      ? `<div class="mismatch-table-wrap hud-panel"><h3 class="subh">불일치 상세 (추정 기여 vs 입력 결과)</h3>
    <table class="data-table"><thead><tr><th>이름</th><th>기여 추정</th><th>결과 점수</th><th>차이</th><th>심각도</th><th>메모</th></tr></thead><tbody>${mmRows}</tbody></table></div>`
      : "";
  const anom = (res.anomaly_alerts || [])
    .map(
      (a) =>
        `<p class="anomaly-line"><strong>${escapeHtml(a.member_name)}</strong> <code>${escapeHtml(a.code)}</code> <span class="sev sev-${escapeHtml(a.severity || "medium")}">${escapeHtml(a.severity || "")}</span> — ${escapeHtml(a.message || "")}</p>`
    )
    .join("");
  const anomBlock =
    res.anomaly_alerts && res.anomaly_alerts.length
      ? `<div class="anomaly-box hud-panel"><h3 class="subh">비정상 행동 탐지 (고급, 참고)</h3>${anom}</div>`
      : "";
  const metaParts: string[] = [];
  if (res.request_id) metaParts.push(`요청 ID <code class="meta-code">${escapeHtml(res.request_id)}</code>`);
  if (res.processing_ms != null) metaParts.push(`처리 ${escapeHtml(String(res.processing_ms))}ms`);
  if (res.generated_at) metaParts.push(escapeHtml(res.generated_at));
  const metaLine =
    metaParts.length > 0
      ? `<p class="result-meta muted small no-print" role="status">${metaParts.join(" · ")}</p>`
      : "";
  const creativeBlock = teamCreativePanelHtml(res);
  const simBlock = teamSimulatorHtml();
  return `
  <section class="panel panel-result hud-panel team-print-root" id="team-printable-root">
    <div class="result-toolbar no-print" role="toolbar" aria-label="결과 내보내기">
      <button type="button" class="btn btn-ghost btn-sm" id="btn-team-export-json">JSON 내보내기</button>
      <button type="button" class="btn btn-ghost btn-sm" id="btn-team-copy-summary">요약 복사</button>
      <button type="button" class="btn btn-ghost btn-sm" id="btn-team-print">인쇄·PDF</button>
    </div>
    ${metaLine}
    <h2>자동 평가 결과 <span class="pill ${res.mode === "ai" ? "pill-on" : "pill-muted"}">${escapeHtml(res.mode)}</span> ${adv}</h2>
    ${res.fairness_notes ? `<p class="prose">${escapeHtml(res.fairness_notes)}</p>` : ""}
    ${res.free_rider_summary ? `<p class="prose free-rider-summary">${escapeHtml(res.free_rider_summary)}</p>` : ""}
    ${creativeBlock}
    ${simBlock}
    ${coSummary}
    ${sigBlock ? `<div class="signals-box">${sigBlock}</div>` : ""}
    <div class="table-scroll-wrap">
    <table class="data-table data-table--team">
      <thead><tr><th>이름</th><th>기여 지수</th><th>의심도</th><th>기여 유형 (자동)</th><th>기술·협업·주도</th><th>근거 요약</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    </div>
    ${net}
    ${mismatchBlock}
    ${anomBlock}
    ${chart}
    <h3 class="subh">팀원별 AI 피드백</h3>
    <div class="member-feedback-grid">${feedbackBlocks}</div>
    <p class="footer-note muted small">${escapeHtml(res.disclaimer || "")}</p>
  </section>`;
}

function teamHtml(): string {
  const blocks = state.team.members.map((m, i) => teamMemberBlock(i, m)).join("");
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">Team Project · Auto Evaluation</p>
    <h1 class="page-title">팀 프로젝트 기여도 자동 평가</h1>
    <p class="lead analyze-lead">
      정량 지표(커밋·PR·태스크 등)·주차별 활동·자기·동료 서술을 바탕으로 <strong>기여 지수</strong>를 산출하고,
      <strong>무임승차 의심</strong>·<strong>기여 유형(개발·문서·리더·서포터)</strong>·<strong>기여–결과 불일치</strong>·<strong>협업 네트워크</strong>·<strong>이상 탐지(고급)</strong>·<strong>기여도 타임라인</strong>·<strong>팀원별 AI 피드백</strong>·
      <strong>창의 인사이트</strong>(팀 스토리라인·역할 밸런스 레이더·면담 질문·설명 카드·가상 시뮬레이터)를 제공합니다.
      OpenAI 키가 있으면 서술 반영 평가·불일치 해설 품질이 올라갑니다. 최종 성적은 교수·기관 규정에 따릅니다.
    </p>
    <div class="pill-row">${providerPills()}</div>
    <p class="muted small team-practice-hint">입력 내용은 브라우저에 <strong>자동 임시 저장</strong>됩니다(성공적으로 평가하면 삭제). 조교·기록용으로 JSON 내보내기·요약 복사를 활용하세요.</p>

    <section class="panel hud-panel team-form-panel">
      <div class="grid-2">
        <div>
          <label class="lbl">프로젝트명</label>
          <input class="txt" id="team_project_name" value="${escapeHtml(state.team.project_name)}" />
        </div>
      </div>
      <label class="lbl">프로젝트 설명</label>
      <textarea class="txt" id="team_project_description" rows="2">${escapeHtml(state.team.project_description)}</textarea>
      <label class="lbl">평가 기준 · 심사기준 (기본값 있음, 수정 가능)</label>
      <textarea class="txt" id="team_evaluation_criteria" rows="6">${escapeHtml(state.team.evaluation_criteria)}</textarea>
      <label class="lbl">협업 네트워크 (선택, JSON 배열)</label>
      <textarea class="txt mono-input" id="team_collaboration_edges" rows="3" placeholder='[{"source":"이름A","target":"이름B","weight":40}]'>${escapeHtml(state.team.collaboration_edges_json)}</textarea>
      <p class="muted small">source·target은 위 멤버 이름과 동일하게. weight는 0–100. 비우면 서버가 기여 지수로 간선을 추정합니다.</p>
      ${blocks}
      <div class="row-actions">
        <button type="button" class="btn btn-ghost" id="btn-team-add">멤버 추가</button>
        <button type="button" class="btn btn-ghost" id="btn-team-remove" ${state.team.members.length <= 1 ? "disabled" : ""}>마지막 멤버 제거</button>
        <button type="button" class="btn btn-primary" id="btn-team-run" ${state.team.loading ? "disabled" : ""}>
          ${state.team.loading ? "자동 평가 중…" : "자동 평가 실행"}
        </button>
      </div>
      ${state.team.error ? `<p class="err">${escapeHtml(state.team.error)}</p>` : ""}
    </section>
    ${
      state.team.loading
        ? `<div class="team-loading hud-panel" aria-live="polite" aria-busy="true">
      <div class="skeleton-line skeleton-line--long"></div>
      <div class="skeleton-line"></div>
      <div class="skeleton-line skeleton-line--med"></div>
      <p class="muted small">다중 분석·피드백 생성 중입니다. 팀 규모·API에 따라 수십 초 걸릴 수 있습니다.</p>
    </div>`
        : ""
    }
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

function syllabusHtml(): string {
  const r = state.syllabus.result;
  const resultBlock = r
    ? `
  <section class="panel panel-result hud-panel">
    <h2>답변 초안 <span class="pill ${r.mode === "ai" ? "pill-on" : "pill-muted"}">${escapeHtml(r.mode)}</span></h2>
    <p class="prose">${escapeHtml(r.answer_draft)}</p>
    ${
      r.citations?.length
        ? `<h4>인용·근거 문장</h4><ul>${r.citations.map((c) => `<li class="prose">${escapeHtml(c)}</li>`).join("")}</ul>`
        : ""
    }
    <p class="footer-note muted small">${escapeHtml(r.caveats || "")}</p>
  </section>`
    : "";
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">강의 운영</p>
    <h1 class="page-title">강의 안내 Q&amp;A 초안</h1>
    <p class="lead analyze-lead">실라버스·공지 텍스트 일부와 학생 질문을 넣으세요. OpenAI가 있으면 근거 중심 답변이, 없으면 키워드 매칭 발췌가 나옵니다.</p>
    <section class="panel hud-panel">
      <label class="lbl">과목명 (선택)</label>
      <input class="txt" id="sy_course" value="${escapeHtml(state.syllabus.course_name)}" />
      <label class="lbl">강의 안내·실라버스 발췌</label>
      <textarea class="txt" id="sy_text" rows="10" placeholder="최소 20자">${escapeHtml(state.syllabus.syllabus_text)}</textarea>
      <label class="lbl">학생 질문</label>
      <textarea class="txt" id="sy_q" rows="3">${escapeHtml(state.syllabus.question)}</textarea>
      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-sy-run" ${state.syllabus.loading ? "disabled" : ""}>
          ${state.syllabus.loading ? "생성 중…" : "초안 만들기"}
        </button>
      </div>
      ${state.syllabus.error ? `<p class="err">${escapeHtml(state.syllabus.error)}</p>` : ""}
    </section>
    ${resultBlock}
  </div>`;
}

function discussionPostBlock(i: number, p: { author: string; text: string }): string {
  return `
  <div class="member-block hud-panel">
    <h4 class="subh">게시 ${i + 1}</h4>
    <div class="grid-2">
      <div><label class="lbl">작성자 라벨 (선택)</label><input class="txt" id="disc_author_${i}" placeholder="익명-A" value="${escapeHtml(p.author)}" /></div>
    </div>
    <label class="lbl">본문</label>
    <textarea class="txt" id="disc_text_${i}" rows="4">${escapeHtml(p.text)}</textarea>
  </div>`;
}

function discussionHtml(): string {
  const blocks = state.discussion.posts.map((p, i) => discussionPostBlock(i, p)).join("");
  const r = state.discussion.result;
  const resultBlock = r
    ? `
  <section class="panel panel-result hud-panel">
    <h2>요약 <span class="pill ${r.mode === "ai" ? "pill-on" : "pill-muted"}">${escapeHtml(r.mode)}</span></h2>
    <p class="prose">${escapeHtml(r.summary)}</p>
    <h4>주제·쟁점</h4>
    <ul>${r.themes.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ul>
    <h4>참여 메모</h4>
    <p class="prose">${escapeHtml(r.participation_notes)}</p>
    <h4>후속 질문 제안</h4>
    <ul>${r.suggested_followups.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    <p class="footer-note muted small">${escapeHtml(r.disclaimer || "")}</p>
  </section>`
    : "";
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">토론·포럼</p>
    <h1 class="page-title">스레드 요약</h1>
    <p class="lead analyze-lead">게시글을 여러 개 붙여 넣을 수 있습니다. OpenAI가 있으면 의미 요약이, 없으면 글자 수·키워드 통계가 나옵니다.</p>
    <section class="panel hud-panel">
      <label class="lbl">스레드 제목 (선택)</label>
      <input class="txt" id="disc_title" value="${escapeHtml(state.discussion.thread_title)}" />
      ${blocks}
      <div class="row-actions">
        <button type="button" class="btn btn-ghost" id="btn-disc-add">게시 추가</button>
        <button type="button" class="btn btn-ghost" id="btn-disc-remove" ${state.discussion.posts.length <= 1 ? "disabled" : ""}>마지막 게시 제거</button>
        <button type="button" class="btn btn-primary" id="btn-disc-run" ${state.discussion.loading ? "disabled" : ""}>
          ${state.discussion.loading ? "요약 중…" : "요약 실행"}
        </button>
      </div>
      ${state.discussion.error ? `<p class="err">${escapeHtml(state.discussion.error)}</p>` : ""}
    </section>
    ${resultBlock}
  </div>`;
}

function llmProviderLabel(provider: string): string {
  const m: Record<string, string> = {
    gemini: "Gemini",
    openai: "ChatGPT",
    claude: "Claude",
    grok: "Grok",
  };
  return m[provider] || provider;
}

function llmCompareHtml(): string {
  const r = state.llmCompare.result;
  const skippedHtml =
    r && r.providers_skipped.length > 0
      ? `<p class="skipped muted small">건너뜀: ${escapeHtml(r.providers_skipped.join(" · "))}</p>`
      : "";
  const cards =
    r?.results
      .map((j) => {
        if (!j.ok) {
          return `<div class="card card-err"><h3>${escapeHtml(llmProviderLabel(j.provider))}</h3><p class="err">${escapeHtml(j.error || "오류")}</p></div>`;
        }
        return `<div class="card">
        <div class="card-head">
          <h3>${escapeHtml(llmProviderLabel(j.provider))}</h3>
          <span class="model-tag">${escapeHtml(j.model_label || "")}</span>
        </div>
        <p class="prose llm-freeform-text">${escapeHtml(j.text)}</p>
      </div>`;
      })
      .join("") ?? "";
  const resultBlock = r
    ? `
  <section class="panel panel-result hud-panel" id="llm-results">
    <h2>모델별 응답</h2>
    ${skippedHtml}
    <div class="card-grid">${cards}</div>
    <p class="footer-note">${escapeHtml(r.disclaimer)}</p>
  </section>`
    : "";
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">4모델 병렬</p>
    <h1 class="page-title">Gemini · ChatGPT · Claude · Grok</h1>
    <p class="lead analyze-lead">동일한 요청을 네 AI에 동시에 보냅니다. <code>backend/.env</code>에 키가 있는 모델만 응답합니다.</p>
    <div class="pill-row">${providerPills()}</div>
    <section class="panel hud-panel">
      <label class="lbl">작업 제목 (선택)</label>
      <input class="txt" id="llm_task_title" placeholder="예: 수업 개선안 검토" value="${escapeHtml(state.llmCompare.task_title)}" />
      <label class="lbl">추가 지시 (시스템 힌트, 선택)</label>
      <textarea class="txt" id="llm_system_hint" rows="2" placeholder="예: bullet으로 짧게">${escapeHtml(state.llmCompare.system_hint)}</textarea>
      <label class="lbl">분석·질문 내용</label>
      <textarea class="txt" id="llm_prompt" rows="10" placeholder="여기에 붙여 넣으면 Gemini, ChatGPT, Claude, Grok이 각각 답합니다.">${escapeHtml(state.llmCompare.prompt)}</textarea>
      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-llm-run" ${state.llmCompare.loading ? "disabled" : ""}>
          ${state.llmCompare.loading ? "요청 중…" : "4개 AI 동시 분석"}
        </button>
      </div>
      ${state.llmCompare.error ? `<p class="err">${escapeHtml(state.llmCompare.error)}</p>` : ""}
    </section>
    ${resultBlock}
  </div>`;
}

function rubricAlignHtml(): string {
  const r = state.rubricAlign.result;
  const resultBlock = r
    ? `
  <section class="panel panel-result hud-panel">
    <h2>정합성 <span class="score-num">${r.alignment_score.toFixed(1)}</span> / 100
      <span class="pill ${r.mode === "ai" ? "pill-on" : "pill-muted"}">${escapeHtml(r.mode)}</span></h2>
    <h4>잘 맞는 루브릭 요소</h4>
    <ul>${r.matched_rubric_points.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    <h4>격차·주의</h4>
    <ul>${r.gaps.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    <h4>제안</h4>
    <p class="prose">${escapeHtml(r.suggestions)}</p>
    <p class="footer-note muted small">${escapeHtml(r.disclaimer || "")}</p>
  </section>`
    : "";
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">채점 품질</p>
    <h1 class="page-title">루브릭·채점 근거 일치 점검</h1>
    <p class="lead analyze-lead">동일 과제의 루브릭과 채점자 코멘트를 넣으세요. OpenAI가 있으면 의미 분석, 없으면 단어 겹침만 봅니다.</p>
    <section class="panel hud-panel">
      <label class="lbl">루브릭 전문</label>
      <textarea class="txt" id="ra_rubric" rows="6">${escapeHtml(state.rubricAlign.rubric)}</textarea>
      <label class="lbl">채점 근거·코멘트</label>
      <textarea class="txt" id="ra_rationale" rows="5">${escapeHtml(state.rubricAlign.grader_rationale)}</textarea>
      <label class="lbl">학생 답안 발췌 (선택)</label>
      <textarea class="txt" id="ra_student" rows="4">${escapeHtml(state.rubricAlign.student_work)}</textarea>
      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-ra-run" ${state.rubricAlign.loading ? "disabled" : ""}>
          ${state.rubricAlign.loading ? "점검 중…" : "정합성 점검"}
        </button>
      </div>
      ${state.rubricAlign.error ? `<p class="err">${escapeHtml(state.rubricAlign.error)}</p>` : ""}
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
    case "team":
      return teamHtml();
    case "at-risk":
      return atRiskHtml();
    case "feedback":
      return feedbackHtml();
    case "syllabus":
      return syllabusHtml();
    case "discussion":
      return discussionHtml();
    case "rubric":
      return rubricAlignHtml();
    case "llm":
      return llmCompareHtml();
    case "analyze":
      return analyzeHtml();
    default:
      return teamHtml();
  }
}

function render(): void {
  const app = document.getElementById("app");
  if (!app) return;
  app.innerHTML = `
  <div class="app-shell">
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
  if (state.view === "syllabus") {
    readSyllabusForm();
  }
  if (state.view === "discussion") {
    readDiscussionForm();
  }
  if (state.view === "rubric") {
    readRubricAlignForm();
  }
  if (state.view === "llm") {
    readLlmCompareForm();
  }
  state.view = v;
  if (v === "team") {
    hydrateTeamDraftIfEmpty();
  }
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

  document.querySelector(".team-form-panel")?.addEventListener("input", scheduleTeamDraftSave);
  document.querySelector(".team-form-panel")?.addEventListener("change", scheduleTeamDraftSave);

  document.getElementById("btn-team-export-json")?.addEventListener("click", () => {
    const r = state.team.result;
    if (!r) return;
    const blob = new Blob([JSON.stringify(r, null, 2)], { type: "application/json;charset=utf-8" });
    const safe = (state.team.project_name || "team-eval").replace(/[\\/:*?"<>|]/g, "_").slice(0, 80);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${safe}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  });

  document.getElementById("btn-team-copy-summary")?.addEventListener("click", async () => {
    const r = state.team.result;
    if (!r) return;
    const text = teamExportSummaryText(r);
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      /* ignore */
    }
  });

  document.getElementById("btn-team-print")?.addEventListener("click", () => {
    window.print();
  });

  document.getElementById("team-sim-member")?.addEventListener("change", updateTeamSimDisplay);
  document.getElementById("team-sim-extra")?.addEventListener("input", updateTeamSimDisplay);
  updateTeamSimDisplay();

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

  document.getElementById("btn-sy-run")?.addEventListener("click", () => {
    void submitSyllabus();
  });

  document.getElementById("btn-disc-add")?.addEventListener("click", () => {
    readDiscussionForm();
    state.discussion.posts.push(emptyDiscussionPost());
    state.view = "discussion";
    render();
  });

  document.getElementById("btn-disc-remove")?.addEventListener("click", () => {
    readDiscussionForm();
    if (state.discussion.posts.length > 1) state.discussion.posts.pop();
    state.view = "discussion";
    render();
  });

  document.getElementById("btn-disc-run")?.addEventListener("click", () => {
    void submitDiscussion();
  });

  document.getElementById("btn-ra-run")?.addEventListener("click", () => {
    void submitRubricAlign();
  });

  document.getElementById("btn-llm-run")?.addEventListener("click", () => {
    void submitLlmCompare();
  });
}

void refreshHealth().then(() => render());
