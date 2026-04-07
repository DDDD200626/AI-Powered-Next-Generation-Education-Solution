import "./style.css";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

const REPO_URL =
  "https://github.com/DDDD200626/AI-Powered-Next-Generation-Education-Solution";

/** 평가 요청 시 AI·문서에 반영되도록 기본 문구 */
const DEFAULT_TEAM_EVALUATION_CRITERIA = `평가 시 참고(수정 가능):
■ 기술적 구현 — 시스템 구조·API·시각화·예외 처리
■ AI 활용 — 다중 모델·생성형 보강(선택)·휴리스틱 폴백
■ 실무 연계 — 팀 과제 운영(지표·동료·결과 점수·협업)과의 연결
■ 분석 깊이 — 기여-결과 불일치·협업 네트워크·역할·이상 탐지·면담 질문·시뮬레이터 등`;

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

/** 백엔드 `/api/health` — 한 번의 요청으로 버전·제공자 확인 */
interface BackendHealth {
  providers?: ProviderKeys;
  version?: string;
  app_name?: string;
  openapi_docs?: string;
}

let lastHealthFetchedAt = 0;
let lastHealthAttemptAt = 0;
const HEALTH_CACHE_MS = 5000;
const HEALTH_RETRY_MS = 4000;
/** 백엔드 미연결 시 주기적으로 /api/health 재시도 */
let healthPollHandle: number | null = null;

function clearHealthPoll(): void {
  if (healthPollHandle != null) {
    window.clearInterval(healthPollHandle);
    healthPollHandle = null;
  }
}

function ensureHealthPollWhenDisconnected(): void {
  if (healthPollHandle != null) return;
  healthPollHandle = window.setInterval(() => {
    void refreshHealth(true).then(() => renderSync());
  }, 6000);
}

/** GET 기본 12s / POST·LLM 180s — 무한 대기 방지 */
function apiFetch(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<Response> {
  const url = path.startsWith("http") ? path : apiUrl(path);
  const timeoutMs =
    init?.timeoutMs ?? (!init?.method || init.method === "GET" ? 12_000 : 180_000);
  const ctrl = new AbortController();
  const tid = window.setTimeout(() => ctrl.abort(), timeoutMs);
  const { timeoutMs: _ignore, ...rest } = init ?? {};
  return fetch(url, { ...rest, signal: ctrl.signal }).finally(() => {
    clearTimeout(tid);
  });
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

interface FreeriderRuleMetrics {
  activity_score?: number;
  collaboration_score?: number;
  consistency_score?: number;
  blended_score?: number;
  final_score?: number;
  penalty_applied?: boolean;
  activity_risk?: boolean;
  collaboration_risk?: boolean;
  consistency_risk?: boolean;
  interaction_risk?: boolean;
  rule_conditions_met?: number;
  risk_level?: string;
  grade_ko?: string;
  analysis_lines?: string[];
}

interface FreeriderDetectionReport {
  basic_low_contribution?: boolean;
  basic_reasons?: string[];
  advanced_pattern_flags?: string[];
  advanced_pattern_score?: number;
  collaboration_isolated?: boolean;
  collaboration_isolation_reasons?: string[];
  below_team_average?: boolean;
  team_mean_contribution?: number;
  delta_vs_team_mean?: number;
  ai_detection_summary?: string;
  rule_metrics?: FreeriderRuleMetrics;
}

interface MemberDashboardRow {
  member_name: string;
  rank: number;
  final_rule_score: number;
  bar_fill_percent: number;
  grade_ko: string;
  rule_conditions_met: number;
  risk_level: string;
  suspected_highlight: boolean;
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
  freerider_detection?: FreeriderDetectionReport;
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
  team_health_score?: number;
  team_health_hint?: string;
}

interface PracticalToolkit {
  teacher_checklist: string[];
  checklist_note?: string;
}

interface RubricMemberRow {
  member_name: string;
  contribution: number;
  collaboration: number;
  persistence: number;
  problem_solving: number;
}

interface RubricReport {
  team_average?: RubricMemberRow;
  members?: RubricMemberRow[];
  criteria_note?: string;
  ai_explanation?: string;
}

interface EvaluationTrustBlock {
  score_0_100?: number;
  level_ko?: string;
  factors?: string[];
  short_note?: string;
}

interface TeamRiskBlock {
  flags?: string[];
  summary_ko?: string;
  ai_team_risk?: string;
}

interface ImprovementChainItem {
  problem?: string;
  explanation?: string;
  suggestion?: string;
  predicted_outcome?: string;
}

interface ImprovementChainBlock {
  headline?: string;
  items?: ImprovementChainItem[];
}

interface TeamEvaluateResponse {
  mode: string;
  members: TeamMemberOut[];
  fairness_notes?: string;
  free_rider_summary?: string;
  freerider_detection_overview?: string;
  product_tagline_ko?: string;
  team_dashboard?: MemberDashboardRow[];
  rubric_report?: RubricReport;
  evaluation_trust?: EvaluationTrustBlock;
  team_risk?: TeamRiskBlock;
  improvement_chain?: ImprovementChainBlock;
  collaboration_network?: NetworkGraph;
  contribution_outcome_summary?: string;
  mismatches?: MismatchItem[];
  anomaly_alerts?: AnomalyAlert[];
  advanced_mode?: string;
  creative_insights?: CreativeInsights;
  practical_toolkit?: PracticalToolkit;
  request_id?: string;
  generated_at?: string;
  processing_ms?: number;
  disclaimer?: string;
}

interface ModelEvalBundle {
  key: string;
  label: string;
  ok: boolean;
  error?: string | null;
  result?: TeamEvaluateResponse | null;
}

interface MemberContributionCompare {
  member_name: string;
  contribution_index_by_model: Record<string, number>;
  spread: number;
  mean: number;
  dimension_spread?: Record<string, number>;
}

interface DivergenceAxisSummary {
  axis_id: string;
  axis_label_ko: string;
  mean_spread: number;
  max_spread_member: string;
  interpretation: string;
}

interface OpinionDivergenceAnalysis {
  primary_axes: DivergenceAxisSummary[];
  criteria_segments: string[];
  criteria_keyword_overlap_note: string;
  narrative: string;
}

interface TrustScoreBlock {
  consistency_0_100: number;
  rubric_alignment_0_100: number;
  explanation_quality_0_100: number;
  overall_trust_0_100: number;
  notes: string[];
}

interface ExplainabilityEntry {
  model_key: string;
  model_label: string;
  member_name: string;
  contribution_index: number;
  technical: number;
  collaboration: number;
  initiative: number;
  evidence_summary: string;
  caveats: string;
  why_one_liner: string;
}

interface EvaluationPipelineStep {
  step: number;
  title_ko: string;
  status: string;
  detail: string;
}

interface TeamCompareResponse {
  request_id?: string;
  generated_at?: string;
  processing_ms?: number;
  product_mode?: string;
  pipeline_steps?: EvaluationPipelineStep[];
  models: ModelEvalBundle[];
  member_comparison: MemberContributionCompare[];
  comparison_summary: string;
  divergence?: OpinionDivergenceAnalysis | null;
  trust_scores?: TrustScoreBlock | null;
  explainability?: ExplainabilityEntry[];
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

interface RubricCriterionGen {
  name: string;
  description: string;
  weight_percent: number;
}

interface RubricGenerateResponse {
  mode: string;
  rubric_markdown: string;
  criteria: RubricCriterionGen[];
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

/** 데모용 샘플 팀 데이터 */
function applyDemoTeamData(): void {
  state.team.project_name = "데모: 풀스택 팀 프로젝트";
  state.team.project_description = "공모전·시연용 샘플입니다. 실제 과제와 무관합니다.";
  state.team.evaluation_criteria = DEFAULT_TEAM_EVALUATION_CRITERIA;
  state.team.collaboration_edges_json = "";
  const a = emptyTeamMember();
  a.name = "김팀장";
  a.role = "리더·백엔드";
  a.commits = "42";
  a.pull_requests = "8";
  a.lines_changed = "2400";
  a.tasks_completed = "7";
  a.meetings_attended = "6";
  a.outcome_score = "88";
  a.self_report = "API 설계·코드 리뷰·스프린트 회의 진행을 맡았습니다.";
  a.peer_notes = "리뷰가 빠릅니다.";
  const b = emptyTeamMember();
  b.name = "이프론트";
  b.role = "프론트엔드";
  b.commits = "35";
  b.pull_requests = "10";
  b.lines_changed = "1800";
  b.tasks_completed = "7";
  b.meetings_attended = "6";
  b.outcome_score = "85";
  b.self_report = "UI 구현, 접근성 개선, 스토리북 작성.";
  b.peer_notes = "";
  const c = emptyTeamMember();
  c.name = "박문서";
  c.role = "문서·기획";
  c.commits = "8";
  c.pull_requests = "3";
  c.lines_changed = "240";
  c.tasks_completed = "9";
  c.meetings_attended = "7";
  c.outcome_score = "82";
  c.self_report = "요구사항 정리·회의록·사용자 가이드 작성.";
  c.peer_notes = "";
  state.team.members = [a, b, c];
  state.team.result = null;
  state.team.error = null;
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

function downloadJsonExport(filenameBase: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json;charset=utf-8" });
  const safe = filenameBase.replace(/[\\/:*?"<>|]/g, "_").slice(0, 80);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${safe}-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
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
  health: BackendHealth | null;
  team: {
    project_name: string;
    project_description: string;
    evaluation_criteria: string;
    members: TeamMemberRow[];
    /** JSON 배열: [{"source":"이름","target":"이름","weight":0-100}] */
    collaboration_edges_json: string;
    result: TeamEvaluateResponse | null;
    loading: boolean;
    compareLoading: boolean;
    compareResult: TeamCompareResponse | null;
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
  rubricGen: {
    course_name: string;
    learning_objectives: string;
    assignment_type: string;
    max_criteria: string;
    result: RubricGenerateResponse | null;
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
    compareLoading: false,
    compareResult: null,
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
  rubricGen: {
    course_name: "",
    learning_objectives: "",
    assignment_type: "",
    max_criteria: "5",
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
let teamSimRafId = 0;

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
  if (res.product_tagline_ko?.trim()) {
    lines.push("");
    lines.push(res.product_tagline_ko.trim());
  }
  if (res.team_dashboard?.length) {
    lines.push("");
    lines.push("[팀 대시보드 · Rule 혼합]");
    res.team_dashboard.forEach((row) => {
      lines.push(
        `#${row.rank} ${row.member_name}: ${row.final_rule_score.toFixed(1)} · ${row.grade_ko || "—"} · ${row.risk_level || "—"}${row.suspected_highlight ? " ⚠" : ""}`
      );
    });
  }
  const et = res.evaluation_trust;
  if (et && et.score_0_100 != null) {
    lines.push("");
    lines.push(`[평가 신뢰도] ${et.score_0_100.toFixed(0)}/100 · ${et.level_ko || "—"}`);
  }
  if (res.team_risk?.summary_ko?.trim()) {
    lines.push("");
    lines.push("[팀 리스크]");
    lines.push(res.team_risk.summary_ko.trim());
  }
  if (res.request_id) lines.push(`요청 ID: ${res.request_id}`);
  if (res.generated_at) lines.push(`생성 시각: ${res.generated_at}`);
  if (res.processing_ms != null) lines.push(`서버 처리: ${res.processing_ms}ms`);
  lines.push("");
  res.members.forEach((m) => {
    lines.push(
      `- ${m.name}: 기여 ${m.contribution_index.toFixed(1)} · 의심도 ${m.free_rider_risk ?? "—"} · ${m.contribution_type_label || "유형 미분류"}`
    );
  });
  if (res.freerider_detection_overview) {
    lines.push("");
    lines.push("[무임승차 자동 탐지 · 4단계]");
    lines.push(res.freerider_detection_overview);
  }
  const ci = res.creative_insights;
  if (ci?.team_health_score != null) {
    lines.push("");
    lines.push(`[팀 협업 건강도(참고)] ${ci.team_health_score.toFixed(1)} / 100`);
    if (ci.team_health_hint) lines.push(ci.team_health_hint);
  }
  const pt = res.practical_toolkit;
  if (pt?.teacher_checklist?.length) {
    lines.push("");
    lines.push("[교육자 실무 체크리스트]");
    pt.teacher_checklist.forEach((x, i) => lines.push(`${i + 1}. ${x}`));
  }
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

async function refreshHealth(force = false): Promise<void> {
  const now = Date.now();
  if (!force && state.health && now - lastHealthFetchedAt < HEALTH_CACHE_MS) return;
  if (!force && !state.health && now - lastHealthAttemptAt < HEALTH_RETRY_MS) return;
  lastHealthAttemptAt = now;
  try {
    const r = await apiFetch("/api/health", { timeoutMs: 10_000 });
    if (!r.ok) {
      state.health = null;
      ensureHealthPollWhenDisconnected();
      return;
    }
    state.health = (await r.json()) as BackendHealth;
    lastHealthFetchedAt = Date.now();
    clearHealthPoll();
  } catch {
    state.health = null;
    ensureHealthPollWhenDisconnected();
  }
}

async function submitAnalyze(): Promise<void> {
  readForm();
  state.error = null;
  state.result = null;
  state.loading = true;
  renderSync();

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
    const r = await apiFetch("/api/analyze", {
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
      renderSync();
      return;
    }
    state.result = data;
    state.view = "analyze";
  } catch (e) {
    state.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.loading = false;
  }
  renderSync();
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
  state.team.compareResult = null;
  state.team.loading = true;
  renderSync();

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
    renderSync();
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
      renderSync();
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
    const r = await apiFetch("/api/team/evaluate", {
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
      renderSync();
      return;
    }
    state.team.result = data;
    clearTeamDraft();
  } catch (e) {
    state.team.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.team.loading = false;
  }
  renderSync();
}

async function submitTeamCompare(): Promise<void> {
  readTeamForm();
  state.team.error = null;
  state.team.result = null;
  state.team.compareResult = null;
  state.team.compareLoading = true;
  renderSync();

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
    state.team.compareLoading = false;
    renderSync();
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
      state.team.error = "협업 네트워크 JSON 형식이 올바르지 않습니다.";
      state.team.compareLoading = false;
      renderSync();
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
    const r = await apiFetch("/api/team/evaluate/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      timeoutMs: 300_000,
    });
    const data = (await r.json()) as TeamCompareResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.team.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.team.compareLoading = false;
      renderSync();
      return;
    }
    state.team.compareResult = data;
  } catch (e) {
    state.team.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.team.compareLoading = false;
  }
  renderSync();
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
  renderSync();

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
    renderSync();
    return;
  }

  const body = {
    course_name: state.atRisk.course_name.trim(),
    student_label: state.atRisk.student_label.trim(),
    weeks,
    notes: state.atRisk.notes,
  };

  try {
    const r = await apiFetch("/api/at-risk/evaluate", {
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
      renderSync();
      return;
    }
    state.atRisk.result = data;
  } catch (e) {
    state.atRisk.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.atRisk.loading = false;
  }
  renderSync();
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
  renderSync();

  if (!state.feedback.rubric.trim() || !state.feedback.submission.trim()) {
    state.feedback.error = "루브릭과 제출물을 입력하세요.";
    state.feedback.loading = false;
    renderSync();
    return;
  }

  const body = {
    rubric: state.feedback.rubric.trim(),
    assignment_prompt: state.feedback.assignment_prompt,
    submission: state.feedback.submission,
  };

  try {
    const r = await apiFetch("/api/feedback/draft", {
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
      renderSync();
      return;
    }
    state.feedback.result = data;
  } catch (e) {
    state.feedback.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.feedback.loading = false;
  }
  renderSync();
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
  renderSync();

  const text = state.syllabus.syllabus_text.trim();
  const q = state.syllabus.question.trim();
  if (text.length < 20 || q.length < 2) {
    state.syllabus.error = "안내 문구는 20자 이상, 질문은 2자 이상 입력하세요.";
    state.syllabus.loading = false;
    renderSync();
    return;
  }

  const body = {
    course_name: state.syllabus.course_name.trim(),
    syllabus_text: text,
    question: q,
  };

  try {
    const r = await apiFetch("/api/course/ask", {
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
      renderSync();
      return;
    }
    state.syllabus.result = data;
  } catch (e) {
    state.syllabus.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.syllabus.loading = false;
  }
  renderSync();
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
  renderSync();

  const posts = state.discussion.posts
    .filter((p) => p.text.trim())
    .map((p) => ({
      author_label: p.author.trim() || "익명",
      text: p.text.trim(),
    }));

  if (posts.length === 0) {
    state.discussion.error = "최소 한 게시글 본문을 입력하세요.";
    state.discussion.loading = false;
    renderSync();
    return;
  }

  const body = {
    thread_title: state.discussion.thread_title.trim(),
    posts,
  };

  try {
    const r = await apiFetch("/api/discussion/synthesize", {
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
      renderSync();
      return;
    }
    state.discussion.result = data;
  } catch (e) {
    state.discussion.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.discussion.loading = false;
  }
  renderSync();
}

function readRubricAlignForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLTextAreaElement | null;
  state.rubricAlign.rubric = g("ra_rubric")?.value ?? "";
  state.rubricAlign.grader_rationale = g("ra_rationale")?.value ?? "";
  state.rubricAlign.student_work = g("ra_student")?.value ?? "";
}

function readRubricGenForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | null;
  state.rubricGen.course_name = g("rg_course")?.value ?? "";
  state.rubricGen.learning_objectives = g("rg_objectives")?.value ?? "";
  state.rubricGen.assignment_type = g("rg_assignment")?.value ?? "";
  state.rubricGen.max_criteria = g("rg_max_criteria")?.value ?? "5";
}

async function submitRubricGenerate(): Promise<void> {
  readRubricGenForm();
  state.rubricGen.error = null;
  state.rubricGen.result = null;
  state.rubricGen.loading = true;
  renderSync();

  const obj = state.rubricGen.learning_objectives.trim();
  if (obj.length < 10) {
    state.rubricGen.error = "학습 목표를 10자 이상 입력하세요.";
    state.rubricGen.loading = false;
    renderSync();
    return;
  }

  let maxC = parseInt(state.rubricGen.max_criteria.trim(), 10);
  if (!Number.isFinite(maxC)) maxC = 5;
  maxC = Math.min(8, Math.max(3, maxC));

  const body = {
    course_name: state.rubricGen.course_name.trim(),
    assignment_type: state.rubricGen.assignment_type.trim(),
    learning_objectives: obj,
    max_criteria: maxC,
  };

  try {
    const r = await apiFetch("/api/rubric/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as RubricGenerateResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.rubricGen.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.rubricGen.loading = false;
      renderSync();
      return;
    }
    state.rubricGen.result = data;
  } catch (e) {
    state.rubricGen.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.rubricGen.loading = false;
  }
  renderSync();
}

async function submitRubricAlign(): Promise<void> {
  readRubricAlignForm();
  state.rubricAlign.error = null;
  state.rubricAlign.result = null;
  state.rubricAlign.loading = true;
  renderSync();

  if (!state.rubricAlign.rubric.trim() || !state.rubricAlign.grader_rationale.trim()) {
    state.rubricAlign.error = "루브릭과 채점 근거를 입력하세요.";
    state.rubricAlign.loading = false;
    renderSync();
    return;
  }

  const body = {
    rubric: state.rubricAlign.rubric.trim(),
    grader_rationale: state.rubricAlign.grader_rationale.trim(),
    student_work_excerpt: state.rubricAlign.student_work.trim(),
  };

  try {
    const r = await apiFetch("/api/rubric/check", {
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
      renderSync();
      return;
    }
    state.rubricAlign.result = data;
  } catch (e) {
    state.rubricAlign.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.rubricAlign.loading = false;
  }
  renderSync();
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
  renderSync();

  const p = state.llmCompare.prompt.trim();
  if (!p) {
    state.llmCompare.error = "분석할 내용을 입력하세요.";
    state.llmCompare.loading = false;
    renderSync();
    return;
  }

  const body = {
    task_title: state.llmCompare.task_title.trim(),
    system_hint: state.llmCompare.system_hint,
    prompt: p,
  };

  try {
    const r = await apiFetch("/api/llm/compare", {
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
      renderSync();
      return;
    }
    state.llmCompare.result = data;
  } catch (e) {
    state.llmCompare.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.llmCompare.loading = false;
  }
  renderSync();
}

function connectionStripHtml(): string {
  const h = state.health;
  const docsHref = API_BASE ? `${API_BASE}/docs` : "/docs";
  const stripClass = h ? "conn-strip conn-strip--live" : "conn-strip conn-strip--partial";
  const dotClass = h ? "api-dot api-dot--on" : "api-dot api-dot--pulse";
  const mainMsg = h
    ? `프론트 ↔ 백엔드 <strong>연결됨</strong> · API <code class="conn-code">${escapeHtml(h.version ?? "?")}</code> · 요청은 <code class="conn-code">/api</code> → 프록시 → <code class="conn-code">:8000</code>`
    : `프론트 ↔ 백엔드 <strong>대기 중</strong> — 브라우저는 <code class="conn-code">/api/health</code>로 확인합니다. 개발: 루트 <code>npm run dev</code>(API+웹) · Docker: <code>8080</code>`;
  const retryBtn = h
    ? ""
    : `<button type="button" class="btn btn-ghost btn-sm" id="btn-health-retry">다시 연결 확인</button>`;
  return `
  <div class="${stripClass}" role="status" aria-live="polite">
    <div class="conn-strip-inner">
      <span class="${dotClass}" aria-hidden="true"></span>
      <span class="conn-msg">${mainMsg}</span>
      <span class="conn-actions">
        ${retryBtn}
        <a class="conn-link" href="${escapeHtml(docsHref)}" target="_blank" rel="noopener noreferrer">OpenAPI</a>
      </span>
    </div>
  </div>`;
}

function providerPills(): string {
  const p = state.health?.providers;
  const ver = state.health?.version;
  const verPill = ver ? `<span class="pill pill-muted">API v${escapeHtml(ver)}</span>` : "";
  const mk = (name: string, on: boolean | undefined) =>
    `<span class="pill ${on ? "pill-on" : "pill-off"}">${name}</span>`;
  if (!p) {
    return verPill
      ? `${verPill} <span class="pill pill-muted">제공자 확인 중…</span>`
      : '<span class="pill pill-muted">백엔드 연결 확인 중…</span>';
  }
  return [verPill, mk("Gemini", p.gemini), mk("ChatGPT", p.openai), mk("Claude", p.claude), mk("Grok", p.grok)]
    .filter(Boolean)
    .join(" ");
}

function navHtml(): string {
  const cur = (v: SiteView) => (state.view === v ? "nav-link active" : "nav-link");
  return `
  <header class="site-header site-header--glass">
    <div class="nav-inner">
      <a href="#" class="brand brand-mark" data-view="team" aria-label="팀 기여도 자동 평가">
        <span class="brand-glow" aria-hidden="true"></span>
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
  const docsHref = API_BASE ? `${API_BASE}/docs` : "/docs";
  const apiMeta =
    state.health?.version != null
      ? ` · <a href="${escapeHtml(docsHref)}" target="_blank" rel="noopener noreferrer">OpenAPI ${escapeHtml(state.health.version)}</a>`
      : ` · <a href="${escapeHtml(docsHref)}" target="_blank" rel="noopener noreferrer">OpenAPI</a>`;
  return `
  <footer class="site-footer">
    <div class="footer-inner">
      <p class="footer-line">
        <a href="${REPO_URL}" target="_blank" rel="noopener noreferrer">저장소</a>
        ${apiMeta}
        · 교육 보조 도구이며 징계·단정에 사용할 수 없습니다.
      </p>
      <p class="footer-muted">팀 프로젝트 기여도 자동 평가 시스템 — 교육 보조·참고용</p>
    </div>
  </footer>`;
}

function hubHtml(): string {
  return `
  <div class="page page-animate home-page">
    <section class="hero-block home-hero-main home-hero-visual">
      <div class="hero-orb hero-orb--a" aria-hidden="true"></div>
      <div class="hero-orb hero-orb--b" aria-hidden="true"></div>
      <p class="eyebrow eyebrow--shine">부가 모듈</p>
      <h1 class="home-headline home-headline--fx">교육 현장 보조 AI</h1>
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
          <h3>루브릭 초안·채점 정합성</h3>
          <p>학습 목표로 루브릭 초안을 만들거나, 기존 루브릭과 채점 코멘트의 일치를 점검합니다.</p>
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

function teamNetworkSvg(net: NetworkGraph | undefined, members?: TeamMemberOut[]): string {
  if (!net?.nodes?.length) return "";
  const iso = new Set<string>();
  if (members) {
    for (const m of members) {
      const id = m.name?.trim();
      if (!id) continue;
      if (m.freerider_detection?.collaboration_isolated || m.freerider_detection?.rule_metrics?.interaction_risk) {
        iso.add(id);
      }
    }
  }
  const maxCi = Math.max(...net.nodes.map((n) => n.contribution_index), 1);
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
      const isHub = n.contribution_index >= maxCi * 0.92 && maxCi > 0;
      const isolated = iso.has(n.id);
      const r = isHub ? 24 : 20;
      const fill = isolated ? "#fef2f2" : isHub ? "#eff6ff" : "#ffffff";
      const stroke = isolated ? "#f87171" : isHub ? "#2563eb" : "#cbd5e1";
      const sw = isolated || isHub ? 2.2 : 1.5;
      const badge = isolated
        ? `<text x="${n.x}" y="${n.y - r - 6}" text-anchor="middle" fill="#f87171" font-size="9" font-weight="700" font-family="system-ui,sans-serif">고립</text>`
        : isHub
          ? `<text x="${n.x}" y="${n.y - r - 6}" text-anchor="middle" fill="#2563eb" font-size="9" font-weight="700" font-family="system-ui,sans-serif">중심</text>`
          : "";
      return `<g class="${isolated ? "network-node--isolated" : ""} ${isHub ? "network-node--hub" : ""}">
      ${badge}
      <circle cx="${n.x}" cy="${n.y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" />
      <text x="${n.x}" y="${n.y + 4}" text-anchor="middle" fill="#0f172a" font-size="10" font-family="system-ui,sans-serif">${escapeHtml(lab)}</text>
      <text x="${n.x}" y="${n.y + (isHub ? 30 : 28)}" text-anchor="middle" fill="#64748b" font-size="8" font-family="system-ui,sans-serif">${n.contribution_index.toFixed(0)}</text>
    </g>`;
    })
    .join("");
  return `
  <div class="network-chart-wrap hud-panel">
    <h3 class="subh">협업 네트워크</h3>
    <p class="muted small">노드=팀원, 선=협업 가중치. <strong>중심</strong>=기여 상위, <strong>고립</strong>=Rule 상호작용 의심. 간선 없으면 서버가 추정합니다.</p>
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
    <h3 class="subh">프로젝트 기간별 기여 비중 (%)</h3>
    <p class="muted small">초반·후반 누가 얼마나 기여했는지 한눈에 봅니다.</p>
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
      `<circle cx="${cx}" cy="${cy}" r="${rmax * t}" fill="none" stroke="#2a3547" stroke-width="1" />`
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
    <path d="${pathD}" fill="rgba(91,143,255,0.2)" stroke="#5b8fff" stroke-width="2" />
  </svg>`;
}

function teamPracticalPanelHtml(res: TeamEvaluateResponse): string {
  const p = res.practical_toolkit;
  if (!p?.teacher_checklist?.length) return "";
  const items = p.teacher_checklist.map((i) => `<li>${escapeHtml(i)}</li>`).join("");
  return `
  <section class="practical-panel hud-panel" aria-labelledby="practical-heading">
    <h3 class="subh" id="practical-heading">교육자 실무 체크리스트</h3>
    <p class="muted small practical-note">${escapeHtml(p.checklist_note || "")}</p>
    <ul class="practical-checklist">${items}</ul>
  </section>`;
}

function teamCreativePanelHtml(res: TeamEvaluateResponse): string {
  const c = res.creative_insights;
  if (!c) return "";
  const rk = c.reflection_kit;
  const hs = c.team_health_score;
  const healthBlock =
    hs != null
      ? `<div class="team-health-meter" role="group" aria-label="팀 협업 건강도 참고">
    <div class="team-health-top">
      <span class="team-health-label">팀 협업 건강도 (참고)</span>
      <span class="team-health-num">${hs.toFixed(1)}<span class="team-health-max">/100</span></span>
    </div>
    <div class="health-bar-wrap" aria-hidden="true"><div class="health-bar-fill" style="width:${Math.min(100, Math.max(0, hs))}%"></div></div>
    <p class="muted small team-health-hint">${escapeHtml(c.team_health_hint || "")}</p>
  </div>`
      : "";
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
    ${healthBlock}
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

/** 슬라이더·셀렉트 입력을 rAF로 합쳐 레이아웃 부담 완화 */
function updateTeamSimDisplay(): void {
  if (teamSimRafId) return;
  teamSimRafId = requestAnimationFrame(() => {
    teamSimRafId = 0;
    updateTeamSimDisplayImpl();
  });
}

function updateTeamSimDisplayImpl(): void {
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

function pipelineStatusLabel(status: string): string {
  if (status === "completed") return "완료";
  if (status === "partial") return "일부";
  if (status === "skipped") return "생략";
  return status;
}

function teamComparePanelHtml(): string {
  const c = state.team.compareResult;
  if (!c) return "";
  const modeBadge =
    c.product_mode === "ai_multi_eval" ? ` <span class="pill pill-on">AI 멀티평가</span>` : "";
  const pipelineBlock = (() => {
    const steps = c.pipeline_steps;
    if (!steps?.length) return "";
    const items = steps
      .map((s) => {
        const cls =
          s.status === "completed"
            ? "eval-pipeline__item--done"
            : s.status === "partial"
              ? "eval-pipeline__item--partial"
              : "eval-pipeline__item--skip";
        return `<li class="eval-pipeline__item ${cls}">
          <div class="eval-pipeline__head">
            <span class="eval-pipeline__n">STEP ${s.step}</span>
            <span class="eval-pipeline__title">${escapeHtml(s.title_ko)}</span>
            <span class="eval-pipeline__badge">${escapeHtml(pipelineStatusLabel(s.status))}</span>
          </div>
          ${s.detail ? `<p class="eval-pipeline__detail muted small">${escapeHtml(s.detail)}</p>` : ""}
        </li>`;
      })
      .join("");
    return `<div class="hud-panel eval-pipeline-wrap">
      <h3 class="subh">처리 파이프라인 (STEP 1–6)</h3>
      <ol class="eval-pipeline">${items}</ol>
    </div>`;
  })();
  const modelPills = c.models
    .map((b) => {
      const err = b.ok ? "" : ` <span class="muted small">(${escapeHtml(b.error || "")})</span>`;
      return `<span class="pill ${b.ok ? "pill-on" : "pill-off"}">${escapeHtml(b.key)} · ${escapeHtml(b.label)}</span>${err}`;
    })
    .join(" ");

  const divergenceBlock = (() => {
    const d = c.divergence;
    if (!d?.primary_axes?.length) return "";
    const axRows = d.primary_axes
      .map(
        (a) =>
          `<tr><td>${escapeHtml(a.axis_label_ko)}</td><td>${a.mean_spread.toFixed(2)}</td><td>${escapeHtml(a.max_spread_member || "—")}</td><td class="muted small">${escapeHtml(a.interpretation)}</td></tr>`
      )
      .join("");
    const crit =
      d.criteria_segments?.length > 0
        ? `<ul class="criteria-list muted small">${d.criteria_segments.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul>`
        : "";
    return `
    <div class="hud-panel team-compare-div">
      <h3 class="subh">STEP 4 · AI 간 차이 분석</h3>
      ${d.narrative ? `<p class="prose">${escapeHtml(d.narrative)}</p>` : ""}
      ${d.criteria_keyword_overlap_note ? `<p class="muted small">${escapeHtml(d.criteria_keyword_overlap_note)}</p>` : ""}
      ${crit ? `<p class="muted small">입력 평가 기준(발췌)</p>${crit}` : ""}
      <div class="table-scroll-wrap">
      <table class="data-table data-table--compact">
        <thead><tr><th>축</th><th>평균 편차</th><th>편차 최대 팀원</th><th>해석</th></tr></thead>
        <tbody>${axRows}</tbody>
      </table></div>
    </div>`;
  })();

  const trustBlock = (() => {
    const t = c.trust_scores;
    if (!t) return "";
    const noteList = (t.notes || []).map((n) => `<li>${escapeHtml(n)}</li>`).join("");
    return `
    <div class="hud-panel team-compare-trust">
      <h3 class="subh">STEP 5 · 신뢰도 계산 (0–100)</h3>
      <div class="trust-score-grid">
        <div><span class="trust-label">일관성</span> <strong>${t.consistency_0_100.toFixed(1)}</strong></div>
        <div><span class="trust-label">루브릭 일치</span> <strong>${t.rubric_alignment_0_100.toFixed(1)}</strong></div>
        <div><span class="trust-label">설명 품질</span> <strong>${t.explanation_quality_0_100.toFixed(1)}</strong></div>
        <div><span class="trust-label">종합</span> <strong>${t.overall_trust_0_100.toFixed(1)}</strong></div>
      </div>
      ${noteList ? `<ul class="muted small">${noteList}</ul>` : ""}
    </div>`;
  })();

  const explainBlock = (() => {
    const ex = c.explainability;
    if (!ex?.length) return "";
    const cards = ex
      .map(
        (e) => `
      <article class="explain-card hud-panel">
        <h4 class="explain-card__title">${escapeHtml(e.model_label)} · ${escapeHtml(e.member_name)}</h4>
        <p class="muted small">기여 ${e.contribution_index.toFixed(1)} · 기술 ${e.technical.toFixed(0)} · 협업 ${e.collaboration.toFixed(0)} · 주도 ${e.initiative.toFixed(0)}</p>
        <p class="prose">${escapeHtml(e.why_one_liner)}</p>
        ${e.evidence_summary ? `<p class="prose small"><strong>근거</strong> ${escapeHtml(e.evidence_summary)}</p>` : ""}
        ${e.caveats ? `<p class="muted small"><strong>주의</strong> ${escapeHtml(e.caveats)}</p>` : ""}
      </article>`
      )
      .join("");
    return `
    <div class="team-compare-explain">
      <h3 class="subh">STEP 6 · 모델별 근거·피드백 (Explainable AI)</h3>
      <div class="explain-grid">${cards}</div>
    </div>`;
  })();

  const keys = [...new Set(c.member_comparison.flatMap((m) => Object.keys(m.contribution_index_by_model)))];
  if (!keys.length) {
    return `
  <section class="panel panel-result hud-panel team-compare-panel">
    <h2>AI 멀티평가${modeBadge}</h2>
    ${pipelineBlock}
    <p class="muted small">${escapeHtml(c.comparison_summary)}</p>
    <div class="pill-row">${modelPills}</div>
    <p class="err">비교할 수 있는 모델 결과가 없습니다. backend/.env 에 각 제공자 API 키를 넣었는지 확인하세요.</p>
    ${divergenceBlock}
    ${trustBlock}
    ${explainBlock}
    <p class="footer-note muted small">${escapeHtml(c.disclaimer || "")}</p>
  </section>`;
  }
  const th = keys.map((k) => `<th>${escapeHtml(k)}</th>`).join("");
  const trs = c.member_comparison
    .map((m) => {
      const tds = keys
        .map((k) => {
          const v = m.contribution_index_by_model[k];
          return `<td>${v !== undefined ? v.toFixed(1) : "—"}</td>`;
        })
        .join("");
      return `<tr><td>${escapeHtml(m.member_name)}</td>${tds}<td>${m.spread.toFixed(2)}</td><td>${m.mean.toFixed(1)}</td></tr>`;
    })
    .join("");
  const meta = [c.request_id ? `요청 ${escapeHtml(c.request_id)}` : "", c.processing_ms != null ? `${c.processing_ms.toFixed(0)} ms` : ""]
    .filter(Boolean)
    .join(" · ");
  return `
  <section class="panel panel-result hud-panel team-compare-panel">
    <h2>AI 멀티평가${modeBadge}</h2>
    ${pipelineBlock}
    <p class="muted small">${escapeHtml(c.comparison_summary)}</p>
    ${meta ? `<p class="muted small">${meta}</p>` : ""}
    <p class="muted small step-inline-hint"><strong>STEP 2</strong> · 연동 모델</p>
    <div class="pill-row">${modelPills}</div>
    <p class="muted small step-inline-hint"><strong>STEP 3</strong> · 루브릭(평가 기준)에 따라 각 모델이 차원·기여 점수를 산출합니다.</p>
    ${divergenceBlock}
    ${trustBlock}
    <h3 class="subh">STEP 6 · 최종 점수 (모델별 기여 지수 비교)</h3>
    <div class="table-scroll-wrap">
      <table class="data-table">
        <thead><tr><th>팀원</th>${th}<th>편차</th><th>평균</th></tr></thead>
        <tbody>${trs}</tbody>
      </table>
    </div>
    ${explainBlock}
    <p class="footer-note muted small">${escapeHtml(c.disclaimer || "")}</p>
  </section>`;
}

function teamDashboardBarHtml(res: TeamEvaluateResponse): string {
  const tag = res.product_tagline_ko?.trim();
  const dash = res.team_dashboard;
  if (!tag && !(dash && dash.length)) return "";
  const tagBlock = tag
    ? `<div class="product-tagline hud-panel"><p class="prose tagline-prose">${escapeHtml(tag)}</p></div>`
    : "";
  const bars = (dash || [])
    .map((row) => {
      const warn = row.suspected_highlight
        ? ` <span class="dash-warn" title="Rule 기준 의심·저점">⚠️</span>`
        : "";
      const risk =
        row.risk_level === "suspected"
          ? `<span class="pill pill-warn pill-sm">위험</span>`
          : row.risk_level === "caution"
            ? `<span class="pill pill-muted pill-sm">주의</span>`
            : `<span class="pill pill-on pill-sm">정상</span>`;
      return `<div class="dash-row">
      <div class="dash-rank">#${row.rank}</div>
      <div class="dash-name">${escapeHtml(row.member_name)}${warn}</div>
      <div class="dash-bar-wrap" role="img" aria-label="기여 점수 막대">
        <div class="dash-bar-fill" style="width:${Math.min(100, row.bar_fill_percent)}%"></div>
      </div>
      <div class="dash-score">${row.final_rule_score.toFixed(1)}</div>
      <div class="dash-grade">${escapeHtml(row.grade_ko)}</div>
      <div class="dash-risk">${risk} <span class="muted small">${row.rule_conditions_met}조건</span></div>
    </div>`;
    })
    .join("");
  const dashBlock =
    dash && dash.length
      ? `<div class="team-dash hud-panel">
      <h3 class="subh">팀 기여 대시보드</h3>
      <p class="muted small">막대·순위 = Rule 혼합(활동×0.4+협업×0.3+시간×0.3), <strong>코드·데이터 산출</strong>. 3지표 이상 위험 시 페널티(×0.6).</p>
      <div class="dash-head"><span>순위</span><span>이름</span><span>막대</span><span>점수</span><span>등급</span><span>판정</span></div>
      ${bars}
    </div>`
      : "";
  return `${tagBlock}${dashBlock}`;
}

function teamContestUpgradeHtml(res: TeamEvaluateResponse): string {
  const rub = res.rubric_report;
  const tr = res.evaluation_trust;
  const risk = res.team_risk;
  const chain = res.improvement_chain;
  const rubRows = (rub?.members || [])
    .map(
      (r) => `<tr>
      <td>${escapeHtml(r.member_name)}</td>
      <td>${r.contribution.toFixed(0)}</td>
      <td>${r.collaboration.toFixed(0)}</td>
      <td>${r.persistence.toFixed(0)}</td>
      <td>${r.problem_solving.toFixed(0)}</td>
    </tr>`
    )
    .join("");
  const ta = rub?.team_average;
  const rubTable =
    rubRows || ta
      ? `<div class="contest-block hud-panel">
      <h3 class="subh">루브릭 4축 (규칙·데이터)</h3>
      <p class="muted small">${escapeHtml(rub?.criteria_note || "")}</p>
      <table class="data-table contest-rubric-table">
        <thead><tr><th>이름</th><th>기여도</th><th>협업 참여</th><th>지속성</th><th>문제 해결 기여</th></tr></thead>
        <tbody>${rubRows}</tbody>
        ${
          ta
            ? `<tfoot><tr><td><strong>${escapeHtml(ta.member_name)}</strong></td>
          <td>${ta.contribution.toFixed(0)}</td><td>${ta.collaboration.toFixed(0)}</td>
          <td>${ta.persistence.toFixed(0)}</td><td>${ta.problem_solving.toFixed(0)}</td></tr></tfoot>`
            : ""
        }
      </table>
    </div>`
      : "";
  const trustPill =
    tr && tr.score_0_100 != null
      ? `<div class="contest-block hud-panel contest-trust">
      <h3 class="subh">평가 신뢰도</h3>
      <p><span class="pill ${tr.level_ko === "높음" ? "pill-on" : tr.level_ko === "중간" ? "pill-muted" : "pill-warn"}">신뢰도: ${escapeHtml(tr.level_ko || "—")}</span>
      <strong class="trust-score">${tr.score_0_100.toFixed(0)}</strong><span class="muted small">/100</span></p>
      <p class="muted small">${escapeHtml(tr.short_note || "")}</p>
      <ul class="fr-ul">${(tr.factors || []).map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul>
    </div>`
      : "";
  const riskBlock =
    risk && (risk.flags?.length || risk.summary_ko)
      ? `<div class="contest-block hud-panel">
      <h3 class="subh">팀 리스크 (전체)</h3>
      <p class="prose small">${escapeHtml(risk.summary_ko || "")}</p>
      <ul class="fr-ul">${(risk.flags || []).map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul>
    </div>`
      : "";
  const chainItems = (chain?.items || [])
    .map(
      (it) => `<article class="chain-item hud-panel">
      <p><strong>문제</strong> ${escapeHtml(it.problem || "")}</p>
      <p class="muted small"><strong>설명</strong> ${escapeHtml(it.explanation || "")}</p>
      <p><strong>개선</strong> ${escapeHtml(it.suggestion || "")}</p>
      <p class="muted small"><strong>기대</strong> ${escapeHtml(it.predicted_outcome || "")}</p>
    </article>`
    )
    .join("");
  const chainBlock = chainItems
    ? `<div class="contest-chain-wrap hud-panel">
      <h3 class="subh">${escapeHtml(chain?.headline || "설명 → 개선 → 기대 효과")}</h3>
      <div class="chain-grid">${chainItems}</div>
    </div>`
    : "";
  if (!rubTable && !trustPill && !riskBlock && !chainBlock) return "";
  return `<section class="contest-upgrade" aria-label="수상권 강화 블록">${rubTable}${trustPill}${riskBlock}${chainBlock}</section>`;
}

function teamFreeriderPanelHtml(res: TeamEvaluateResponse): string {
  const ov = res.freerider_detection_overview?.trim();
  if (!ov && !(res.members || []).some((m) => m.freerider_detection)) return "";
  const cards = (res.members || [])
    .map((m) => {
      const fr = m.freerider_detection;
      if (!fr) return "";
      const rm = fr.rule_metrics;
      const ruleBlock =
        rm && (rm.final_score != null || rm.grade_ko)
          ? `<div class="fr-layer fr-rule">
        <strong>Rule 혼합</strong>
        <p class="muted small">점수 = 활동×0.4 + 협업×0.3 + 시간×0.3 → <strong>${(rm.blended_score ?? 0).toFixed(1)}</strong>${rm.penalty_applied ? ` → 페널티 적용 <strong>${(rm.final_score ?? 0).toFixed(1)}</strong> (×0.6)` : ` → 최종 <strong>${(rm.final_score ?? 0).toFixed(1)}</strong>`}</p>
        <p class="muted small">활동 ${(rm.activity_score ?? 0).toFixed(1)} · 협업 ${(rm.collaboration_score ?? 0).toFixed(1)} · 시간 ${(rm.consistency_score ?? 0).toFixed(1)} · 4지표 중 ${rm.rule_conditions_met ?? 0}개 위험</p>
        <p><span class="fr-badge fr-badge--grade">${escapeHtml(rm.grade_ko || "")}</span> · ${escapeHtml(rm.risk_level === "suspected" ? "의심(3+)" : rm.risk_level === "caution" ? "주의(2)" : "정상(≤1)")}</p>
        <ul class="fr-ul">${(rm.analysis_lines || []).map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul>
      </div>`
          : "";
      const b1 = fr.basic_low_contribution
        ? `<span class="fr-badge fr-badge--basic">Low Contribution</span>`
        : `<span class="muted small">1차 해당 없음</span>`;
      const b3 = fr.collaboration_isolated
        ? `<span class="fr-badge fr-badge--iso">협업 고립 사용자</span>`
        : `<span class="muted small">3차 해당 없음</span>`;
      const b4 = fr.below_team_average
        ? `<span class="fr-badge fr-badge--avg">팀 평균 대비 낮음</span>`
        : `<span class="muted small">4차 해당 없음</span>`;
      const advList = (fr.advanced_pattern_flags || []).length
        ? `<ul class="fr-ul">${(fr.advanced_pattern_flags || []).map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul>`
        : `<p class="muted small">2차 비정상 패턴 신호 없음</p>`;
      const ai = fr.ai_detection_summary?.trim()
        ? `<div class="fr-layer fr-ai"><strong>AI 설명</strong> <span class="muted small">(탐지·점수는 Rule·데이터)</span><p class="prose small">${escapeHtml(fr.ai_detection_summary || "")}</p></div>`
        : "";
      return `
    <article class="fr-card hud-panel">
      <h4>${escapeHtml(m.name)}</h4>
      ${ruleBlock}
      <div class="fr-layer"><strong>1️⃣ 기본 탐지</strong> ${b1}<ul class="fr-ul">${(fr.basic_reasons || []).map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul></div>
      <div class="fr-layer"><strong>2️⃣ 고급 패턴</strong> <span class="muted small">패턴 점수 ${(fr.advanced_pattern_score ?? 0).toFixed(1)}</span>${advList}</div>
      <div class="fr-layer"><strong>3️⃣ 협업 고립</strong> ${b3}<ul class="fr-ul">${(fr.collaboration_isolation_reasons || []).map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul></div>
      <div class="fr-layer"><strong>4️⃣ 팀 평균 비교</strong> ${b4} <span class="muted small">팀 평균 ${(fr.team_mean_contribution ?? 0).toFixed(1)} · Δ ${(fr.delta_vs_team_mean ?? 0).toFixed(1)}</span></div>
      ${ai}
    </article>`;
    })
    .join("");
  return `
  <div class="freerider-panel hud-panel" id="freerider-detection-panel">
    <h3 class="subh">무임승차 탐지 — 탐지·점수는 Rule, 설명은 AI(선택)</h3>
    ${ov ? `<p class="prose muted small">${escapeHtml(ov)}</p>` : ""}
    <div class="fr-cards-grid">${cards}</div>
  </div>`;
}

function teamResultHtml(): string {
  const res = state.team.result;
  if (!res) return "";
  const rows = res.members
    .map((m) => {
      const sus = m.free_rider_suspected
        ? `<span class="free-rider-badge" title="자동 의심">무임승차 의심</span>`
        : `<span class="pill pill-muted">—</span>`;
      const fr = m.freerider_detection;
      const frMini =
        fr &&
        [
          fr.basic_low_contribution ? `<span class="fr-badge fr-badge--mini fr-badge--basic" title="1차">Low</span>` : "",
          (fr.advanced_pattern_flags || []).length ? `<span class="fr-badge fr-badge--mini fr-badge--adv" title="2차">패턴</span>` : "",
          fr.collaboration_isolated ? `<span class="fr-badge fr-badge--mini fr-badge--iso" title="3차">고립</span>` : "",
          fr.below_team_average ? `<span class="fr-badge fr-badge--mini fr-badge--avg" title="4차">평균↓</span>` : "",
        ]
          .filter(Boolean)
          .join(" ");
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
      <td>${escapeHtml(m.name)} ${sus} ${frMini || ""}</td>
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
  const net = teamNetworkSvg(res.collaboration_network, res.members);
  const dashBlock = teamDashboardBarHtml(res);
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
  const practicalBlock = teamPracticalPanelHtml(res);
  const freeriderBlock = teamFreeriderPanelHtml(res);
  const contestBlock = teamContestUpgradeHtml(res);
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
    ${dashBlock}
    ${freeriderBlock}
    ${contestBlock}
    ${creativeBlock}
    ${practicalBlock}
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
    <h3 class="subh">팀원별 피드백</h3>
    <p class="muted small">기여 지수·근거는 규칙·데이터 기반입니다. 아래 문단은 서술 보강·면담용(키 있을 때).</p>
    <div class="member-feedback-grid">${feedbackBlocks}</div>
    <p class="footer-note muted small">${escapeHtml(res.disclaimer || "")}</p>
  </section>`;
}

function teamHtml(): string {
  const blocks = state.team.members.map((m, i) => teamMemberBlock(i, m)).join("");
  return `
  <div class="page page-analyze page-animate analyze-page">
    <p class="eyebrow eyebrow--shine">Team Project · Auto Evaluation</p>
    <h1 class="page-title page-title--hero">팀 프로젝트 기여도 자동 평가</h1>
    <p class="lead analyze-lead">
      정량 지표·주차별 활동·서술을 바탕으로 <strong>기여 지수와 Rule 판단은 규칙·데이터로</strong> 산출합니다.
      <strong>무임승차 탐지·네트워크·타임라인</strong>은 코드 기반이며, OpenAI 키가 있으면 <strong>설명·해석·피드백 문장</strong> 품질이 올라갑니다(점수 자체를 AI가 매기지는 않음).
      <strong>창의 인사이트</strong>·팀원별 문단은 참고용입니다. 최종 성적은 교수·기관 규정에 따릅니다.
    </p>
    <div class="pill-row">${providerPills()}</div>
    <section class="panel hud-panel team-arch-panel">
      <h3 class="subh">전체 시스템 구조 (핵심 5단계)</h3>
      <p class="muted small">데이터 수집 → 정제 → 분석 → AI 해석 → 결과 출력</p>
      <div class="team-arch-grid">
        <article class="team-arch-step"><strong>1) 데이터 수집</strong><p class="muted small">Git·문서·협업 로그·시간 데이터를 모아 “얼마나”보다 “어떻게 참여했는지”를 봅니다.</p></article>
        <article class="team-arch-step"><strong>2) 데이터 정제</strong><p class="muted small">사용자 단위로 묶고 팀 평균·비교값을 계산해 분석 가능한 형태로 만듭니다.</p></article>
        <article class="team-arch-step"><strong>3) 분석 엔진</strong><p class="muted small">기여도·Rule 혼합 점수는 <strong>데이터·규칙으로만</strong> 산출합니다. AI는 점수 계산에 쓰지 않습니다.</p></article>
        <article class="team-arch-step"><strong>4) AI 해석</strong><p class="muted small">무임승차 설명·협업 구조 해석·피드백 문장·개선 제안 등 <strong>이해·설명</strong>에만 사용합니다.</p></article>
        <article class="team-arch-step"><strong>5) 결과 출력</strong><p class="muted small">팀 대시보드·개인 상세·협업 네트워크·타임라인으로 심사위원이 빠르게 이해하게 합니다.</p></article>
      </div>
      <p class="muted small team-arch-line">AI는 점수를 만드는 것이 아니라 점수를 이해시키는 역할입니다.</p>
      <p class="muted small team-arch-connection">백엔드 연결: 프론트는 <code>/api</code>로 요청하고 개발 시 Vite 프록시가 <code>http://127.0.0.1:8000</code>으로 전달합니다.</p>
    </section>
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
      <label class="lbl">평가 기준 (기본값 있음, 수정 가능)</label>
      <textarea class="txt" id="team_evaluation_criteria" rows="6">${escapeHtml(state.team.evaluation_criteria)}</textarea>
      <label class="lbl">협업 네트워크 (선택, JSON 배열)</label>
      <textarea class="txt mono-input" id="team_collaboration_edges" rows="3" placeholder='[{"source":"이름A","target":"이름B","weight":40}]'>${escapeHtml(state.team.collaboration_edges_json)}</textarea>
      <p class="muted small">source·target은 위 멤버 이름과 동일하게. weight는 0–100. 비우면 서버가 기여 지수로 간선을 추정합니다.</p>
      ${blocks}
      <div class="row-actions">
        <button type="button" class="btn btn-ghost" id="btn-team-demo">데모 데이터 입력</button>
        <button type="button" class="btn btn-ghost" id="btn-team-add">멤버 추가</button>
        <button type="button" class="btn btn-ghost" id="btn-team-remove" ${state.team.members.length <= 1 ? "disabled" : ""}>마지막 멤버 제거</button>
        <button type="button" class="btn btn-primary" id="btn-team-run" ${state.team.loading || state.team.compareLoading ? "disabled" : ""}>
          ${state.team.loading ? "자동 평가 중…" : "자동 평가 실행"}
        </button>
        <button type="button" class="btn btn-ghost" id="btn-team-compare" ${state.team.loading || state.team.compareLoading ? "disabled" : ""}>
          ${state.team.compareLoading ? "AI 멀티평가 실행 중…" : "AI 멀티평가 (ChatGPT·Gemini·Claude·Grok)"}
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
    ${
      state.team.compareLoading
        ? `<div class="team-loading hud-panel" aria-live="polite" aria-busy="true">
      <div class="skeleton-line skeleton-line--long"></div>
      <p class="muted small">각 제공자 최신 모델로 순차·병렬 호출 중입니다. 키가 모두 있으면 최대 수 분 걸릴 수 있습니다.</p>
    </div>`
        : ""
    }
    ${teamComparePanelHtml()}
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

function rubricGenResultHtml(): string {
  const r = state.rubricGen.result;
  if (!r) return "";
  const rows = r.criteria
    .map(
      (c) =>
        `<tr><td>${escapeHtml(c.name)}</td><td class="muted small">${escapeHtml(c.description)}</td><td>${c.weight_percent.toFixed(1)}%</td></tr>`
    )
    .join("");
  return `
  <section class="panel panel-result hud-panel">
    <div class="result-toolbar no-print" role="toolbar" aria-label="루브릭 초안 내보내기">
      <button type="button" class="btn btn-ghost btn-sm" id="btn-rg-export-json">JSON 내보내기</button>
      <button type="button" class="btn btn-ghost btn-sm" id="btn-rg-copy-md">마크다운 복사</button>
    </div>
    <h2>루브릭 초안 <span class="pill ${r.mode === "ai" ? "pill-on" : "pill-muted"}">${escapeHtml(r.mode)}</span></h2>
    <div class="table-scroll-wrap">
      <table class="data-table"><thead><tr><th>항목</th><th>기준</th><th>배점</th></tr></thead><tbody>${rows}</tbody></table>
    </div>
    <h3 class="subh">마크다운 전문</h3>
    <pre class="rubric-md-preview mono-input">${escapeHtml(r.rubric_markdown)}</pre>
    <p class="footer-note muted small">${escapeHtml(r.disclaimer || "")}</p>
  </section>`;
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
    <div class="result-toolbar no-print" role="toolbar" aria-label="결과 내보내기">
      <button type="button" class="btn btn-ghost btn-sm" id="btn-llm-export-json">JSON 내보내기</button>
    </div>
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
    <p class="lead analyze-lead">학습 목표에서 <strong>루브릭 초안</strong>을 만들거나, 기존 루브릭과 채점 코멘트의 <strong>정합성</strong>을 점검할 수 있습니다. OpenAI가 있으면 서술 품질이 올라갑니다.</p>
    <section class="panel hud-panel">
      <h4 class="subh">1) 학습 목표 → 루브릭 초안</h4>
      <div class="grid-2">
        <div><label class="lbl">과목명 (선택)</label><input class="txt" id="rg_course" value="${escapeHtml(state.rubricGen.course_name)}" placeholder="예: 소프트웨어공학" /></div>
        <div><label class="lbl">과제 유형 (선택)</label><input class="txt" id="rg_assignment" value="${escapeHtml(state.rubricGen.assignment_type)}" placeholder="팀 프로젝트, 보고서…" /></div>
      </div>
      <div class="grid-2">
        <div>
          <label class="lbl">항목 수 (3–8)</label>
          <input class="txt" id="rg_max_criteria" type="number" min="3" max="8" step="1" value="${escapeHtml(state.rubricGen.max_criteria)}" />
        </div>
      </div>
      <label class="lbl">학습 목표·성과 목표 (10자 이상)</label>
      <textarea class="txt" id="rg_objectives" rows="5" placeholder="예: REST API 설계·문서화, 보안·오류 처리 등">${escapeHtml(state.rubricGen.learning_objectives)}</textarea>
      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-rg-run" ${state.rubricGen.loading ? "disabled" : ""}>
          ${state.rubricGen.loading ? "생성 중…" : "루브릭 초안 생성"}
        </button>
      </div>
      ${state.rubricGen.error ? `<p class="err">${escapeHtml(state.rubricGen.error)}</p>` : ""}
    </section>
    ${rubricGenResultHtml()}
    <section class="panel hud-panel">
      <h4 class="subh">2) 루브릭·채점 근거 정합성 점검</h4>
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
    <div class="result-toolbar no-print" role="toolbar" aria-label="결과 내보내기">
      <button type="button" class="btn btn-ghost btn-sm" id="btn-analyze-export-json">JSON 내보내기</button>
    </div>
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

let renderRafId = 0;

/** 즉시 DOM 반영 — 로딩 표시·비동기 완료 후에는 이쪽(다음 프레임 지연 없음) */
function renderSync(): void {
  if (renderRafId) {
    cancelAnimationFrame(renderRafId);
    renderRafId = 0;
  }
  const app = document.getElementById("app");
  if (!app) return;
  app.innerHTML = `
  <div class="app-shell app-shell--fx">
    ${navHtml()}
    ${connectionStripHtml()}
    <main class="site-main site-main--fx">${mainContentHtml()}</main>
    ${footerHtml()}
  </div>`;
  wire();
}

/** 동기 스택에서 연속 호출 시 한 프레임으로 합쳐 reflow 비용 절감 */
function render(): void {
  if (renderRafId) return;
  renderRafId = requestAnimationFrame(() => {
    renderRafId = 0;
    renderSync();
  });
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
    readRubricGenForm();
  }
  if (state.view === "llm") {
    readLlmCompareForm();
  }
  state.view = v;
  if (v === "team") {
    hydrateTeamDraftIfEmpty();
  }
  void refreshHealth(true).then(() => {
    renderSync();
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

  document.getElementById("btn-team-demo")?.addEventListener("click", () => {
    applyDemoTeamData();
    state.view = "team";
    render();
    scheduleTeamDraftSave();
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

  document.getElementById("btn-team-compare")?.addEventListener("click", () => {
    void submitTeamCompare();
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

  document.getElementById("btn-rg-run")?.addEventListener("click", () => {
    void submitRubricGenerate();
  });

  document.getElementById("btn-rg-export-json")?.addEventListener("click", () => {
    const x = state.rubricGen.result;
    if (!x) return;
    downloadJsonExport("rubric-draft", x);
  });

  document.getElementById("btn-rg-copy-md")?.addEventListener("click", async () => {
    const x = state.rubricGen.result;
    if (!x) return;
    try {
      await navigator.clipboard.writeText(x.rubric_markdown);
    } catch {
      /* ignore */
    }
  });

  document.getElementById("btn-analyze-export-json")?.addEventListener("click", () => {
    if (!state.result) return;
    downloadJsonExport("analyze-learning-exam", state.result);
  });

  document.getElementById("btn-llm-export-json")?.addEventListener("click", () => {
    if (!state.llmCompare.result) return;
    downloadJsonExport("llm-compare", state.llmCompare.result);
  });

  document.getElementById("btn-llm-run")?.addEventListener("click", () => {
    void submitLlmCompare();
  });

  document.getElementById("btn-health-retry")?.addEventListener("click", () => {
    void refreshHealth(true).then(() => renderSync());
  });

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      void refreshHealth(true).then(() => renderSync());
    }
  });
}

/** 백엔드 health 전에도 셸을 먼저 그려 빈 화면을 막음 */
function boot(): void {
  renderSync();
  void refreshHealth(true).then(() => renderSync());
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
