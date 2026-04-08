"""
Git 기반 팀 데이터 평가 — Score Engine(계산) / Anomaly(판단) / AI(설명) 분리.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from edu_tools.team_data_store import db_profile, record_team_report
from edu_tools.team_ml_model import (
    append_samples,
    feature_row,
    load_model,
    predict_score,
    train_if_needed,
)
from edu_tools.team_web_priors import get_web_priors
from learning_analysis.llm_clients import get_openai_client

router = APIRouter()

DEFAULT_WEIGHTS = {
    "commit": 0.25,
    "pr": 0.20,
    "lines": 0.20,
    "attendance": 0.15,
    "self_report": 0.20,
}


class TeamUserIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    commits: int = Field(0, ge=0)
    prs: int = Field(0, ge=0)
    lines: int = Field(0, ge=0)
    attendance: float = Field(0.0, ge=0, le=100)
    selfReport: str = Field("", max_length=12000)


class TeamReportRequest(BaseModel):
    project_name: str = Field("", max_length=300)
    teamData: list[TeamUserIn] = Field(..., min_length=1, max_length=40)


class ScoreBreakdown(BaseModel):
    commits: float
    prs: float
    lines: float
    attendance: float
    self_report: float


class WeightedPoints(BaseModel):
    commits: float
    prs: float
    lines: float
    attendance: float
    self_report: float


class DataReliability(BaseModel):
    score_0_100: float
    git_ratio_percent: float
    self_report_ratio_percent: float
    note: str = ""


class ScoreResult(BaseModel):
    member_name: str
    rawScore: float
    normalizedScore: float
    rank: int
    top_percent: float
    breakdown: ScoreBreakdown
    weighted_points: WeightedPoints
    pct_vs_team_mean: float
    data_reliability: DataReliability
    dl_score: float | None = Field(
        default=None, ge=0, le=100, description="경량 신경망 기반 데이터 학습 점수(보강 신호)"
    )
    dl_confidence: float | None = Field(
        default=None, ge=0, le=100, description="입력 밀도·분산 기반 신뢰도"
    )
    blendedScore: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="최종 점수: 0.7 * normalizedScore + 0.3 * dl_score",
    )


class AnomalyResult(BaseModel):
    member_name: str
    flags: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    member_name: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    position_in_team: str = ""
    recommended_actions: list[str] = Field(default_factory=list)


class TeamReportResponse(BaseModel):
    scores: list[ScoreResult]
    anomalies: list[AnomalyResult]
    analysis: list[AnalysisResult]
    edge_cases: list[str] = Field(default_factory=list)
    trust_scores: dict[str, float] = Field(default_factory=dict)
    dl_model_info: dict[str, Any] = Field(default_factory=dict)
    evaluation_log: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = Field(
        default="정량 점수는 결정론적으로 계산되며, AI는 설명만 생성합니다."
    )


def _load_weights() -> dict[str, float]:
    p = Path(__file__).with_name("data").joinpath("team_eval_weights.json")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        keys = ("commit", "pr", "lines", "attendance", "self_report")
        out: dict[str, float] = {}
        s = 0.0
        for k in keys:
            v = float(data.get(k, DEFAULT_WEIGHTS[k]))
            if v < 0:
                v = DEFAULT_WEIGHTS[k]
            out[k] = v
            s += v
        if s <= 1e-9:
            return DEFAULT_WEIGHTS.copy()
        return {k: out[k] / s for k in keys}
    except Exception:
        return DEFAULT_WEIGHTS.copy()


def _word_count(s: str) -> int:
    return len([x for x in s.split() if x.strip()])


def _median(vals: list[float]) -> float:
    if not vals:
        return 1.0
    a = sorted(vals)
    n = len(a)
    m = n // 2
    return float(a[m]) if n % 2 else (float(a[m - 1]) + float(a[m])) / 2.0


def _scaled_by_team(value: float, base: float, cap_mult: float = 2.5) -> float:
    m = max(base, 1e-6)
    return min(100.0, 100.0 * min(cap_mult, value / m) / cap_mult)


def run_score_engine(users: list[TeamUserIn]) -> tuple[list[ScoreResult], dict[str, float]]:
    w = _load_weights()
    commits = [float(u.commits) for u in users]
    prs = [float(u.prs) for u in users]
    lines = [float(u.lines) for u in users]
    atts = [float(u.attendance) for u in users]
    words = [float(_word_count(u.selfReport)) for u in users]

    med = {
        "commit": _median(commits),
        "pr": _median(prs),
        "lines": _median(lines),
        "attendance": _median(atts) if max(atts) > 1e-9 else 50.0,
        "self_report": _median(words) if max(words) > 1e-9 else 1.0,
    }

    bds: list[ScoreBreakdown] = []
    pts: list[WeightedPoints] = []
    raws: list[float] = []
    trust: dict[str, float] = {}

    for i, u in enumerate(users):
        bd = ScoreBreakdown(
            commits=_scaled_by_team(commits[i], med["commit"]),
            prs=_scaled_by_team(prs[i], med["pr"]),
            lines=_scaled_by_team(lines[i], med["lines"]),
            attendance=min(100.0, atts[i]),
            self_report=_scaled_by_team(words[i], med["self_report"]),
        )
        pt = WeightedPoints(
            commits=round(w["commit"] * bd.commits, 2),
            prs=round(w["pr"] * bd.prs, 2),
            lines=round(w["lines"] * bd.lines, 2),
            attendance=round(w["attendance"] * bd.attendance, 2),
            self_report=round(w["self_report"] * bd.self_report, 2),
        )
        raw = pt.commits + pt.prs + pt.lines + pt.attendance + pt.self_report
        bds.append(bd)
        pts.append(pt)
        raws.append(raw)

        data_core = (pt.commits + pt.prs + pt.lines + pt.attendance) / max(
            w["commit"] + w["pr"] + w["lines"] + w["attendance"], 1e-9
        )
        gap = abs(data_core - bd.self_report)
        tr = max(0.0, min(100.0, 100.0 - gap * 0.8))
        trust[u.name.strip()] = round(tr, 1)

    n = len(users)
    mean_raw = sum(raws) / max(n, 1)
    mx, mi = max(raws), min(raws)
    span = max(mx - mi, 1e-6)
    ranked = sorted(range(n), key=lambda i: raws[i], reverse=True)
    rank = [0] * n
    for r, idx in enumerate(ranked, start=1):
        rank[idx] = r

    out: list[ScoreResult] = []
    for i, u in enumerate(users):
        top = 100.0 * rank[i] / max(n, 1)
        git_ratio = 100.0 * (w["commit"] + w["pr"] + w["lines"] + w["attendance"])
        self_ratio = 100.0 * w["self_report"]
        rel = DataReliability(
            score_0_100=trust[u.name.strip()],
            git_ratio_percent=round(git_ratio, 1),
            self_report_ratio_percent=round(self_ratio, 1),
            note="자기서술 비중 높음" if self_ratio >= 30 else "",
        )
        out.append(
            ScoreResult(
                member_name=u.name.strip(),
                rawScore=round(raws[i], 2),
                normalizedScore=round(100.0 * (raws[i] - mi) / span, 2),
                rank=rank[i],
                top_percent=round(top, 1),
                breakdown=bds[i],
                weighted_points=pts[i],
                pct_vs_team_mean=round(
                    ((raws[i] - mean_raw) / max(abs(mean_raw), 1e-9)) * 100.0, 1
                ),
                data_reliability=rel,
            )
        )
    return out, trust


def apply_dl_scores(users: list[TeamUserIn], scores: list[ScoreResult]) -> dict[str, Any]:
    """학습 데이터 기반 ML 보강 점수.

    - 요청마다 (feature, target=normalizedScore) 샘플을 누적
    - 샘플 수가 쌓이면 자동 재학습
    - 저장된 최신 모델로 dl_score 산출
    """
    if not users or not scores:
        return {"model_name": "team-ml", "enabled": False}

    priors, priors_meta = get_web_priors()
    samples: list[dict[str, Any]] = []
    for u, s in zip(users, scores):
        x = feature_row(u.commits, u.prs, u.lines, u.attendance, u.selfReport)
        samples.append(
            {
                "x": x,
                "y": float(s.normalizedScore),
                "member": u.name.strip(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    append_samples(samples)
    train_info = train_if_needed()
    model = load_model()
    if not model:
        for s in scores:
            s.dl_score = None
            s.dl_confidence = None
            s.blendedScore = float(s.normalizedScore)
        return {
            "model_name": "team-ml",
            "enabled": False,
            "reason": "model_not_ready",
            "sample_count": train_info.get("sample_count", 0),
            "blend_formula": "0.7*normalizedScore + 0.3*dl_score",
        }

    conf_base = max(35.0, min(95.0, 40.0 + model.sample_count * 0.6))
    for u, s in zip(users, scores):
        x = feature_row(u.commits, u.prs, u.lines, u.attendance, u.selfReport)
        dl_score = predict_score(model, x)
        # Optional web priors adjustment (small bounded delta)
        commit_gap = min(1.0, max(0.0, x[0] / math.log1p(30))) - (priors["commit_expectation"] / 100.0)
        pr_gap = min(1.0, max(0.0, x[1] / math.log1p(12))) - (priors["pr_expectation"] / 100.0)
        att_gap = x[3] - (priors["attendance_expectation"] / 100.0)
        prior_delta = (commit_gap * 5.0) + (pr_gap * 4.0) + (att_gap * 3.0)
        dl_score = max(0.0, min(100.0, dl_score + prior_delta))
        # 입력 다양도 기반 confidence 보정
        x_min, x_max = min(x), max(x)
        spread = x_max - x_min
        conf = max(0.0, min(100.0, conf_base + spread * 8.0))
        s.dl_score = round(dl_score, 2)
        s.dl_confidence = round(conf, 1)
        s.blendedScore = round(0.7 * s.normalizedScore + 0.3 * s.dl_score, 2)

    return {
        "model_name": "team-ml-linear-sgd",
        "enabled": True,
        "model_version": model.version,
        "trained_at": model.trained_at,
        "sample_count": model.sample_count,
        "auto_retrain": bool(train_info.get("trained", False)),
        "web_priors": priors_meta,
        "input_features": [
            "log_commits",
            "log_prs",
            "log_lines",
            "attendance_norm",
            "log_self_report_words",
        ],
        "blend_formula": "0.7*normalizedScore + 0.3*dl_score",
    }


def run_anomaly_detector(users: list[TeamUserIn], scores: list[ScoreResult]) -> list[AnomalyResult]:
    team_mean = sum(s.rawScore for s in scores) / max(len(scores), 1)
    med_lines = _median([float(u.lines) for u in users])
    out: list[AnomalyResult] = []
    for u, s in zip(users, scores):
        flags: list[str] = []
        wc = _word_count(u.selfReport)
        if u.commits == 0 and wc >= 35:
            flags.append("커밋 0 + 자기서술 과다: 데이터 신뢰도 확인 필요")
        if u.lines > max(1.0, med_lines) * 2 and u.prs == 0:
            flags.append("코드량 대비 PR 부족: 협업 점검 필요")
        if u.attendance < 40 and s.rawScore >= team_mean * 1.15 and team_mean > 1e-9:
            flags.append("출석 낮음 대비 점수 높음: 이상 패턴 점검")
        if u.commits == 0 and u.prs == 0 and u.lines == 0 and wc < 5:
            flags.append("평가 불가에 가까운 데이터 부족")
        out.append(AnomalyResult(member_name=u.name.strip(), flags=flags))
    return out


def detect_edge_cases(users: list[TeamUserIn], scores: list[ScoreResult]) -> list[str]:
    notes: list[str] = []
    git_total = sum((u.commits + u.prs + u.lines) for u in users)
    self_total = sum(_word_count(u.selfReport) for u in users)
    if git_total == 0 and self_total == 0:
        notes.append("CASE: 데이터 없음 — 평가 불가")
    if scores:
        raws = sorted((s.rawScore for s in scores), reverse=True)
        if len(raws) >= 2 and raws[0] >= max(1e-9, sum(raws) * 0.65):
            notes.append("CASE: 한 명 집중 — 팀 불균형 가능")
        if all(s.normalizedScore < 35 for s in scores):
            notes.append("CASE: 전반 저조 — 팀 전체 성과 저조")
    return notes


def _extract_json(text: str) -> dict[str, Any]:
    t = text.strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        t = m.group(0)
    return json.loads(t)


def _fallback_analysis(
    users: list[TeamUserIn], scores: list[ScoreResult], anomalies: list[AnomalyResult]
) -> list[AnalysisResult]:
    by_anom = {a.member_name: a.flags for a in anomalies}
    by_score = {s.member_name: s for s in scores}
    out: list[AnalysisResult] = []
    n = len(scores)
    for u in users:
        name = u.name.strip()
        s = by_score.get(name)
        if not s:
            continue
        flags = by_anom.get(name, [])
        rec = ["PR 리뷰 참여를 주차 목표에 포함하세요."]
        if any("협업" in f or "PR" in f for f in flags):
            rec = ["PR 분할 제출 + 리뷰 코멘트 기록을 늘리세요."]
        out.append(
            AnalysisResult(
                member_name=name,
                summary=f"{name}의 기여도는 {s.normalizedScore:.0f}/100, 팀 평균 대비 {s.pct_vs_team_mean:+.1f}%입니다.",
                strengths=["Git 지표 기반으로 산정 근거를 추적할 수 있습니다."],
                weaknesses=flags[:3] or ["특이 이상 없음"],
                position_in_team=f"팀 {n}명 중 {s.rank}위 (상위 {s.top_percent:.1f}%).",
                recommended_actions=rec,
            )
        )
    return out


def run_ai_analyzer(
    users: list[TeamUserIn], scores: list[ScoreResult], anomalies: list[AnomalyResult]
) -> list[AnalysisResult]:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        return _fallback_analysis(users, scores, anomalies)
    payload = {
        "members": [
            {
                "member_name": s.member_name,
                "score": s.model_dump(),
                "anomaly_flags": next((a.flags for a in anomalies if a.member_name == s.member_name), []),
                "self_report_excerpt": next((u.selfReport for u in users if u.name.strip() == s.member_name), "")[:600],
            }
            for s in scores
        ]
    }
    sys = (
        "점수는 절대 바꾸지 말고 설명만 하세요. JSON 객체로만 출력: "
        '{"members":[{"member_name":"","summary":"","strengths":[],"weaknesses":[],"position_in_team":"","recommended_actions":[]}]}'
    )
    try:
        client = get_openai_client(key)
        res = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = _extract_json(res.choices[0].message.content or "{}")
        arr = data.get("members") if isinstance(data, dict) else None
        if not isinstance(arr, list):
            return _fallback_analysis(users, scores, anomalies)
        out: list[AnalysisResult] = []
        for it in arr:
            if not isinstance(it, dict):
                continue
            out.append(
                AnalysisResult(
                    member_name=str(it.get("member_name", ""))[:120],
                    summary=str(it.get("summary", ""))[:1500],
                    strengths=[str(x) for x in (it.get("strengths") or [])][:5],
                    weaknesses=[str(x) for x in (it.get("weaknesses") or [])][:5],
                    position_in_team=str(it.get("position_in_team", ""))[:400],
                    recommended_actions=[str(x) for x in (it.get("recommended_actions") or [])][:5],
                )
            )
        return out if len(out) == len(scores) else _fallback_analysis(users, scores, anomalies)
    except Exception:
        return _fallback_analysis(users, scores, anomalies)


@router.post("/report", response_model=TeamReportResponse)
def post_team_report(body: TeamReportRequest) -> TeamReportResponse:
    users = body.teamData
    scores, trust = run_score_engine(users)
    dl_model_info = apply_dl_scores(users, scores)
    anomalies = run_anomaly_detector(users, scores)
    analysis = run_ai_analyzer(users, scores, anomalies)
    edge_cases = detect_edge_cases(users, scores)

    req_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    input_data = [u.model_dump() for u in users]
    input_blob = json.dumps(input_data, ensure_ascii=False, sort_keys=True)
    input_hash = hashlib.sha256(input_blob.encode("utf-8")).hexdigest()[:24]
    evaluation_log = {
        "request_id": req_id,
        "timestamp": ts,
        "project_name": body.project_name.strip(),
        "input_hash": input_hash,
        "input_data": input_data,
        "score": [s.model_dump() for s in scores],
        "dl_model_info": dl_model_info,
        "anomaly": [a.model_dump() for a in anomalies],
        "ai_result": [a.model_dump() for a in analysis],
        "edge_cases": edge_cases,
    }
    try:
        record_team_report(
            request_id=req_id,
            project_name=body.project_name.strip(),
            users=input_data,
            scores=[s.model_dump() for s in scores],
            anomalies=[a.model_dump() for a in anomalies],
            trust_scores=trust,
            dl_model_info=dl_model_info,
        )
        evaluation_log["db_profile"] = db_profile()
    except Exception:
        # DB 저장 실패가 평가 API 실패로 이어지지 않도록 방어
        evaluation_log["db_profile"] = {"error": "db_store_failed"}
    return TeamReportResponse(
        scores=scores,
        anomalies=anomalies,
        analysis=analysis,
        edge_cases=edge_cases,
        trust_scores=trust,
        dl_model_info=dl_model_info,
        evaluation_log=evaluation_log,
    )
