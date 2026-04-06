"""AI(가능 시) + 휴리스틱 기반 팀 기여도 평가."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from team_eval.schemas import (
    DimensionScores,
    MemberEvaluation,
    MemberInput,
    TeamEvaluateRequest,
    TeamEvaluateResponse,
)


DISCLAIMER_KO = (
    "본 결과는 입력된 수치·서술을 바탕으로 한 자동 추정입니다. "
    "커밋 수·라인 수만으로 공정한 평가가 불가능한 경우가 많으며, "
    "교육기관의 공식 채점·최종 판단을 대체하지 않습니다."
)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    return json.loads(text)


def _heuristic_evaluate(req: TeamEvaluateRequest) -> TeamEvaluateResponse:
    members = req.members
    n = len(members)
    commits = [float(m.commits or 0) for m in members]
    tasks = [float(m.tasks_completed or 0) for m in members]
    lines = [float(m.lines_changed or 0) for m in members]
    prs = [float(m.pull_requests or 0) for m in members]
    meet = [float(m.meetings_attended or 0) for m in members]
    words = [max(0, len((m.self_report or "").split())) for m in members]

    def norm(vals: list[float]) -> list[float]:
        s = sum(vals) or 1.0
        return [v / s for v in vals]

    nc = norm(commits)
    nt = norm(tasks)
    nl = norm(lines)
    npr = norm(prs)
    nw = norm([float(w) for w in words])
    nmeet = norm(meet)

    out_members: list[MemberEvaluation] = []
    raw_scores: list[float] = []
    for i, m in enumerate(members):
        # 가중: 정량 + 서술 가중 (서술만 있는 팀도 분산 가능)
        blend = (
            0.22 * nc[i]
            + 0.18 * nt[i]
            + 0.18 * nl[i]
            + 0.12 * npr[i]
            + 0.30 * nw[i]
        )
        raw_scores.append(blend)

    # 0–100 기여 지수 (상대 분배)
    raw_sum = sum(raw_scores) or 1.0
    for i, m in enumerate(members):
        idx = 100.0 * (raw_scores[i] / raw_sum)
        # 차원: 기술은 정량 비중, 협업은 회의·서술, 주도성은 태스크·PR
        q = nc[i] + nl[i] + npr[i] + 1e-6
        tech = min(
            100.0,
            30.0 + 70.0 * (0.45 * nc[i] + 0.35 * nl[i] + 0.20 * npr[i]) / q,
        )
        collab = min(100.0, 35.0 + 45.0 * nw[i] + 20.0 * nmeet[i])
        init_ = min(100.0, 32.0 + 40.0 * nt[i] + 28.0 * npr[i])

        out_members.append(
            MemberEvaluation(
                name=m.name,
                role=m.role,
                contribution_index=round(idx, 1),
                dimensions=DimensionScores(
                    technical=round(float(tech), 1),
                    collaboration=round(float(collab), 1),
                    initiative=round(float(init_), 1),
                ),
                evidence_summary=(
                    f"정량 입력 기준 상대 비중(커밋·태스크·PR·라인·자기서술)을 합성했습니다. "
                    f"동일 기여를 서술로도 설명했는지 확인이 필요합니다."
                ),
                caveats="OpenAI 키가 없어 휴리스틱만 사용했습니다. 가능하면 AI 모드로 재평가하세요.",
            )
        )

    imbalance = ""
    if n > 1 and commits and max(commits) > 0:
        mx = max(commits)
        if mx / (sum(commits) / n) > 2.5:
            imbalance = "커밋 수 편차가 큽니다. 리뷰·기획·문서 기여는 수치에 잘 안 잡힙니다."

    return TeamEvaluateResponse(
        mode="heuristic",
        project_summary=(req.project_description or "")[:2000],
        fairness_notes=imbalance or "정량 지표는 참고용이며, 역할 분담·기술 난이도는 반영되지 않았습니다.",
        members=out_members,
        disclaimer=DISCLAIMER_KO,
    )


def _build_prompt(req: TeamEvaluateRequest) -> str:
    payload = {
        "project_name": req.project_name,
        "project_description": req.project_description,
        "evaluation_criteria": req.evaluation_criteria,
        "members": [
            {
                "name": m.name,
                "role": m.role,
                "commits": m.commits,
                "pull_requests": m.pull_requests,
                "lines_changed": m.lines_changed,
                "tasks_completed": m.tasks_completed,
                "meetings_attended": m.meetings_attended,
                "self_report": m.self_report,
                "peer_notes": m.peer_notes,
            }
            for m in req.members
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def evaluate_with_openai(req: TeamEvaluateRequest, api_key: str) -> TeamEvaluateResponse:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    sys = """당신은 대학 교육 맥락의 팀 프로젝트 기여도를 공정하게 평가하는 조교입니다.
반드시 아래 JSON 스키마만 출력하세요. 한국어 문자열을 사용합니다.
스키마:
{
  "project_summary": "한 문단 요약",
  "fairness_notes": "공정성·한계 안내",
  "members": [
    {
      "name": "이름",
      "role": "역할",
      "contribution_index": 0-100 숫자,
      "dimensions": { "technical": 0-100, "collaboration": 0-100, "initiative": 0-100 },
      "evidence_summary": "근거 요약",
      "caveats": "주의점"
    }
  ]
}
원칙:
- 커밋·라인 수만으로 순위를 매기지 말고, 문서·리뷰·기획·디버깅 등 비가시 기여를 self_report/peer_notes에서 반영하세요.
- 수치가 비어 있으면 서술과 역할을 기준으로 합리적으로 추정하세요.
- 모든 멤버의 contribution_index 합이 100이 될 필요는 없습니다(상대 비교 지수).
- 편향을 줄이기 위해 극단적 0 또는 100은 남용하지 마세요."""

    user = f"평가 데이터:\n{_build_prompt(req)}"

    try:
        res = client.chat.completions.create(
            model=os.environ.get("OPENAI_EVAL_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        content = (res.choices[0].message.content or "").strip()
        data = _parse_json_loose(content)
    except (json.JSONDecodeError, Exception):
        return _heuristic_evaluate(req)

    rows = [r for r in (data.get("members") or []) if isinstance(r, dict)]
    by_name = {str(r.get("name") or "").strip(): r for r in rows}

    members_out: list[MemberEvaluation] = []
    for m in req.members:
        raw = by_name.get(m.name.strip()) or by_name.get(m.name)
        if raw is None and len(rows) == len(req.members):
            raw = rows[len(members_out)]
        if raw is None:
            return _heuristic_evaluate(req)
        name = str(raw.get("name") or m.name)
        d = raw.get("dimensions") or {}
        members_out.append(
            MemberEvaluation(
                name=name,
                role=str(raw.get("role") or m.role),
                contribution_index=min(
                    100.0,
                    max(0.0, _safe_float(raw.get("contribution_index"), 50.0)),
                ),
                dimensions=DimensionScores(
                    technical=min(100.0, max(0.0, _safe_float(d.get("technical"), 50.0))),
                    collaboration=min(100.0, max(0.0, _safe_float(d.get("collaboration"), 50.0))),
                    initiative=min(100.0, max(0.0, _safe_float(d.get("initiative"), 50.0))),
                ),
                evidence_summary=str(raw.get("evidence_summary") or "")[:2000],
                caveats=str(raw.get("caveats") or "")[:1500],
            )
        )

    if len(members_out) != len(req.members):
        return _heuristic_evaluate(req)

    return TeamEvaluateResponse(
        mode="ai",
        project_summary=str(data.get("project_summary") or "")[:3000],
        fairness_notes=str(data.get("fairness_notes") or "")[:2000],
        members=members_out,
        disclaimer=DISCLAIMER_KO,
    )


def evaluate(req: TeamEvaluateRequest) -> TeamEvaluateResponse:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        return evaluate_with_openai(req, key)
    return _heuristic_evaluate(req)
