

from __future__ import annotations

import json
import os
import re
from typing import Any

from edu_tools.team_unified_eval import (
    TeamReportRequest,
    TeamUserIn,
    apply_dl_scores,
    run_score_engine,
)


def _extract_json_obj(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        t = m.group(0)
    return json.loads(t)


def _llm_baseline_scores_openai(users: list[TeamUserIn]) -> dict[str, float]:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        return {}
    from learning_analysis.llm_clients import get_openai_client

    members = [
        {
            "member_name": u.name.strip(),
            "commits": u.commits,
            "prs": u.prs,
            "lines_changed": u.lines,
            "attendance_0_100": u.attendance,
            "self_report_excerpt": (u.selfReport or "")[:900],
        }
        for u in users
    ]
    sys_msg = (
        "You output ONLY valid JSON. Task: assign each member a team contribution score from 0 to 100 "
        "based ONLY on the numeric metrics and short excerpt provided. No prose. "
        'Schema: {"scores":[{"member_name":"string","score":number}]}. '
        "Use one score per member_name exactly as given."
    )
    model = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    try:
        client = get_openai_client(key)
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": json.dumps({"members": members}, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = res.choices[0].message.content or "{}"
        data = _extract_json_obj(raw)
        arr = data.get("scores")
        if not isinstance(arr, list):
            return {}
        out: dict[str, float] = {}
        for it in arr:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("member_name", "")).strip()
            try:
                sc = float(it.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            out[nm.lower()] = max(0.0, min(100.0, sc))
        return out
    except Exception:
        return {}


def run_narrow_benchmark(body: TeamReportRequest) -> dict[str, Any]:
    """기준: 룰 엔진 normalizedScore. 낮은 MAE가 동일 기준에 더 가깝다."""
    users = body.teamData
    scores, _ = run_score_engine(users)
    dl_info = apply_dl_scores(users, scores, request_id="", learning=False)
    if dl_info.get("enabled") is False:
        return {
            "status": "dl_not_ready",
            "interpretation_ko": "저장된 전용 DL/선형 모델이 없어 룰 점수만으로는 비교할 수 없습니다. 평가를 몇 번 돌려 학습 샘플을 쌓으세요.",
            "reference": "rule_engine_normalizedScore",
            "mae_team_dl": None,
            "mae_llm_baseline_openai": None,
            "winner": None,
            "llm_available": False,
            "dl_model_info": dl_info,
            "per_member": [],
        }

    ref = [float(s.normalizedScore) for s in scores]
    dl_vals: list[float] = []
    for s in scores:
        if s.dl_score is not None:
            dl_vals.append(float(s.dl_score))
        else:
            dl_vals.append(float(s.normalizedScore))
    n = len(ref)
    mae_dl = sum(abs(ref[i] - dl_vals[i]) for i in range(n)) / max(n, 1)

    llm_map = _llm_baseline_scores_openai(users)
    llm_vals: list[float] = []
    if llm_map:
        for s in scores:
            key = s.member_name.strip().lower()
            llm_vals.append(llm_map.get(key, float(s.normalizedScore)))
        mae_llm = sum(abs(ref[i] - llm_vals[i]) for i in range(n)) / max(n, 1)
    else:
        llm_vals = []
        mae_llm = None

    winner: str | None = None
    if mae_llm is not None:
        if mae_dl < mae_llm - 1e-6:
            winner = "team_dl"
        elif mae_llm < mae_dl - 1e-6:
            winner = "llm_baseline"
        else:
            winner = "tie"

    per_member: list[dict[str, Any]] = []
    for i, s in enumerate(scores):
        row: dict[str, Any] = {
            "member_name": s.member_name,
            "reference_rule_normalized": round(ref[i], 2),
            "team_dl": round(dl_vals[i], 2),
        }
        if llm_vals:
            row["llm_baseline_openai"] = round(llm_vals[i], 2)
        per_member.append(row)

    return {
        "status": "ok",
        "interpretation_ko": (
            "같은 팀 입력에 대해 '룰 엔진 정규화 점수'에 누가 더 가까운지(MAE) 비교. "
            "범용 LLM 전체 능력 순위가 아니라 이 좁은 수치 과제에서의 오차다."
        ),
        "reference": "rule_engine_normalizedScore",
        "mae_team_dl": round(mae_dl, 4),
        "mae_llm_baseline_openai": round(mae_llm, 4) if mae_llm is not None else None,
        "winner": winner,
        "llm_available": mae_llm is not None,
        "dl_model_info": dl_info,
        "per_member": per_member,
    }
