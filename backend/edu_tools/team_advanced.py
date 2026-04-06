"""기여–결과 불일치, 협업 네트워크, 역할 4유형, 비정상 행위 탐지(고급)."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from pydantic import BaseModel, Field


def _openai_timeout_sec() -> float:
    raw = (os.environ.get("OPENAI_TIMEOUT_SEC") or "120").strip()
    try:
        v = float(raw)
    except ValueError:
        v = 120.0
    return max(5.0, min(600.0, v))


class CollaborationEdgeIn(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    weight: float = Field(5.0, ge=0, le=100, description="상호작용·협업 강도")


class MismatchItem(BaseModel):
    member_name: str
    contribution_index: float
    outcome_score: float | None = None
    gap: float | None = None
    note: str = ""
    severity: str = "low"


class NetworkNode(BaseModel):
    id: str
    label: str
    x: float
    y: float
    contribution_index: float = 0


class NetworkEdge(BaseModel):
    source: str
    target: str
    weight: float = 1.0


class NetworkGraph(BaseModel):
    nodes: list[NetworkNode] = Field(default_factory=list)
    edges: list[NetworkEdge] = Field(default_factory=list)


class AnomalyAlert(BaseModel):
    member_name: str
    code: str
    severity: str = "medium"
    message: str = ""


ROLE_TYPES = ("개발형", "문서형", "리더형", "서포터형")


def _layout_circular(names: list[str], cis: list[float]) -> list[NetworkNode]:
    n = len(names)
    nodes: list[NetworkNode] = []
    for i, name in enumerate(names):
        ang = 2 * math.pi * i / n - math.pi / 2 if n else 0
        x = 260 + 200 * math.cos(ang)
        y = 240 + 200 * math.sin(ang)
        ci = cis[i] if i < len(cis) else 0.0
        nodes.append(NetworkNode(id=name, label=name, x=x, y=y, contribution_index=round(ci, 1)))
    return nodes


def _normalize_role_scores(d: dict[str, float]) -> dict[str, float]:
    s = sum(max(0, v) for v in d.values()) or 1.0
    return {k: round(100.0 * max(0, d.get(k, 0)) / s, 1) for k in ("dev", "doc", "leader", "supporter")}


def heuristic_roles(
    commits: float,
    lines: float,
    prs: float,
    tasks: float,
    meetings: float,
    words: float,
    tech: float,
    collab: float,
    init_: float,
) -> tuple[str, dict[str, float]]:
    """개발·문서·리더·서포터 점수 휴리스틱."""
    dev_raw = 0.35 * min(commits / 50.0, 1.0) * 100 + 0.25 * min(lines / 5000.0, 1.0) * 100 + 0.2 * min(prs / 10.0, 1.0) * 100 + 0.2 * (tech / 100.0) * 100
    doc_raw = 0.55 * min(words / 200.0, 1.0) * 100 + 0.25 * (collab / 100.0) * 100 + 0.2 * min(tasks / 15.0, 1.0) * 100
    leader_raw = 0.4 * min(meetings / 8.0, 1.0) * 100 + 0.35 * (init_ / 100.0) * 100 + 0.25 * (collab / 100.0) * 100
    supporter_raw = 0.5 * (collab / 100.0) * 100 + 0.25 * min(tasks / 12.0, 1.0) * 100 + 0.25 * (100 - min(dev_raw, 100)) * 0.01 * 50

    raw = {
        "dev": max(0, dev_raw),
        "doc": max(0, doc_raw),
        "leader": max(0, leader_raw),
        "supporter": max(0, supporter_raw),
    }
    scores = _normalize_role_scores(raw)
    primary = max(scores, key=scores.get)
    label_map = {"dev": "개발형", "doc": "문서형", "leader": "리더형", "supporter": "서포터형"}
    return label_map[primary], scores


def build_network(
    member_names: list[str],
    contribution_indices: list[float],
    edges_in: list[CollaborationEdgeIn] | None,
) -> NetworkGraph:
    nodes = _layout_circular(member_names, contribution_indices)
    name_set = set(member_names)
    edges: list[NetworkEdge] = []
    if edges_in:
        for e in edges_in:
            a, b = e.source.strip(), e.target.strip()
            if a in name_set and b in name_set and a != b:
                edges.append(NetworkEdge(source=a, target=b, weight=min(100, max(0, e.weight))))
    if not edges and len(member_names) >= 2:
        for i in range(len(member_names)):
            for j in range(i + 1, len(member_names)):
                wi = contribution_indices[i] if i < len(contribution_indices) else 0
                wj = contribution_indices[j] if j < len(contribution_indices) else 0
                w = min(wi, wj) * 0.5 + 10.0
                edges.append(NetworkEdge(source=member_names[i], target=member_names[j], weight=round(min(100, w), 1)))
    return NetworkGraph(nodes=nodes, edges=edges)


def compute_mismatches(
    names: list[str],
    contribution_indices: list[float],
    outcome_scores: list[float | None],
) -> tuple[list[MismatchItem], str]:
    items: list[MismatchItem] = []
    lines: list[str] = []
    for i, name in enumerate(names):
        oc = outcome_scores[i] if i < len(outcome_scores) else None
        ci = contribution_indices[i] if i < len(contribution_indices) else 0
        if oc is None:
            continue
        gap = abs(ci - oc)
        sev = "low"
        if gap >= 35:
            sev = "high"
        elif gap >= 20:
            sev = "medium"
        note = f"추정 기여 {ci:.1f} vs 결과 점수 {oc:.1f}, 차이 {gap:.1f}."
        if gap >= 15:
            items.append(
                MismatchItem(
                    member_name=name,
                    contribution_index=round(ci, 1),
                    outcome_score=round(oc, 1),
                    gap=round(gap, 1),
                    note=note,
                    severity=sev,
                )
            )
            lines.append(f"{name}: {note}")
    summary = (
        "기여 지수(활동 추정)와 입력한 결과 점수를 비교했습니다. 차이가 크면 산출 방식·평가 기준·누락 활동을 검토하세요.\n"
        + ("\n".join(lines) if lines else "결과 점수가 입력된 멤버가 없거나 차이가 작습니다.")
    )
    return items, summary


def compute_anomalies(
    names: list[str],
    contribution_indices: list[float],
    free_rider: list[bool],
    mismatches: list[MismatchItem],
    edges: list[NetworkEdge],
) -> list[AnomalyAlert]:
    alerts: list[AnomalyAlert] = []
    incoming: dict[str, int] = {n: 0 for n in names}
    for e in edges:
        if e.target in incoming:
            incoming[e.target] += 1
        if e.source in incoming:
            pass
    for i, name in enumerate(names):
        if free_rider[i] if i < len(free_rider) else False:
            alerts.append(
                AnomalyAlert(
                    member_name=name,
                    code="FREE_RIDER",
                    severity="high",
                    message="무임승차 의심 신호와 연계된 비정상 패턴(고급).",
                )
            )
        if incoming.get(name, 0) == 0 and len(names) > 2 and len(edges) > 0:
            alerts.append(
                AnomalyAlert(
                    member_name=name,
                    code="NETWORK_ISOLATE",
                    severity="medium",
                    message="협업 네트워크상 유입 상호작용이 거의 없습니다(입력 간선 확인).",
                )
            )
    for m in mismatches:
        if m.severity == "high":
            alerts.append(
                AnomalyAlert(
                    member_name=m.member_name,
                    code="CONTRIBUTION_OUTCOME_GAP",
                    severity="high",
                    message=f"기여 추정과 결과 점수 괴리가 큽니다(차이 {m.gap}).",
                )
            )
    seen: set[tuple[str, str]] = set()
    uniq: list[AnomalyAlert] = []
    for a in alerts:
        k = (a.member_name, a.code)
        if k not in seen:
            seen.add(k)
            uniq.append(a)
    return uniq[:12]


def openai_enrich_advanced(
    project: dict[str, Any],
    members_payload: list[dict[str, Any]],
    heuristic_bundle: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    from openai import OpenAI

    sys = """교육용 팀 분석가입니다. JSON만 출력하세요.
{
  "contribution_outcome_comment": "기여-결과 불일치에 대한 한국어 해설(2~4문장)",
  "roles": [{"name":"이름","primary":"개발형|문서형|리더형|서포터형 중 하나","rationale":"한 줄"}],
  "anomaly_notes": [{"name":"이름","note":"비정상 의심 보조 설명, 없으면 생략"}]
}
휴리스틱과 크게 다르지 않아도 되나, peer_notes·self_report 맥락을 반영하세요."""

    client = OpenAI(api_key=api_key, timeout=_openai_timeout_sec())
    user = json.dumps(
        {"project": project, "members": members_payload, "heuristic": heuristic_bundle},
        ensure_ascii=False,
    )
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.25,
    )
    raw = (res.choices[0].message.content or "{}").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        raw = m.group(0)
    return json.loads(raw)
