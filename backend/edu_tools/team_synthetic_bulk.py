"""
합성 팀·멤버 샘플 대량 생성 → team_ml_dataset.jsonl 추가.

실제 교수/조교 라벨을 대체하지 않으며, 데이터 파이프라인·학습 스트레스 테스트·
분포 탐색용이다. 운영 평가에는 반드시 실제 누적 샘플을 함께 사용할 것.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

from edu_tools.team_ml_model import append_samples, build_feature_vector, hybrid_dl_target
from edu_tools.team_unified_eval import TeamUserIn, _median, run_score_engine

_LOREM = [
    "API",
    "DB",
    "테스트",
    "리뷰",
    "배포",
    "문서",
    "회의",
    "리팩터",
    "버그",
    "기능",
    "프론트",
    "백엔드",
    "Docker",
    "CI",
    "PR",
    "커밋",
    "협업",
    "요구사항",
    "설계",
    "구현",
    "코드",
]


def _rand_self_report(rng: random.Random, max_words: int) -> str:
    n = rng.randint(0, max_words)
    if n <= 0:
        return ""
    return " ".join(rng.choice(_LOREM) for _ in range(n))


def _rand_member(rng: random.Random, name: str) -> TeamUserIn:
    commits = min(900, max(0, int(rng.gammavariate(2.2, 28))))
    prs = min(150, max(0, int(rng.gammavariate(1.3, 5))))
    lines = min(120000, max(0, int(rng.lognormvariate(8.0, 1.45))))
    att = rng.uniform(30.0, 100.0)
    nw = rng.randint(0, 320)
    sr = _rand_self_report(rng, min(nw, 120))
    return TeamUserIn(
        name=name,
        commits=commits,
        prs=prs,
        lines=lines,
        attendance=round(att, 1),
        selfReport=sr,
    )


def iter_synthetic_rows(target_members: int, seed: int = 42):
    """팀 단위로 Score Engine과 동일 규칙으로 (x,y) 행 생성."""
    rng = random.Random(seed)
    team_i = 0
    emitted = 0
    while emitted < target_members:
        room = target_members - emitted
        if room <= 0:
            break
        if room == 1:
            ts = 1
        else:
            ts = rng.randint(2, min(14, room))
        users: list[TeamUserIn] = []
        for j in range(ts):
            users.append(_rand_member(rng, f"syn_t{team_i}_m{j}_{rng.getrandbits(24):x}"))
        scores, _trust = run_score_engine(users)
        commits = [float(u.commits) for u in users]
        prs = [float(u.prs) for u in users]
        lines = [float(u.lines) for u in users]
        atts = [float(u.attendance) for u in users]
        words = [float(len((u.selfReport or "").split())) for u in users]
        m_c = _median(commits)
        m_p = _median(prs)
        m_l = _median(lines)
        m_a = _median(atts) if max(atts) > 1e-9 else 50.0
        m_w = _median(words) if max(words) > 1e-9 else 1.0
        team_n = len(users)
        ts_iso = datetime.now(timezone.utc).isoformat()
        for u, s in zip(users, scores):
            hb, hr, hd = 0.5, 0.5, 0.0
            x = build_feature_vector(
                u.commits,
                u.prs,
                u.lines,
                u.attendance,
                u.selfReport,
                member_rank=int(s.rank),
                team_size=team_n,
                median_commits=m_c,
                median_prs=m_p,
                median_lines=m_l,
                median_attendance=m_a,
                median_words=m_w,
                hist_blend=hb,
                hist_rule=hr,
                hist_density=hd,
            )
            y = hybrid_dl_target(
                float(s.normalizedScore),
                u.commits,
                u.prs,
                u.lines,
                u.attendance,
                u.selfReport,
            )
            yield {
                "x": x,
                "y": y,
                "member": u.name.strip(),
                "created_at": ts_iso,
                "synthetic": True,
                "synthetic_team": team_i,
            }
            emitted += 1
        team_i += 1


def bulk_append_synthetic(target_members: int, seed: int = 42, batch_size: int = 8000) -> int:
    """합성 행을 batch_size 단위로 append_samples. 추가된 총 행 수 반환."""
    buf: list[dict] = []
    total = 0
    for row in iter_synthetic_rows(target_members, seed=seed):
        buf.append(row)
        if len(buf) >= batch_size:
            total += int(append_samples(buf).get("written", 0))
            buf = []
    if buf:
        total += int(append_samples(buf).get("written", 0))
    return total
