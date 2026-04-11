#!/usr/bin/env python3
"""고정 시나리오(default/uneven/small)로 POST /api/team/report 를 연속 호출해
입력 형태별 점수 분포·순위 민감도를 한 JSON에 모읍니다(서버 불필요·TestClient).

  cd backend && python scripts/dl_smoke_benchmark.py

출력: edu_tools/data/experiment_snapshots/smoke_<UTC>.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

SNAP_DIR = BACKEND_ROOT / "edu_tools" / "data" / "experiment_snapshots"


def _load_demo_scenarios() -> dict:
    demo_path = Path(__file__).resolve().parent / "demo_team_report.py"
    spec = importlib.util.spec_from_file_location("demo_team_report", demo_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("demo_team_report.py 로드 실패")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.SCENARIOS)


def _post(payload: dict) -> dict:
    from fastapi.testclient import TestClient

    from learning_analysis.main import app

    with TestClient(app) as client:
        r = client.post("/api/team/report", json=payload)
        r.raise_for_status()
        return r.json()


def _summarize_scenario(data: dict) -> dict[str, object]:
    scores = data.get("scores") or []
    blended = [float(s.get("blendedScore") or 0) for s in scores if s.get("blendedScore") is not None]
    norms = [float(s.get("normalizedScore") or 0) for s in scores]
    names = [str(s.get("member_name", "?")) for s in scores]
    order = sorted(range(len(names)), key=lambda i: -blended[i]) if blended else []
    rank_by_name = {names[i]: scores[i].get("rank") for i in range(len(names))}
    dl_info = data.get("dl_model_info") or {}
    q = dl_info.get("quality") if isinstance(dl_info, dict) else {}
    return {
        "n_members": len(scores),
        "member_names": names,
        "blended_min": min(blended) if blended else None,
        "blended_max": max(blended) if blended else None,
        "blended_spread": (max(blended) - min(blended)) if len(blended) > 1 else None,
        "normalized_spread": (max(norms) - min(norms)) if len(norms) > 1 else None,
        "rank_by_member": rank_by_name,
        "top_member_by_blended": names[order[0]] if order and names else None,
        "dl_enabled": dl_info.get("enabled") if isinstance(dl_info, dict) else None,
        "validation_mae_mean": q.get("validation_mae_mean") if isinstance(q, dict) else None,
        "cv_mae_mean": q.get("cv_mae_mean") if isinstance(q, dict) else None,
        "cv_vs_validation_gap": q.get("cv_vs_validation_gap") if isinstance(q, dict) else None,
    }


def main() -> int:
    scenarios = _load_demo_scenarios()
    out: dict[str, object] = {
        "schema": "dl_smoke_benchmark_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note_ko": (
            "동일한 저장된 PyTorch 가중치로 시나리오만 바꿔 호출합니다. "
            "일반화·안정성은 CV/holdout 스냅샷과 함께 보는 것이 좋습니다."
        ),
        "scenarios": {},
    }

    for name, payload in scenarios.items():
        data = _post(payload)
        block = _summarize_scenario(data)
        block["project_name"] = payload.get("project_name")
        out["scenarios"][name] = block

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = SNAP_DIR / f"smoke_{ts}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== DL 스모크 벤치마크 (시나리오별 요약) ===")
    print(f"저장: {path}\n")
    for name, block in (out["scenarios"] or {}).items():
        b = block if isinstance(block, dict) else {}
        print(
            f"  [{name}] n={b.get('n_members')} spread_blended={b.get('blended_spread')} "
            f"top={b.get('top_member_by_blended')} val_MAE={b.get('validation_mae_mean')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
