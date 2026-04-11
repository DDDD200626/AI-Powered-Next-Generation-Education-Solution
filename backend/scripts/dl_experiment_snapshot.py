#!/usr/bin/env python3
"""현재 학습 산출물·데이터셋·환경을 JSON으로 스냅샷(실험 비교·재현용).

  cd backend && python scripts/dl_experiment_snapshot.py
  cd backend && python scripts/dl_experiment_snapshot.py --print-only

출력: edu_tools/data/experiment_snapshots/snapshot_<UTC>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

SNAP_DIR = BACKEND_ROOT / "edu_tools" / "data" / "experiment_snapshots"


def main() -> int:
    from edu_tools.experiment_reporting import collect_dl_experiment_context

    ap = argparse.ArgumentParser(description="DL 실험 컨텍스트 스냅샷")
    ap.add_argument("--print-only", action="store_true", help="파일 저장 없이 stdout에만 출력")
    args = ap.parse_args()

    ctx = collect_dl_experiment_context()
    text = json.dumps(ctx, ensure_ascii=False, indent=2)

    if args.print_only:
        print(text)
        return 0

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = SNAP_DIR / f"snapshot_{ts}.json"
    path.write_text(text, encoding="utf-8")

    print("=== DL 실험 스냅샷 ===")
    print(f"저장: {path}")
    m = ctx.get("metrics_summary") or {}
    print(
        f"표본 sample_count={m.get('sample_count')} | "
        f"val_MAE={m.get('validation_mae_mean')} | "
        f"CV_MAE={m.get('cv_mae_mean')} | "
        f"CV−val_gap={m.get('cv_vs_validation_gap')} | "
        f"holdout_MAE={m.get('holdout_time_mae')}"
    )
    print(f"dataset_sha256={ctx.get('dataset_sha256')}")
    print(f"git={ctx.get('git_commit_short')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
