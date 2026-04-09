"""
합성 팀 기여 샘플을 대량으로 team_ml_dataset.jsonl 에 추가합니다.

사용 (저장소 루트, backend 가 PYTHONPATH 에 있어야 함):
  python scripts/bulk_synthetic_team_data.py 100000
  set TEAM_TRAIN_RESERVOIR_MAX=131072 && python -c "from edu_tools.team_torch_model import train_torch_if_needed; print(train_torch_if_needed())"

기본은 reservoir 미사용(0). 수백만 줄일 때만 TEAM_TRAIN_RESERVOIR_MAX 를 설정하세요.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from edu_tools.team_synthetic_bulk import bulk_append_synthetic  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="합성 팀 ML 샘플 대량 추가")
    p.add_argument("members", type=int, help="추가할 멤버(행) 수 목표")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=8000)
    args = p.parse_args()
    if args.members < 1:
        p.error("members >= 1")
    n = bulk_append_synthetic(args.members, seed=args.seed, batch_size=args.batch)
    print(f"append_samples rows: {n}")


if __name__ == "__main__":
    main()
