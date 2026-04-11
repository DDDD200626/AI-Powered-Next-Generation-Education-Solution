"""팀 학습 산출물(JSONL·선형·PyTorch) 타임스탬프 폴더로 백업."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edu_tools.team_ml_model import DATA_DIR, DATASET_PATH, MODEL_PATH
from edu_tools.team_torch_model import TORCH_META_PATH, TORCH_MODEL_PATH


def _backup_root() -> Path:
    raw = (os.environ.get("TEAM_DATA_BACKUP_ROOT") or "").strip()
    if raw:
        return Path(raw)
    return DATA_DIR / "backups"


def backup_team_training_artifacts(*, reason: str = "manual") -> dict[str, Any]:
    """dataset + team_ml_model.json + team_torch_model.pt/meta 를 한 폴더에 복사."""
    root = _backup_root()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = root / f"{ts}_{reason}".replace(" ", "_")
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in (DATASET_PATH, MODEL_PATH, TORCH_MODEL_PATH, TORCH_META_PATH):
        try:
            if src.is_file():
                shutil.copy2(src, dest / src.name)
                copied.append(str((dest / src.name).resolve()))
        except OSError:
            continue
    ext_cache = DATA_DIR / "team_external_train.cache.jsonl"
    if ext_cache.is_file():
        try:
            shutil.copy2(ext_cache, dest / ext_cache.name)
            copied.append(str((dest / ext_cache.name).resolve()))
        except OSError:
            pass
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "copied_files": [Path(p).name for p in copied],
    }
    (dest / "backup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "backup_dir": str(dest.resolve()),
        "copied": copied,
        "reason": reason,
    }
