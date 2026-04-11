"""pytest 부트스트랩 — 저장소 기본 PyTorch 프로필이 xxl이라도 CI·로컬 테스트는 경량으로 고정."""

from __future__ import annotations

import os


def pytest_configure(config) -> None:  # noqa: ARG001
    # 명시적으로 TEAM_TORCH_MODEL_SIZE를 준 경우만 존중(미설정일 때만 standard 주입)
    if not (os.environ.get("TEAM_TORCH_MODEL_SIZE") or "").strip():
        os.environ["TEAM_TORCH_MODEL_SIZE"] = "standard"
    # 앱 lifespan 백그라운드 재학습 비활성(테스트·CI에서 CPU·중복 학습 방지)
    os.environ["EDUSIGNAL_AUTO_RETRAIN_BACKGROUND"] = "0"
    # Python 3.14+ 에서 google.genai 내부 typing 경고(런타임 무관)
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*_UnionGenericAlias.*:DeprecationWarning:google.genai.types",
    )
