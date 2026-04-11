"""EduSignal 전용: 새 팀 평가 샘이 쌓일 때 백그라운드에서 PyTorch·경량 모델 재학습을 주기적으로 시도."""

from __future__ import annotations

import asyncio
import logging
import os
import threading

_log = logging.getLogger(__name__)

_stop: asyncio.Event | None = None
_task: asyncio.Task[None] | None = None
_retrain_lock = threading.Lock()


def background_autoretrain_enabled() -> bool:
    v = (os.environ.get("EDUSIGNAL_AUTO_RETRAIN_BACKGROUND", "1") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def autoretrain_interval_sec() -> int:
    raw = (os.environ.get("EDUSIGNAL_AUTO_RETRAIN_SEC") or "900").strip()
    try:
        return max(120, int(raw))
    except ValueError:
        return 900


def _tick_sync() -> None:
    if not _retrain_lock.acquire(blocking=False):
        return
    try:
        from edu_tools.team_ml_model import train_if_needed
        from edu_tools.team_torch_model import torch_available, train_torch_if_needed

        if torch_available():
            train_torch_if_needed()
        train_if_needed()
    except Exception as ex:
        _log.debug("edusignal autoretrain tick: %s", ex)
    finally:
        _retrain_lock.release()


async def _worker() -> None:
    while _stop is not None and not _stop.is_set():
        try:
            await asyncio.wait_for(_stop.wait(), timeout=float(autoretrain_interval_sec()))
            break
        except TimeoutError:
            pass
        if _stop is None or _stop.is_set():
            break
        if not background_autoretrain_enabled():
            continue
        await asyncio.to_thread(_tick_sync)


async def start_edusignal_background_autoretrain() -> None:
    global _stop, _task
    if not background_autoretrain_enabled():
        _log.info("edusignal background autoretrain disabled (EDUSIGNAL_AUTO_RETRAIN_BACKGROUND)")
        return
    if _task is not None and not _task.done():
        return
    _stop = asyncio.Event()
    _task = asyncio.create_task(_worker(), name="edusignal_team_autoretrain")
    _log.info(
        "edusignal background autoretrain started (every %ss)",
        autoretrain_interval_sec(),
    )


async def stop_edusignal_background_autoretrain() -> None:
    global _stop, _task
    if _stop is not None:
        _stop.set()
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
    _stop = None
    _task = None
