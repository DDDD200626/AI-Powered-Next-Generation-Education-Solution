"""Optional web benchmark priors for contribution scoring.

This module does NOT crawl the whole internet.
It only fetches a single allowlisted JSON URL (if configured) and caches it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

DATA_DIR = Path(__file__).with_name("data")
CACHE_PATH = DATA_DIR / "web_priors_cache.json"
CACHE_TTL_SEC = 24 * 60 * 60


def _enabled() -> bool:
    return os.environ.get("TEAM_WEB_PRIORS_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")


def _source_url() -> str:
    return os.environ.get("TEAM_WEB_PRIORS_URL", "").strip()


def _default_priors() -> dict[str, float]:
    # fallback neutral priors (0-100 scale expectations for contribution signals)
    return {
        "commit_expectation": 55.0,
        "pr_expectation": 52.0,
        "lines_expectation": 54.0,
        "attendance_expectation": 70.0,
        "self_report_words_expectation": 45.0,
    }


def _load_cache() -> dict[str, Any] | None:
    if not CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return raw
    except Exception:
        return None


def _save_cache(payload: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_remote_json(url: str) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                return data
            return None
    except Exception:
        return None


def get_web_priors() -> tuple[dict[str, float], dict[str, Any]]:
    """Return (priors, metadata)."""
    priors = _default_priors()
    meta: dict[str, Any] = {"enabled": False, "source": "local-default"}

    if not _enabled():
        return priors, meta

    url = _source_url()
    if not url:
        return priors, {"enabled": True, "source": "local-default", "reason": "no_url"}

    now = int(time.time())
    cached = _load_cache()
    if cached and int(cached.get("fetched_at", 0)) + CACHE_TTL_SEC > now:
        p = cached.get("priors")
        if isinstance(p, dict):
            priors.update({k: float(v) for k, v in p.items() if k in priors})
            return priors, {"enabled": True, "source": "cache", "url": url, "fetched_at": cached.get("fetched_at")}

    remote = _fetch_remote_json(url)
    if isinstance(remote, dict):
        p = remote.get("priors")
        if isinstance(p, dict):
            priors.update({k: float(v) for k, v in p.items() if k in priors})
            _save_cache({"url": url, "fetched_at": now, "priors": priors})
            return priors, {"enabled": True, "source": "web", "url": url, "fetched_at": now}

    if cached and isinstance(cached.get("priors"), dict):
        priors.update({k: float(v) for k, v in cached["priors"].items() if k in priors})
        return priors, {"enabled": True, "source": "stale-cache", "url": url}

    return priors, {"enabled": True, "source": "local-default", "url": url, "reason": "fetch_failed"}

