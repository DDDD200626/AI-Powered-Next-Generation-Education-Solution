"""Optional external benchmark priors for contribution scoring.

- 단일 **승인된** JSON URL 또는 **로컬 파일**만 사용 (전체 웹 크롤링 없음).
- 기대값(priors) + 선택적 **gap 가중·강도**로 DL 보조 점수를 소폭 보정(상한 있음).
"""

from __future__ import annotations

import json
import math
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


def _local_file_path() -> str:
    return os.environ.get("TEAM_WEB_PRIORS_FILE", "").strip()


def _env_strength() -> float:
    raw = (os.environ.get("TEAM_WEB_PRIORS_STRENGTH", "1") or "1").strip()
    try:
        v = float(raw)
    except ValueError:
        return 1.0
    return max(0.0, min(2.0, v))


def _default_priors() -> dict[str, float]:
    return {
        "commit_expectation": 55.0,
        "pr_expectation": 52.0,
        "lines_expectation": 54.0,
        "attendance_expectation": 70.0,
        "self_report_words_expectation": 45.0,
    }


def _default_adjustment() -> dict[str, Any]:
    """기본: 기존 3항(commit·pr·att)만, 라인·서술 가중 0으로 과거 동작과 동일."""
    return {
        "prior_strength": 1.0,
        "gap_weights": {
            "commit": 5.0,
            "pr": 4.0,
            "lines": 0.0,
            "attendance": 3.0,
            "self_report": 0.0,
        },
        "max_delta": 12.0,
        "source_title": None,
        "methodology_note_ko": None,
        "cohort": None,
        "valid_until": None,
    }


def _clamp_priors(p: dict[str, Any]) -> dict[str, float]:
    out = _default_priors()
    for k in out:
        if k in p:
            try:
                out[k] = max(0.0, min(100.0, float(p[k])))
            except (TypeError, ValueError):
                pass
    return out


def _default_benchmark_inference() -> dict[str, Any]:
    """외부 JSON 기대값만으로 정렬 점수(0–100)를 만들고 NN dl_score와 혼합(학습 불필요)."""
    return {
        "enabled": False,
        "blend_weight": 0.3,
        "note_ko": "",
    }


def _merge_benchmark_inference(base: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    bi = raw.get("benchmark_inference")
    if not isinstance(bi, dict):
        return base
    out = {**base}
    if "enabled" in bi:
        out["enabled"] = str(bi["enabled"]).strip().lower() in ("1", "true", "yes", "on")
    if "blend_weight" in bi:
        try:
            out["blend_weight"] = max(0.0, min(1.0, float(bi["blend_weight"])))
        except (TypeError, ValueError):
            pass
    if bi.get("note_ko") is not None:
        out["note_ko"] = str(bi["note_ko"])[:2000]
    return out


def _merge_adjustment(base: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    adj = {**base}
    if "prior_strength" in raw:
        try:
            adj["prior_strength"] = max(0.0, min(1.5, float(raw["prior_strength"])))
        except (TypeError, ValueError):
            pass
    if "max_delta" in raw:
        try:
            adj["max_delta"] = max(1.0, min(25.0, float(raw["max_delta"])))
        except (TypeError, ValueError):
            pass
    gw = raw.get("gap_weights")
    if isinstance(gw, dict):
        cur = dict(adj["gap_weights"])
        for key in cur:
            if key in gw:
                try:
                    cur[key] = max(0.0, min(12.0, float(gw[key])))
                except (TypeError, ValueError):
                    pass
        adj["gap_weights"] = cur
    for meta_key in ("source_title", "methodology_note_ko", "cohort", "valid_until"):
        if meta_key in raw and raw[meta_key] is not None:
            adj[meta_key] = str(raw[meta_key])[:2000]
    return adj


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


def _read_json_file(path_str: str) -> dict[str, Any] | None:
    p = Path(path_str)
    if not p.is_file():
        p2 = DATA_DIR / path_str
        if p2.is_file():
            p = p2
        else:
            return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def get_web_priors() -> tuple[dict[str, float], dict[str, Any]]:
    """Return (priors, metadata) including ``prior_adjustment`` for DL delta transparency."""
    priors = _default_priors()
    adjustment = _default_adjustment()
    benchmark_inference = _default_benchmark_inference()
    meta: dict[str, Any] = {
        "enabled": False,
        "source": "local-default",
        "prior_adjustment": adjustment,
        "benchmark_inference": benchmark_inference,
    }

    # 1) 로컬 파일 (교육기관·레포에 커밋 가능한 벤치마크)
    lf = _local_file_path()
    if lf:
        local_doc = _read_json_file(lf)
        if local_doc:
            if isinstance(local_doc.get("priors"), dict):
                priors = _clamp_priors(local_doc["priors"])
            adjustment = _merge_adjustment(adjustment, local_doc)
            benchmark_inference = _merge_benchmark_inference(benchmark_inference, local_doc)
            meta["source"] = "local-file"
            meta["local_path"] = lf
            meta["enabled"] = True

    # 2) 원격 URL (단일 allowlist, 캐시 TTL)
    if _enabled():
        url = _source_url()
        if not url and not lf:
            meta["enabled"] = bool(meta.get("enabled"))
            meta.setdefault("reason", "no_url")
        elif url:
            now = int(time.time())
            cached = _load_cache()
            if cached and int(cached.get("fetched_at", 0)) + CACHE_TTL_SEC > now:
                p = cached.get("priors")
                doc = cached.get("document")
                if isinstance(p, dict):
                    priors = _clamp_priors(p)
                if isinstance(doc, dict):
                    adjustment = _merge_adjustment(adjustment, doc)
                    benchmark_inference = _merge_benchmark_inference(benchmark_inference, doc)
                meta.update(
                    {
                        "enabled": True,
                        "source": "cache",
                        "url": url,
                        "fetched_at": cached.get("fetched_at"),
                    }
                )
            else:
                remote = _fetch_remote_json(url)
                if isinstance(remote, dict):
                    if isinstance(remote.get("priors"), dict):
                        priors = _clamp_priors(remote["priors"])
                    adjustment = _merge_adjustment(adjustment, remote)
                    benchmark_inference = _merge_benchmark_inference(benchmark_inference, remote)
                    meta.update(
                        {
                            "enabled": True,
                            "source": "web",
                            "url": url,
                            "fetched_at": now,
                        }
                    )
                    _save_cache(
                        {
                            "url": url,
                            "fetched_at": now,
                            "priors": priors,
                            "document": {
                                k: v
                                for k, v in remote.items()
                                if k != "priors"
                            },
                        }
                    )
                elif cached and isinstance(cached.get("priors"), dict):
                    priors = _clamp_priors(cached["priors"])
                    doc = cached.get("document")
                    if isinstance(doc, dict):
                        adjustment = _merge_adjustment(adjustment, doc)
                        benchmark_inference = _merge_benchmark_inference(benchmark_inference, doc)
                    meta.update(
                        {
                            "enabled": True,
                            "source": "stale-cache",
                            "url": url,
                            "reason": "fetch_failed",
                        }
                    )
                else:
                    meta.update({"enabled": True, "source": "local-default", "url": url, "reason": "fetch_failed"})

    # 3) 전역 강도 (환경): 외부 자료 영향을 줄이거나 키울 때
    env_s = _env_strength()
    adjustment["prior_strength"] = max(0.0, min(1.5, float(adjustment["prior_strength"]) * env_s))
    adjustment["env_strength_multiplier"] = env_s
    meta["prior_adjustment"] = adjustment

    # 외부 정렬 점수: 환경으로 켜거나 끄기 (JSON보다 우선)
    benv = (os.environ.get("TEAM_BENCHMARK_INFERENCE_ENABLED", "") or "").strip().lower()
    if benv in ("1", "true", "yes", "on"):
        benchmark_inference["enabled"] = True
    elif benv in ("0", "false", "no", "off"):
        benchmark_inference["enabled"] = False
    bw_env = (os.environ.get("TEAM_BENCHMARK_BLEND_WEIGHT", "") or "").strip()
    if bw_env:
        try:
            benchmark_inference["blend_weight"] = max(0.0, min(1.0, float(bw_env)))
        except ValueError:
            pass
    benchmark_inference["blend_weight"] = max(0.0, min(1.0, float(benchmark_inference.get("blend_weight", 0.3))))
    meta["benchmark_inference"] = benchmark_inference

    return priors, meta


def compute_prior_score_delta(
    x: list[float],
    priors: dict[str, float],
    adjustment: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """피처 앞쪽 5개와 기대값 차이로 소폭 보정량을 계산. (투명성용 gap 스칼라 반환)"""
    cg = min(1.0, max(0.0, x[0] / math.log1p(30))) - (priors["commit_expectation"] / 100.0)
    pg = min(1.0, max(0.0, x[1] / math.log1p(12))) - (priors["pr_expectation"] / 100.0)
    lg = min(1.0, max(0.0, x[2] / math.log1p(12000))) - (priors["lines_expectation"] / 100.0)
    ag = x[3] - (priors["attendance_expectation"] / 100.0)
    sg = max(-1.0, min(1.0, x[4] - (priors["self_report_words_expectation"] / 100.0)))

    gw = adjustment.get("gap_weights") or {}
    wc = float(gw.get("commit", 5.0))
    wp = float(gw.get("pr", 4.0))
    wl = float(gw.get("lines", 0.0))
    wa = float(gw.get("attendance", 3.0))
    ws = float(gw.get("self_report", 0.0))

    raw = wc * cg + wp * pg + wl * lg + wa * ag + ws * sg
    ps = float(adjustment.get("prior_strength", 1.0))
    md = float(adjustment.get("max_delta", 12.0))
    delta = max(-md, min(md, raw * ps))
    gaps = {
        "commit_gap": round(cg, 4),
        "pr_gap": round(pg, 4),
        "lines_gap": round(lg, 4),
        "attendance_gap": round(ag, 4),
        "self_report_gap": round(sg, 4),
        "raw_weighted_sum": round(raw, 4),
        "prior_strength_applied": round(ps, 4),
        "delta_applied": round(delta, 4),
    }
    return delta, gaps


def compute_external_benchmark_score(
    x: list[float],
    priors: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """외부 JSON의 기대값(priors) 대비 입력 정렬도를 0–100으로 환산(추가 학습 없음).

    평균 절대 갭이 작을수록 외부 벤치마크 분포와 잘 맞는 것으로 본다.
    """
    cg = min(1.0, max(0.0, x[0] / math.log1p(30))) - (priors["commit_expectation"] / 100.0)
    pg = min(1.0, max(0.0, x[1] / math.log1p(12))) - (priors["pr_expectation"] / 100.0)
    lg = min(1.0, max(0.0, x[2] / math.log1p(12000))) - (priors["lines_expectation"] / 100.0)
    ag = x[3] - (priors["attendance_expectation"] / 100.0)
    sg = max(-1.0, min(1.0, x[4] - (priors["self_report_words_expectation"] / 100.0)))
    m = (abs(cg) + abs(pg) + abs(lg) + abs(ag) + abs(sg)) / 5.0
    score = max(0.0, min(100.0, 100.0 * (1.0 - min(1.0, m * 1.12))))
    detail = {
        "mean_abs_gap": round(m, 4),
        "commit_gap": round(cg, 4),
        "pr_gap": round(pg, 4),
        "lines_gap": round(lg, 4),
        "attendance_gap": round(ag, 4),
        "self_report_gap": round(sg, 4),
    }
    return score, detail
