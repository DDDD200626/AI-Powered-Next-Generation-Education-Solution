"""Team contribution data store (SQLite).

Keeps project-local, reproducible training/evaluation data.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).with_name("data")
DB_PATH = DATA_DIR / "team_contrib.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    c = _conn()
    try:
        cur = c.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS teams (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              request_id TEXT UNIQUE,
              project_name TEXT,
              created_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS members (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              team_id INTEGER,
              member_name TEXT,
              role TEXT,
              FOREIGN KEY(team_id) REFERENCES teams(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS contribution_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              team_id INTEGER,
              member_name TEXT,
              commits INTEGER,
              prs INTEGER,
              lines_changed INTEGER,
              attendance REAL,
              self_report_words INTEGER,
              rule_score REAL,
              dl_score REAL,
              blended_score REAL,
              trust_score REAL,
              anomaly_count INTEGER,
              payload_json TEXT,
              created_at TEXT,
              FOREIGN KEY(team_id) REFERENCES teams(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              request_id TEXT,
              model_name TEXT,
              model_version TEXT,
              sample_count INTEGER,
              auto_retrain INTEGER,
              created_at TEXT
            )
            """
        )
        c.commit()
    finally:
        c.close()


def record_team_report(
    *,
    request_id: str,
    project_name: str,
    users: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    trust_scores: dict[str, float],
    dl_model_info: dict[str, Any] | None,
) -> None:
    init_db()
    c = _conn()
    try:
        cur = c.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO teams(request_id, project_name, created_at) VALUES (?, ?, ?)",
            (request_id, project_name, _utc_now()),
        )
        cur.execute("SELECT id FROM teams WHERE request_id = ?", (request_id,))
        row = cur.fetchone()
        if not row:
            c.commit()
            return
        team_id = int(row["id"])

        for u in users:
            cur.execute(
                "INSERT INTO members(team_id, member_name, role) VALUES (?, ?, ?)",
                (team_id, str(u.get("name", "")), str(u.get("role", ""))),
            )

        anom_by_name: dict[str, int] = {}
        for a in anomalies:
            nm = str(a.get("member_name", "")).strip()
            flags = a.get("flags") or []
            anom_by_name[nm] = len(flags if isinstance(flags, list) else [])

        user_by_name = {str(u.get("name", "")).strip(): u for u in users}
        for s in scores:
            name = str(s.get("member_name", "")).strip()
            u = user_by_name.get(name, {})
            self_words = len([x for x in str(u.get("selfReport", "")).split() if x.strip()])
            payload = {"user": u, "score": s}
            cur.execute(
                """
                INSERT INTO contribution_events(
                  team_id, member_name, commits, prs, lines_changed, attendance, self_report_words,
                  rule_score, dl_score, blended_score, trust_score, anomaly_count, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    team_id,
                    name,
                    int(u.get("commits", 0) or 0),
                    int(u.get("prs", 0) or 0),
                    int(u.get("lines", 0) or 0),
                    float(u.get("attendance", 0.0) or 0.0),
                    self_words,
                    float(s.get("normalizedScore", 0.0) or 0.0),
                    float(s.get("dl_score", 0.0) or 0.0) if s.get("dl_score") is not None else None,
                    float(s.get("blendedScore", 0.0) or 0.0) if s.get("blendedScore") is not None else None,
                    float(trust_scores.get(name, 0.0)),
                    int(anom_by_name.get(name, 0)),
                    json.dumps(payload, ensure_ascii=False),
                    _utc_now(),
                ),
            )

        if dl_model_info:
            cur.execute(
                """
                INSERT INTO model_runs(request_id, model_name, model_version, sample_count, auto_retrain, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    str(dl_model_info.get("model_name", "")),
                    str(dl_model_info.get("model_version", "")),
                    int(dl_model_info.get("sample_count", 0) or 0),
                    1 if dl_model_info.get("auto_retrain") else 0,
                    _utc_now(),
                ),
            )
        c.commit()
    finally:
        c.close()


def db_profile() -> dict[str, int]:
    init_db()
    c = _conn()
    try:
        cur = c.cursor()
        out: dict[str, int] = {}
        for table in ("teams", "members", "contribution_events", "model_runs"):
            cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
            out[table] = int(cur.fetchone()["n"])
        return out
    finally:
        c.close()


def contribution_trends(days: int = 30, member_name: str | None = None) -> dict[str, Any]:
    """날짜별 기여도 추세(룰/ML/혼합) 집계."""
    init_db()
    c = _conn()
    try:
        cur = c.cursor()
        days = max(1, min(365, int(days)))
        params: list[Any] = [f"-{days} day"]
        where = "WHERE datetime(created_at) >= datetime('now', ?)"
        if member_name and member_name.strip():
            where += " AND member_name = ?"
            params.append(member_name.strip())

        cur.execute(
            f"""
            SELECT
              date(created_at) AS d,
              member_name,
              ROUND(AVG(rule_score), 2) AS rule_avg,
              ROUND(AVG(dl_score), 2) AS dl_avg,
              ROUND(AVG(blended_score), 2) AS blended_avg,
              COUNT(*) AS samples
            FROM contribution_events
            {where}
            GROUP BY date(created_at), member_name
            ORDER BY d ASC, member_name ASC
            """,
            params,
        )
        rows = cur.fetchall()

        series: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            name = str(r["member_name"])
            series.setdefault(name, []).append(
                {
                    "date": str(r["d"]),
                    "rule_score": float(r["rule_avg"] or 0.0),
                    "dl_score": float(r["dl_avg"] or 0.0),
                    "blended_score": float(r["blended_avg"] or 0.0),
                    "samples": int(r["samples"] or 0),
                }
            )

        return {
            "window_days": days,
            "member_filter": member_name.strip() if member_name else None,
            "members": sorted(series.keys()),
            "series": series,
        }
    finally:
        c.close()

