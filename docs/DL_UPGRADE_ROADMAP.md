# DL Upgrade Roadmap (Ceiling Plan)

This plan defines the practical upper bound for the current project setup.

## Milestone 1 - Reproducibility and Stability
- Deterministic training seed flow across CV/bootstrap/training.
- Schema-stable API outputs for CI and production parity.
- Baseline regression tests for model metadata and API contracts.

## Milestone 2 - Reliability-Aware Scoring
- Dynamic blending between rule score and DL score based on confidence/uncertainty.
- Feature drift signal from serving batch vs training feature statistics.
- Degradation-safe fallback path when model artifacts are unavailable.

## Milestone 3 - Generalization and Robustness
- Rolling time split and group-aware validation strategy tuning.
- Better uncertainty calibration and OOD behavior controls.
- Drift-triggered retrain policy hooks.

## Milestone 4 - Operational Excellence
- Training cache and incremental retraining strategies.
- Model promotion gates based on holdout metrics.
- Structured model card fields for contest and audit evidence.

## Milestone 5 - Explainability and Governance
- Stronger factor attribution consistency checks.
- Score-diff narrative between model versions.
- Fairness and human-review safeguards in evaluator UX.
