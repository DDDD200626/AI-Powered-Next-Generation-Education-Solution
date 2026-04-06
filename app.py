"""
한국 주식 방향 예측 데모 — CLI (Streamlit 미사용).

예시:
  python app.py --ticker 005930.KS --period 5y
  python app.py --mode krx --krx-limit 10 --period 1y
"""

from __future__ import annotations

import argparse

import pandas as pd

from data_loader import load_data
from features import add_features
from krx_tickers import get_krx_yahoo_tickers
from model import predict_next_day, train_model


def run_cli(
    *,
    ticker: str,
    period: str,
    mode: str,
    krx_limit: int,
) -> None:
    if mode == "single":
        tickers = [ticker]
    else:
        tickers = get_krx_yahoo_tickers(limit=krx_limit, verify_with_yfinance=False)

    results: list[tuple[str, object, list[str], pd.DataFrame]] = []
    for t in tickers:
        try:
            df_raw = load_data(ticker=t, period=period)
            if df_raw.empty:
                continue
            df_feat = add_features(df_raw)
            model, feature_cols = train_model(df_feat)
            results.append((t, model, feature_cols, df_feat))
        except Exception:
            continue

    if not results:
        print("예측에 사용할 수 있는 종목이 없습니다.")
        return

    print(f"모델 학습 완료 (종목 수: {len(results)})\n")
    print("=== 내일 방향 예측 ===")
    for t, model, feature_cols, df_feat in results:
        try:
            print(f"\n--- {t} ---")
            predict_next_day(model, feature_cols, df_feat, ticker=t)
        except Exception:
            print(f"{t}: 예측 실패")

    _, _, _, df_feat = results[-1]
    print("\n=== 최근 가격·이동평균 (마지막 처리 종목 기준) ===")
    price_cols = [c for c in ["Close", "ma5", "ma20"] if c in df_feat.columns]
    if price_cols:
        print(df_feat[price_cols].tail(12).to_string())
    if "rsi14" in df_feat.columns:
        print("\n=== RSI(14) 최근 ===")
        print(df_feat[["rsi14"]].tail(12).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description="한국 주식 방향 예측 CLI")
    p.add_argument("--ticker", default="005930.KS", help="야후 티커 형식")
    p.add_argument("--period", default="5y", choices=["1y", "3y", "5y"])
    p.add_argument("--mode", default="single", choices=["single", "krx"])
    p.add_argument("--krx-limit", type=int, default=50, help="KRX 모드일 때 종목 수 상한")
    args = p.parse_args()
    run_cli(
        ticker=args.ticker.strip(),
        period=args.period,
        mode=args.mode,
        krx_limit=max(1, args.krx_limit),
    )


if __name__ == "__main__":
    main()
