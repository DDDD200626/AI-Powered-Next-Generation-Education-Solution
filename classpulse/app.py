"""
ClassPulse 백엔드만 기동할 때 사용합니다.

백엔드: python -m classpulse.app  → http://127.0.0.1:8000  (API, /docs)
프론트: frontend/ 에서 npm run dev → http://127.0.0.1:5173  (Vite가 /api 프록시)
"""

from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run("classpulse.api_app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
