"""
MCP 근거 수집 시간을 측정한다. 파일 목록 캐시가 있을 때와 없을 때를 비교한다.

MCP 왕복은 한 번에 1.5~2초쯤 걸린다. 목록 조회와 파일 읽기를 순차로 돌리면
LLM을 부르기도 전에 시간이 다 가므로, 같은 단계는 병렬로 보내고 목록은 캐싱한다.

실행: .venv/bin/python scripts/bench_mcp.py
필요: .env에 GITHUB_TOKEN, GITHUB_REPO
"""
import asyncio
import logging
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import mcp_client  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

QUESTIONS = [
    "이 챗봇은 어떻게 만들어졌나요?",
    "코사인 유사도 검색 코드 어떻게 구현했어요?",
    "이 프로젝트 파일 구조 알려줘",
]


def pad(text: str, width: int) -> str:
    """한글은 터미널에서 2칸을 차지하므로 표시 폭 기준으로 채운다."""
    display = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - display)


async def measure(question: str) -> tuple[float, list[dict]]:
    started = time.perf_counter()
    chunks = await mcp_client.collect_code_context(question)
    return time.perf_counter() - started, chunks


async def main() -> None:
    if not mcp_client.is_configured():
        raise SystemExit("GITHUB_TOKEN / GITHUB_REPO가 없습니다. .env를 확인해주세요.")

    print()
    print(f"대상 리포: {mcp_client.repo_slug()}")
    print("-" * 74)
    print(f"{pad('질문', 40)} {pad('목록 캐시', 12)} {'소요':>8s}   근거 파일")
    print("-" * 74)

    for index, question in enumerate(QUESTIONS):
        if index == 0:
            # 첫 측정만 캐시를 비워 콜드 상태를 만든다.
            mcp_client._listing_cache = None
            cache_state = "없음"
        else:
            cache_state = "적용"

        elapsed, chunks = await measure(question)
        names = ", ".join(c["section"] for c in chunks) or "(없음)"
        print(f"{pad(question, 40)} {pad(cache_state, 12)} {elapsed:6.2f}초   {names[:40]}")

    print("-" * 74)
    print("  첫 질문은 목록 조회 왕복 두 번이 더 붙는다. 이후는 파일 읽기만 남는다.")
    print()
    await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
