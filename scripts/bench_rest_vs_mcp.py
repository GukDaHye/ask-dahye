"""
같은 파일 하나를 읽을 때 REST와 MCP가 각각 얼마나 걸리는지 잰다.

MCP를 고른 이유는 성능이 아니라 프로토콜을 직접 구현해보는 것이었다.
그 선택의 대가가 실제로 얼마인지 숫자로 남겨둔다.

실행: .venv/bin/python scripts/bench_rest_vs_mcp.py
필요: .env에 GITHUB_TOKEN, GITHUB_REPO
"""
import asyncio
import os
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import mcp_client  # noqa: E402

TARGET_PATH = "search.py"
ROUNDS = 5


def pad(text: str, width: int) -> str:
    display = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - display)


async def rest_read(client: httpx.AsyncClient, owner: str, repo: str) -> float:
    started = time.perf_counter()
    response = await client.get(
        f"https://api.github.com/repos/{owner}/{repo}/contents/{TARGET_PATH}",
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github.raw+json",
        },
    )
    response.raise_for_status()
    return time.perf_counter() - started


async def mcp_read() -> float:
    started = time.perf_counter()
    await mcp_client.read_file(TARGET_PATH)
    return time.perf_counter() - started


async def main() -> None:
    if not mcp_client.is_configured():
        raise SystemExit("GITHUB_TOKEN / GITHUB_REPO가 없습니다. .env를 확인해주세요.")

    owner, _, repo = (mcp_client.repo_slug() or "/").partition("/")

    async with httpx.AsyncClient(timeout=20) as client:
        await rest_read(client, owner, repo)  # 커넥션 수립은 측정에서 제외
        rest = [await rest_read(client, owner, repo) for _ in range(ROUNDS)]

    started = time.perf_counter()
    await mcp_client.read_file(TARGET_PATH)  # 세션 초기화가 포함된 첫 호출
    first_call = time.perf_counter() - started
    mcp = [await mcp_read() for _ in range(ROUNDS)]

    rest_median = statistics.median(rest)
    mcp_median = statistics.median(mcp)

    print()
    print(f"파일 하나 읽기 — {owner}/{repo}/{TARGET_PATH} ({ROUNDS}회 중앙값)")
    print("-" * 58)
    print(f"  {pad('REST GET /contents', 34)} {rest_median * 1000:7.0f}ms")
    print(f"  {pad('MCP tools/call (세션 재사용)', 34)} {mcp_median * 1000:7.0f}ms")
    print(f"  {pad('MCP 첫 호출 (initialize 포함)', 34)} {first_call * 1000:7.0f}ms")
    print("-" * 58)
    print(f"  호출당 배수: {mcp_median / rest_median:.1f}x")
    print("  세션 초기화는 프로세스당 한 번이므로 질문마다 붙지는 않는다.")
    print()

    await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
