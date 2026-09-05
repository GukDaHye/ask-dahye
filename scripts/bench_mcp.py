"""
MCP 근거 수집 시간을 측정한다.

두 가지를 잰다.
  1) 순차 호출 vs 병렬 호출 — 같은 질문을 콜드 상태에서 두 방식으로 재서 비교한다.
  2) 목록 캐시 없음 vs 적용 — 캐싱이 이후 질문에 주는 효과를 본다.

순차 구현은 프로덕션 코드에 남아 있지 않다(mcp_client는 asyncio.gather로 고정돼 있다).
그래서 여기서 mcp_client가 참조하는 asyncio.gather만 순차 실행으로 바꿔 끼운다.
프로덕션 코드를 측정용으로 더럽히지 않으면서 두 방식을 같은 실행에서 재기 위해서다.

실행: .venv/bin/python scripts/bench_mcp.py
필요: .env에 GITHUB_TOKEN, GITHUB_REPO
"""
import asyncio
import contextlib
import logging
import statistics
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
REPEAT = 3


class _SequentialAsyncio:
    """gather만 순차 실행으로 바꾸고 나머지 속성은 실제 asyncio로 넘긴다.

    mcp_client가 모듈 전역에서 asyncio.Lock()을 이미 만들어 뒀으므로(import 시점),
    import 이후에 바꿔 끼우는 것은 안전하다. asyncio.sleep은 그대로 위임된다.
    """

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def gather(self, *awaitables, return_exceptions=False):
        results = []
        for awaitable in awaitables:
            try:
                results.append(await awaitable)
            except BaseException as error:  # noqa: BLE001 - gather의 동작을 그대로 흉내낸다
                if not return_exceptions:
                    raise
                results.append(error)
        return results


@contextlib.contextmanager
def sequential_mode():
    original = mcp_client.asyncio
    mcp_client.asyncio = _SequentialAsyncio()
    try:
        yield
    finally:
        mcp_client.asyncio = original


def pad(text: str, width: int) -> str:
    """한글은 터미널에서 2칸을 차지하므로 표시 폭 기준으로 채운다."""
    display = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - display)


async def measure(question: str, *, sequential: bool = False) -> tuple[float, list[dict]]:
    stack = sequential_mode() if sequential else contextlib.nullcontext()
    with stack:
        started = time.perf_counter()
        chunks = await mcp_client.collect_code_context(question)
        return time.perf_counter() - started, chunks


def summarize(label: str, values: list[float]) -> str:
    return (
        f"  {pad(label, 26)} {statistics.median(values):6.2f}초 (중앙값)"
        f"   범위 {min(values):.2f}~{max(values):.2f}초   n={len(values)}"
    )


async def compare_policies() -> None:
    """순차 vs 병렬. 매번 목록 캐시를 비워 같은 콜드 조건에서 잰다."""
    question = QUESTIONS[0]
    print()
    print(f'1) 순차 호출 vs 병렬 호출 — 콜드 상태, 질문 "{question}"')
    print("-" * 74)

    timings: dict[str, list[float]] = {"순차 (수정 전)": [], "병렬 (수정 후)": []}
    for _ in range(REPEAT):
        for label, is_sequential in (("순차 (수정 전)", True), ("병렬 (수정 후)", False)):
            mcp_client._listing_cache = None
            elapsed, _ = await measure(question, sequential=is_sequential)
            timings[label].append(elapsed)

    for label, values in timings.items():
        print(summarize(label, values))
    print("-" * 74)
    print("  순차는 목록 조회와 파일 읽기를 한 번에 하나씩 보낸다. 왕복 수만큼 시간이 쌓인다.")


async def compare_cache() -> None:
    """목록 캐시 없음 vs 적용. 병렬 구현 기준."""
    print()
    print("2) 목록 캐시 없음 vs 적용 — 병렬 구현")
    print("-" * 74)
    print(f"{pad('질문', 40)} {pad('목록 캐시', 12)} {'소요':>8s}   근거 파일")
    print("-" * 74)

    cold: list[float] = []
    warm: list[float] = []
    for _ in range(REPEAT):
        for index, question in enumerate(QUESTIONS):
            if index == 0:
                mcp_client._listing_cache = None
                state = "없음"
            else:
                state = "적용"
            elapsed, chunks = await measure(question)
            (cold if index == 0 else warm).append(elapsed)
            names = ", ".join(c["section"] for c in chunks) or "(없음)"
            print(f"{pad(question, 40)} {pad(state, 12)} {elapsed:6.2f}초   {names[:38]}")

    print("-" * 74)
    print(summarize("캐시 없음 (첫 질문)", cold))
    print(summarize("캐시 적용 (이후 질문)", warm))


async def main() -> None:
    if not mcp_client.is_configured():
        raise SystemExit("GITHUB_TOKEN / GITHUB_REPO가 없습니다. .env를 확인해주세요.")

    print()
    print(f"대상 리포: {mcp_client.repo_slug()}  ·  각 조건 {REPEAT}회 반복")
    await compare_policies()
    await compare_cache()
    print()
    await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
