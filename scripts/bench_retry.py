"""
재시도 정책에 따라 fallback까지 걸리는 시간이 얼마나 달라지는지 실측한다.

존재하지 않는 모델명으로 호출해 확정된 실패(404)를 만든 뒤,
  - 순진한 정책: APIStatusError를 전부 재시도 -> 1s + 2s + 4s를 헛되게 기다린다
  - 수정한 정책: 4xx는 재시도하지 않는다 -> 즉시 fallback
두 경우를 같은 조건에서 재서 비교한다.

실행: .venv/bin/python scripts/bench_retry.py
"""
import asyncio
import logging
import sys
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from main import RETRY_DELAYS, is_retryable  # noqa: E402

# main을 import하면 logging이 켜진다. 측정 결과만 깔끔하게 남도록 HTTP 로그는 끈다(스크린샷용).
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

BROKEN_MODEL = "model-does-not-exist"
MESSAGES = [{"role": "user", "content": "테스트"}]


def pad(text: str, width: int) -> str:
    """한글은 터미널에서 2칸을 차지하므로 표시 폭 기준으로 채운다."""
    display = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - display)


def retry_everything(error: Exception) -> bool:
    """수정 전 정책. 4xx도 재시도 대상에 들어간다."""
    return isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError))


async def measure(client: AsyncOpenAI, policy, label: str) -> None:
    started = time.perf_counter()
    attempts = 0
    for delay in (0, *RETRY_DELAYS):
        if delay:
            await asyncio.sleep(delay)
        attempts += 1
        try:
            await client.chat.completions.create(model=BROKEN_MODEL, messages=MESSAGES, stream=True)
            break
        except Exception as error:
            if not policy(error):
                break
    elapsed = time.perf_counter() - started
    print(f"  {pad(label, 26)} 시도 {attempts}회   fallback까지 {elapsed:5.2f}초")


async def main() -> None:
    client = AsyncOpenAI()
    print()
    print("존재하지 않는 모델명(404)으로 확정된 실패를 만든 뒤 fallback까지의 시간 측정")
    print("-" * 68)
    await measure(client, retry_everything, "수정 전 (전부 재시도)")
    await measure(client, is_retryable, "수정 후 (4xx 제외)")
    print("-" * 68)
    print("  4xx는 몇 번을 불러도 결과가 같으므로 기다릴 이유가 없다.")
    print()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
