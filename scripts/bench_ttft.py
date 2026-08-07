"""
모델·설정별 TTFT(첫 토큰까지 걸린 시간)를 측정한다.
추론 모델은 답을 내기 전에 내부 추론을 먼저 돌려서 첫 토큰이 늦게 나온다.

실행: .venv/bin/python scripts/bench_ttft.py
"""
import sys
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

MESSAGES = [
    {"role": "system", "content": "한국어로 3문장 이내로 답하라."},
    {"role": "user", "content": "쿠버네티스에서 OOMKilled가 왜 발생하는지 설명해줘."},
]

CONFIGS = [
    ("gpt-5.4-mini 기본", {"model": "gpt-5.4-mini", "max_completion_tokens": 400}),
    ("gpt-5.4-mini reasoning=minimal", {"model": "gpt-5.4-mini", "max_completion_tokens": 400, "reasoning_effort": "minimal"}),
    ("gpt-5.4-mini reasoning=low", {"model": "gpt-5.4-mini", "max_completion_tokens": 400, "reasoning_effort": "low"}),
    ("gpt-4.1-mini (비추론)", {"model": "gpt-4.1-mini", "max_tokens": 400}),
]


def pad(text: str, width: int) -> str:
    """한글은 터미널에서 2칸을 차지하므로 표시 폭 기준으로 채운다(스크린샷 정렬용)."""
    display = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - display)


client = OpenAI()

print()
print(f"{pad('설정', 38)} {'TTFT':>9s} {'total':>9s}   {'출력':>4s}")
print("-" * 70)

for label, kwargs in CONFIGS:
    started = time.perf_counter()
    first_token_at = None
    chars = 0
    try:
        stream = client.chat.completions.create(messages=MESSAGES, stream=True, **kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if first_token_at is None:
                    first_token_at = time.perf_counter() - started
                chars += len(delta)
        total = time.perf_counter() - started
        print(f"{pad(label, 38)} {first_token_at * 1000:7.0f}ms {total * 1000:7.0f}ms  {chars:4d}자")
    except Exception as error:
        print(f"{pad(label, 38)} {pad('미지원', 9)}   {pad('—', 8)}  {type(error).__name__}")

print()
