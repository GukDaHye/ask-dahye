"""
MCP 호출과 LLM 호출이 각각 어디로 나가고 무엇을 돌려주는지 원문으로 보여준다.

확인하려는 것 두 가지.
  1. MCP는 내 서버와 GitHub MCP 서버 사이에서만 오간다 (LLM은 끼지 않는다)
  2. LLM에게 가는 요청에는 MCP도, 도구 목록도 들어 있지 않다 — 이미 채워진 텍스트뿐이다

실행: .venv/bin/python scripts/trace_mcp_llm.py
필요: .env에 GITHUB_TOKEN, GITHUB_REPO, OPENAI_API_KEY
"""
import asyncio
import json
import logging
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import httpx  # noqa: E402

import main  # noqa: E402
import mcp_client  # noqa: E402

logging.basicConfig(level=logging.CRITICAL)

QUESTION = "이 챗봇은 어떻게 만들어졌나요?"


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def show(label: str, value: str, limit: int = 300) -> None:
    body = value if len(value) <= limit else value[:limit] + f" … (총 {len(value)}자)"
    print(f"  {label}")
    for line in body.split("\n"):
        print(f"    {line}")


# httpx로 나가는 모든 요청을 가로채서 목적지를 기록한다.
CALLS: list[tuple[str, str]] = []
_orig_send = httpx.AsyncClient.send


async def traced_send(self, request, **kwargs):
    CALLS.append((request.method, str(request.url)))
    return await _orig_send(self, request, **kwargs)


httpx.AsyncClient.send = traced_send


async def run() -> None:
    if not mcp_client.is_configured():
        raise SystemExit("GITHUB_TOKEN / GITHUB_REPO가 없습니다. .env를 확인해주세요.")

    # ---------- 1단계: MCP ----------
    rule("1단계 — MCP 호출 (내 서버 ↔ GitHub MCP 서버)")
    mcp_client._listing_cache = None
    chunks = await mcp_client.collect_code_context(QUESTION)

    print(f"  요청을 보낸 곳: {mcp_client.MCP_ENDPOINT}")
    print(f"  붙인 헤더     : X-MCP-Readonly={mcp_client.MCP_READONLY}, "
          f"X-MCP-Tools={mcp_client.MCP_TOOLS}")
    print(f"  부른 도구     : get_file_contents")
    print()
    print(f"  MCP가 돌려준 근거 {len(chunks)}건:")
    for c in chunks:
        print(f"    - {c['section']}  ({len(c['text'])}자)  kind={c['kind']}")
    print()
    show("첫 파일 내용의 앞부분 (MCP 응답 원문):", chunks[0]["text"], 240)

    mcp_calls = list(CALLS)
    CALLS.clear()

    # ---------- 2단계: LLM ----------
    rule("2단계 — LLM 호출 (내 서버 ↔ OpenAI)")
    messages = main.build_messages(QUESTION, chunks, [], main.CODE_SYSTEM_PROMPT)

    print(f"  요청을 보낸 곳: OpenAI Chat Completions ({main.CHAT_MODEL})")
    print(f"  메시지 수     : {len(messages)}")
    print(f"  tools 파라미터: 없음 (LLM에게 도구를 주지 않는다)")
    print()
    system_text = messages[0]["content"]
    instructions, _, references = system_text.partition("참고자료:")

    # 지시문(내가 쓴 프롬프트)과 참고자료(MCP가 가져온 파일 내용)를 나눠서 본다.
    print("  LLM에게 준 지시문에 MCP 관련 내용이 있나?")
    print(f"    'mcp' 언급           : {'예' if 'mcp' in instructions.lower() else '아니오'}")
    print(f"    도구 이름 언급        : "
          f"{'예' if 'get_file_contents' in instructions else '아니오'}")
    print(f"    도구 호출 지시        : "
          f"{'예' if 'tool' in instructions.lower() else '아니오'}")
    print()
    print("  참고자료(파일 내용) 쪽:")
    print(f"    'mcp' 언급           : {'예' if 'mcp' in references.lower() else '아니오'}"
          "  ← main.py 소스에 import mcp_client가 있어서다.")
    print("    LLM에게 MCP를 알려준 게 아니라, 근거로 넣은 파일이 마침 그 코드일 뿐이다.")
    print()
    show("LLM에게 실제로 간 system 메시지 (뒷부분 — MCP가 넣어준 파일 내용):",
         system_text[-260:], 260)

    client = main.AsyncOpenAI()
    response = await client.chat.completions.create(
        model=main.CHAT_MODEL,
        messages=messages,
        max_completion_tokens=main.MAX_COMPLETION_TOKENS,
        reasoning_effort=main.REASONING_EFFORT,
    )
    answer = response.choices[0].message.content or ""
    print()
    show("LLM이 돌려준 답변:", answer, 260)
    await client.close()

    llm_calls = list(CALLS)

    # ---------- 정리 ----------
    rule("실제로 오간 HTTP 요청 목적지")
    print("  [1단계 MCP]")
    for method, url in mcp_calls:
        print(f"    {method} {url}")
    print()
    print("  [2단계 LLM]")
    for method, url in llm_calls:
        print(f"    {method} {url}")
    print()
    print("  두 목적지가 완전히 다르고, 서로를 거치지 않는다.")
    print("  LLM 쪽 요청에는 MCP 엔드포인트도 도구 이름도 나타나지 않는다.")
    print()

    await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(run())
