"""
MCP 권한 제한이 실제로 동작하는지 보여준다.

토큰은 fine-grained PAT(단일 리포, Contents Read-only)이지만 거기에만 기대지 않는다.
세 겹으로 막아두고, 각 겹이 실제로 거부하는지 확인한다.

  1) 애플리케이션 — 허용 리포 목록 밖이면 요청조차 보내지 않는다
  2) MCP 서버    — X-MCP-Tools로 노출 도구를 하나로 좁힌다(쓰기 도구는 물론 다른 읽기 도구도 거부)
  3) 토큰        — 위 둘을 뚫어도 PAT 권한이 막는다(여기서는 확인만 안내)

실행: .venv/bin/python scripts/demo_mcp_scope.py
필요: .env에 GITHUB_TOKEN, GITHUB_REPO
"""
import asyncio
import logging
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import mcp_client  # noqa: E402

logging.basicConfig(level=logging.CRITICAL)


def pad(text: str, width: int) -> str:
    display = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - display)


async def try_repo(slug: str) -> None:
    try:
        await mcp_client.list_files("/", slug=slug)
        print(f"  {pad(slug, 32)} 허용됨")
    except mcp_client.McpUnavailable as error:
        print(f"  {pad(slug, 32)} 차단 — {error}")


async def try_tool(name: str, note: str) -> None:
    owner, repo = slug_parts()
    try:
        await mcp_client._call_tool(name, {"owner": owner, "repo": repo, "path": "/"})
        print(f"  {pad(name, 24)} {pad(note, 14)} 허용됨")
    except mcp_client.McpUnavailable as error:
        reason = str(error).split(":", 2)[-1].strip()[:48]
        print(f"  {pad(name, 24)} {pad(note, 14)} 거부 — {reason}")


def slug_parts() -> tuple[str, str]:
    owner, _, repo = (mcp_client.repo_slug() or "/").partition("/")
    return owner, repo


async def main() -> None:
    if not mcp_client.is_configured():
        raise SystemExit("GITHUB_TOKEN / GITHUB_REPO가 없습니다. .env를 확인해주세요.")

    allowed = mcp_client.allowed_slugs()
    print()
    print("=" * 66)
    print(" 1겹 — 애플리케이션: 허용 리포 목록 밖이면 요청을 보내지 않는다")
    print("=" * 66)
    print(f"  허용 목록: {allowed}")
    print()
    for slug in (allowed[0], "GukDaHye/page72", "someone-else/private-repo"):
        await try_repo(slug)

    print()
    print("=" * 66)
    print(" 2겹 — MCP 서버: X-MCP-Tools로 노출 도구를 하나로 좁혔다")
    print("=" * 66)
    print(f"  허용 도구: {mcp_client.MCP_TOOLS}")
    print()
    await try_tool("get_file_contents", "(허용)")
    await try_tool("list_branches", "(읽기)")
    await try_tool("create_or_update_file", "(쓰기)")

    print()
    print("=" * 66)
    print(" 3겹 — 토큰: fine-grained PAT, 단일 리포, Contents Read-only")
    print("=" * 66)
    print("  위 두 겹을 뚫어도 토큰 권한에서 막힌다.")
    print("  GitHub 설정 > Developer settings > Fine-grained tokens 에서 확인 가능.")
    print()

    await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
