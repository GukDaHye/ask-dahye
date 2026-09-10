"""M. MCP 실패 → RAG 폴백 (L0) — M-06

MCP에서 무슨 일이 나든 챗봇은 답을 해야 한다. 여기서 예외가 새어 나가면
사용자는 에러 화면을 본다. 폴백 신호는 "빈 목록"이고, 호출자가 그걸 보고 RAG로 내려간다.
"""
import asyncio

import mcp_client
import main


def collect(question: str = "이 챗봇은 어떻게 만들어졌나요?") -> list[dict]:
    return asyncio.run(main.collect_mcp_chunks(question))


def raising(error: Exception):
    async def _stub(question, **kwargs):
        raise error
    return _stub


def test_m06_known_mcp_failure_falls_back(monkeypatch):
    monkeypatch.setattr(mcp_client, "collect_code_context", raising(mcp_client.McpUnavailable("401")))
    assert collect() == []


def test_m06b_unexpected_exception_also_falls_back(monkeypatch):
    """예상 못 한 오류에도 죽지 않아야 한다. 여기가 뚫리면 500이 사용자에게 간다."""
    monkeypatch.setattr(mcp_client, "collect_code_context", raising(RuntimeError("boom")))
    assert collect() == []


def test_m06c_timeout_falls_back(monkeypatch):
    monkeypatch.setattr(mcp_client, "collect_code_context", raising(TimeoutError()))
    assert collect() == []


def test_successful_collection_is_passed_through(monkeypatch):
    """폴백만 되면 안 된다. 성공 경로가 살아 있는지도 같이 본다."""
    async def _stub(question, **kwargs):
        return [{"project": "ask-dahye", "section": "main.py", "kind": "github"}]

    monkeypatch.setattr(mcp_client, "collect_code_context", _stub)
    assert collect()[0]["kind"] == "github"


def test_mcp_backoff_is_shorter_than_llm_backoff():
    """MCP는 폴백 경로가 있다. 7초를 기다린 뒤 폴백하느니 빨리 포기하는 게 낫다."""
    assert sum(mcp_client.RETRY_DELAYS) < sum(main.RETRY_DELAYS)
