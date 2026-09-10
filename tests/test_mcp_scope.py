"""S. MCP 권한 차단 (L0) — S-02 ~ S-05, S-09

토큰이 단일 리포 read-only PAT이지만 거기에만 기대지 않는다.
애플리케이션 층에서 먼저 막아 의도하지 않은 요청이 아예 나가지 않게 한다.
"""
import pytest

import mcp_client


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "GukDaHye/ask-dahye")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")


def test_s02_other_own_repo_is_rejected(configured):
    """내 다른 리포라도 허용 목록 밖이면 요청을 보내지 않는다."""
    with pytest.raises(mcp_client.McpUnavailable, match="허용되지 않은"):
        mcp_client._resolve_slug("GukDaHye/page72")


def test_s03_third_party_repo_is_rejected(configured):
    with pytest.raises(mcp_client.McpUnavailable, match="허용되지 않은"):
        mcp_client._resolve_slug("someone-else/private-repo")


def test_s02b_allowed_repo_passes(configured):
    assert mcp_client._resolve_slug("GukDaHye/ask-dahye") == "GukDaHye/ask-dahye"
    assert mcp_client._resolve_slug(None) == "GukDaHye/ask-dahye"


def test_s04_missing_repo_setting_is_rejected(monkeypatch):
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    assert mcp_client.allowed_slugs() == ()
    with pytest.raises(mcp_client.McpUnavailable):
        mcp_client._resolve_slug(None)


def test_s05_malformed_slug_is_rejected(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "abc")
    with pytest.raises(mcp_client.McpUnavailable, match="owner/repo"):
        mcp_client._owner_and_repo(None)


def test_is_configured_needs_both_values(monkeypatch):
    """둘 중 하나만 있으면 MCP를 시도하지 않고 RAG로 간다."""
    monkeypatch.setenv("GITHUB_REPO", "GukDaHye/ask-dahye")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert mcp_client.is_configured() is False


def test_s09_toolsets_header_is_absent(configured):
    """X-MCP-Toolsets를 X-MCP-Tools와 같이 보내면 두 값이 합집합으로 처리돼
    toolset 전체가 다시 열린다. 좁히려고 넣은 헤더가 오히려 제한을 푸는 경우다."""
    headers = mcp_client._headers()
    assert "X-MCP-Toolsets" not in headers
    assert headers["X-MCP-Readonly"] == "true"
    assert headers["X-MCP-Tools"] == "get_file_contents"


def test_only_one_tool_is_exposed():
    """허용 도구가 늘면 이 테스트가 먼저 빨개진다."""
    assert mcp_client.MCP_TOOLS == "get_file_contents"
