"""F. MCP 파일 선택 (L0) — F-01 ~ F-08

파일 내용을 임베딩해 고르지 않는다. 그러려면 모든 파일을 먼저 받아와야 해서
MCP 왕복과 토큰 비용이 파일 수만큼 늘어난다. 파일명 매칭의 한계는 README에 적어뒀다.
"""
import pytest

import mcp_client


def entry(path: str, size: int = 1000, kind: str = "file") -> dict:
    return {"path": path, "name": path.rsplit("/", 1)[-1], "size": size, "type": kind}


def best(question: str, paths: list[str]) -> str:
    entries = [entry(p) for p in paths]
    return sorted(entries, key=lambda e: mcp_client._rank(e, question))[0]["path"]


CANDIDATES = ["main.py", "search.py", "mcp_client.py", "build_embeddings.py", "static/index.html"]


@pytest.mark.parametrize("case,question,expected", [
    ("F-01", "코사인 유사도 검색 코드 어떻게 짰어요?", "search.py"),
    ("F-02", "mcp 어떻게 붙였어요?", "mcp_client.py"),
    ("F-03", "재시도 로직 보여줘", "main.py"),
    ("F-03b", "스트리밍 어떻게 처리해요?", "main.py"),
    ("F-03c", "임베딩 어떻게 만들었어요?", "build_embeddings.py"),
    ("F-03d", "출처 칩 화면은 어떻게 그려요?", "static/index.html"),
], ids=lambda v: v if isinstance(v, str) and v.startswith("F-") else "")
def test_concept_hints_map_korean_terms_to_files(case, question, expected):
    """한국어 질문에는 영어 파일명이 안 들어 있다. CONCEPT_HINTS가 그 간극을 메운다."""
    assert best(question, CANDIDATES) == expected


def test_f04_hidden_files_are_excluded():
    """.env가 근거로 올라가면 안 된다."""
    assert mcp_client._is_source_file(entry(".env")) is False
    assert mcp_client._is_source_file(entry(".gitignore")) is False


def test_f05_oversized_files_are_excluded():
    assert mcp_client._is_source_file(entry("big.py", size=mcp_client.MAX_FILE_BYTES + 1)) is False
    assert mcp_client._is_source_file(entry("small.py", size=mcp_client.MAX_FILE_BYTES)) is True


def test_f06_only_source_extensions_and_dockerfile_pass():
    assert mcp_client._is_source_file(entry("README.md")) is False
    assert mcp_client._is_source_file(entry("Dockerfile")) is True
    assert mcp_client._is_source_file(entry("main.py")) is True


def test_f06b_directories_are_not_source_files():
    assert mcp_client._is_source_file(entry("scripts", kind="dir")) is False


def test_f07_shallow_paths_win_ties():
    """루트의 main.py가 깊은 유틸보다 설명력이 크다."""
    question = "이 프로젝트 어떻게 만들어졌어요?"
    assert best(question, ["a/b/c/main.py", "main.py"]) == "main.py"


def test_f08_unknown_concept_falls_back_to_entrypoints():
    """CONCEPT_HINTS에 없는 주제여도 예외 없이 진입점을 고른다."""
    assert best("양자컴퓨팅 어떻게 붙였어요?", CANDIDATES) == "main.py"


def test_exact_filename_in_question_wins():
    assert best("search.py 보여줘", CANDIDATES) == "search.py"
