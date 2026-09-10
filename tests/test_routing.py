"""T. 키워드 라우팅 (L0) — T-01 ~ T-08

임베딩 분류기 대신 키워드로 분기한다. 판단 근거가 눈에 보이고, 오분류가 나오면
CODE_KEYWORDS만 고치면 되기 때문이다. 그래서 목록 자체를 테스트로 고정한다.
"""
import pytest

import main

MCP_QUESTIONS = [
    ("T-01", "이 챗봇은 어떻게 만들어졌나요?"),
    ("T-02", "코사인 유사도 검색 코드 어떻게 구현했어요?"),
    ("T-03", "이 프로젝트 파일 구조 알려줘"),
    ("T-04", "main 파일에서 재시도 로직 소스 어떻게 짰어요?"),
]

RAG_QUESTIONS = [
    ("T-05", "비포펫에서 이미지 업로드는 어떻게 처리했나요?"),
    ("T-06", "티켓팅 인프라에서 왜 NLB를 선택했나요?"),
    ("T-07", "돌봄플러그에서 손상된 IoT 데이터는 어떻게 걸러냈나요?"),
]


@pytest.mark.parametrize("case,question", MCP_QUESTIONS, ids=[c for c, _ in MCP_QUESTIONS])
def test_code_questions_route_to_mcp(case, question):
    assert main.is_code_question(question) is True


@pytest.mark.parametrize("case,question", RAG_QUESTIONS, ids=[c for c, _ in RAG_QUESTIONS])
def test_project_questions_route_to_rag(case, question):
    """프로젝트 경험 질문은 지식베이스가 답한다. 여기가 MCP로 새면 답변 품질이 떨어진다."""
    assert main.is_code_question(question) is False


def test_t08_matching_is_case_insensitive():
    assert main.is_code_question("GitHub 코드 보여줘") is True
    assert main.is_code_question("GITHUB 레포 알려줘") is True


def test_sample_chip_keyword_is_present():
    """결함 3-8이 다시 들어오는 걸 막는다. 목록에서 이 값이 빠지면 샘플 칩이 RAG로 샌다."""
    assert "만들어졌" in main.CODE_KEYWORDS
