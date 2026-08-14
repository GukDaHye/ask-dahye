"""
질문이 어느 경로로 가는지 보여준다(1단계 키워드 라우팅).

임베딩 분류기 대신 키워드로 분기한 이유는 판단 근거가 눈에 보이기 때문이다.
오분류가 나오면 main.py의 CODE_KEYWORDS만 고치면 된다.

실행: .venv/bin/python scripts/demo_routing.py
서버가 필요 없다. 라우팅 함수만 직접 부른다.
"""
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402

QUESTIONS = [
    "이 챗봇은 어떻게 만들어졌나요?",
    "코사인 유사도 검색 코드 어떻게 구현했어요?",
    "이 프로젝트 파일 구조 알려줘",
    "main 파일에서 재시도 로직 소스 어떻게 짰어요?",
    "비포펫에서 이미지 업로드는 어떻게 처리했나요?",
    "티켓팅 인프라에서 왜 NLB를 선택했나요?",
    "돌봄플러그에서 손상된 IoT 데이터는 어떻게 걸러냈나요?",
]


def pad(text: str, width: int) -> str:
    display = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - display)


def matched_keywords(question: str) -> str:
    lowered = question.lower()
    hits = [k for k in main.CODE_KEYWORDS if k in lowered]
    return ", ".join(hits) if hits else "-"


print()
print(f"{pad('질문', 56)} {pad('경로', 6)} 매칭된 키워드")
print("-" * 88)
for question in QUESTIONS:
    route = "MCP" if main.is_code_question(question) else "RAG"
    print(f"{pad(question, 56)} {pad(route, 6)} {matched_keywords(question)}")
print("-" * 88)
print("  MCP = GitHub 리포의 실제 소스를 근거로 답한다")
print("  RAG = 지식베이스 텍스트를 근거로 답한다")
print()
