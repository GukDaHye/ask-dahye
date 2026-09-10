"""고치기 전 구현(v1) — 회귀 테스트가 실제로 결함을 잡는지 증명하기 위한 것.

프로덕션에서 쓰지 않는다. `test_regression.py`가 같은 케이스를 v1과 현재 구현에
나란히 걸어, **v1에서 반드시 실패(RED)하는지** 확인한다.
v1이 통과해버리면 그 테스트는 아무것도 잡지 않는 것이므로 xfail(strict=True)로 실패시킨다.

## 출처 구분 — 4건 중 2건만 실물이다

첫 커밋(`ea45109`)이 1일차와 2일차 작업을 한꺼번에 담고 있다(git init을 2일차 끝에 했다).
그래서 2일차 안에서 고쳐진 결함은 옛 코드가 히스토리에 없다.

  [실물]   git에서 꺼낸 실제 코드
  [재구성] docs/ai_tooling_log.md의 서술로 되살린 코드. 실물이 아니다.
"""
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

NO_INFO_TEXT = "이 지식베이스에는 해당 내용이 없습니다."
MAX_SOURCE_CHIPS = 3


def is_no_info_answer_v1(answer: str) -> bool:
    """[실물] ea45109:main.py:276 — `if NO_INFO_TEXT in answer or not answer:`

    결함 3-4. 부분 문자열 검사라, 근거 있는 답 끝에 단서가 붙으면
    ("...전체 구현은 이 지식베이스에는 해당 내용이 없습니다") 거절로 오판해
    유효한 답변의 출처 칩까지 전부 사라졌다.
    """
    return NO_INFO_TEXT in answer or not answer


# [실물] de61c5b^:main.py:87 — "만들어졌"이 없다.
# 결함 3-8. 샘플 칩 문구 "이 챗봇은 어떻게 만들어졌나요?"가 MCP 경로로 가지 않았다.
CODE_KEYWORDS_V1 = (
    "코드", "구현", "소스", "어떻게 짰", "어떻게 만들었", "파일 구조", "디렉토리",
    "함수", "클래스", "리팩터", "리팩토링", "레포", "리포", "깃허브", "github",
)


def is_code_question_v1(question: str) -> bool:
    """[실물] de61c5b^:main.py:240 — 키워드 목록만 v1이다."""
    lowered = question.lower()
    return any(keyword in lowered for keyword in CODE_KEYWORDS_V1)


# [재구성] docs/ai_tooling_log.md 3-1
# "RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError)로
#  작성돼 있었는데, APIStatusError가 4xx까지 포함한다."
# 첫 커밋 시점에 이미 고쳐져 있어 git에 실물이 없다.
RETRYABLE_V1 = (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError)


def is_retryable_v1(error: Exception) -> bool:
    """[재구성] 결함 3-1. 4xx도 재시도 대상이라 확정된 실패에 1+2+4초를 기다렸다."""
    return isinstance(error, RETRYABLE_V1)


def source_chips_v1(chunks: list[dict]) -> list[dict]:
    """[재구성] docs/ai_tooling_log.md 3-6 — "검색된 top-5를 그대로 출처 칩으로 띄우도록 짰다."

    비포펫 질문에서 0.307로 겨우 임계값을 넘긴 무관한 청크가 근거처럼 표시됐다.
    첫 커밋 시점에 이미 `[:MAX_SOURCE_CHIPS]`가 들어 있어 git에 실물이 없다.
    """
    return [
        {"project": c["project"], "section": c["section"], "link": c["source_link"]}
        for c in chunks
    ]
