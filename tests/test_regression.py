"""회귀 4건 — ai_tooling_log 3장의 정확성 결함이 다시 들어오는지 감시한다.

각 케이스를 **현재 구현과 v1에 나란히** 건다.
  current -> PASS 여야 한다 (고쳐졌으므로)
  v1      -> XFAIL 이어야 한다 (결함이 있으므로 반드시 실패)

`strict=True`라서 v1이 통과해버리면 그 순간 스위트가 빨개진다.
"아무것도 잡지 않는 테스트"를 테스트가 스스로 잡는 구조다.
"""
import httpx
import pytest
from openai import APIStatusError, RateLimitError

import main
from tests import legacy


def status_error(code: int, cls=APIStatusError) -> Exception:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return cls("boom", response=httpx.Response(code, request=request), body=None)


def chunk(index: int) -> dict:
    return {
        "project": f"프로젝트{index}",
        "section": "PROBLEM",
        "source_link": f"#p{index}",
        "text": "본문",
    }


# --------------------------------------------------------------------------
# G-05 — 결함 3-4. 근거 있는 답변 끝에 붙은 단서를 거절로 오판했다.
# --------------------------------------------------------------------------
ANSWER_WITH_TRAILING_CAVEAT = (
    "재시도는 스트림을 여는 시점까지만 적용합니다. 토큰이 이미 흘러나온 뒤에 재시도하면 "
    "앞부분이 중복된 답변이 만들어지기 때문입니다. 5xx와 429, 타임아웃만 재시도하고 "
    "4xx는 즉시 폴백합니다. 다만 전체 구현은 " + main.NO_INFO_TEXT
)


@pytest.mark.parametrize("judge", [
    pytest.param(main.is_no_info_answer, id="current"),
    pytest.param(
        legacy.is_no_info_answer_v1, id="v1",
        marks=pytest.mark.xfail(strict=True, reason="결함 3-4: 부분 문자열 검사라 뒤에 붙은 단서를 거절로 오판"),
    ),
])
def test_g05_trailing_caveat_is_not_a_refusal(judge):
    """100자가 넘는 유효 답변 끝에 NO_INFO 문구가 붙어도 거절이 아니다."""
    assert len(ANSWER_WITH_TRAILING_CAVEAT) >= 100
    assert judge(ANSWER_WITH_TRAILING_CAVEAT) is False


# --------------------------------------------------------------------------
# P-04 — 결함 3-1. 4xx를 재시도 대상에 넣어 확정된 실패에 7초를 기다렸다.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("judge", [
    pytest.param(main.is_retryable, id="current"),
    pytest.param(
        legacy.is_retryable_v1, id="v1",
        marks=pytest.mark.xfail(strict=True, reason="결함 3-1: APIStatusError 전체를 재시도 대상으로 잡음"),
    ),
])
def test_p04_client_error_is_not_retryable(judge):
    """400(잘못된 모델명)은 몇 번을 불러도 같은 결과다. 재시도하지 않는다."""
    assert judge(status_error(400)) is False


@pytest.mark.parametrize("judge", [
    pytest.param(main.is_retryable, id="current"),
    pytest.param(
        legacy.is_retryable_v1, id="v1",
        marks=pytest.mark.xfail(strict=True, reason="결함 3-1: 401도 재시도했다"),
    ),
])
def test_p05_auth_error_is_not_retryable(judge):
    """401(인증 실패)도 마찬가지."""
    assert judge(status_error(401)) is False


# --------------------------------------------------------------------------
# C-01 — 결함 3-6. 답변에 쓰이지 않은 출처를 노출했다.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("build", [
    pytest.param(main.source_chips, id="current"),
    pytest.param(
        legacy.source_chips_v1, id="v1",
        marks=pytest.mark.xfail(strict=True, reason="결함 3-6: top-5를 그대로 칩으로 띄웠다"),
    ),
])
def test_c01_chips_are_capped_at_three(build):
    """프롬프트에는 5개를 넣지만 사용자에게 보여줄 칩은 상위 3개다."""
    assert len(build([chunk(i) for i in range(5)])) == main.MAX_SOURCE_CHIPS


# --------------------------------------------------------------------------
# T-01 — 결함 3-8. 샘플 칩 문구가 라우팅을 통과하지 못했다.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("route", [
    pytest.param(main.is_code_question, id="current"),
    pytest.param(
        legacy.is_code_question_v1, id="v1",
        marks=pytest.mark.xfail(strict=True, reason='결함 3-8: 키워드에 "어떻게 만들었"만 있고 "만들어졌"이 없었다'),
    ),
])
def test_t01_sample_chip_question_routes_to_mcp(route):
    """입력창 아래 샘플 칩 문구다. 이게 RAG로 가면 MCP 연동이 화면에서 안 보인다."""
    assert route("이 챗봇은 어떻게 만들어졌나요?") is True
