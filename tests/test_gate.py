"""G. No-Info 게이트 판정 (L0) — G-03 ~ G-07

판정 기준은 "문구로 답을 시작했는가"다. 거절이면 문구가 앞에 오고, 단서면 뒤에 붙는다.
"""
import main

NO_INFO = main.NO_INFO_TEXT


def test_g03_empty_answer_is_refusal():
    assert main.is_no_info_answer("") is True


def test_g04_answer_starting_with_phrase_is_refusal():
    assert main.is_no_info_answer(NO_INFO) is True
    assert main.is_no_info_answer(f"{NO_INFO} 다른 질문을 해주세요.") is True


def test_g05_long_answer_with_trailing_phrase_is_valid():
    """G-05. 유효 답변 끝의 단서. 회귀 상세는 test_regression.py에 있다."""
    answer = "가" * 200 + NO_INFO
    assert main.is_no_info_answer(answer) is False


def test_g06_short_answer_containing_phrase_is_refusal():
    """100자 미만이면 문구가 어디에 있든 거절로 본다."""
    answer = "음, " + NO_INFO
    assert len(answer) < 100
    assert main.is_no_info_answer(answer) is True


def test_g07_phrase_after_60_chars_in_long_answer_is_valid():
    """앞 60자 안에 없고 답변이 100자 이상이면 단서로 본다."""
    answer = "나" * 61 + NO_INFO + "다" * 50
    assert len(answer) >= 100
    assert NO_INFO not in answer[:60]
    assert main.is_no_info_answer(answer) is False
