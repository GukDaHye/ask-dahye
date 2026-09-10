"""O. 운영 한도 (L0) — O-04 ~ O-07, O-09

check_limits는 모듈 전역 딕셔너리를 쓴다. 케이스마다 초기화해서 서로 오염되지 않게 한다.
"""
import time
from collections import deque

import pytest

import main


@pytest.fixture(autouse=True)
def reset_counters():
    main._ip_hits.clear()
    main._daily.update(date=time.strftime("%Y-%m-%d"), count=0)
    yield
    main._ip_hits.clear()
    main._daily.update(date=time.strftime("%Y-%m-%d"), count=0)


def test_o04_requests_within_the_limit_pass():
    for _ in range(main.IP_MAX_REQUESTS):
        assert main.check_limits("1.1.1.1") is None


def test_o04b_request_over_the_ip_limit_is_blocked():
    for _ in range(main.IP_MAX_REQUESTS):
        main.check_limits("1.1.1.1")
    message = main.check_limits("1.1.1.1")
    assert message is not None
    assert "빠르게" in message


def test_o04c_limit_is_tracked_per_ip():
    """한 사람이 막혔다고 다른 방문자까지 막히면 안 된다."""
    for _ in range(main.IP_MAX_REQUESTS):
        main.check_limits("1.1.1.1")
    assert main.check_limits("2.2.2.2") is None


def test_o05_daily_cap_blocks_regardless_of_ip():
    """비용 방어. IP를 바꿔도 총량은 막힌다."""
    main._daily["count"] = main.DAILY_MAX_REQUESTS
    message = main.check_limits("3.3.3.3")
    assert message is not None
    assert "오늘" in message


def test_o06_daily_counter_resets_on_date_change():
    main._daily.update(date="2000-01-01", count=main.DAILY_MAX_REQUESTS)
    assert main.check_limits("4.4.4.4") is None
    assert main._daily["date"] == time.strftime("%Y-%m-%d")


def test_o07_old_hits_outside_the_window_are_dropped():
    """60초 창을 벗어난 기록은 큐에서 빠져야 한다. 안 빠지면 영구 차단이 된다."""
    stale = time.time() - main.IP_WINDOW_SECONDS - 1
    main._ip_hits["5.5.5.5"] = deque([stale] * main.IP_MAX_REQUESTS)
    assert main.check_limits("5.5.5.5") is None


def test_o09_session_history_keeps_only_three_turns():
    history = deque(maxlen=main.MAX_HISTORY_TURNS)
    for i in range(5):
        history.append({"question": f"q{i}", "answer": f"a{i}"})
    assert len(history) == 3
    assert [t["question"] for t in history] == ["q2", "q3", "q4"]


def test_question_length_cap_is_enforced_by_the_schema():
    """O-01. 300자 초과는 서버가 아니라 Pydantic이 422로 막는다."""
    from pydantic import ValidationError

    main.AskRequest(question="가" * main.MAX_QUESTION_CHARS, session_id="s")
    with pytest.raises(ValidationError):
        main.AskRequest(question="가" * (main.MAX_QUESTION_CHARS + 1), session_id="s")
    with pytest.raises(ValidationError):
        main.AskRequest(question="", session_id="s")
