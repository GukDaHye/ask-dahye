"""P. 재시도 정책 (L0) — P-01 ~ P-08

기준은 "다시 부르면 결과가 달라질 수 있는가"다.
"""
import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

import main

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def status_error(code: int, cls=APIStatusError) -> Exception:
    return cls("boom", response=httpx.Response(code, request=REQUEST), body=None)


def test_p01_rate_limit_is_retryable():
    """429는 APIStatusError의 서브클래스다. 검사 순서가 뒤집히면 4xx로 분류돼 즉시 폴백한다."""
    error = status_error(429, RateLimitError)
    assert isinstance(error, APIStatusError)
    assert main.is_retryable(error) is True


@pytest.mark.parametrize("code", [500, 502, 503])
def test_p02_p03_server_errors_are_retryable(code):
    assert main.is_retryable(status_error(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_p04_p05_client_errors_are_not_retryable(code):
    """확정된 실패다. 재시도하면 사용자가 결과가 정해진 실패를 기다린다."""
    assert main.is_retryable(status_error(code)) is False


def test_p06_timeout_is_retryable():
    assert main.is_retryable(APITimeoutError(request=REQUEST)) is True


def test_p07_connection_error_is_retryable():
    assert main.is_retryable(APIConnectionError(message="down", request=REQUEST)) is True


@pytest.mark.parametrize("error", [ValueError("bad"), KeyError("k"), RuntimeError("x")])
def test_p08_unrelated_exceptions_are_not_retryable(error):
    assert main.is_retryable(error) is False


def test_backoff_is_three_attempts():
    """1+2+4초. 이 값이 늘면 폴백 도달 시간 실측치가 무효가 된다."""
    assert main.RETRY_DELAYS == (1, 2, 4)
