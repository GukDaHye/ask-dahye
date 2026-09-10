"""L. 관측성 (L0) — L-01

결함 3-3. 상태 출력을 print()로 짰더니 stdout이 터미널이 아닐 때 버퍼링돼서,
서버 출력을 파일로 리다이렉트하자 로그가 사라졌다.

테스트가 통과해도 존재할 수 있는 종류의 결함이라, 대장에 없으면 다시 들어온다.
그래서 동작이 아니라 소스를 검사한다.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVER_MODULES = ["main.py", "search.py", "mcp_client.py"]


def print_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


@pytest.mark.parametrize("module", SERVER_MODULES)
def test_l01_server_modules_do_not_use_print(module):
    """search.py의 print는 __main__ CLI 블록 안에만 허용된다."""
    path = ROOT / module
    lines = print_calls(path)
    if module == "search.py":
        source = path.read_text(encoding="utf-8").splitlines()
        cli_start = next(i for i, l in enumerate(source, 1) if l.startswith('if __name__'))
        lines = [n for n in lines if n < cli_start]
    assert lines == [], f"{module}에 print() 호출이 남아 있다 (줄 {lines})"


def test_logging_is_configured_in_main():
    import main
    assert main.log.name == "ask_dahye"
