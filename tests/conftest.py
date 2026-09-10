"""프로젝트 루트를 import 경로에 넣는다.

main.py는 import 시점에 API 키를 요구하지 않는다(클라이언트는 lifespan에서 만든다).
그래서 서버를 띄우지 않고 순수 함수만 직접 부를 수 있다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
