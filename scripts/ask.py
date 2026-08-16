"""
서버에 질문 하나를 던지고 결과를 요약해서 보여준다.

curl로 직접 부르면 토큰마다 SSE 프레임이 하나씩 나와서 화면이 수백 줄이 된다.
스크린샷을 찍거나 동작만 확인할 때 쓰려고 만든 것이다.

실행:
    .venv/bin/python scripts/ask.py "질문"
    .venv/bin/python scripts/ask.py "질문" --port 8099
"""
import argparse
import json
import sys
import unicodedata
import urllib.request

DEFAULT_PORT = 8000


def pad(text: str, width: int) -> str:
    display = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - display)


def ask(question: str, port: int, session: str) -> None:
    payload = json.dumps({"question": question, "session_id": session}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    tokens: list[str] = []
    events: list[str] = []
    sources: list[dict] | None = None
    message: str | None = None
    current: str | None = None

    with urllib.request.urlopen(request) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip()
            if line.startswith("event:"):
                current = line[6:].strip()
                events.append(current)
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if current == "token":
                    tokens.append(data["t"])
                elif current in ("no_info", "fallback", "error"):
                    message = data.get("message")
                elif current == "sources":
                    sources = data.get("sources")

    answer = "".join(tokens).strip() or message or ""
    flow = " -> ".join(dict.fromkeys(events))

    print()
    print(f"  {pad('질문', 10)} {question}")
    print(f"  {pad('이벤트', 10)} {flow}")
    print(f"  {pad('답변', 10)} {answer[:120]}{'…' if len(answer) > 120 else ''}")
    if sources is not None:
        if sources:
            for index, source in enumerate(sources):
                label = "출처" if index == 0 else ""
                kind = source.get("kind", "kb")
                print(f"  {pad(label, 10)} [{kind}] {source.get('project')} · {source.get('section')}")
        else:
            print(f"  {pad('출처', 10)} (없음 — 근거 없음으로 판단)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="질문을 던지고 결과를 요약해서 출력한다.")
    parser.add_argument("question", help="질문 내용")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"서버 포트 (기본 {DEFAULT_PORT})")
    parser.add_argument("--session", default="cli", help="세션 ID (기본 cli)")
    args = parser.parse_args()

    try:
        ask(args.question, args.port, args.session)
    except urllib.error.URLError as error:
        raise SystemExit(f"서버에 연결할 수 없습니다 (포트 {args.port}): {error.reason}")


if __name__ == "__main__":
    main()
