#!/usr/bin/env bash
# No-Info 2단 게이트를 나란히 보여준다.
#   1차 — 검색 단계에서 차단. LLM을 호출하지 않는다(토큰 비용 0).
#   2차 — 임계값은 통과하지만 LLM이 근거 없음으로 판단해 차단한다.
#
# 서버 로그에 게이트 판정이 찍히므로, 이 스크립트와 서버 터미널을 같이 캡처하면
# "1차는 LLM을 부르지도 않았다"는 게 증거로 남는다.
#
# 실행: bash scripts/demo_gates.sh
set -u

URL="http://127.0.0.1:8000"

if ! curl -sf -o /dev/null "$URL/healthz"; then
  echo "서버가 응답하지 않습니다. 먼저 .venv/bin/uvicorn main:app 을 실행해주세요."
  exit 1
fi

ask() {
  curl -sN -X POST "$URL/ask" -H 'Content-Type: application/json' \
    -d "{\"question\":\"$1\",\"session_id\":\"gate-demo-$2\"}"
}

# SSE 프레임에서 사람이 읽을 부분만 뽑는다.
summarize() {
  python3 -c '
import json, sys
tokens, event_names, sources, message = [], [], None, None
frame_event = None
for line in sys.stdin:
    line = line.rstrip()
    if line.startswith("event:"):
        frame_event = line[6:].strip()
        event_names.append(frame_event)
    elif line.startswith("data:"):
        payload = json.loads(line[5:].strip())
        if frame_event == "token":
            tokens.append(payload["t"])
        elif frame_event == "no_info":
            message = payload["message"]
        elif frame_event == "sources":
            sources = payload["sources"]
answer = "".join(tokens).strip() or message or ""
flow = " -> ".join(dict.fromkeys(event_names))
chips = len(sources) if sources is not None else 0
print("  이벤트    :", flow)
print("  답변      :", answer)
print("  출처 칩   :", str(chips) + "개")
'
}

echo ""
echo "===================================================================="
echo " 1차 게이트 — 검색 단계에서 차단 (LLM 호출 없음)"
echo " Q. 리액트 네이티브로 앱을 만든 경험이 있나요?"
echo "===================================================================="
ask "리액트 네이티브로 앱을 만든 경험이 있나요?" 1 | summarize

echo ""
echo "===================================================================="
echo " 2차 게이트 — 임계값은 통과, 시스템 프롬프트가 차단"
echo " Q. GraphQL API를 설계해본 적 있나요?"
echo "===================================================================="
ask "GraphQL API를 설계해본 적 있나요?" 2 | summarize

echo ""
echo "--------------------------------------------------------------------"
echo " 서버 터미널의 로그를 함께 보면 1차는 'LLM 호출 생략'이 찍혀 있다."
echo "--------------------------------------------------------------------"
echo ""
