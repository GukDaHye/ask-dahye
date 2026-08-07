#!/usr/bin/env bash
# 동시 요청이 Semaphore(1)로 순차 처리되는지 확인한다.
# 뒤 요청은 앞 요청이 끝날 때까지 기다리므로 응답 시간이 계단식으로 늘어난다.
#
# 실행: 서버를 먼저 띄운 뒤 (.venv/bin/uvicorn main:app)
#       bash scripts/bench_concurrent.sh
set -u

URL="http://127.0.0.1:8000"
COUNT=6
QUESTION="돌봄플러그에서 CRC 검증은 어떻게 했나요?"

if ! curl -sf -o /dev/null "$URL/healthz"; then
  echo "서버가 응답하지 않습니다. 먼저 .venv/bin/uvicorn main:app 을 실행해주세요."
  exit 1
fi

echo ""
echo "동시 요청 ${COUNT}건 — Semaphore(1) 순차 처리 확인"
echo "----------------------------------------------------"

TMP=$(mktemp)
for i in $(seq 1 "$COUNT"); do
  (
    elapsed=$(curl -sN -o /dev/null -w '%{time_total}' -X POST "$URL/ask" \
      -H 'Content-Type: application/json' \
      -d "{\"question\":\"${QUESTION}\",\"session_id\":\"bench-${i}\"}")
    printf '%s %s\n' "$elapsed" "$i" >> "$TMP"
  ) &
done
wait

sort -n "$TMP" | awk '{ printf "  완료 %d번째   %6.1fs   (요청 #%s)\n", NR, $1, $2 }'
rm -f "$TMP"

echo "----------------------------------------------------"
echo "응답 시간이 계단식으로 늘어나면 순차 처리가 동작한 것입니다."
echo ""
