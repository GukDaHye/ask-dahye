"""
검색 함수를 여러 질문으로 일괄 테스트하고 결과를 CSV로 남긴다.
블로그용 수치/경험담을 쌓을 때는 이 스크립트에 질문을 추가해서 반복 실행하면 된다.
실행: python test_queries.py
출력: data/test_results.csv (query, category, top1_project, top1_section, top1_score, hits, latency_ms)
"""
import csv
import time

from search import search

# category: "on-topic" (KB에 명확히 있는 질문), "adjacent" (주제는 맞지만 사실은 없는 질문), "off-topic" (완전 무관)
TEST_QUERIES: list[tuple[str, str]] = [
    ("on-topic", "티켓팅 인프라 프로젝트에서 왜 그 기술을 선택했나요?"),
    ("on-topic", "비포펫에서 이미지 업로드는 어떻게 처리했나요?"),
    ("on-topic", "72page에서 정산 처리는 어떻게 자동화했나요?"),
    ("on-topic", "돌봄플러그에서 손상된 IoT 데이터는 어떻게 걸러냈나요?"),
    ("on-topic", "일영랩 프로젝트에서 권한 로직은 어디서 관리했나요?"),
    ("on-topic", "KNS 프로젝트에서 무중단 배포는 어떻게 구현했나요?"),
    ("adjacent", "쿠버네티스 대신 Docker Swarm을 선택한 이유는?"),
    ("adjacent", "Kafka로 메시지 큐를 구성한 이유는?"),
    ("off-topic", "리액트 네이티브로 앱을 만든 경험이 있나요?"),
    ("off-topic", "GraphQL API를 설계해본 적 있나요?"),
]


def run() -> None:
    rows = []
    for category, query in TEST_QUERIES:
        t0 = time.perf_counter()
        results = search(query)
        latency_ms = (time.perf_counter() - t0) * 1000

        top1 = results[0] if results else None
        rows.append({
            "query": query,
            "category": category,
            "top1_project": top1["project"] if top1 else "",
            "top1_section": top1["section"] if top1 else "",
            "top1_score": f"{top1['score']:.4f}" if top1 else "",
            "hits": len(results),
            "latency_ms": f"{latency_ms:.0f}",
        })
        label = f"{top1['project']} · {top1['section']} ({top1['score']:.3f})" if top1 else "No-Info"
        print(f"[{category:9s}] {query}\n    -> {label}  ({latency_ms:.0f}ms, hits={len(results)})")

    with open("data/test_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n결과 저장 완료 -> data/test_results.csv")


if __name__ == "__main__":
    run()
