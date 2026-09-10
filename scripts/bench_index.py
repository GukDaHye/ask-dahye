"""인덱스 로딩 시간 실측 — `data/embeddings.npy` + `data/chunks_meta.json`.

README의 "벡터DB 대신 numpy" 근거로 쓰는 수치다. 이 값이 밀리초 단위라서
Pinecone·Chroma의 인덱싱 지연과 인프라 관리 비용을 감수할 이유가 없다는 논거가 성립한다.

bench_retry·bench_mcp와 달리 별도 기록 문서를 두지 않는다. 외부 API를 부르지 않아
언제 돌려도 같은 조건이고, 이 스크립트를 그 자리에서 다시 돌리면 되기 때문이다.

실행: .venv/bin/python scripts/bench_index.py
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import search  # noqa: E402

RUNS = 15


def measure() -> float:
    # load_index는 모듈 전역에 캐싱한다. 매번 비워야 실제 로딩을 잰다.
    search._embeddings = None
    search._metadata = None
    start = time.perf_counter()
    search.load_index()
    return (time.perf_counter() - start) * 1000


def main() -> None:
    embeddings, metadata = search.load_index()
    samples = sorted(measure() for _ in range(RUNS))

    print()
    print(f"  대상        {search.EMBEDDINGS_PATH} {embeddings.shape} + {search.META_PATH} ({len(metadata)}개)")
    print(f"  파일 크기   {Path(search.EMBEDDINGS_PATH).stat().st_size / 1024:.0f}KB + "
          f"{Path(search.META_PATH).stat().st_size / 1024:.0f}KB")
    print(f"  실행 횟수   {RUNS}회")
    print()
    print(f"  중앙값      {statistics.median(samples):.2f}ms")
    print(f"  범위        {samples[0]:.2f} ~ {samples[-1]:.2f}ms")
    print()
    print("  주의: OS 파일 캐시가 더워진 상태의 값이다. 프로세스 첫 기동에서는 더 걸릴 수 있다.")
    print()


if __name__ == "__main__":
    main()
