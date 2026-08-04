"""
1일차: 사전 계산된 문서 임베딩(numpy)과 쿼리 임베딩 간 코사인 유사도 검색.
로컬 테스트: python search.py "티켓팅 인프라 프로젝트에서 왜 그 기술을 선택했나요?"
"""
import json
import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBEDDINGS_PATH = "data/embeddings.npy"
META_PATH = "data/chunks_meta.json"

_client: OpenAI | None = None
_embeddings: np.ndarray | None = None
_metadata: list[dict] | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인해주세요.")
        _client = OpenAI(api_key=api_key)
    return _client


def load_index() -> tuple[np.ndarray, list[dict]]:
    """임베딩 배열과 메타데이터를 메모리에 캐싱해 반환한다. 서버 시작 시 한 번만 호출하면 된다."""
    global _embeddings, _metadata
    if _embeddings is None or _metadata is None:
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(META_PATH):
            raise RuntimeError("임베딩 파일이 없습니다. 먼저 build_embeddings.py를 실행해주세요.")
        _embeddings = np.load(EMBEDDINGS_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            _metadata = json.load(f)
    return _embeddings, _metadata


def embed_query(query: str) -> np.ndarray:
    response = get_client().embeddings.create(model=EMBED_MODEL, input=[query])
    return np.array(response.data[0].embedding, dtype=np.float32)


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """matrix: (N, dim), vector: (dim,) -> (N,) 코사인 유사도"""
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    dot_products = matrix @ vector
    return dot_products / (matrix_norms * vector_norm + 1e-8)


def search(query: str, top_k: int = 5, threshold: float = 0.3) -> list[dict]:
    """쿼리와 가장 유사한 청크 top_k개를 유사도 내림차순으로 반환한다.
    threshold 미만인 결과는 제외한다 (No-Info 처리 기준)."""
    embeddings, metadata = load_index()
    query_vector = embed_query(query)
    scores = cosine_similarity(embeddings, query_vector)

    ranked_indices = np.argsort(-scores)[:top_k]
    results = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score < threshold:
            continue
        chunk = metadata[idx]
        results.append({**chunk, "score": score})
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('사용법: python search.py "질문 내용"')
    query_text = sys.argv[1]
    top_results = search(query_text)

    if not top_results:
        print("이 지식베이스에는 해당 내용이 없습니다.")
    else:
        for rank, result in enumerate(top_results, start=1):
            print(f"[{rank}] score={result['score']:.4f}  {result['project']} · {result['section']}")
            print(f"    {result['text'][:100]}...")
            print(f"    {result['source_link']}")
