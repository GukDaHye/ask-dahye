"""
1일차: knowledge_base.json의 청크를 OpenAI 임베딩으로 변환해 numpy 배열로 저장한다.
실행: python build_embeddings.py
출력: data/embeddings.npy (N x dim float32), data/chunks_meta.json (임베딩 행과 1:1 매핑되는 청크 메타데이터)
"""
import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
KB_PATH = "knowledge_base.json"
EMBEDDINGS_PATH = "data/embeddings.npy"
META_PATH = "data/chunks_meta.json"


def load_chunks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        kb = json.load(f)
    return kb["chunks"]


def build_embedding_input(chunk: dict) -> str:
    # 프로젝트/섹션명을 본문 앞에 붙여 청크만으로는 드러나지 않는 맥락(어떤 프로젝트인지)을 임베딩에 포함시킨다.
    return f"[{chunk['project']} · {chunk['section']}]\n{chunk['text']}"


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일에 값을 채워주세요 (.env.example 참고).")

    chunks = load_chunks(KB_PATH)
    inputs = [build_embedding_input(c) for c in chunks]

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=EMBED_MODEL, input=inputs)

    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)

    os.makedirs("data", exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"임베딩 {embeddings.shape[0]}개 (dim={embeddings.shape[1]}) 저장 완료 -> {EMBEDDINGS_PATH}")
    print(f"메타데이터 저장 완료 -> {META_PATH}")


if __name__ == "__main__":
    main()
