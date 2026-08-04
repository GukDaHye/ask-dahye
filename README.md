# Ask Dahye

포트폴리오 RAG 챗봇. 질문을 임베딩해 사전 계산된 문서 임베딩과 코사인 유사도로 비교하고,
검색된 청크를 근거로 LLM이 답변을 스트리밍한다. 벡터DB 없이 numpy 배열 전수 비교로 처리한다.

## 구성

| 파일 | 역할 |
|---|---|
| `knowledge_base.json` | 지식베이스 원본 (청크 23개) |
| `build_embeddings.py` | 청크 → 임베딩 → `data/embeddings.npy` 생성 |
| `search.py` | 코사인 유사도 검색 (CLI 테스트 가능) |
| `test_queries.py` | 질문 일괄 테스트 → `data/test_results.csv` |
| `main.py` | FastAPI 서버 (검색 + 프롬프트 조립 + SSE 스트리밍) |
| `static/index.html` | 프론트 (SSE 수신, Thinking → Streaming → Done) |

## 로컬 실행

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env    # OPENAI_API_KEY 채우기
.venv/bin/python build_embeddings.py
.venv/bin/uvicorn main:app --reload
```

`http://localhost:8000` 접속.

## 검색만 따로 테스트

```bash
.venv/bin/python search.py "티켓팅 인프라에서 왜 그 기술을 선택했나요?"
.venv/bin/python test_queries.py
```

## 설계 메모

**검색된 청크를 전부 프롬프트에 넣는다.** 1일차 테스트에서 top-1이 오답인 경우를 봤다
(KNS 무중단 배포 질문: 정답 청크가 0.4034로 2위, 무관한 티켓팅 청크가 0.4121로 1위).
top-1만 신뢰하면 틀린 프로젝트를 근거로 답한다.

**No-Info를 2단으로 막는다.** 유사도 임계값만으로는 부족하다. GraphQL 질문은 임계값(0.3)을
넘는 청크가 4개 나왔지만 지식베이스에 GraphQL 경험은 없다. 코사인 유사도는 "주제가 비슷하다"를
재는 것이지 "사실이 맞다"를 재지 않는다. 그래서 임계값(1차)과 시스템 프롬프트(2차)로 이중 차단한다.

**후속 질문은 직전 질문을 붙여 임베딩한다.** "그 프로젝트에서 다른 이슈는 없었어?"는 문장만으로는
검색이 불가능하다. 대화 이력을 LLM 프롬프트에만 넣으면 검색 단계가 여전히 실패한다.

**4xx는 재시도하지 않는다.** 잘못된 모델명이나 인증 실패는 몇 번을 불러도 같은 결과다.
일시적 5xx·429·타임아웃만 exponential backoff(1s, 2s, 4s)로 재시도하고, 전부 실패하면
검색된 청크 링크를 보여주는 fallback으로 전환한다.

**서버리스를 쓰지 않는다.** 임베딩 메모리 캐싱, 세션 dict, 동시 실행 상한이 모두 프로세스 수명에
의존한다. 콜드스타트마다 초기화되면 이 설계가 성립하지 않아 상시 프로세스 방식으로 배포한다.

## 배포 (Railway)

1. Railway에서 이 저장소를 연결하거나 `railway up`으로 업로드
2. 환경변수 `OPENAI_API_KEY` 설정 (`.env`는 커밋되지 않는다)
3. `Procfile`의 `$PORT`를 Railway가 주입한다
4. 헬스체크 경로: `/healthz`
