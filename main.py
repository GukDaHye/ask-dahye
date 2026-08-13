"""
2일차: FastAPI 서버 — 검색 → 프롬프트 조립 → LLM 스트리밍 릴레이.
실행: uvicorn main:app --reload

1일차 테스트에서 확인한 두 가지가 이 파일의 설계를 결정했다.
  - top-1이 오답인 경우가 있었다(KNS 질문: 정답이 2위) -> 검색된 청크를 전부 프롬프트에 넣고 LLM이 고르게 한다.
  - 유사도 임계값을 넘겨도 근거가 없는 경우가 있었다(GraphQL 질문) -> 임계값(1차)과 시스템 프롬프트(2차)로 이중 차단한다.
"""
import asyncio
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

import mcp_client
from search import EMBED_MODEL, cosine_similarity, load_index

load_dotenv()

# print()는 stdout이 tty가 아닐 때 버퍼링돼서 배포 환경에서 로그가 늦게 뜨거나 유실된다.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ask_dahye")

# reasoning 모델이라 기본값은 첫 토큰까지 1.2초가 걸린다. low로 낮추면 1초 이내로 줄어 스트리밍 체감이 낫다.
# (minimal은 이 모델에서 지원하지 않는다.)
# 모델명을 환경변수로 뺀 이유: fallback 로직을 로컬에서 테스트할 때 존재하지 않는 모델명을 넣어 실패를 유도한다.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-5.4-mini")
REASONING_EFFORT = "low"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
MAX_HISTORY_TURNS = 3

# 프롬프트에는 top-5를 다 넣지만(1일차에 top-1이 오답인 경우를 봤다) 화면 칩은 상위 3개만 보여준다.
# 임계값을 겨우 넘긴 청크까지 출처로 띄우면 답변에 쓰이지 않은 프로젝트가 근거처럼 보인다.
MAX_SOURCE_CHIPS = 3

# 비용 방어: 공개 URL이라 질문 하나가 그대로 토큰 비용이 된다.
MAX_QUESTION_CHARS = 300
MAX_COMPLETION_TOKENS = 700
IP_WINDOW_SECONDS = 60
IP_MAX_REQUESTS = 10
DAILY_MAX_REQUESTS = 500

# 재시도: 스트림을 여는 데까지만 적용한다(토큰이 나오기 시작한 뒤에 재시도하면 답변이 중복된다).
RETRY_DELAYS = (1, 2, 4)

NO_INFO_TEXT = "이 지식베이스에는 해당 내용이 없습니다."

SYSTEM_PROMPT = f"""너는 국다혜(백엔드 엔지니어)의 포트폴리오를 근거로 답하는 어시스턴트다.

규칙:
1. 아래 참고자료에 있는 내용만 근거로 답한다. 자료에 없는 기술·경험·수치는 절대 만들어내지 않는다.
2. 참고자료로 답할 수 없는 질문이면 "{NO_INFO_TEXT}"라고만 답하고 추측하지 않는다.
3. 참고자료가 질문과 주제만 겹치고 실제 답이 없는 경우도 2번에 해당한다.
   (예: 자료에 EKS 이야기만 있는데 Docker Swarm을 왜 골랐냐고 묻는 경우)
4. 한국어로 3~5문장, 담백한 존댓말로 답한다. 수치는 자료에 있는 값만 그대로 인용한다.
5. 참고자료 번호나 "참고자료에 따르면" 같은 표현은 쓰지 않는다. 출처는 화면에서 따로 보여준다.
6. 마크다운을 쓰지 않는다. 백틱, 별표, 목록 기호 없이 일반 문장으로만 답한다(화면이 평문으로 렌더링한다)."""

# 코드 질문(MCP 경로)용 프롬프트. 근거가 문서가 아니라 실제 파일이라 지시가 달라진다.
CODE_SYSTEM_PROMPT = f"""너는 국다혜의 GitHub 리포 코드를 근거로 답하는 어시스턴트다.

규칙:
1. 아래 참고자료는 실제 리포에서 읽어온 파일 내용이다. 파일에 있는 것만 근거로 답한다.
2. 파일에 없는 구현·기술·수치는 만들어내지 않는다. 답할 수 없으면 "{NO_INFO_TEXT}"라고만 답한다.
3. 질문이 이 파일들과 무관하면 억지로 연결하지 말고 2번대로 답한다.
4. 한국어로 3~6문장, 담백한 존댓말로 답한다. 근거가 된 파일명은 언급해도 좋다.
5. 코드를 길게 그대로 붙여넣지 말고, 무엇을 어떻게 처리하는지 설명한다.
6. 참고자료에 "(파일 목록)" 항목이 있으면 그것이 리포의 실제 파일 목록이다.
   구조를 묻는 질문에는 그 목록을 근거로 답한다. 목록이 있는데 "없습니다"라고 답하지 않는다.
   파일 경로를 나열할 때는 줄바꿈으로 구분한다(화면이 줄바꿈을 그대로 렌더링한다).
7. 마크다운을 쓰지 않는다. 백틱, 별표, 하이픈 목록 기호 없이 쓴다."""

# 1단계 라우팅용 키워드. 임베딩 분류기 대신 키워드를 쓴 이유는 판단 근거가 로그에 그대로
# 남고 눈으로 검증할 수 있어서다. 오분류가 보이면 이 목록만 고치면 된다.
CODE_KEYWORDS = (
    "코드", "구현", "소스", "어떻게 짰", "어떻게 만들었", "만들어졌",
    "파일 구조", "디렉토리", "함수", "클래스", "리팩터", "리팩토링",
    "레포", "리포", "깃허브", "github",
)

_client: AsyncOpenAI | None = None
_sessions: dict[str, deque] = {}
_ip_hits: dict[str, deque] = {}
_daily = {"date": "", "count": 0}

# 429가 나면 순차 처리로 흘린다. 완전한 분산 큐는 이 트래픽 규모에 과설계이므로 단일 프로세스 안에서 해결한다.
# asyncio.Queue + 워커 풀 대신 Semaphore를 쓴 이유: 토큰을 요청 핸들러가 직접 릴레이해야 하는데,
# 큐로 넘기면 응답 토큰을 되돌려받는 큐가 하나 더 필요해진다. 동시 실행 상한이라는 목적은 동일하다.
_llm_gate = asyncio.Semaphore(1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str = Field(min_length=1, max_length=64)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 임베딩은 서버 시작 시 한 번만 메모리에 올린다. 요청마다 재계산하지 않는 게 응답 지연을 줄이는 핵심.
    embeddings, metadata = load_index()
    log.info("임베딩 로드 완료: %d개 청크, dim=%d", embeddings.shape[0], embeddings.shape[1])
    global _client
    _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if mcp_client.is_configured():
        log.info("GitHub MCP 연동 활성 (%s)", mcp_client.repo_slug())
    else:
        log.info("GitHub MCP 미설정 — 코드 질문도 RAG 경로로 처리한다")
    yield
    await _client.close()
    await mcp_client.close()


app = FastAPI(title="Ask Dahye", lifespan=lifespan)


def client_ip(request: Request) -> str:
    # Railway는 프록시 뒤에 있어서 remote address가 전부 같게 보인다.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_limits(ip: str) -> str | None:
    """한도를 넘었으면 사용자에게 보여줄 메시지를, 통과했으면 None을 반환한다."""
    now = time.time()

    today = time.strftime("%Y-%m-%d")
    if _daily["date"] != today:
        _daily.update(date=today, count=0)
    if _daily["count"] >= DAILY_MAX_REQUESTS:
        return "오늘 받을 수 있는 질문 수를 넘었습니다. 내일 다시 시도해주세요."

    hits = _ip_hits.setdefault(ip, deque())
    while hits and now - hits[0] > IP_WINDOW_SECONDS:
        hits.popleft()
    if len(hits) >= IP_MAX_REQUESTS:
        return "질문이 너무 빠르게 들어왔습니다. 잠시 후 다시 시도해주세요."

    hits.append(now)
    _daily["count"] += 1
    return None


def build_search_query(question: str, history: deque) -> str:
    """검색용 쿼리를 만든다.

    "그 프로젝트에서 다른 이슈는 없었어?" 같은 후속 질문은 질문 문장만으로는 검색이 불가능하다
    ("그 프로젝트"가 무엇인지 임베딩에 정보가 없다). 직전 질문을 앞에 붙여 맥락을 보강한다.
    """
    if history:
        return f"{history[-1]['question']}\n{question}"
    return question


async def embed(text: str) -> np.ndarray:
    response = await _client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(response.data[0].embedding, dtype=np.float32)


def top_chunks(query_vector: np.ndarray) -> list[dict]:
    embeddings, metadata = load_index()
    scores = cosine_similarity(embeddings, query_vector)
    ranked = np.argsort(-scores)[:TOP_K]
    return [
        {**metadata[i], "score": float(scores[i])}
        for i in ranked
        if float(scores[i]) >= SIMILARITY_THRESHOLD
    ]


def format_reference(chunk: dict) -> str:
    # MCP로 가져온 파일은 경계를 분명히 해서 넣는다. 지식베이스 청크는 기존 형식 유지.
    if chunk.get("kind") == "github":
        return f"[파일: {chunk['section']}]\n<<<\n{chunk['text']}\n>>>"
    return f"[{chunk['project']} · {chunk['section']}]\n{chunk['text']}"


def build_messages(
    question: str, chunks: list[dict], history: deque, system_prompt: str = SYSTEM_PROMPT
) -> list[dict]:
    references = "\n\n".join(format_reference(c) for c in chunks)
    messages = [{"role": "system", "content": f"{system_prompt}\n\n참고자료:\n{references}"}]
    for turn in history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def source_chips(chunks: list[dict]) -> list[dict]:
    return [
        {
            "project": c["project"],
            "section": c["section"],
            "link": c["source_link"],
            # 프론트가 GitHub 파일 칩과 지식베이스 칩을 구분해 표시하는 데 쓴다.
            "kind": c.get("kind", "kb"),
        }
        for c in chunks[:MAX_SOURCE_CHIPS]
    ]


def is_no_info_answer(answer: str) -> bool:
    """LLM이 "근거 없음"으로 답한 것인지 판정한다(2차 게이트).

    단순 부분 문자열 검사로는 안 된다. 근거 있는 답을 한 뒤 마지막에 단서를 붙이는 패턴이
    흔한데("...전체 구현은 이 지식베이스에는 해당 내용이 없습니다"), 그걸 거절로 보면
    유효한 답변의 출처 칩까지 사라진다. MCP 경로에서는 파일을 일부만 넣기 때문에
    이 패턴이 특히 자주 나온다.

    그래서 "문구로 답을 시작했는가"를 기준으로 본다. 거절이면 문구가 앞에 오고,
    단서면 뒤에 붙는다.
    """
    if not answer:
        return True
    if NO_INFO_TEXT in answer[:60]:
        return True
    # 짧은 답변에 문구만 들어 있으면 거절로 본다.
    return len(answer) < 100 and NO_INFO_TEXT in answer


def is_code_question(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in CODE_KEYWORDS)


async def collect_mcp_chunks(question: str) -> list[dict]:
    """MCP 경로로 근거를 모은다. 실패하면 빈 목록을 반환해 호출자가 RAG로 폴백하게 한다."""
    try:
        return await mcp_client.collect_code_context(question)
    except mcp_client.McpUnavailable as error:
        log.warning("MCP 사용 불가 -> RAG 폴백: %s", error)
        return []
    except Exception as error:
        # 예상 못 한 오류에도 챗봇이 죽지 않아야 한다.
        log.error("MCP 예상 외 오류 -> RAG 폴백: %s: %s", type(error).__name__, error)
        return []


def is_retryable(error: Exception) -> bool:
    """재시도해서 결과가 달라질 수 있는 실패인지 판단한다.

    RateLimitError(429)는 APIStatusError의 서브클래스라 먼저 확인한다.
    잘못된 모델명·인증 실패 같은 4xx는 몇 번을 다시 불러도 같은 결과이므로 재시도하지 않는다.
    (전부 재시도하면 fallback까지 7초를 헛되게 기다린다.)
    """
    if isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code >= 500
    return False


async def open_stream(messages: list[dict]):
    """스트림을 여는 데까지만 재시도한다. 전부 실패하면 마지막 예외를 올린다."""
    last_error: Exception | None = None
    for delay in (0, *RETRY_DELAYS):
        if delay:
            await asyncio.sleep(delay)
        try:
            return await _client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort=REASONING_EFFORT,
                stream=True,
            )
        except Exception as error:
            last_error = error
            if not is_retryable(error):
                log.error("재시도 불가 오류: %s: %s", type(error).__name__, error)
                break
            log.warning("재시도 대상 오류: %s: %s", type(error).__name__, error)
    raise last_error


async def answer_stream(question: str, session_id: str, ip: str):
    limit_message = check_limits(ip)
    if limit_message:
        yield sse("error", {"message": limit_message})
        return

    history = _sessions.setdefault(session_id, deque(maxlen=MAX_HISTORY_TURNS))

    # ---- 1단계 라우팅: 코드 질문이면 MCP 경로, 아니면 기존 RAG 경로 ----
    chunks: list[dict] = []
    route = "rag"
    if is_code_question(question):
        if not mcp_client.is_configured():
            log.info("라우팅: 코드 질문이지만 GITHUB_TOKEN/GITHUB_REPO 미설정 -> RAG 경로")
        else:
            chunks = await collect_mcp_chunks(question)
            if chunks:
                route = "mcp"
                log.info("라우팅: MCP 경로 (%s, 파일 %d개)", mcp_client.repo_slug(), len(chunks))
            else:
                route = "mcp->rag"
                log.info("라우팅: MCP 근거 없음 -> RAG 경로로 폴백")
    else:
        log.info("라우팅: RAG 경로 (코드 키워드 없음)")

    # ---- 기존 RAG 경로 (검증 완료. MCP가 근거를 못 구했을 때도 여기로 내려온다) ----
    if not chunks:
        try:
            query_vector = await embed(build_search_query(question, history))
        except Exception as error:
            log.error("임베딩 호출 실패: %s: %s", type(error).__name__, error)
            yield sse("error", {"message": "지금은 답변이 어렵습니다. 잠시 후 다시 시도해주세요."})
            return

        chunks = top_chunks(query_vector)
        log.info("질문: %s | 임계값 %.2f 통과 청크 %d개", question, SIMILARITY_THRESHOLD, len(chunks))

    # 1차 게이트: 근거가 하나도 없으면 LLM을 부르지 않고 바로 No-Info로 끝낸다.
    # MCP 경로에서 관련 파일을 못 찾은 경우도 RAG를 거쳐 여기서 함께 걸린다.
    if not chunks:
        log.info("1차 게이트 차단 -> LLM 호출 생략 (토큰 비용 0)")
        yield sse("no_info", {"message": NO_INFO_TEXT})
        return

    log.info("LLM 호출 (경로 %s, 근거 %d개 전달)", route, len(chunks))
    messages = build_messages(
        question, chunks, history, CODE_SYSTEM_PROMPT if route == "mcp" else SYSTEM_PROMPT
    )

    async with _llm_gate:
        try:
            stream = await open_stream(messages)
        except Exception as error:
            # 재시도가 전부 실패하면 빈 에러 화면 대신 검색된 청크 링크라도 보여준다.
            log.error("스트림 열기 실패 -> fallback: %s: %s", type(error).__name__, error)
            yield sse("fallback", {
                "message": "지금은 답변 생성이 어렵습니다. 대신 관련 프로젝트를 안내합니다.",
                "sources": source_chips(chunks),
            })
            return

        collected: list[str] = []
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    collected.append(delta)
                    yield sse("token", {"t": delta})
        except Exception as error:
            # 토큰이 나오기 시작한 뒤의 실패는 재시도하면 답변이 중복되므로, 받은 만큼만 두고 끊는다.
            log.error("스트리밍 중단: %s: %s", type(error).__name__, error)
            if not collected:
                yield sse("fallback", {
                    "message": "지금은 답변 생성이 어렵습니다. 대신 관련 프로젝트를 안내합니다.",
                    "sources": source_chips(chunks),
                })
                return

    answer = "".join(collected).strip()

    # 2차 게이트: 임계값은 넘었지만 LLM이 근거 없다고 판단한 경우. 이때 출처 칩을 띄우면
    # "내용이 없습니다 + 출처 3개"가 되어 화면이 모순되므로 칩을 숨기고,
    # 1차 게이트와 같은 회색 No-Info 톤으로 보이도록 프론트에 알린다.
    if is_no_info_answer(answer):
        log.info("2차 게이트 차단 -> LLM이 근거 없음으로 판단, 출처 칩 숨김")
        yield sse("sources", {"sources": [], "no_info": True})
    else:
        log.info("답변 완료 (%d자, 출처 칩 %d개)", len(answer), len(source_chips(chunks)))
        yield sse("sources", {"sources": source_chips(chunks), "no_info": False})
        history.append({"question": question, "answer": answer})

    yield sse("done", {})


@app.post("/ask")
async def ask(payload: AskRequest, request: Request):
    return StreamingResponse(
        answer_stream(payload.question.strip(), payload.session_id, client_ip(request)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/healthz")
async def healthz():
    embeddings, _ = load_index()
    return {"status": "ok", "chunks": int(embeddings.shape[0])}


# 정적 HTML을 같은 서버에서 서빙한다. 배포가 한 곳으로 끝나고 CORS 설정도 필요없어진다.
# /ask, /healthz 라우트를 가리지 않도록 마지막에 마운트한다.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
