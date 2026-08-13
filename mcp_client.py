"""
GitHub MCP 연동. 코드 관련 질문이 들어오면 지식베이스 텍스트 대신
실제 리포의 파일을 근거로 쓰기 위해 공식 GitHub MCP 서버를 호출한다.

기존 RAG 경로와 분리된 모듈이다. 여기서 실패하면 McpUnavailable을 올리고,
호출한 쪽(main.py)이 그걸 잡아서 RAG 경로로 되돌린다. 챗봇이 죽지 않는 게 우선이다.

전송 방식은 원격 streamable HTTP(JSON-RPC)를 골랐다. 로컬 Docker 컨테이너나
바이너리를 띄우지 않아도 되므로 상시 프로세스 배포 환경에 그대로 올라간다.

읽기 전용 도구 두 개만 쓴다.
  - list_files: 디렉토리 목록
  - read_file : 파일 내용
GitHub MCP 서버에서는 둘 다 get_file_contents 하나로 처리되고,
경로가 디렉토리면 목록, 파일이면 내용을 돌려준다.

전부 async인 이유: MCP 왕복이 한 번에 1.5~2초쯤 걸린다. 목록과 읽기를 순차로 돌리면
LLM을 부르기도 전에 12초가 넘어가 스트리밍 체감이 무너졌다. 같은 단계의 호출은 묶어서 보낸다.
"""
import asyncio
import json
import logging
import os
import time

import httpx

log = logging.getLogger("ask_dahye.mcp")

MCP_ENDPOINT = "https://api.githubcopilot.com/mcp/"
PROTOCOL_VERSION = "2025-06-18"

# 노출 도구를 서버 쪽에서 아예 좁힌다. 토큰이 Contents Read-only 단일 리포 PAT이지만
# 방어를 한 겹 더 둔다. 이 헤더를 걸면 쓰기 도구는 물론 허용 목록 밖의 읽기 도구까지
# "unknown tool"로 거부된다(list_branches, create_or_update_file 모두 거부되는 것을 확인).
MCP_READONLY = "true"
MCP_TOOLSETS = "repos"
MCP_TOOLS = "get_file_contents"

REQUEST_TIMEOUT = 15.0

# main.py는 (1, 2, 4)를 쓰지만 여기는 더 짧게 잡았다.
# MCP가 실패하면 RAG 경로로 되돌아가므로, 7초를 기다린 뒤 폴백하는 것보다
# 빨리 포기하고 답을 주는 편이 사용자 입장에서 낫다.
RETRY_DELAYS = (1, 2)

# 비용·지연 방어: 리포가 크면 목록 조회와 파일 읽기가 무한정 늘어난다.
MAX_LIST_DIRS = 6
MAX_CODE_FILES = 3
MAX_FILE_CHARS = 6000
MAX_FILE_BYTES = 40_000

# 파일 목록은 캐싱한다. 리포 구조는 질문마다 바뀌지 않는데 목록 조회가 왕복 두 번을 차지해서,
# 캐시가 살아 있으면 코드 질문이 5초에서 2초 수준으로 줄어든다(임베딩을 시작 시 한 번만
# 올리는 것과 같은 판단). 대신 커밋 직후 최대 이 시간만큼 옛 구조를 볼 수 있다.
LISTING_TTL_SECONDS = 300

SOURCE_EXTENSIONS = (
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".dart",
    ".tf", ".yaml", ".yml", ".sh", ".sql", ".html", ".css",
)
# 파일명만 보고도 중심 파일로 볼 만한 것들. 질문에 단서가 없을 때 우선순위를 준다.
ENTRYPOINT_HINTS = ("main", "app", "index", "server", "settings", "config", "views", "models")

# 한국어 질문에는 영어 파일명이 안 들어 있다("코사인 유사도 검색 코드" -> search.py를 못 고름).
# 개념어를 파일명 조각에 이어주는 최소한의 사전을 둔다. 키워드 라우팅과 같은 성격으로,
# 오분류가 보이면 이 표만 고치면 된다. (한계는 README에 적어뒀다.)
CONCEPT_HINTS: dict[str, tuple[str, ...]] = {
    "검색": ("search",),
    "유사도": ("search",),
    "코사인": ("search",),
    "임베딩": ("embed",),
    "재시도": ("main",),
    "스트리밍": ("main",),
    "프롬프트": ("main",),
    "라우팅": ("main",),
    "큐": ("main",),
    "프론트": ("index",),
    "화면": ("index",),
    "출처": ("index",),
    "테스트": ("test",),
    "벤치마크": ("bench",),
    "mcp": ("mcp_client",),
    "깃허브": ("mcp_client",),
    "github": ("mcp_client",),
}

# 파일 구조를 묻는 질문. 이때는 목록 자체가 근거가 된다.
STRUCTURE_KEYWORDS = ("파일 구조", "디렉토리", "폴더", "구조 알려", "구조가 어떻", "프로젝트 구조")

_client: httpx.AsyncClient | None = None
_session_id: str | None = None
# 동시 요청이 각자 initialize를 부르지 않도록 세션 생성을 직렬화한다.
_session_lock = asyncio.Lock()
# (만료 시각, 항목 목록). repo_slug이 바뀌면 버린다.
_listing_cache: tuple[str, float, list[dict]] | None = None


class McpUnavailable(Exception):
    """MCP 경로를 쓸 수 없다. 호출자는 RAG 경로로 폴백해야 한다."""


def allowed_slugs() -> tuple[str, ...]:
    """접근을 허용하는 리포 목록.

    지금은 GITHUB_REPO 하나뿐이다. 이 챗봇 자신의 소스를 근거로 "어떻게 만들어졌나"에
    답하는 것이 목적이므로 다중 리포는 범위가 아니다.

    목록 형태로 둔 이유는 넓힐 때 이 함수와 호출부의 slug 인자만 손대면 되게 하려는 것이다.
    라우팅(질문 -> 어느 리포) 자체는 구현하지 않았다.
    """
    slug = os.environ.get("GITHUB_REPO")
    return (slug,) if slug else ()


def repo_slug() -> str | None:
    """기본 대상 리포. 로그와 표시용."""
    allowed = allowed_slugs()
    return allowed[0] if allowed else None


def is_configured() -> bool:
    """환경변수가 다 있는지 확인한다. 없으면 라우팅 단계에서 MCP를 아예 시도하지 않는다."""
    return bool(os.environ.get("GITHUB_TOKEN") and repo_slug())


def _resolve_slug(slug: str | None) -> str:
    """대상 리포를 확정한다. 허용 목록에 없는 리포는 요청조차 보내지 않는다.

    토큰이 단일 리포 fine-grained PAT이라 다른 리포를 부르면 어차피 404가 나지만,
    코드 쪽에서 먼저 막아 의도하지 않은 접근이 나가지 않게 한다.
    """
    allowed = allowed_slugs()
    if not allowed:
        raise McpUnavailable("GITHUB_REPO가 설정되지 않았다")
    target = slug or allowed[0]
    if target not in allowed:
        raise McpUnavailable(f"허용되지 않은 리포 접근 시도: {target!r}")
    return target


def _owner_and_repo(slug: str | None = None) -> tuple[str, str]:
    target = _resolve_slug(slug)
    owner, _, repo = target.partition("/")
    if not owner or not repo:
        raise McpUnavailable(f"GITHUB_REPO 형식이 owner/repo가 아니다: {target!r}")
    return owner, repo


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    return _client


async def close() -> None:
    """서버 종료 시 커넥션을 정리한다."""
    global _client, _session_id, _listing_cache
    if _client is not None:
        await _client.aclose()
        _client = None
    _session_id = None
    _listing_cache = None


def _headers() -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Content-Type": "application/json",
        # 응답이 SSE 프레임으로 오는 경우가 있어 둘 다 받는다고 알린다.
        "Accept": "application/json, text/event-stream",
        "X-MCP-Readonly": MCP_READONLY,
        "X-MCP-Toolsets": MCP_TOOLSETS,
        "X-MCP-Tools": MCP_TOOLS,
    }
    if _session_id:
        headers["Mcp-Session-Id"] = _session_id
    return headers


def _parse_response(body: str) -> dict:
    """JSON 본문 또는 SSE 프레임에서 JSON-RPC 응답을 꺼낸다."""
    body = body.strip()
    if body.startswith("{"):
        return json.loads(body)
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload.startswith("{"):
                return json.loads(payload)
    raise McpUnavailable("MCP 응답을 해석할 수 없다")


async def _post(payload: dict) -> httpx.Response:
    """5xx와 타임아웃만 재시도한다. 4xx는 몇 번을 불러도 같은 결과라 즉시 실패시킨다."""
    last_error: Exception | None = None
    for delay in (0, *RETRY_DELAYS):
        if delay:
            await asyncio.sleep(delay)
        try:
            response = await _get_client().post(MCP_ENDPOINT, headers=_headers(), json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as error:
            last_error = error
            log.warning("MCP 통신 오류, 재시도 대상: %s: %s", type(error).__name__, error)
            continue

        if response.status_code < 400:
            return response

        last_error = McpUnavailable(f"MCP HTTP {response.status_code}: {response.text[:200]}")
        if response.status_code < 500:
            log.error("MCP %d 응답 — 재시도하지 않는다", response.status_code)
            raise last_error
        log.warning("MCP %d 응답, 재시도 대상", response.status_code)

    raise McpUnavailable(f"MCP 재시도 실패: {last_error}")


async def _initialize() -> None:
    """세션을 열고 세션 ID를 캐싱한다."""
    global _session_id
    async with _session_lock:
        _session_id = None
        response = await _post({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "ask-dahye", "version": "0.1"},
            },
        })
        _parse_response(response.text)  # 오류 응답이면 여기서 걸러진다

        session_id = response.headers.get("mcp-session-id")
        if not session_id:
            raise McpUnavailable("MCP 세션 ID를 받지 못했다")
        _session_id = session_id

        # 프로토콜 규약: initialize 뒤에 초기화 완료를 알린다. 응답 본문은 없다.
        await _get_client().post(
            MCP_ENDPOINT,
            headers=_headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        log.info("MCP 세션 생성 (readonly=%s, toolsets=%s)", MCP_READONLY, MCP_TOOLSETS)


async def _call_tool(name: str, arguments: dict) -> list[dict]:
    """도구를 호출해 content 목록을 반환한다. 세션이 끊겨 있으면 한 번 다시 연다."""
    if not _session_id:
        await _initialize()

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    try:
        message = _parse_response((await _post(payload)).text)
    except McpUnavailable as error:
        # 세션 만료로 4xx가 났을 수 있다. 딱 한 번만 다시 열고 재시도한다.
        if "HTTP 4" not in str(error):
            raise
        log.info("MCP 세션 재생성 후 재시도")
        await _initialize()
        message = _parse_response((await _post(payload)).text)

    if "error" in message:
        raise McpUnavailable(f"MCP 도구 오류({name}): {message['error']}")

    result = message.get("result") or {}
    if result.get("isError"):
        raise McpUnavailable(f"MCP 도구 실패({name}): {str(result.get('content'))[:200]}")
    return result.get("content") or []


async def list_files(path: str = "/", *, slug: str | None = None) -> list[dict]:
    """디렉토리 목록을 반환한다. 각 항목은 type/name/path/size를 가진다."""
    owner, repo = _owner_and_repo(slug)
    content = await _call_tool("get_file_contents", {"owner": owner, "repo": repo, "path": path})

    for item in content:
        if item.get("type") != "text":
            continue
        try:
            entries = json.loads(item.get("text", ""))
        except json.JSONDecodeError:
            continue
        if isinstance(entries, list):
            return entries
    return []


async def read_file(path: str, *, slug: str | None = None) -> str | None:
    """파일 내용을 반환한다. 텍스트로 못 읽으면 None."""
    owner, repo = _owner_and_repo(slug)
    content = await _call_tool("get_file_contents", {"owner": owner, "repo": repo, "path": path})

    # 파일 내용은 text가 아니라 resource 항목으로 온다(text는 "다운로드 완료" 안내문).
    for item in content:
        if item.get("type") == "resource":
            text = (item.get("resource") or {}).get("text")
            if text:
                return text
    return None


def _is_source_file(entry: dict) -> bool:
    if entry.get("type") != "file":
        return False
    name = entry.get("name", "")
    if name.startswith("."):
        return False
    if (entry.get("size") or 0) > MAX_FILE_BYTES:
        return False
    return name.endswith(SOURCE_EXTENSIONS) or name == "Dockerfile"


def _rank(entry: dict, question: str) -> tuple[int, int]:
    """질문과 파일명이 겹치는 정도로 순위를 매긴다.

    파일 내용을 임베딩해 고르지 않는 이유: 그러려면 모든 파일을 먼저 받아와야 해서
    MCP 왕복과 토큰 비용이 파일 수만큼 늘어난다. 이 규모에서는 파일명 매칭으로 충분하다.
    """
    name = entry.get("name", "").rsplit(".", 1)[0].lower()
    lowered = question.lower()

    score = 0
    if name and name in lowered:
        score += 10
    # 한국어 개념어 -> 파일명 조각 매칭
    for concept, fragments in CONCEPT_HINTS.items():
        if concept in lowered and any(fragment in name for fragment in fragments):
            score += 8
    if name in ENTRYPOINT_HINTS:
        score += 3
    # 얕은 경로를 선호한다(루트의 main.py가 깊은 유틸보다 설명력이 크다).
    return (-score, entry.get("path", "").count("/"))


async def _gather_listings(paths: list[str], slug: str) -> list[dict]:
    """여러 디렉토리를 한꺼번에 조회한다. 개별 실패는 무시하고 얻은 것만 모은다."""
    results = await asyncio.gather(
        *(list_files(p, slug=slug) for p in paths), return_exceptions=True
    )
    entries: list[dict] = []
    for path, result in zip(paths, results):
        if isinstance(result, BaseException):
            log.warning("MCP 목록 조회 실패(%s): %s", path, result)
            continue
        entries.extend(result)
    return entries


async def _repo_entries(slug: str) -> list[dict]:
    """리포의 파일·디렉토리 목록을 모은다(루트 + 한 단계 하위). TTL 동안 캐싱한다.

    캐시 키에 slug를 넣어둔 이유는 대상 리포가 바뀌어도 옛 목록을 쓰지 않게 하려는 것이다.
    """
    global _listing_cache

    if _listing_cache and _listing_cache[0] == slug and time.monotonic() < _listing_cache[1]:
        return _listing_cache[2]

    # 1) 루트를 먼저 훑는다. 세션도 여기서 열린다.
    entries = await list_files("/", slug=slug)

    # 2) 하위 디렉토리를 한 단계 내려간다. 순차로 돌면 디렉토리 수만큼 느려지므로 병렬로 보낸다.
    subdirs = [
        e["path"] for e in entries
        if e.get("type") == "dir" and not (e.get("name") or "").startswith(".")
    ][:MAX_LIST_DIRS]
    if subdirs:
        entries.extend(await _gather_listings(subdirs, slug))

    _listing_cache = (slug, time.monotonic() + LISTING_TTL_SECONDS, entries)
    log.info("MCP: 파일 목록 캐싱 (%d개 항목, 디렉토리 %d개 조회)", len(entries), len(subdirs) + 1)
    return entries


async def collect_code_context(question: str, *, slug: str | None = None) -> list[dict]:
    """리포에서 근거가 될 파일을 모아 RAG 청크와 같은 모양의 dict 목록으로 반환한다.

    같은 모양으로 맞추는 게 핵심이다. 프롬프트 조립, 2단 게이트, 출처 칩, SSE 릴레이가
    전부 이 모양을 전제로 이미 동작하므로 기존 코드를 고치지 않아도 된다.

    slug를 생략하면 GITHUB_REPO를 쓴다. 다중 리포로 넓힐 때 호출부에서 넘기면 되도록
    인자만 열어뒀고, 어느 리포로 보낼지 판단하는 로직은 넣지 않았다.
    """
    target = _resolve_slug(slug)
    owner, repo = _owner_and_repo(target)
    entries = await _repo_entries(target)

    chunks: list[dict] = []

    # 구조를 묻는 질문이면 파일 목록 자체가 근거다. 목록은 이미 캐시에 있어 추가 호출이 없다.
    if any(keyword in question.lower() for keyword in STRUCTURE_KEYWORDS):
        paths = sorted(e["path"] for e in entries if e.get("type") == "file")
        if paths:
            chunks.append({
                "project": f"{repo} 코드",
                "section": "(파일 목록)",
                "text": "\n".join(paths),
                "source_link": f"https://github.com/{owner}/{repo}",
                "kind": "github",
            })

    candidates = sorted((e for e in entries if _is_source_file(e)), key=lambda e: _rank(e, question))
    if not candidates:
        log.info("MCP: 근거로 쓸 소스 파일이 없다")
        return chunks

    # 상위 후보 파일을 병렬로 읽는다.
    picked = candidates[:MAX_CODE_FILES]
    texts = await asyncio.gather(
        *(read_file(e["path"], slug=target) for e in picked), return_exceptions=True
    )

    for entry, text in zip(picked, texts):
        if isinstance(text, BaseException):
            log.warning("MCP 파일 읽기 실패(%s): %s", entry["path"], text)
            continue
        if not text:
            continue
        truncated = text[:MAX_FILE_CHARS]
        if len(text) > MAX_FILE_CHARS:
            truncated += "\n... (이하 생략)"
        chunks.append({
            "project": f"{repo} 코드",
            "section": entry["path"],
            "text": truncated,
            "source_link": f"https://github.com/{owner}/{repo}/blob/HEAD/{entry['path']}",
            # 출처 칩을 지식베이스 청크와 구분하기 위한 표시.
            "kind": "github",
        })

    log.info("MCP: 파일 %d개 수집 (후보 %d개)", len(chunks), len(candidates))
    return chunks
