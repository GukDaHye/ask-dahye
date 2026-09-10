"""R. 검색 쿼리 조립과 결과 필터 (L0) — R-11 ~ R-14"""
from collections import deque

import numpy as np

import main


def history_with(*questions) -> deque:
    turns = deque(maxlen=main.MAX_HISTORY_TURNS)
    for question in questions:
        turns.append({"question": question, "answer": "답변"})
    return turns


def test_r11_followup_question_gets_previous_turn_prepended():
    """"그 프로젝트"만으로는 임베딩에 정보가 없다. 직전 질문을 붙여 맥락을 보강한다."""
    query = main.build_search_query("그 프로젝트 다른 이슈는?", history_with("티켓팅 인프라 얘기해줘"))
    assert query == "티켓팅 인프라 얘기해줘\n그 프로젝트 다른 이슈는?"


def test_r11b_only_the_latest_turn_is_attached():
    """2턴 이상 거슬러 올라가지 않는다. 붙이는 건 직전 1턴뿐이다."""
    query = main.build_search_query("그건?", history_with("첫 질문", "둘째 질문"))
    assert query.startswith("둘째 질문\n")
    assert "첫 질문" not in query


def test_r12_first_question_is_used_as_is():
    assert main.build_search_query("비포펫 알려줘", deque()) == "비포펫 알려줘"


def test_r13_orthogonal_vector_returns_no_chunks():
    """임계값 0.3을 넘는 청크가 없으면 빈 목록. 1차 게이트가 여기에 걸린다."""
    noise = np.random.default_rng(0).normal(size=1536).astype(np.float32)
    assert main.top_chunks(noise) == []


def test_r14_result_is_capped_at_top_k():
    embeddings, _ = main.load_index()
    assert len(main.top_chunks(embeddings.mean(axis=0))) == main.TOP_K


def test_r14b_every_returned_chunk_clears_the_threshold():
    embeddings, _ = main.load_index()
    results = main.top_chunks(embeddings[0])
    assert results
    assert all(c["score"] >= main.SIMILARITY_THRESHOLD for c in results)


def test_results_are_sorted_by_score_descending():
    embeddings, _ = main.load_index()
    scores = [c["score"] for c in main.top_chunks(embeddings.mean(axis=0))]
    assert scores == sorted(scores, reverse=True)
