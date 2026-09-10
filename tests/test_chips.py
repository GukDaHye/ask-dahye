"""C. 출처 칩 (L0) — C-01 ~ C-04

프롬프트에 넣는 근거(5개)와 사용자에게 보여주는 출처(3개)는 다른 범위다.
"""
import main


def chunk(index: int, **extra) -> dict:
    return {
        "project": f"프로젝트{index}",
        "section": "PROBLEM",
        "source_link": f"#p{index}",
        "text": "본문",
        **extra,
    }


def test_c01_five_chunks_yield_three_chips():
    assert len(main.source_chips([chunk(i) for i in range(5)])) == 3


def test_c02_two_chunks_yield_two_chips():
    assert len(main.source_chips([chunk(i) for i in range(2)])) == 2


def test_c03_knowledge_base_chunk_defaults_to_kb_kind():
    """프론트가 칩 모양을 kind로 고른다. 기본값이 빠지면 GitHub 칩으로 잘못 그려진다."""
    assert main.source_chips([chunk(0)])[0]["kind"] == "kb"


def test_c04_mcp_chunk_keeps_github_kind():
    assert main.source_chips([chunk(0, kind="github")])[0]["kind"] == "github"


def test_chip_shape_has_all_fields_front_needs():
    chip = main.source_chips([chunk(0)])[0]
    assert set(chip) == {"project", "section", "link", "kind"}
