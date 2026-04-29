from huginn.retrieve.basic import score_query_against_text


def test_score_query_against_text_prefers_more_overlapping_terms() -> None:
    strong = score_query_against_text("project atlas budget", "atlas project budget budget")
    weak = score_query_against_text("project atlas budget", "atlas notes appendix")

    assert strong > weak
    assert strong > 0


def test_score_query_against_text_returns_zero_for_no_overlap() -> None:
    assert score_query_against_text("atlas budget", "completely unrelated words") == 0.0
