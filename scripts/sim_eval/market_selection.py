"""Shared, outcome-blind rules for selecting cricket winner markets."""
from __future__ import annotations


MONEYLINE_TYPE = "moneyline"


def classify_h2h(question, title, sports_market_type) -> tuple[bool | None, str | None]:
    """Classify H2H from the two sealed structural signals, failing closed."""
    by_identity = None
    if question is not None and title is not None:
        by_identity = str(question).strip() == str(title).strip()
    by_type = None
    if sports_market_type:
        by_type = str(sports_market_type) == MONEYLINE_TYPE
    if by_type is None and by_identity is None:
        return None, "no_h2h_evidence"
    if by_type is not None and by_identity is not None and by_type != by_identity:
        return None, "h2h_rule_conflict"
    return (by_type if by_type is not None else by_identity), None
