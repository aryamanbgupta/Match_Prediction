"""BR2 gate-matrix finding (2026-09-10): the v2 odds rows carry a timestamp
inside ``odds.winner``; the one-sided-book guard must not treat it as a side."""
from sim_eval.loaders import BettingOddsLoader


def test_timestamped_two_sided_book_yields_probabilities():
    probs = BettingOddsLoader.get_implied_probabilities(
        {"West Indies": 3.3333, "South Africa": 1.4286, "timestamp": "2026-01-27T15:00:19Z"}
    )
    assert set(probs) == {"West Indies", "South Africa"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_one_sided_book_still_fails_closed():
    assert BettingOddsLoader.get_implied_probabilities(
        {"A": 1.9, "B": "n/a", "timestamp": "2026-01-01T00:00:00Z"}
    ) == {}
