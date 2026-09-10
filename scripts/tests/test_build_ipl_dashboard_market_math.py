from build_ipl_dashboard import compute_bet


def test_dashboard_devig_can_flip_raw_no_bet_to_positive_edge_bet():
    prediction = {
        "team1": "A",
        "team2": "B",
        "p_team1": 0.51,
        "p_team2": 0.49,
    }
    odds = {
        "odds": {"winner": {"A": 1.9, "B": 1.9}},
        "actual_winner": "A",
    }
    raw_edges = {
        "A": 0.51 - 1.0 / 1.9,
        "B": 0.49 - 1.0 / 1.9,
    }
    assert max(raw_edges.values()) < 0.0

    decision = compute_bet(prediction, odds)

    assert decision["placed"] is True
    assert decision["bet_team"] == "A"
    assert decision["best_edge"] > 0.0
    assert decision["market_prob"] == {"A": 0.5, "B": 0.5}
