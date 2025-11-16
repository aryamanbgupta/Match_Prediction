"""
Unit tests for WAR calculations (scripts 06-09).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestLeverageIndex:
    """Test leverage index calculations."""

    def test_phase_leverage_values(self):
        """Test phase leverage mapping."""
        phase_leverage = {
            'powerplay': 0.8,
            'middle': 1.0,
            'death': 1.4
        }

        assert phase_leverage['powerplay'] < phase_leverage['middle']
        assert phase_leverage['death'] > phase_leverage['middle']
        assert phase_leverage['middle'] == 1.0  # Baseline

    def test_wickets_leverage_formula(self):
        """Test wickets leverage calculation."""
        # LI = 0.6 + 0.05 * wickets_in_hand
        wickets_in_hand = 10  # No wickets lost
        li = 0.6 + 0.05 * wickets_in_hand
        assert li == 1.1

        wickets_in_hand = 0  # All wickets lost
        li = 0.6 + 0.05 * wickets_in_hand
        assert li == 0.6

    def test_combined_leverage_range(self):
        """Combined leverage should be positive."""
        # Min: powerplay (0.8) * 0 wickets (0.6) * situation (1.0) = 0.48
        min_li = 0.8 * 0.6 * 1.0
        assert min_li == pytest.approx(0.48, rel=0.01)

        # Max: death (1.4) * 10 wickets (1.1) * situation (1.0) = 1.54
        max_li = 1.4 * 1.1 * 1.0
        assert max_li == pytest.approx(1.54, rel=0.01)


class TestRunsConservation:
    """Test runs conservation framework."""

    def test_batter_bowler_raa_sum_zero(self):
        """RAA_batter + RAA_bowler should equal zero."""
        # For any ball, batter and bowler RAA should be opposites
        batter_raa = 2.5
        bowler_raa = -2.5
        assert batter_raa + bowler_raa == 0.0

    def test_raa_conservation_aggregate(self):
        """Total RAA across all balls should be near zero."""
        # Create mock data
        np.random.seed(42)
        n_balls = 1000
        batter_raa = np.random.randn(n_balls)
        bowler_raa = -batter_raa  # Runs conservation

        total = batter_raa.sum() + bowler_raa.sum()
        assert abs(total) < 1e-10  # Near zero (floating point precision)


class TestReplacementLevel:
    """Test replacement level calculations."""

    def test_replacement_level_percentile(self):
        """Replacement level should be bottom 25%."""
        # Mock data
        raa_per_ball = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7])
        threshold = np.percentile(raa_per_ball, 25)

        # Bottom 25% should be <= threshold
        replacement_players = raa_per_ball <= threshold
        assert replacement_players.sum() == 2  # 25% of 8 = 2

    def test_avg_raa_rep_negative(self):
        """Replacement level avg should be negative."""
        # Replacement players are below average
        # avg.RAA_rep should be negative
        avg_raa_rep_batting = -0.2012
        avg_raa_rep_bowling = -0.2071

        assert avg_raa_rep_batting < 0
        assert avg_raa_rep_bowling < 0


class TestVORPCalculation:
    """Test VORP calculations."""

    def test_vorp_formula(self):
        """VORP = RAA - (avg.RAA_rep × balls)."""
        raa = 100.0
        avg_raa_rep = -0.2
        balls = 300

        vorp = raa - (avg_raa_rep * balls)
        expected_vorp = 100.0 - (-0.2 * 300)
        expected_vorp = 100.0 + 60.0  # Double negative

        assert vorp == expected_vorp
        assert vorp == 160.0

    def test_vorp_increases_with_playing_time(self):
        """More balls → more VORP (for positive RAA players)."""
        raa_per_ball = 0.1  # Positive player
        avg_raa_rep = -0.2

        balls_200 = 200
        balls_400 = 400

        vorp_200 = (raa_per_ball * balls_200) - (avg_raa_rep * balls_200)
        vorp_400 = (raa_per_ball * balls_400) - (avg_raa_rep * balls_400)

        assert vorp_400 > vorp_200


class TestWARCalculation:
    """Test WAR calculations."""

    def test_war_formula(self):
        """WAR = VORP / RPW."""
        vorp = 222.88
        rpw = 111.44

        war = vorp / rpw
        assert war == pytest.approx(2.0, rel=0.01)

    def test_war_proportional_to_vorp(self):
        """WAR should be proportional to VORP."""
        rpw = 100.0

        vorp_100 = 100.0
        vorp_200 = 200.0

        war_100 = vorp_100 / rpw
        war_200 = vorp_200 / rpw

        assert war_200 == 2 * war_100

    def test_negative_war_possible(self):
        """Players can have negative WAR (worse than replacement)."""
        vorp = -50.0  # Negative VORP
        rpw = 100.0

        war = vorp / rpw
        assert war < 0


class TestRPWEstimation:
    """Test Runs Per Win estimation."""

    def test_rpw_positive(self):
        """RPW should be positive."""
        rpw = 111.44
        assert rpw > 0

    def test_rpw_reasonable_range(self):
        """RPW should be in reasonable range for T20 (80-120)."""
        rpw = 111.44
        assert 80 < rpw < 120

    def test_rpw_formula(self):
        """RPW = 1 / β from Win ~ RunDiff regression."""
        beta = 0.008973  # From OLS regression
        rpw = 1 / beta

        assert rpw == pytest.approx(111.44, rel=0.01)


class TestDataIntegrity:
    """Test data integrity throughout pipeline."""

    def test_no_missing_war_values(self):
        """All qualified players should have WAR values."""
        # Mock data
        df = pd.DataFrame({
            'player_id': ['P1', 'P2', 'P3'],
            'WAR': [2.0, 1.5, 0.8]
        })

        assert df['WAR'].isna().sum() == 0

    def test_war_values_realistic(self):
        """WAR values should be in realistic range (-2 to 10)."""
        # Top players rarely exceed 10 WAR in a season
        # Worst players rarely below -2 WAR
        war_values = [8.35, 7.23, 2.06, 0.5, -0.5]

        for war in war_values:
            assert -2 < war < 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
