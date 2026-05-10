import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import time

from sim_v1_2 import SimulationEngine, SimulationConfig, MatchState, ResultAggregator
from .loaders import BettingOddsLoader

# Betting configuration
BET_EDGE_THRESHOLD = 0.0  # Minimum edge required to place bet (0 = any positive edge)


@dataclass
class MatchEvaluationResult:
    """Results for a single match evaluation"""
    match_id: str
    team1: str
    team2: str

    # Simulation results
    simulated_win_prob: Dict[str, float]  # team -> probability
    simulated_scores: Dict[str, Dict[str, float]]  # team -> {mean, std, percentiles}

    # Betting comparison
    market_win_prob: Dict[str, float]  # team -> implied probability
    market_odds: Dict[str, float]  # team -> decimal odds

    actual_winner: Optional[str]

    # Metrics
    log_loss: float  # Single match log loss
    brier_score: float  # Single match brier score
    edge: Dict[str, float]  # team -> edge over market

    realized_pnl: Optional[float]

    # Kelly Criterion and EV metrics
    expected_value: float = 0.0  # Expected value of the bet
    full_kelly_fraction: float = 0.0  # Full Kelly optimal stake
    fractional_kelly_fraction: float = 0.0  # 25% Kelly stake
    full_kelly_pnl: Optional[float] = None  # P&L with Full Kelly
    fractional_kelly_pnl: Optional[float] = None  # P&L with Fractional Kelly

    # Metadata
    n_simulations: int = 0
    simulation_time: float = 0.0


@dataclass
class OverallEvaluationResults:
    """Aggregated results across all matches"""
    n_matches: int

    # Overall metrics
    avg_log_loss: float
    avg_brier_score: float

    # Calibration data
    calibration_bins: List[Tuple[float, float, int]]  # (predicted, actual, count)

    # Edge analysis
    avg_edge: float  # Average absolute edge (magnitude of disagreement)
    avg_signed_edge: float  # Average signed edge (positive = overconfident, negative = underconfident)
    profitable_bets: int  # Count where model has positive edge

    # Actual betting performance (flat staking)
    total_pnl: float  # Total profit/loss
    roi: float  # Return on investment
    win_rate: float  # Percentage of winning bets
    bets_placed: int  # Number of bets actually placed

    # Kelly Criterion performance
    total_expected_value: float = 0.0  # Sum of all EVs
    full_kelly_total_pnl: float = 0.0  # Total P&L with Full Kelly
    full_kelly_roi: float = 0.0  # Full Kelly ROI
    full_kelly_win_rate: float = 0.0  # Full Kelly win rate
    full_kelly_bets_placed: int = 0  # Bets placed with Full Kelly
    fractional_kelly_total_pnl: float = 0.0  # Total P&L with Fractional Kelly
    fractional_kelly_roi: float = 0.0  # Fractional Kelly ROI
    fractional_kelly_win_rate: float = 0.0  # Fractional Kelly win rate
    fractional_kelly_bets_placed: int = 0  # Bets placed with Fractional Kelly

    # Risk-adjusted returns
    sharpe_ratio_flat: float = 0.0  # Sharpe ratio for flat staking
    sharpe_ratio_full_kelly: float = 0.0  # Sharpe ratio for Full Kelly
    sharpe_ratio_fractional_kelly: float = 0.0  # Sharpe ratio for Fractional Kelly

    # Performance by match type
    favorite_stats: Optional[Dict] = None  # Stats for favorites (odds < 2.0)
    underdog_stats: Optional[Dict] = None  # Stats for underdogs (odds >= 2.0)

    # Per match results for detailed analysis
    match_results: List[MatchEvaluationResult] = field(default_factory=list)

    # Summary stats
    total_simulation_time: float = 0.0

    # Calibration comparison (populated when --calibrate is used)
    calibration_method: Optional[str] = None
    pre_calibration_ece: Optional[float] = None
    post_calibration_ece: Optional[float] = None
    pre_calibration_log_loss: Optional[float] = None
    post_calibration_log_loss: Optional[float] = None
    pre_calibration_brier: Optional[float] = None
    post_calibration_brier: Optional[float] = None

    # Bootstrap 95% CIs (percentile method) — populated by _aggregate_results.
    avg_log_loss_ci_low: float = float('nan')
    avg_log_loss_ci_high: float = float('nan')
    flat_roi_ci_low: float = float('nan')
    flat_roi_ci_high: float = float('nan')


class MatchLevelEvaluator:
    """Evaluates match predictions against betting odds"""
    
    def __init__(self, model, simulation_engine: SimulationEngine,
                 n_simulations: int = 1000, parallel: bool = True,
                 bootstrap_resamples: int = 1000):
        """
        Args:
            model: The prediction model (XGBoost, etc.)
            simulation_engine: Engine to run match simulations
            n_simulations: Number of simulations per match
            parallel: Enable parallel processing for simulations
            bootstrap_resamples: Resamples used for percentile-method 95% CIs
                on log-loss / flat-betting ROI.
        """
        self.model = model
        self.engine = simulation_engine
        self.n_simulations = n_simulations
        self.parallel = parallel
        self.bootstrap_resamples = bootstrap_resamples
    
    def evaluate_all(self, matches: List[Tuple[str, MatchState]], 
                     odds_lookup: Dict[str, Dict]) -> OverallEvaluationResults:
        """Evaluate all matches against betting odds
        
        Design decisions:
        - Process matches sequentially (could parallelize later)
        - Skip matches without odds rather than fail
        - Aggregate metrics incrementally
        """
        match_results = []
        total_time = 0
        
        print(f"\nEvaluating {len(matches)} matches with {self.n_simulations} simulations each...")
        
        for i, (match_id, match_state) in enumerate(matches):
            print(f"\n[{i+1}/{len(matches)}] Evaluating {match_id}")
            
            # Check if we have odds for this match
            if match_id not in odds_lookup:
                print(f"  Warning: No odds found for {match_id}, skipping...")
                continue
            
            # Evaluate single match
            try:
                result = self._evaluate_single_match(match_id, match_state, odds_lookup[match_id])
                match_results.append(result)
                total_time += result.simulation_time
                
                # Print summary
                print(f"  Simulated: {match_state.team1} {result.simulated_win_prob[match_state.team1]:.1%} vs "
                      f"{match_state.team2} {result.simulated_win_prob[match_state.team2]:.1%}")
                print(f"  Market:    {match_state.team1} {result.market_win_prob.get(match_state.team1, 0):.1%} vs "
                      f"{match_state.team2} {result.market_win_prob.get(match_state.team2, 0):.1%}")
                if result.actual_winner:
                    print(f"  Actual Winner: {result.actual_winner}")
                print(f"  Log Loss: {result.log_loss:.3f}, Edge: {max(result.edge.values()):.1%}")
                
            except Exception as e:
                print(f"  Error evaluating match: {e}")
                continue
        
        # Aggregate results
        overall_results = self._aggregate_results(match_results, total_time)

        return overall_results

    def evaluate_all_with_calibration(self, matches: List[Tuple[str, MatchState]],
                                       odds_lookup: Dict[str, Dict],
                                       calibration_method: str = 'platt') -> OverallEvaluationResults:
        """Two-pass evaluation: simulate all, then calibrate, then compute metrics.

        Pass 1: Run all simulations and collect raw win probabilities.
        Pass 2: Fit LOOCV calibration, recompute metrics with calibrated probs.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from calibration import PlattCalibrator, IsotonicCalibrator, compute_ece

        print(f"\nEvaluating {len(matches)} matches with {self.n_simulations} sims each "
              f"(+ {calibration_method} calibration)...")

        # ── Pass 1: run all simulations, collect raw probabilities ──
        raw_data = []  # list of dicts with match info + raw probs
        total_time = 0

        for i, (match_id, match_state) in enumerate(matches):
            print(f"\n[{i+1}/{len(matches)}] Simulating {match_id}")

            if match_id not in odds_lookup:
                print(f"  Warning: No odds found for {match_id}, skipping...")
                continue

            try:
                start_time = time.time()
                config = SimulationConfig(
                    n_simulations=self.n_simulations,
                    parallel=self.parallel,
                    random_seed=42,
                    verbose=False
                )
                sim_results = self.engine.simulate_multiple(match_state, config)
                aggregated = ResultAggregator.aggregate(sim_results)
                sim_time = time.time() - start_time
                total_time += sim_time

                team1 = match_state.team1
                team2 = match_state.team2

                # Normalize to exclude ties
                t1_raw = aggregated['win_probability'][team1]
                t2_raw = aggregated['win_probability'][team2]
                total_prob = t1_raw + t2_raw
                if total_prob > 0:
                    t1_norm = t1_raw / total_prob
                else:
                    t1_norm = 0.5

                odds_data = odds_lookup[match_id]
                market_odds = odds_data.get('odds', {}).get('winner', {})
                market_win_prob = BettingOddsLoader.get_implied_probabilities(market_odds)
                actual_winner = odds_data.get('actual_winner')

                print(f"  Raw sim: {team1} {t1_norm:.1%} vs {team2} {1-t1_norm:.1%}")
                if actual_winner:
                    print(f"  Actual: {actual_winner}")

                raw_data.append({
                    'match_id': match_id,
                    'match_state': match_state,
                    'team1': team1,
                    'team2': team2,
                    't1_prob_raw': t1_norm,
                    'odds_data': odds_data,
                    'market_odds': market_odds,
                    'market_win_prob': market_win_prob,
                    'actual_winner': actual_winner,
                    'aggregated': aggregated,
                    'sim_time': sim_time,
                })
            except Exception as e:
                print(f"  Error simulating match: {e}")
                continue

        if not raw_data:
            print("No matches with odds to evaluate!")
            return self._aggregate_results([], total_time)

        # ── Calibration ──
        # Build arrays: raw team1 probability + binary outcome
        raw_probs = np.array([d['t1_prob_raw'] for d in raw_data])
        has_outcome = np.array([d['actual_winner'] is not None for d in raw_data])
        actual_outcomes = np.array([
            1.0 if d['actual_winner'] == d['team1'] else 0.0
            for d in raw_data
        ])

        # Only calibrate on matches with known outcomes
        cal_mask = has_outcome
        cal_probs = raw_probs[cal_mask]
        cal_outcomes = actual_outcomes[cal_mask]

        print(f"\n--- Calibration ({calibration_method}) ---")
        print(f"  Matches with outcomes: {cal_mask.sum()}")
        print(f"  Raw prob range: [{raw_probs.min():.3f}, {raw_probs.max():.3f}]")
        print(f"  Raw prob std: {raw_probs.std():.4f}")

        if raw_probs.std() < 0.02:
            print(f"  WARNING: Probability spread too low for calibration to be effective.")

        # Compute pre-calibration ECE
        pre_ece = compute_ece(cal_probs, cal_outcomes, n_bins=7, strategy='quantile')
        pre_log_loss = float(-np.mean(
            cal_outcomes * np.log(np.clip(cal_probs, 1e-15, 1)) +
            (1 - cal_outcomes) * np.log(np.clip(1 - cal_probs, 1e-15, 1))
        ))
        pre_brier = float(np.mean((cal_probs - cal_outcomes) ** 2))
        print(f"  Pre-calibration ECE: {pre_ece:.4f}")
        print(f"  Pre-calibration Log Loss: {pre_log_loss:.4f}")
        print(f"  Pre-calibration Brier: {pre_brier:.4f}")

        # Fit LOOCV calibration on matches with outcomes
        if calibration_method == 'isotonic':
            calibrator = IsotonicCalibrator()
        else:
            calibrator = PlattCalibrator()

        calibrated_cal = calibrator.fit_loocv(cal_probs, cal_outcomes)

        # Build calibrated array for ALL matches:
        # - Matches with outcomes: use LOOCV predictions (unbiased)
        # - Matches without outcomes: use full-data calibrator
        calibrated_all = calibrator.predict(raw_probs)
        # Overwrite outcome-matches with their LOOCV predictions
        cal_indices = np.where(cal_mask)[0]
        for j, idx in enumerate(cal_indices):
            calibrated_all[idx] = calibrated_cal[j]

        post_ece = compute_ece(calibrated_cal, cal_outcomes, n_bins=7, strategy='quantile')
        post_log_loss = float(-np.mean(
            cal_outcomes * np.log(np.clip(calibrated_cal, 1e-15, 1)) +
            (1 - cal_outcomes) * np.log(np.clip(1 - calibrated_cal, 1e-15, 1))
        ))
        post_brier = float(np.mean((calibrated_cal - cal_outcomes) ** 2))
        print(f"  Post-calibration ECE: {post_ece:.4f}")
        print(f"  Post-calibration Log Loss: {post_log_loss:.4f}")
        print(f"  Post-calibration Brier: {post_brier:.4f}")

        if isinstance(calibrator, PlattCalibrator):
            print(f"  Platt params: a={calibrator.a:.4f}, b={calibrator.b:.4f}")
        print(f"  ECE improvement: {pre_ece - post_ece:+.4f}")

        # ── Pass 2: build MatchEvaluationResults with calibrated probs ──
        match_results = []
        cal_idx = 0  # index into calibrated_cal for matches with outcomes

        for i, d in enumerate(raw_data):
            # Get calibrated team1 probability
            t1_cal = float(calibrated_all[i])
            t2_cal = 1.0 - t1_cal

            # Clip to [5%, 95%]
            PROB_FLOOR = 0.05
            PROB_CEILING = 0.95
            t1_cal = max(PROB_FLOOR, min(PROB_CEILING, t1_cal))
            t2_cal = max(PROB_FLOOR, min(PROB_CEILING, t2_cal))
            clip_total = t1_cal + t2_cal
            t1_cal /= clip_total
            t2_cal /= clip_total

            simulated_win_prob = {d['team1']: t1_cal, d['team2']: t2_cal}

            simulated_scores = {
                d['team1']: d['aggregated']['score_stats'][d['team1']],
                d['team2']: d['aggregated']['score_stats'][d['team2']],
            }

            log_loss = self._calculate_log_loss(simulated_win_prob, d['actual_winner'],
                                                 d['team1'], d['team2'])
            brier_score = self._calculate_brier_score(simulated_win_prob, d['actual_winner'],
                                                       d['team1'], d['team2'])
            edge = self._calculate_edge(simulated_win_prob, d['market_win_prob'])
            realized_pnl = self._calculate_realized_pnl(edge, d['market_odds'], d['actual_winner'])

            # Kelly and EV
            best_team = None
            best_edge = 0.0
            for team, team_edge in edge.items():
                if team_edge > best_edge:
                    best_edge = team_edge
                    best_team = team

            if best_team and best_edge > BET_EDGE_THRESHOLD and best_team in d['market_odds']:
                win_prob = simulated_win_prob[best_team]
                odds = d['market_odds'][best_team]
                expected_value = self._calculate_expected_value(win_prob, odds)
                full_kelly_fraction = self._calculate_kelly_fraction(win_prob, odds)
                full_kelly_pnl = self._calculate_kelly_pnl(full_kelly_fraction, odds, best_team, d['actual_winner'])
                fractional_kelly_fraction = full_kelly_fraction * 0.25
                fractional_kelly_pnl = self._calculate_kelly_pnl(fractional_kelly_fraction, odds, best_team, d['actual_winner'])
            else:
                expected_value = 0.0
                full_kelly_fraction = 0.0
                fractional_kelly_fraction = 0.0
                full_kelly_pnl = None
                fractional_kelly_pnl = None

            result = MatchEvaluationResult(
                match_id=d['match_id'],
                team1=d['team1'],
                team2=d['team2'],
                simulated_win_prob=simulated_win_prob,
                simulated_scores=simulated_scores,
                market_win_prob=d['market_win_prob'],
                market_odds=d['market_odds'],
                actual_winner=d['actual_winner'],
                log_loss=log_loss,
                brier_score=brier_score,
                edge=edge,
                realized_pnl=realized_pnl,
                expected_value=expected_value,
                full_kelly_fraction=full_kelly_fraction,
                fractional_kelly_fraction=fractional_kelly_fraction,
                full_kelly_pnl=full_kelly_pnl,
                fractional_kelly_pnl=fractional_kelly_pnl,
                n_simulations=self.n_simulations,
                simulation_time=d['sim_time'],
            )
            match_results.append(result)

        overall = self._aggregate_results(match_results, total_time)

        # Attach calibration comparison data
        overall.calibration_method = calibration_method
        overall.pre_calibration_ece = pre_ece
        overall.post_calibration_ece = post_ece
        overall.pre_calibration_log_loss = pre_log_loss
        overall.post_calibration_log_loss = post_log_loss
        overall.pre_calibration_brier = pre_brier
        overall.post_calibration_brier = post_brier

        return overall

    def _evaluate_single_match(self, match_id: str, match_state: MatchState,
                              odds_data: Dict) -> MatchEvaluationResult:
        """Evaluate a single match
        
        Design decisions:
        - Run all simulations at once for efficiency
        - Calculate multiple metrics for comprehensive evaluation
        - Store raw results for later analysis
        """
        start_time = time.time()
        
        # Run simulations
        config = SimulationConfig(
            n_simulations=self.n_simulations,
            parallel=self.parallel,
            random_seed=42,  # Fixed for reproducibility
            verbose=False
        )
        
        sim_results = self.engine.simulate_multiple(match_state, config)
        
        # Aggregate simulation results
        aggregated = ResultAggregator.aggregate(sim_results)
        
        # Extract win probabilities
        team1 = match_state.team1
        team2 = match_state.team2

        # Get raw probabilities from simulation
        team1_prob_raw = aggregated['win_probability'][team1]
        team2_prob_raw = aggregated['win_probability'][team2]

        # Step 1: Normalize to exclude ties (so probabilities sum to 1.0)
        total = team1_prob_raw + team2_prob_raw
        if total > 0:
            team1_prob_norm = team1_prob_raw / total
            team2_prob_norm = team2_prob_raw / total
        else:
            # Fallback for edge case (all ties)
            team1_prob_norm = 0.5
            team2_prob_norm = 0.5

        # Step 2: Clip to avoid extreme predictions (5%-95%)
        PROB_FLOOR = 0.05
        PROB_CEILING = 0.95
        team1_prob = max(PROB_FLOOR, min(PROB_CEILING, team1_prob_norm))
        team2_prob = max(PROB_FLOOR, min(PROB_CEILING, team2_prob_norm))

        # Re-normalize after clipping to ensure they sum to 1.0
        clip_total = team1_prob + team2_prob
        simulated_win_prob = {
            team1: team1_prob / clip_total,
            team2: team2_prob / clip_total
        }
        
        # Extract score statistics
        simulated_scores = {
            team1: aggregated['score_stats'][team1],
            team2: aggregated['score_stats'][team2]
        }
        
        # Get market probabilities
        market_odds = odds_data.get('odds', {}).get('winner', {})
        market_win_prob = BettingOddsLoader.get_implied_probabilities(market_odds)
        actual_winner = odds_data.get('actual_winner')

        # Calculate metrics
        log_loss = self._calculate_log_loss(simulated_win_prob, actual_winner, team1, team2)
        brier_score = self._calculate_brier_score(simulated_win_prob, actual_winner, team1, team2)
        edge = self._calculate_edge(simulated_win_prob, market_win_prob)
        realized_pnl = self._calculate_realized_pnl(edge, market_odds, actual_winner)

        # Calculate Kelly Criterion and EV metrics
        best_team = None
        best_edge = 0.0
        for team, team_edge in edge.items():
            if team_edge > best_edge:
                best_edge = team_edge
                best_team = team

        # Only calculate Kelly/EV if we have a positive edge bet
        if best_team and best_edge > BET_EDGE_THRESHOLD and best_team in market_odds:
            win_prob = simulated_win_prob[best_team]
            odds = market_odds[best_team]

            # Expected value
            expected_value = self._calculate_expected_value(win_prob, odds)

            # Full Kelly
            full_kelly_fraction = self._calculate_kelly_fraction(win_prob, odds)
            full_kelly_pnl = self._calculate_kelly_pnl(full_kelly_fraction, odds, best_team, actual_winner)

            # Fractional Kelly (25%)
            fractional_kelly_fraction = full_kelly_fraction * 0.25
            fractional_kelly_pnl = self._calculate_kelly_pnl(fractional_kelly_fraction, odds, best_team, actual_winner)
        else:
            expected_value = 0.0
            full_kelly_fraction = 0.0
            fractional_kelly_fraction = 0.0
            full_kelly_pnl = None
            fractional_kelly_pnl = None

        simulation_time = time.time() - start_time

        return MatchEvaluationResult(
            match_id=match_id,
            team1=team1,
            team2=team2,
            simulated_win_prob=simulated_win_prob,
            simulated_scores=simulated_scores,
            market_win_prob=market_win_prob,
            market_odds=market_odds,
            actual_winner=actual_winner,
            log_loss=log_loss,
            brier_score=brier_score,
            edge=edge,
            realized_pnl=realized_pnl,
            expected_value=expected_value,
            full_kelly_fraction=full_kelly_fraction,
            fractional_kelly_fraction=fractional_kelly_fraction,
            full_kelly_pnl=full_kelly_pnl,
            fractional_kelly_pnl=fractional_kelly_pnl,
            n_simulations=self.n_simulations,
            simulation_time=simulation_time
        )
    
    def _calculate_log_loss(self, sim_prob: Dict[str, float], actual_winner: Optional[str],
                           team1: str, team2: str) -> float:
        """Calculate binary log loss against actual match outcome

        Log loss = -log(p) where p is predicted probability of actual outcome
        Lower is better (0 = perfect prediction)

        Args:
            sim_prob: Simulated win probabilities {team: probability}
            actual_winner: Name of team that actually won
            team1, team2: Team names for validation
        """
        if not actual_winner:
            return np.nan

        # Get predicted probability for the team that actually won
        p_predicted = sim_prob.get(actual_winner, 0.5)

        # Clip to avoid log(0)
        p_predicted = np.clip(p_predicted, 1e-15, 1 - 1e-15)

        # Binary log loss: -log(probability of actual outcome)
        log_loss = -np.log(p_predicted)

        return log_loss
    
    def _calculate_brier_score(self, sim_prob: Dict[str, float], actual_winner: Optional[str],
                              team1: str, team2: str) -> float:
        """Calculate Brier score against actual match outcome

        Brier score = (p - actual)^2 where actual is 0 or 1
        Lower is better (0 = perfect prediction)

        For team1: if team1 won, actual=1, else actual=0
        """
        if not actual_winner:
            return np.nan

        # Get predicted probability for team1
        p_team1 = sim_prob.get(team1, 0.5)

        # Actual outcome: 1 if team1 won, 0 if team1 lost
        actual = 1.0 if actual_winner == team1 else 0.0

        # Brier score: squared difference between prediction and actual
        return (p_team1 - actual) ** 2
    
    def _calculate_edge(self, sim_prob: Dict[str, float], market_prob: Dict[str, float]) -> Dict[str, float]:
        """Calculate edge over market for each team
        
        Edge = Our probability - Market probability
        Positive edge suggests value bet
        """
        edge = {}
        for team in sim_prob:
            if team in market_prob:
                edge[team] = sim_prob[team] - market_prob[team]
            else:
                edge[team] = 0.0
        
        return edge
    
    def _calculate_realized_pnl(self, edge: Dict[str, float], market_odds: Dict[str, float], 
                               actual_winner: Optional[str]) -> Optional[float]:
        """Calculate realized profit/loss if betting on positive edge
        
        Assumes unit stake on team with highest positive edge
        Returns profit/loss amount (e.g., +1.5 for 150% return, -1.0 for total loss)
        """
        if not actual_winner or not edge or not market_odds:
            return None
        
        # Find team with highest positive edge
        best_team = None
        best_edge = 0.0

        for team, team_edge in edge.items():
            if team_edge > best_edge:
                best_edge = team_edge
                best_team = team

        # If edge doesn't meet threshold, no bet
        if not best_team or best_edge <= BET_EDGE_THRESHOLD:
            return 0.0
        
        # Calculate P&L
        if best_team == actual_winner:
            # Win: Return is (odds - 1) since stake is returned
            return float(market_odds.get(best_team, 0)) - 1.0
        else:
            # Loss: Lose the stake
            return -1.0

    def _calculate_expected_value(self, win_prob: float, odds: float) -> float:
        """Calculate expected value of a bet

        EV = (win_prob × profit) - (loss_prob × stake)
           = (p × (odds - 1)) - ((1 - p) × 1)
           = p × odds - 1

        Args:
            win_prob: Our estimated win probability
            odds: Decimal odds for the bet

        Returns:
            Expected value in units (e.g., 0.15 = 15% expected return)
        """
        if odds <= 1.0:
            return 0.0

        # EV = probability × profit - (1 - probability) × loss
        profit = odds - 1.0  # Net profit if win (odds - stake)
        loss = 1.0  # Lose the stake

        ev = (win_prob * profit) - ((1 - win_prob) * loss)
        return ev

    def _calculate_kelly_fraction(self, win_prob: float, odds: float) -> float:
        """Calculate Kelly Criterion optimal stake fraction

        Kelly formula: f* = (bp - q) / b
        where:
            b = odds - 1 (net odds)
            p = win probability
            q = 1 - p (loss probability)

        Args:
            win_prob: Our estimated win probability
            odds: Decimal odds for the bet

        Returns:
            Optimal fraction of bankroll to bet (e.g., 0.15 = 15%)
            Returns 0 if no edge or negative Kelly
        """
        if odds <= 1.0 or win_prob <= 0 or win_prob >= 1:
            return 0.0

        b = odds - 1.0  # Net odds
        p = win_prob
        q = 1.0 - p

        # Kelly fraction
        kelly = (b * p - q) / b

        # Only bet if Kelly is positive (we have edge)
        if kelly <= 0:
            return 0.0

        # Return full Kelly (no cap as requested)
        return kelly

    def _calculate_kelly_pnl(self, kelly_fraction: float, odds: float,
                             bet_team: str, actual_winner: Optional[str]) -> Optional[float]:
        """Calculate P&L for a Kelly-sized bet

        Args:
            kelly_fraction: Fraction of bankroll to bet
            odds: Decimal odds
            bet_team: Team we bet on
            actual_winner: Team that actually won

        Returns:
            P&L as fraction of bankroll (e.g., 0.15 = 15% gain, -0.10 = 10% loss)
        """
        if kelly_fraction <= 0 or not actual_winner:
            return None

        # Calculate P&L
        if bet_team == actual_winner:
            # Win: profit = stake × (odds - 1)
            return kelly_fraction * (odds - 1.0)
        else:
            # Loss: lose the stake
            return -kelly_fraction

    def _bootstrap_ci(self, values: List[float], n_resamples: int = None,
                      ci: float = 0.95, seed: int = 42,
                      strata: Optional[List] = None) -> Tuple[float, float]:
        """Percentile-method bootstrap CI for the mean of `values`.

        If `strata` is provided (one label per value), resampling happens
        within each stratum, preserving stratum sizes — a standard
        stratified bootstrap. This widens the CI when within-stratum
        variance is lower than between-stratum variance, which is the
        honest framing for cross-tier / cross-period match samples.

        Returns (low, high). Returns (nan, nan) on empty input.
        """
        if not values:
            return (float('nan'), float('nan'))
        if n_resamples is None:
            n_resamples = self.bootstrap_resamples
        if n_resamples <= 0:
            return (float('nan'), float('nan'))

        arr = np.asarray(values, dtype=float)
        n = len(arr)
        rng = np.random.default_rng(seed)

        if strata is None:
            idx = rng.integers(0, n, size=(n_resamples, n))
            means = arr[idx].mean(axis=1)
        else:
            if len(strata) != n:
                raise ValueError(
                    f"strata length {len(strata)} does not match values "
                    f"length {n}")
            stratum_to_idx: Dict = {}
            for i, s in enumerate(strata):
                stratum_to_idx.setdefault(s, []).append(i)
            sums = np.zeros(n_resamples)
            for s, members in stratum_to_idx.items():
                m = len(members)
                resampled = rng.integers(0, m, size=(n_resamples, m))
                sums += arr[np.asarray(members)][resampled].sum(axis=1)
            means = sums / n
        alpha = (1 - ci) / 2
        return (
            float(np.quantile(means, alpha)),
            float(np.quantile(means, 1 - alpha)),
        )

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio for a series of returns

        Sharpe = mean(returns) / std(returns) × sqrt(n)

        Args:
            returns: List of returns (P&L values)

        Returns:
            Sharpe ratio (higher is better, >1 is good, >2 is excellent)
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)  # Sample std

        if std_return == 0:
            return 0.0

        # Sharpe ratio (not annualized since we don't have time units)
        sharpe = mean_return / std_return * np.sqrt(len(returns))

        return sharpe

    def _split_by_favorite_underdog(self, match_results: List[MatchEvaluationResult]
                                   ) -> Tuple[Dict, Dict]:
        """Split results into favorites vs underdogs and calculate stats

        Favorite: odds < 2.0 (implied probability > 50%)
        Underdog: odds >= 2.0 (implied probability <= 50%)

        Args:
            match_results: List of all match results

        Returns:
            (favorite_stats, underdog_stats) dictionaries
        """
        favorite_results = []
        underdog_results = []

        # Split matches
        for result in match_results:
            if not result.market_odds:
                continue

            # Get the team we would bet on (highest edge)
            if not result.edge:
                continue

            best_team = max(result.edge, key=result.edge.get)
            best_edge = result.edge[best_team]

            if best_edge <= BET_EDGE_THRESHOLD:
                continue  # No bet placed

            odds = result.market_odds.get(best_team, 0)

            if odds == 0:
                continue

            # Categorize
            if odds < 2.0:
                favorite_results.append((result, best_team))
            else:
                underdog_results.append((result, best_team))

        # Calculate stats for each category
        def calculate_category_stats(category_results):
            if not category_results:
                return None

            total_matches = len(category_results)
            wins = sum(1 for r, team in category_results if r.actual_winner == team)

            flat_pnl = sum(r.realized_pnl for r, _ in category_results
                          if r.realized_pnl is not None and r.realized_pnl != 0)
            flat_bets = sum(1 for r, _ in category_results
                           if r.realized_pnl is not None and r.realized_pnl != 0)

            full_kelly_pnl = sum(r.full_kelly_pnl for r, _ in category_results
                                if r.full_kelly_pnl is not None)

            edges = [r.edge[team] for r, team in category_results if r.edge]

            return {
                'n_matches': total_matches,
                'win_rate': wins / total_matches if total_matches > 0 else 0,
                'flat_roi': (flat_pnl / flat_bets * 100) if flat_bets > 0 else 0,
                'flat_pnl': flat_pnl,
                'full_kelly_pnl': full_kelly_pnl,
                'avg_edge': np.mean(edges) if edges else 0,
                'bets_placed': flat_bets
            }

        favorite_stats = calculate_category_stats(favorite_results)
        underdog_stats = calculate_category_stats(underdog_results)

        return favorite_stats, underdog_stats

    def _aggregate_results(self, match_results: List[MatchEvaluationResult], 
                          total_time: float) -> OverallEvaluationResults:
        """Aggregate individual match results
        
        Design decisions:
        - Weight all matches equally
        - Calculate calibration in bins
        - Track profitable betting opportunities
        """
        if not match_results:
            return OverallEvaluationResults(
                n_matches=0,
                avg_log_loss=np.nan,
                avg_brier_score=np.nan,
                calibration_bins=[],
                avg_edge=0.0,
                avg_signed_edge=0.0,
                profitable_bets=0,
                total_pnl=0.0,
                roi=0.0,
                win_rate=0.0,
                bets_placed=0,
                match_results=[],
                total_simulation_time=0.0
            )
        
        # Calculate averages
        log_losses = [r.log_loss for r in match_results if not np.isnan(r.log_loss)]
        brier_scores = [r.brier_score for r in match_results if not np.isnan(r.brier_score)]
        
        avg_log_loss = np.mean(log_losses) if log_losses else np.nan
        avg_brier_score = np.mean(brier_scores) if brier_scores else np.nan
        
        # Calculate calibration
        calibration_bins = self._calculate_calibration(match_results)
        
        # Edge analysis
        all_edges = []  # Absolute edges
        all_signed_edges = []  # Signed edges (correct = positive, wrong = negative)
        profitable_bets = 0  # Matches with at least one positive edge

        # Actual betting performance
        total_pnl = 0.0
        bets_placed = 0
        winning_bets = 0

        for result in match_results:
            # Count matches with ANY positive edge (not sum of teams)
            has_positive_edge = any(edge > BET_EDGE_THRESHOLD for edge in result.edge.values())
            if has_positive_edge:
                profitable_bets += 1

            # Track all edges
            for team, edge in result.edge.items():
                all_edges.append(abs(edge))  # Absolute for magnitude

                # Signed edge: positive if correct, negative if wrong
                if result.actual_winner:
                    if result.actual_winner == team:
                        all_signed_edges.append(edge)  # Correct prediction
                    else:
                        all_signed_edges.append(-edge)  # Wrong prediction

            # Track actual P&L
            if result.realized_pnl is not None:
                if result.realized_pnl != 0:  # A bet was placed
                    total_pnl += result.realized_pnl
                    bets_placed += 1
                    if result.realized_pnl > 0:
                        winning_bets += 1

        avg_edge = np.mean(all_edges) if all_edges else 0.0
        avg_signed_edge = np.mean(all_signed_edges) if all_signed_edges else 0.0
        roi = (total_pnl / bets_placed * 100) if bets_placed > 0 else 0.0
        win_rate = (winning_bets / bets_placed) if bets_placed > 0 else 0.0

        # Kelly Criterion and EV aggregation
        total_ev = 0.0
        full_kelly_pnl = 0.0
        full_kelly_wins = 0
        full_kelly_bets = 0
        fractional_kelly_pnl = 0.0
        fractional_kelly_wins = 0
        fractional_kelly_bets = 0
        flat_returns = []
        full_kelly_returns = []
        fractional_kelly_returns = []

        for result in match_results:
            # Track EV
            total_ev += result.expected_value

            # Track Full Kelly metrics
            if result.full_kelly_pnl is not None:
                full_kelly_pnl += result.full_kelly_pnl
                full_kelly_bets += 1
                full_kelly_returns.append(result.full_kelly_pnl)
                if result.full_kelly_pnl > 0:
                    full_kelly_wins += 1

            # Track Fractional Kelly metrics
            if result.fractional_kelly_pnl is not None:
                fractional_kelly_pnl += result.fractional_kelly_pnl
                fractional_kelly_bets += 1
                fractional_kelly_returns.append(result.fractional_kelly_pnl)
                if result.fractional_kelly_pnl > 0:
                    fractional_kelly_wins += 1

            # Track flat returns for Sharpe
            if result.realized_pnl is not None and result.realized_pnl != 0:
                flat_returns.append(result.realized_pnl)

        # Calculate Kelly ROIs and win rates
        full_kelly_roi = (full_kelly_pnl / full_kelly_bets * 100) if full_kelly_bets > 0 else 0.0
        full_kelly_win_rate = (full_kelly_wins / full_kelly_bets) if full_kelly_bets > 0 else 0.0
        fractional_kelly_roi = (fractional_kelly_pnl / fractional_kelly_bets * 100) if fractional_kelly_bets > 0 else 0.0
        fractional_kelly_win_rate = (fractional_kelly_wins / fractional_kelly_bets) if fractional_kelly_bets > 0 else 0.0

        # Calculate Sharpe ratios
        sharpe_flat = self._calculate_sharpe_ratio(flat_returns)
        sharpe_full_kelly = self._calculate_sharpe_ratio(full_kelly_returns)
        sharpe_fractional_kelly = self._calculate_sharpe_ratio(fractional_kelly_returns)

        # Bootstrap CIs on per-match log loss and per-bet flat P&L. ROI CI is
        # the P&L-mean CI ×100 (ROI = mean(P&L) × 100 by definition). Resamples
        # are seeded for reproducibility across the three liquidity slices.
        ll_lo, ll_hi = self._bootstrap_ci(log_losses)
        roi_lo, roi_hi = self._bootstrap_ci(flat_returns)
        flat_roi_ci_low = roi_lo * 100 if not np.isnan(roi_lo) else float('nan')
        flat_roi_ci_high = roi_hi * 100 if not np.isnan(roi_hi) else float('nan')

        # Split by favorite/underdog
        favorite_stats, underdog_stats = self._split_by_favorite_underdog(match_results)

        return OverallEvaluationResults(
            n_matches=len(match_results),
            avg_log_loss=avg_log_loss,
            avg_brier_score=avg_brier_score,
            calibration_bins=calibration_bins,
            avg_edge=avg_edge,
            avg_signed_edge=avg_signed_edge,
            profitable_bets=profitable_bets,
            total_pnl=total_pnl,
            roi=roi,
            win_rate=win_rate,
            bets_placed=bets_placed,
            total_expected_value=total_ev,
            full_kelly_total_pnl=full_kelly_pnl,
            full_kelly_roi=full_kelly_roi,
            full_kelly_win_rate=full_kelly_win_rate,
            full_kelly_bets_placed=full_kelly_bets,
            fractional_kelly_total_pnl=fractional_kelly_pnl,
            fractional_kelly_roi=fractional_kelly_roi,
            fractional_kelly_win_rate=fractional_kelly_win_rate,
            fractional_kelly_bets_placed=fractional_kelly_bets,
            sharpe_ratio_flat=sharpe_flat,
            sharpe_ratio_full_kelly=sharpe_full_kelly,
            sharpe_ratio_fractional_kelly=sharpe_fractional_kelly,
            favorite_stats=favorite_stats,
            underdog_stats=underdog_stats,
            match_results=match_results,
            total_simulation_time=total_time,
            avg_log_loss_ci_low=ll_lo,
            avg_log_loss_ci_high=ll_hi,
            flat_roi_ci_low=flat_roi_ci_low,
            flat_roi_ci_high=flat_roi_ci_high,
        )
    
    def _calculate_calibration(self, match_results: List[MatchEvaluationResult],
                              n_bins: int = 10) -> List[Tuple[float, float, int]]:
        """Calculate calibration statistics against actual outcomes

        For each probability bin, what fraction actually won?
        This measures true calibration: if model says 70%, do we win 70% of the time?

        Returns list of (predicted_prob, actual_win_rate, count) tuples
        """
        bins = np.linspace(0, 1, n_bins + 1)
        calibration_data = []

        # Collect all predictions with actual outcomes
        predictions = []  # List of (predicted_prob, did_win)
        for result in match_results:
            if not result.actual_winner:
                continue  # Skip matches without outcomes

            for team, sim_prob in result.simulated_win_prob.items():
                did_win = 1.0 if result.actual_winner == team else 0.0
                predictions.append((sim_prob, did_win))

        if not predictions:
            return []  # No matches with outcomes

        # Bin predictions
        pred_probs = np.array([p[0] for p in predictions])
        actual_wins = np.array([p[1] for p in predictions])

        for i in range(n_bins):
            bin_mask = (pred_probs >= bins[i]) & (pred_probs < bins[i + 1])

            if np.sum(bin_mask) > 0:
                avg_predicted = np.mean(pred_probs[bin_mask])
                actual_win_rate = np.mean(actual_wins[bin_mask])
                count = int(np.sum(bin_mask))

                calibration_data.append((avg_predicted, actual_win_rate, count))

        return calibration_data


def print_evaluation_summary(results: OverallEvaluationResults):
    """Pretty print evaluation results"""
    print("\n" + "="*60)
    print("MATCH LEVEL EVALUATION SUMMARY".center(60))
    print("="*60)
    
    print(f"\nMatches evaluated: {results.n_matches}")
    print(f"Total simulation time: {results.total_simulation_time:.1f}s")
    print(f"Average time per match: {results.total_simulation_time/results.n_matches:.1f}s")
    
    print(f"\n--- Performance Metrics ---")
    if not np.isnan(results.avg_log_loss_ci_low):
        print(f"Average Log Loss: {results.avg_log_loss:.4f}  "
              f"[95% CI: {results.avg_log_loss_ci_low:.4f}, {results.avg_log_loss_ci_high:.4f}]")
    else:
        print(f"Average Log Loss: {results.avg_log_loss:.4f}")
    print(f"Average Brier Score: {results.avg_brier_score:.4f}")
    print(f"Average Edge (magnitude): {results.avg_edge:.1%}")
    print(f"Average Signed Edge: {results.avg_signed_edge:+.1%} ({'overconfident' if results.avg_signed_edge < 0 else 'underconfident' if results.avg_signed_edge > 0 else 'neutral'})")
    print(f"Matches with model-market disagreement: {results.profitable_bets}/{results.n_matches}")
    
    print(f"\n--- Betting Strategy Comparison ---")
    print(f"\nFlat Staking (1 unit per bet):")
    print(f"  Total P&L: {results.total_pnl:+.2f} units")
    if not np.isnan(results.flat_roi_ci_low):
        print(f"  ROI: {results.roi:+.1f}%  "
              f"[95% CI: {results.flat_roi_ci_low:+.1f}%, {results.flat_roi_ci_high:+.1f}%]")
    else:
        print(f"  ROI: {results.roi:+.1f}%")
    print(f"  Sharpe Ratio: {results.sharpe_ratio_flat:.2f}")
    print(f"  Win Rate: {results.win_rate:.1%}")
    print(f"  Bets Placed: {results.bets_placed}")

    print(f"\nFull Kelly Criterion:")
    print(f"  Total P&L: {results.full_kelly_total_pnl:+.2f} units")
    print(f"  ROI: {results.full_kelly_roi:+.1f}%")
    print(f"  Sharpe Ratio: {results.sharpe_ratio_full_kelly:.2f}")
    print(f"  Win Rate: {results.full_kelly_win_rate:.1%}")
    print(f"  Bets Placed: {results.full_kelly_bets_placed}")

    print(f"\nFractional Kelly (25%):")
    print(f"  Total P&L: {results.fractional_kelly_total_pnl:+.2f} units")
    print(f"  ROI: {results.fractional_kelly_roi:+.1f}%")
    print(f"  Sharpe Ratio: {results.sharpe_ratio_fractional_kelly:.2f} {'⭐ BEST' if results.sharpe_ratio_fractional_kelly >= max(results.sharpe_ratio_flat, results.sharpe_ratio_full_kelly) else ''}")
    print(f"  Win Rate: {results.fractional_kelly_win_rate:.1%}")
    print(f"  Bets Placed: {results.fractional_kelly_bets_placed}")

    print(f"\n--- Expected Value Analysis ---")
    print(f"Total Expected Value: {results.total_expected_value:+.2f} units")
    flat_ev_diff = results.total_expected_value - results.total_pnl
    full_kelly_ev_diff = results.total_expected_value - results.full_kelly_total_pnl
    frac_kelly_ev_diff = results.total_expected_value - results.fractional_kelly_total_pnl
    print(f"Flat Staking (EV vs Realized): {flat_ev_diff:+.2f} {'(unlucky)' if flat_ev_diff > 1 else '(lucky)' if flat_ev_diff < -1 else '(neutral)'}")
    print(f"Full Kelly (EV vs Realized): {full_kelly_ev_diff:+.2f} {'(unlucky)' if full_kelly_ev_diff > 1 else '(lucky)' if full_kelly_ev_diff < -1 else '(neutral)'}")
    print(f"Fractional Kelly (EV vs Realized): {frac_kelly_ev_diff:+.2f} {'(unlucky)' if frac_kelly_ev_diff > 1 else '(lucky)' if frac_kelly_ev_diff < -1 else '(neutral)'}")

    print(f"\n--- Performance by Match Type ---")
    if results.favorite_stats:
        fav = results.favorite_stats
        print(f"\nFavorites (odds < 2.0):")
        print(f"  Matches: {fav['n_matches']}")
        print(f"  Win Rate: {fav['win_rate']:.1%}")
        print(f"  Flat ROI: {fav['flat_roi']:+.1f}%")
        print(f"  Full Kelly P&L: {fav['full_kelly_pnl']:+.2f} units")
        print(f"  Average Edge: {fav['avg_edge']:.1%}")
        print(f"  Bets Placed: {fav['bets_placed']}")

    if results.underdog_stats:
        und = results.underdog_stats
        print(f"\nUnderdogs (odds >= 2.0):")
        print(f"  Matches: {und['n_matches']}")
        print(f"  Win Rate: {und['win_rate']:.1%}")
        print(f"  Flat ROI: {und['flat_roi']:+.1f}%")
        print(f"  Full Kelly P&L: {und['full_kelly_pnl']:+.2f} units")
        print(f"  Average Edge: {und['avg_edge']:.1%}")
        print(f"  Bets Placed: {und['bets_placed']}")

    print(f"\n--- Calibration Analysis ---")
    print("Predicted probability vs Actual win rate:")
    print("(Perfect calibration: predicted = actual)")
    for pred, actual, count in results.calibration_bins:
        if count > 0:
            diff = actual - pred
            print(f"  Predicted: {pred:.1%}, Actual: {actual:.1%}, Diff: {diff:+.1%} (n={count})")

    # Show calibration comparison if calibration was applied
    if results.calibration_method is not None:
        print(f"\n--- Calibration Comparison ({results.calibration_method}) ---")
        print(f"                   Before    After    Change")
        print(f"  ECE:           {results.pre_calibration_ece:>7.4f}  {results.post_calibration_ece:>7.4f}  "
              f"{results.post_calibration_ece - results.pre_calibration_ece:>+7.4f}")
        print(f"  Log Loss:      {results.pre_calibration_log_loss:>7.4f}  {results.post_calibration_log_loss:>7.4f}  "
              f"{results.post_calibration_log_loss - results.pre_calibration_log_loss:>+7.4f}")
        print(f"  Brier Score:   {results.pre_calibration_brier:>7.4f}  {results.post_calibration_brier:>7.4f}  "
              f"{results.post_calibration_brier - results.pre_calibration_brier:>+7.4f}")

    print(f"\n--- Predictions by Signed Edge ---")
    print("(Positive = correct prediction, Negative = incorrect prediction)")

    def get_signed_edge(match):
        """Calculate signed edge: positive if correct, negative if wrong"""
        best_team = max(match.edge, key=match.edge.get)
        edge = match.edge[best_team]

        if not match.actual_winner:
            return None  # Skip matches without results

        # Positive edge if we predicted the winner correctly, negative if wrong
        if match.actual_winner == best_team:
            return edge  # Correct prediction
        else:
            return -edge  # Wrong prediction

    # Filter to matches with actual results and calculate signed edges
    matches_with_results = [
        (match, get_signed_edge(match))
        for match in results.match_results
        if get_signed_edge(match) is not None
    ]

    # Sort by signed edge (best correct predictions first)
    sorted_matches = sorted(matches_with_results,
                          key=lambda x: x[1],
                          reverse=True)

    # Show top 10 (or all if fewer)
    num_to_show = min(10, len(sorted_matches))

    for match, signed_edge in sorted_matches[:num_to_show]:
        best_team = max(match.edge, key=match.edge.get)
        edge = match.edge[best_team]
        is_correct = signed_edge > 0

        print(f"\n  {match.match_id}")
        print(f"    Bet on: {best_team}")
        print(f"    Model: {match.simulated_win_prob[best_team]:.1%}, "
              f"Market: {match.market_win_prob.get(best_team, 0):.1%}")
        print(f"    Edge: {edge:.1%} | Signed Edge: {signed_edge:+.1%} "
              f"({'✓ CORRECT' if is_correct else '✗ WRONG'})")

        outcome = "WON" if is_correct else "LOST"
        pnl = match.realized_pnl if match.realized_pnl is not None else 0
        print(f"    Result: {outcome} (Actual winner: {match.actual_winner}, P&L: {pnl:+.2f})")