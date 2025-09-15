import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time

from sim_v1_2 import SimulationEngine, SimulationConfig, MatchState, ResultAggregator
from .loaders import BettingOddsLoader


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
    
    # Metadata
    n_simulations: int
    simulation_time: float


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
    avg_edge: float  # Average absolute edge
    profitable_bets: int  # Count where model has positive edge

    # Actual betting performance
    total_pnl: float  # Total profit/loss
    roi: float  # Return on investment
    win_rate: float  # Percentage of winning bets
    
    # Per match results for detailed analysis
    match_results: List[MatchEvaluationResult]
    
    # Summary stats
    total_simulation_time: float


class MatchLevelEvaluator:
    """Evaluates match predictions against betting odds"""
    
    def __init__(self, model, simulation_engine: SimulationEngine, n_simulations: int = 1000):
        """
        Args:
            model: The prediction model (XGBoost, etc.)
            simulation_engine: Engine to run match simulations
            n_simulations: Number of simulations per match
        """
        self.model = model
        self.engine = simulation_engine
        self.n_simulations = n_simulations
    
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
            parallel=True,
            random_seed=42,  # Fixed for reproducibility
            verbose=False
        )
        
        sim_results = self.engine.simulate_multiple(match_state, config)
        
        # Aggregate simulation results
        aggregated = ResultAggregator.aggregate(sim_results)
        
        # Extract win probabilities
        team1 = match_state.team1
        team2 = match_state.team2
        
        simulated_win_prob = {
            team1: aggregated['win_probability'][team1],
            team2: aggregated['win_probability'][team2]
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
        log_loss = self._calculate_log_loss(simulated_win_prob, market_win_prob, team1)
        brier_score = self._calculate_brier_score(simulated_win_prob, market_win_prob, team1)
        edge = self._calculate_edge(simulated_win_prob, market_win_prob)
        realized_pnl = self._calculate_realized_pnl(edge, market_odds, actual_winner)

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
            n_simulations=self.n_simulations,
            simulation_time=simulation_time
        )
    
    def _calculate_log_loss(self, sim_prob: Dict[str, float], market_prob: Dict[str, float], 
                           team1: str) -> float:
        """Calculate log loss for binary outcome
        
        Design decision: Use team1 win as the positive class
        """
        if team1 not in market_prob:
            return np.nan
        
        # Get probabilities for team1 winning
        p_sim = sim_prob.get(team1, 0.5)
        p_market = market_prob.get(team1, 0.5)
        
        # Clip to avoid log(0)
        p_market = np.clip(p_market, 1e-15, 1 - 1e-15)
        
        # Binary cross entropy
        # Note: We're comparing our prediction to market "truth"
        log_loss = -(p_sim * np.log(p_market) + (1 - p_sim) * np.log(1 - p_market))
        
        return log_loss
    
    def _calculate_brier_score(self, sim_prob: Dict[str, float], market_prob: Dict[str, float],
                              team1: str) -> float:
        """Calculate Brier score for binary outcome"""
        if team1 not in market_prob:
            return np.nan
        
        p_sim = sim_prob.get(team1, 0.5)
        p_market = market_prob.get(team1, 0.5)
        
        # Squared difference
        return (p_sim - p_market) ** 2
    
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
        
        # If no positive edge, no bet
        if not best_team or best_edge <= 0:
            return 0.0
        
        # Calculate P&L
        if best_team == actual_winner:
            # Win: Return is (odds - 1) since stake is returned
            return float(market_odds.get(best_team, 0)) - 1.0
        else:
            # Loss: Lose the stake
            return -1.0

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
                profitable_bets=0,
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
        all_edges = []
        profitable_bets = 0
        
        # Actual betting performance
        total_pnl = 0.0
        bets_placed = 0
        winning_bets = 0

        for result in match_results:
            for team, edge in result.edge.items():
                all_edges.append(abs(edge))
                if edge > 0.05:  # 5% edge threshold
                    profitable_bets += 1
            # Track actual P&L
            if result.realized_pnl is not None:
                if result.realized_pnl != 0:  # A bet was placed
                    total_pnl += result.realized_pnl
                    bets_placed += 1
                    if result.realized_pnl > 0:
                        winning_bets += 1

        avg_edge = np.mean(all_edges) if all_edges else 0.0
        roi = (total_pnl / bets_placed * 100) if bets_placed > 0 else 0.0
        win_rate = (winning_bets / bets_placed) if bets_placed > 0 else 0.0

        return OverallEvaluationResults(
            n_matches=len(match_results),
            avg_log_loss=avg_log_loss,
            avg_brier_score=avg_brier_score,
            calibration_bins=calibration_bins,
            avg_edge=avg_edge,
            profitable_bets=profitable_bets,
            total_pnl=total_pnl,
            roi=roi,
            win_rate=win_rate,
            match_results=match_results,
            total_simulation_time=total_time
        )
    
    def _calculate_calibration(self, match_results: List[MatchEvaluationResult], 
                              n_bins: int = 10) -> List[Tuple[float, float, int]]:
        """Calculate calibration statistics
        
        For each probability bin, what fraction actually won?
        Note: We can't know actual outcomes without match results
        
        Design decision: For now, return binned comparison against market
        In production, would need actual match outcomes
        """
        bins = np.linspace(0, 1, n_bins + 1)
        calibration_data = []
        
        # Collect all predictions
        predictions = []
        for result in match_results:
            for team, sim_prob in result.simulated_win_prob.items():
                if team in result.market_win_prob:
                    predictions.append((sim_prob, result.market_win_prob[team]))
        
        # Bin predictions
        for i in range(n_bins):
            bin_mask = (np.array([p[0] for p in predictions]) >= bins[i]) & \
                      (np.array([p[0] for p in predictions]) < bins[i + 1])
            
            bin_predictions = [p for j, p in enumerate(predictions) if bin_mask[j]]
            
            if bin_predictions:
                avg_predicted = np.mean([p[0] for p in bin_predictions])
                avg_market = np.mean([p[1] for p in bin_predictions])
                count = len(bin_predictions)
                
                calibration_data.append((avg_predicted, avg_market, count))
        
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
    print(f"Average Log Loss: {results.avg_log_loss:.4f}")
    print(f"Average Brier Score: {results.avg_brier_score:.4f}")
    print(f"Average Edge: {results.avg_edge:.1%}")
    print(f"Profitable opportunities (>5% edge): {results.profitable_bets}")
    
    print(f"\n--- Actual Betting Performance ---")
    print(f"Total P&L: {results.total_pnl:+.2f} units")
    print(f"ROI: {results.roi:+.1f}%")
    print(f"Win Rate: {results.win_rate:.1%}")
    print(f"Bets Placed: {int(results.total_pnl != 0)}")  # Count of matches with bets

    print(f"\n--- Calibration Analysis ---")
    print("Predicted vs Market probabilities:")
    for pred, market, count in results.calibration_bins:
        if count > 0:
            print(f"  Predicted: {pred:.1%}, Market: {market:.1%} (n={count})")
    
    print(f"\n--- Top Value Bets ---")
    # Sort by edge
    sorted_matches = sorted(results.match_results, 
                          key=lambda x: max(x.edge.values()), 
                          reverse=True)
    
    for match in sorted_matches[:5]:
        best_team = max(match.edge, key=match.edge.get)
        edge = match.edge[best_team]
        if edge > 0:
            print(f"  {match.match_id}")
            print(f"    Bet on: {best_team}")
            print(f"    Model: {match.simulated_win_prob[best_team]:.1%}, "
                  f"Market: {match.market_win_prob.get(best_team, 0):.1%}, "
                  f"Edge: {edge:.1%}")
            if match.actual_winner:
                outcome = "WON" if match.actual_winner == best_team else "LOST"
                pnl = match.realized_pnl if match.realized_pnl is not None else 0
                print(f"    Result: {outcome} (Actual winner: {match.actual_winner}, P&L: {pnl:+.2f})")