Make the outcome categories: squish/ round down to 0,1,2,4,6,W
Use the split to make pipeline and split the processed data
Make unknown player encoding (bottom 5-10%) to train for unknown/new players
Betting sim: currently the odds json generation ignored 30% of the matches, fix that.
eval functions graph generation
Add Expected Value (EV) tracking to distinguish +EV bets from lucky/unlucky outcomes
Add Sharpe Ratio calculation for risk-adjusted returns in betting evaluation
Implement Kelly Criterion for optimal bet sizing based on edge and bankroll
