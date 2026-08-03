# Women's model vs market — women's market odds joined to the I12 w1 (T20I) fixture pool

Source: `data/womens_polymarket/betting_odds_womens_w1.json`  
Capture: `/Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_match_odds_strict_female_t20-hundred_2025-07-01_2026-08-01.json`  
Generated: 2026-08-01T23:01:06Z

Coinflip LL = 0.6931. Log loss and accuracy only — no ROI (invariant 7 requires the I3 block contract).

Rows marked **!** are slices where the market itself scored worse than a coinflip. On those, beating the market is not evidence of skill — the line carries no information to beat.

| slice | n | market LL | market acc | xgb_match_w1_base LL | xgb_match_w1_base acc | xgb_match_w1_swap LL | xgb_match_w1_swap acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| test | 46 | 0.4056 | 0.739 | 0.4864 | 0.783 | 0.5126 | 0.761 |
| test >=$10,000 | 9 | 0.3869 | 0.667 | 0.6017 | 0.778 | 0.5578 | 0.667 |
| test >=$50,000 **!** | 3 | 0.7659 | 0.333 | 0.7839 | 0.667 | 0.7083 | 0.333 |
| test >=$100,000 **!** | 2 | 0.7661 | 0.500 | 0.9859 | 0.500 | 0.6517 | 0.500 |
| golden | 129 | 0.4895 | 0.775 | 0.5159 | 0.752 | 0.5085 | 0.783 |
| golden >=$10,000 | 94 | 0.5689 | 0.713 | 0.5676 | 0.713 | 0.5569 | 0.766 |
| golden >=$50,000 | 48 | 0.5093 | 0.729 | 0.5842 | 0.688 | 0.5619 | 0.771 |
| golden >=$100,000 | 38 | 0.4871 | 0.737 | 0.5676 | 0.711 | 0.5550 | 0.789 |
| ALL | 175 | 0.4674 | 0.766 | 0.5082 | 0.760 | 0.5096 | 0.777 |
| ALL >=$10,000 | 103 | 0.5530 | 0.709 | 0.5706 | 0.718 | 0.5570 | 0.757 |
| ALL >=$50,000 | 51 | 0.5244 | 0.706 | 0.5959 | 0.686 | 0.5705 | 0.745 |
| ALL >=$100,000 | 40 | 0.5010 | 0.725 | 0.5885 | 0.700 | 0.5599 | 0.775 |
