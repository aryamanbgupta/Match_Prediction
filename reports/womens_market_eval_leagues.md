# Women's model vs market — women's market odds joined to the I12-L w2 (leagues) fixture pool

Source: `data/womens_polymarket_leagues/betting_odds_womens_w2.json`  
Capture: `/Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_match_odds_strict_female_t20-hundred_2025-07-01_2026-08-01.json`  
Generated: 2026-08-01T23:01:07Z

Coinflip LL = 0.6931. Log loss and accuracy only — no ROI (invariant 7 requires the I3 block contract).

Rows marked **!** are slices where the market itself scored worse than a coinflip. On those, beating the market is not evidence of skill — the line carries no information to beat.

| slice | n | market LL | market acc | xgb_match_w2_base LL | xgb_match_w2_base acc | xgb_match_w2_swap LL | xgb_match_w2_swap acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| test | 10 | 0.6739 | 0.500 | 0.7561 | 0.400 | 0.7432 | 0.600 |
| test >=$10,000 | 10 | 0.6739 | 0.500 | 0.7561 | 0.400 | 0.7432 | 0.600 |
| test >=$50,000 **!** | 4 | 0.7447 | 0.250 | 0.6917 | 0.500 | 0.6455 | 0.750 |
| test >=$100,000 **!** | 2 | 0.7394 | 0.500 | 0.8776 | 0.000 | 0.6552 | 0.500 |
| golden | 41 | 0.6853 | 0.610 | 0.6968 | 0.561 | 0.7530 | 0.488 |
| golden >=$10,000 **!** | 17 | 0.7074 | 0.529 | 0.6667 | 0.529 | 0.7114 | 0.412 |
| golden >=$50,000 | 8 | 0.6900 | 0.625 | 0.6346 | 0.500 | 0.7329 | 0.375 |
| golden >=$100,000 **!** | 6 | 0.7018 | 0.500 | 0.5956 | 0.667 | 0.7269 | 0.500 |
| ALL | 51 | 0.6831 | 0.588 | 0.7084 | 0.529 | 0.7511 | 0.510 |
| ALL >=$10,000 **!** | 27 | 0.6950 | 0.519 | 0.6998 | 0.481 | 0.7232 | 0.481 |
| ALL >=$50,000 **!** | 12 | 0.7082 | 0.500 | 0.6536 | 0.500 | 0.7038 | 0.500 |
| ALL >=$100,000 **!** | 8 | 0.7112 | 0.500 | 0.6661 | 0.500 | 0.7090 | 0.500 |
