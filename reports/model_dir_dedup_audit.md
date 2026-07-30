# Model-directory dedup and branch-hygiene audit

## V7 sweep model directories

Audited against the shipped `models/xgb_v3` directory. Sizes are allocated
directory sizes from `du -sk`, converted to MiB. “Identical to shipped” means
that the candidate's model and every encoder/calibrator pickle it contains
have the same SHA-256 as the same-named shipped artifact. Shipped-only
additions do not make an older directory unique.

| Directory | Size | Identical to shipped | Recommendation |
|---|---:|:---:|---|
| `models/xgb_v3_phase6_k10` | 130.98 MiB | No | **Keep** — unique sweep model bytes |
| `models/xgb_v3_phase6_k100` | 125.66 MiB | No | **Keep** — unique sweep model bytes |
| `models/xgb_v3_phase6_k300` | 123.57 MiB | No | **Keep** — unique sweep model bytes |
| `models/xgb_v3_v6_backup` | 127.02 MiB | No | **Keep** — unique flat-shrinkage rollback model |
| `models/xgb_v3_phase5_k30` | 128.63 MiB | Yes | **Delete** after approval — redundant subset of shipped artifacts |

No directory was deleted.

### SHA-256 evidence

The three legacy encoders are byte-identical in all five candidates and the
shipped directory:

| Artifact | Shipped SHA-256 | Candidate result |
|---|---|---|
| `batter_encoder_v3.pkl` | `94ed4c061945e0d6fd055dd46f48c2f349b8ab2651f2a482560e99c44b610e90` | 5/5 match |
| `bowler_encoder_v3.pkl` | `554c1987b5d13d4d94a84b0f118cb85ba5d41d837b0f9861861f1dbb571f89ca` | 5/5 match |
| `matchup_encoder_v3.pkl` | `051f45780937859ef6981d3e1d3afcf84277e2d02f655409b1d043a53d384ab2` | 5/5 match |

Model hashes distinguish every retained variant:

| Directory | `xgboost_model_v3.pkl` SHA-256 | Matches shipped |
|---|---|:---:|
| `models/xgb_v3` | `5400df329221d8a85f36eea793821459c39bd9fbd35a30d72e6f3900d3d491ac` | — |
| `models/xgb_v3_phase6_k10` | `803aeb215c1ffae4eb015ead4d6e14b37812eb7130434853f7a9eebe225ed338` | No |
| `models/xgb_v3_phase6_k100` | `6ca536f39f8f10cba8be62fbd8666c623449dd12f8244a87d0e7c1913d1084f7` | No |
| `models/xgb_v3_phase6_k300` | `4903f065dac6831924cb26f35135801d74e291978f8f403a1bd52e652f099266` | No |
| `models/xgb_v3_v6_backup` | `2d733e3b8bd1bec4e599c41a4851129f960cc9523cc5b0eec587e6f5c7cae939` | No |
| `models/xgb_v3_phase5_k30` | `5400df329221d8a85f36eea793821459c39bd9fbd35a30d72e6f3900d3d491ac` | Yes |

`models/xgb_v3_phase5_k30` has six files; every one is byte-identical to its
same-named shipped file, including feature columns and feature importance.
The shipped directory is a superset: its venue encoder, vector-scaling
calibrator, and outcome-distribution sidecar are absent from the Phase 5
directory. None of the five candidates contains a calibrator or venue encoder,
so there is no conflicting candidate hash for those shipped-only artifacts.

## Branch hygiene

Tip dates are commit dates. The reachability count is the requested
`git log main..<branch> --oneline | wc -l` result.

| Branch | Tip date | Tip subject | Commits not reachable from `main` | Recommendation |
|---|---|---|---:|---|
| `fixes/sim_improvements` | 2025-10-09 | Fix critical metric calculation issues | 0 | **Delete** — fully reachable from `main` |
| `feature/player-stats-cache` | 2025-10-15 | Add comprehensive documentation and update cache system references | 0 | **Delete** — fully reachable from `main` |
| `features/cricinfo-features` | 2025-12-26 | added v1 lstm model | 0 | **Delete** — fully reachable from `main` |
| `features/mlp-model` | 2025-12-31 | modified mlp | 0 | **Delete** — fully reachable from `main` |
| `features/llm-model` | 2026-02-23 | added possible improvements | 0 | **Delete** — fully reachable from `main` |
| `features/transformer-model` | 2026-03-20 | fixed eval bug and re-ran for results fo rall models | 0 | **Delete** — fully reachable from `main` |
| `backup-pre-rewrite` | 2026-05-28 | chore: stop tracking tmp/ scratch space; gitignore it | 15 | **Delete** — May rewrite verified; see below |

No branch was deleted.

### Pre-rewrite backup verification

`git cherry main backup-pre-rewrite` marks 13 of the 15 non-reachable commits
as patch-equivalent on `main`. The two unmatched commits are `07e4443` (“misc
files”) and `685e863` (“stop tracking tmp scratch space”):

- all durable paths introduced by `07e4443` are present on `main`, including
  the project overview, prop reports, parallel-run helper, and prop backtest;
- the scratch profile and `tmp/golden_inclusive/` artifacts are absent from
  `main`, as intended;
- `tmp/` is ignored on `main`.

The backup therefore preserves old hashes, but no wanted content that is
missing from the rewritten history. Its 15-count is rewrite topology rather
than evidence of an unmerged feature.
