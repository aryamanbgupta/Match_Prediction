You are one iteration of an unattended overnight research loop on this repo.

Read `program.md` at the repo root and follow its PROTOCOL exactly, top to
bottom. Non-negotiables for this session:

- ONE idea only, from `research/IDEAS.md`. When it is logged and committed, stop.
- Obey every rule in the DO NOT CHEAT section. In particular: never touch
  `data/golden/`, never modify the eval framework, report eval numbers
  verbatim.
- Verdicts use the dual-metric rule: both gate metrics improve → LANDED;
  exactly one → TABLED (revert code, keep the idea for future combinations);
  neither → FAILED (revert code).
- Never `git push`. Never `git reset`. Discard only via `git revert`.
- If the queue has no PENDING ideas, follow PROTOCOL step 1 (combine TABLED
  ideas, or write the night summary and `touch research/STOP`).
