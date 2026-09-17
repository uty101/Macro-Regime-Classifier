# Macro Regime Classifier and Factor Timing

A monthly macro regime classifier built from US rates, inflation, growth, the dollar, oil and equity volatility, estimated so that the regime at month-end t uses only data published by month-end t (ALFRED vintages for revised series).
Regimes come from an expanding-window Gaussian HMM (primary), a Gaussian mixture and a growth/inflation rules quadrant, each anchored so that state numbers mean the same thing across refits.
Regimes condition monthly Fama-French factor returns, with stationary-block-bootstrap intervals, and drive a timed allocation over SMB, HML, RMW, CMA and UMD tested out of sample after a 1-month lag and 20 bp costs.
The write-up answers four questions: how many persistent states there are, whether factor premia differ by regime beyond bootstrap noise, whether timing beats static equal weight net of lag and costs, and how much of the benefit disappears going from smoothed to filtered regimes.
Everything runs from `python -m regime.run` and `pytest`; the plan is in `PLAN.md`, the conventions in `CLAUDE.md` and `docs/CONVENTIONS_RESOLVED.md`.

## Results

Results pending.
