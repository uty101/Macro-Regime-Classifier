# Allocation universe

The timed allocation is over A = {SMB, HML, RMW, CMA, UMD}; the static benchmark is 1/5 on A, rebalanced monthly.
Mkt-RF is excluded from allocation: a long-only market sleeve inside a book of self-financing long-short sleeves changes the leverage interpretation. It still appears in every conditional-statistics table.
Weights blend 1/5 with positive-part trailing conditional Sharpe shares (η ∈ {0.25, 0.5, 1.0}), falling back to 1/5 when the regime has fewer than `min_regime_obs` (24) prior observations, no positive Sharpe, or is unassigned.
Costs are c × ½ Σ|Δw| with c ∈ {10, 20, 50} bp (headline 20 bp), deducted in the month the weights earn.
Sleeve-internal turnover is not modelled and is identical across static and timed.
