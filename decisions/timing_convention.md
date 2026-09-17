# Timing convention

Decision date t is the last calendar day of month t; the decision is executed at the close of the first trading day of t+1 and earns the factor return of month t+1 (lag 0) or t+2 (lag 1, the headline).
Market series use the last non-missing daily observation dated on or before t, looking back at most 10 calendar days; otherwise the month is NaN and reported.
Revised series (CPIAUCSL, INDPRO, UNRATE) use the ALFRED vintage as of t: for each observation month, the release with the greatest `realtime_start` ≤ t, and every lag of that series comes from the same vintage.
Lags are rows of the decision-date index (12 rows = "12 months ago"), never calendar-day arithmetic.
Everything downstream reads the single as-of panel from `regime/data/asof.py`; nothing downstream touches raw series.
