# Review — Section 2: Features

## Section

This is Section 2 of `PLAN.md` (steps 2.1 to 2.4): the ten raw features from the as-of panel, the DTWEXM/DTWEXBGS change splice, the real-time standardisation with the first-window and expanding rule, `model_input` with its dropped-row table, the feature sanity table and the ten-panel review chart, and section 2 of `python -m regime.run`. **Every step completed**, in one sitting on 22 September 2026, no `OPEN.md` item, no definition changed. Before starting, `data/processed/asof_panel.parquet` was rebuilt with `python -m regime.run --section 1` (no `--pull`): 439 rows × 16 columns, 1990-01-31 to 2026-07-31; section 2 reads only that file.

The headline facts: `model_input` drops exactly one date from 1991-01-31 onward, **2026-02-28**, because `cpi_m3` is NaN in the as-of-2026-02-28 vintage (October 2025 CPI was never published) — the row is reported, not patched; `dollar_chg12` at the splice date 2007-01-31 is the DTWEXBGS change, −1.4864877746844383, and the two change series correlate 0.9656 over the 156-month overlap; the first-window mean and standard deviation of every z column are 0 and 1 to 2.6e-16 and 1.2e-16 (168 rows per core column, 12 for `breakeven_chg12`); no non-positive WTI or VIX month-end exists (the 2020-04-20 negative WTI print is skipped by the 10-day month-end rule, and 2020-04-30 is 19.23), so no NaN from a log occurs; `python -m regime.run --section 2` runs in about 2 seconds and every file it writes (`features_raw.parquet`, `features_z.parquet`, `dropped_rows.csv`, `feature_sanity.csv`, `features_review.png`) is byte-identical across two consecutive runs.

One small reading the plan leaves implicit, recorded here rather than in `OPEN.md` because it changes no number: the `missing` column of `dropped_rows.csv` holds the NaN column names of that date joined by `;` (one row per dropped date). `tests/test_run.py::test_unbuilt_section_raises` now asserts on section 3, since section 2 is built; its intent (an unbuilt section raises `NotImplementedError("section N not built")`) is unchanged.

## Steps completed

- `step 2.1: build_raw_features for the ten columns, splice_dollar_chg12 and model_input with the hand-built 14-row and rows-not-days tests` — `b698ca1`
- `step 2.2: dollar splice tests, switching exactly at 2007-01-31 and with no level jump` — `4cfcc1f`
- `step 2.3: standardise with the first-window and expanding rule; spike-after-t, first-window-rule and model_input tests` — `9c39d6c`
- `step 2.4: feature_sanity table, features_review_chart and section 2 of run.py from the as-of panel` — `7fdfbff`

`splice_dollar_chg12` and `standardise` were written in step 2.1's `regime/features.py` (the plan says the splice is written in 2.1 and proven in 2.2; `standardise` sits in the same module and is proven in 2.3), so commits 2.2 and 2.3 add tests only.

## Evidence

### Step 2.1 — `raw.loc["2004-12-31"]`, `raw.loc[sample_end]`, `raw.tail(3)`, and the single dropped row 2026-02-28 with its panel row

```
$ .venv/Scripts/python.exe -c "<build_raw_features / standardise / model_input on data/processed/asof_panel.parquet>"
raw.loc[2004-12-31]
dgs10_level         4.240000
dgs10_chg12        -0.030000
slope_2s10s         1.160000
cpi_3m_ann          3.856015
indpro_chg12        3.744999
dollar_chg12       -6.568366
oil_chg12          27.272681
log_vix             2.587012
unrate_chg12       -0.500000
breakeven_chg12     0.290000
raw.loc[sample_end]
dgs10_level         4.750000
dgs10_chg12         0.380000
slope_2s10s         0.470000
cpi_3m_ann          2.783726
indpro_chg12        1.137590
dollar_chg12       -1.671452
oil_chg12          20.258111
log_vix             2.771964
unrate_chg12        0.100000
breakeven_chg12    -0.110000
raw.tail(3)
            dgs10_level  dgs10_chg12  slope_2s10s  cpi_3m_ann  indpro_chg12  dollar_chg12  oil_chg12   log_vix  unrate_chg12  breakeven_chg12
date                                                                                                                                         
2026-05-31         4.45         0.04         0.47    7.319774      1.344065     -2.401239  39.422965  2.729159           0.1             0.04
2026-06-30         4.44         0.20         0.30    8.204073      1.652191      1.261596   6.227351  2.800325           0.0            -0.05
2026-07-31         4.75         0.38         0.47    2.783726      1.137590     -1.671452  20.258111  2.771964           0.1            -0.11
dropped
         date     missing
0  2026-02-28  cpi_3m_ann
panel row 2026-02-28
dgs10                              3.97
dgs2                               3.38
dtwexbgs                       117.8223
dtwexm                              NaN
wti                               66.96
vix                               19.86
t10yie                             2.25
cpi_m                           326.588
cpi_m3                              NaN
indpro_m                       102.3412
indpro_m12                     100.0647
unrate_m                            4.3
unrate_m12                          4.0
cpi_obs_month       2026-01-31 00:00:00
indpro_obs_month    2026-01-31 00:00:00
unrate_obs_month    2026-01-31 00:00:00
raw row
dgs10_level        3.970000
dgs10_chg12       -0.270000
slope_2s10s        0.590000
cpi_3m_ann              NaN
indpro_chg12       2.249535
dollar_chg12      -8.504172
oil_chg12         -4.397115
log_vix            2.988708
unrate_chg12       0.300000
breakeven_chg12   -0.130000
model_input shape (426, 8) 1991-01-31 2026-07-31
n rows features_from..end 427
non-positive log inputs: wti<=0 0 vix<=0 0
```

The dropped date's panel row shows `cpi_m = 326.588` for observation month 2026-01 and `cpi_m3 = NaN`: the as-of-2026-02-28 vintage carries no value for October 2025. Every other raw column on that date is valid. No other date is dropped; `model_input` is 426 × 8 from 1991-01-31 to 2026-07-31 (427 rows in z, minus one).

### Step 2.2 — the 12 rows either side of 2007-01-31, the splice-date value, the overlap correlation and the five largest absolute differences

```
$ .venv/Scripts/python.exe -c "<m = ln(dtwexm / dtwexm.shift(12)) × 100; b likewise for dtwexbgs; splice_dollar_chg12(panel, cfg)>"
12 rows either side of 2007-01-31
             dtwexm  dtwexbgs    m_chg12   b_chg12  dollar_chg12
date                                                            
2006-01-31  84.1424   99.4311   2.978958       NaN      2.978958
2006-02-28  84.7180   99.7695   4.725336       NaN      4.725336
2006-03-31  85.1759  100.5600   4.100135       NaN      4.100135
2006-04-30  81.8354   98.1412  -0.672503       NaN     -0.672503
2006-05-31  80.5146   97.7705  -4.959856       NaN     -4.959856
2006-06-30  81.3415   98.2483  -4.797639       NaN     -4.797639
2006-07-31  81.5974   97.9996  -4.789256       NaN     -4.789256
2006-08-31  81.2524   97.6440  -3.158957       NaN     -3.158957
2006-09-30  81.9707   98.1054  -2.896024       NaN     -2.896024
2006-10-31  81.5503   97.3901  -4.711156       NaN     -4.711156
2006-11-30  80.3180   96.5779  -7.284576       NaN     -7.284576
2006-12-31  81.5007   97.0312  -5.405424       NaN     -5.405424
2007-01-31  82.6003   97.9640  -1.849729 -1.486488     -1.486488
2007-02-28  81.4734   97.2386  -3.905151 -2.569477     -2.569477
2007-03-31  80.6091   96.3183  -5.510698 -4.309623     -4.309623
2007-04-30  78.9872   94.8365  -3.542410 -3.425290     -3.425290
2007-05-31  79.0364   94.3215  -1.853003 -3.591374     -3.591374
2007-06-30  78.7698   94.1393  -3.212667 -4.272235     -4.272235
2007-07-31  77.7477   93.4264  -4.832843 -4.778944     -4.778944
2007-08-31  77.5001   93.6674  -4.728113 -4.157800     -4.157800
2007-09-30  74.5149   90.9446  -9.536276 -7.579188     -7.579188
2007-10-31  72.7723   89.1389 -11.388462 -8.852874     -8.852874
2007-11-30  73.1659   89.6212  -9.326429 -7.475804     -7.475804
2007-12-31  73.3779   89.6304 -10.498881 -7.933803     -7.933803
2008-01-31  72.5045   88.6547 -13.036468 -9.985102     -9.985102
dollar_chg12 at 2007-01-31 = -1.4864877746844383
overlap rows 156 2007-01-31 2019-12-31
corr m_chg12 vs b_chg12 over overlap = 0.9655842667203641
5 largest abs differences
              m_chg12    b_chg12  abs_diff
date                                      
2015-03-31  18.349387  13.105015  5.244372
2016-05-31   0.014643   5.192473  5.177830
2016-07-31  -2.075007   2.802828  4.877835
2009-08-31   0.077635   4.943151  4.865517
2016-03-31  -2.577695   2.281581  4.859276
dtwexm last valid 2019-12-31 dtwexbgs first valid 2006-01-31
```

Before 2007-01-31 `dollar_chg12` equals `m_chg12` on every row; from 2007-01-31 it equals `b_chg12`, whose first valid row is exactly the splice date (DTWEXBGS begins 2006-01-31 in the panel). DTWEXM's last month-end is 2019-12-31, so the overlap for the additional evidence is 2007-01-31 to 2019-12-31, 156 rows: correlation 0.9655842667203641; the largest absolute gap between the two change series is 5.24 points (2015-03-31, the broad index rising less than the major-currencies index), the next four are in 2009-08 and 2016. No level jump: the splice is on 12-row changes, and the row-before/row-at values (−5.405424 → −1.486488) are two changes of two different indices, not a level.

### Step 2.3 — `z.loc[["1991-01-31", "2004-12-31", "2005-01-31"]]`, `z.tail(3)`, first-window moments of z with the rows counted, and the dropped-row table

```
$ .venv/Scripts/python.exe -c "<standardise(build_raw_features(panel, cfg), cfg); model_input>"
z rows 427 1991-01-31 2026-07-31
z.loc[[1991-01-31, 2004-12-31, 2005-01-31]]
            dgs10_level  dgs10_chg12  slope_2s10s  cpi_3m_ann  indpro_chg12  dollar_chg12  oil_chg12   log_vix  unrate_chg12  breakeven_chg12
date                                                                                                                                         
1991-01-31     1.860011    -0.097686    -0.145572    1.880358     -1.212649     -1.337797  -0.268487  0.362554      1.151368              NaN
2004-12-31    -1.364082     0.287847     0.061196    1.024596      0.510128     -0.915481   0.845154 -1.090552     -0.736461        -0.639965
2005-01-31    -1.435935     0.297309    -0.293960    0.321819      0.682159     -0.640696   1.155991 -1.197269     -0.444449        -0.940619
z.tail(3)
            dgs10_level  dgs10_chg12  slope_2s10s  cpi_3m_ann  indpro_chg12  dollar_chg12  oil_chg12   log_vix  unrate_chg12  breakeven_chg12
date                                                                                                                                         
2026-05-31     0.186355     0.183176    -0.596364    1.861494     -0.084299     -0.427949   1.101306 -0.522329      0.085829         0.049290
2026-06-30     0.180631     0.363214    -0.780496    2.196021     -0.006061      0.149804   0.095275 -0.309721      0.020198        -0.133202
2026-07-31     0.351031     0.565632    -0.593860    0.061880     -0.136705     -0.313112   0.520167 -0.393851      0.085781        -0.254753
first-window moments of z (rows counted per column)
                 n_rows_in_window        mean_z  std_z_ddof1  max_abs_mean  abs_std_minus_1
dgs10_level                   168 -4.229421e-17          1.0  4.229421e-17     1.110223e-16
dgs10_chg12                   168  2.312965e-18          1.0  2.312965e-18     0.000000e+00
slope_2s10s                   168 -4.229421e-17          1.0  4.229421e-17     0.000000e+00
cpi_3m_ann                    168 -1.057355e-17          1.0  1.057355e-17     0.000000e+00
indpro_chg12                  168 -1.057355e-16          1.0  1.057355e-16     0.000000e+00
dollar_chg12                  168  1.057355e-17          1.0  1.057355e-17     0.000000e+00
oil_chg12                     168  0.000000e+00          1.0  0.000000e+00     0.000000e+00
log_vix                       168  1.268826e-16          1.0  1.268826e-16     0.000000e+00
unrate_chg12                  168  0.000000e+00          1.0  0.000000e+00     1.110223e-16
breakeven_chg12                12  2.590520e-16          1.0  2.590520e-16     0.000000e+00
max |mean| = 2.590520390792032e-16  max |std-1| = 1.1102230246251565e-16
raw first-window rows for breakeven_chg12 (its non-NaN rows in the window)
date
2004-01-31    0.50
2004-02-29    0.44
2004-03-31    0.58
2004-04-30    0.69
2004-05-31    1.06
2004-06-30    0.88
2004-07-31    0.41
2004-08-31    0.19
2004-09-30    0.36
2004-10-31    0.02
2004-11-30    0.30
2004-12-31    0.29
z NaN count per column from features_from
dgs10_level          0
dgs10_chg12          0
slope_2s10s          0
cpi_3m_ann           1
indpro_chg12         0
dollar_chg12         0
oil_chg12            0
log_vix              0
unrate_chg12         0
breakeven_chg12    156
dropped rows
         date     missing
0  2026-02-28  cpi_3m_ann
model_input (426, 8)
features_z sha256 6d1223f17d6cb476fd57477df982e52de8281f2a5a0a1129a687d6c2da7d9f50
```

For the nine columns valid throughout, the first window 1991-01-31 to 2004-12-31 has 168 rows; `breakeven_chg12` (T10YIE begins 2003-01-31, so its 12-row change begins 2004-01-31) has 12 rows in the window, printed above, and its window moments are over those 12 rows — the same rule over its own non-NaN rows. The z NaN counts from 1991-01-31 are 1 for `cpi_3m_ann` (2026-02-28) and 156 for `breakeven_chg12` (1991-01-31 to 2003-12-31), zero elsewhere. The SHA-256 of `features_z.parquet` is shown and matches the run-twice check below.

### Step 2.4 — `feature_sanity.csv` in full (370 rows = 10 features × 37 calendar years 1990–2026)

```
$ .venv/Scripts/python.exe -c "<pd.read_csv('outputs/tables/feature_sanity.csv').to_string()>"
feature_sanity.csv, 370 rows
             feature  year         min         max  n_nan
0        dgs10_level  1990    8.080000    9.040000      0
1        dgs10_level  1991    6.710000    8.240000      0
2        dgs10_level  1992    6.370000    7.610000      0
3        dgs10_level  1993    5.400000    6.390000      0
4        dgs10_level  1994    5.700000    7.910000      0
5        dgs10_level  1995    5.580000    7.600000      0
6        dgs10_level  1996    5.600000    6.960000      0
7        dgs10_level  1997    5.750000    6.920000      0
8        dgs10_level  1998    4.440000    5.680000      0
9        dgs10_level  1999    4.660000    6.450000      0
10       dgs10_level  2000    5.120000    6.680000      0
11       dgs10_level  2001    4.300000    5.430000      0
12       dgs10_level  2002    3.630000    5.420000      0
13       dgs10_level  2003    3.370000    4.490000      0
14       dgs10_level  2004    3.860000    4.660000      0
15       dgs10_level  2005    3.940000    4.570000      0
16       dgs10_level  2006    4.460000    5.150000      0
17       dgs10_level  2007    3.970000    5.030000      0
18       dgs10_level  2008    2.250000    4.060000      0
19       dgs10_level  2009    2.710000    3.850000      0
20       dgs10_level  2010    2.470000    3.840000      0
21       dgs10_level  2011    1.890000    3.470000      0
22       dgs10_level  2012    1.510000    2.230000      0
23       dgs10_level  2013    1.700000    3.040000      0
24       dgs10_level  2014    2.170000    2.730000      0
25       dgs10_level  2015    1.680000    2.350000      0
26       dgs10_level  2016    1.460000    2.450000      0
27       dgs10_level  2017    2.120000    2.450000      0
28       dgs10_level  2018    2.690000    3.150000      0
29       dgs10_level  2019    1.500000    2.730000      0
30       dgs10_level  2020    0.550000    1.510000      0
31       dgs10_level  2021    1.110000    1.740000      0
32       dgs10_level  2022    1.790000    4.100000      0
33       dgs10_level  2023    3.440000    4.880000      0
34       dgs10_level  2024    3.810000    4.690000      0
35       dgs10_level  2025    4.020000    4.580000      0
36       dgs10_level  2026    3.970000    4.750000      0
37       dgs10_chg12  1990         NaN         NaN     12
38       dgs10_chg12  1991   -1.370000   -0.160000      0
39       dgs10_chg12  1992   -1.480000   -0.010000      0
40       dgs10_chg12  1993   -1.560000   -0.870000      0
41       dgs10_chg12  1994   -0.690000    2.380000      0
42       dgs10_chg12  1995   -2.260000    1.900000      0
43       dgs10_chg12  1996   -2.000000    0.850000      0
44       dgs10_chg12  1997   -0.780000    0.930000      0
45       dgs10_chg12  1998   -1.680000   -0.520000      0
46       dgs10_chg12  1999   -0.870000    1.800000      0
47       dgs10_chg12  2000   -1.330000    2.020000      0
48       dgs10_chg12  2001   -1.500000   -0.050000      0
49       dgs10_chg12  2002   -1.240000    0.490000      0
50       dgs10_chg12  2003   -1.710000    0.440000      0
51       dgs10_chg12  2004   -0.320000    1.290000      0
52       dgs10_chg12  2005   -0.680000    0.640000      0
53       dgs10_chg12  2006   -0.030000    1.210000      0
54       dgs10_chg12  2007   -0.670000    0.300000      0
55       dgs10_chg12  2008   -1.790000   -0.470000      0
56       dgs10_chg12  2009   -0.800000    1.600000      0
57       dgs10_chg12  2010   -0.930000    1.130000      0
58       dgs10_chg12  2011   -1.410000    0.210000      0
59       dgs10_chg12  2012   -1.590000   -0.110000      0
60       dgs10_chg12  2013   -0.360000    1.260000      0
61       dgs10_chg12  2014   -0.870000    0.970000      0
62       dgs10_chg12  2015   -0.990000    0.100000      0
63       dgs10_chg12  2016   -0.860000    0.260000      0
64       dgs10_chg12  2017   -0.050000    0.840000      0
65       dgs10_chg12  2018    0.270000    0.770000      0
66       dgs10_chg12  2019   -1.460000   -0.090000      0
67       dgs10_chg12  2020   -1.870000   -0.780000      0
68       dgs10_chg12  2021   -0.400000    1.040000      0
69       dgs10_chg12  2022    0.390000    2.550000      0
70       dgs10_chg12  2023    0.000000    2.090000      0
71       dgs10_chg12  2024   -0.780000    1.250000      0
72       dgs10_chg12  2025   -0.520000    0.590000      0
73       dgs10_chg12  2026   -0.320000    0.380000      0
74       slope_2s10s  1990    0.010000    0.930000      0
75       slope_2s10s  1991    0.980000    2.000000      0
76       slope_2s10s  1992    1.940000    2.570000      0
77       slope_2s10s  1993    1.440000    2.220000      0
78       slope_2s10s  1994    0.150000    1.580000      0
79       slope_2s10s  1995    0.340000    0.570000      0
80       slope_2s10s  1996    0.470000    0.690000      0
81       slope_2s10s  1997    0.090000    0.590000      0
82       slope_2s10s  1998   -0.050000    0.520000      0
83       slope_2s10s  1999    0.080000    0.290000      0
84       slope_2s10s  2000   -0.470000    0.070000      0
85       slope_2s10s  2001    0.510000    2.000000      0
86       slope_2s10s  2002    1.700000    2.280000      0
87       slope_2s10s  2003    2.040000    2.690000      0
88       slope_2s10s  2004    1.160000    2.330000      0
89       slope_2s10s  2005   -0.020000    0.850000      0
90       slope_2s10s  2006   -0.160000    0.200000      0
91       slope_2s10s  2007   -0.110000    0.990000      0
92       slope_2s10s  2008    1.360000    2.450000      0
93       slope_2s10s  2009    1.900000    2.710000      0
94       slope_2s10s  2010    2.000000    2.820000      0
95       slope_2s10s  2011    1.640000    2.840000      0
96       slope_2s10s  2012    1.280000    1.900000      0
97       slope_2s10s  2013    1.480000    2.660000      0
98       slope_2s10s  2014    1.500000    2.330000      0
99       slope_2s10s  2015    1.210000    1.710000      0
100      slope_2s10s  2016    0.780000    1.260000      0
101      slope_2s10s  2017    0.510000    1.260000      0
102      slope_2s10s  2018    0.210000    0.620000      0
103      slope_2s10s  2019    0.000000    0.340000      0
104      slope_2s10s  2020    0.180000    0.800000      0
105      slope_2s10s  2021    0.790000    1.580000      0
106      slope_2s10s  2022   -0.700000    0.610000      0
107      slope_2s10s  2023   -1.060000   -0.190000      0
108      slope_2s10s  2024   -0.390000    0.330000      0
109      slope_2s10s  2025    0.250000    0.710000      0
110      slope_2s10s  2026    0.300000    0.740000      0
111       cpi_3m_ann  1990    3.154333    8.862562      0
112       cpi_3m_ann  1991    1.491955    4.918385      0
113       cpi_3m_ann  1992    2.310374    4.111069      0
114       cpi_3m_ann  1993    0.834778    4.002368      0
115       cpi_3m_ann  1994    1.931645    3.845718      0
116       cpi_3m_ann  1995    1.842663    3.497821      0
117       cpi_3m_ann  1996    1.798863    4.459801      0
118       cpi_3m_ann  1997    0.753056    3.336256      0
119       cpi_3m_ann  1998    0.247295    2.492092      0
120       cpi_3m_ann  1999    1.208811    4.154660      0
121       cpi_3m_ann  2000    2.161435    5.809649      0
122       cpi_3m_ann  2001    0.000000    4.432114      0
123       cpi_3m_ann  2002   -2.004949    4.348441      0
124       cpi_3m_ann  2003   -0.650934    5.163139      0
125       cpi_3m_ann  2004    0.000000    5.476702      0
126       cpi_3m_ann  2005    1.264473    9.380500      0
127       cpi_3m_ann  2006   -3.869882    5.745096      0
128       cpi_3m_ann  2007    0.197482    7.005499      0
129       cpi_3m_ann  2008  -10.219221   10.571707      0
130       cpi_3m_ann  2009  -12.729620    4.882201      0
131       cpi_3m_ann  2010   -1.461637    3.283554      0
132       cpi_3m_ann  2011    0.801450    6.240591      0
133       cpi_3m_ann  2012   -0.835439    5.401578      0
134       cpi_3m_ann  2013   -1.603872    3.205196      0
135       cpi_3m_ann  2014   -0.665482    3.530786      0
136       cpi_3m_ann  2015   -5.075550    3.636986      0
137       cpi_3m_ann  2016   -0.994998    3.468603      0
138       cpi_3m_ann  2017   -0.994405    4.314815      0
139       cpi_3m_ann  2018    1.238231    4.388652      0
140       cpi_3m_ann  2019   -0.191323    3.666689      0
141       cpi_3m_ann  2020   -4.972251    6.261482      0
142       cpi_3m_ann  2021    2.429115    9.680486      0
143       cpi_3m_ann  2022    1.955065   11.274095      0
144       cpi_3m_ann  2023    1.833467    4.878781      0
145       cpi_3m_ann  2024    0.418270    4.627334      0
146       cpi_3m_ann  2025    1.010767    4.539686      0
147       cpi_3m_ann  2026    2.065330    8.204073      1
148     indpro_chg12  1990   -0.556588    2.284247      0
149     indpro_chg12  1991   -3.459906   -0.462750      0
150     indpro_chg12  1992    0.093765    2.527041      0
151     indpro_chg12  1993    2.845534    4.420609      0
152     indpro_chg12  1994    4.576842    6.448226      0
153     indpro_chg12  1995    1.890726    6.088101      0
154     indpro_chg12  1996    0.082068    4.310324      0
155     indpro_chg12  1997    3.668826    5.455093      0
156     indpro_chg12  1998    1.461185    5.776504      0
157     indpro_chg12  1999    1.572313    4.219140      0
158     indpro_chg12  2000    4.629701    5.884451      0
159     indpro_chg12  2001   -6.624189    3.101099      0
160     indpro_chg12  2002   -5.960497    1.776582      0
161     indpro_chg12  2003   -1.389826    2.078374      0
162     indpro_chg12  2004    2.301274    6.106776      0
163     indpro_chg12  2005    1.922380    4.266701      0
164     indpro_chg12  2006    2.789031    5.424264      0
165     indpro_chg12  2007    1.341761    3.377042      0
166     indpro_chg12  2008   -5.659494    2.258191      0
167     indpro_chg12  2009  -14.651666   -5.212627      0
168     indpro_chg12  2010   -2.033629    7.898601      0
169     indpro_chg12  2011    3.173901    5.740099      0
170     indpro_chg12  2012    1.721995    5.031514      0
171     indpro_chg12  2013    1.413842    3.412325      0
172     indpro_chg12  2014    2.795442    5.083563      0
173     indpro_chg12  2015   -1.181070    4.762984      0
174     indpro_chg12  2016   -2.048126   -0.533678      0
175     indpro_chg12  2017    0.005069    3.295438      0
176     indpro_chg12  2018    3.401545    5.014984      0
177     indpro_chg12  2019   -1.139234    3.874886      0
178     indpro_chg12  2020  -16.567051    0.036867      0
179     indpro_chg12  2021   -4.341899   16.202678      0
180     indpro_chg12  2022    2.476413    7.188395      0
181     indpro_chg12  2023   -0.682422    1.632167      0
182     indpro_chg12  2024   -0.902818    1.563399      0
183     indpro_chg12  2025    0.547359    2.487621      0
184     indpro_chg12  2026    0.738947    2.249535      0
185     dollar_chg12  1990         NaN         NaN     12
186     dollar_chg12  1991   -9.426583    3.671534      0
187     dollar_chg12  1992  -10.521599    8.236700      0
188     dollar_chg12  1993   -1.760170    9.260927      0
189     dollar_chg12  1994   -7.386648    3.328872      0
190     dollar_chg12  1995  -10.821531   -0.246579      0
191     dollar_chg12  1996    0.687924    8.415445      0
192     dollar_chg12  1997    4.719101   10.197928      0
193     dollar_chg12  1998   -3.501460    8.026318      0
194     dollar_chg12  1999   -5.365996    1.252804      0
195     dollar_chg12  2000    0.101522   11.258733      0
196     dollar_chg12  2001    1.941877   11.671133      0
197     dollar_chg12  2002   -9.756753    8.170172      0
198     dollar_chg12  2003  -16.640760   -8.720171      0
199     dollar_chg12  2004  -13.812762   -4.351722      0
200     dollar_chg12  2005   -7.512061    7.901324      0
201     dollar_chg12  2006   -7.284576    4.725336      0
202     dollar_chg12  2007   -8.852874   -1.486488      0
203     dollar_chg12  2008  -10.936029   12.591216      0
204     dollar_chg12  2009  -10.656826   18.547075      0
205     dollar_chg12  2010  -11.360487    1.083873      0
206     dollar_chg12  2011  -10.558644    2.089932      0
207     dollar_chg12  2012   -1.463492    8.861567      0
208     dollar_chg12  2013   -1.145035    2.797643      0
209     dollar_chg12  2014   -1.172108    8.917890      0
210     dollar_chg12  2015   10.266155   15.459834      0
211     dollar_chg12  2016    0.848265    9.035471      0
212     dollar_chg12  2017   -7.242568    5.332119      0
213     dollar_chg12  2018   -8.484698    5.244729      0
214     dollar_chg12  2019   -0.780419    6.883370      0
215     dollar_chg12  2020   -3.382018    6.223312      0
216     dollar_chg12  2021   -8.799848    3.500889      0
217     dollar_chg12  2022    1.221084   11.169090      0
218     dollar_chg12  2023   -4.169244    5.168372      0
219     dollar_chg12  2024   -0.838531    8.771687      0
220     dollar_chg12  2025   -7.659176    6.470063      0
221     dollar_chg12  2026   -8.596163    1.261596      0
222        oil_chg12  1990         NaN         NaN     12
223        oil_chg12  1991  -57.471787   19.162748      0
224        oil_chg12  1992  -14.573867    6.039571      0
225        oil_chg12  1993  -31.736402    9.389861      0
226        oil_chg12  1994  -34.957136   22.497415      0
227        oil_chg12  1995  -14.158627   28.594566      0
228        oil_chg12  1996   -4.595328   33.695720      0
229        oil_chg12  1997  -38.350719   31.354860      0
230        oil_chg12  1998  -53.171411  -25.624055      0
231        oil_chg12  1999  -29.526449   78.268395      0
232        oil_chg12  2000    3.658945   90.960720      0
233        oil_chg12  2001  -54.646256   10.232207      0
234        oil_chg12  2002  -37.297965   44.700829      0
235        oil_chg12  2003   -7.604613   53.071778      0
236        oil_chg12  2004   -1.867160   57.146634      0
237        oil_chg12  2005   14.400169   48.559889      0
238        oil_chg12  2006   -5.128535   37.799085      0
239        oil_chg12  2007  -15.407700   47.221508      0
240        oil_chg12  2008  -76.609336   68.774367      0
241        oil_chg12  2009  -85.743620   57.663856      0
242        oil_chg12  2010    2.762685   61.314307      0
243        oil_chg12  2011   -1.284006   32.774702      0
244        oil_chg12  2012  -17.143652   15.518180      0
245        oil_chg12  2013  -15.146161   17.666679      0
246        oil_chg12  2014  -60.795403   11.757754      0
247        oil_chg12  2015  -75.539762  -36.432137      0
248        oil_chg12  2016  -42.022030   36.991840      0
249        oil_chg12  2017   -4.773417   50.038647      0
250        oil_chg12  2018  -29.199171   47.674422      0
251        oil_chg12  2019  -30.199950   30.317604      0
252        oil_chg12  2020 -119.975174   -4.288268      0
253        oil_chg12  2021    1.118192  119.456835      0
254        oil_chg12  2022    3.576010   54.518552      0
255        oil_chg12  2023  -51.840219   12.742783      0
256        oil_chg12  2024  -27.785210   15.891067      0
257        oil_chg12  2025  -33.791057   -4.614553      0
258        oil_chg12  2026  -12.160003   60.122337      0
259          log_vix  1990    2.740840    3.402530      0
260          log_vix  1991    2.671386    3.055415      0
261          log_vix  1992    2.531313    2.856470      0
262          log_vix  1993    2.421257    2.621766      0
263          log_vix  1994    2.363680    3.017983      0
264          log_vix  1995    2.431857    2.626840      0
265          log_vix  1996    2.528126    3.040706      0
266          log_vix  1997    2.954389    3.557916      0
267          log_vix  1998    2.920470    3.790533      0
268          log_vix  1999    3.048799    3.327910      0
269          log_vix  2000    2.823757    3.389462      0
270          log_vix  2001    2.947592    3.513335      0
271          log_vix  2002    2.856470    3.681099      0
272          log_vix  2003    2.778819    3.439456      0
273          log_vix  2004    2.583243    2.844328      0
274          log_vix  2005    2.448416    2.729159      0
275          log_vix  2006    2.389680    2.799717      0
276          log_vix  2007    2.343727    3.157851      0
277          log_vix  2008    2.880882    4.092510      0
278          log_vix  2009    3.076390    3.836221      0
279          log_vix  2010    2.867331    3.542118      0
280          log_vix  2011    2.691243    3.760269      0
281          log_vix  2012    2.740840    3.180551      0
282          log_vix  2013    2.541602    2.833801      0
283          log_vix  2014    2.433613    2.954910      0
284          log_vix  2015    2.494857    3.347445      0
285          log_vix  2016    2.474014    3.022861      0
286          log_vix  2017    2.252344    2.558776      0
287          log_vix  2018    2.494857    3.235536      0
288          log_vix  2019    2.535283    2.943386      0
289          log_vix  2020    2.935982    3.980429      0
290          log_vix  2021    2.761907    3.499231      0
291          log_vix  2022    3.023347    3.508556      0
292          log_vix  2023    2.521721    3.030134      0
293          log_vix  2024    2.520917    3.142427      0
294          log_vix  2025    2.704711    3.206803      0
295          log_vix  2026    2.729159    3.228826      0
296     unrate_chg12  1990   -0.100000    0.600000      0
297     unrate_chg12  1991    0.800000    1.700000      0
298     unrate_chg12  1992    0.300000    1.000000      0
299     unrate_chg12  1993   -0.900000    0.200000      0
300     unrate_chg12  1994   -0.900000   -0.400000      0
301     unrate_chg12  1995   -1.200000    0.000000      0
302     unrate_chg12  1996   -0.500000    0.200000      0
303     unrate_chg12  1997   -0.700000   -0.200000      0
304     unrate_chg12  1998   -0.700000   -0.200000      0
305     unrate_chg12  1999   -0.500000    0.000000      0
306     unrate_chg12  2000   -0.400000   -0.100000      0
307     unrate_chg12  2001   -0.100000    1.700000      0
308     unrate_chg12  2002    0.300000    1.800000      0
309     unrate_chg12  2003    0.000000    0.600000      0
310     unrate_chg12  2004   -0.700000   -0.100000      0
311     unrate_chg12  2005   -0.600000   -0.200000      0
312     unrate_chg12  2006   -0.600000   -0.200000      0
313     unrate_chg12  2007   -0.400000    0.300000      0
314     unrate_chg12  2008    0.300000    2.000000      0
315     unrate_chg12  2009    2.300000    3.900000      0
316     unrate_chg12  2010   -0.500000    2.600000      0
317     unrate_chg12  2011   -1.200000   -0.300000      0
318     unrate_chg12  2012   -1.200000   -0.700000      0
319     unrate_chg12  2013   -0.800000   -0.400000      0
320     unrate_chg12  2014   -1.400000   -0.800000      0
321     unrate_chg12  2015   -1.200000   -0.700000      0
322     unrate_chg12  2016   -0.800000   -0.100000      0
323     unrate_chg12  2017   -0.700000   -0.100000      0
324     unrate_chg12  2018   -0.700000   -0.300000      0
325     unrate_chg12  2019   -0.300000   -0.100000      0
326     unrate_chg12  2020   -0.400000   11.100000      0
327     unrate_chg12  2021   -8.700000    3.100000      0
328     unrate_chg12  2022   -2.800000   -0.500000      0
329     unrate_chg12  2023   -0.600000    0.300000      0
330     unrate_chg12  2024    0.200000    0.800000      0
331     unrate_chg12  2025    0.000000    0.400000      0
332     unrate_chg12  2026    0.000000    0.300000      0
333  breakeven_chg12  1990         NaN         NaN     12
334  breakeven_chg12  1991         NaN         NaN     12
335  breakeven_chg12  1992         NaN         NaN     12
336  breakeven_chg12  1993         NaN         NaN     12
337  breakeven_chg12  1994         NaN         NaN     12
338  breakeven_chg12  1995         NaN         NaN     12
339  breakeven_chg12  1996         NaN         NaN     12
340  breakeven_chg12  1997         NaN         NaN     12
341  breakeven_chg12  1998         NaN         NaN     12
342  breakeven_chg12  1999         NaN         NaN     12
343  breakeven_chg12  2000         NaN         NaN     12
344  breakeven_chg12  2001         NaN         NaN     12
345  breakeven_chg12  2002         NaN         NaN     12
346  breakeven_chg12  2003         NaN         NaN     12
347  breakeven_chg12  2004    0.020000    1.060000      0
348  breakeven_chg12  2005   -0.290000    0.330000      0
349  breakeven_chg12  2006   -0.300000    0.340000      0
350  breakeven_chg12  2007   -0.300000    0.070000      0
351  breakeven_chg12  2008   -2.200000    0.130000      0
352  breakeven_chg12  2009   -1.470000    2.260000      0
353  breakeven_chg12  2010   -0.120000    1.190000      0
354  breakeven_chg12  2011   -0.340000    0.640000      0
355  breakeven_chg12  2012   -0.320000    0.670000      0
356  breakeven_chg12  2013   -0.330000    0.480000      0
357  breakeven_chg12  2014   -0.560000    0.270000      0
358  breakeven_chg12  2015   -0.560000   -0.140000      0
359  breakeven_chg12  2016   -0.470000    0.410000      0
360  breakeven_chg12  2017   -0.080000    0.640000      0
361  breakeven_chg12  2018   -0.250000    0.380000      0
362  breakeven_chg12  2019   -0.610000    0.060000      0
363  breakeven_chg12  2020   -1.010000    0.250000      0
364  breakeven_chg12  2021    0.480000    1.500000      0
365  breakeven_chg12  2022   -0.260000    0.470000      0
366  breakeven_chg12  2023   -0.700000    0.200000      0
367  breakeven_chg12  2024   -0.170000    0.230000      0
368  breakeven_chg12  2025   -0.180000    0.260000      0
369  breakeven_chg12  2026   -0.130000    0.230000      0
```

The 1990 rows of every 12-row-change column have `n_nan = 12` (nothing to lag to); `breakeven_chg12` is NaN through 2003 (T10YIE begins 2003-01-31); `cpi_3m_ann` 2026 has `n_nan = 1` (2026-02-28). The 2026 rows cover seven months (to `sample.end` 2026-07-31).

### Step 2.4 — the dates each plausibility sentence was checked against

```
$ .venv/Scripts/python.exe -c "<raw = pd.read_parquet(cfg.outputs_features_raw); panel = pd.read_parquet(cfg.outputs_asof_panel); ...>"
== spot checks ==
cpi_3m_ann 2008-10..2009-03
date
2008-10-31     2.619608
2008-11-30    -4.433828
2008-12-31   -10.219221
2009-01-31   -12.729620
2009-02-28    -8.416223
2009-03-31    -0.479294
cpi_3m_ann 2020-03..2020-08
date
2020-03-31    1.915683
2020-04-30   -0.757770
2020-05-31   -4.434004
2020-06-30   -4.972251
2020-07-31   -1.141030
2020-08-31    4.482952
log_vix 2008-09..2009-01 and 2020-02..2020-05
date
2008-09-30    3.673512
2008-10-31    4.092510
2008-11-30    4.022490
2008-12-31    3.688879
2009-01-31    3.803101
date
2020-02-29    3.691626
2020-03-31    3.980429
2020-04-30    3.530763
2020-05-31    3.314550
log_vix argmax 2008-10-31 4.092509546276371 argmin 2017-09-30 2.252343876557299
dgs10_chg12 2022
date
2022-01-31    0.68
2022-02-28    0.39
2022-03-31    0.58
2022-04-30    1.24
2022-05-31    1.27
2022-06-30    1.53
2022-07-31    1.43
2022-08-31    1.85
2022-09-30    2.31
2022-10-31    2.55
2022-11-30    2.25
2022-12-31    2.36
oil_chg12 2020-03..2021-05 with panel wti
              wti  wti_lag12   oil_chg12
date                                    
2020-03-31  20.51      60.19 -107.659365
2020-04-30  19.23      63.83 -119.975174
2020-05-31  35.57      53.49  -40.799213
2020-06-30  39.27      58.20  -39.342449
2020-07-31  40.10      58.53  -37.816311
2020-08-31  42.61      55.07  -25.651614
2020-09-30  40.05      54.09  -30.052065
2020-10-31  35.64      54.02  -41.588575
2020-11-30  45.20      58.12  -25.141275
2020-12-31  48.35      61.14  -23.470009
2021-01-31  52.16      51.58    1.118192
2021-02-28  61.55      44.83   31.697229
2021-03-31  59.19      20.51  105.984004
2021-04-30  63.50      19.23  119.456835
2021-05-31  66.31      35.57   62.283813
indpro_chg12 2009-01..2009-12 and 2020-03..2021-06
date
2009-01-31    -8.143041
2009-02-28   -10.512721
2009-03-31   -12.594243
2009-04-30   -13.678986
2009-05-31   -13.408495
2009-06-30   -14.397900
2009-07-31   -14.651666
2009-08-31   -13.984770
2009-09-30   -11.352783
2009-10-31    -6.273118
2009-11-30    -7.362581
2009-12-31    -5.212627
date
2020-03-31     0.036867
2020-04-30    -5.642699
2020-05-31   -16.304346
2020-06-30   -16.567051
2020-07-31   -11.446090
2020-08-31    -8.529543
2020-09-30    -8.047147
2020-10-31    -7.556039
2020-11-30    -5.487438
2020-12-31    -5.661715
2021-01-31    -3.648239
2021-02-28    -1.848933
2021-03-31    -4.341899
2021-04-30     1.009884
2021-05-31    16.202678
2021-06-30    15.120644
unrate_chg12 2020-03..2021-06
date
2020-03-31    -0.3
2020-04-30     0.6
2020-05-31    11.1
2020-06-30     9.7
2020-07-31     7.4
2020-08-31     6.5
2020-09-30     4.7
2020-10-31     4.4
2020-11-30     3.3
2020-12-31     3.2
2021-01-31     3.1
2021-02-28     2.8
2021-03-31     2.7
2021-04-30     1.6
2021-05-31    -8.7
2021-06-30    -7.5
slope_2s10s 2000, 2006-07, 2019, 2022-2024 extremes
date
2000-04-30   -0.45
2000-12-31    0.01
2006-12-31   -0.11
2007-06-30    0.16
2019-08-31    0.00
2023-06-30   -1.06
2023-07-31   -0.91
2024-08-31    0.00
slope argmin 2023-06-30 -1.06
dgs10_level max/min 1990-04-30 9.04 2020-07-31 0.55
dollar_chg12 max/min 2009-02-28 18.54707496102916 2003-12-31 -16.640759658368264
breakeven_chg12 2008-09..2009-06 + 2021
date
2008-09-30   -0.72
2008-10-31   -1.47
2008-11-30   -2.01
2008-12-31   -2.20
2009-01-31   -1.20
2009-02-28   -1.47
2009-03-31   -1.06
2009-04-30   -0.80
2009-05-31   -0.68
2009-06-30   -0.76
date
2021-01-31    0.48
2021-02-28    0.74
2021-03-31    1.50
2021-04-30    1.34
2021-05-31    1.27
2021-06-30    0.98
2021-07-31    0.85
2021-08-31    0.53
2021-09-30    0.74
2021-10-31    0.81
2021-11-30    0.73
2021-12-31    0.57
breakeven first valid 2004-01-31 t10yie first valid 2003-01-31
```

Chart: [`outputs/charts/features_review.png`](../outputs/charts/features_review.png) (5 × 2 panels, 1991-01-31 to 2026-07-31, dashed line at 2004-12-31, 300 dpi, 975 kB, `metadata={"Software": None}`).

Plausibility, one sentence per feature, each naming the rows above:

- `dgs10_level`: range 0.55 (2020-07-31) to 9.04 (1990-04-30, in the table's 1990 row; the 1991-onward maximum is 8.24 in 1991) — the 2020 trough and the 1990 peak are the known extremes of the 10-year yield, and the table's 2022 row (1.79 → 4.10) matches the hiking year.
- `dgs10_chg12`: the 2022 rows climb from 0.68 in January to 2.55 at 2022-10-31 and end the year at 2.36, the largest 12-month rise in the sample, consistent with the 2022 tightening; 1994 (max 2.38) and 1995 (min −2.26) are the other known swings.
- `slope_2s10s`: near zero or negative at 2000-04-30 (−0.45), 2006-12-31 (−0.11), 2019-08-31 (0.00) and 2024-08-31 (0.00), and the sample minimum −1.06 at 2023-06-30, the deepest inversion of the period — all four known inversion episodes are where they should be.
- `cpi_3m_ann`: negative through 2008-Q4 into 2009-Q1 (−4.43 at 2008-11-30, −10.22 at 2008-12-31, −12.73 at 2009-01-31; annualised 3-month rates so large in magnitude because of the oil collapse) and negative in 2020-Q2 (−4.43 at 2020-05-31, −4.97 at 2020-06-30), positive elsewhere with the 2021–22 peak visible in the table; each value refers to CPI of month t−1 by the as-of rule.
- `indpro_chg12`: the troughs of −14.65 at 2009-07-31 and −16.57 at 2020-06-30 are the two recessions, followed by the mirror-image +16.20 at 2021-05-31 (12-row base effect); otherwise within about ±7.
- `dollar_chg12`: max 18.55 at 2009-02-28 (flight to the dollar) and min −16.64 at 2003-12-31 (the 2003 dollar slide), both on the expected side; the splice rows above show no discontinuity at 2007-01-31.
- `oil_chg12`: −107.66 at 2020-03-31 and −119.98 at 2020-04-30 are large and negative because WTI month-ends were 20.51 and 19.23 against 60.19 and 63.83 a year earlier — the negative 2020-04-20 print is not captured by the 10-day month-end rule (2020-04-30 is 19.23 > 0), so no log of a non-positive value occurs and no row is dropped for it; the +119.46 at 2021-04-30 is the base effect of the same rows twelve later.
- `log_vix`: the sample maximum 4.0925 is at 2008-10-31 (VIX ≈ 59.9 at month-end), 2008-11-30 is 4.02, and the 2020 peak is 3.98 at 2020-03-31 (VIX ≈ 53.5); the minimum 2.25 at 2017-09-30 is the 2017 low-volatility year — the "2008-11 peak" in the plan's example is the second-highest reading, the month-end series peaking one month earlier.
- `unrate_chg12`: +11.1 at 2020-05-31 (the April-2020 unemployment spike, one month lagged by the as-of rule) then −8.7 at 2021-05-31 as it drops out of the base; 2009 rises to about +4 in the table; plausible.
- `breakeven_chg12`: −2.20 at 2008-12-31 (breakevens collapsing in the crisis) and +1.50 at 2021-03-31 (the 2021 reflation), first valid 2004-01-31 as designed; plausible.

### Section 2 run and determinism

```
$ time .venv/Scripts/python.exe -m regime.run --section 2
INFO regime: as-of panel read from data/processed/asof_panel.parquet: 439 rows x 16 columns
INFO regime: raw features written to data/processed/features_raw.parquet: 439 rows x 10 columns
INFO regime: z written to data/processed/features_z.parquet: 427 rows from 1991-01-31
INFO regime: model input 426 rows x 8 columns; 1 dropped rows written to data/processed/dropped_rows.csv
INFO regime: dropped 2026-02-28: cpi_3m_ann
INFO regime: feature_sanity.csv written: 370 rows
INFO regime: features review chart written to outputs\charts\features_review.png

real	0m2.098s
user	0m0.000s
sys	0m0.000s
```

SHA-256 of the five files after the first run, identical after a second run (`diff` of the two hash lists printed `IDENTICAL`):

```
9584b4b69ee8d2600cccd336f68f6def73a7d07543b6280cda806a18689f275d *data/processed/features_raw.parquet
6d1223f17d6cb476fd57477df982e52de8281f2a5a0a1129a687d6c2da7d9f50 *data/processed/features_z.parquet
9ae1a26f1c3595e992be2f89c5b85869a281472907a6e1d0c6a56b54373f15ac *outputs/charts/features_review.png
a8f11d04bbb5417d5ec0b3aa4abc1fb706c863642dce863b402c307f84dce094 *outputs/tables/feature_sanity.csv
96d136eac67265ef1954c7697f58353438773dd7337d9f13bdcfc3fad302a816 *data/processed/dropped_rows.csv
```

## Tests run

Exact command and full output, unedited (41 tests: the 29 of section 1 plus 12 new — 2 in step 2.1, 2 in 2.2, 4 parametrised spike cases + 2 in 2.3, 1 table and 1 chart in 2.4; `test_unbuilt_section_raises` now targets section 3):

```
$ .venv/Scripts/python.exe -m pytest -q
.........................................                                [100%]
41 passed in 4.44s
```

## Runtime per step

| step | wall-clock | machine | notes |
|---|---|---|---|
| 2.1 | ~22 min authoring; compute < 1 s | Windows 11 laptop, Python 3.11 venv | 11:15–11:38; `features.py` and all of `tests/test_features.py` were written and debugged together (one fixture placed the lagged row at row 0, fixed in the test), then committed per step |
| 2.2 | < 1 min; compute < 1 s | same | tests only; commit 4cfcc1f at 11:38:29 |
| 2.3 | < 1 min; compute < 1 s | same | tests only; commit 9c39d6c at 11:38:47 |
| 2.4 | ~12 min authoring; `run --section 2` 2.1 s | same | includes two full section-2 runs for the hash comparison; commit 7fdfbff at 11:40:30 |

No step approached the 20-minute threshold of `run.step_timeout_minutes`; nothing was reduced.

## Not verified

- The plausibility sentences compare against economic knowledge, not against an independent data source; no external series was fetched (no test touches the network).
- `feature_sanity` and the chart cover `raw` as built from the full panel (1990-01-31 onward); the chart is restricted to `features_from`–`sample_end`, the table is not, so its 1990 rows are all-NaN for the lagged columns. The plan does not fix the table's date range; reported, not chosen.
- The chart was inspected visually once (rendered, all ten panels titled and populated, dashed line at 2004-12-31); no automated check on its content beyond existence and size > 10 kB.
- `standardise` under a `dataclasses.replace(cfg, sample_features_from=..., sample_first_window_end=...)` config (the section 6.3 robustness variant, convention 13) is not exercised in this section; the function takes every date from `cfg`, but the variant has not been run.
- Wall-clock per step is from commit timestamps and the session log, not from an instrumented timer; `outputs/tables/runtime.csv` is a step 7.3 deliverable.

## Open questions

None appended to `decisions/OPEN.md`. No step was left incomplete.

## Files changed

Step 2.1 (`b698ca1`):
- `regime/features.py` — added: `splice_dollar_chg12`, `build_raw_features`, `standardise`, `model_input`, `_log`, `_write`
- `tests/test_features.py` — added: `test_raw_feature_definitions_by_hand`, `test_lags_are_rows_not_days`

Step 2.2 (`4cfcc1f`):
- `tests/test_features.py` — added: `test_dollar_splice_switches_exactly_at_splice_date`, `test_dollar_splice_has_no_level_jump`

Step 2.3 (`9c39d6c`):
- `tests/test_features.py` — added: `test_spike_after_t_does_not_change_z_at_t` (parametrised over 2004-12-31, 2005-01-31, 2009-12-31, 2026-06-30), `test_first_window_rule`, `test_model_input_drops_nan_rows_and_writes_them`

Step 2.4 (`7fdfbff`):
- `regime/tables.py` — added: `feature_sanity`, `FEATURE_SANITY_COLUMNS`
- `regime/charts.py` — added: `features_review_chart` (Agg backend, `metadata={"Software": None}`)
- `regime/run.py` — `section_2` built; sections 3–7 still `_not_built`
- `tests/test_run.py` — `test_unbuilt_section_raises` asserts on section 3
- `tests/test_tables.py`, `tests/test_charts.py` — added
- `outputs/tables/feature_sanity.csv`, `outputs/charts/features_review.png` — committed outputs of the run

Review:
- `review/section_2.md` — this file

Not committed (regenerated, gitignored): `data/processed/features_raw.parquet`, `data/processed/features_z.parquet`, `data/processed/dropped_rows.csv`.

## Reviewer reads

1. `regime/features.py` — the ten definitions against the step 2.1 table; `splice_dollar_chg12` splices changes (`m.where(index < splice, b)`); `standardise` overwrites the expanding moments with the window moments for rows ≤ `first_window_end` and uses pandas' NaN-skipping `expanding()` after it; `model_input` drops on any NaN in the requested columns.
2. `tests/test_features.py` — the hand-built panel (row 1 is the lag of row 13), the two splice tests, the spike parametrisation and its comment on why first-window rows are excluded, and the first-window-rule test including the 12-row `breakeven_chg12` window.
3. This file's step 2.1 and 2.3 evidence — the single dropped date and its panel row; the first-window moments table with row counts.
4. `regime/run.py::section_2` — the order raw → z → model_input → sanity → chart, all from `cfg` paths.
