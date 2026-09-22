# Review — Section 4: Conditional statistics

## Section

Section 4 of `PLAN.md`, steps 4.1 to 4.5, as amended by the reviewer's answers in
`decisions/section_3_review.md` and by the `section 4: reviewer decisions and plan amendments`
commit. It joins each regime labelling to the factor return of the month after the decision
date, reports the conditional mean, standard deviation and Sharpe per factor per state for the
four label sources, attaches stationary block-bootstrap intervals to every cell and to every
pairwise state difference, measures how much of each conditional Sharpe is hindsight by
comparing the filtered and smoothed labellings, and skips the optional project 1 join.

**Every step completed.** The section did not stop early: no test failed at any step (rule 4),
no step approached its 20-minute threshold (rule 10), and nothing was appended to
`decisions/OPEN.md` (rule 1). This is the first section that reads a factor return; sections 1
to 3 were re-run first, without `--pull`, and the `classifier_diagnostics.csv` they printed is
identical to the one recorded in `instructions/03b_section_3_revision.status.md`, so nothing
below rests on a changed classifier.

## Steps completed

- `04: instructions` — `13dfcfe`
- `section 4: reviewer decisions and plan amendments` — `f6cd731`
- `step 4.1: join the decision at t to the factor returns of month t+1` — `c3ed4d1`
- `step 4.2: conditional statistics for the four label sources, and the refit split` — `14cba40`
- `step 4.3: stationary block bootstrap, pairwise state differences and the pooled table` — `034b24a`
- `step 4.4: the filtered-minus-smoothed Sharpe gap, both labellings in one draw` — `895a41c`
- `step 4.5: the project 1 conditional join, or a logged skip` — `4a82ffb`

## Evidence

### The join puts the month t+1 return beside the decision at t (step 4.1)

The four-row hand example of `tests/test_conditional.py::test_join_alignment_on_four_rows`,
printed. Five decision dates, five factor months, every return distinct, so a one-row
misalignment could not survive. Inputs:

```
            label  assigned
date                       
2020-01-31      0      True
2020-02-29      1      True
2020-03-31      2      True
2020-04-30      0      True
2020-05-31      1      True
```

The factor file — the return **of** each month:

```
            Mkt-RF   SMB
date                    
2020-01-31     1.0  10.0
2020-02-29     2.0  20.0
2020-03-31     3.0  30.0
2020-04-30     4.0  40.0
2020-05-31     5.0  50.0
```

`join_next_return(labels, factors, cfg)`:

```
            label  assigned  Mkt-RF   SMB
date                                     
2020-01-31      0      True     2.0  20.0
2020-02-29      1      True     3.0  30.0
2020-03-31      2      True     4.0  40.0
2020-04-30      0      True     5.0  50.0
```

The decision at 2020-01-31 earns 2.0, the February return, and so on. The 2020-05-31 decision
date has no t+1 return and is absent rather than carrying a NaN.

### The real join: 258 out-of-sample dates, 2004-12-31 to 2026-06-30, identical for all four sources

`join_next_return(sources["hmm_filtered"], factors, cfg)`, head and tail:

```
            label  assigned  Mkt-RF     SMB     HML     RMW     CMA     UMD
date                                                                       
2004-12-31      0      True -0.0275 -0.0114  0.0210  0.0282 -0.0138  0.0312
2005-01-31      0      True  0.0188 -0.0032  0.0144  0.0136 -0.0017  0.0343
2005-02-28      0      True -0.0193 -0.0141  0.0208  0.0046  0.0134  0.0043
```

```
            label  assigned  Mkt-RF     SMB     HML     RMW     CMA     UMD
date                                                                       
2026-04-30      1      True  0.0491 -0.0265 -0.0231 -0.0816 -0.0146  0.0139
2026-05-31      1      True -0.0107  0.0436  0.0358 -0.0507  0.0345  0.0462
2026-06-30      1      True -0.0060 -0.0012  0.0211  0.1105  0.0234 -0.1225
```

The last row is the decision at 2026-06-30. Its returns are the 2026-07 returns, which is the
last month in the French file (`sample.end = 2026-07-31`, convention 10). Both rows of the
factor file, so the shift can be checked by eye:

```
            Mkt-RF     SMB     HML     RMW     CMA     UMD
date                                                      
2026-06-30 -0.0107  0.0436  0.0358 -0.0507  0.0345  0.0462
2026-07-31 -0.0060 -0.0012  0.0211  0.1105  0.0234 -0.1225
```

Span per source:

```
      source  n_dates      first       last  index_equals_hmm_filtered
hmm_filtered      258 2004-12-31 2026-06-30                       True
hmm_smoothed      258 2004-12-31 2026-06-30                       True
gmm_filtered      258 2004-12-31 2026-06-30                       True
       rules      258 2004-12-31 2026-06-30                       True
```

The four sources are compared on identical dates, as PLAN.md's out-of-sample convention
requires. They coincide without any intersection step because the one model-input row dropped
in section 2 (2026-02-28, missing `cpi_3m_ann`) is also the one row the rules labeller cannot
produce, both features being NaN there.

### No source leaves an out-of-sample date unassigned (step 4.2)

```
      source  n_unassigned  n_assigned
hmm_filtered             0         258
hmm_smoothed             0         258
gmm_filtered             0         258
       rules             0         258
```

`assigned` is `max probability > 0.5` for the three probabilistic sources and always true for
rules. Every filtered, smoothed and GMM row over this window has a maximum probability above
0.5, so no cell below is computed on fewer rows than its `n` column says, and the list of
unassigned dates the plan asks for is empty.

### The four conditional tables in full (steps 4.2 and 4.3, nine columns after the bootstrap)

`outputs/tables/conditional_stats_hmm_filtered.csv`:

```
factor  state   n  ann_mean  ann_std    sharpe  sharpe_p05  sharpe_p95  excludes_zero
Mkt-RF      0  91  0.048303 0.140195  0.344543   -0.406935    1.344566          False
Mkt-RF      1  61  0.146656 0.134906  1.087095    0.588614    1.773611           True
Mkt-RF      2 106  0.122672 0.172274  0.712075    0.174042    1.282782           True
   SMB      0  91 -0.012976 0.086864 -0.149381   -0.712789    0.421516          False
   SMB      1  61 -0.027639 0.104311 -0.264971   -0.766503    0.364228          False
   SMB      2 106  0.019381 0.089124  0.217462   -0.347618    0.787990          False
   HML      0  91 -0.039890 0.086716 -0.460008   -1.054890    0.177871          False
   HML      1  61  0.026990 0.106786  0.252751   -0.395023    1.043652          False
   HML      2 106  0.011004 0.129359  0.085064   -0.595287    0.701788          False
   RMW      0  91  0.048541 0.051418  0.944037    0.287502    1.646790           True
   RMW      1  61 -0.000944 0.099138 -0.009525   -0.698222    0.666484          False
   RMW      2 106  0.037947 0.070457  0.538587   -0.067646    1.053632          False
   CMA      0  91 -0.018145 0.050791 -0.357253   -0.917908    0.158767          False
   CMA      1  61 -0.027718 0.087694 -0.316075   -1.012381    0.391057          False
   CMA      2 106  0.038355 0.067444  0.568694   -0.175451    1.144325          False
   UMD      0  91  0.094378 0.124003  0.761095    0.306703    1.246084           True
   UMD      1  61  0.016584 0.139121  0.119203   -0.452916    0.755164          False
   UMD      2 106 -0.047434 0.180358 -0.262999   -0.814666    0.544030          False
```

`outputs/tables/conditional_stats_hmm_smoothed.csv`:

```
factor  state   n  ann_mean  ann_std    sharpe  sharpe_p05  sharpe_p95  excludes_zero
Mkt-RF      0  73  0.034356 0.164484  0.208872   -0.333099    0.796343          False
Mkt-RF      1  38  0.183379 0.126715  1.447179    0.831730    2.224928           True
Mkt-RF      2 147  0.114751 0.153052  0.749751    0.126092    1.454759           True
   SMB      0  73 -0.039074 0.079980 -0.488549   -1.166534    0.173549          False
   SMB      1  38 -0.045316 0.117134 -0.386870   -1.011039    0.427573          False
   SMB      2 147  0.025592 0.089992  0.284379   -0.175197    0.702523          False
   HML      0  73 -0.073759 0.106802 -0.690613   -1.396792    0.046111          False
   HML      1  38  0.031895 0.123533  0.258188   -0.525306    1.470212          False
   HML      2 147  0.022824 0.108104  0.211134   -0.266005    0.677394          False
   RMW      0  73 -0.009468 0.077828 -0.121659   -0.617235    0.433556          False
   RMW      1  38  0.034232 0.081414  0.420465   -0.751838    1.604595          False
   RMW      2 147  0.052873 0.067215  0.786628    0.312995    1.281851           True
   CMA      0  73 -0.004011 0.070680 -0.056748   -0.667961    0.510168          False
   CMA      1  38 -0.028516 0.093612 -0.304616   -1.276144    0.820039          False
   CMA      2 147  0.014286 0.058608  0.243751   -0.299661    0.689223          False
   UMD      0  73  0.080893 0.127600  0.633960    0.122313    1.316172           True
   UMD      1  38  0.011274 0.136015  0.082885   -0.614736    1.077583          False
   UMD      2 147 -0.011984 0.168612 -0.071072   -0.516549    0.560425          False
```

`outputs/tables/conditional_stats_gmm_filtered.csv`:

```
factor  state   n  ann_mean  ann_std    sharpe  sharpe_p05  sharpe_p95  excludes_zero
Mkt-RF      0  95  0.065179 0.140058  0.465371   -0.289216    1.422927          False
Mkt-RF      1 144  0.093817 0.159296  0.588944    0.144304    1.098885           True
Mkt-RF      2  19  0.349642 0.154858  2.257820    1.348802    3.344888           True
   SMB      0  95  0.034282 0.090151  0.380275   -0.174182    0.916829          False
   SMB      1 144 -0.032208 0.088499 -0.363940   -0.772943    0.066170          False
   SMB      2  19  0.029937 0.121863  0.245660   -0.697066    1.337288          False
   HML      0  95  0.004257 0.085277  0.049918   -0.573626    0.650351          False
   HML      1 144 -0.001242 0.122509 -0.010135   -0.541520    0.472262          False
   HML      2  19 -0.054884 0.129128 -0.425037   -1.960137    1.002169          False
   RMW      0  95  0.050526 0.056574  0.893101    0.270966    1.602559           True
   RMW      1 144  0.013175 0.073002  0.180475   -0.451293    0.754780          False
   RMW      2  19  0.088674 0.125120  0.708707   -0.730266    1.970914          False
   CMA      0  95 -0.004295 0.052336 -0.082060   -0.683092    0.546057          False
   CMA      1 144  0.005800 0.074203  0.078163   -0.468796    0.538554          False
   CMA      2  19  0.015600 0.089692  0.173929   -1.282635    1.407531          False
   UMD      0  95  0.064168 0.125191  0.512565   -0.084838    1.142859          False
   UMD      1 144  0.006125 0.165314  0.037051   -0.509238    0.792056          False
   UMD      2  19 -0.126632 0.184068 -0.687962   -1.689876    0.935166          False
```

`outputs/tables/conditional_stats_rules.csv`:

```
factor  state  n  ann_mean  ann_std    sharpe  sharpe_p05  sharpe_p95  excludes_zero
Mkt-RF      0 74  0.168551 0.153688  1.096712    0.304879    1.993839           True
Mkt-RF      1 55  0.140596 0.147903  0.950595    0.321756    1.892029           True
Mkt-RF      2 79  0.082770 0.160492  0.515725   -0.226429    1.449251          False
Mkt-RF      3 50 -0.007992 0.143661 -0.055631   -0.590688    0.715184          False
   SMB      0 74  0.030097 0.095854  0.313991   -0.265241    0.852633          False
   SMB      1 55 -0.032378 0.075903 -0.426572   -1.123206    0.248803          False
   SMB      2 79 -0.001048 0.107713 -0.009731   -0.625721    0.628162          False
   SMB      3 50 -0.023520 0.074906 -0.313995   -1.308237    0.627836          False
   HML      0 74 -0.030422 0.110644 -0.274951   -0.935401    0.396927          False
   HML      1 55 -0.023127 0.074180 -0.311772   -1.074634    0.620543          False
   HML      2 79  0.002203 0.126019  0.017478   -0.700254    0.853032          False
   HML      3 50  0.050640 0.118361  0.427842   -0.691599    1.126082          False
   RMW      0 74  0.015957 0.061043  0.261400   -0.376325    0.918663          False
   RMW      1 55  0.014335 0.055931  0.256291   -0.446799    1.031153          False
   RMW      2 79  0.028739 0.089972  0.319425   -0.273944    1.090436          False
   RMW      3 50  0.082848 0.074311  1.114888    0.384456    1.934144           True
   CMA      0 74  0.012341 0.062882  0.196250   -0.417354    0.761658          False
   CMA      1 55  0.002553 0.048322  0.052827   -0.661360    0.756210          False
   CMA      2 79 -0.047787 0.067341 -0.709630   -1.342359   -0.033386           True
   CMA      3 50  0.068904 0.088796  0.775984   -0.139094    1.551155          False
   UMD      0 74  0.000876 0.208884  0.004192   -0.598613    1.067685          False
   UMD      1 55 -0.017716 0.096474 -0.183640   -0.708227    0.474821          False
   UMD      2 79 -0.014354 0.157369 -0.091215   -0.653580    0.526235          False
   UMD      3 50  0.132312 0.085040  1.555871    0.632615    2.624191           True
```

### The pooled comparison every conditional Sharpe has to be read against

`outputs/tables/unconditional_stats.csv` — the same 258 dates, every month included, no state
conditioning and no `assigned` filter:

```
factor   n  ann_mean  ann_std    sharpe  sharpe_p05  sharpe_p95
Mkt-RF 258  0.102112 0.152959  0.667575    0.258644    1.096812
   SMB 258 -0.003149 0.091978 -0.034235   -0.357822    0.301642
   HML 258 -0.003167 0.110440 -0.028680   -0.433182    0.376164
   RMW 258  0.032488 0.072659  0.447135    0.092335    0.816471
   CMA 258  0.002805 0.068059  0.041209   -0.358389    0.403102
   UMD 258  0.017721 0.153339  0.115567   -0.244422    0.574285
```

Over this window only Mkt-RF (0.668) and RMW (0.447) have an unconditional interval that
excludes zero. SMB, HML and CMA are flat to slightly negative. This is the baseline: a
conditional Sharpe of 0.7 in some state is not a finding if the factor pays 0.67 in every
month.

### Sharpe by factor and state, per source, with n in brackets

hmm_filtered:

```
state             0            1             2
factor                                        
Mkt-RF   0.345 (91)   1.087 (61)   0.712 (106)
SMB     -0.149 (91)  -0.265 (61)   0.217 (106)
HML     -0.460 (91)   0.253 (61)   0.085 (106)
RMW      0.944 (91)  -0.010 (61)   0.539 (106)
CMA     -0.357 (91)  -0.316 (61)   0.569 (106)
UMD      0.761 (91)   0.119 (61)  -0.263 (106)
```

hmm_smoothed:

```
state             0            1             2
factor                                        
Mkt-RF   0.209 (73)   1.447 (38)   0.750 (147)
SMB     -0.489 (73)  -0.387 (38)   0.284 (147)
HML     -0.691 (73)   0.258 (38)   0.211 (147)
RMW     -0.122 (73)   0.420 (38)   0.787 (147)
CMA     -0.057 (73)  -0.305 (38)   0.244 (147)
UMD      0.634 (73)   0.083 (38)  -0.071 (147)
```

gmm_filtered:

```
state             0             1            2
factor                                        
Mkt-RF   0.465 (95)   0.589 (144)   2.258 (19)
SMB      0.380 (95)  -0.364 (144)   0.246 (19)
HML      0.050 (95)  -0.010 (144)  -0.425 (19)
RMW      0.893 (95)   0.180 (144)   0.709 (19)
CMA     -0.082 (95)   0.078 (144)   0.174 (19)
UMD      0.513 (95)   0.037 (144)  -0.688 (19)
```

rules:

```
state             0            1            2            3
factor                                                    
Mkt-RF   1.097 (74)   0.951 (55)   0.516 (79)  -0.056 (50)
SMB      0.314 (74)  -0.427 (55)  -0.010 (79)  -0.314 (50)
HML     -0.275 (74)  -0.312 (55)   0.017 (79)   0.428 (50)
RMW      0.261 (74)   0.256 (55)   0.319 (79)   1.115 (50)
CMA      0.196 (74)   0.053 (55)  -0.710 (79)   0.776 (50)
UMD      0.004 (74)  -0.184 (55)  -0.091 (79)   1.556 (50)
```

### Cells and pairwise differences that exclude zero (question 2 input)

The count that answers PLAN.md question 2 is the **pairwise** one (step 7.4 as amended): a
Sharpe whose interval excludes zero says the factor pays in that state, not that it pays
*differently* there.

```
      source  cells_excl_zero  cells  diffs_excl_zero  diffs
hmm_filtered                4     18                1     18
hmm_smoothed                4     18                3     18
gmm_filtered                3     18                3     18
       rules                5     24                7     36
```

`rules` has 4 states and therefore 6 pairs per factor (36 differences); the HMM and GMM sources
have `primary_K = 3` states and 3 pairs per factor (18 differences).

The rows behind those counts. Cells first, then differences:

**hmm_filtered — cells that exclude zero:**

```
factor  state   n  ann_mean  ann_std   sharpe  sharpe_p05  sharpe_p95  excludes_zero
Mkt-RF      1  61  0.146656 0.134906 1.087095    0.588614    1.773611           True
Mkt-RF      2 106  0.122672 0.172274 0.712075    0.174042    1.282782           True
   RMW      0  91  0.048541 0.051418 0.944037    0.287502    1.646790           True
   UMD      0  91  0.094378 0.124003 0.761095    0.306703    1.246084           True
```

**hmm_filtered — differences that exclude zero:**

```
factor  state_a  state_b  sharpe_diff  diff_p05  diff_p95  excludes_zero  n_a  n_b
   UMD        0        2    -1.024094 -1.740244 -0.112286           True   91  106
```

**hmm_smoothed — cells that exclude zero:**

```
factor  state   n  ann_mean  ann_std   sharpe  sharpe_p05  sharpe_p95  excludes_zero
Mkt-RF      1  38  0.183379 0.126715 1.447179    0.831730    2.224928           True
Mkt-RF      2 147  0.114751 0.153052 0.749751    0.126092    1.454759           True
   RMW      2 147  0.052873 0.067215 0.786628    0.312995    1.281851           True
   UMD      0  73  0.080893 0.127600 0.633960    0.122313    1.316172           True
```

**hmm_smoothed — differences that exclude zero:**

```
factor  state_a  state_b  sharpe_diff  diff_p05  diff_p95  excludes_zero  n_a  n_b
Mkt-RF        0        1     1.238306  0.350368  2.179252           True   73   38
   HML        0        2     0.901747  0.146182  1.666262           True   73  147
   RMW        0        2     0.908286  0.143180  1.587192           True   73  147
```

**gmm_filtered — cells that exclude zero:**

```
factor  state   n  ann_mean  ann_std   sharpe  sharpe_p05  sharpe_p95  excludes_zero
Mkt-RF      1 144  0.093817 0.159296 0.588944    0.144304    1.098885           True
Mkt-RF      2  19  0.349642 0.154858 2.257820    1.348802    3.344888           True
   RMW      0  95  0.050526 0.056574 0.893101    0.270966    1.602559           True
```

**gmm_filtered — differences that exclude zero:**

```
factor  state_a  state_b  sharpe_diff  diff_p05  diff_p95  excludes_zero  n_a  n_b
Mkt-RF        0        2     1.792449  0.475613  3.105939           True   95   19
Mkt-RF        1        2     1.668876  0.557260  2.957805           True  144   19
   SMB        0        1    -0.744214 -1.413798 -0.049759           True   95  144
```

**rules — cells that exclude zero:**

```
factor  state  n  ann_mean  ann_std    sharpe  sharpe_p05  sharpe_p95  excludes_zero
Mkt-RF      0 74  0.168551 0.153688  1.096712    0.304879    1.993839           True
Mkt-RF      1 55  0.140596 0.147903  0.950595    0.321756    1.892029           True
   RMW      3 50  0.082848 0.074311  1.114888    0.384456    1.934144           True
   CMA      2 79 -0.047787 0.067341 -0.709630   -1.342359   -0.033386           True
   UMD      3 50  0.132312 0.085040  1.555871    0.632615    2.624191           True
```

**rules — differences that exclude zero:**

```
factor  state_a  state_b  sharpe_diff  diff_p05  diff_p95  excludes_zero  n_a  n_b
Mkt-RF        0        3    -1.152343 -2.159682 -0.038015           True   74   50
Mkt-RF        1        3    -1.006226 -2.077680 -0.027726           True   55   50
   CMA        0        2    -0.905880 -1.667487 -0.049272           True   74   79
   CMA        2        3     1.485613  0.284733  2.482774           True   79   50
   UMD        0        3     1.551678  0.208743  2.769463           True   74   50
   UMD        1        3     1.739510  0.490343  3.036931           True   55   50
   UMD        2        3     1.647086  0.429111  2.929772           True   79   50
```

The headline: **1 of 18** pairwise differences excludes zero for `hmm_filtered`, the only
source the strategy is allowed to use out of sample — UMD, state 0 against state 2, −1.024
with a [−1.740, −0.112] interval. The hindsight `hmm_smoothed` labelling finds 3, the GMM 3,
and the backward-looking `rules` quadrants 7 of 36. On this evidence the filtered HMM
separates the factor premia far less than any of the three comparisons, and the single
difference it does find is on momentum, not on the value/quality factors the regime story is
usually told about.

### The two conditional_differences tables in full, for the filtered and smoothed HMM

`outputs/tables/conditional_differences_hmm_filtered.csv`:

```
factor  state_a  state_b  sharpe_diff  diff_p05  diff_p95  excludes_zero  n_a  n_b
Mkt-RF        0        1     0.742552 -0.380484  1.795091          False   91   61
Mkt-RF        0        2     0.367532 -0.739228  1.300040          False   91  106
Mkt-RF        1        2    -0.375020 -1.214048  0.356088          False   61  106
   SMB        0        1    -0.115590 -0.861296  0.731170          False   91   61
   SMB        0        2     0.366844 -0.411102  1.246239          False   91  106
   SMB        1        2     0.482433 -0.394042  1.241632          False   61  106
   HML        0        1     0.712759 -0.155488  1.662482          False   91   61
   HML        0        2     0.545071 -0.274983  1.385765          False   91  106
   HML        1        2    -0.167687 -1.230079  0.731154          False   61  106
   RMW        0        1    -0.953562 -2.019260  0.044757          False   91   61
   RMW        0        2    -0.405450 -1.425986  0.439806          False   91  106
   RMW        1        2     0.548112 -0.360901  1.387597          False   61  106
   CMA        0        1     0.041177 -0.833387  0.969108          False   91   61
   CMA        0        2     0.925947 -0.021524  1.767060          False   91  106
   CMA        1        2     0.884769 -0.159326  1.784932          False   61  106
   UMD        0        1    -0.641892 -1.370189  0.133812          False   91   61
   UMD        0        2    -1.024094 -1.740244 -0.112286           True   91  106
   UMD        1        2    -0.382202 -1.221237  0.571151          False   61  106
```

`outputs/tables/conditional_differences_hmm_smoothed.csv`:

```
factor  state_a  state_b  sharpe_diff  diff_p05  diff_p95  excludes_zero  n_a  n_b
Mkt-RF        0        1     1.238306  0.350368  2.179252           True   73   38
Mkt-RF        0        2     0.540879 -0.284552  1.397612          False   73  147
Mkt-RF        1        2    -0.697427 -1.710025  0.203037          False   38  147
   SMB        0        1     0.101679 -0.833588  1.147757          False   73   38
   SMB        0        2     0.772928 -0.069503  1.622147          False   73  147
   SMB        1        2     0.671249 -0.231438  1.423834          False   38  147
   HML        0        1     0.948800 -0.093489  2.382678          False   73   38
   HML        0        2     0.901747  0.146182  1.666262           True   73  147
   HML        1        2    -0.047053 -1.319811  0.834427          False   38  147
   RMW        0        1     0.542124 -0.843929  1.833900          False   73   38
   RMW        0        2     0.908286  0.143180  1.587192           True   73  147
   RMW        1        2     0.366162 -0.833876  1.648526          False   38  147
   CMA        0        1    -0.247868 -1.433237  1.071372          False   73   38
   CMA        0        2     0.300499 -0.376599  0.979162          False   73  147
   CMA        1        2     0.548367 -0.742290  1.631183          False   38  147
   UMD        0        1    -0.551075 -1.547176  0.612732          False   73   38
   UMD        0        2    -0.705033 -1.483281  0.053970          False   73  147
   UMD        1        2    -0.153958 -1.170729  0.787597          False   38  147
```

### The refit split: what the 45% costs (step 4.2, the reviewer's Q1)

`decisions/section_3_review.md` Q1 accepts that 9 of the 20 filtered label changes land on a
31 December refit date and asks for the cost to be measured rather than engineered away. Every
out-of-sample date is flagged by whether the filtered label run containing it *began* on a
refit date. The run starts, with the flag:

```
            label  run_started_on_refit
date                                   
2004-12-31      0                  True
2005-12-31      1                  True
2006-12-31      0                  True
2007-07-31      1                 False
2007-09-30      0                 False
2008-12-31      2                  True
2009-10-31      0                 False
2009-12-31      2                  True
2013-12-31      0                  True
2014-12-31      2                  True
2016-09-30      0                 False
2016-10-31      2                 False
2016-12-31      0                  True
2019-12-31      2                  True
2020-04-30      0                 False
2020-07-31      2                 False
2020-12-31      0                  True
2021-02-28      2                 False
2022-06-30      1                 False
2025-04-30      0                 False
2025-05-31      1                 False
```

164 of the 258 dates sit in a run that began on a refit date, 94 do not.

`outputs/tables/conditional_stats_refit_split.csv` in full:

```
factor  state  run_started_on_refit  n  ann_mean  ann_std    sharpe
Mkt-RF      0                 False 22 -0.118309 0.207242 -0.570875
Mkt-RF      0                  True 69  0.101426 0.108457  0.935174
Mkt-RF      1                 False 49  0.158327 0.147391  1.074197
Mkt-RF      1                  True 12  0.099000 0.064948  1.524307
Mkt-RF      2                 False 23  0.112017 0.180808  0.619538
Mkt-RF      2                  True 83  0.125624 0.170963  0.734804
   SMB      0                 False 22  0.013855 0.096135  0.144116
   SMB      0                  True 69 -0.021530 0.084301 -0.255398
   SMB      1                 False 49 -0.039355 0.108113 -0.364017
   SMB      1                  True 12  0.020200 0.089923  0.224635
   SMB      2                 False 23  0.044974 0.109376  0.411187
   SMB      2                  True 83  0.012289 0.083318  0.147497
   HML      0                 False 22 -0.050673 0.104767 -0.483673
   HML      0                  True 69 -0.036452 0.080986 -0.450104
   HML      1                 False 49  0.009184 0.116369  0.078919
   HML      1                  True 12  0.099700 0.050095  1.990233
   HML      2                 False 23  0.231287 0.176090  1.313459
   HML      2                  True 83 -0.050039 0.107980 -0.463405
   RMW      0                 False 22  0.161455 0.050010  3.228473
   RMW      0                  True 69  0.012539 0.047681  0.262979
   RMW      1                 False 49 -0.007984 0.109312 -0.073035
   RMW      1                  True 12  0.027800 0.037070  0.749930
   RMW      2                 False 23  0.151461 0.101120  1.497835
   RMW      2                  True 83  0.006492 0.056820  0.114247
   CMA      0                 False 22  0.021000 0.051012  0.411669
   CMA      0                  True 69 -0.030626 0.050559 -0.605755
   CMA      1                 False 49 -0.052261 0.095164 -0.549172
   CMA      1                  True 12  0.072500 0.035956  2.016344
   CMA      2                 False 23  0.140243 0.104938  1.336438
   CMA      2                  True 83  0.010120 0.050598  0.200019
   UMD      0                 False 22  0.248345 0.173237  1.433556
   UMD      0                  True 69  0.045287 0.101195  0.447520
   UMD      1                 False 49  0.036294 0.151544  0.239494
   UMD      1                  True 12 -0.063900 0.068002 -0.939684
   UMD      2                 False 23 -0.037461 0.133767 -0.280046
   UMD      2                  True 83 -0.050198 0.191962 -0.261498
```

This is the finding the reviewer asked for and it is not small. Reading down the `sharpe`
column, the two halves of a state frequently disagree in sign, and in three cells they
disagree by more than a whole unit of Sharpe:

- **RMW, state 0**: 3.228 on the 22 dates whose run did not start on a refit, 0.263 on the 69
  dates whose run did. The state's pooled figure, 0.944, is one of the four cells whose
  bootstrap interval excludes zero, and it is an average of those two.
- **HML, state 2**: 1.313 against −0.463.
- **CMA, state 1**: −0.549 against 2.016, on 49 and 12 dates.
- **Mkt-RF, state 0**: −0.571 against 0.935.

The split is information only and carries no bootstrap, so none of these gaps has an interval
and none is a significance claim; several rest on twelve or twenty-two months. What it
establishes is narrower and sufficient: the conditional statistics of `hmm_filtered` are not
stable across the refit schedule, so a state's pooled Sharpe is partly a statement about when
the model was last refitted. Section 7's write-up reports this table beside the question 2
count.

### Bootstrap settings, and the NaN-replication check

`StationaryBootstrap(block_size=6, joined_frame, seed=20260917)`, 2000 replications,
percentiles 0.05 and 0.95 — all from `config.toml`, nothing hard-coded. The whole joined frame
is resampled jointly, one row being one (label, assigned, six returns) tuple, so a label can
never be paired with another month's return.

A state with fewer than two rows in a replication has no sample standard deviation, so its
Sharpe is NaN for that replication and the percentiles use `np.nanquantile`. The session set a
5% reporting threshold. Nothing comes near it — the rows of
`outputs/tables/bootstrap_nan_replications.csv` with any NaN at all:

```
      source factor  state  n_replications  n_nan  share_nan
gmm_filtered Mkt-RF      2            2000      1     0.0005
gmm_filtered    SMB      2            2000      1     0.0005
gmm_filtered    HML      2            2000      1     0.0005
gmm_filtered    RMW      2            2000      1     0.0005
gmm_filtered    CMA      2            2000      1     0.0005
gmm_filtered    UMD      2            2000      1     0.0005
```

Maximum `share_nan` over all 78 rows: 0.000500. Rows above 5%: 0.

The only affected cells are the three states of `gmm_filtered`, whose state 2 holds 19 of the
258 dates; a stationary-bootstrap draw occasionally contains fewer than two of them. One
replication in 2000 is far below the threshold and no interval below is materially affected.

### The filtered/smoothed gap (step 4.4)

`outputs/tables/filtered_smoothed_gap.csv` in full — smoothed Sharpe minus filtered Sharpe,
both labellings resampled in the same draw:

```
factor  state       gap   gap_p05   gap_p95
Mkt-RF      0 -0.135670 -1.238004  0.754707
Mkt-RF      1  0.360084 -0.284289  1.026063
Mkt-RF      2  0.037677 -0.586075  0.672053
   SMB      0 -0.339167 -1.066862  0.353638
   SMB      1 -0.121899 -0.554719  0.297752
   SMB      2  0.066917 -0.388635  0.454746
   HML      0 -0.230605 -1.024985  0.533849
   HML      1  0.005437 -0.380035  0.623029
   HML      2  0.126071 -0.274896  0.599473
   RMW      0 -1.065696 -1.900999 -0.294969
   RMW      1  0.429990 -0.318265  1.295200
   RMW      2  0.248040 -0.071779  0.702799
   CMA      0  0.300504 -0.439539  1.035691
   CMA      1  0.011459 -0.662595  0.661643
   CMA      2 -0.324943 -0.746757  0.127955
   UMD      0 -0.127134 -0.654867  0.482525
   UMD      1 -0.036317 -0.477775  0.671045
   UMD      2  0.191927 -0.134686  0.515041
```

Mean absolute gap over the 18 cells = 0.231085, from this column:

```
factor  state       gap  abs_gap
Mkt-RF      0 -0.135670 0.135670
Mkt-RF      1  0.360084 0.360084
Mkt-RF      2  0.037677 0.037677
   SMB      0 -0.339167 0.339167
   SMB      1 -0.121899 0.121899
   SMB      2  0.066917 0.066917
   HML      0 -0.230605 0.230605
   HML      1  0.005437 0.005437
   HML      2  0.126071 0.126071
   RMW      0 -1.065696 1.065696
   RMW      1  0.429990 0.429990
   RMW      2  0.248040 0.248040
   CMA      0  0.300504 0.300504
   CMA      1  0.011459 0.011459
   CMA      2 -0.324943 0.324943
   UMD      0 -0.127134 0.127134
   UMD      1 -0.036317 0.036317
   UMD      2  0.191927 0.191927
```

Every interval in the table contains zero except RMW state 0 ([−1.901, −0.295]). The mean
absolute gap of 0.231 is therefore an average of differences that are individually
indistinguishable from noise at these sample sizes; it is reported as a magnitude, not as
evidence that hindsight pays.

The underlying label agreement over the same window:

```
agreement 176 of 258 = 0.682171
```

The confusion table:

```
smoothed   0   1   2
filtered            
0         41   0  50
1         23  38   0
2          9   0  97
```

0.682171 matches `filtered_smoothed_agreement` in `classifier_diagnostics.csv` for the core
set, which is the section 3 number, so the re-run of sections 1 to 3 reproduced the classifier
exactly. The off-diagonal is one-sided: the filter spends 50 dates in state 0 that the
smoother calls state 2, and 23 in state 1 that it calls state 0 — it is late leaving a regime,
not early entering one.

### Mean monthly return per state beside the unconditional mean

Requested so a state's Sharpe can be read against the size of the return that produced it.
Monthly decimals, not annualised:

```
      source factor  state   n  mean_monthly  uncond_mean_monthly  difference
hmm_filtered Mkt-RF      0  91      0.004025             0.008509   -0.004484
hmm_filtered Mkt-RF      1  61      0.012221             0.008509    0.003712
hmm_filtered Mkt-RF      2 106      0.010223             0.008509    0.001713
hmm_filtered    SMB      0  91     -0.001081            -0.000262   -0.000819
hmm_filtered    SMB      1  61     -0.002303            -0.000262   -0.002041
hmm_filtered    SMB      2 106      0.001615            -0.000262    0.001877
hmm_filtered    HML      0  91     -0.003324            -0.000264   -0.003060
hmm_filtered    HML      1  61      0.002249            -0.000264    0.002513
hmm_filtered    HML      2 106      0.000917            -0.000264    0.001181
hmm_filtered    RMW      0  91      0.004045             0.002707    0.001338
hmm_filtered    RMW      1  61     -0.000079             0.002707   -0.002786
hmm_filtered    RMW      2 106      0.003162             0.002707    0.000455
hmm_filtered    CMA      0  91     -0.001512             0.000234   -0.001746
hmm_filtered    CMA      1  61     -0.002310             0.000234   -0.002544
hmm_filtered    CMA      2 106      0.003196             0.000234    0.002963
hmm_filtered    UMD      0  91      0.007865             0.001477    0.006388
hmm_filtered    UMD      1  61      0.001382             0.001477   -0.000095
hmm_filtered    UMD      2 106     -0.003953             0.001477   -0.005430
hmm_smoothed Mkt-RF      0  73      0.002863             0.008509   -0.005646
hmm_smoothed Mkt-RF      1  38      0.015282             0.008509    0.006772
hmm_smoothed Mkt-RF      2 147      0.009563             0.008509    0.001053
hmm_smoothed    SMB      0  73     -0.003256            -0.000262   -0.002994
hmm_smoothed    SMB      1  38     -0.003776            -0.000262   -0.003514
hmm_smoothed    SMB      2 147      0.002133            -0.000262    0.002395
hmm_smoothed    HML      0  73     -0.006147            -0.000264   -0.005883
hmm_smoothed    HML      1  38      0.002658            -0.000264    0.002922
hmm_smoothed    HML      2 147      0.001902            -0.000264    0.002166
hmm_smoothed    RMW      0  73     -0.000789             0.002707   -0.003496
hmm_smoothed    RMW      1  38      0.002853             0.002707    0.000145
hmm_smoothed    RMW      2 147      0.004406             0.002707    0.001699
hmm_smoothed    CMA      0  73     -0.000334             0.000234   -0.000568
hmm_smoothed    CMA      1  38     -0.002376             0.000234   -0.002610
hmm_smoothed    CMA      2 147      0.001190             0.000234    0.000957
hmm_smoothed    UMD      0  73      0.006741             0.001477    0.005264
hmm_smoothed    UMD      1  38      0.000939             0.001477   -0.000537
hmm_smoothed    UMD      2 147     -0.000999             0.001477   -0.002475
gmm_filtered Mkt-RF      0  95      0.005432             0.008509   -0.003078
gmm_filtered Mkt-RF      1 144      0.007818             0.008509   -0.000691
gmm_filtered Mkt-RF      2  19      0.029137             0.008509    0.020628
gmm_filtered    SMB      0  95      0.002857            -0.000262    0.003119
gmm_filtered    SMB      1 144     -0.002684            -0.000262   -0.002422
gmm_filtered    SMB      2  19      0.002495            -0.000262    0.002757
gmm_filtered    HML      0  95      0.000355            -0.000264    0.000619
gmm_filtered    HML      1 144     -0.000103            -0.000264    0.000160
gmm_filtered    HML      2  19     -0.004574            -0.000264   -0.004310
gmm_filtered    RMW      0  95      0.004211             0.002707    0.001503
gmm_filtered    RMW      1 144      0.001098             0.002707   -0.001609
gmm_filtered    RMW      2  19      0.007389             0.002707    0.004682
gmm_filtered    CMA      0  95     -0.000358             0.000234   -0.000592
gmm_filtered    CMA      1 144      0.000483             0.000234    0.000250
gmm_filtered    CMA      2  19      0.001300             0.000234    0.001066
gmm_filtered    UMD      0  95      0.005347             0.001477    0.003871
gmm_filtered    UMD      1 144      0.000510             0.001477   -0.000966
gmm_filtered    UMD      2  19     -0.010553             0.001477   -0.012029
       rules Mkt-RF      0  74      0.014046             0.008509    0.005537
       rules Mkt-RF      1  55      0.011716             0.008509    0.003207
       rules Mkt-RF      2  79      0.006897             0.008509   -0.001612
       rules Mkt-RF      3  50     -0.000666             0.008509   -0.009175
       rules    SMB      0  74      0.002508            -0.000262    0.002771
       rules    SMB      1  55     -0.002698            -0.000262   -0.002436
       rules    SMB      2  79     -0.000087            -0.000262    0.000175
       rules    SMB      3  50     -0.001960            -0.000262   -0.001698
       rules    HML      0  74     -0.002535            -0.000264   -0.002271
       rules    HML      1  55     -0.001927            -0.000264   -0.001663
       rules    HML      2  79      0.000184            -0.000264    0.000447
       rules    HML      3  50      0.004220            -0.000264    0.004484
       rules    RMW      0  74      0.001330             0.002707   -0.001378
       rules    RMW      1  55      0.001195             0.002707   -0.001513
       rules    RMW      2  79      0.002395             0.002707   -0.000312
       rules    RMW      3  50      0.006904             0.002707    0.004197
       rules    CMA      0  74      0.001028             0.000234    0.000795
       rules    CMA      1  55      0.000213             0.000234   -0.000021
       rules    CMA      2  79     -0.003982             0.000234   -0.004216
       rules    CMA      3  50      0.005742             0.000234    0.005508
       rules    UMD      0  74      0.000073             0.001477   -0.001404
       rules    UMD      1  55     -0.001476             0.001477   -0.002953
       rules    UMD      2  79     -0.001196             0.001477   -0.002673
       rules    UMD      3  50      0.011026             0.001477    0.009549
```

The differences are tens of basis points a month. The largest for `hmm_filtered` is UMD state
2, 5.4 bp a month below the pooled mean; the largest anywhere is `gmm_filtered` Mkt-RF state 2,
2.06 percentage points a month above it, on 19 months.

### Does any state difference rest on a handful of months?

For every source, factor and state: the three largest absolute monthly returns inside the
state with their dates, the state's total return, and the mean of the months that remain once
those three are removed.

**hmm_filtered:**

```
factor  state   n     date_1     date_2     date_3   ret_1   ret_2   ret_3  sum_all  mean_ex_top3
Mkt-RF      0  91 2008-09-30 2018-11-30 2008-08-31 -0.1720 -0.0955 -0.0935   0.3663      0.008265
Mkt-RF      1  61 2026-03-31 2022-06-30 2022-08-31  0.0995  0.0957 -0.0934   0.7455      0.011098
Mkt-RF      2 106 2020-03-31 2020-02-29 2020-10-31  0.1358 -0.1335  0.1244   1.0836      0.009290
   SMB      0  91 2020-12-31 2009-11-30 2017-08-31  0.0681  0.0614  0.0476  -0.0984     -0.003131
   SMB      1  61 2024-06-30 2023-11-30 2023-02-28  0.0833  0.0725 -0.0689  -0.1405     -0.003921
   SMB      2 106 2020-02-29 2020-10-31 2009-03-31 -0.0818  0.0712  0.0708   0.1712      0.001078
   HML      0  91 2021-01-31 2019-08-31 2008-10-31  0.0720  0.0676 -0.0621  -0.3025     -0.004318
   HML      1  61 2023-02-28 2022-09-30 2023-04-30 -0.0905  0.0797 -0.0779   0.1372      0.003895
   HML      2 106 2020-02-29 2021-12-31 2008-12-31 -0.1383  0.1286 -0.1101   0.0972      0.002107
   RMW      0  91 2008-05-31 2008-10-31 2013-12-31  0.0497  0.0465 -0.0391   0.3681      0.003534
   RMW      1  61 2026-06-30 2026-04-30 2022-10-31  0.1105 -0.0816  0.0629  -0.0048     -0.001666
   RMW      2 106 2021-10-31 2021-02-28 2021-06-30  0.0719  0.0635  0.0550   0.3352      0.001406
   CMA      0  91 2020-12-31 2018-09-30 2019-08-31  0.0497  0.0356  0.0338  -0.1376     -0.002917
   CMA      1  61 2023-04-30 2022-06-30 2022-09-30 -0.0706 -0.0685  0.0642  -0.1409     -0.001138
   CMA      2 106 2021-12-31 2022-03-31 2022-05-31  0.0773  0.0592 -0.0458   0.3388      0.002409
   UMD      0  91 2008-05-31 2018-12-31 2021-01-31  0.1273 -0.0864 -0.0799   0.7157      0.008576
   UMD      1  61 2022-12-31 2026-06-30 2026-03-31 -0.1621 -0.1225  0.0951   0.0843      0.004721
   UMD      2 106 2009-03-31 2020-10-31 2009-04-30 -0.3436 -0.1260 -0.1254  -0.4190      0.001709
```

**hmm_smoothed:**

```
factor  state   n     date_1     date_2     date_3   ret_1   ret_2   ret_3  sum_all  mean_ex_top3
Mkt-RF      0  73 2020-03-31 2020-02-29 2026-03-31  0.1358 -0.1335  0.0995   0.2090      0.001531
Mkt-RF      1  38 2023-10-31 2022-09-30 2022-12-31  0.0888  0.0786  0.0660   0.5807      0.009923
Mkt-RF      2 147 2008-09-30 2020-10-31 2011-09-30 -0.1720  0.1244  0.1134   1.4057      0.009305
   SMB      0  73 2020-02-29 2005-12-31 2018-04-30 -0.0818  0.0583  0.0471  -0.2377     -0.003733
   SMB      1  38 2024-06-30 2023-11-30 2023-02-28  0.0833  0.0725 -0.0689  -0.1435     -0.006583
   SMB      2 147 2020-10-31 2009-03-31 2016-10-31  0.0712  0.0708  0.0697   0.3135      0.000707
   HML      0  73 2020-02-29 2022-04-30 2019-08-31 -0.1383  0.0844  0.0676  -0.4487     -0.006606
   HML      1  38 2023-02-28 2022-09-30 2023-04-30 -0.0905  0.0797 -0.0779   0.1010      0.005420
   HML      2 147 2021-12-31 2008-12-31 2016-10-31  0.1286 -0.1101  0.0832   0.2796      0.001235
   RMW      0  73 2026-06-30 2026-04-30 2026-05-31  0.1105 -0.0816 -0.0507  -0.0576     -0.000511
   RMW      1  38 2022-10-31 2025-09-30 2023-10-31  0.0629 -0.0525 -0.0372   0.1084      0.003863
   RMW      2 147 2021-10-31 2021-02-28 2021-06-30  0.0719  0.0635  0.0550   0.6477      0.003176
   CMA      0  73 2022-06-30 2022-03-31 2022-05-31 -0.0685  0.0592 -0.0458  -0.0244      0.000439
   CMA      1  38 2023-04-30 2022-09-30 2026-01-31 -0.0706  0.0642  0.0504  -0.0903     -0.003837
   CMA      2 147 2021-12-31 2020-12-31 2021-11-30  0.0773  0.0497  0.0429   0.1750      0.000035
   UMD      0  73 2026-06-30 2026-03-31 2018-12-31 -0.1225  0.0951 -0.0864   0.4921      0.008656
   UMD      1  38 2022-12-31 2023-11-30 2023-12-31 -0.1621 -0.0550  0.0546   0.0357      0.005663
   UMD      2 147 2009-03-31 2008-05-31 2020-10-31 -0.3436  0.1273 -0.1260  -0.1468      0.001358
```

**gmm_filtered:**

```
factor  state   n     date_1     date_2     date_3   ret_1   ret_2   ret_3  sum_all  mean_ex_top3
Mkt-RF      0  95 2008-09-30 2020-10-31 2008-08-31 -0.1720  0.1244 -0.0935   0.5160      0.007142
Mkt-RF      1 144 2020-03-31 2020-02-29 2011-09-30  0.1358 -0.1335  0.1134   1.1258      0.007164
Mkt-RF      2  19 2010-08-31 2023-10-31 2022-09-30  0.0955  0.0888  0.0786   0.5536      0.018169
   SMB      0  95 2020-10-31 2016-10-31 2020-12-31  0.0712  0.0697  0.0681   0.2714      0.000678
   SMB      1 144 2024-06-30 2020-02-29 2009-03-31  0.0833 -0.0818  0.0708  -0.3865     -0.003254
   SMB      2  19 2023-11-30 2023-02-28 2024-10-31  0.0725 -0.0689  0.0463   0.0474     -0.000156
   HML      0  95 2016-10-31 2008-10-31 2008-08-31  0.0832 -0.0621  0.0565   0.0337     -0.000477
   HML      1 144 2020-02-29 2021-12-31 2008-12-31 -0.1383  0.1286 -0.1101  -0.0149      0.000744
   HML      2  19 2023-02-28 2022-09-30 2023-11-30 -0.0905  0.0797  0.0487  -0.0869     -0.007800
   RMW      0  95 2008-05-31 2008-10-31 2020-07-31  0.0497  0.0465  0.0427   0.4000      0.002838
   RMW      1 144 2026-04-30 2021-10-31 2021-02-28 -0.0816  0.0719  0.0635   0.1581      0.000740
   RMW      2  19 2026-06-30 2022-10-31 2023-10-31  0.1105  0.0629 -0.0372   0.1404      0.000263
   CMA      0  95 2020-12-31 2016-10-31 2020-04-30  0.0497  0.0370 -0.0316  -0.0340     -0.000968
   CMA      1 144 2021-12-31 2023-04-30 2022-06-30  0.0773 -0.0706 -0.0685   0.0696      0.000932
   CMA      2  19 2022-09-30 2022-12-31 2010-11-30  0.0642 -0.0441  0.0317   0.0247     -0.001694
   UMD      0  95 2008-05-31 2020-10-31 2008-09-30  0.1273 -0.1260  0.0784   0.5080      0.004655
   UMD      1 144 2009-03-31 2009-04-30 2009-02-28 -0.3436 -0.1254 -0.1180   0.0735      0.004684
   UMD      2  19 2022-12-31 2026-06-30 2023-11-30 -0.1621 -0.1225 -0.0550  -0.2005      0.008694
```

**rules:**

```
factor  state  n     date_1     date_2     date_3   ret_1   ret_2   ret_3  sum_all  mean_ex_top3
Mkt-RF      0 74 2020-03-31 2009-03-31 2009-01-31  0.1358  0.1017 -0.1014   1.0394      0.012723
Mkt-RF      1 55 2011-09-30 2018-11-30 2010-08-31  0.1134 -0.0955  0.0955   0.6444      0.010212
Mkt-RF      2 79 2008-09-30 2020-02-29 2020-10-31 -0.1720 -0.1335  0.1244   0.5449      0.009553
Mkt-RF      3 50 2022-06-30 2022-03-31 2022-08-31  0.0957 -0.0941 -0.0934  -0.0333      0.001245
   SMB      0 74 2009-03-31 2016-10-31 2020-12-31  0.0708  0.0697  0.0681   0.1856     -0.000324
   SMB      1 55 2018-09-30 2005-03-31 2010-08-31 -0.0436 -0.0399  0.0365  -0.1484     -0.001950
   SMB      2 79 2024-06-30 2020-02-29 2023-11-30  0.0833 -0.0818  0.0725  -0.0069     -0.001064
   SMB      3 50 2018-04-30 2021-06-30 2014-06-30  0.0471 -0.0455 -0.0419  -0.0980     -0.001228
   HML      0 74 2008-12-31 2016-10-31 2021-01-31 -0.1101  0.0832  0.0720  -0.1876     -0.003277
   HML      1 55 2014-02-28 2010-05-31 2019-02-28  0.0490 -0.0473 -0.0416  -0.1060     -0.001271
   HML      2 79 2020-02-29 2023-02-28 2023-04-30 -0.1383 -0.0905 -0.0779   0.0145      0.004226
   HML      3 50 2021-12-31 2022-04-30 2022-09-30  0.1286  0.0844  0.0797   0.2110     -0.001738
   RMW      0 74 2008-05-31 2008-10-31 2020-07-31  0.0497  0.0465  0.0427   0.0984     -0.000570
   RMW      1 55 2022-10-31 2013-12-31 2010-11-30  0.0629 -0.0391 -0.0342   0.0657      0.001463
   RMW      2 79 2026-06-30 2026-04-30 2021-02-28  0.1105 -0.0816  0.0635   0.1892      0.001274
   RMW      3 50 2021-10-31 2021-06-30 2021-11-30  0.0719  0.0550  0.0488   0.3452      0.003606
   CMA      0 74 2026-01-31 2020-12-31 2016-10-31  0.0504  0.0497  0.0370   0.0761     -0.000859
   CMA      1 55 2018-09-30 2010-11-30 2022-10-31  0.0356  0.0317  0.0316   0.0117     -0.001677
   CMA      2 79 2023-04-30 2022-12-31 2025-09-30 -0.0706 -0.0441 -0.0407  -0.3146     -0.002095
   CMA      3 50 2021-12-31 2022-06-30 2022-09-30  0.0773 -0.0685  0.0642   0.2871      0.004555
   UMD      0 74 2009-03-31 2008-05-31 2009-04-30 -0.3436  0.1273 -0.1254   0.0054      0.004889
   UMD      1 55 2018-12-31 2011-12-31 2015-03-31 -0.0864 -0.0801 -0.0726  -0.0812      0.003037
   UMD      2 79 2022-12-31 2020-10-31 2026-06-30 -0.1621 -0.1260 -0.1225  -0.0945      0.004159
   UMD      3 50 2012-04-30 2022-03-31 2022-11-30  0.0645  0.0498  0.0494   0.5513      0.008247
```

The answer is yes in specific, nameable places:

- **UMD, `hmm_filtered` state 2**: the state's total is −0.419 and its single largest month is
  2009-03-31 at −0.3436, the momentum crash. Removing the three largest months turns the state
  mean from −0.40 bp to +17 bp a month. The one pairwise difference that excludes zero for
  `hmm_filtered` is UMD state 0 against state 2, so that finding leans on the momentum crash
  being inside state 2.
- **Mkt-RF, `gmm_filtered` state 2**: 19 months, Sharpe 2.258, the largest cell in any table,
  and its three largest months are 2020-10, 2020-04 and 2009-04 — a state that is essentially
  the post-crash rebound months.
- **RMW, `hmm_filtered` state 0**: total 0.368 with 2008-05 (+0.0497) and 2008-10 (+0.0465) at
  the top, but 91 months and a mean excluding the top three of 35 bp, so this one does not
  collapse when its extremes are removed. It is, however, the cell the refit split above
  disagrees most violently about.

Nothing here was corrected or excluded. It is reported so the write-up does not present a
three-month artefact as a regime effect.

### Project 1 is absent (step 4.5)

`python -m regime.run --section 4`, last two lines:

```
INFO regime: project1: absent
INFO regime: project1: absent, conditional join skipped
```

No `conditional_stats_project1.csv` was written. Convention 7 holds: nothing in section 4
depends on the file being present. The present branch is not untested dead code —
`tests/test_project1.py::test_conditional_join_runs_when_present` writes a two-factor parquet
to `tmp_path` and runs the same bootstrap through it.

## Tests run

Exact command and full output, unedited:

```
$ .venv/Scripts/python.exe -m pytest -q
........................................................................ [ 88%]
.........                                                                [100%]
81 passed in 7.93s
```

Section 4 added 14 tests to `tests/test_conditional.py` and 2 to `tests/test_project1.py`, and
changed one line of `tests/test_run.py::test_unbuilt_section_raises`, which asserted that
section 4 was not built and now asserts it of section 5.

## Runtime per step

Wall-clock of each step's computation, measured by re-running it with
`time.perf_counter()` after the section had completed:

| step | wall-clock | machine | notes |
|---|---|---|---|
| 4.1 | 0.02 s | Windows 11, this laptop | four joins |
| 4.2 | 0.08 s | | four conditional tables plus the refit split |
| 4.3 | 2.36 s | | four bootstraps of 2000 replications, plus the unconditional one |
| 4.4 | 0.59 s | | one bootstrap of 2000 replications over both labellings |
| 4.5 | 0.00 s | | the absent branch |

The threshold for every Section 4 step is 20 minutes (`run.step_timeout_minutes`); no step is
within three orders of magnitude of it. Nothing was reduced: `bootstrap.n_replications` is the
configured 2000 and `bootstrap.block_size` the configured 6 in every table above. The
rebuild of sections 1 to 3 that preceded the section was not separately timed and is not a
section 4 step; its own runtimes are in `review/section_3.md`.

## Not verified

- **The bootstrap intervals are not calibrated, only computed.** No coverage study was run
  against a known data-generating process; `block_size = 6` is taken from config and its
  sensitivity is step 6.5, not this section.
- **The refit-split table has no bootstrap**, by specification. None of the differences between
  its two halves has an interval, and several rest on 12 to 23 months. The table establishes
  instability, not significance.
- **Nothing here tests that the states mean the same thing across sources.** Chained anchoring
  (convention 16) is what makes state k comparable between filtered, smoothed and GMM, and it
  was verified in section 3; section 4 relies on it and does not re-check it. The `rules`
  states are a different partition entirely and its state numbers are quadrant codes, not
  anchored states — a `rules` state 2 has nothing to do with an HMM state 2.
- **No multiple-testing adjustment.** 18 cells and 18 differences per source are read
  independently; at a 90% interval roughly 2 of 18 would be expected to exclude zero by chance.
  `hmm_filtered` finds 1 of 18 differences, which is below that.
- **The out-of-sample window is not independent of the first fit.** It starts at
  `first_window_end`, the date of the first refit, so the first month's parameters were fitted
  on data up to and including that date (convention 4). That is the specified protocol, not a
  defect, but the first row is not a forecast.
- **The project 1 present-branch has never run on real project 1 data**, only on the synthetic
  frame in its test.
- **`unconditional_stats.csv` is written once, from the headline source's index.** It does not
  depend on labels, and all four sources share an index, so it is the same table for all four;
  that identity is asserted by the span table above, not by a test.

## Open questions

None. Nothing was appended to `decisions/OPEN.md` this section, and no step was left incomplete.

The two judgement calls that came up were both already answered in writing before the section
began, so neither became an `OPEN.md` item:

1. Whether to report question 2 as a per-cell or a pairwise count — fixed by amendment 4 of the
   `section 4: reviewer decisions and plan amendments` commit, which the reviewer's message
   specified.
2. What to do with a state holding fewer than two rows in a replication — fixed by the session
   message (NaN for that replication, `np.nanquantile`, report the counts).

## Files changed

**`section 4: reviewer decisions and plan amendments`** — `decisions/section_3_review.md`
(added), `PLAN.md` (steps 4.2, 4.3 and 7.4 amended).

**Step 4.1** — `regime/conditional.py` (`assert_complete_monthly`, `join_next_return`),
`tests/test_conditional.py` (added).

**Step 4.2** — `regime/conditional.py` (`states_of`, `_moments`,
`conditional_stats_from_joined`, `conditional_stats`, `unassigned_dates`,
`run_started_on_refit`, `conditional_stats_refit_split`), `regime/run.py`
(`load_label_sources`, `section_4`), `tests/test_conditional.py`, `tests/test_run.py`.

**Step 4.3** — `regime/conditional.py` (`excludes_zero`, `_sharpe_by_state`, `state_pairs`,
`_replication_matrices`, `bootstrap_conditional`, `block_bootstrap_ci`,
`conditional_differences`, `unconditional_stats`), `regime/run.py`,
`tests/test_conditional.py`.

**Step 4.4** — `regime/conditional.py` (`join_both`, `_gap_states`, `_gap_matrix`,
`filtered_smoothed_gap`), `regime/run.py`, `tests/test_conditional.py`.

**Step 4.5** — `regime/run.py` (`project1_conditional`), `tests/test_project1.py`.

Outputs written by `python -m regime.run --section 4` (regenerated, not committed — convention
14): `conditional_stats_{hmm_filtered,hmm_smoothed,gmm_filtered,rules}.csv`,
`conditional_differences_{...}.csv`, `conditional_stats_refit_split.csv`,
`unconditional_stats.csv`, `bootstrap_nan_replications.csv`, `filtered_smoothed_gap.csv`.

## Reviewer reads

1. `decisions/section_3_review.md` — that the four answers are recorded as given and that
   nothing in it quietly reopens `decisions/primary_feature_set.md`.
2. `regime/conditional.py`, `join_next_return` and `assert_complete_monthly` — the only place
   the timing convention is applied in this section. If this is right, nothing downstream can
   look ahead.
3. `regime/conditional.py`, `_replication_matrices` — that the label, the `assigned` flag and
   the six returns travel in one bootstrap row, and that each pairwise difference is formed
   inside its own replication.
4. The "Cells and pairwise differences that exclude zero" subsection above — the 1-of-18 for
   `hmm_filtered` against 7-of-36 for `rules` is the substantive result of the section, and it
   is not the result the project was hoping for.
5. The "refit split" subsection — the RMW state 0 split of 3.228 against 0.263 is the measured
   cost of the Q1 finding the reviewer accepted.
6. The "handful of months" subsection — specifically UMD `hmm_filtered` state 2 and the
   2009-03 momentum crash, because the section's one significant pairwise difference depends
   on it.
7. `PLAN.md` steps 4.2, 4.3 and 7.4 — that the amendments match the reviewer's message and add
   nothing else.
