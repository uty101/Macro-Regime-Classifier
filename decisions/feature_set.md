# Feature set

Core d = 8: `dgs10_level`, `dgs10_chg12`, `slope_2s10s`, `cpi_3m_ann`, `indpro_chg12`, `dollar_chg12`, `oil_chg12`, `log_vix`, defined exactly as in `PLAN.md` section 2.
Robustness only (section 6): `unrate_chg12` and `breakeven_chg12`, valid from 2004-01-31.
The core set is fixed now and is not changed after conditional factor returns have been computed.
Standardisation is real-time: first-window mean/sd up to `first_window_end`, expanding thereafter, ddof = 1, computed once into `data/processed/features_z.parquet` and never recomputed at refits.
Rows with any NaN core feature are dropped from model input and their dates reported.
