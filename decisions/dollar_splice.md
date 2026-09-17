# Dollar splice

`dollar_chg12` is the 12-row log change × 100 of a trade-weighted dollar index.
For decision dates up to and including 2006-11-30 the change comes from DTWEXM (major currencies, 1973 to 2020-01).
From 2006-12-31 onward (`features.dollar_splice_date` in config) the change comes from DTWEXBGS (broad, from 2006-01).
The change series is spliced, never the levels, so the different index bases cannot create a jump.
A test asserts the spliced change series has no level jump at the splice date.
