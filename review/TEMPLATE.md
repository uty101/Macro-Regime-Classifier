# Review — Section X: <title>

## Section

One paragraph: which section of `PLAN.md` this is, what it was meant to produce, and whether every step completed. If the section stopped early (rule 4), name the step and why.

## Steps completed

One line per step, in order, with the commit hash:

- `step X.1: <message>` — `<hash>`
- `step X.2: <message>` — `<hash>`

## Evidence

One subsection per numeric claim made anywhere in this file or in the outputs of this section. Every aggregate number is backed by the raw rows it was computed from, printed with `to_string()`. Never `.describe()` or `.mean()` alone.

### <claim 1, e.g. "CPI obs month at t is t−1 for 97.4% of decision dates">

The command or function that produced the rows, then the rows:

```
<DataFrame.to_string() output>
```

### <claim 2>

...

## Tests run

Exact command and full output, unedited:

```
$ pytest -q
<full output>
```

## Runtime per step

| step | wall-clock | machine | notes |
|---|---|---|---|
| X.1 | m:ss | | |

Any step over 20 minutes is reported here and was not worked around by reducing restarts, replications or the grid (rule 10).

## Not verified

Everything this section did not check, one line each. Nothing is silently omitted.

## Open questions

Items appended to `decisions/OPEN.md` this section, each with its two options, and which steps were left incomplete because of them. "None" if nothing was appended.

## Files changed

Every file added, modified or deleted, grouped by step.

## Reviewer reads

Ordered list, the shortest set sufficient to clear the section: the files the reviewer must read, with one line on what to look for in each.
