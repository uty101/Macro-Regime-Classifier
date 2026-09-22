# Instruction history before rule 11

Rule 11 — save the session's instruction message verbatim before any other
work, and write a status file at the end — was added during the section 3
revision (`03b`). **Everything before that point has no verbatim instruction
file: those messages were given in chat and were not saved.** This file is the
back-fill. It lists what each session was and which commits it produced, so the
reviewer can find the work; it is not a reconstruction of what was asked, and
nothing here should be read as the wording of an instruction.

The authoritative record of what the project is meant to do remains `PLAN.md`,
`CLAUDE.md` and `docs/CONVENTIONS_RESOLVED.md`. The review files in `review/`
record what each session actually produced.

## Sessions

| session | file | commits |
|---|---|---|
| 0 — plan and scaffolding | not saved | `c2a83b7`, `063f430` |
| 1 — foundation and data | not saved | `ff81cbf`, `1fb67ac`, `7cbe025`, `ff9e132`, `61ac025`, `24322a6`, `89b9992`, `c468292`, `327b8db`, `3f26374`, `da2c490` |
| 2 — features | not saved | `b698ca1`, `4cfcc1f`, `9c39d6c`, `7fdfbff`, `4e652e3` |
| 3 — classifiers | not saved | `00b918f`, `3b4c4f0`, `7faad6e`, `7830040`, `26baae4`, `1f5e09e`, `5ca844c`, `7ddc1df`, `67a15b8`, `e12f2cb` |
| — (housekeeping) | not saved | `13730b1` |
| 3b — section 3 revision | `03b_section_3_revision.md` | this session |

Notes on the sessions that did not run straight through, from their review
files and commit messages:

- **Session 1** stopped twice. `7cbe025` records a stop before step 1.3 because
  `FRED_API_KEY` was not set; `24322a6` records steps 1.5 and 1.6 completed out
  of order while 1.3, 1.4 and 1.7 still waited on that key. The key arrived and
  the section finished at `3f26374`. `da2c490` is a post-review fix to the
  as-of rule for missing releases.
- **Session 3** ran straight through, steps 3.1 to 3.8, and is reviewed in
  `review/section_3.md`. The revision `03b` supersedes part of it: see the
  Revision section of that file.

## Numbering from here

`NN` is the section number, zero-padded, with a letter suffix for a revision of
an already-built section: `03b` is the revision of section 3. The next new
section is `04`.
