# TC1 verified task-completion qualification

Captured: 2026-07-21

Model: native `mlx-community/Qwen3-Coder-Next-4bit`, affine Q4, 48 layers, 512 experts per
layer, K=10, approximately 40.5 GiB resident expert corpus

Post-fix source: `862233f6099bb421803698d86d95aa5623e04d7f`
(`fix: report strict completion evidence`)

Fixture: `fixtures/harness-task-completion/tc1-task.txt` with
`fixtures/harness-task-completion/tc1-contract.json`

## Hypothesis and contract

TC1 tests whether the native model can use pb's harness-owned all-create work-unit controller to
finish a complete strict workflow. The accepted plan must create two exact files. While either is
missing, pb binds `write_file.path` to the first missing accepted-plan path and withholds the
implementation terminal.

Success requires:

- only `alpha.txt` and `beta.txt` change;
- both exact content checks pass;
- fresh code review reads both final paths and cites current check evidence;
- pb creates a semantic task-owned commit;
- the final worktree is clean; and
- the run exits zero with `contract_status=satisfied` and `verified_completed=true`.

The locked model settings were temperature 0, top-k 1, seed 0, 1,024 maximum generated tokens,
131,072 context tokens, and four model-driven steps per strict stage. Every run used a new preserved
scratch root.

## Initial qualification and reporting fix

Initial run `1784667572614-75379-0` on fixture source
`2c308b616da9c5a823b779af95aed0b7c0766f65` completed the full contract in 561,696 ms, seven model
calls, 35,013 prompt tokens, 1,115 generated tokens, and 7.24 Wh. It produced task commit
`f4da7a5a925f82a6c129850f284ff2731d8568d9`; the exact check, commit identity, delta, and clean
worktree passed independent review.

The successful run exposed two diagnostic defects, not completion defects. Strict delivery recorded
the executed required check but left `checks_planned` empty because that audit field depended on a
legacy handoff event. The journal also classified proactive next-path work-unit guidance as a model
limitation. Commit `862233f6` records every observed strict check in the planned audit and classifies
the deterministic guidance as positive evidence. A scripted event-journal fixture locks both
behaviors.

## Locked post-fix series

Trials 2–4 used the identical post-fix source, release binary, model, sampling, task, contract, and
limits. They ran sequentially on the same Mac, so the series proves functional repeatability but
must not be interpreted as three independent cold-start performance samples.

| Trial | Run ID | Task commit | Verified | Model calls | Prompt | Generated | Wall ms | Energy Wh |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | `1784668516949-33851-0` | `56443bb6d7db99c4d0fb8f9a5fdd16bf127d7877` | yes | 7 | 35,013 | 1,131 | 435,365 | 4.98 |
| 3 | `1784668983662-35797-0` | `b9bcdd7f6ba0cef3824256e7ee439495807eab7b` | yes | 7 | 35,013 | 1,115 | 317,781 | 3.90 |
| 4 | `1784669344810-37478-0` | `604c90de41db636b7729bd0beacec65f5f21830b` | yes | 7 | 35,013 | 1,115 | 308,619 | 3.97 |

Post-fix aggregate:

- verified completion: 3/3;
- false verified completion and forbidden mutation: 0;
- stage shape: one planning, one plan-review, three implementation, and two code-review calls in
  every trial;
- median wall time: 317,781 ms; mean wall time: 353,922 ms;
- median energy: 3.97 Wh; mean energy: 4.28 Wh;
- mean generated tokens: 1,120; and
- independent contract check, commit-at-`HEAD`, two-path delta, and clean-worktree audit: 3/3.

Disk-prefix reuse was active for four calls in trial 2 and every call in trials 3 and 4. Fresh
prefill fell from 22,163 tokens in trial 2 to 13,133 in trials 3 and 4 while each rendered transcript
still totaled 35,013 prompt tokens.

## Findings

1. **Positive evidence — next-path control works with the native model.** Each run accepted the same
   two-step plan, bound the first implementation action to `alpha.txt`, advanced only after its
   atomic creation, bound the second action to `beta.txt`, and withheld implementation submission
   until both paths existed.
2. **Positive evidence — the complete acceptance chain is achievable.** The model reached trusted
   checks, fresh review, managed semantic commit, a clean worktree, and externally verified
   completion without a repair or rejected action.
3. **P2 efficiency — workflow overhead remains high for a tiny task.** Even with prefix reuse, the
   post-fix median was about 5.3 minutes and 3.97 Wh for 49 output bytes. The semantic stages are
   behaving correctly, but acceptance skeletons, useful independent creation batches, and smaller
   stable stage prefixes should be evaluated before scaling the corpus.
4. **P2 diagnostic defect — fixed.** Strict planned-check reporting and work-unit classification are
   truthful in every post-fix journal.

## Disposition

TC1 is qualified and remains a regression gate. The next artifact-quality experiment is TC2's
dependency-free slugify task with an independent behavior oracle and model-authored tests. The
browser-game rerun remains deferred until the smaller useful-coding gate identifies whether the
next failure is output shaping, semantic implementation, review, or efficiency.
