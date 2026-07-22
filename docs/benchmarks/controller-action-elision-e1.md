# Controller action-elision E1 screening run

Captured: 2026-07-22

> Subsequent product decision: transcript-shaped renderings were removed. pb now uses intrinsic,
> truthful controller blocks only; this report preserves the historical experiment.

Plan: [Deterministic controller actions](../controller-action-elision-plan.md)

This first preserved real-model screen tests the fresh-review observation path. It is qualification
evidence for the hidden harness implementation, not a prompt-representation promotion decision.
The checked-in `fix_average_divisor` TC3 fixture was prepared into four fresh Git workspaces and run
sequentially with the same model, contract, task, generation limit, sampling, and seed.

## Locked configuration

| Field | Value |
| --- | --- |
| Model | `hf://mlx-community/Qwen3-Coder-Next-4bit` |
| Backend | local FlashMoe `flashmoe-v2-mlxq4`, K=10 |
| Context | 131,072 tokens |
| Generation | 1,024 maximum new tokens per model turn |
| Sampling | temperature 0, top-k 1, seed 0 |
| Fixture | `fix_average_divisor` |
| Contract | exact `average.mjs` scope, required Deno regression, fresh review, semantic commit, clean worktree |
| Execution | sequential on the same host; energy is measured but not a controlled power trial |

The model chose two planning reads in every arm before an accepted plan existed. pb correctly left
those model-authored reads native. The experimental difference began at fresh code review: native
required a model `inspect_change` call, while each controller arm injected a full current
controller observation and exposed the independent review terminal immediately.

## Results

| Arm | Verified | Model calls | Review steps | Prompt tokens | Generated | Wall | Energy | Controller prompt bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | yes | 8 | 2 | 37,003 | 919 | 366,547 ms | 4.448 Wh | 0 |
| Controller block | yes | 7 | 1 | 32,574 | 882 | 320,129 ms | 4.036 Wh | 1,590 |
| Disclosed tool transcript | yes | 7 | 1 | 32,745 | 964 | 338,007 ms | 3.842 Wh | 1,781 |
| Compatibility tool transcript | yes | 7 | 1 | 32,676 | 1,004 | 337,270 ms | 4.043 Wh | 1,514 |

Against native, every controller arm removed one of eight model invocations and one of two review
steps. Controller block reduced wall time by 12.7%, prompt tokens by 12.0%, and measured energy by
9.3%. Disclosed transcript reduced wall time by 7.8%, prompt tokens by 11.5%, and measured energy by
13.6%. Compatibility transcript reduced wall time by 8.0%, prompt tokens by 11.7%, and measured
energy by 9.1%. One sequential run per arm is insufficient to rank latency or energy.

All four runs produced the same `average.mjs` SHA-256
`a7ee67d8d1c9e4367f2f27481e670fe2a825f3cec6605b30dc1f920beef97910` and the same binary diff
SHA-256 `356ca306c0e4b838346462e98fab457afe4a3e5716d38e39408395d9b1c37fd1`. Independent reruns of
`deno test average.test.mjs` passed. Each workspace changed only `average.mjs`, ended clean, and had
one task-owned semantic commit on top of its fixture baseline.

## Provenance and containment audit

- Native emitted one model `inspect_change` call and no controller observation.
- Each controller arm emitted one durable `controller_observation` with
  `actual_origin=controller`, `operation=inspect_change`, `coverage=full`, and authority effects
  limited to prompt context and review coverage.
- Controller arms emitted zero model `inspect_change` calls. Their prompt-only compatibility calls
  therefore did not appear as model-authored tool events.
- Every arm still required a model-authored `submit_code_review` verdict, the trusted regression,
  managed commit, and clean-worktree gate.
- No arm had a rejected workflow action, repair cycle, evidence invalidation, forbidden mutation,
  false completion, or false model attribution.

The human-readable journal originally omitted these additive experiment counters even though the
run index preserved them. The implementation was corrected after this audit so future journals
show rendering, controller observation count and prompt bytes, coverage, mutations, and closures.

## Limitations and decision

This fixture does not qualify accepted-plan read elision: the model read the target during planning,
and current small-file evidence then carried into implementation. Scripted end-to-end coverage
proves the no-model-read path, but a real-model read-rendering comparison still needs a fixture or
restored checkpoint that begins with an accepted plan and no current read evidence.

The review observation payloads were also not byte-identical. Their semantic file, diff, check
status, and output were identical, but volatile check durations made the rendered results 1,401 or
1,402 bytes and changed their range hashes. That violates the locked E1 representation-only
comparison requirement, so this run cannot select among controller block, disclosed transcript,
and compatibility transcript.

The valid native control was also re-prepared after the sandbox-only attempt failed, so its initial
fixture commit OID differs from the three controller arms even though its task bytes, tree, prompt
token count through planning, contract, and final binary diff are identical. A promotion-quality
runner must clone one immutable prepared baseline for every arm.

The safe interim decision is therefore:

1. keep all rendering choices hidden and harness-only;
2. treat the one-call reduction and truthful containment as positive implementation evidence;
3. do not promote or prefer the undisclosed compatibility transcript;
4. add a byte-locked accepted-plan read fixture before the repeated two-arm series; and
5. use controller block as the conservative reference arm unless later repeated evidence shows a
   material transcript advantage.

## Preserved evidence

| Classification | Scratch root |
| --- | --- |
| Valid native control | `/private/tmp/pb-action-elision-e1-20260722-native-metal` |
| Valid controller block | `/private/tmp/pb-action-elision-e1-20260722-controller-block` |
| Valid disclosed transcript | `/private/tmp/pb-action-elision-e1-20260722-disclosed-tool-transcript` |
| Valid compatibility transcript | `/private/tmp/pb-action-elision-e1-20260722-compatibility-tool-transcript` |
| Excluded experiment error | `/private/tmp/pb-action-elision-e1-20260722-native` |

The excluded run stopped before inference because the filesystem sandbox exposed no default Metal
device. It changed no task files and is retained as an experiment-error record. The four valid runs
used explicit host Metal access and retained cumulative events, per-run events, journals,
checkpoints, run indexes, Git history, and final workspaces.
