# Harness review rubric

Rank the behavior of pb, not the aesthetic quality of the model's artifact. First classify the cause;
then assign priority by impact on a trustworthy harness experiment.

## Classification

- **pb defect:** parser, tool, safety, persistence, completion/review gate, reporting, recovery, resource,
  or termination behavior violated the explicit experiment contract.
- **model limitation:** the model produced poor code, weak design, malformed calls, unsupported claims,
  or instruction drift that pb safely bounded and reported.
- **experiment error:** the profile, prompt, quoting, model, budget, or supervisor action could not test
  the intended property.
- **positive evidence:** pb correctly rejected, contained, persisted, or reported problematic output.

A CSS/JavaScript bug, weak visual design, or incomplete artifact is not a pb defect by itself. Escalate
it only when pb skipped a required check, accepted an unsupported completion/review, reported false
state, or allowed a forbidden effect.

## Priority

- **P0 — invalid or unsafe experiment:** crash, data loss, escaped mutation, unbounded resource use,
  completion bypass, or false success under an explicit acceptance contract.
- **P1 — harness correctness or reliability:** reproducible tool/parser failure, skipped required check,
  stale or unsupported review, incorrect journal/status/diff, missing required commit, or broken recovery.
- **P2 — efficiency or diagnostic quality:** avoidable turns, repeated inference, context blow-up, poor
  corrective feedback, weak observability, or costly behavior that remains bounded and truthful.
- **P3 — ergonomics or hypothesis:** useful polish, clearer presentation, or an improvement idea that
  still needs controlled evidence.

Model limitations and experiment errors may be important context, but do not assign them pb-defect
priority merely because they prevented a polished artifact.

## Evidence checklist

For each run ask:

1. Did pb execute only the intended parsed action and contain forbidden effects?
2. Were required reads, commands, tests, commit, and review actually evidenced?
3. Did completion and review gates use current workspace state?
4. Did status, diff, journal, and final reporting match Git and event state?
5. Did retry, loop, step, token, and resource bounds terminate predictably?
6. Is the observation caused by pb, the model, or the experiment configuration?

## Observation format

```markdown
### P1 — Short actionable title

- Classification: pb defect | model limitation | experiment error | positive evidence.
- Evidence: event, run path, commit, diff, check output, or deterministic reproduction.
- Impact: effect on safety, trust, diagnosability, or evaluation cost.
- Disposition: fixed commit, planned, accepted model limit, invalid configuration, or positive result.
- Recommendation: smallest action and measurable success condition.
```

Merge duplicate symptoms with one cause. Record positive evidence when a gate or safety mechanism works;
do not let a bad model response hide correct harness containment.

## Improvement plan

Order remaining pb work by priority and expected benefit. For each item state:

1. the observed harness problem and preserved run;
2. the proposed pb change;
3. a deterministic fixture or benchmark that proves improvement;
4. dependencies and risks;
5. whether another model run is necessary.

Prefer deterministic transcript/tool fixtures over expensive free-model reruns. Keep speculative model
quality work separate from validated harness defects.
