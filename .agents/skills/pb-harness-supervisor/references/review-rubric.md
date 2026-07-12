# Harness review rubric

Use this rubric after each `pb harness agent` run. Rank observed behavior by user impact, not by how
interesting it is internally. Keep facts separate from inferences.

## Priority

- **P0 — invalid experiment:** crash, data loss, unsafe behavior, false completion, unusable artifact,
  harness bypass, or a blocker that prevents evaluating the requested task.
- **P1 — major correctness or reliability:** reproducible agent failure, incorrect tool behavior,
  missing required commit, dirty final workspace, unsupported test claim, or a serious recovery gap.
- **P2 — quality or efficiency:** avoidable latency, excessive tool/model work, weak diagnostics,
  incomplete tests or documentation, or output that works but falls materially short of the task.
- **P3 — polish or opportunity:** non-blocking ergonomics, clearer presentation, useful metrics, or an
  improvement hypothesis that still needs evidence.

## Observation format

Use one entry per distinct issue:

```markdown
### P1 — Short actionable title

- Evidence: event type or line, scratch path, commit, diff, test output, or reproduced behavior.
- Impact: what failed or became harder, slower, or less trustworthy.
- Disposition: fixed in pb commit, generated-output issue, planned, not reproducible, or accepted.
- Recommendation: smallest next action and a measurable success condition.
```

Merge duplicate symptoms that share one cause. Promote or demote a finding when later runs supply
better evidence, and record why. Never infer a pb defect solely from an imperfect generated artifact;
inspect the event/tool trace and reproduce the behavior first.

## Improvement plan

Order remaining work by priority, then expected benefit. For each item state:

1. the observed problem and supporting run;
2. the proposed pb change;
3. the verification or benchmark that would prove improvement;
4. dependencies or risks;
5. whether it is required before another harness experiment.

Keep speculative ideas separate from validated findings. A successful run can still produce P2/P3
improvements, but it must not leave a known P0/P1 issue disguised as future work.
