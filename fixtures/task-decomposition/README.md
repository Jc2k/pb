# Task decomposition corpus

This corpus locks the controller outcomes required by the Task decomposition design before the
runtime types exist. It separates deterministic validation failures from semantic review failures
and records the expected dispatch boundary for accepted plans.

Run its schema and coverage check with:

```bash
deno task test:task-decomposition
```

`corpus.json` is test input, not model output. Later implementation Tasks must run these same cases
through the production validator and preserve the expected outcomes. Numeric values inside the
`model_authored_numeric_budget` case are deliberately forbidden proposal fields.
