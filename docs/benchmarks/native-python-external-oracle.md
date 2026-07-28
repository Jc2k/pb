# Native Python external-oracle interface

**Status: Design record; checked CI interface, no runtime provider**

The native Python layer uses exact-pinned Astral `ty` internally. A second checker can be useful for
finding false rejections, missed errors, and profile assumptions, but disagreement between two type
checkers is evidence to investigate—not permission to constrain generation. This interface keeps
that research path outside the pb binary and hidden harness.

## Boundary

`scripts/python-semantic-oracle.ts` is a Deno development tool. It has no Cargo dependency and does
not download or invoke a Python checker. The ordinary CI job tests its materialization and artifact
validation using only Deno and Git. A research or CI job may independently provision an exact
checker release and run it over the materialized trees.

The checker must not call back into pb or affect token masks. Its result contains case identifiers,
verdicts, stable diagnostic identifiers, provider identity, and configuration identity—never source
excerpts, messages, or file contents. Unknown fields fail closed. The result is bound to the exact
corpus SHA-256 and must cover every case exactly once.

## Materialize complete states

Use a new output directory:

```sh
deno run --allow-read --allow-write --allow-run=git \
  scripts/python-semantic-oracle.ts materialize \
  --corpus fixtures/control-collar/semantic-python-v1.json \
  --output /tmp/python-oracle-run
```

The command creates:

- `baseline/project`, the unchanged first-party project;
- `dependencies`, the frozen static dependency inputs;
- `cases/<id>/project`, one complete post-mutation project for every case; and
- `oracle-request-v1.json`, the content-free, corpus-bound request manifest.

All four mutation shapes are materialized. Canonical patches must pass Git's independent check and
application, edited text must match exactly once, creates cannot overwrite, and replacements require
a regular existing file. The resulting tree rejects symlinks, special files, traversal, ambiguity,
and file/byte overflow.

An external runner must analyze the baseline and each complete candidate with the same pinned
provider/configuration/dependency roots. It reports newly introduced diagnostic debt rather than
requiring a globally error-free baseline. This is important for the checked baseline-debt cases and
for realistic public projects.

## Result contract

The independently produced result has this versioned shape:

```json
{
  "version": 1,
  "corpus_sha256": "<64 lowercase hex characters>",
  "provider": {
    "name": "independent_checker",
    "version": "<exact release>",
    "configuration_sha256": "<64 lowercase hex characters>"
  },
  "cases": [
    {
      "id": "annotated_invalid_argument",
      "outcome": "reject",
      "diagnostic_ids": ["stable-provider-rule"]
    }
  ]
}
```

`outcome` is `allow`, `reject`, or `unknown`. Diagnostic identifiers must be sorted, unique, and
bounded. A runner should use `unknown` when its profile, dependency view, or diagnostic mapping is
incomplete rather than guessing.

Compare it with the materialized request:

```sh
deno run --allow-read scripts/python-semantic-oracle.ts compare \
  --request /tmp/python-oracle-run/oracle-request-v1.json \
  --result /tmp/python-oracle-result.json
```

The report separates agreement, disagreement, and unknown case IDs. Comparison is audit-only by
default. `--require-agreement` exits nonzero for a deliberately curated corpus whose cross-checker
parity has already been reviewed; it must not be used to silently make an external checker the
production authority.

## Current evidence and next gate

`deno task test:python-oracle` proves deterministic 24-case materialization, every mutation shape,
baseline preservation, external-dependency placement, content-free exact-schema enforcement,
stale/incomplete artifact rejection, and deterministic disagreement/unknown classification. It runs
in ordinary CI.

No independently provisioned checker result or public-project corpus is checked in yet. Before an
external comparison can gate changes, record the exact provider/configuration identity, review every
initial disagreement, add an approved content-free baseline, and keep the provider installation in
the research job—not in pb's Cargo graph, release bundle, default setup, or hidden harness.
