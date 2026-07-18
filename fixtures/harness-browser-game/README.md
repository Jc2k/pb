# Offline browser-game harness fixture

Copy `check-offline-web.sh` into the root of a fresh harness scratch workspace and use
`contract.json` as the trusted contract. The named `offline_dependencies` check uses only POSIX
shell, `find`, and `grep`; it rejects HTTP(S), protocol-relative, npm, and JSR references in HTML,
CSS, and JavaScript without downloading project dependencies.

The checker itself is fixture infrastructure, not an allowed task output. Add a separate named
logic check appropriate to the task before using this contract for an acceptance run.
