# Documentation site

The documentation is an mdBook: a Rust static-site generator over the Markdown in `docs/`. The
curated user and architecture chapters sit alongside the detailed engineering records so design
history remains searchable without dominating the main product story.

## Tooling

CI pins mdBook `0.5.3`. Install the same version locally:

```bash
cargo install mdbook --version 0.5.3 --locked
```

Deno is used only for the deterministic rendered-site check already shared by the web project.

## Build and test

```bash
deno task build:docs
deno task test:docs
```

The build writes `site/`, which is ignored by Git. `test:docs` builds from scratch, walks the
rendered output, checks every local page, asset, and fragment link, validates the manifest icons,
and asserts the required safe-area/PWA metadata on every HTML page. It intentionally does not make
network requests to external links.

For a local preview with live rebuilds:

```bash
mdbook serve --open
```

## Structure

- `docs/user/` contains task-oriented guidance and practical privacy controls.
- `docs/architecture/` explores shipped behavior through workflows, authority, privacy, and user
  contracts.
- top-level long-form plans remain engineering records.
- `docs/benchmarks/` preserves evaluation evidence linked from its owning plan.
- `docs/theme/` contains only pb-specific overrides; mdBook retains ownership of the rest of its
  accessible navigation and search behavior.
- `book.toml` owns renderer settings, repository links, and the `/pb/` Pages base path.
- `docs/SUMMARY.md` is the navigation and inclusion contract.

## Writing rules

Curated pages should describe current product behavior. Use the status vocabulary from the site
home page: **Shipped**, **Configurable**, and **Design record**. Do not promote a planned hardening
item into a security claim merely because it appears in an engineering record.

When behavior changes:

1. Update the relevant user-facing task page.
2. Update the architecture contract that explains why it changed.
3. Keep the detailed source-of-truth plan aligned when its critical convention requires it.
4. Add the page to `SUMMARY.md` if it introduces a new chapter.
5. Run `deno task test:docs` before committing.

## Continuous delivery

The `docs` GitHub Actions job runs on every pushed branch and pull-request commit. It builds and
tests the site independently of the Rust application test job. On `main`, its Pages artifact waits
for the semantic-release stage and is then deployed by the same `ci-release` workflow. Documentation
changes therefore ship from the normal release pipeline even when a commit does not create a new
binary version.
