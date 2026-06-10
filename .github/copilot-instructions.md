When creating commits in this repository, always use semantic commit messages following Conventional Commits (for example: `feat: ...`, `fix: ...`, `chore: ...`).

## `pb init` command

`src/init.rs` implements the `pb init` subcommand. It inspects a project and automatically configures `.pb/environment.toml`.

**Whenever you add a new feature that introduces per-project configuration** (new file-based signals, new container options, new agent doc locations, new language ecosystems, etc.), you **must** also update `src/init.rs`:

1. Add a field to `ProjectInspection` for the new signal.
2. Detect the signal inside `inspect()`.
3. Update `suggest_environment()` (or a helper) to act on the new signal if it influences the environment config.
4. Add unit tests for the new detection and suggestion logic.

## Web UI

The web app is a PWA that supports "Add to Home Screen". When working on the web UI:

- The viewport meta tag must include `viewport-fit=cover` so the app renders correctly on devices with notches, Dynamic Island, and rounded corners (e.g. iPhones).
- The CSS must use `env(safe-area-inset-top/right/bottom/left)` to ensure content is not obscured by device hardware or the system status bar.
- Do not remove the `<meta name="apple-mobile-web-app-capable">` or `<link rel="manifest">` tags; they are required for standalone (no browser chrome) behaviour when installed to the home screen.
