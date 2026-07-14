# Power estimation on macOS

pb estimates the incremental, device-side energy of a task. It measures the
whole Mac and deliberately does not attribute power to a PID. The canonical
unit in events and persistence is joules; Wh and kWh are display units.

## Why the old estimator was wrong

The previous vendored `macpow` implementation was a stub rather than a working
sampler. Its configured sampling interval was unused, its SoC value was always
zero, and it copied one battery-or-adapter value into both system and adapter
fields. On mains, that value was adapter capacity rather than live system
consumption. While charging, taking the absolute value of battery current also
confused energy entering the battery with energy consumed by the computer.

Two endpoint readings were then averaged over an operation. That misses bursty
work between the endpoints. Independently measuring every parallel tool and a
parent sub-agent call also counted the same system interval multiple times.

## Measurement contract

Every top-level turn has one authoritative measurement scope. The reported
`total_energy_joules` is used for session and project totals. LLM and tool
measurements are diagnostics and are not added to produce that total:

1. `gross_energy_joules` is energy at the selected whole-device source.
2. `adjusted_energy_joules` removes independently measured display energy.
3. `total_energy_joules` removes the rolling idle baseline when one is
   available; otherwise it conservatively equals the display-adjusted value.

The event also records source, wall time, measured time, coverage, whether the
display and baseline exclusions applied, and whether one unchanged provenance
covered the turn. A source change or sensor gap preserves the measured joules
but marks the result partial instead of silently presenting it as complete.

Parallel MCP calls share one batch measurement. A sub-agent's leaf LLM/tool
diagnostics are retained, but the enclosing `sub_agent` tool span is not
measured again. A non-blocking inter-process lock allows only one pb process to
claim the system meter; an overlapping process reports task energy as
unavailable rather than duplicating the same whole-device energy.

## Device power on mains and battery

On Apple Silicon, pb selects the first usable reading from:

1. AppleSMC `PSTR`, the direct total-system power rail.
2. AppleSmartBattery `PowerTelemetryData.SystemLoad`, in milliwatts.
3. Battery voltage multiplied by negated signed current, only when external
   power is disconnected and the battery is actually discharging.

Adapter capacity, negotiated adapter power, and battery charging power are
never treated as consumption. In particular, `SystemPowerIn` includes input and
charging effects, while `SystemEnergyConsumed` is energy for a telemetry
interval rather than an instantaneous milliwatt reading. SMC decoding accepts
only known numeric types and reconnects after repeated read failures.

This is a device-side estimate on both mains and battery. It does not claim to
measure wall-socket energy: adapter conversion loss is outside the Mac's
telemetry and would need a separate smart-plug source.

## Exclusions

pb removes costs only when there is a defensible measurement boundary:

- Built-in display/backlight power from AppleSMC `PBwo` or `PDBR`.
- Internal and external display-controller domains from IOReport Energy Model
  channels beginning `DISP` or `DISPEXT`.
- The low 20th percentile of an idle rolling window, maintained separately for
  mains and battery. The baseline is frozen while a pb scope is active. A new
  one-shot process performs a short calibration before work begins.

The idle baseline is the safe way to remove persistent non-task costs such as
already-playing audio, keyboard illumination, attached USB devices, background
services, and normal idle SoC activity without inventing component models.

Networking, storage, memory, fans, and accelerator work triggered by the task
remain included. A short post-task tail captures immediate fan and I/O
settling. Audio, USB, keyboard, or network activity that starts during a task
also remains included because macOS does not expose a reliable independent
energy rail that proves it is unrelated. pb does not subtract TDP tables,
volume-based audio guesses, brightness formulas, or per-process utilization.

## Sampling and integration

The in-process sampler reads every 250 ms. It integrates the last observed
power with zero-order hold, capped at a two-second sample age. Snapshot polling
does not alter the result. Source, AC/battery mode, and exclusion changes advance
a provenance generation. SMC, IOReport, and registry failures fail closed.

## Persistence and presentation

Each continuation appends a timestamped turn record. Session totals are the sum
of turn records, not the latest turn, and project totals use the authoritative
task energy and wall runtime. “Today” uses each turn's start/end window and
apportions a turn that crosses local midnight.

The CLI and web UI choose J, Wh, or kWh by magnitude and expose gross,
display-adjusted, baseline-adjusted, source, coverage, and diagnostic breakdown
details. The playful comparison is intentionally one fixed reference: the time
a 10 W LED bulb would use the same energy. It changes only the time unit, never
the appliance or sentence semantics.

## Platform limitations

- The estimator is available on Apple Silicon macOS. Other platforms report it
  as unavailable.
- AppleSMC, IOReport, and `PowerTelemetryData` are undocumented interfaces and
  may change on future hardware. Unknown values are rejected rather than
  guessed.
- A battery-mode hardware check requires running the validation matrix while
  unplugged; unit tests cover signed-current and mode-selection behavior when a
  battery run is not available.
