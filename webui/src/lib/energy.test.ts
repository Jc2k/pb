import { equal } from "node:assert/strict";
import {
  formatEnergy,
  ledEquivalent,
  metricEnergyJoules,
  metricRuntimeMs,
} from "./energy.ts";
import type { SessionMetricsSnapshot } from "../types/index.ts";

function currentMetrics(
  values: Partial<SessionMetricsSnapshot>,
): SessionMetricsSnapshot {
  return {
    llm_invocations: 0,
    llm_runtime_ms: 0,
    prompt_tokens: 0,
    generated_tokens: 0,
    tool_calls: 0,
    tool_runtime_ms: 0,
    cache_persistence_queued_checkpoints: 0,
    cache_persistence_completed_checkpoints: 0,
    cache_persistence_wall_ms: 0,
    cache_persistence_failures: 0,
    wall_runtime_ms: 0,
    display_energy_excluded: false,
    idle_baseline_applied: false,
    energy_complete: false,
    energy_exclusive: false,
    ...values,
    started_at_ms: values.started_at_ms ?? 0,
    ended_at_ms: values.ended_at_ms ?? 0,
  };
}

Deno.test("formatEnergy keeps ordinary task estimates visible without zero kWh", () => {
  equal(formatEnergy(0), "0.00 J");
  equal(formatEnergy(0.42), "0.42 J");
  equal(formatEnergy(42), "42.0 J");
  equal(formatEnergy(36_000), "10.0 Wh");
  equal(formatEnergy(3_600_000), "1.000 kWh");
});

Deno.test("LED comparison uses one explicit 10 W appliance and sane time units", () => {
  equal(ledEquivalent(38), "3.8 seconds");
  equal(ledEquivalent(600), "60 seconds");
  equal(ledEquivalent(36_000), "60 minutes");
  equal(ledEquivalent(216_000), "6 hours");
});

Deno.test("canonical task totals override overlapping diagnostic breakdowns", () => {
  const metrics = currentMetrics({
    llm_invocations: 2,
    llm_runtime_ms: 8_000,
    prompt_tokens: 10,
    generated_tokens: 5,
    tool_calls: 2,
    tool_runtime_ms: 9_000,
    wall_runtime_ms: 10_000,
    total_energy_joules: 100,
    llm_energy_joules: 80,
    tool_energy_joules: 70,
  });
  equal(metricEnergyJoules(metrics), 100);
  equal(metricRuntimeMs(metrics), 10_000);
});

Deno.test("current snapshots never promote diagnostic spans when task attribution is unavailable", () => {
  const metrics = currentMetrics({
    llm_invocations: 1,
    llm_runtime_ms: 1_000,
    prompt_tokens: 1,
    generated_tokens: 1,
    tool_calls: 1,
    tool_runtime_ms: 1_000,
    wall_runtime_ms: 1_500,
    llm_energy_joules: 20,
    tool_energy_joules: 20,
  });
  equal(metricEnergyJoules(metrics), undefined);
});
