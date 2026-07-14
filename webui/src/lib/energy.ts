import type { SessionMetricsSnapshot } from "../types";

export function metricEnergyJoules(
  metrics: SessionMetricsSnapshot,
): number | undefined {
  if (metrics.total_energy_joules !== undefined) {
    return metrics.total_energy_joules;
  }
  // Current snapshots never promote overlapping diagnostic spans when the
  // authoritative outer scope was unavailable (for example, another pb
  // process owned the meter). The fallback below is only for legacy records.
  if ((metrics.wall_runtime_ms ?? 0) > 0) return undefined;
  const legacy = (metrics.llm_energy_joules ?? 0) +
    (metrics.tool_energy_joules ?? 0);
  if (legacy > 0) return legacy;
  const legacyKwh = (metrics.llm_energy_kwh ?? 0) +
    (metrics.tool_energy_kwh ?? 0);
  return legacyKwh > 0 ? legacyKwh * 3_600_000 : undefined;
}

export function metricRuntimeMs(metrics: SessionMetricsSnapshot): number {
  return metrics.wall_runtime_ms ||
    metrics.llm_runtime_ms + metrics.tool_runtime_ms;
}

export function formatEnergy(joules?: number): string {
  if (joules === undefined || !Number.isFinite(joules) || joules < 0) {
    return "Not available";
  }
  if (joules < 1) return `${joules.toFixed(2)} J`;
  if (joules < 1_000) {
    return `${joules.toFixed(joules < 10 ? 2 : joules < 100 ? 1 : 0)} J`;
  }
  const wattHours = joules / 3_600;
  if (wattHours < 1_000) {
    return `${
      wattHours.toFixed(wattHours < 10 ? 2 : wattHours < 100 ? 1 : 0)
    } Wh`;
  }
  const kwh = wattHours / 1_000;
  return `${kwh.toFixed(kwh < 10 ? 3 : kwh < 100 ? 2 : 1)} kWh`;
}

export function formatPower(watts?: number): string {
  if (watts === undefined || !Number.isFinite(watts) || watts < 0) {
    return "Not available";
  }
  return `${watts.toFixed(watts < 10 ? 2 : watts < 100 ? 1 : 0)} W`;
}

export function ledEquivalent(joules: number): string | undefined {
  if (!Number.isFinite(joules) || joules <= 0) return undefined;
  const seconds = joules / 10; // A fixed, explicit 10 W LED bulb.
  if (seconds < 90) {
    return `${formatAmount(seconds)} ${plural(seconds, "second")}`;
  }
  const minutes = seconds / 60;
  if (minutes < 90) {
    return `${formatAmount(minutes)} ${plural(minutes, "minute")}`;
  }
  const hours = minutes / 60;
  return `${formatAmount(hours)} ${plural(hours, "hour")}`;
}

function formatAmount(value: number): string {
  if (value >= 10) return value.toFixed(0);
  if (value >= 1) return Number(value.toFixed(1)).toString();
  return value.toFixed(2);
}

function plural(value: number, singular: string): string {
  return Math.abs(value - 1) < 0.05 ? singular : `${singular}s`;
}
