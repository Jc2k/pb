use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergySource {
    SmcSystemTotal,
    PowerTelemetry,
    BatteryVoltageCurrent,
}

impl EnergySource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SmcSystemTotal => "smc_system_total",
            Self::PowerTelemetry => "power_telemetry",
            Self::BatteryVoltageCurrent => "battery_voltage_current",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EnergySample {
    pub watts: f64,
    integrated: Option<IntegratedEnergy>,
}

#[derive(Debug, Clone, Copy)]
struct IntegratedEnergy {
    workload_joules: f64,
    adjusted_joules: f64,
    gross_joules: f64,
    measured_seconds: f64,
    generation: u64,
    source: EnergySource,
    display_excluded: bool,
    baseline_available: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct EnergyEstimate {
    /// Best estimate of incremental task energy. When an idle baseline is not
    /// available this conservatively falls back to display-adjusted system energy.
    pub joules: f64,
    pub kwh: f64,
    pub average_watts: f64,
    pub gross_joules: f64,
    pub adjusted_joules: f64,
    pub measured_seconds: f64,
    pub elapsed_seconds: f64,
    pub coverage: f64,
    pub complete: bool,
    pub source: EnergySource,
    pub display_excluded: bool,
    pub baseline_applied: bool,
    /// False when another pb process already owns the system power meter. In
    /// that case the estimate is deliberately not attributed to this task.
    pub exclusive: bool,
}

impl EnergyEstimate {
    fn from_joules(
        joules: f64,
        adjusted_joules: f64,
        gross_joules: f64,
        measured_seconds: f64,
        elapsed_seconds: f64,
        source: EnergySource,
        display_excluded: bool,
        baseline_applied: bool,
        complete: bool,
        exclusive: bool,
    ) -> Option<Self> {
        if !joules.is_finite() || joules < 0.0 || elapsed_seconds <= 0.0 {
            return None;
        }
        let coverage = (measured_seconds / elapsed_seconds).clamp(0.0, 1.0);
        Some(Self {
            joules,
            kwh: joules / 3_600_000.0,
            average_watts: joules / elapsed_seconds,
            gross_joules,
            adjusted_joules,
            measured_seconds,
            elapsed_seconds,
            coverage,
            complete: complete && coverage >= 0.98,
            source,
            display_excluded,
            baseline_applied,
            exclusive,
        })
    }
}

impl EnergySample {
    pub fn estimate_since(
        self,
        end: Option<EnergySample>,
        started: Instant,
    ) -> Option<EnergyEstimate> {
        self.estimate_since_with_ownership(end, started, true)
    }

    fn estimate_since_with_ownership(
        self,
        end: Option<EnergySample>,
        started: Instant,
        exclusive: bool,
    ) -> Option<EnergyEstimate> {
        if !exclusive {
            return None;
        }
        let elapsed_seconds = started.elapsed().as_secs_f64();
        if elapsed_seconds <= 0.0 || !self.watts.is_finite() || self.watts < 0.0 {
            return None;
        }

        if let Some(start) = self.integrated {
            let end = end?.integrated?;
            let workload_joules = end.workload_joules - start.workload_joules;
            let adjusted_joules = end.adjusted_joules - start.adjusted_joules;
            let gross_joules = end.gross_joules - start.gross_joules;
            let measured_seconds = end.measured_seconds - start.measured_seconds;
            let same_provenance = start.generation == end.generation
                && start.source == end.source
                && start.display_excluded == end.display_excluded;
            return EnergyEstimate::from_joules(
                workload_joules,
                adjusted_joules,
                gross_joules,
                measured_seconds,
                elapsed_seconds,
                end.source,
                end.display_excluded,
                start.baseline_available && end.baseline_available,
                same_provenance,
                exclusive,
            );
        }

        if self.watts <= 0.0 {
            return None;
        }
        let end = end?;
        let end_watts = (end.watts.is_finite() && end.watts > 0.0).then_some(end.watts)?;
        let average_watts = (self.watts + end_watts) / 2.0;
        let joules = average_watts * elapsed_seconds;
        EnergyEstimate::from_joules(
            joules,
            joules,
            joules,
            elapsed_seconds,
            elapsed_seconds,
            EnergySource::SmcSystemTotal,
            false,
            false,
            false,
            exclusive,
        )
    }
}

/// An outer task/turn measurement scope. Only one pb process can own the
/// system-wide meter at a time, avoiding false per-task attribution when two
/// independent pb processes overlap.
pub struct EnergyScope {
    started: Instant,
    start: Option<EnergySample>,
    exclusive: bool,
    finished: bool,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    _ownership: Option<macos::MeasurementOwnership>,
}

impl EnergyScope {
    pub fn finish(mut self) -> Option<EnergyEstimate> {
        // Include a short, bounded tail for fans and storage/network work that
        // is causally triggered by the task but settles just after its final
        // model/tool action.
        if self.exclusive {
            std::thread::sleep(std::time::Duration::from_millis(750));
        }
        let estimate = self.start.and_then(|sample| {
            sample.estimate_since_with_ownership(sample_end(), self.started, self.exclusive)
        });
        self.end();
        estimate
    }

    fn end(&mut self) {
        if self.finished {
            return;
        }
        platform_end_scope();
        self.finished = true;
    }
}

impl Drop for EnergyScope {
    fn drop(&mut self) {
        self.end();
    }
}

pub fn scope() -> EnergyScope {
    let ownership = platform_begin_scope();
    let exclusive = platform_scope_is_exclusive(&ownership);
    EnergyScope {
        started: Instant::now(),
        start: sample(),
        exclusive,
        finished: false,
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        _ownership: ownership,
    }
}

pub fn sample() -> Option<EnergySample> {
    platform_sample()
}

fn sample_end() -> Option<EnergySample> {
    sample()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod macos;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn platform_sample() -> Option<EnergySample> {
    macos::sample().map(|snapshot| EnergySample {
        watts: snapshot.workload_watts,
        integrated: Some(IntegratedEnergy {
            workload_joules: snapshot.workload_joules,
            adjusted_joules: snapshot.adjusted_joules,
            gross_joules: snapshot.gross_joules,
            measured_seconds: snapshot.measured_seconds,
            generation: snapshot.generation,
            source: snapshot.source,
            display_excluded: snapshot.display_excluded,
            baseline_available: snapshot.baseline_available,
        }),
    })
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn platform_begin_scope() -> Option<macos::MeasurementOwnership> {
    macos::begin_scope()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn platform_scope_is_exclusive(ownership: &Option<macos::MeasurementOwnership>) -> bool {
    ownership.is_some()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn platform_end_scope() {
    macos::end_scope();
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn platform_sample() -> Option<EnergySample> {
    None
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn platform_begin_scope() {}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn platform_scope_is_exclusive(_: &()) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn platform_end_scope() {}

#[cfg(test)]
mod tests {
    use super::{EnergySample, EnergySource, IntegratedEnergy};
    use std::thread;
    use std::time::{Duration, Instant};

    fn integrated(joules: f64, generation: u64) -> IntegratedEnergy {
        IntegratedEnergy {
            workload_joules: joules,
            adjusted_joules: joules + 1.0,
            gross_joules: joules + 2.0,
            measured_seconds: joules / 10.0,
            generation,
            source: EnergySource::SmcSystemTotal,
            display_excluded: true,
            baseline_available: true,
        }
    }

    #[test]
    fn estimates_energy_from_average_power_and_duration() {
        let started = Instant::now();
        thread::sleep(Duration::from_millis(1));
        let estimate = EnergySample {
            watts: 100.0,
            integrated: None,
        }
        .estimate_since(
            Some(EnergySample {
                watts: 200.0,
                integrated: None,
            }),
            started,
        )
        .expect("positive power should estimate energy");
        assert!(estimate.joules > 0.0);
        assert!((estimate.average_watts - 150.0).abs() <= f64::EPSILON * 150.0);
        assert!(!estimate.complete);
    }

    #[test]
    fn prefers_integrated_energy_and_reports_breakdown() {
        let started = Instant::now();
        thread::sleep(Duration::from_millis(1));
        let estimate = EnergySample {
            watts: 5.0,
            integrated: Some(integrated(10.0, 1)),
        }
        .estimate_since(
            Some(EnergySample {
                watts: 5.0,
                integrated: Some(integrated(11.0, 1)),
            }),
            started,
        )
        .expect("integrated energy should produce an estimate");
        assert_eq!(estimate.joules, 1.0);
        assert_eq!(estimate.adjusted_joules, 1.0);
        assert_eq!(estimate.gross_joules, 1.0);
        assert!(estimate.display_excluded);
        assert!(estimate.baseline_applied);
    }

    #[test]
    fn integrated_measurement_can_start_at_the_idle_baseline() {
        let started = Instant::now();
        thread::sleep(Duration::from_millis(1));
        let estimate = EnergySample {
            watts: 0.0,
            integrated: Some(integrated(10.0, 1)),
        }
        .estimate_since(
            Some(EnergySample {
                watts: 5.0,
                integrated: Some(integrated(11.0, 1)),
            }),
            started,
        )
        .expect("an idle start is valid for cumulative integration");
        assert_eq!(estimate.joules, 1.0);
    }

    #[test]
    fn marks_provenance_changes_incomplete_without_losing_measured_energy() {
        let started = Instant::now();
        thread::sleep(Duration::from_millis(1));
        let estimate = EnergySample {
            watts: 5.0,
            integrated: Some(integrated(10.0, 1)),
        }
        .estimate_since(
            Some(EnergySample {
                watts: 5.0,
                integrated: Some(integrated(11.0, 2)),
            }),
            started,
        )
        .expect("measured energy remains useful across a marked transition");
        assert_eq!(estimate.joules, 1.0);
        assert!(!estimate.complete);
    }

    #[test]
    fn requires_an_end_snapshot_for_integrated_measurements() {
        let started = Instant::now();
        assert!(
            EnergySample {
                watts: 5.0,
                integrated: Some(integrated(10.0, 1))
            }
            .estimate_since(None, started)
            .is_none()
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "hardware smoke test"]
    fn live_task_scope_calibrates_and_reports_provenance() {
        let scope = super::scope();
        thread::sleep(Duration::from_secs(1));
        let estimate = scope
            .finish()
            .expect("this Apple Silicon Mac should expose a system power source");
        eprintln!(
            "task {:.3} J, gross {:.3} J, adjusted {:.3} J, {:.1}% coverage, source {}",
            estimate.joules,
            estimate.gross_joules,
            estimate.adjusted_joules,
            estimate.coverage * 100.0,
            estimate.source.as_str()
        );
        assert!(estimate.exclusive);
        assert!(estimate.baseline_applied);
        assert!(estimate.coverage >= 0.9);
        assert!(estimate.gross_joules >= estimate.adjusted_joules);
        assert!(estimate.adjusted_joules >= estimate.joules);
    }
}
