use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct EnergySample {
    pub watts: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct EnergyEstimate {
    pub joules: f64,
    pub kwh: f64,
    pub average_watts: f64,
}

impl EnergySample {
    pub fn estimate_since(
        self,
        end: Option<EnergySample>,
        started: Instant,
    ) -> Option<EnergyEstimate> {
        let seconds = started.elapsed().as_secs_f64();
        if seconds <= 0.0 || !self.watts.is_finite() || self.watts <= 0.0 {
            return None;
        }
        let end_watts = end
            .map(|sample| sample.watts)
            .filter(|watts| watts.is_finite() && *watts > 0.0)
            .unwrap_or(self.watts);
        let average_watts = (self.watts + end_watts) / 2.0;
        let joules = average_watts * seconds;
        Some(EnergyEstimate {
            joules,
            kwh: joules / 3_600_000.0,
            average_watts,
        })
    }
}

pub fn sample() -> Option<EnergySample> {
    platform_sample()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn platform_sample() -> Option<EnergySample> {
    use std::sync::{Mutex, OnceLock};

    static SAMPLER: OnceLock<Mutex<macpow::metrics::Sampler>> = OnceLock::new();
    let sampler = SAMPLER.get_or_init(|| Mutex::new(macpow::metrics::Sampler::new(250)));
    let snapshot = sampler.lock().ok()?.snapshot();
    watts_from_metrics(&snapshot).map(|watts| EnergySample { watts })
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn watts_from_metrics(metrics: &macpow::types::Metrics) -> Option<f64> {
    [
        metrics.sys_power_w,
        metrics.soc.total_w,
        metrics.adapter_power_w,
    ]
    .into_iter()
    .map(f64::from)
    .find(|watts| watts.is_finite() && *watts > 0.0)
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn platform_sample() -> Option<EnergySample> {
    None
}

#[cfg(test)]
mod tests {
    use super::EnergySample;
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn estimates_energy_from_average_power_and_duration() {
        let started = Instant::now();
        thread::sleep(Duration::from_millis(1));
        let estimate = EnergySample { watts: 100.0 }
            .estimate_since(Some(EnergySample { watts: 200.0 }), started)
            .expect("positive power should estimate energy");
        assert!(estimate.joules > 0.0);
        assert!((estimate.kwh - estimate.joules / 3_600_000.0).abs() < f64::EPSILON);
        assert_eq!(estimate.average_watts, 150.0);
    }

    #[test]
    fn ignores_invalid_power_samples() {
        let started = Instant::now();
        assert!(
            EnergySample { watts: 0.0 }
                .estimate_since(None, started)
                .is_none()
        );
        assert!(
            EnergySample { watts: f64::NAN }
                .estimate_since(None, started)
                .is_none()
        );
    }
}
