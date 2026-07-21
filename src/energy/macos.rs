use super::EnergySource;
use std::collections::{HashMap, VecDeque};
use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::os::raw::{c_char, c_void};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);
const MAX_SAMPLE_AGE: Duration = Duration::from_secs(2);
const MAX_SANE_SYSTEM_WATTS: f64 = 1_000.0;
const BASELINE_WINDOW: usize = 240;
const MIN_BASELINE_SAMPLES: usize = 8;

pub(super) struct PowerSnapshot {
    pub workload_watts: f64,
    pub workload_joules: f64,
    pub adjusted_joules: f64,
    pub gross_joules: f64,
    pub measured_seconds: f64,
    pub generation: u64,
    pub source: EnergySource,
    pub display_excluded: bool,
    pub baseline_available: bool,
}

pub(super) fn sample() -> Option<PowerSnapshot> {
    sampler().snapshot()
}

fn sampler() -> &'static PowerSampler {
    static SAMPLER: OnceLock<PowerSampler> = OnceLock::new();
    SAMPLER.get_or_init(PowerSampler::start)
}

pub(super) struct MeasurementOwnership {
    _lock: File,
}

impl Drop for MeasurementOwnership {
    fn drop(&mut self) {
        unsafe {
            let _ = libc::flock(self._lock.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

pub(super) fn begin_scope() -> Option<MeasurementOwnership> {
    let ownership = acquire_measurement_lock();
    let sampler = sampler();
    if ownership.is_some() {
        sampler.wait_for_baseline(Duration::from_millis(2_250));
    }
    sampler.begin_scope();
    ownership
}

pub(super) fn end_scope() {
    sampler().end_scope();
}

fn acquire_measurement_lock() -> Option<MeasurementOwnership> {
    let user_id = unsafe { libc::geteuid() };
    let path = std::env::temp_dir().join(format!("pb-system-energy-{user_id}.lock"));
    acquire_measurement_lock_at(&path)
}

fn acquire_measurement_lock_at(path: &Path) -> Option<MeasurementOwnership> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .ok()?;
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    (result == 0).then_some(MeasurementOwnership { _lock: file })
}

struct PowerSampler {
    state: Arc<Mutex<IntegratedPower>>,
}

impl PowerSampler {
    fn start() -> Self {
        let state = Arc::new(Mutex::new(IntegratedPower::default()));
        let mut reader = PowerReader::new();
        if let Some(reading) = reader.read_power() {
            if let Ok(mut state) = state.lock() {
                state.observe(reading, Instant::now());
            }
        }

        let thread_state = Arc::clone(&state);
        let _ = std::thread::Builder::new()
            .name("pb-power-sampler".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(SAMPLE_INTERVAL);
                    let Some(reading) = reader.read_power() else {
                        continue;
                    };
                    if let Ok(mut state) = thread_state.lock() {
                        state.observe(reading, Instant::now());
                    }
                }
            });

        Self { state }
    }

    fn snapshot(&self) -> Option<PowerSnapshot> {
        self.state.lock().ok()?.snapshot(Instant::now())
    }

    fn begin_scope(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.advance(Instant::now());
            state.active_scopes = state.active_scopes.saturating_add(1);
        }
    }

    fn end_scope(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.advance(Instant::now());
            state.active_scopes = state.active_scopes.saturating_sub(1);
        }
    }

    fn wait_for_baseline(&self, timeout: Duration) {
        let started = Instant::now();
        while started.elapsed() < timeout {
            if self
                .state
                .lock()
                .ok()
                .is_some_and(|state| state.current_baseline_available())
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PowerMode {
    Mains,
    Battery,
    Unknown,
}

#[derive(Debug, Clone, Copy)]
struct PowerReading {
    gross_watts: f64,
    adjusted_watts: f64,
    workload_watts: f64,
    source: EnergySource,
    mode: PowerMode,
    display_excluded: bool,
    display_signature: u8,
    baseline_available: bool,
}

#[derive(Default)]
struct IntegratedPower {
    workload_joules: f64,
    adjusted_joules: f64,
    gross_joules: f64,
    measured_seconds: f64,
    generation: u64,
    integrated_at: Option<Instant>,
    last_sensor_at: Option<Instant>,
    reading: Option<PowerReading>,
    active_scopes: usize,
    mains_baseline: VecDeque<f64>,
    battery_baseline: VecDeque<f64>,
    baseline_reset_pending: bool,
    stale_generation_marked: bool,
}

impl IntegratedPower {
    fn observe(&mut self, mut reading: PowerReading, now: Instant) {
        let stale = self
            .last_sensor_at
            .is_some_and(|last| now.saturating_duration_since(last) > MAX_SAMPLE_AGE);
        self.advance(now);
        let provenance_changed = self.reading.is_some_and(|previous| {
            previous.source != reading.source
                || previous.mode != reading.mode
                || previous.display_excluded != reading.display_excluded
                || previous.display_signature != reading.display_signature
        });
        if (stale && !self.stale_generation_marked) || provenance_changed {
            self.generation = self.generation.wrapping_add(1);
        }
        self.stale_generation_marked = false;

        if self.active_scopes == 0 {
            if provenance_changed || self.baseline_reset_pending {
                self.clear_baseline(reading.mode);
                self.baseline_reset_pending = false;
            }
            self.record_baseline(reading.mode, reading.adjusted_watts);
        } else if provenance_changed {
            self.baseline_reset_pending = true;
        }
        let baseline = self.baseline(reading.mode);
        reading.baseline_available = baseline.is_some();
        reading.workload_watts = baseline
            .map(|idle| (reading.adjusted_watts - idle).max(0.0))
            .unwrap_or(reading.adjusted_watts);
        self.reading = Some(reading);
        self.integrated_at = Some(now);
        self.last_sensor_at = Some(now);
    }

    fn snapshot(&mut self, now: Instant) -> Option<PowerSnapshot> {
        let reading = self.reading?;
        let last_sensor_at = self.last_sensor_at?;
        let stale = now.saturating_duration_since(last_sensor_at) > MAX_SAMPLE_AGE;
        self.advance(now);
        if stale && !self.stale_generation_marked {
            self.generation = self.generation.wrapping_add(1);
            self.stale_generation_marked = true;
        }
        Some(PowerSnapshot {
            workload_watts: reading.workload_watts,
            workload_joules: self.workload_joules,
            adjusted_joules: self.adjusted_joules,
            gross_joules: self.gross_joules,
            measured_seconds: self.measured_seconds,
            generation: self.generation,
            source: reading.source,
            display_excluded: reading.display_excluded,
            baseline_available: reading.baseline_available,
        })
    }

    /// Integrate the last observed value (zero-order hold). Advancing for a
    /// snapshot and advancing at the next sensor sample are mathematically
    /// identical, so measurement cannot change merely because UI code polls.
    fn advance(&mut self, now: Instant) {
        let (Some(reading), Some(integrated_at), Some(last_sensor_at)) =
            (self.reading, self.integrated_at, self.last_sensor_at)
        else {
            return;
        };
        let valid_until = last_sensor_at.checked_add(MAX_SAMPLE_AGE).unwrap_or(now);
        let end = now.min(valid_until);
        if end <= integrated_at {
            return;
        }
        let seconds = end.saturating_duration_since(integrated_at).as_secs_f64();
        self.workload_joules += reading.workload_watts * seconds;
        self.adjusted_joules += reading.adjusted_watts * seconds;
        self.gross_joules += reading.gross_watts * seconds;
        self.measured_seconds += seconds;
        self.integrated_at = Some(end);
    }

    fn record_baseline(&mut self, mode: PowerMode, watts: f64) {
        let samples = match mode {
            PowerMode::Mains => &mut self.mains_baseline,
            PowerMode::Battery => &mut self.battery_baseline,
            PowerMode::Unknown => return,
        };
        samples.push_back(watts);
        if samples.len() > BASELINE_WINDOW {
            samples.pop_front();
        }
    }

    fn clear_baseline(&mut self, mode: PowerMode) {
        match mode {
            PowerMode::Mains => self.mains_baseline.clear(),
            PowerMode::Battery => self.battery_baseline.clear(),
            PowerMode::Unknown => {}
        }
    }

    fn baseline(&self, mode: PowerMode) -> Option<f64> {
        let samples = match mode {
            PowerMode::Mains => &self.mains_baseline,
            PowerMode::Battery => &self.battery_baseline,
            PowerMode::Unknown => return None,
        };
        if samples.len() < MIN_BASELINE_SAMPLES {
            return None;
        }
        let mut sorted = samples.iter().copied().collect::<Vec<_>>();
        sorted.sort_by(f64::total_cmp);
        let index = ((sorted.len() - 1) as f64 * 0.2).round() as usize;
        sorted.get(index).copied()
    }

    fn current_baseline_available(&self) -> bool {
        self.reading
            .is_some_and(|reading| self.baseline(reading.mode).is_some())
    }
}

struct PowerReader {
    smc: Option<SmcConnection>,
    smc_failures: usize,
    display: Option<IoReportDisplayReader>,
}

impl PowerReader {
    fn new() -> Self {
        Self {
            smc: SmcConnection::open(),
            smc_failures: 0,
            display: IoReportDisplayReader::new(),
        }
    }

    fn read_power(&mut self) -> Option<PowerReading> {
        if self.smc.is_none() {
            self.smc = SmcConnection::open();
        }
        let (smc_system_watts, smc_display_watts) = if let Some(smc) = self.smc.as_mut() {
            let system = smc.read_f32("PSTR").map(f64::from).and_then(valid_watts);
            let display = smc
                .read_f32("PBwo")
                .or_else(|| smc.read_f32("PDBR"))
                .map(f64::from)
                .and_then(valid_watts);
            if system.is_none() && display.is_none() {
                self.smc_failures += 1;
                if self.smc_failures >= 4 {
                    self.smc = None;
                    self.smc_failures = 0;
                }
            } else {
                self.smc_failures = 0;
            }
            (system, display)
        } else {
            (None, None)
        };

        let registry = read_registry_power().unwrap_or_default();
        let (gross_watts, source) = if let Some(watts) = smc_system_watts {
            (watts, EnergySource::SmcSystemTotal)
        } else if let Some(watts) = registry.telemetry_watts() {
            (watts, EnergySource::PowerTelemetry)
        } else {
            (
                registry.battery_discharge_watts()?,
                EnergySource::BatteryVoltageCurrent,
            )
        };
        let mode = match registry.external_connected {
            Some(true) => PowerMode::Mains,
            Some(false) => PowerMode::Battery,
            None => PowerMode::Unknown,
        };
        let io_display_watts = self
            .display
            .as_mut()
            .and_then(IoReportDisplayReader::read_watts);
        let display_watts = combine_display_power(smc_display_watts, io_display_watts);
        let adjusted_watts = subtract_display_power(gross_watts, display_watts)?;
        let display_excluded = display_watts.is_some() && adjusted_watts < gross_watts;
        let display_signature = if display_excluded {
            u8::from(smc_display_watts.and_then(valid_watts).is_some())
                | (u8::from(io_display_watts.and_then(valid_watts).is_some()) << 1)
        } else {
            0
        };
        Some(PowerReading {
            gross_watts,
            adjusted_watts,
            workload_watts: adjusted_watts,
            source,
            mode,
            display_excluded,
            display_signature,
            baseline_available: false,
        })
    }
}

fn valid_watts(watts: f64) -> Option<f64> {
    (watts.is_finite() && watts > 0.0 && watts < MAX_SANE_SYSTEM_WATTS).then_some(watts)
}

fn subtract_display_power(system_watts: f64, display_watts: Option<f64>) -> Option<f64> {
    let system_watts = valid_watts(system_watts)?;
    let Some(display_watts) = display_watts.and_then(valid_watts) else {
        return Some(system_watts);
    };
    valid_watts(system_watts - display_watts).or(Some(system_watts))
}

fn combine_display_power(smc: Option<f64>, ioreport: Option<f64>) -> Option<f64> {
    match (smc.and_then(valid_watts), ioreport.and_then(valid_watts)) {
        // PBwo/PDBR is commonly the panel/backlight while DISP/DISPEXT covers
        // display-controller domains, so both independently measured costs are
        // excluded when present.
        (Some(backlight), Some(controller)) => Some(backlight + controller),
        (Some(backlight), None) => Some(backlight),
        (None, Some(controller)) => Some(controller),
        (None, None) => None,
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct RegistryPower {
    system_load_mw: Option<i64>,
    external_connected: Option<bool>,
    voltage_mv: Option<i64>,
    amperage_ma: Option<i64>,
}

impl RegistryPower {
    #[cfg(test)]
    fn system_watts(self) -> Option<f64> {
        self.telemetry_watts()
            .or_else(|| self.battery_discharge_watts())
    }

    fn telemetry_watts(self) -> Option<f64> {
        self.system_load_mw
            .map(|milliwatts| milliwatts as f64 / 1_000.0)
            .and_then(valid_watts)
    }

    fn battery_discharge_watts(self) -> Option<f64> {
        if self.external_connected != Some(false) {
            return None;
        }
        let voltage_mv = self.voltage_mv?;
        let amperage_ma = self.amperage_ma?;
        if voltage_mv <= 0 || amperage_ma >= 0 {
            return None;
        }
        valid_watts(voltage_mv as f64 * -(amperage_ma as f64) / 1_000_000.0)
    }
}

fn read_registry_power() -> Option<RegistryPower> {
    let props = smart_battery_properties()?;
    let reading = RegistryPower {
        // SystemLoad is an instantaneous device-load reading in milliwatts.
        // SystemEnergyConsumed is energy accumulated over a telemetry interval;
        // interpreting that value as milliwatts under-reports system power.
        system_load_mw: dict_dictionary(props, "PowerTelemetryData")
            .and_then(|telemetry| dict_i64(telemetry, "SystemLoad")),
        external_connected: dict_bool(props, "ExternalConnected"),
        voltage_mv: dict_i64(props, "AppleRawBatteryVoltage")
            .or_else(|| dict_i64(props, "Voltage")),
        amperage_ma: dict_i64(props, "InstantAmperage").or_else(|| dict_i64(props, "Amperage")),
    };
    unsafe { CFRelease(props as _) };
    Some(reading)
}

/// Minimal IOReport reader for the display-controller energy channels. This
/// intentionally subscribes only to the Energy Model group and uses deltas;
/// absolute counter values are not power readings.
struct IoReportDisplayReader {
    subscription: *mut c_void,
    channels: CfMutableDictionaryRef,
    previous_sample: Option<CfDictionaryRef>,
    previous_at: Instant,
}

// SAFETY: the reader and its retained CoreFoundation objects are transferred
// once to the sampler thread, then read and released only on that thread.
unsafe impl Send for IoReportDisplayReader {}

impl IoReportDisplayReader {
    fn new() -> Option<Self> {
        unsafe {
            let group = cf_string("Energy Model")?;
            let copied = IOReportCopyChannelsInGroup(group, ptr::null(), 0, 0, 0);
            CFRelease(group as _);
            if copied.is_null() {
                return None;
            }
            let channels = CFDictionaryCreateMutableCopy(ptr::null(), 0, copied);
            CFRelease(copied as _);
            if channels.is_null() {
                return None;
            }
            let mut subscribed_channels: CfMutableDictionaryRef = ptr::null_mut();
            let subscription = IOReportCreateSubscription(
                ptr::null_mut(),
                channels,
                &mut subscribed_channels,
                0,
                ptr::null(),
            );
            CFRelease(channels as _);
            if subscription.is_null() || subscribed_channels.is_null() {
                if !subscription.is_null() {
                    CFRelease(subscription as _);
                }
                if !subscribed_channels.is_null() {
                    CFRelease(subscribed_channels as _);
                }
                return None;
            }
            Some(Self {
                subscription,
                channels: subscribed_channels,
                previous_sample: None,
                previous_at: Instant::now(),
            })
        }
    }

    fn read_watts(&mut self) -> Option<f64> {
        unsafe {
            let current = IOReportCreateSamples(self.subscription, self.channels, ptr::null_mut());
            if current.is_null() {
                return None;
            }
            let now = Instant::now();
            let Some(previous) = self.previous_sample.replace(current) else {
                self.previous_at = now;
                return None;
            };
            let seconds = now
                .saturating_duration_since(std::mem::replace(&mut self.previous_at, now))
                .as_secs_f64();
            let delta = IOReportCreateSamplesDelta(previous, current, ptr::null_mut());
            CFRelease(previous as _);
            if delta.is_null() || seconds <= 0.0 {
                if !delta.is_null() {
                    CFRelease(delta as _);
                }
                return None;
            }
            let watts = display_watts_from_delta(delta, seconds);
            CFRelease(delta as _);
            watts.and_then(valid_watts)
        }
    }
}

impl Drop for IoReportDisplayReader {
    fn drop(&mut self) {
        unsafe {
            if let Some(previous) = self.previous_sample.take() {
                CFRelease(previous as _);
            }
            if !self.subscription.is_null() {
                CFRelease(self.subscription as _);
            }
            if !self.channels.is_null() {
                CFRelease(self.channels as _);
            }
        }
    }
}

unsafe fn display_watts_from_delta(delta: CfDictionaryRef, seconds: f64) -> Option<f64> {
    let channels = dict_value(delta, "IOReportChannels")? as CfArrayRef;
    let count = unsafe { CFArrayGetCount(channels) };
    let mut total = 0.0;
    let mut found = false;
    for index in 0..count {
        let channel = unsafe { CFArrayGetValueAtIndex(channels, index) } as CfDictionaryRef;
        if channel.is_null() {
            continue;
        }
        let group = unsafe { IOReportChannelGetGroup(channel) };
        let name = unsafe { IOReportChannelGetChannelName(channel) };
        let unit = unsafe { IOReportChannelGetUnitLabel(channel) };
        if cf_string_value(group).as_deref() != Some("Energy Model") {
            continue;
        }
        let Some(name) = cf_string_value(name) else {
            continue;
        };
        if !(name.starts_with("DISP") || name.starts_with("DISPEXT")) {
            continue;
        }
        let value = unsafe { IOReportSimpleGetIntegerValue(channel, 0) };
        if value <= 0 {
            continue;
        }
        let joules = match cf_string_value(unit).as_deref() {
            Some("mJ") => value as f64 / 1_000.0,
            Some("uJ") | Some("µJ") => value as f64 / 1_000_000.0,
            Some("nJ") => value as f64 / 1_000_000_000.0,
            _ => continue,
        };
        total += joules / seconds;
        found = true;
    }
    found.then_some(total)
}

// AppleSMC user-client protocol. The C layout and selectors are stable across
// Intel and Apple Silicon; only the keys exposed by a particular Mac vary.
const KERNEL_INDEX_SMC: u32 = 2;
const SMC_CMD_READ_KEYINFO: u8 = 9;
const SMC_CMD_READ_BYTES: u8 = 5;
const TYPE_FLT: u32 = u32::from_be_bytes(*b"flt ");
const TYPE_IOF: u32 = u32::from_be_bytes(*b"iof ");
const TYPE_SP78: u32 = u32::from_be_bytes(*b"sp78");

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SmcVersion {
    major: u8,
    minor: u8,
    build: u8,
    reserved: u8,
    release: u16,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SmcPowerLimitData {
    version: u16,
    length: u16,
    cpu_power_limit: u32,
    gpu_power_limit: u32,
    memory_power_limit: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SmcKeyInfo {
    data_size: u32,
    data_type: u32,
    data_attributes: u8,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SmcKeyData {
    key: u32,
    version: SmcVersion,
    power_limit_data: SmcPowerLimitData,
    key_info: SmcKeyInfo,
    result: u8,
    status: u8,
    data8: u8,
    data32: u32,
    bytes: [u8; 32],
}

struct SmcConnection {
    connection: u32,
    key_info: HashMap<u32, SmcKeyInfo>,
}

// SAFETY: the connection is created and used only by the sampler thread after
// startup, and is closed on that same thread when the reader is dropped.
unsafe impl Send for SmcConnection {}

impl SmcConnection {
    fn open() -> Option<Self> {
        unsafe {
            let matching = IOServiceMatching(c"AppleSMC".as_ptr());
            if matching.is_null() {
                return None;
            }
            let service = IOServiceGetMatchingService(0, matching);
            if service == 0 {
                return None;
            }
            let mut connection = 0;
            let result = IOServiceOpen(service, mach_task_self(), 0, &mut connection);
            let _ = IOObjectRelease(service);
            if result != 0 {
                return None;
            }
            Some(Self {
                connection,
                key_info: HashMap::new(),
            })
        }
    }

    fn read_f32(&mut self, key: &str) -> Option<f32> {
        let key = fourcc(key)?;
        let info = self.read_key_info(key)?;
        if info.data_size == 0 || info.data_type == 0 {
            return None;
        }

        let input = SmcKeyData {
            key,
            key_info: info,
            data8: SMC_CMD_READ_BYTES,
            ..Default::default()
        };
        let output = self.call(&input)?;
        let value = match info.data_type {
            TYPE_FLT | TYPE_IOF if info.data_size >= 4 => f32::from_le_bytes([
                output.bytes[0],
                output.bytes[1],
                output.bytes[2],
                output.bytes[3],
            ]),
            TYPE_SP78 if info.data_size >= 2 => {
                i16::from_be_bytes([output.bytes[0], output.bytes[1]]) as f32 / 256.0
            }
            _ => return None,
        };
        value.is_finite().then_some(value)
    }

    fn read_key_info(&mut self, key: u32) -> Option<SmcKeyInfo> {
        if let Some(info) = self.key_info.get(&key) {
            return Some(*info);
        }
        let input = SmcKeyData {
            key,
            data8: SMC_CMD_READ_KEYINFO,
            ..Default::default()
        };
        let info = self.call(&input)?.key_info;
        self.key_info.insert(key, info);
        Some(info)
    }

    fn call(&self, input: &SmcKeyData) -> Option<SmcKeyData> {
        unsafe {
            let mut output = SmcKeyData::default();
            let mut output_size = std::mem::size_of::<SmcKeyData>();
            let result = IOConnectCallStructMethod(
                self.connection,
                KERNEL_INDEX_SMC,
                input as *const SmcKeyData as *const u8,
                std::mem::size_of::<SmcKeyData>(),
                &mut output as *mut SmcKeyData as *mut u8,
                &mut output_size,
            );
            (result == 0 && output.result == 0).then_some(output)
        }
    }
}

impl Drop for SmcConnection {
    fn drop(&mut self) {
        unsafe {
            let _ = IOServiceClose(self.connection);
        }
    }
}

fn fourcc(value: &str) -> Option<u32> {
    let bytes: [u8; 4] = value.as_bytes().try_into().ok()?;
    Some(u32::from_be_bytes(bytes))
}

type CfAllocatorRef = *const c_void;
type CfArrayRef = *const c_void;
type CfDictionaryRef = *const c_void;
type CfMutableDictionaryRef = *mut c_void;
type CfStringRef = *const c_void;
type CfTypeRef = *const c_void;
type CfTypeId = usize;
type CfIndex = isize;
type IoObject = u32;

const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
const K_CF_NUMBER_SINT64_TYPE: i32 = 4;

fn cf_string(value: &str) -> Option<CfStringRef> {
    let value = CString::new(value).ok()?;
    unsafe {
        let output =
            CFStringCreateWithCString(ptr::null(), value.as_ptr(), K_CF_STRING_ENCODING_UTF8);
        (!output.is_null()).then_some(output)
    }
}

fn cf_string_value(value: CfStringRef) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let mut buffer = vec![0_i8; 512];
    let copied = unsafe {
        CFStringGetCString(
            value,
            buffer.as_mut_ptr(),
            buffer.len() as CfIndex,
            K_CF_STRING_ENCODING_UTF8,
        )
    };
    if copied == 0 {
        return None;
    }
    unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) }
        .to_str()
        .ok()
        .map(ToOwned::to_owned)
}

fn smart_battery_properties() -> Option<CfMutableDictionaryRef> {
    unsafe {
        let matching = IOServiceMatching(c"AppleSmartBattery".as_ptr());
        if matching.is_null() {
            return None;
        }
        let service = IOServiceGetMatchingService(0, matching);
        if service == 0 {
            return None;
        }
        let mut properties = ptr::null_mut();
        let result = IORegistryEntryCreateCFProperties(service, &mut properties, ptr::null(), 0);
        let _ = IOObjectRelease(service);
        (result == 0 && !properties.is_null()).then_some(properties)
    }
}

fn dict_value(dict: CfDictionaryRef, key: &str) -> Option<CfTypeRef> {
    let key = CString::new(key).ok()?;
    unsafe {
        let key_ref =
            CFStringCreateWithCString(ptr::null(), key.as_ptr(), K_CF_STRING_ENCODING_UTF8);
        if key_ref.is_null() {
            return None;
        }
        let value = CFDictionaryGetValue(dict, key_ref as _);
        CFRelease(key_ref as _);
        (!value.is_null()).then_some(value)
    }
}

fn dict_dictionary(dict: CfDictionaryRef, key: &str) -> Option<CfDictionaryRef> {
    let value = dict_value(dict, key)?;
    unsafe { (CFGetTypeID(value) == CFDictionaryGetTypeID()).then_some(value as _) }
}

fn dict_i64(dict: CfDictionaryRef, key: &str) -> Option<i64> {
    let value = dict_value(dict, key)?;
    unsafe {
        if CFGetTypeID(value) != CFNumberGetTypeID() {
            return None;
        }
        let mut output = 0_i64;
        (CFNumberGetValue(
            value,
            K_CF_NUMBER_SINT64_TYPE,
            &mut output as *mut _ as *mut c_void,
        ) != 0)
            .then_some(output)
    }
}

fn dict_bool(dict: CfDictionaryRef, key: &str) -> Option<bool> {
    let value = dict_value(dict, key)?;
    unsafe {
        if CFGetTypeID(value) == CFBooleanGetTypeID() {
            Some(CFBooleanGetValue(value) != 0)
        } else if CFGetTypeID(value) == CFNumberGetTypeID() {
            let mut output = 0_i64;
            (CFNumberGetValue(
                value,
                K_CF_NUMBER_SINT64_TYPE,
                &mut output as *mut _ as *mut c_void,
            ) != 0)
                .then_some(output != 0)
        } else {
            None
        }
    }
}

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const c_char) -> *mut c_void;
    fn IOServiceGetMatchingService(main_port: u32, matching: *const c_void) -> IoObject;
    fn IORegistryEntryCreateCFProperties(
        entry: IoObject,
        properties: *mut CfMutableDictionaryRef,
        allocator: CfAllocatorRef,
        options: u32,
    ) -> i32;
    fn IOServiceOpen(
        service: IoObject,
        owning_task: u32,
        connection_type: u32,
        connection: *mut u32,
    ) -> i32;
    fn IOServiceClose(connection: u32) -> i32;
    fn IOConnectCallStructMethod(
        connection: u32,
        selector: u32,
        input: *const u8,
        input_size: usize,
        output: *mut u8,
        output_size: *mut usize,
    ) -> i32;
    fn IOObjectRelease(object: IoObject) -> i32;
    fn mach_task_self() -> u32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFGetTypeID(value: CfTypeRef) -> CfTypeId;
    fn CFDictionaryGetTypeID() -> CfTypeId;
    fn CFNumberGetTypeID() -> CfTypeId;
    fn CFBooleanGetTypeID() -> CfTypeId;
    fn CFDictionaryGetValue(dict: CfDictionaryRef, key: *const c_void) -> CfTypeRef;
    fn CFDictionaryCreateMutableCopy(
        allocator: CfAllocatorRef,
        capacity: CfIndex,
        dictionary: CfDictionaryRef,
    ) -> CfMutableDictionaryRef;
    fn CFArrayGetCount(array: CfArrayRef) -> CfIndex;
    fn CFArrayGetValueAtIndex(array: CfArrayRef, index: CfIndex) -> CfTypeRef;
    fn CFNumberGetValue(number: CfTypeRef, number_type: i32, value: *mut c_void) -> u8;
    fn CFBooleanGetValue(boolean: CfTypeRef) -> u8;
    fn CFRelease(value: CfTypeRef);
    fn CFStringCreateWithCString(
        allocator: CfAllocatorRef,
        string: *const c_char,
        encoding: u32,
    ) -> CfStringRef;
    fn CFStringGetCString(
        string: CfStringRef,
        buffer: *mut c_char,
        buffer_size: CfIndex,
        encoding: u32,
    ) -> u8;
}

#[link(name = "IOReport", kind = "dylib")]
unsafe extern "C" {
    fn IOReportCopyChannelsInGroup(
        group: CfStringRef,
        subgroup: CfStringRef,
        channel_id: u64,
        flags: u64,
        options: u64,
    ) -> CfDictionaryRef;
    fn IOReportCreateSubscription(
        allocator: *mut c_void,
        channels: CfMutableDictionaryRef,
        subscribed_channels: *mut CfMutableDictionaryRef,
        channel_id: u64,
        options: CfTypeRef,
    ) -> *mut c_void;
    fn IOReportCreateSamples(
        subscription: *mut c_void,
        channels: CfMutableDictionaryRef,
        options: *mut c_void,
    ) -> CfDictionaryRef;
    fn IOReportCreateSamplesDelta(
        previous: CfDictionaryRef,
        current: CfDictionaryRef,
        options: *mut c_void,
    ) -> CfDictionaryRef;
    fn IOReportChannelGetGroup(channel: CfDictionaryRef) -> CfStringRef;
    fn IOReportChannelGetChannelName(channel: CfDictionaryRef) -> CfStringRef;
    fn IOReportChannelGetUnitLabel(channel: CfDictionaryRef) -> CfStringRef;
    fn IOReportSimpleGetIntegerValue(channel: CfDictionaryRef, index: i32) -> i64;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(watts: f64) -> PowerReading {
        reading_for_mode(watts, PowerMode::Unknown)
    }

    fn reading_for_mode(watts: f64, mode: PowerMode) -> PowerReading {
        PowerReading {
            gross_watts: watts,
            adjusted_watts: watts,
            workload_watts: watts,
            source: EnergySource::SmcSystemTotal,
            mode,
            display_excluded: false,
            display_signature: 0,
            baseline_available: false,
        }
    }

    #[test]
    fn uses_system_telemetry_on_mains_without_counting_adapter_capacity() {
        let reading = RegistryPower {
            system_load_mw: Some(23_828),
            external_connected: Some(true),
            voltage_mv: Some(12_156),
            amperage_ma: Some(5_497),
        };

        assert_eq!(reading.system_watts(), Some(23.828));
    }

    #[test]
    fn battery_fallback_requires_a_real_discharge() {
        let discharging = RegistryPower {
            external_connected: Some(false),
            voltage_mv: Some(12_000),
            amperage_ma: Some(-2_000),
            ..Default::default()
        };
        let charging = RegistryPower {
            external_connected: Some(true),
            voltage_mv: Some(12_000),
            amperage_ma: Some(2_000),
            ..Default::default()
        };

        assert_eq!(discharging.system_watts(), Some(24.0));
        assert_eq!(charging.system_watts(), None);
    }

    #[test]
    fn subtracts_measured_display_power_but_rejects_inconsistent_values() {
        assert_eq!(subtract_display_power(20.0, Some(4.0)), Some(16.0));
        assert_eq!(subtract_display_power(2.0, Some(4.0)), Some(2.0));
    }

    #[test]
    fn integrates_intermediate_power_samples() {
        let start = Instant::now();
        let mut power = IntegratedPower::default();
        power.observe(reading(10.0), start);
        power.observe(reading(30.0), start + Duration::from_secs(1));
        let snapshot = power
            .snapshot(start + Duration::from_secs(2))
            .expect("recent power should produce a snapshot");

        assert_eq!(snapshot.workload_watts, 30.0);
        assert_eq!(snapshot.workload_joules, 40.0);
    }

    #[test]
    fn snapshots_do_not_change_zero_order_hold_integration() {
        let start = Instant::now();
        let mut frequently_polled = IntegratedPower::default();
        let mut polled_once = IntegratedPower::default();
        frequently_polled.observe(reading(10.0), start);
        polled_once.observe(reading(10.0), start);

        frequently_polled.snapshot(start + Duration::from_millis(400));
        frequently_polled.snapshot(start + Duration::from_millis(700));
        let frequent = frequently_polled
            .snapshot(start + Duration::from_secs(1))
            .unwrap();
        let once = polled_once
            .snapshot(start + Duration::from_secs(1))
            .unwrap();

        assert_eq!(frequent.workload_joules, once.workload_joules);
        assert_eq!(frequent.measured_seconds, once.measured_seconds);
    }

    #[test]
    fn baseline_is_separate_by_power_mode_and_frozen_during_work() {
        let start = Instant::now();
        let mut power = IntegratedPower::default();
        for index in 0..MIN_BASELINE_SAMPLES {
            power.observe(
                reading_for_mode(10.0, PowerMode::Mains),
                start + SAMPLE_INTERVAL * index as u32,
            );
        }
        assert_eq!(power.baseline(PowerMode::Mains), Some(10.0));
        assert_eq!(power.baseline(PowerMode::Battery), None);

        power.active_scopes = 1;
        let work_started = start + SAMPLE_INTERVAL * MIN_BASELINE_SAMPLES as u32;
        power.observe(reading_for_mode(16.0, PowerMode::Mains), work_started);
        let snapshot = power
            .snapshot(work_started + Duration::from_secs(1))
            .unwrap();
        assert_eq!(snapshot.workload_watts, 6.0);
        assert!(snapshot.baseline_available);
        assert_eq!(power.mains_baseline.len(), MIN_BASELINE_SAMPLES);
    }

    #[test]
    fn combines_independent_backlight_and_display_controller_costs() {
        assert_eq!(combine_display_power(Some(3.0), Some(2.0)), Some(5.0));
        assert_eq!(combine_display_power(None, Some(2.0)), Some(2.0));
    }

    #[test]
    fn system_meter_has_one_cross_process_owner() {
        let directory = tempfile::tempdir().unwrap();
        let lock_path = directory.path().join("system-energy.lock");
        let first = acquire_measurement_lock_at(&lock_path);
        assert!(first.is_some());
        assert!(acquire_measurement_lock_at(&lock_path).is_none());
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("energy::macos::tests::measurement_lock_child_probe")
            .env("PB_TEST_SYSTEM_ENERGY_LOCK_PATH", &lock_path)
            .output()
            .unwrap();
        assert!(
            child.status.success(),
            "child ownership probe failed: stdout={} stderr={}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        drop(first);
        assert!(acquire_measurement_lock_at(&lock_path).is_some());
    }

    #[test]
    fn measurement_lock_child_probe() {
        let Some(path) = std::env::var_os("PB_TEST_SYSTEM_ENERGY_LOCK_PATH") else {
            return;
        };
        assert!(acquire_measurement_lock_at(Path::new(&path)).is_none());
    }

    #[test]
    fn invalidates_integration_across_a_sensor_gap() {
        let start = Instant::now();
        let mut power = IntegratedPower::default();
        power.observe(reading(10.0), start);
        let before = power
            .snapshot(start + Duration::from_secs(1))
            .expect("recent power should produce a snapshot");
        power.observe(
            reading(20.0),
            start + MAX_SAMPLE_AGE + Duration::from_secs(1),
        );
        let after = power
            .snapshot(start + MAX_SAMPLE_AGE + Duration::from_secs(2))
            .expect("recovered power should produce a snapshot");

        assert_ne!(before.generation, after.generation);
        assert_eq!(after.workload_joules - before.workload_joules, 30.0);
    }

    #[test]
    fn snapshot_preserves_partial_energy_when_sensor_is_currently_stale() {
        let start = Instant::now();
        let mut power = IntegratedPower::default();
        power.observe(reading(10.0), start);
        let before = power.snapshot(start).unwrap();
        let after = power
            .snapshot(start + MAX_SAMPLE_AGE + Duration::from_secs(1))
            .expect("stale sensors retain bounded counters for partial coverage");

        assert_ne!(before.generation, after.generation);
        assert_eq!(after.workload_joules - before.workload_joules, 20.0);
        assert_eq!(after.measured_seconds - before.measured_seconds, 2.0);
    }

    #[test]
    fn live_power_sample_is_sane_when_available() {
        // Allow IOReport to produce at least one delta as well as exercising
        // the immediate SMC/registry path.
        let _ = super::sample();
        std::thread::sleep(SAMPLE_INTERVAL * 3);
        if let Some(sample) = super::sample() {
            eprintln!("estimated workload power: {:.3} W", sample.workload_watts);
            assert!(sample.workload_watts >= 0.0 && sample.workload_watts < MAX_SANE_SYSTEM_WATTS);
        }
    }
}
