#[cfg(not(target_os = "macos"))]
compile_error!("macpow supports macOS targets only.");

pub mod metrics {
    use crate::types::Metrics;

    #[derive(Debug)]
    pub struct Sampler {
        _interval_ms: u64,
    }

    impl Sampler {
        pub fn new(interval_ms: u64) -> Self {
            Self { _interval_ms: interval_ms }
        }

        pub fn snapshot(&self) -> Metrics {
            let mut metrics = Metrics::default();
            if let Some(power) = crate::power::read_battery_or_adapter_power_watts() {
                metrics.sys_power_w = power;
                metrics.adapter_power_w = power;
            }
            metrics
        }
    }
}

pub mod types {
    #[derive(Debug, Clone, Default)]
    pub struct Metrics {
        pub soc: SocPower,
        pub sys_power_w: f32,
        pub adapter_power_w: f32,
    }

    #[derive(Debug, Clone, Default)]
    pub struct SocPower {
        pub total_w: f32,
    }
}

#[cfg(target_os = "macos")]
mod power {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_void};
    use std::ptr;

    type CfAllocatorRef = *const c_void;
    type CfDictionaryRef = *const c_void;
    type CfMutableDictionaryRef = *mut c_void;
    type CfStringRef = *const c_void;
    type CfTypeRef = *const c_void;
    type IoIteratorT = u32;
    type IoObjectT = u32;
    type KernReturnT = i32;

    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    const K_CF_NUMBER_SINT32_TYPE: i32 = 3;

    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOServiceMatching(name: *const c_char) -> CfMutableDictionaryRef;
        fn IOServiceGetMatchingServices(
            master_port: u32,
            matching: CfDictionaryRef,
            existing: *mut IoIteratorT,
        ) -> KernReturnT;
        fn IOIteratorNext(iterator: IoIteratorT) -> IoObjectT;
        fn IOObjectRelease(object: IoObjectT) -> KernReturnT;
        fn IORegistryEntryCreateCFProperties(
            entry: IoObjectT,
            properties: *mut CfMutableDictionaryRef,
            allocator: CfAllocatorRef,
            options: u32,
        ) -> KernReturnT;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFDictionaryGetValue(the_dict: CfDictionaryRef, key: *const c_void) -> *const c_void;
        fn CFNumberGetValue(number: CfTypeRef, the_type: i32, value_ptr: *mut c_void) -> bool;
        fn CFRelease(cf: *const c_void);
        fn CFStringCreateWithCString(
            alloc: CfAllocatorRef,
            c_str: *const c_char,
            encoding: u32,
        ) -> CfStringRef;
    }

    pub fn read_battery_or_adapter_power_watts() -> Option<f32> {
        let props = smart_battery_properties()?;
        let voltage_mv = dict_i32(props, "Voltage").unwrap_or_default();
        let amperage_ma = dict_i32(props, "Amperage").unwrap_or_default();
        let battery_watts = ((voltage_mv as f64 * amperage_ma as f64).abs() / 1_000_000.0) as f32;
        let adapter_watts = dict_i32(props, "AdapterPower")
            .map(|raw| f32::from_bits(raw as u32))
            .filter(|watts| watts.is_finite() && *watts > 0.0)
            .unwrap_or_default();
        unsafe { CFRelease(props as _) };
        [adapter_watts, battery_watts]
            .into_iter()
            .find(|watts| watts.is_finite() && *watts > 0.0)
    }

    fn smart_battery_properties() -> Option<CfMutableDictionaryRef> {
        unsafe {
            let service_name = CString::new("AppleSmartBattery").ok()?;
            let matching = IOServiceMatching(service_name.as_ptr());
            if matching.is_null() {
                return None;
            }
            let mut iterator = 0;
            if IOServiceGetMatchingServices(0, matching as _, &mut iterator) != 0 {
                return None;
            }
            let service = IOIteratorNext(iterator);
            let _ = IOObjectRelease(iterator);
            if service == 0 {
                return None;
            }
            let mut props = ptr::null_mut();
            let result = IORegistryEntryCreateCFProperties(service, &mut props, ptr::null(), 0);
            let _ = IOObjectRelease(service);
            if result != 0 || props.is_null() {
                return None;
            }
            Some(props)
        }
    }

    fn dict_i32(dict: CfDictionaryRef, key: &str) -> Option<i32> {
        unsafe {
            let key = CString::new(key).ok()?;
            let key_ref = CFStringCreateWithCString(ptr::null(), key.as_ptr(), K_CF_STRING_ENCODING_UTF8);
            if key_ref.is_null() {
                return None;
            }
            let value = CFDictionaryGetValue(dict, key_ref as _);
            CFRelease(key_ref as _);
            if value.is_null() {
                return None;
            }
            let mut out = 0i32;
            if CFNumberGetValue(value as _, K_CF_NUMBER_SINT32_TYPE, &mut out as *mut _ as _) {
                Some(out)
            } else {
                None
            }
        }
    }
}
