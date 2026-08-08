//! Keep macOS awake while the daemon is actively processing queued work.

#[cfg(target_os = "macos")]
use macos::MacOsSleepAssertion as PlatformSleepAssertion;

#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
struct PlatformSleepAssertion;

#[cfg(not(target_os = "macos"))]
impl PlatformSleepAssertion {
    fn release(self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SleepPreventionStatus {
    pub enabled: bool,
    pub supported: bool,
    pub active: bool,
    pub error: Option<String>,
}

#[derive(Debug)]
pub struct SleepPrevention {
    enabled: bool,
    working: bool,
    assertion: Option<PlatformSleepAssertion>,
    last_error: Option<String>,
}

impl SleepPrevention {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            working: false,
            assertion: None,
            last_error: None,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.reconcile();
    }

    pub fn set_working(&mut self, working: bool) {
        self.working = working;
        self.reconcile();
    }

    pub fn status(&mut self) -> SleepPreventionStatus {
        self.reconcile();
        SleepPreventionStatus {
            enabled: self.enabled,
            supported: supported(),
            active: self.assertion.is_some(),
            error: self.last_error.clone(),
        }
    }

    fn reconcile(&mut self) {
        let should_be_active = desired_assertion(self.enabled, self.working, supported());
        if should_be_active && self.assertion.is_none() {
            match start_assertion() {
                Ok(assertion) => {
                    self.assertion = Some(assertion);
                    self.last_error = None;
                }
                Err(error) => self.last_error = Some(error),
            }
        } else if !should_be_active && self.assertion.is_some() {
            self.release_assertion();
        } else if !should_be_active {
            self.last_error = None;
        }
    }

    fn release_assertion(&mut self) {
        let Some(assertion) = self.assertion.take() else {
            return;
        };
        match assertion.release() {
            Ok(()) => self.last_error = None,
            Err(error) => {
                self.last_error = Some(error);
            }
        }
    }
}

impl Drop for SleepPrevention {
    fn drop(&mut self) {
        self.release_assertion();
    }
}

const fn desired_assertion(enabled: bool, working: bool, supported: bool) -> bool {
    enabled && working && supported
}

const fn supported() -> bool {
    cfg!(target_os = "macos")
}

#[cfg(target_os = "macos")]
fn start_assertion() -> Result<PlatformSleepAssertion, String> {
    PlatformSleepAssertion::create()
}

#[cfg(not(target_os = "macos"))]
fn start_assertion() -> Result<PlatformSleepAssertion, String> {
    Err("preventing idle system sleep is only supported on macOS".to_string())
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::{CString, c_char, c_void};
    use std::ptr;

    type CfAllocatorRef = *const c_void;
    type CfStringRef = *const c_void;
    type CfTypeRef = *const c_void;
    type IoPmAssertionId = u32;
    type IoPmAssertionLevel = u32;
    type IoReturn = i32;

    const CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    const IO_PM_ASSERTION_LEVEL_ON: IoPmAssertionLevel = 255;
    const IO_RETURN_SUCCESS: IoReturn = 0;

    #[derive(Debug)]
    pub(super) struct MacOsSleepAssertion {
        assertion_id: Option<IoPmAssertionId>,
    }

    impl MacOsSleepAssertion {
        pub(super) fn create() -> Result<Self, String> {
            let assertion_type = OwnedCfString::new("PreventUserIdleSystemSleep")?;
            let assertion_name = OwnedCfString::new("pb processing queued work")?;
            let mut assertion_id = 0;

            let result = unsafe {
                IOPMAssertionCreateWithName(
                    assertion_type.as_ptr(),
                    IO_PM_ASSERTION_LEVEL_ON,
                    assertion_name.as_ptr(),
                    &mut assertion_id,
                )
            };
            if result != IO_RETURN_SUCCESS {
                return Err(format!(
                    "failed to create macOS idle-sleep assertion: IOKit returned {result:#010x}"
                ));
            }

            Ok(Self {
                assertion_id: Some(assertion_id),
            })
        }

        pub(super) fn release(mut self) -> Result<(), String> {
            self.release_inner()
        }

        fn release_inner(&mut self) -> Result<(), String> {
            let Some(assertion_id) = self.assertion_id.take() else {
                return Ok(());
            };
            let result = unsafe { IOPMAssertionRelease(assertion_id) };
            if result == IO_RETURN_SUCCESS {
                Ok(())
            } else {
                Err(format!(
                    "failed to release macOS idle-sleep assertion: IOKit returned {result:#010x}"
                ))
            }
        }
    }

    impl Drop for MacOsSleepAssertion {
        fn drop(&mut self) {
            let _ = self.release_inner();
        }
    }

    #[derive(Debug)]
    struct OwnedCfString(CfStringRef);

    impl OwnedCfString {
        fn new(value: &str) -> Result<Self, String> {
            let value = CString::new(value).map_err(|_| {
                "macOS sleep assertion strings cannot contain NUL bytes".to_string()
            })?;
            let string = unsafe {
                CFStringCreateWithCString(ptr::null(), value.as_ptr(), CF_STRING_ENCODING_UTF8)
            };
            if string.is_null() {
                Err("failed to create a Core Foundation sleep assertion string".to_string())
            } else {
                Ok(Self(string))
            }
        }

        fn as_ptr(&self) -> CfStringRef {
            self.0
        }
    }

    impl Drop for OwnedCfString {
        fn drop(&mut self) {
            unsafe { CFRelease(self.0) };
        }
    }

    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOPMAssertionCreateWithName(
            assertion_type: CfStringRef,
            assertion_level: IoPmAssertionLevel,
            assertion_name: CfStringRef,
            assertion_id: *mut IoPmAssertionId,
        ) -> IoReturn;
        fn IOPMAssertionRelease(assertion_id: IoPmAssertionId) -> IoReturn;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithCString(
            allocator: CfAllocatorRef,
            string: *const c_char,
            encoding: u32,
        ) -> CfStringRef;
        fn CFRelease(value: CfTypeRef);
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    use super::SleepPrevention;
    use super::desired_assertion;

    #[test]
    fn assertion_requires_support_setting_and_active_work() {
        assert!(desired_assertion(true, true, true));
        assert!(!desired_assertion(false, true, true));
        assert!(!desired_assertion(true, false, true));
        assert!(!desired_assertion(true, true, false));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn live_setting_changes_start_and_release_the_assertion() {
        let mut prevention = SleepPrevention::new(false);
        prevention.set_working(true);
        assert!(!prevention.status().active);

        prevention.set_enabled(true);
        assert!(prevention.status().active);

        prevention.set_enabled(false);
        assert!(!prevention.status().active);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn finishing_work_releases_the_assertion() {
        let mut prevention = SleepPrevention::new(true);
        prevention.set_working(true);
        assert!(prevention.status().active);

        prevention.set_working(false);
        assert!(!prevention.status().active);
    }
}
