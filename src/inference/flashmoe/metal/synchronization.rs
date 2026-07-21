use std::sync::{Mutex, MutexGuard};

/// Recover data only for infallible cleanup paths. Normal execution should
/// continue surfacing poisoned state as an error instead of using it.
pub(super) fn lock_for_drop<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Mutable variant used by owners whose `Drop` implementation has exclusive
/// access to the mutex and must not skip cleanup after a panic.
pub(super) fn get_mut_for_drop<T>(mutex: &mut Mutex<T>) -> &mut T {
    mutex
        .get_mut()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;

    #[test]
    fn cleanup_accessors_recover_poisoned_data() {
        let mut mutex = Mutex::new(1usize);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut value = mutex.lock().unwrap();
            *value = 2;
            panic!("intentional cleanup poison");
        }));
        assert!(result.is_err());

        assert_eq!(*lock_for_drop(&mutex), 2);
        *get_mut_for_drop(&mut mutex) = 3;
        assert_eq!(*get_mut_for_drop(&mut mutex), 3);
    }
}
