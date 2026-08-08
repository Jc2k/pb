//! HTTP listener ownership for direct development and macOS launchd services.

use anyhow::{Context, Result};
use std::net::{IpAddr, SocketAddr};

#[cfg(target_os = "macos")]
pub(crate) const LAUNCHD_SOCKET_NAME: &str = "HttpListener";
pub(crate) const BONJOUR_SERVICE_TYPE: &str = "_http._tcp";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ListenerSource {
    Direct,
    #[cfg(target_os = "macos")]
    Launchd,
}

pub(crate) struct HttpListener {
    pub(crate) listener: tokio::net::TcpListener,
    pub(crate) source: ListenerSource,
    pub(crate) wake_advertised: bool,
    #[cfg(target_os = "macos")]
    _bonjour: Option<macos::BonjourRegistration>,
}

pub(crate) fn socket_addr(host: &str, port: u16) -> Result<SocketAddr> {
    let normalized_host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    let ip: IpAddr = normalized_host
        .parse()
        .with_context(|| format!("invalid web listen address {host:?}; expected an IP address"))?;
    Ok(SocketAddr::new(ip, port))
}

#[cfg(any(test, target_os = "macos"))]
pub(crate) fn is_network_visible(addr: SocketAddr) -> bool {
    !addr.ip().is_loopback()
}

pub(crate) async fn acquire(expected: SocketAddr) -> Result<HttpListener> {
    #[cfg(target_os = "macos")]
    if let Some(listener) = macos::take_launchd_listener(expected)? {
        return Ok(HttpListener {
            listener,
            source: ListenerSource::Launchd,
            wake_advertised: is_network_visible(expected),
            _bonjour: None,
        });
    }

    let listener = tokio::net::TcpListener::bind(expected)
        .await
        .with_context(|| format!("failed to bind pb HTTP listener at {expected}"))?;

    #[cfg(target_os = "macos")]
    let bonjour = if is_network_visible(expected) {
        let port = listener
            .local_addr()
            .context("failed to inspect the pb HTTP listener")?
            .port();
        match macos::BonjourRegistration::register(port) {
            Ok(registration) => Some(registration),
            Err(error) => {
                eprintln!(
                    "warning: pb HTTP is available, but Bonjour wake registration failed: {error:#}"
                );
                None
            }
        }
    } else {
        None
    };

    Ok(HttpListener {
        listener,
        source: ListenerSource::Direct,
        wake_advertised: {
            #[cfg(target_os = "macos")]
            {
                bonjour.is_some()
            }
            #[cfg(not(target_os = "macos"))]
            {
                false
            }
        },
        #[cfg(target_os = "macos")]
        _bonjour: bonjour,
    })
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{BONJOUR_SERVICE_TYPE, LAUNCHD_SOCKET_NAME};
    use anyhow::{Context, Result, bail};
    use std::ffi::{c_char, c_int, c_void};
    use std::net::{SocketAddr, TcpListener as StdTcpListener};
    use std::os::fd::FromRawFd;
    use std::{ptr, slice};

    type DnsServiceRef = *mut c_void;
    type DnsServiceFlags = u32;
    type DnsServiceError = i32;
    type DnsServiceRegisterReply = Option<
        unsafe extern "C" fn(
            DnsServiceRef,
            DnsServiceFlags,
            DnsServiceError,
            *const c_char,
            *const c_char,
            *const c_char,
            *mut c_void,
        ),
    >;

    const DNS_SERVICE_ERR_NO_ERROR: DnsServiceError = 0;
    const HTTP_TXT_RECORD: &[u8] = b"\x06path=/";

    pub(super) fn take_launchd_listener(
        expected: SocketAddr,
    ) -> Result<Option<tokio::net::TcpListener>> {
        let mut raw_fds: *mut c_int = ptr::null_mut();
        let mut count = 0_usize;
        let result = unsafe {
            launch_activate_socket(
                c"HttpListener".as_ptr(),
                &mut raw_fds as *mut _,
                &mut count as *mut _,
            )
        };

        if result == libc::ENOENT || result == libc::ESRCH {
            return Ok(None);
        }
        if result != 0 {
            bail!(
                "failed to activate launchd socket {LAUNCHD_SOCKET_NAME:?}: {}",
                std::io::Error::from_raw_os_error(result)
            );
        }
        if raw_fds.is_null() {
            bail!("launchd activated {LAUNCHD_SOCKET_NAME:?} without a file descriptor");
        }
        if count == 0 {
            unsafe { libc::free(raw_fds.cast()) };
            bail!("launchd activated {LAUNCHD_SOCKET_NAME:?} without a file descriptor");
        }

        let descriptors = unsafe { slice::from_raw_parts(raw_fds, count).to_vec() };
        unsafe { libc::free(raw_fds.cast()) };
        if descriptors.len() != 1 {
            for descriptor in descriptors {
                unsafe { libc::close(descriptor) };
            }
            bail!(
                "launchd activated {count} descriptors for {LAUNCHD_SOCKET_NAME:?}; expected exactly one"
            );
        }

        let listener = unsafe { StdTcpListener::from_raw_fd(descriptors[0]) };
        let actual = listener
            .local_addr()
            .context("failed to inspect the launchd HTTP listener")?;
        if !listener_matches(expected, actual) {
            bail!(
                "launchd HTTP listener {actual} does not match configured address {expected}; run `pb self refresh-service`"
            );
        }
        listener
            .set_nonblocking(true)
            .context("failed to make the launchd HTTP listener nonblocking")?;
        tokio::net::TcpListener::from_std(listener)
            .context("failed to adopt the launchd HTTP listener")
            .map(Some)
    }

    fn listener_matches(expected: SocketAddr, actual: SocketAddr) -> bool {
        expected.port() == actual.port()
            && (expected.ip().is_unspecified() || expected.ip() == actual.ip())
    }

    #[derive(Debug)]
    pub(super) struct BonjourRegistration(DnsServiceRef);

    impl BonjourRegistration {
        pub(super) fn register(port: u16) -> Result<Self> {
            let mut service_ref = ptr::null_mut();
            let result = unsafe {
                DNSServiceRegister(
                    &mut service_ref,
                    0,
                    0,
                    ptr::null(),
                    c"_http._tcp".as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    port.to_be(),
                    HTTP_TXT_RECORD.len() as u16,
                    HTTP_TXT_RECORD.as_ptr().cast(),
                    None,
                    ptr::null_mut(),
                )
            };
            if result != DNS_SERVICE_ERR_NO_ERROR {
                bail!("DNSServiceRegister({BONJOUR_SERVICE_TYPE}) returned error {result}");
            }
            if service_ref.is_null() {
                bail!("DNSServiceRegister({BONJOUR_SERVICE_TYPE}) returned no service reference");
            }
            Ok(Self(service_ref))
        }
    }

    impl Drop for BonjourRegistration {
        fn drop(&mut self) {
            unsafe { DNSServiceRefDeallocate(self.0) };
        }
    }

    unsafe extern "C" {
        fn launch_activate_socket(
            name: *const c_char,
            fds: *mut *mut c_int,
            count: *mut usize,
        ) -> c_int;
        fn DNSServiceRegister(
            service_ref: *mut DnsServiceRef,
            flags: DnsServiceFlags,
            interface_index: u32,
            name: *const c_char,
            registration_type: *const c_char,
            domain: *const c_char,
            host: *const c_char,
            port: u16,
            txt_len: u16,
            txt_record: *const c_void,
            callback: DnsServiceRegisterReply,
            context: *mut c_void,
        ) -> DnsServiceError;
        fn DNSServiceRefDeallocate(service_ref: DnsServiceRef);
    }

    #[cfg(test)]
    mod tests {
        use super::{BonjourRegistration, listener_matches};
        use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

        #[test]
        fn activated_listener_must_match_port_and_non_wildcard_address() {
            let wildcard = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 8311);
            let loopback = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8311);
            let other_loopback = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 8311);
            assert!(listener_matches(wildcard, wildcard));
            assert!(listener_matches(loopback, loopback));
            assert!(!listener_matches(loopback, wildcard));
            assert!(!listener_matches(loopback, other_loopback));
            assert!(!listener_matches(
                loopback,
                SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8312)
            ));
        }

        #[test]
        #[ignore = "requires access to macOS mDNSResponder"]
        fn native_bonjour_registration_round_trips() {
            let registration = BonjourRegistration::register(49_153).unwrap();
            drop(registration);
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    use super::{ListenerSource, acquire};
    use super::{is_network_visible, socket_addr};
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

    #[test]
    fn only_non_loopback_addresses_are_network_visible() {
        assert!(!is_network_visible(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            8311
        )));
        assert!(!is_network_visible(SocketAddr::new(
            IpAddr::V6(Ipv6Addr::LOCALHOST),
            8311
        )));
        assert!(is_network_visible(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            8311
        )));
    }

    #[test]
    fn socket_address_accepts_ipv4_and_ipv6_without_string_reassembly() {
        assert_eq!(
            socket_addr("127.0.0.1", 8311).unwrap(),
            "127.0.0.1:8311".parse().unwrap()
        );
        assert_eq!(
            socket_addr("::1", 8311).unwrap(),
            "[::1]:8311".parse().unwrap()
        );
        assert_eq!(
            socket_addr("[::1]", 8311).unwrap(),
            "[::1]:8311".parse().unwrap()
        );
        assert!(socket_addr("localhost", 8311).is_err());
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    #[ignore = "requires permission to bind a TCP socket and access macOS mDNSResponder"]
    async fn direct_development_listener_binds_and_advertises_without_launchd() {
        let expected = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0);
        let acquired = acquire(expected).await.unwrap();
        assert_eq!(acquired.source, ListenerSource::Direct);
        assert!(acquired.listener.local_addr().unwrap().port() > 0);
        assert!(acquired.wake_advertised);
    }
}
