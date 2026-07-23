use anyhow::{Context, Result, bail};
use reqwest::Url;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};

pub(crate) fn validate_public_http_url(url: &Url) -> Result<()> {
    if !matches!(url.scheme(), "http" | "https") {
        bail!("only http and https URLs are supported");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("URLs with embedded credentials are not allowed");
    }
    let host = url.host_str().context("URL is missing a host")?;
    let normalized_host = host.trim_end_matches('.').to_ascii_lowercase();
    if normalized_host == "localhost"
        || normalized_host.ends_with(".localhost")
        || normalized_host == "local"
        || normalized_host.ends_with(".local")
    {
        bail!("local network URLs are not allowed");
    }
    let ip_literal = normalized_host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(&normalized_host);
    if let Ok(ip) = ip_literal.parse::<IpAddr>()
        && is_non_public_ip(ip)
    {
        bail!("private or loopback IP URLs are not allowed");
    }
    Ok(())
}

pub(crate) fn validate_resolved_public_addresses(
    host: &str,
    addresses: &[SocketAddr],
) -> Result<SocketAddr> {
    if addresses.is_empty() {
        bail!("public network host '{host}' resolved to no addresses");
    }
    if let Some(address) = addresses
        .iter()
        .find(|address| is_non_public_ip(address.ip()))
    {
        bail!(
            "public network host '{host}' resolved to private or special-use address {}",
            address.ip()
        );
    }
    Ok(addresses[0])
}

pub(crate) fn resolve_public_target_blocking(url: &Url) -> Result<(String, SocketAddr)> {
    validate_public_http_url(url)?;
    let host = url.host_str().context("URL is missing a host")?.to_string();
    let port = url
        .port_or_known_default()
        .context("URL scheme has no known default port")?;
    let addresses = (host.as_str(), port)
        .to_socket_addrs()
        .with_context(|| format!("failed to resolve public network host '{host}'"))?
        .collect::<Vec<_>>();
    let address = validate_resolved_public_addresses(&host, &addresses)?;
    Ok((host, address))
}

fn is_non_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.octets()[0] == 0
                || ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.is_unspecified()
                || is_shared_v4(ip)
                || ip.is_multicast()
                || is_benchmark_v4(ip)
                || is_protocol_assignment_v4(ip)
                || ip.octets()[0] >= 240
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
                || is_documentation_v6(ip)
                || ip.is_multicast()
                || is_site_local_v6(ip)
                || ip.to_ipv4_mapped().is_some()
                || !is_global_unicast_v6(ip)
        }
    }
}

fn is_benchmark_v4(ip: Ipv4Addr) -> bool {
    matches!(ip.octets(), [198, second, ..] if matches!(second, 18 | 19))
}

fn is_protocol_assignment_v4(ip: Ipv4Addr) -> bool {
    matches!(ip.octets(), [192, 0, 0, _])
}

fn is_shared_v4(ip: Ipv4Addr) -> bool {
    matches!(ip.octets(), [100, second_octet, ..] if (64..=127).contains(&second_octet))
}

fn is_documentation_v6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    matches!(segments, [0x2001, 0x0db8, ..]) || segments[0] & 0xfff0 == 0x3ff0
}

fn is_site_local_v6(ip: Ipv6Addr) -> bool {
    ip.segments()[0] & 0xffc0 == 0xfec0
}

fn is_global_unicast_v6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    if segments[0] & 0xe000 != 0x2000 {
        return false;
    }

    // IANA's special-purpose IPv6 registry reserves 2001::/23 and 2002::/16.
    // Treat the entire allocations as non-public: accepting selected anycast
    // exceptions here would weaken the SSRF boundary for no registry use case.
    !(segments[0] == 0x2001 && segments[1] <= 0x01ff) && segments[0] != 0x2002
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_url_rejects_local_names_and_credentials() {
        for value in [
            "http://localhost:8080",
            "http://LOCALHOST./",
            "http://service.LOCAL/",
            "http://user@example.com/",
        ] {
            assert!(
                validate_public_http_url(&Url::parse(value).unwrap()).is_err(),
                "accepted {value}"
            );
        }
    }

    #[test]
    fn public_url_rejects_special_addresses() {
        for value in [
            "https://0.1.2.3/",
            "https://127.0.0.1/",
            "https://100.64.0.1/",
            "https://198.18.0.1/",
            "https://[::1]/",
            "https://[64:ff9b:1::1]/",
            "https://[100::1]/",
            "https://[2001:1::1]/",
            "https://[2001:db8::1]/",
            "https://[2002::1]/",
            "https://[3fff::1]/",
            "https://[5f00::1]/",
            "https://[::ffff:5db8:d822]/",
        ] {
            assert!(validate_public_http_url(&Url::parse(value).unwrap()).is_err());
        }
    }

    #[test]
    fn resolved_public_host_rejects_any_private_answer() {
        let addresses = [
            SocketAddr::from(([93, 184, 216, 34], 443)),
            SocketAddr::from(([127, 0, 0, 1], 443)),
        ];
        let error = validate_resolved_public_addresses("public.example", &addresses)
            .unwrap_err()
            .to_string();
        assert!(error.contains("private or special-use"), "{error}");
    }

    #[test]
    fn public_url_accepts_public_https() {
        validate_public_http_url(&Url::parse("https://example.com/path").unwrap()).unwrap();
        validate_public_http_url(&Url::parse("https://[2606:4700:4700::1111]/").unwrap()).unwrap();
    }

    #[test]
    fn resolved_public_host_rejects_omitted_special_use_ranges() {
        for ip in [
            "0.1.2.3",
            "64:ff9b:1::1",
            "100::1",
            "2001:1::1",
            "2002::1",
            "3fff::1",
            "5f00::1",
        ] {
            let address = SocketAddr::new(ip.parse().unwrap(), 443);
            assert!(
                validate_resolved_public_addresses("registry.example", &[address]).is_err(),
                "accepted {ip}"
            );
        }
    }
}
