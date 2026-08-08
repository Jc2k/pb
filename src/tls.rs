/// Installs pb's process-wide Rust-only TLS cryptography provider.
///
/// Rustls permits one successful process-wide installation. A concurrent caller may observe that
/// another caller won the race after the initial check, which is also a valid initialized state.
pub(crate) fn install_default_crypto_provider() {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls_graviola::default_provider().install_default();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rust_only_provider_builds_a_reqwest_client() {
        super::install_default_crypto_provider();
        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
        reqwest::Client::builder()
            .build()
            .expect("reqwest should use the installed Rust-only TLS provider");
    }
}
