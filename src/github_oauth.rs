use anyhow::{Context, Result, anyhow, bail};
use axum::http::StatusCode;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use url::Url;

const AUTHORIZE_URL: &str = "https://github.com/login/oauth/authorize";
const TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const CALLBACK_PATH: &str = "/auth/github/callback";
const CALLBACK_WAIT: Duration = Duration::from_secs(180);
const OAUTH_STATE_BYTES: usize = 32;
const OAUTH_STATE_LENGTH: usize = (OAUTH_STATE_BYTES * 4 + 2) / 3;

#[derive(Debug, Clone)]
pub struct OAuthRequest {
    pub state: String,
    pub code_verifier: String,
    pub authorize_url: String,
    pub redirect_uri: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthCallback {
    pub state: String,
    pub code: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

pub fn redirect_uri(listen: &str, port: u16) -> String {
    let host = match listen {
        "" | "0.0.0.0" | "::" => "127.0.0.1",
        value => value,
    };
    let host = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_string()
    };
    format!("http://{host}:{port}{CALLBACK_PATH}")
}

pub fn callback_bind_addr(listen: &str, port: u16) -> Result<SocketAddr> {
    let host = match listen {
        "" | "0.0.0.0" | "::" => "127.0.0.1",
        value => value,
    };
    format!("{host}:{port}")
        .parse()
        .with_context(|| format!("invalid GitHub OAuth callback address {host}:{port}"))
}

pub fn begin(client_id: &str, redirect_uri: &str, scopes: &[&str]) -> Result<OAuthRequest> {
    let state = random_urlsafe(OAUTH_STATE_BYTES)?;
    let code_verifier = random_urlsafe(64)?;
    let code_challenge = pkce_challenge(&code_verifier);
    let mut url = Url::parse(AUTHORIZE_URL)?;
    url.query_pairs_mut()
        .append_pair("client_id", client_id)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("scope", &scopes.join(" "))
        .append_pair("state", &state)
        .append_pair("code_challenge", &code_challenge)
        .append_pair("code_challenge_method", "S256");
    Ok(OAuthRequest {
        state,
        code_verifier,
        authorize_url: url.into(),
        redirect_uri: redirect_uri.to_string(),
    })
}

pub fn callback_path_for_state(state: &str) -> Result<PathBuf> {
    callback_state_path(state, "toml")
}

pub fn prepare_callback(state: &str) -> Result<()> {
    let callback_path = callback_path_for_state(state)?;
    remove_file_if_exists(&callback_path).with_context(|| {
        format!(
            "failed to remove stale OAuth callback {}",
            callback_path.display()
        )
    })?;
    let pending_path = pending_callback_path_for_state(state)?;
    if let Some(parent) = pending_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    match write_secret_file(&pending_path, state) {
        Ok(()) => Ok(()),
        Err(err) => {
            let _ = remove_file_if_exists(&pending_path);
            Err(err).with_context(|| {
                format!(
                    "failed to prepare GitHub OAuth callback {}",
                    pending_path.display()
                )
            })
        }
    }
}

pub fn persist_callback_from_query(query: &HashMap<String, String>) -> (StatusCode, String) {
    match callback_from_query(query)
        .and_then(|callback| persist_callback(&callback).map(|_| callback))
    {
        Ok(callback) if callback.error.is_none() => (
            StatusCode::OK,
            html_page("GitHub authorization complete", "You can return to pb."),
        ),
        Ok(callback) => (
            StatusCode::BAD_REQUEST,
            html_page(
                "GitHub authorization failed",
                callback
                    .error_description
                    .as_deref()
                    .or(callback.error.as_deref())
                    .unwrap_or("GitHub returned an OAuth error."),
            ),
        ),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            html_page("GitHub authorization failed", &err.to_string()),
        ),
    }
}

pub fn try_start_callback_listener(addr: SocketAddr) -> Option<TcpListener> {
    let listener = TcpListener::bind(addr).ok()?;
    listener.set_nonblocking(true).ok()?;
    Some(listener)
}

pub fn wait_for_callback(state: &str, listener: Option<TcpListener>) -> Result<OAuthCallback> {
    let _pending_guard = PendingCallbackGuard {
        path: pending_callback_path_for_state(state)?,
    };
    let deadline = Instant::now() + CALLBACK_WAIT;
    while Instant::now() < deadline {
        if let Some(listener) = &listener {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let mut buffer = [0_u8; 8192];
                    let len = stream.read(&mut buffer).unwrap_or(0);
                    let request = String::from_utf8_lossy(&buffer[..len]);
                    let callback = callback_from_http_request(&request)?;
                    let status = if callback.error.is_none() {
                        "200 OK"
                    } else {
                        "400 Bad Request"
                    };
                    let body = if callback.error.is_none() {
                        html_page("GitHub authorization complete", "You can return to pb.")
                    } else {
                        html_page(
                            "GitHub authorization failed",
                            callback
                                .error_description
                                .as_deref()
                                .unwrap_or("GitHub returned an OAuth error."),
                        )
                    };
                    let response = format!(
                        "HTTP/1.1 {status}\r\ncontent-type: text/html; charset=utf-8\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes());
                    return validate_state(state, callback);
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(err) => return Err(err).context("failed to accept GitHub OAuth callback"),
            }
        }
        if let Some(callback) = read_persisted_callback(state)? {
            return validate_state(state, callback);
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    bail!("timed out waiting for GitHub OAuth callback on {CALLBACK_PATH}")
}

pub async fn exchange_code(
    client_id: &str,
    code: &str,
    code_verifier: &str,
    redirect_uri: &str,
) -> Result<String> {
    crate::tls::install_default_crypto_provider();
    let response = reqwest::Client::new()
        .post(TOKEN_URL)
        .header(reqwest::header::ACCEPT, "application/json")
        .form(&[
            ("client_id", client_id),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("code_verifier", code_verifier),
        ])
        .send()
        .await
        .context("failed to exchange GitHub OAuth code")?
        .error_for_status()
        .context("GitHub OAuth token endpoint returned an error")?
        .json::<TokenResponse>()
        .await
        .context("failed to parse GitHub OAuth token response")?;
    if let Some(token) = response
        .access_token
        .filter(|token| !token.trim().is_empty())
    {
        return Ok(token);
    }
    bail!(
        "GitHub OAuth token exchange failed: {}",
        response
            .error_description
            .or(response.error)
            .unwrap_or_else(|| "missing access_token".to_string())
    )
}

pub fn token_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("github-token"))
}

pub fn write_token(token: &str) -> Result<PathBuf> {
    let path = token_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    write_secret_file(&path, token)
        .with_context(|| format!("failed to write GitHub token to {}", path.display()))?;
    Ok(path)
}

fn config_dir() -> Result<PathBuf> {
    let config_dir = dirs::config_dir().context("cannot determine config directory")?;
    Ok(config_dir.join("pb"))
}

fn callback_from_query(query: &HashMap<String, String>) -> Result<OAuthCallback> {
    let state = query
        .get("state")
        .filter(|value| !value.is_empty())
        .cloned()
        .ok_or_else(|| anyhow!("missing OAuth state"))?;
    Ok(OAuthCallback {
        state,
        code: query.get("code").filter(|value| !value.is_empty()).cloned(),
        error: query
            .get("error")
            .filter(|value| !value.is_empty())
            .cloned(),
        error_description: query
            .get("error_description")
            .filter(|value| !value.is_empty())
            .cloned(),
    })
}

fn callback_from_http_request(request: &str) -> Result<OAuthCallback> {
    let request_line = request.lines().next().unwrap_or_default();
    let target = request_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow!("invalid OAuth callback request"))?;
    let url = Url::parse(&format!("http://127.0.0.1{target}"))?;
    let query = url
        .query_pairs()
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<HashMap<_, _>>();
    callback_from_query(&query)
}

fn persist_callback(callback: &OAuthCallback) -> Result<()> {
    let path = callback_path_for_state(&callback.state)?;
    let pending_path = pending_callback_path_for_state(&callback.state)?;
    persist_callback_to_paths(callback, &path, &pending_path)
}

fn persist_callback_to_paths(
    callback: &OAuthCallback,
    path: &std::path::Path,
    pending_path: &std::path::Path,
) -> Result<()> {
    match std::fs::remove_file(pending_path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            bail!("GitHub OAuth callback state is not pending")
        }
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "failed to consume pending OAuth callback {}",
                    pending_path.display()
                )
            });
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let text = toml::to_string(&CallbackFile::from(callback))
        .context("failed to encode OAuth callback")?;
    write_secret_file(&path, &text)
}

fn read_persisted_callback(state: &str) -> Result<Option<OAuthCallback>> {
    let path = callback_path_for_state(state)?;
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read OAuth callback {}", path.display()))?;
    let _ = std::fs::remove_file(&path);
    let file: CallbackFile = toml::from_str(&text).context("failed to parse OAuth callback")?;
    Ok(Some(file.into()))
}

fn validate_state(expected_state: &str, callback: OAuthCallback) -> Result<OAuthCallback> {
    if callback.state != expected_state {
        bail!("GitHub OAuth callback state did not match the login request");
    }
    if let Some(error) = &callback.error {
        bail!(
            "GitHub OAuth authorization failed: {}",
            callback.error_description.as_deref().unwrap_or(error)
        );
    }
    if callback.code.as_ref().is_none_or(|code| code.is_empty()) {
        bail!("GitHub OAuth callback did not include an authorization code");
    }
    Ok(callback)
}

fn validate_state_component(state: &str) -> Result<()> {
    let is_urlsafe = state
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'));
    if state.len() != OAUTH_STATE_LENGTH || !is_urlsafe {
        bail!("invalid OAuth state");
    }
    Ok(())
}

fn callback_state_path(state: &str, extension: &str) -> Result<PathBuf> {
    validate_state_component(state)?;
    Ok(config_dir()?
        .join("github-oauth")
        .join(format!("{state}.{extension}")))
}

fn pending_callback_path_for_state(state: &str) -> Result<PathBuf> {
    callback_state_path(state, "pending")
}

fn remove_file_if_exists(path: &std::path::Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

fn random_urlsafe(bytes: usize) -> Result<String> {
    let mut random = vec![0_u8; bytes];
    getrandom::getrandom(&mut random)
        .map_err(|err| anyhow!("failed to generate OAuth random bytes: {err}"))?;
    Ok(URL_SAFE_NO_PAD.encode(random))
}

fn pkce_challenge(code_verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(code_verifier.as_bytes()))
}

fn html_page(title: &str, message: &str) -> String {
    format!(
        "<!doctype html><meta charset=\"utf-8\"><title>{}</title><body><h1>{}</h1><p>{}</p></body>",
        escape_html(title),
        escape_html(title),
        escape_html(message)
    )
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CallbackFile {
    state: String,
    code: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

impl From<&OAuthCallback> for CallbackFile {
    fn from(callback: &OAuthCallback) -> Self {
        Self {
            state: callback.state.clone(),
            code: callback.code.clone(),
            error: callback.error.clone(),
            error_description: callback.error_description.clone(),
        }
    }
}

impl From<CallbackFile> for OAuthCallback {
    fn from(file: CallbackFile) -> Self {
        Self {
            state: file.state,
            code: file.code,
            error: file.error,
            error_description: file.error_description,
        }
    }
}

struct PendingCallbackGuard {
    path: PathBuf,
}

impl Drop for PendingCallbackGuard {
    fn drop(&mut self) {
        let _ = remove_file_if_exists(&self.path);
    }
}

#[cfg(unix)]
fn write_secret_file(path: &std::path::Path, contents: &str) -> Result<()> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let mut file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(path)?;
    file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    file.set_len(0)?;
    file.write_all(contents.as_bytes())?;
    Ok(())
}

#[cfg(not(unix))]
fn write_secret_file(path: &std::path::Path, contents: &str) -> Result<()> {
    std::fs::write(path, contents)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redirect_uri_uses_loopback_for_wildcard_listeners() {
        assert_eq!(
            redirect_uri("0.0.0.0", 8311),
            "http://127.0.0.1:8311/auth/github/callback"
        );
        assert_eq!(
            redirect_uri("::", 8311),
            "http://127.0.0.1:8311/auth/github/callback"
        );
    }

    #[test]
    fn redirect_uri_brackets_ipv6_literals() {
        assert_eq!(
            redirect_uri("::1", 8311),
            "http://[::1]:8311/auth/github/callback"
        );
    }

    #[test]
    fn callback_query_requires_state() {
        let query = HashMap::from([("code".to_string(), "abc".to_string())]);
        assert!(callback_from_query(&query).is_err());
    }

    #[test]
    fn callback_paths_reject_unsafe_state_components() {
        for state in [
            "",
            "../callback",
            "nested/callback",
            "state.toml",
            "state%2f",
            &"A".repeat(OAUTH_STATE_LENGTH - 1),
            &"A".repeat(OAUTH_STATE_LENGTH + 1),
        ] {
            assert!(
                callback_path_for_state(state).is_err(),
                "accepted {state:?}"
            );
        }
        assert!(callback_path_for_state(&"A".repeat(OAUTH_STATE_LENGTH)).is_ok());
    }

    #[test]
    fn callback_persistence_requires_and_consumes_pending_state() {
        let directory = tempfile::tempdir().unwrap();
        let state = "A".repeat(OAUTH_STATE_LENGTH);
        let callback = OAuthCallback {
            state,
            code: Some("code".to_string()),
            error: None,
            error_description: None,
        };
        let callback_path = directory.path().join("callback.toml");
        let pending_path = directory.path().join("callback.pending");

        let missing =
            persist_callback_to_paths(&callback, &callback_path, &pending_path).unwrap_err();
        assert!(missing.to_string().contains("is not pending"));

        write_secret_file(&pending_path, &callback.state).unwrap();
        persist_callback_to_paths(&callback, &callback_path, &pending_path).unwrap();

        assert!(!pending_path.exists());
        let persisted: CallbackFile =
            toml::from_str(&std::fs::read_to_string(&callback_path).unwrap()).unwrap();
        assert_eq!(persisted.state, callback.state);
        assert_eq!(persisted.code.as_deref(), Some("code"));
        let replay =
            persist_callback_to_paths(&callback, &callback_path, &pending_path).unwrap_err();
        assert!(replay.to_string().contains("is not pending"));
    }

    #[cfg(unix)]
    #[test]
    fn secret_file_permissions_are_restricted_when_overwriting() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("secret");
        std::fs::write(&path, "old secret").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        write_secret_file(&path, "new secret").unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        assert_eq!(std::fs::read_to_string(path).unwrap(), "new secret");
    }
}
