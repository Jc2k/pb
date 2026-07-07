//! Built-in Safari WebDriver tools for local browser diagnostics.
//!
//! This first version intentionally targets Safari via `/usr/bin/safaridriver` on macOS.
//! It uses WebDriver plus page-side JavaScript instrumentation for console and network
//! information because Safari WebDriver does not expose the full Web Inspector protocol.
//! Native debugger breakpoints, source-map debugging, and complete performance tracing are
//! outside this initial WebDriver-based scope.

#[cfg(target_os = "macos")]
use anyhow::{Context, anyhow};
use anyhow::{Result, bail};
#[cfg(target_os = "macos")]
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
#[cfg(target_os = "macos")]
use futures::future::LocalBoxFuture;
use serde_json::{Value, json};
#[cfg(target_os = "macos")]
use std::process::{Child, Command, Stdio};
#[cfg(target_os = "macos")]
use std::sync::{Mutex, OnceLock};
#[cfg(target_os = "macos")]
use std::time::Duration;
#[cfg(target_os = "macos")]
use tokio::net::TcpListener;
#[cfg(target_os = "macos")]
use tokio::time::timeout;
#[cfg(target_os = "macos")]
use tokio::time::{Instant, sleep};

#[cfg(target_os = "macos")]
const SAFARIDRIVER: &str = "/usr/bin/safaridriver";
#[cfg(target_os = "macos")]
const TOOL_TIMEOUT: Duration = Duration::from_secs(30);
#[cfg(target_os = "macos")]
const WAIT_POLL: Duration = Duration::from_millis(200);

#[cfg(target_os = "macos")]
static SESSION: OnceLock<Mutex<Option<BrowserSession>>> = OnceLock::new();

#[cfg(target_os = "macos")]
struct BrowserSession {
    client: WebDriverClient,
    driver: Child,
    port: u16,
}

#[cfg(target_os = "macos")]
impl Drop for BrowserSession {
    fn drop(&mut self) {
        let _ = self.driver.kill();
        let _ = self.driver.wait();
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct WebDriverClient {
    http: reqwest::Client,
    endpoint: String,
    session_id: String,
}

#[cfg(target_os = "macos")]
struct WebDriverElement {
    client: WebDriverClient,
    id: String,
}

#[cfg(target_os = "macos")]
impl WebDriverClient {
    async fn connect(endpoint: &str) -> Result<Self> {
        let http = reqwest::Client::new();
        let response: Value = http
            .post(format!("{endpoint}/session"))
            .json(&json!({"capabilities":{"alwaysMatch":{"browserName":"safari"}}}))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let session_id = response
            .get("value")
            .and_then(|value| value.get("sessionId"))
            .or_else(|| response.get("sessionId"))
            .and_then(Value::as_str)
            .context("safaridriver did not return a WebDriver session id")?
            .to_string();
        Ok(Self {
            http,
            endpoint: endpoint.to_string(),
            session_id,
        })
    }

    async fn command(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value> {
        let url = format!("{}/session/{}{}", self.endpoint, self.session_id, path);
        let request = self.http.request(method, url);
        let request = if let Some(body) = body {
            request.json(&body)
        } else {
            request
        };
        let response = request.send().await?.error_for_status()?;
        let value: Value = response.json().await?;
        if let Some(error) = value
            .get("value")
            .and_then(|value| value.get("error"))
            .and_then(Value::as_str)
        {
            let message = value
                .get("value")
                .and_then(|value| value.get("message"))
                .and_then(Value::as_str)
                .unwrap_or(error);
            bail!("WebDriver command failed: {message}");
        }
        Ok(value.get("value").cloned().unwrap_or(value))
    }

    async fn goto(&self, url: &str) -> Result<()> {
        self.command(reqwest::Method::POST, "/url", Some(json!({"url": url})))
            .await?;
        Ok(())
    }

    async fn current_url(&self) -> Result<url::Url> {
        let value = self.command(reqwest::Method::GET, "/url", None).await?;
        let url = value
            .as_str()
            .context("safaridriver returned a non-string current URL")?;
        Ok(url::Url::parse(url)?)
    }

    async fn execute(&self, script: &str, args: Vec<Value>) -> Result<Value> {
        self.command(
            reqwest::Method::POST,
            "/execute/sync",
            Some(json!({"script": script, "args": args})),
        )
        .await
    }

    async fn find_css(&self, selector: &str) -> Result<WebDriverElement> {
        let value = self
            .command(
                reqwest::Method::POST,
                "/element",
                Some(json!({"using": "css selector", "value": selector})),
            )
            .await?;
        let id = element_id(&value).context("safaridriver returned an element without an id")?;
        Ok(WebDriverElement {
            client: self.clone(),
            id,
        })
    }

    async fn screenshot(&self) -> Result<Vec<u8>> {
        let value = self
            .command(reqwest::Method::GET, "/screenshot", None)
            .await?;
        decode_screenshot(value)
    }

    async fn refresh(&self) -> Result<()> {
        self.command(reqwest::Method::POST, "/refresh", Some(json!({})))
            .await?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.http
            .delete(format!("{}/session/{}", self.endpoint, self.session_id))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl WebDriverElement {
    fn to_json(&self) -> Value {
        json!({"element-6066-11e4-a52e-4f735466cecf": self.id})
    }

    async fn click(&self) -> Result<()> {
        self.client
            .command(
                reqwest::Method::POST,
                &format!("/element/{}/click", self.id),
                Some(json!({})),
            )
            .await?;
        Ok(())
    }

    async fn send_keys(&self, text: &str) -> Result<()> {
        self.client
            .command(
                reqwest::Method::POST,
                &format!("/element/{}/value", self.id),
                Some(json!({"text": text})),
            )
            .await?;
        Ok(())
    }

    async fn screenshot(&self) -> Result<Vec<u8>> {
        let value = self
            .client
            .command(
                reqwest::Method::GET,
                &format!("/element/{}/screenshot", self.id),
                None,
            )
            .await?;
        decode_screenshot(value)
    }
}

#[cfg(target_os = "macos")]
fn element_id(value: &Value) -> Option<String> {
    value
        .get("element-6066-11e4-a52e-4f735466cecf")
        .or_else(|| value.get("ELEMENT"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

#[cfg(target_os = "macos")]
fn decode_screenshot(value: Value) -> Result<Vec<u8>> {
    let data = value
        .as_str()
        .context("safaridriver returned a non-string screenshot")?;
    BASE64
        .decode(data)
        .context("safaridriver returned invalid base64 screenshot data")
}

#[cfg(target_os = "macos")]
pub fn call_tool(tool: &str, arguments: &Value) -> Result<String> {
    block_on(async move {
        match tool {
            "browser_open" => open(arguments).await,
            "browser_snapshot" => with_client(|c| Box::pin(async move { snapshot(c).await })).await,
            "browser_interact" => {
                let arguments = arguments.clone();
                with_client(|c| Box::pin(async move { interact(c, &arguments).await })).await
            }
            "browser_dom" => {
                let arguments = arguments.clone();
                with_client(|c| Box::pin(async move { dom(c, &arguments).await })).await
            }
            "browser_console" => {
                with_client(|c| {
                    Box::pin(async move {
                        js_json(c, "return window.__pbBrowserDebug?.console || [];").await
                    })
                })
                .await
            }
            "browser_network" => {
                with_client(|c| {
                    Box::pin(async move {
                        js_json(c, "return window.__pbBrowserDebug?.network || [];").await
                    })
                })
                .await
            }
            "browser_evaluate" => {
                let arguments = arguments.clone();
                with_client(|c| Box::pin(async move { evaluate(c, &arguments).await })).await
            }
            "browser_storage" => {
                let arguments = arguments.clone();
                with_client(|c| Box::pin(async move { storage(c, &arguments).await })).await
            }
            "browser_wait" => {
                let arguments = arguments.clone();
                with_client(|c| Box::pin(async move { wait_for(c, &arguments).await })).await
            }
            "browser_reload" => {
                let arguments = arguments.clone();
                with_client(|c| Box::pin(async move { reload(c, &arguments).await })).await
            }
            "browser_screenshot" => {
                let arguments = arguments.clone();
                with_client(|c| Box::pin(async move { screenshot(c, &arguments).await })).await
            }
            "browser_debug_report" => {
                with_client(|c| Box::pin(async move { debug_report(c).await })).await
            }
            "react_tree" | "react_component" | "react_find" | "react_renders" | "react_errors" => {
                react_unsupported(tool)
            }
            "browser_close" => close().await,
            _ => bail!("unknown browser tool: {tool}"),
        }
    })
}

#[cfg(not(target_os = "macos"))]
pub fn call_tool(tool: &str, _arguments: &Value) -> Result<String> {
    match tool {
        "react_tree" | "react_component" | "react_find" | "react_renders" | "react_errors" => {
            react_unsupported(tool)
        }
        "browser_open"
        | "browser_snapshot"
        | "browser_interact"
        | "browser_dom"
        | "browser_console"
        | "browser_network"
        | "browser_evaluate"
        | "browser_storage"
        | "browser_wait"
        | "browser_reload"
        | "browser_screenshot"
        | "browser_debug_report"
        | "browser_close" => bail!(
            "Safari automation is only available on macOS. On macOS, enable it once with: safaridriver --enable"
        ),
        _ => bail!("unknown browser tool: {tool}"),
    }
}

#[cfg(target_os = "macos")]
fn block_on<F: std::future::Future<Output = Result<String>>>(future: F) -> Result<String> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to create browser tool runtime")?
        .block_on(async move {
            timeout(TOOL_TIMEOUT, future)
                .await
                .context("browser tool timed out")?
        })
}

#[cfg(target_os = "macos")]
#[allow(clippy::await_holding_lock)]
async fn with_client(
    f: impl for<'a> FnOnce(&'a mut WebDriverClient) -> LocalBoxFuture<'a, Result<String>>,
) -> Result<String> {
    let mut session = SESSION
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| anyhow!("browser session lock poisoned"))?;
    let session = session
        .as_mut()
        .context("no Safari WebDriver session is open; call browser_open(url) first")?;
    if session.driver.try_wait()?.is_some() {
        bail!(
            "safaridriver exited unexpectedly; call browser_close then browser_open to restart it"
        );
    }
    f(&mut session.client).await
}

#[cfg(target_os = "macos")]
async fn open(arguments: &Value) -> Result<String> {
    let url = arguments
        .get("url")
        .and_then(Value::as_str)
        .context("browser_open requires string argument: url")?;
    let mut existing = {
        let mut guard = SESSION
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map_err(|_| anyhow!("browser session lock poisoned"))?;
        guard.take()
    };
    if existing.is_none() {
        existing = Some(start_session().await?);
    }
    let mut session = existing.unwrap();
    inject_instrumentation(&mut session.client).await?;
    session
        .client
        .goto(url)
        .await
        .with_context(|| format!("failed to navigate Safari to {url}"))?;
    inject_instrumentation(&mut session.client).await?;
    let current = session
        .client
        .current_url()
        .await
        .context("failed to read current URL")?;
    let port = session.port;
    *SESSION
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| anyhow!("browser session lock poisoned"))? = Some(session);
    Ok(
        json!({"ok": true, "url": current.as_str(), "driver": SAFARIDRIVER, "port": port})
            .to_string(),
    )
}

#[cfg(target_os = "macos")]
async fn start_session() -> Result<BrowserSession> {
    #[cfg(not(target_os = "macos"))]
    bail!(
        "Safari automation is only available on macOS. On macOS, enable it once with: safaridriver --enable"
    );
    #[cfg(target_os = "macos")]
    {
        if !std::path::Path::new(SAFARIDRIVER).exists() {
            bail!(
                "/usr/bin/safaridriver was not found. Safari automation requires macOS Safari and must be enabled with: safaridriver --enable"
            );
        }
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .context("failed to reserve local safaridriver port")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let mut driver = Command::new(SAFARIDRIVER).arg("-p").arg(port.to_string()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().context("failed to launch /usr/bin/safaridriver; enable Safari automation with: safaridriver --enable")?;
        let endpoint = format!("http://127.0.0.1:{port}");
        let deadline = Instant::now() + Duration::from_secs(5);
        let client = loop {
            if driver.try_wait()?.is_some() {
                bail!(
                    "safaridriver exited during startup. Enable Safari automation with: safaridriver --enable"
                );
            }
            match WebDriverClient::connect(&endpoint).await {
                Ok(client) => break client,
                Err(err) if Instant::now() < deadline => {
                    let _ = err;
                    sleep(WAIT_POLL).await;
                }
                Err(err) => bail!(
                    "failed to create Safari WebDriver session: {err}. If automation is disabled, run: safaridriver --enable"
                ),
            }
        };
        Ok(BrowserSession {
            client,
            driver,
            port,
        })
    }
}

#[cfg(target_os = "macos")]
async fn inject_instrumentation(client: &mut WebDriverClient) -> Result<()> {
    let script = include_str!("browser_instrumentation.js");
    let _ = client.execute(script, vec![]).await;
    Ok(())
}

#[cfg(target_os = "macos")]
async fn js_json(client: &mut WebDriverClient, script: &str) -> Result<String> {
    inject_instrumentation(client).await?;
    Ok(client
        .execute(script, vec![])
        .await
        .context("browser JavaScript execution failed")?
        .to_string())
}

#[cfg(target_os = "macos")]
async fn snapshot(client: &mut WebDriverClient) -> Result<String> {
    js_json(client, r#"return Array.from(document.querySelectorAll('body *')).slice(0,500).map((el,i)=>{ if(!el.dataset.pbRef) el.dataset.pbRef='pb-'+i+'-'+Math.random().toString(36).slice(2); const r=el.getBoundingClientRect(); return {ref:el.dataset.pbRef, tag:el.tagName.toLowerCase(), id:el.id||null, classes:Array.from(el.classList), role:el.getAttribute('role'), name:el.getAttribute('aria-label')||el.innerText?.trim().slice(0,120)||el.getAttribute('title')||'', visible:!!(r.width||r.height), bounds:{x:r.x,y:r.y,width:r.width,height:r.height}};});"#).await
}

#[cfg(target_os = "macos")]
async fn resolve_element(client: &mut WebDriverClient, target: &str) -> Result<WebDriverElement> {
    let selector = if target.starts_with("pb-") {
        format!("[data-pb-ref='{target}']")
    } else {
        target.to_string()
    };
    client
        .find_css(&selector)
        .await
        .with_context(|| format!("failed to find element: {target}"))
}

#[cfg(target_os = "macos")]
async fn interact(client: &mut WebDriverClient, arguments: &Value) -> Result<String> {
    let action = arguments
        .get("action")
        .and_then(Value::as_str)
        .context("browser_interact requires action")?;
    let target = arguments
        .get("target")
        .and_then(Value::as_str)
        .context("browser_interact requires target")?;
    let value = arguments.get("value").and_then(Value::as_str).unwrap_or("");
    let el = resolve_element(client, target).await?;
    match action {
        "click" => el.click().await?,
        "type" => el.send_keys(value).await?,
        "focus" => {
            client
                .execute("arguments[0].focus();", vec![el.to_json()])
                .await?;
        }
        "submit" => {
            client.execute("arguments[0].requestSubmit ? arguments[0].requestSubmit() : arguments[0].submit();", vec![el.to_json()]).await?;
        }
        "select" => {
            client.execute("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('change',{bubbles:true}));", vec![el.to_json(), json!(value)]).await?;
        }
        "hover" => {
            client
                .execute(
                    "arguments[0].dispatchEvent(new MouseEvent('mouseover',{bubbles:true}));",
                    vec![el.to_json()],
                )
                .await?;
        }
        _ => bail!("unsupported browser_interact action: {action}"),
    }
    Ok(json!({"ok": true, "action": action, "target": target}).to_string())
}

#[cfg(target_os = "macos")]
async fn dom(client: &mut WebDriverClient, arguments: &Value) -> Result<String> {
    let target = arguments
        .get("target")
        .and_then(Value::as_str)
        .context("browser_dom requires target")?;
    let el = resolve_element(client, target).await?;
    Ok(client.execute(r#"const el=arguments[0], r=el.getBoundingClientRect(), cs=getComputedStyle(el); return {html:el.outerHTML, text:el.innerText||el.textContent||'', attributes:Object.fromEntries(Array.from(el.attributes).map(a=>[a.name,a.value])), styles:{display:cs.display, visibility:cs.visibility, opacity:cs.opacity, position:cs.position}, bounds:{x:r.x,y:r.y,width:r.width,height:r.height}, visible:!!(r.width||r.height)&&cs.visibility!=='hidden'&&cs.display!=='none'};"#, vec![el.to_json()]).await?.to_string())
}

#[cfg(target_os = "macos")]
async fn evaluate(client: &mut WebDriverClient, arguments: &Value) -> Result<String> {
    let script = arguments
        .get("script")
        .and_then(Value::as_str)
        .context("browser_evaluate requires script")?;
    js_json(client, &format!("return (async()=>{{ {script} }})()")).await
}

#[cfg(target_os = "macos")]
async fn storage(client: &mut WebDriverClient, arguments: &Value) -> Result<String> {
    inject_instrumentation(client).await?;
    if let Some(clear) = arguments.get("clear").and_then(Value::as_bool)
        && clear
    {
        client
            .execute("localStorage.clear(); sessionStorage.clear();", vec![])
            .await?;
    }
    Ok(client.execute("return {localStorage:{...localStorage}, sessionStorage:{...sessionStorage}, cookies:document.cookie};", vec![]).await?.to_string())
}

#[cfg(target_os = "macos")]
async fn wait_for(client: &mut WebDriverClient, arguments: &Value) -> Result<String> {
    let condition = arguments
        .get("condition")
        .and_then(Value::as_str)
        .context("browser_wait requires condition")?;
    let target = arguments
        .get("target")
        .and_then(Value::as_str)
        .unwrap_or("");
    let deadline = Instant::now()
        + Duration::from_secs(
            arguments
                .get("timeout_ms")
                .and_then(Value::as_u64)
                .unwrap_or(5000)
                / 1000
                + 1,
        );
    loop {
        let ok = match condition {
            "element" => client.find_css(target).await.is_ok(),
            "text" => client
                .execute(
                    "return document.body && document.body.innerText.includes(arguments[0]);",
                    vec![json!(target)],
                )
                .await?
                .as_bool()
                .unwrap_or(false),
            "url" => client.current_url().await?.as_str().contains(target),
            "javascript" => client
                .execute(target, vec![])
                .await?
                .as_bool()
                .unwrap_or(false),
            _ => bail!("unsupported browser_wait condition: {condition}"),
        };
        if ok {
            return Ok(json!({"ok":true,"condition":condition,"target":target}).to_string());
        }
        if Instant::now() >= deadline {
            bail!("timed out waiting for {condition}: {target}");
        }
        sleep(WAIT_POLL).await;
    }
}

#[cfg(target_os = "macos")]
async fn reload(client: &mut WebDriverClient, arguments: &Value) -> Result<String> {
    if arguments
        .get("clear_storage")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let _ = client
            .execute("localStorage.clear(); sessionStorage.clear();", vec![])
            .await;
    }
    client.refresh().await?;
    inject_instrumentation(client).await?;
    Ok(json!({"ok":true,"url":client.current_url().await?.as_str()}).to_string())
}

fn react_unsupported(tool: &str) -> Result<String> {
    Ok(json!({"ok": false, "tool": tool, "unsupported": true, "reason": "React diagnostics require the React DevTools global hook to be available in the page; this first WebDriver implementation reports unsupported when the hook is absent or inaccessible."}).to_string())
}

#[cfg(target_os = "macos")]
async fn screenshot(client: &mut WebDriverClient, arguments: &Value) -> Result<String> {
    let bytes = if let Some(target) = arguments.get("target").and_then(Value::as_str) {
        resolve_element(client, target).await?.screenshot().await?
    } else {
        client.screenshot().await?
    };
    Ok(json!({"mime":"image/png", "base64": BASE64.encode(bytes)}).to_string())
}

#[cfg(target_os = "macos")]
async fn debug_report(client: &mut WebDriverClient) -> Result<String> {
    inject_instrumentation(client).await?;
    let url = client.current_url().await.ok().map(|u| u.to_string());
    let shot = client.screenshot().await.ok().map(|b| BASE64.encode(b));
    let dom = client
        .execute(
            "return document.documentElement.outerHTML.slice(0,100000);",
            vec![],
        )
        .await
        .ok();
    let console = client
        .execute("return window.__pbBrowserDebug?.console || [];", vec![])
        .await
        .ok();
    let network = client.execute("return (window.__pbBrowserDebug?.network || []).filter(r=>r.error || (r.status && r.status >= 400));", vec![]).await.ok();
    let storage = client.execute("return {localStorage:{...localStorage}, sessionStorage:{...sessionStorage}, cookies:document.cookie};", vec![]).await.ok();
    Ok(json!({"url":url,"screenshot":{"mime":"image/png","base64":shot},"dom":dom,"console":console,"failed_requests":network,"storage":storage,"scope_note":"Full Safari Web Inspector features such as native debugger breakpoints, source-map debugging, and complete performance tracing are outside this WebDriver-based scope."}).to_string())
}

#[cfg(target_os = "macos")]
async fn close() -> Result<String> {
    let session = SESSION
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| anyhow!("browser session lock poisoned"))?
        .take();
    if let Some(mut session) = session {
        let _ = session.client.close().await;
        let _ = session.driver.kill();
        let _ = session.driver.wait();
    }
    Ok(json!({"ok": true, "closed": true}).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn browser_open_explains_safaridriver_enable_when_unavailable() {
        let err = call_tool("browser_open", &json!({"url":"http://127.0.0.1/"})).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("Safari automation is only available on macOS"));
        assert!(text.contains("safaridriver --enable"));
    }

    #[test]
    fn react_tools_report_optional_unsupported_capability() {
        let value: Value =
            serde_json::from_str(&call_tool("react_tree", &json!({})).unwrap()).unwrap();
        assert_eq!(value["unsupported"], true);
        assert!(
            value["reason"]
                .as_str()
                .unwrap()
                .contains("React DevTools global hook")
        );
    }
}
