//! Built-in Safari WebDriver tools for local browser diagnostics.
//!
//! This first version intentionally targets Safari via `/usr/bin/safaridriver` on macOS.
//! It uses WebDriver plus page-side JavaScript instrumentation for console and network
//! information because Safari WebDriver does not expose the full Web Inspector protocol.
//! Native debugger breakpoints, source-map debugging, and complete performance tracing are
//! outside this initial WebDriver-based scope.

use anyhow::{Context, Result, anyhow, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use fantoccini::{Client, ClientBuilder, Locator};
use serde_json::{Value, json};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::time::{Instant, sleep, timeout};

const SAFARIDRIVER: &str = "/usr/bin/safaridriver";
const TOOL_TIMEOUT: Duration = Duration::from_secs(30);
const WAIT_POLL: Duration = Duration::from_millis(200);

static SESSION: OnceLock<Mutex<Option<BrowserSession>>> = OnceLock::new();

struct BrowserSession {
    client: Client,
    driver: Child,
    port: u16,
}

impl Drop for BrowserSession {
    fn drop(&mut self) {
        let _ = self.driver.kill();
        let _ = self.driver.wait();
    }
}

pub fn call_tool(tool: &str, arguments: &Value) -> Result<String> {
    block_on(async move {
        match tool {
            "browser_open" => open(arguments).await,
            "browser_snapshot" => with_client(|c| Box::pin(async move { snapshot(c).await })).await,
            "browser_interact" => {
                with_client(|c| Box::pin(async move { interact(c, arguments).await })).await
            }
            "browser_dom" => {
                with_client(|c| Box::pin(async move { dom(c, arguments).await })).await
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
                with_client(|c| Box::pin(async move { evaluate(c, arguments).await })).await
            }
            "browser_storage" => {
                with_client(|c| Box::pin(async move { storage(c, arguments).await })).await
            }
            "browser_wait" => {
                with_client(|c| Box::pin(async move { wait_for(c, arguments).await })).await
            }
            "browser_reload" => {
                with_client(|c| Box::pin(async move { reload(c, arguments).await })).await
            }
            "browser_screenshot" => {
                with_client(|c| Box::pin(async move { screenshot(c, arguments).await })).await
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

fn block_on<F: std::future::Future<Output = Result<String>>>(future: F) -> Result<String> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to create browser tool runtime")?
        .block_on(timeout(TOOL_TIMEOUT, future))
        .context("browser tool timed out")?
}

async fn with_client<F>(f: impl FnOnce(&mut Client) -> F) -> Result<String>
where
    F: std::future::Future<Output = Result<String>>,
{
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
            match ClientBuilder::native().connect(&endpoint).await {
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

async fn inject_instrumentation(client: &mut Client) -> Result<()> {
    let script = include_str!("browser_instrumentation.js");
    let _ = client.execute(script, vec![]).await;
    Ok(())
}

async fn js_json(client: &mut Client, script: &str) -> Result<String> {
    inject_instrumentation(client).await?;
    Ok(client
        .execute(script, vec![])
        .await
        .context("browser JavaScript execution failed")?
        .to_string())
}

async fn snapshot(client: &mut Client) -> Result<String> {
    js_json(client, r#"return Array.from(document.querySelectorAll('body *')).slice(0,500).map((el,i)=>{ if(!el.dataset.pbRef) el.dataset.pbRef='pb-'+i+'-'+Math.random().toString(36).slice(2); const r=el.getBoundingClientRect(); return {ref:el.dataset.pbRef, tag:el.tagName.toLowerCase(), id:el.id||null, classes:Array.from(el.classList), role:el.getAttribute('role'), name:el.getAttribute('aria-label')||el.innerText?.trim().slice(0,120)||el.getAttribute('title')||'', visible:!!(r.width||r.height), bounds:{x:r.x,y:r.y,width:r.width,height:r.height}};});"#).await
}

async fn resolve_element(
    client: &mut Client,
    target: &str,
) -> Result<fantoccini::elements::Element> {
    let selector = if target.starts_with("pb-") {
        format!("[data-pb-ref='{target}']")
    } else {
        target.to_string()
    };
    client
        .find(Locator::Css(&selector))
        .await
        .with_context(|| format!("failed to find element: {target}"))
}

async fn interact(client: &mut Client, arguments: &Value) -> Result<String> {
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
                .execute("arguments[0].focus();", vec![el.to_json()?])
                .await?;
        }
        "submit" => {
            client.execute("arguments[0].requestSubmit ? arguments[0].requestSubmit() : arguments[0].submit();", vec![el.to_json()?]).await?;
        }
        "select" => {
            client.execute("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('change',{bubbles:true}));", vec![el.to_json()?, json!(value)]).await?;
        }
        "hover" => {
            client
                .execute(
                    "arguments[0].dispatchEvent(new MouseEvent('mouseover',{bubbles:true}));",
                    vec![el.to_json()?],
                )
                .await?;
        }
        _ => bail!("unsupported browser_interact action: {action}"),
    }
    Ok(json!({"ok": true, "action": action, "target": target}).to_string())
}

async fn dom(client: &mut Client, arguments: &Value) -> Result<String> {
    let target = arguments
        .get("target")
        .and_then(Value::as_str)
        .context("browser_dom requires target")?;
    let el = resolve_element(client, target).await?;
    Ok(client.execute(r#"const el=arguments[0], r=el.getBoundingClientRect(), cs=getComputedStyle(el); return {html:el.outerHTML, text:el.innerText||el.textContent||'', attributes:Object.fromEntries(Array.from(el.attributes).map(a=>[a.name,a.value])), styles:{display:cs.display, visibility:cs.visibility, opacity:cs.opacity, position:cs.position}, bounds:{x:r.x,y:r.y,width:r.width,height:r.height}, visible:!!(r.width||r.height)&&cs.visibility!=='hidden'&&cs.display!=='none'};"#, vec![el.to_json()?]).await?.to_string())
}

async fn evaluate(client: &mut Client, arguments: &Value) -> Result<String> {
    let script = arguments
        .get("script")
        .and_then(Value::as_str)
        .context("browser_evaluate requires script")?;
    js_json(client, &format!("return (async()=>{{ {script} }})()")).await
}

async fn storage(client: &mut Client, arguments: &Value) -> Result<String> {
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

async fn wait_for(client: &mut Client, arguments: &Value) -> Result<String> {
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
            "element" => client.find(Locator::Css(target)).await.is_ok(),
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

async fn reload(client: &mut Client, arguments: &Value) -> Result<String> {
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

async fn screenshot(client: &mut Client, arguments: &Value) -> Result<String> {
    let bytes = if let Some(target) = arguments.get("target").and_then(Value::as_str) {
        resolve_element(client, target).await?.screenshot().await?
    } else {
        client.screenshot().await?
    };
    Ok(json!({"mime":"image/png", "base64": BASE64.encode(bytes)}).to_string())
}

async fn debug_report(client: &mut Client) -> Result<String> {
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
