//! `pb service` — integrate `pb serve` with launchd (macOS).
//!
//! Sub-commands:
//! - `start`   — ask launchd to start the service immediately
//! - `stop`    — ask launchd to stop the service
//! - `restart` — restart the service immediately

use anyhow::{Context, Result, bail};
use std::path::PathBuf;

use crate::ServeArgs;

pub const LABEL: &str = "com.jc2k.pb";

/// Path to the LaunchAgent plist file.
pub fn plist_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home
        .join("Library/LaunchAgents")
        .join(format!("{LABEL}.plist")))
}

/// Render the plist XML for a given `pb serve` invocation.
#[cfg(target_os = "macos")]
fn render_plist(exe: &str, args: &ServeArgs) -> String {
    let mut program_args = vec![
        format!("        <string>{exe}</string>"),
        "        <string>serve</string>".to_string(),
        "        <string>--host</string>".to_string(),
        format!("        <string>{}</string>", args.host),
        "        <string>--port</string>".to_string(),
        format!("        <string>{}</string>", args.port),
        "        <string>--model</string>".to_string(),
        format!("        <string>{}</string>", args.model),
        "        <string>--gpu-layers</string>".to_string(),
        format!("        <string>{}</string>", args.gpu_layers),
        "        <string>--max-steps</string>".to_string(),
        format!("        <string>{}</string>", args.max_steps),
        "        <string>--max-tokens</string>".to_string(),
        format!("        <string>{}</string>", args.max_tokens),
        "        <string>--ctx-size</string>".to_string(),
        format!("        <string>{}</string>", args.ctx_size),
        "        <string>--temperature</string>".to_string(),
        format!("        <string>{}</string>", args.temperature),
        "        <string>--top-k</string>".to_string(),
        format!("        <string>{}</string>", args.top_k),
        "        <string>--seed</string>".to_string(),
        format!("        <string>{}</string>", args.seed),
    ];

    if let Some(ref model_dir) = args.model_dir {
        program_args.push("        <string>--model-dir</string>".to_string());
        program_args.push(format!(
            "        <string>{}</string>",
            model_dir.display()
        ));
    }
    if let Some(ref workdir) = args.workdir {
        program_args.push("        <string>--workdir</string>".to_string());
        program_args.push(format!(
            "        <string>{}</string>",
            workdir.display()
        ));
    }
    if let Some(threads) = args.threads {
        program_args.push("        <string>--threads</string>".to_string());
        program_args.push(format!("        <string>{threads}</string>"));
    }
    if let Some(threads_batch) = args.threads_batch {
        program_args.push("        <string>--threads-batch</string>".to_string());
        program_args.push(format!("        <string>{threads_batch}</string>"));
    }

    let log_dir = dirs::home_dir()
        .map(|h| h.join("Library/Logs"))
        .unwrap_or_else(|| PathBuf::from("/tmp"));

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
{args_str}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_out}</string>
    <key>StandardErrorPath</key>
    <string>{log_err}</string>
</dict>
</plist>
"#,
        label = LABEL,
        args_str = program_args.join("\n"),
        log_out = log_dir.join("pb.stdout.log").display(),
        log_err = log_dir.join("pb.stderr.log").display(),
    )
}

/// Install the LaunchAgent plist for a given pb binary and load it with launchctl.
pub fn install(exe: &str, args: &ServeArgs) -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = exe;
        let _ = args;
        bail!("pb service is only supported on macOS");
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        let plist = plist_path()?;
        let plist_dir = plist.parent().expect("plist path has no parent");
        std::fs::create_dir_all(plist_dir)
            .with_context(|| format!("failed to create {}", plist_dir.display()))?;

        let content = render_plist(exe, args);
        std::fs::write(&plist, &content)
            .with_context(|| format!("failed to write {}", plist.display()))?;
        println!("Wrote {}", plist.display());

        let status = Command::new("launchctl")
            .args([
                "load",
                "-w",
                plist
                    .to_str()
                    .context("plist path contains invalid UTF-8")?,
            ])
            .status()
            .context("failed to run launchctl")?;
        if !status.success() {
            bail!("launchctl load failed (exit {})", status);
        }
        println!("Service {LABEL} loaded. It will start automatically on login.");
        Ok(())
    }
}

/// Unload the service and remove the LaunchAgent plist.
pub fn uninstall() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    bail!("pb service is only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        let plist = plist_path()?;

        if plist.exists() {
            let status = Command::new("launchctl")
                .args([
                    "unload",
                    "-w",
                    plist
                        .to_str()
                        .context("plist path contains invalid UTF-8")?,
                ])
                .status()
                .context("failed to run launchctl")?;
            if !status.success() {
                bail!("launchctl unload failed (exit {})", status);
            }
            std::fs::remove_file(&plist)
                .with_context(|| format!("failed to remove {}", plist.display()))?;
            println!("Service {LABEL} unloaded and plist removed.");
        } else {
            println!("No plist found at {}; nothing to uninstall.", plist.display());
        }
        Ok(())
    }
}

/// `pb service start` — start the service immediately via launchctl.
pub fn start() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    bail!("pb service is only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        let status = Command::new("launchctl")
            .args(["start", LABEL])
            .status()
            .context("failed to run launchctl")?;
        if !status.success() {
            bail!(
                "launchctl start failed (exit {}). Is the service enabled?",
                status
            );
        }
        println!("Service {LABEL} started.");
        Ok(())
    }
}

/// `pb service stop` — stop the service immediately via launchctl.
pub fn stop() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    bail!("pb service is only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        let status = Command::new("launchctl")
            .args(["stop", LABEL])
            .status()
            .context("failed to run launchctl")?;
        if !status.success() {
            bail!(
                "launchctl stop failed (exit {}). Is the service running?",
                status
            );
        }
        println!("Service {LABEL} stopped.");
        Ok(())
    }
}

/// Return true when launchd knows about the pb service.
pub fn is_loaded() -> Result<bool> {
    #[cfg(not(target_os = "macos"))]
    bail!("pb service is only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        let uid = std::process::Command::new("id")
            .arg("-u")
            .output()
            .context("failed to determine current user id")?;
        if !uid.status.success() {
            bail!("id -u failed (exit {})", uid.status);
        }
        let uid = String::from_utf8(uid.stdout)
            .context("id -u output was not UTF-8")?
            .trim()
            .to_owned();
        let status = Command::new("launchctl")
            .args(["print", &format!("gui/{uid}/{LABEL}")])
            .status()
            .context("failed to run launchctl")?;
        Ok(status.success())
    }
}

/// `pb service restart` — restart the service via launchctl.
pub fn restart() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    bail!("pb service is only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        let _ = stop();
        start()
    }
}

/// Restart the service if loaded; otherwise start it from the LaunchAgent plist.
pub fn restart_or_start() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    bail!("pb service is only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        if is_loaded()? {
            restart()
        } else {
            let plist = plist_path()?;
            if plist.exists() {
                use std::process::Command;

                let status = Command::new("launchctl")
                    .args([
                        "load",
                        "-w",
                        plist
                            .to_str()
                            .context("plist path contains invalid UTF-8")?,
                    ])
                    .status()
                    .context("failed to run launchctl")?;
                if !status.success() {
                    bail!("launchctl load failed (exit {})", status);
                }
                println!("Service {LABEL} loaded and started.");
                Ok(())
            } else {
                start()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use crate::{DEFAULT_AGENT_MAX_STEPS, DEFAULT_AGENT_MAX_TOKENS, DEFAULT_MODEL, default_gpu_layers};

    #[cfg(target_os = "macos")]
    fn default_serve_args() -> ServeArgs {
        ServeArgs {
            host: "127.0.0.1".to_string(),
            port: 8311,
            model: DEFAULT_MODEL.to_string(),
            model_dir: None,
            workdir: None,
            max_steps: DEFAULT_AGENT_MAX_STEPS,
            max_tokens: DEFAULT_AGENT_MAX_TOKENS,
            ctx_size: 8192,
            threads: None,
            threads_batch: None,
            gpu_layers: default_gpu_layers(),
            temperature: 0.2,
            top_k: 40,
            seed: 1337,
        }
    }

    #[test]
    fn test_plist_path_contains_label() {
        let path = plist_path().unwrap();
        assert!(path.to_string_lossy().contains(LABEL));
        assert!(path.extension().map(|e| e == "plist").unwrap_or(false));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_render_plist_contains_label() {
        let args = default_serve_args();
        let exe = "/usr/local/bin/pb";
        let plist = render_plist(exe, &args);
        assert!(plist.contains(LABEL));
        assert!(plist.contains(exe));
        assert!(plist.contains("serve"));
        assert!(plist.contains("127.0.0.1"));
        assert!(plist.contains("8311"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_render_plist_with_optional_args() {
        let mut args = default_serve_args();
        args.threads = Some(4);
        args.threads_batch = Some(8);
        args.model_dir = Some(PathBuf::from("/models"));
        args.workdir = Some(PathBuf::from("/workspace"));
        let plist = render_plist("/usr/local/bin/pb", &args);
        assert!(plist.contains("--threads"));
        assert!(plist.contains("--threads-batch"));
        assert!(plist.contains("--model-dir"));
        assert!(plist.contains("--workdir"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_render_plist_log_paths() {
        let args = default_serve_args();
        let plist = render_plist("/usr/local/bin/pb", &args);
        assert!(plist.contains("pb.stdout.log"));
        assert!(plist.contains("pb.stderr.log"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_plist_valid_xml_structure() {
        let args = default_serve_args();
        let plist = render_plist("/usr/local/bin/pb", &args);
        assert!(plist.starts_with("<?xml"));
        assert!(plist.contains("<plist version=\"1.0\">"));
        assert!(plist.contains("</plist>"));
        assert!(plist.contains("<key>Label</key>"));
        assert!(plist.contains("<key>ProgramArguments</key>"));
        assert!(plist.contains("<key>RunAtLoad</key>"));
        assert!(plist.contains("<key>KeepAlive</key>"));
    }
}
