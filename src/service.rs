//! `pb service` — integrate `pb serve` with launchd (macOS).
//!
//! Sub-commands:
//! - `enable`  — write a LaunchAgent plist referencing the current binary and load it
//! - `disable` — unload the LaunchAgent and remove the plist
//! - `start`   — ask launchd to start the service immediately
//! - `stop`    — ask launchd to stop the service

use anyhow::{Context, Result, bail};
use std::path::PathBuf;

use crate::ServeArgs;

pub const LABEL: &str = "com.jc2k.pb";
pub const TRAY_LABEL: &str = "com.jc2k.pb.tray";

/// Path to the pb serve LaunchAgent plist file.
pub fn plist_path() -> Result<PathBuf> {
    launch_agent_plist_path(LABEL)
}

/// Path to the menu bar tray LaunchAgent plist file.
pub fn tray_plist_path() -> Result<PathBuf> {
    launch_agent_plist_path(TRAY_LABEL)
}

fn launch_agent_plist_path(label: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home
        .join("Library/LaunchAgents")
        .join(format!("{label}.plist")))
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
        program_args.push(format!("        <string>{}</string>", model_dir.display()));
    }
    if let Some(ref workdir) = args.workdir {
        program_args.push("        <string>--workdir</string>".to_string());
        program_args.push(format!("        <string>{}</string>", workdir.display()));
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

#[cfg(target_os = "macos")]
fn render_tray_plist(exe: &str, args: &ServeArgs) -> String {
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
        <string>{exe}</string>
        <string>tray</string>
        <string>--host</string>
        <string>{host}</string>
        <string>--port</string>
        <string>{port}</string>
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
        label = TRAY_LABEL,
        exe = exe,
        host = args.host,
        port = args.port,
        log_out = log_dir.join("pb.tray.stdout.log").display(),
        log_err = log_dir.join("pb.tray.stderr.log").display(),
    )
}

/// `pb service enable` — write the plist and load it with launchctl.
pub fn enable(args: &ServeArgs) -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = args;
        bail!("pb service is only supported on macOS");
    }

    #[cfg(target_os = "macos")]
    {
        let exe = std::env::current_exe()
            .context("cannot determine path to pb binary")?
            .to_string_lossy()
            .into_owned();

        let plist = plist_path()?;
        let tray_plist = tray_plist_path()?;
        let plist_dir = plist.parent().expect("plist path has no parent");
        std::fs::create_dir_all(plist_dir)
            .with_context(|| format!("failed to create {}", plist_dir.display()))?;

        write_plist(&plist, &render_plist(&exe, args))?;
        write_plist(&tray_plist, &render_tray_plist(&exe, args))?;

        load_plist(&plist)?;
        load_plist(&tray_plist)?;

        println!(
            "Services {LABEL} and {TRAY_LABEL} loaded. They will start automatically on login."
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn write_plist(path: &PathBuf, content: &str) -> Result<()> {
    std::fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))?;
    println!("Wrote {}", path.display());
    Ok(())
}

#[cfg(target_os = "macos")]
fn load_plist(path: &PathBuf) -> Result<()> {
    use std::process::Command;

    let status = Command::new("launchctl")
        .args([
            "load",
            "-w",
            path.to_str().context("plist path contains invalid UTF-8")?,
        ])
        .status()
        .context("failed to run launchctl")?;
    if !status.success() {
        bail!(
            "launchctl load failed for {} (exit {})",
            path.display(),
            status
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn unload_and_remove_plist(label: &str, path: &PathBuf) -> Result<()> {
    use std::process::Command;

    if path.exists() {
        let status = Command::new("launchctl")
            .args([
                "unload",
                "-w",
                path.to_str().context("plist path contains invalid UTF-8")?,
            ])
            .status()
            .context("failed to run launchctl")?;
        if !status.success() {
            bail!("launchctl unload failed for {label} (exit {})", status);
        }
        std::fs::remove_file(path)
            .with_context(|| format!("failed to remove {}", path.display()))?;
        println!("Service {label} unloaded and plist removed.");
    } else {
        println!(
            "No plist found at {}; nothing to disable for {label}.",
            path.display()
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn launchctl_signal(action: &str, label: &str) -> Result<()> {
    use std::process::Command;

    let status = Command::new("launchctl")
        .args([action, label])
        .status()
        .context("failed to run launchctl")?;
    if !status.success() {
        bail!("launchctl {action} failed for {label} (exit {status})");
    }
    Ok(())
}

/// `pb service disable` — unload the service and remove the plist.
pub fn disable() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    bail!("pb service is only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        unload_and_remove_plist(TRAY_LABEL, &tray_plist_path()?)?;
        unload_and_remove_plist(LABEL, &plist_path()?)?;
        Ok(())
    }
}

/// `pb service start` — start the service immediately via launchctl.
pub fn start() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    bail!("pb service is only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        launchctl_signal("start", LABEL)?;
        launchctl_signal("start", TRAY_LABEL)?;
        println!("Services {LABEL} and {TRAY_LABEL} started.");
        Ok(())
    }
}

/// `pb service stop` — stop the service immediately via launchctl.
pub fn stop() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    bail!("pb service is only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        launchctl_signal("stop", TRAY_LABEL)?;
        launchctl_signal("stop", LABEL)?;
        println!("Services {LABEL} and {TRAY_LABEL} stopped.");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use crate::{
        DEFAULT_AGENT_MAX_STEPS, DEFAULT_AGENT_MAX_TOKENS, DEFAULT_MODEL, default_gpu_layers,
    };

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

    #[test]
    fn test_tray_plist_path_contains_label() {
        let path = tray_plist_path().unwrap();
        assert!(path.to_string_lossy().contains(TRAY_LABEL));
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
