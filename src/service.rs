//! `pb service` — integrate `pb serve` with launchd (macOS).
//!
//! Sub-commands:
//! - `start`   — ask launchd to start the service immediately
//! - `stop`    — ask launchd to stop the service
//! - `restart` — ask launchd to restart the service

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

pub const LABEL: &str = "com.jc2k.pb";
pub const LEGACY_TRAY_LABEL: &str = "com.jc2k.pb.tray";

/// Path to the pb serve LaunchAgent plist file.
pub fn plist_path() -> Result<PathBuf> {
    launch_agent_plist_path(LABEL)
}

/// Path to the pre-merged menu bar tray LaunchAgent plist file.
///
/// The tray now starts inside `pb serve`, but service removal still cleans up
/// older installs that wrote a second LaunchAgent.
pub fn legacy_tray_plist_path() -> Result<PathBuf> {
    launch_agent_plist_path(LEGACY_TRAY_LABEL)
}

fn launch_agent_plist_path(label: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home
        .join("Library/LaunchAgents")
        .join(format!("{label}.plist")))
}

/// Render the plist XML for `pb serve`.
#[cfg(target_os = "macos")]
fn render_plist(exe: &str) -> String {
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
        <string>serve</string>
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
        log_out = log_dir.join("pb.stdout.log").display(),
        log_err = log_dir.join("pb.stderr.log").display(),
    )
}

/// Install the LaunchAgent plist for a specific pb binary path and load it with launchctl.
pub fn install(exe: &Path) -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = exe;
        bail!("pb service is only supported on macOS");
    }

    #[cfg(target_os = "macos")]
    {
        let exe = exe.to_string_lossy().into_owned();

        let plist = plist_path()?;
        let legacy_tray_plist = legacy_tray_plist_path()?;
        let plist_dir = plist.parent().expect("plist path has no parent");
        std::fs::create_dir_all(plist_dir)
            .with_context(|| format!("failed to create {}", plist_dir.display()))?;

        write_plist(&plist, &render_plist(&exe))?;
        if legacy_tray_plist.exists() {
            unload_plist(LEGACY_TRAY_LABEL, &legacy_tray_plist)?;
            remove_plist(LEGACY_TRAY_LABEL, &legacy_tray_plist)?;
        }

        load_plist(&plist)?;

        println!("Service {LABEL} loaded. It will start automatically on login.");
        Ok(())
    }
}

/// Rewrite the installed LaunchAgent plist so it points at the given binary path.
///
/// If the rendered plist changed, unload and load it so launchd observes the new
/// configuration before the service is restarted or started.
pub fn refresh_plist_and_reload_if_changed(exe: &Path) -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = exe;
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        let plist = plist_path()?;
        let legacy_tray_plist = legacy_tray_plist_path()?;
        if !plist.exists() && !legacy_tray_plist.exists() {
            println!("pb launchd service is not installed; skipping plist refresh.");
            return Ok(());
        }

        if let Some(plist_dir) = plist.parent() {
            std::fs::create_dir_all(plist_dir)
                .with_context(|| format!("failed to create {}", plist_dir.display()))?;
        }

        let was_loaded = service_is_loaded(LABEL)?;
        let exe = exe.to_string_lossy().into_owned();
        let plist_changed = write_plist_if_changed(&plist, &render_plist(&exe))?;
        if plist_changed {
            if was_loaded {
                unload_plist(LABEL, &plist)?;
            }
            load_plist(&plist)?;
        } else if !was_loaded {
            load_plist(&plist)?;
        }

        if legacy_tray_plist.exists() {
            unload_plist_if_loaded(LEGACY_TRAY_LABEL, &legacy_tray_plist)?;
            remove_plist(LEGACY_TRAY_LABEL, &legacy_tray_plist)?;
        }

        restart_or_start_if_installed()
    }
}

/// Rewrite the installed LaunchAgent plist so it points at the given binary path.
pub fn refresh_plist_if_installed(exe: &Path) -> Result<()> {
    refresh_plist_and_reload_if_changed(exe)
}

#[cfg(target_os = "macos")]
fn write_plist(path: &PathBuf, content: &str) -> Result<()> {
    std::fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))?;
    println!("Wrote {}", path.display());
    Ok(())
}

#[cfg(target_os = "macos")]
fn write_plist_if_changed(path: &PathBuf, content: &str) -> Result<bool> {
    if path.exists() {
        let existing = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if existing == content {
            println!("{} is already up to date.", path.display());
            return Ok(false);
        }
    }

    write_plist(path, content)?;
    Ok(true)
}

#[cfg(target_os = "macos")]
fn load_plist(path: &Path) -> Result<()> {
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
fn unload_plist(label: &str, path: &Path) -> Result<()> {
    use std::process::Command;

    if !path.exists() {
        println!(
            "No plist found at {}; nothing to unload for {label}.",
            path.display()
        );
        return Ok(());
    }

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
    println!("Service {label} unloaded.");
    Ok(())
}

#[cfg(target_os = "macos")]
fn remove_plist(label: &str, path: &PathBuf) -> Result<()> {
    if path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("failed to remove {}", path.display()))?;
        println!("Removed plist for {label}: {}", path.display());
    } else {
        println!(
            "No plist found at {}; nothing to remove for {label}.",
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

#[cfg(target_os = "macos")]
fn service_is_loaded(label: &str) -> Result<bool> {
    use std::process::Command;

    let uid_output = Command::new("id")
        .arg("-u")
        .output()
        .context("failed to determine current user id")?;
    if !uid_output.status.success() {
        bail!("id -u failed (exit {})", uid_output.status);
    }
    let uid = String::from_utf8_lossy(&uid_output.stdout);
    let domain_target = format!("gui/{}/{}", uid.trim(), label);

    let output = Command::new("launchctl")
        .args(["print", &domain_target])
        .output()
        .context("failed to run launchctl")?;
    Ok(output.status.success())
}

#[cfg(target_os = "macos")]
fn service_is_running(label: &str) -> Result<bool> {
    use std::process::Command;

    let uid_output = Command::new("id")
        .arg("-u")
        .output()
        .context("failed to determine current user id")?;
    if !uid_output.status.success() {
        bail!("id -u failed (exit {})", uid_output.status);
    }
    let uid = String::from_utf8_lossy(&uid_output.stdout);
    let domain_target = format!("gui/{}/{}", uid.trim(), label);

    let output = Command::new("launchctl")
        .args(["print", &domain_target])
        .output()
        .context("failed to run launchctl")?;
    if !output.status.success() {
        return Ok(false);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().any(|line| {
        let trimmed = line.trim_start();
        trimmed.starts_with("state = running") || trimmed.starts_with("pid =")
    }))
}

#[cfg(target_os = "macos")]
fn unload_plist_if_loaded(label: &str, path: &Path) -> Result<()> {
    if service_is_loaded(label)? {
        unload_plist(label, path)?;
    }
    Ok(())
}

/// Stop the service, remove LaunchAgent plists, and leave the binary untouched.
pub fn remove() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    bail!("pb service is only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        let legacy_tray = legacy_tray_plist_path()?;
        let serve = plist_path()?;
        unload_plist(LEGACY_TRAY_LABEL, &legacy_tray)?;
        unload_plist(LABEL, &serve)?;
        remove_plist(LEGACY_TRAY_LABEL, &legacy_tray)?;
        remove_plist(LABEL, &serve)?;
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
        let _ = launchctl_signal("stop", LEGACY_TRAY_LABEL);
        launchctl_signal("stop", LABEL)?;
        println!("Service {LABEL} stopped.");
        Ok(())
    }
}

/// `pb service restart` — stop and then start the service via launchctl.
pub fn restart() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    bail!("pb service is only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        let _ = launchctl_signal("stop", LEGACY_TRAY_LABEL);
        let _ = launchctl_signal("stop", LABEL);
        launchctl_signal("start", LABEL)?;
        println!("Service {LABEL} restarted.");
        Ok(())
    }
}

/// Restart installed services if they are running; otherwise start them.
pub fn restart_or_start_if_installed() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    return Ok(());

    #[cfg(target_os = "macos")]
    {
        if !plist_path()?.exists() && !legacy_tray_plist_path()?.exists() {
            println!("pb launchd service is not installed; skipping service restart.");
            return Ok(());
        }

        if service_is_running(LABEL)? || service_is_running(LEGACY_TRAY_LABEL)? {
            restart()
        } else {
            start()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plist_path_contains_label() {
        let path = plist_path().unwrap();
        assert!(path.to_string_lossy().contains(LABEL));
        assert!(path.extension().map(|e| e == "plist").unwrap_or(false));
    }

    #[test]
    fn test_legacy_tray_plist_path_contains_label() {
        let path = legacy_tray_plist_path().unwrap();
        assert!(path.to_string_lossy().contains(LEGACY_TRAY_LABEL));
        assert!(path.extension().map(|e| e == "plist").unwrap_or(false));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_render_plist_contains_label() {
        let exe = "/usr/local/bin/pb";
        let plist = render_plist(exe);
        assert!(plist.contains(LABEL));
        assert!(plist.contains(exe));
        assert!(plist.contains("serve"));
        assert!(!plist.contains("--host"));
        assert!(!plist.contains("--port"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_render_plist_has_no_pb_configuration_args() {
        let plist = render_plist("/usr/local/bin/pb");
        assert!(!plist.contains("--host"));
        assert!(!plist.contains("--port"));
        assert!(!plist.contains("--model"));
        assert!(!plist.contains("--workdir"));
        assert!(!plist.contains("--socket-path"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_render_plist_log_paths() {
        let plist = render_plist("/usr/local/bin/pb");
        assert!(plist.contains("pb.stdout.log"));
        assert!(plist.contains("pb.stderr.log"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_plist_valid_xml_structure() {
        let plist = render_plist("/usr/local/bin/pb");
        assert!(plist.starts_with("<?xml"));
        assert!(plist.contains("<plist version=\"1.0\">"));
        assert!(plist.contains("</plist>"));
        assert!(plist.contains("<key>Label</key>"));
        assert!(plist.contains("<key>ProgramArguments</key>"));
        assert!(plist.contains("<key>RunAtLoad</key>"));
        assert!(plist.contains("<key>KeepAlive</key>"));
    }
}
