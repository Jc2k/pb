use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use crate::container::{
    ContainerHandle, ContainerLaunchSpec, ContainerMount, ContainerResources, ContainerRuntime,
    NetworkSpec, VolumeSpec,
};
use crate::session_workspace::WorkspaceStrategy;

const WORKLOAD: &str = r#"set -eu
rm -rf /benchmark/pb-fs-bench
mkdir -p /benchmark/pb-fs-bench
i=0
while test "$i" -lt 2000; do
  printf 'pub const VALUE_%s: usize = %s;\n' "$i" "$i" > "/benchmark/pb-fs-bench/file_$i.rs"
  i=$((i + 1))
done
find /benchmark/pb-fs-bench -type f | sort | xargs cat | cksum
find /benchmark/pb-fs-bench -type f -exec test -s {} \;
"#;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilesystemBenchmarkSample {
    pub strategy: WorkspaceStrategy,
    pub elapsed_ms: u64,
    pub checksum: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilesystemBenchmarkReport {
    pub version: u32,
    pub runtime: String,
    pub runtime_version: String,
    pub image_fingerprint_sha256: String,
    pub iterations: u32,
    pub samples: Vec<FilesystemBenchmarkSample>,
    pub checksums_match: bool,
    pub volume_sync_implemented: bool,
    pub selected_strategy: WorkspaceStrategy,
    pub decision: String,
}

pub fn run(
    runtime: Box<dyn ContainerRuntime>,
    image: &str,
    repository_root: &Path,
    iterations: u32,
) -> Result<FilesystemBenchmarkReport> {
    if iterations == 0 {
        bail!("filesystem benchmark iterations must be greater than zero");
    }
    if !runtime.image_exists(image)? {
        bail!("benchmark image '{image}' is not local; run pb env start first");
    }
    let info = runtime.info()?;
    let image_fingerprint =
        crate::environment_lock::sha256(runtime.image_fingerprint(image)?.as_bytes());
    let nonce = &crate::environment_lock::sha256(
        format!(
            "{}\0{}\0{}",
            repository_root.display(),
            image_fingerprint,
            std::process::id()
        )
        .as_bytes(),
    )[..12];
    let network_name = format!("pb-bench-net-{nonce}");
    let volume_name = format!("pb-bench-volume-{nonce}");
    let labels = BTreeMap::from([
        ("dev.pb.managed".to_string(), "true".to_string()),
        ("dev.pb.project".to_string(), "benchmark".to_string()),
        ("dev.pb.session".to_string(), nonce.to_string()),
        (
            "dev.pb.role".to_string(),
            "filesystem-benchmark".to_string(),
        ),
    ]);
    runtime.create_internal_network(&NetworkSpec {
        name: network_name.clone(),
        labels: labels.clone(),
    })?;
    let volume_spec = VolumeSpec {
        name: volume_name.clone(),
        labels: labels.clone(),
    };
    if let Err(error) = runtime.ensure_volume(&volume_spec) {
        let _ = runtime.remove_network(&network_name);
        return Err(error).context("failed to create filesystem benchmark volume");
    }
    if let Err(error) = runtime.ensure_volume(&volume_spec) {
        let _ = runtime.remove_volume(&volume_name);
        let _ = runtime.remove_network(&network_name);
        return Err(error).context("failed to validate filesystem benchmark volume ownership");
    }
    let bind_root = tempfile::tempdir().context("failed to create bind benchmark directory")?;
    let mut samples = Vec::new();
    let result = (|| {
        for (strategy, mount) in [
            (
                WorkspaceStrategy::WorktreeBind,
                ContainerMount::bind(bind_root.path(), "/benchmark", false),
            ),
            (
                WorkspaceStrategy::ContainerVolume,
                ContainerMount::volume(&volume_name, "/benchmark"),
            ),
        ] {
            let name = format!(
                "pb-bench-{}-{nonce}",
                match strategy {
                    WorkspaceStrategy::WorktreeBind => "bind",
                    WorkspaceStrategy::ContainerVolume => "volume",
                }
            );
            let container_id = runtime.create(&ContainerLaunchSpec {
                name,
                image: image.to_string(),
                workdir: "/benchmark".to_string(),
                mounts: vec![mount],
                labels: labels.clone(),
                env: BTreeMap::new(),
                network: Some(network_name.clone()),
                resources: ContainerResources {
                    cpus: 2,
                    memory_mb: 2_048,
                },
                tmpfs: vec!["/tmp".to_string(), "/run".to_string()],
                read_only_root: true,
            })?;
            let handle = ContainerHandle {
                runtime: crate::container::runtime_for_binary(&info.binary)?,
                container_id,
                network: None,
            };
            for _ in 0..iterations {
                let started = Instant::now();
                let checksum = handle.exec(WORKLOAD)?;
                samples.push(FilesystemBenchmarkSample {
                    strategy,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                    checksum,
                });
            }
            drop(handle);
        }
        Ok::<_, anyhow::Error>(())
    })();
    let volume_cleanup = runtime
        .remove_volume(&volume_name)
        .context("failed to remove filesystem benchmark volume");
    let network_cleanup = runtime
        .remove_network(&network_name)
        .context("failed to remove filesystem benchmark network");
    result?;
    volume_cleanup?;
    network_cleanup?;

    let bind_checksums = samples
        .iter()
        .filter(|sample| sample.strategy == WorkspaceStrategy::WorktreeBind)
        .map(|sample| sample.checksum.as_str())
        .collect::<Vec<_>>();
    let volume_checksums = samples
        .iter()
        .filter(|sample| sample.strategy == WorkspaceStrategy::ContainerVolume)
        .map(|sample| sample.checksum.as_str())
        .collect::<Vec<_>>();
    let checksums_match = !bind_checksums.is_empty()
        && bind_checksums
            .iter()
            .all(|checksum| *checksum == bind_checksums[0])
        && volume_checksums
            .iter()
            .all(|checksum| *checksum == bind_checksums[0]);
    let volume_sync_implemented = false;
    let selected_strategy = WorkspaceStrategy::WorktreeBind;
    Ok(FilesystemBenchmarkReport {
        version: 1,
        runtime: info.binary,
        runtime_version: info.version,
        image_fingerprint_sha256: image_fingerprint,
        iterations,
        samples,
        checksums_match,
        volume_sync_implemented,
        selected_strategy,
        decision: "worktree_bind remains mandatory: it preserves live edits, Git semantics, review, and crash recovery; container_volume is not eligible until transactional synchronization exists".to_string(),
    })
}

pub fn save_report(report: &FilesystemBenchmarkReport, repository_root: &Path) -> Result<()> {
    let path = repository_root
        .join(".pb")
        .join("benchmarks")
        .join("workspace-filesystem.json");
    let parent = path
        .parent()
        .context("benchmark report path has no parent")?;
    std::fs::create_dir_all(parent)?;
    let temp = parent.join(format!(".workspace-filesystem.{}.tmp", std::process::id()));
    std::fs::write(&temp, serde_json::to_vec_pretty(report)?)?;
    std::fs::rename(&temp, &path).with_context(|| format!("failed to replace {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_schema_keeps_volume_ineligible_without_sync() {
        let report = FilesystemBenchmarkReport {
            version: 1,
            runtime: "container".to_string(),
            runtime_version: "1.0.0".to_string(),
            image_fingerprint_sha256: crate::environment_lock::sha256(b"image"),
            iterations: 1,
            samples: Vec::new(),
            checksums_match: true,
            volume_sync_implemented: false,
            selected_strategy: WorkspaceStrategy::WorktreeBind,
            decision: "correctness gate".to_string(),
        };
        assert_eq!(report.selected_strategy, WorkspaceStrategy::WorktreeBind);
        assert!(!report.volume_sync_implemented);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a running Apple container runtime and local or pullable Alpine image"]
    fn apple_bind_and_volume_benchmark_is_correct_and_leak_free() -> Result<()> {
        use crate::container::{AppleContainerRuntime, ContainerRuntime};

        let runtime = AppleContainerRuntime;
        let image = "docker.io/library/alpine:3.20";
        if !runtime.image_exists(image)? {
            runtime.pull(image)?;
        }
        let repository = tempfile::tempdir()?;
        let report = run(Box::new(AppleContainerRuntime), image, repository.path(), 2)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        assert!(report.checksums_match);
        assert_eq!(report.samples.len(), 4);
        assert_eq!(report.selected_strategy, WorkspaceStrategy::WorktreeBind);
        assert!(!runtime.list_managed_containers()?.iter().any(|resource| {
            resource
                .labels
                .get("dev.pb.role")
                .is_some_and(|role| role == "filesystem-benchmark")
        }));
        assert!(!runtime.list_managed_networks()?.iter().any(|resource| {
            resource
                .labels
                .get("dev.pb.role")
                .is_some_and(|role| role == "filesystem-benchmark")
        }));
        Ok(())
    }
}
