use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use futures::{StreamExt, stream};
use reqwest::header::ACCEPT;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio::time::{Duration, sleep};

const DEFAULT_MODEL: &str = "qwen3-coder-next";
const OLLAMA_REGISTRY: &str = "https://registry.ollama.ai";

#[derive(Parser, Debug)]
#[command(name = "pb", about = "A local coding agent CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Self-management commands
    #[command(name = "self")]
    SelfCmd {
        #[command(subcommand)]
        command: SelfCommand,
    },
    /// Pull model blobs from the Ollama-compatible registry
    Pull(PullArgs),
    /// Run a simple local coding agent workflow with streamed progress
    Agent {
        /// Task to execute
        task: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum SelfCommand {
    /// Update pb from the latest GitHub release
    Update,
}

#[derive(Args, Debug)]
pub struct PullArgs {
    /// Model name in Ollama library
    #[arg(default_value = DEFAULT_MODEL)]
    pub model: String,

    /// Number of blobs to process in each recovery-safe batch
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    /// Maximum parallel blob downloads within each batch
    #[arg(long, default_value_t = default_parallelism())]
    pub parallel: usize,

    /// Number of retries per blob download
    #[arg(long, default_value_t = 4)]
    pub retries: u32,

    /// Output directory for downloaded blobs
    #[arg(long)]
    pub out_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
struct ManifestDescriptor {
    digest: String,
    size: u64,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    config: ManifestDescriptor,
    layers: Vec<ManifestDescriptor>,
}

#[derive(Debug)]
enum AgentEvent {
    Started(String),
    Step {
        index: usize,
        total: usize,
        message: String,
    },
    Complete,
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::SelfCmd { command } => match command {
            SelfCommand::Update => run_self_update(),
        },
        Commands::Pull(args) => pull_model(&args).await,
        Commands::Agent { task } => run_agent(&task).await,
    }
}

fn run_self_update() -> Result<()> {
    let target = release_target();
    let status = self_update::backends::github::Update::configure()
        .repo_owner("Jc2k")
        .repo_name("pb")
        .bin_name("pb")
        .target(&target)
        .show_download_progress(true)
        .current_version(self_update::cargo_crate_version!())
        .build()
        .context("failed to build self-update configuration")?
        .update()
        .context("self-update failed")?;

    println!("Updated to {}", status.version());
    Ok(())
}

pub async fn pull_model(args: &PullArgs) -> Result<()> {
    if args.batch_size == 0 {
        bail!("batch-size must be greater than 0");
    }
    if args.parallel == 0 {
        bail!("parallel must be greater than 0");
    }

    let output_root = args.out_dir.clone().unwrap_or_else(default_pull_dir);
    tokio::fs::create_dir_all(&output_root)
        .await
        .with_context(|| {
            format!(
                "failed to create output directory {}",
                output_root.display()
            )
        })?;

    let client = reqwest::Client::builder()
        .user_agent("pb/0.1.0")
        .build()
        .context("failed to build HTTP client")?;

    let manifest = fetch_manifest(&client, &args.model).await?;
    let descriptors = descriptors_from_manifest(manifest);

    let total = descriptors.len();
    let mut completed = 0usize;

    for batch in descriptors.chunks(args.batch_size) {
        println!("Starting batch with {} blobs", batch.len());

        let futures = stream::iter(batch.iter().cloned().map(|descriptor| {
            let client = client.clone();
            let model = args.model.clone();
            let output_root = output_root.clone();
            async move {
                download_blob_with_retry(&client, &model, &output_root, &descriptor, args.retries)
                    .await
            }
        }))
        .buffer_unordered(args.parallel)
        .collect::<Vec<_>>();

        let results = futures.await;
        for result in results {
            result?;
            completed += 1;
            println!("Progress: {completed}/{total} blobs downloaded");
        }
    }

    println!(
        "Pull complete: {} blobs available in {}",
        total,
        output_root.display()
    );
    Ok(())
}

async fn fetch_manifest(client: &reqwest::Client, model: &str) -> Result<Manifest> {
    let url = format!("{OLLAMA_REGISTRY}/v2/library/{model}/manifests/latest");

    client
        .get(url)
        .header(
            ACCEPT,
            "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json",
        )
        .send()
        .await
        .context("failed to request model manifest")?
        .error_for_status()
        .context("failed to fetch model manifest")?
        .json::<Manifest>()
        .await
        .context("failed to decode model manifest")
}

fn descriptors_from_manifest(manifest: Manifest) -> Vec<ManifestDescriptor> {
    let mut descriptors = Vec::with_capacity(manifest.layers.len() + 1);
    descriptors.push(manifest.config);
    descriptors.extend(manifest.layers);
    descriptors
}

async fn download_blob_with_retry(
    client: &reqwest::Client,
    model: &str,
    output_root: &Path,
    descriptor: &ManifestDescriptor,
    retries: u32,
) -> Result<()> {
    let blob_path = blob_path(output_root, model, &descriptor.digest);

    if tokio::fs::try_exists(&blob_path).await.unwrap_or(false) {
        println!("Skipping existing blob {}", descriptor.digest);
        return Ok(());
    }

    if let Some(parent) = blob_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    let mut attempt = 0u32;
    loop {
        attempt += 1;

        let result = download_blob(
            client,
            model,
            &descriptor.digest,
            descriptor.size,
            &blob_path,
        )
        .await;
        match result {
            Ok(_) => return Ok(()),
            Err(err) if attempt <= retries => {
                let backoff = Duration::from_millis(250 * u64::from(attempt));
                eprintln!(
                    "Blob {} failed (attempt {attempt}/{retries}), retrying in {:?}: {err}",
                    descriptor.digest, backoff
                );
                sleep(backoff).await;
            }
            Err(err) => return Err(err),
        }
    }
}

async fn download_blob(
    client: &reqwest::Client,
    model: &str,
    digest: &str,
    expected_size: u64,
    destination: &Path,
) -> Result<()> {
    let url = format!("{OLLAMA_REGISTRY}/v2/library/{model}/blobs/{digest}");
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("failed to request blob {digest}"))?
        .error_for_status()
        .with_context(|| format!("failed to download blob {digest}"))?;

    let mut file = tokio::fs::File::create(destination)
        .await
        .with_context(|| format!("failed to create {}", destination.display()))?;

    let mut stream = response.bytes_stream();
    let mut bytes_written: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("failed reading stream for {digest}"))?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("failed writing blob {digest}"))?;
        bytes_written += chunk.len() as u64;
    }

    file.flush().await.context("failed to flush blob file")?;

    if bytes_written != expected_size {
        bail!("blob {digest} size mismatch: expected {expected_size}, wrote {bytes_written}");
    }

    Ok(())
}

pub async fn run_agent(task: &str) -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<AgentEvent>(32);
    let task_name = task.to_owned();

    tokio::spawn(async move {
        let _ = tx.send(AgentEvent::Started(task_name)).await;

        let steps = [
            "Understand task context",
            "Build a short plan",
            "Apply code changes",
            "Run checks and summarize",
        ];

        for (index, step) in steps.into_iter().enumerate() {
            let _ = tx
                .send(AgentEvent::Step {
                    index: index + 1,
                    total: steps.len(),
                    message: step.to_string(),
                })
                .await;
            sleep(Duration::from_millis(250)).await;
        }

        let _ = tx.send(AgentEvent::Complete).await;
    });

    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::Started(task_name) => {
                println!("agent started: {task_name}");
            }
            AgentEvent::Step {
                index,
                total,
                message,
            } => {
                println!("[{index}/{total}] {message}");
            }
            AgentEvent::Complete => {
                println!("agent complete");
            }
        }
    }

    Ok(())
}

fn release_target() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match (os, arch) {
        ("macos", "aarch64") => "aarch64-apple-darwin".to_owned(),
        ("macos", "x86_64") => "x86_64-apple-darwin".to_owned(),
        ("linux", "aarch64") => "aarch64-unknown-linux-musl".to_owned(),
        ("linux", "x86_64") => "x86_64-unknown-linux-musl".to_owned(),
        ("windows", "x86_64") => "x86_64-pc-windows-msvc".to_owned(),
        _ => format!("{arch}-{os}"),
    }
}

fn default_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(32))
        .unwrap_or(8)
}

fn default_pull_dir() -> PathBuf {
    let mut root = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    root.push(".pb");
    root.push("models");
    root
}

fn blob_path(root: &Path, model: &str, digest: &str) -> PathBuf {
    let mut path = root.join(model);
    path.push(digest.replace(':', "_"));
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_include_config_then_layers() {
        let manifest = Manifest {
            config: ManifestDescriptor {
                digest: "sha256:config".to_string(),
                size: 1,
            },
            layers: vec![
                ManifestDescriptor {
                    digest: "sha256:layer1".to_string(),
                    size: 2,
                },
                ManifestDescriptor {
                    digest: "sha256:layer2".to_string(),
                    size: 3,
                },
            ],
        };

        let descriptors = descriptors_from_manifest(manifest);
        assert_eq!(descriptors.len(), 3);
        assert_eq!(descriptors[0].digest, "sha256:config");
        assert_eq!(descriptors[1].digest, "sha256:layer1");
        assert_eq!(descriptors[2].digest, "sha256:layer2");
    }

    #[test]
    fn blob_path_sanitizes_digest() {
        let path = blob_path(Path::new("/tmp/output"), "qwen3-coder-next", "sha256:abcd");
        assert!(path.ends_with("qwen3-coder-next/sha256_abcd"));
    }

    #[test]
    fn default_parallelism_is_non_zero() {
        assert!(default_parallelism() > 0);
    }

    #[test]
    fn release_target_is_not_empty() {
        assert!(!release_target().is_empty());
    }
}
