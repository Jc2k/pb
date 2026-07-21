use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::agent_context;
use crate::checks::{CheckEvidenceLedger, check_evidence_is_current, plan_checks_for_paths};
use crate::workspace::{
    ContentSnapshot, PathContent, RepositoryContext, WorkspaceGraph, WorkspaceGraphSource,
};

pub(crate) const MAX_REPOSITORY_BRIEF_CHARS: usize = 16_000;
pub(crate) const MAX_CHANGE_MANIFEST_CHARS: usize = 16_000;
const MAX_INSTRUCTION_EXCERPT_CHARS: usize = 1_000;
const MIN_INSTRUCTION_EXCERPT_CHARS: usize = 240;
const MAX_CHECK_EXCERPT_CHARS: usize = 480;
const MAX_CHECK_BRIEF_CHARS: usize = 8_000;
const MIN_INSPECTION_CHARS: usize = 512;
const MAX_INSPECTION_FILE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_INSPECTION_PROCESS_BYTES: usize = 8 * 1024 * 1024;
const MAX_INSTRUCTION_FILE_BYTES: u64 = 256 * 1024;

fn read_bounded_bytes(path: &Path, max_bytes: u64, label: &str) -> Result<Vec<u8>> {
    let file =
        File::open(path).with_context(|| format!("failed to open {label} {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to stat {label} {}", path.display()))?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        bail!(
            "{label} {} exceeds the {max_bytes}-byte input bound",
            path.display()
        );
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {label} {}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        bail!(
            "{label} {} grew beyond the {max_bytes}-byte input bound",
            path.display()
        );
    }
    Ok(bytes)
}

fn run_bounded_command(mut command: Command, label: &str) -> Result<std::process::Output> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to start {label}"))?;
    let mut stdout = child
        .stdout
        .take()
        .context("command stdout was unavailable")?;
    let mut stderr = child
        .stderr
        .take()
        .context("command stderr was unavailable")?;
    let stdout_thread = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .by_ref()
            .take((MAX_INSPECTION_PROCESS_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let stderr_thread = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr
            .by_ref()
            .take((MAX_INSPECTION_PROCESS_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let status = child
        .wait()
        .with_context(|| format!("failed to wait for {label}"))?;
    let stdout = stdout_thread
        .join()
        .map_err(|_| anyhow::anyhow!("{label} stdout reader panicked"))??;
    let stderr = stderr_thread
        .join()
        .map_err(|_| anyhow::anyhow!("{label} stderr reader panicked"))??;
    if stdout.len() > MAX_INSPECTION_PROCESS_BYTES || stderr.len() > MAX_INSPECTION_PROCESS_BYTES {
        bail!("{label} exceeded the {MAX_INSPECTION_PROCESS_BYTES}-byte output bound");
    }
    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct RepositoryBrief {
    schema_version: u32,
    full_graph_sha256: String,
    graph_source: WorkspaceGraphSource,
    focus_root: String,
    components: Vec<BriefComponent>,
    executors: Vec<BriefExecutor>,
    checks: Vec<BriefCheck>,
    tasks: Vec<BriefTask>,
    manifests: Vec<String>,
    entry_points: Vec<String>,
    project_instructions: Vec<BriefInstruction>,
    top_level_paths: Vec<String>,
    dirty_paths: Vec<String>,
    omitted: BriefOmissions,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct BriefComponent {
    id: String,
    root: String,
    executor: String,
    depends_on: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct BriefExecutor {
    id: String,
    kind: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct BriefCheck {
    id: String,
    label: String,
    executor: String,
    components: Vec<String>,
    depends_on: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct BriefTask {
    id: String,
    label: String,
    executor: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct BriefInstruction {
    path: String,
    sha256: String,
    excerpt: String,
    evidence_only: bool,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
struct BriefOmissions {
    components: usize,
    executors: usize,
    checks: usize,
    tasks: usize,
    manifests: usize,
    entry_points: usize,
    project_instructions: usize,
    top_level_paths: usize,
    dirty_paths: usize,
}

impl RepositoryBrief {
    pub(crate) fn build(
        graph: &WorkspaceGraph,
        repository: &RepositoryContext,
        workspace_root: &Path,
        full_graph_sha256: &str,
    ) -> Result<Self> {
        let focus_root = relative_focus_root(repository)?;
        let mut roots = graph
            .components
            .values()
            .map(|component| component.root.clone())
            .collect::<BTreeSet<_>>();
        roots.insert(".".to_string());
        roots.insert(focus_root.clone());
        let mut focus_ancestor = PathBuf::from(&focus_root);
        while let Some(parent) = focus_ancestor.parent() {
            if parent.as_os_str().is_empty() {
                break;
            }
            roots.insert(parent.to_string_lossy().replace('\\', "/"));
            focus_ancestor = parent.to_path_buf();
        }

        let dirty_paths = changed_paths_in_workspace(repository, workspace_root)?;
        let relevant_components = graph
            .components
            .values()
            .filter(|component| {
                path_within_root(&focus_root, &component.root)
                    || dirty_paths
                        .iter()
                        .any(|path| path_within_root(path, &component.root))
            })
            .map(|component| component.id.clone())
            .collect::<BTreeSet<_>>();
        let mut relevant_checks = plan_checks_for_paths(graph, dirty_paths.clone())?
            .checks
            .into_iter()
            .collect::<BTreeSet<_>>();
        relevant_checks.extend(
            graph
                .checks
                .values()
                .filter(|check| {
                    check
                        .components
                        .iter()
                        .any(|component| relevant_components.contains(component))
                })
                .map(|check| check.id.clone()),
        );
        let mut components = graph
            .components
            .values()
            .map(|component| BriefComponent {
                id: component.id.clone(),
                root: component.root.clone(),
                executor: component.executor.clone(),
                depends_on: component.depends_on.clone(),
            })
            .collect::<Vec<_>>();
        components.sort_by_key(|component| {
            (
                !relevant_components.contains(&component.id),
                component.id.clone(),
            )
        });
        let mut checks = graph
            .checks
            .values()
            .map(|check| BriefCheck {
                id: check.id.clone(),
                label: truncate_chars(&check.label, 240),
                executor: check.executor.clone(),
                components: check.components.clone(),
                depends_on: check.depends_on.clone(),
            })
            .collect::<Vec<_>>();
        checks.sort_by_key(|check| (!relevant_checks.contains(&check.id), check.id.clone()));
        let relevant_executors = components
            .iter()
            .filter(|component| relevant_components.contains(&component.id))
            .map(|component| component.executor.clone())
            .chain(
                checks
                    .iter()
                    .filter(|check| relevant_checks.contains(&check.id))
                    .map(|check| check.executor.clone()),
            )
            .collect::<BTreeSet<_>>();
        let mut executors = graph
            .executors
            .values()
            .map(|executor| BriefExecutor {
                id: executor.id.clone(),
                kind: format!("{:?}", executor.kind).to_ascii_lowercase(),
            })
            .collect::<Vec<_>>();
        executors.sort_by_key(|executor| {
            (
                !relevant_executors.contains(&executor.id),
                executor.id.clone(),
            )
        });

        let mut brief = Self {
            schema_version: 1,
            full_graph_sha256: full_graph_sha256.to_string(),
            graph_source: graph.source,
            focus_root,
            components,
            executors,
            checks,
            tasks: graph
                .tasks
                .values()
                .map(|task| BriefTask {
                    id: task.id.clone(),
                    label: truncate_chars(&task.label, 240),
                    executor: task.executor.clone(),
                })
                .collect(),
            manifests: discover_named_paths(workspace_root, &roots, MANIFEST_NAMES)?,
            entry_points: discover_named_paths(workspace_root, &roots, ENTRY_POINT_NAMES)?,
            project_instructions: discover_project_instructions(workspace_root, &roots)?,
            top_level_paths: top_level_paths(workspace_root)?,
            dirty_paths,
            omitted: BriefOmissions::default(),
        };
        brief.fit_to_bound()?;
        Ok(brief)
    }

    pub(crate) fn to_pretty_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("failed to serialize repository brief")
    }

    fn fit_to_bound(&mut self) -> Result<()> {
        while self.serialized_chars()? > MAX_REPOSITORY_BRIEF_CHARS {
            if let Some(instruction) =
                self.project_instructions
                    .iter_mut()
                    .rev()
                    .find(|instruction| {
                        instruction.excerpt.chars().count() > MIN_INSTRUCTION_EXCERPT_CHARS
                    })
            {
                instruction.excerpt = truncate_chars(
                    &instruction.excerpt,
                    instruction
                        .excerpt
                        .chars()
                        .count()
                        .saturating_div(2)
                        .max(MIN_INSTRUCTION_EXCERPT_CHARS),
                );
                continue;
            }
            // Trim breadth evenly so a large early category cannot consume the
            // entire budget and erase the most relevant item from later ones.
            let mut trimmed_round = false;
            trimmed_round |= pop_counted_above(
                &mut self.top_level_paths,
                &mut self.omitted.top_level_paths,
                1,
            );
            trimmed_round |= pop_counted_above(&mut self.tasks, &mut self.omitted.tasks, 1);
            trimmed_round |= pop_counted_above(&mut self.checks, &mut self.omitted.checks, 1);
            trimmed_round |=
                pop_counted_above(&mut self.components, &mut self.omitted.components, 1);
            trimmed_round |= pop_counted_above(&mut self.executors, &mut self.omitted.executors, 1);
            trimmed_round |=
                pop_counted_above(&mut self.entry_points, &mut self.omitted.entry_points, 1);
            trimmed_round |= pop_counted_above(
                &mut self.project_instructions,
                &mut self.omitted.project_instructions,
                1,
            );
            trimmed_round |= pop_counted_above(&mut self.manifests, &mut self.omitted.manifests, 1);
            trimmed_round |=
                pop_counted_above(&mut self.dirty_paths, &mut self.omitted.dirty_paths, 1);
            if trimmed_round {
                continue;
            }

            // Extremely large individual graph records or paths may make the
            // representative set itself too large. Preserve a valid bounded
            // brief by dropping the least essential remaining categories in a
            // deterministic order; the omission counters keep that explicit.
            if pop_counted(&mut self.top_level_paths, &mut self.omitted.top_level_paths)
                || pop_counted(&mut self.tasks, &mut self.omitted.tasks)
                || pop_counted(&mut self.entry_points, &mut self.omitted.entry_points)
                || pop_counted(&mut self.dirty_paths, &mut self.omitted.dirty_paths)
                || pop_counted(
                    &mut self.project_instructions,
                    &mut self.omitted.project_instructions,
                )
                || pop_counted(&mut self.manifests, &mut self.omitted.manifests)
                || pop_counted(&mut self.executors, &mut self.omitted.executors)
                || pop_counted(&mut self.checks, &mut self.omitted.checks)
                || pop_counted(&mut self.components, &mut self.omitted.components)
            {
                continue;
            }
            bail!(
                "repository brief anchors exceed the {MAX_REPOSITORY_BRIEF_CHARS}-character bound"
            );
        }
        Ok(())
    }

    fn serialized_chars(&self) -> Result<usize> {
        Ok(serde_json::to_string_pretty(self)
            .context("failed to measure repository brief")?
            .chars()
            .count())
    }
}

fn path_within_root(path: &str, root: &str) -> bool {
    path == "."
        || root == "."
        || path == root
        || path
            .strip_prefix(root)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn pop_counted<T>(values: &mut Vec<T>, omitted: &mut usize) -> bool {
    if values.pop().is_some() {
        *omitted = omitted.saturating_add(1);
        true
    } else {
        false
    }
}

fn pop_counted_above<T>(values: &mut Vec<T>, omitted: &mut usize, minimum: usize) -> bool {
    if values.len() <= minimum {
        return false;
    }
    pop_counted(values, omitted)
}

fn relative_focus_root(repository: &RepositoryContext) -> Result<String> {
    let relative = repository
        .focus_root
        .strip_prefix(&repository.repo_root)
        .context("repository focus root is outside its repository root")?;
    if relative.as_os_str().is_empty() {
        Ok(".".to_string())
    } else {
        Ok(relative.to_string_lossy().replace('\\', "/"))
    }
}

const MANIFEST_NAMES: &[&str] = &[
    "Cargo.toml",
    "package.json",
    "deno.json",
    "deno.jsonc",
    "pyproject.toml",
    "go.mod",
    "Gemfile",
    "pom.xml",
    "build.gradle",
    "Makefile",
];

const ENTRY_POINT_NAMES: &[&str] = &[
    "src/main.rs",
    "src/lib.rs",
    "src/index.ts",
    "src/index.tsx",
    "src/main.ts",
    "src/main.tsx",
    "main.py",
    "app.py",
    "main.go",
    "index.js",
];

fn discover_named_paths(
    workspace_root: &Path,
    roots: &BTreeSet<String>,
    names: &[&str],
) -> Result<Vec<String>> {
    let mut found = BTreeSet::new();
    for root in roots {
        for name in names {
            let relative = if root == "." {
                (*name).to_string()
            } else {
                format!("{root}/{name}")
            };
            let path = workspace_root.join(&relative);
            if path.is_file() {
                found.insert(relative);
            }
        }
    }
    Ok(found.into_iter().collect())
}

fn discover_project_instructions(
    workspace_root: &Path,
    roots: &BTreeSet<String>,
) -> Result<Vec<BriefInstruction>> {
    let mut candidates = BTreeSet::from([
        "AGENTS.md".to_string(),
        "CLAUDE.md".to_string(),
        ".github/copilot-instructions.md".to_string(),
    ]);
    for root in roots {
        if root != "." {
            candidates.insert(format!("{root}/AGENTS.md"));
            candidates.insert(format!("{root}/CLAUDE.md"));
        }
    }
    let mut instructions = Vec::new();
    for relative in candidates {
        let path = workspace_root.join(&relative);
        if !path.is_file() {
            continue;
        }
        let bytes = read_bounded_bytes(&path, MAX_INSTRUCTION_FILE_BYTES, "project instructions")
            .with_context(|| format!("failed to read project instructions {relative}"))?;
        if bytes.contains(&0) {
            continue;
        }
        let text = String::from_utf8_lossy(&bytes);
        instructions.push(BriefInstruction {
            path: relative,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            excerpt: truncate_chars(&text, MAX_INSTRUCTION_EXCERPT_CHARS),
            evidence_only: true,
        });
    }
    Ok(instructions)
}

fn top_level_paths(workspace_root: &Path) -> Result<Vec<String>> {
    let mut paths = std::fs::read_dir(workspace_root)
        .with_context(|| format!("failed to list workspace root {}", workspace_root.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
        .filter(|name| name != ".git" && name != ".pb")
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

pub(crate) fn changed_paths_in_workspace(
    repository: &RepositoryContext,
    workspace_root: &Path,
) -> Result<Vec<String>> {
    let current = ContentSnapshot::capture(workspace_root)?;
    Ok(repository.task_baseline.content.changed_paths(&current))
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ChangeStatus {
    Added,
    Modified,
    Deleted,
    Renamed,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReviewContentKind {
    Text,
    Binary,
    Symlink,
    Directory,
    Other,
    Missing,
    Unknown,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ChangeManifestEntry {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_path: Option<String>,
    pub status: ChangeStatus,
    pub content_kind: ReviewContentKind,
    pub inspection: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ChangeManifest {
    schema_version: u32,
    checked_content_fingerprint: String,
    entries: Vec<ChangeManifestEntry>,
}

impl ChangeManifest {
    pub(crate) fn build(
        repository: &RepositoryContext,
        workspace_root: &Path,
        checked_content_fingerprint: &str,
    ) -> Result<Self> {
        let current = ContentSnapshot::capture(workspace_root)?;
        if current.fingerprint != checked_content_fingerprint {
            bail!(
                "change manifest fingerprint {} differs from checked content {}",
                current.fingerprint,
                checked_content_fingerprint
            );
        }
        let before = &repository.task_baseline.content;
        let changed = before.changed_paths(&current);
        let mut deleted = BTreeMap::<(String, String), Vec<String>>::new();
        let mut added = BTreeMap::<(String, String), Vec<String>>::new();
        for path in &changed {
            match (
                present_content(before.paths.get(path)),
                present_content(current.paths.get(path)),
            ) {
                (Some(content), None) => deleted
                    .entry((content.kind.clone(), content.fingerprint.clone()))
                    .or_default()
                    .push(path.clone()),
                (None, Some(content)) => added
                    .entry((content.kind.clone(), content.fingerprint.clone()))
                    .or_default()
                    .push(path.clone()),
                _ => {}
            }
        }
        let mut renamed_old = HashSet::new();
        let mut renamed_new = BTreeMap::new();
        for (fingerprint, old_paths) in &deleted {
            let Some(new_paths) = added.get(fingerprint) else {
                continue;
            };
            if old_paths.len() == 1 && new_paths.len() == 1 {
                renamed_old.insert(old_paths[0].clone());
                renamed_new.insert(new_paths[0].clone(), old_paths[0].clone());
            }
        }

        let mut entries = Vec::new();
        for path in changed {
            if renamed_old.contains(&path) {
                continue;
            }
            let previous_path = renamed_new.get(&path).cloned();
            let status = if previous_path.is_some() {
                ChangeStatus::Renamed
            } else {
                match (
                    present_content(before.paths.get(&path)),
                    present_content(current.paths.get(&path)),
                ) {
                    (None, Some(_)) => ChangeStatus::Added,
                    (Some(_), None) => ChangeStatus::Deleted,
                    (Some(_), Some(_)) => ChangeStatus::Modified,
                    (None, None) => continue,
                }
            };
            let content_kind = content_kind(
                present_content(current.paths.get(&path))
                    .or_else(|| present_content(before.paths.get(&path))),
                workspace_root.join(&path),
                status,
            )?;
            let current_reviewable = status != ChangeStatus::Deleted
                && reviewable_text_file(&workspace_root.join(&path))?;
            let inspection = if current_reviewable {
                "inspect_change_required"
            } else {
                match (status, content_kind) {
                    (ChangeStatus::Deleted, _) => "diff_only",
                    (_, ReviewContentKind::Binary) => "metadata_and_checks",
                    _ => "metadata",
                }
            };
            entries.push(ChangeManifestEntry {
                path,
                previous_path,
                status,
                content_kind,
                inspection,
            });
        }
        entries.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(Self {
            schema_version: 1,
            checked_content_fingerprint: checked_content_fingerprint.to_string(),
            entries,
        })
    }

    pub(crate) fn to_prompt_json(&self) -> Result<String> {
        let pretty = serde_json::to_string_pretty(self)
            .context("failed to serialize changed-path manifest")?;
        if pretty.chars().count() <= MAX_CHANGE_MANIFEST_CHARS {
            return Ok(pretty);
        }
        let compact =
            serde_json::to_string(self).context("failed to compact changed-path manifest")?;
        if compact.chars().count() <= MAX_CHANGE_MANIFEST_CHARS {
            return Ok(compact);
        }
        bail!(
            "changed-path manifest exceeds the {MAX_CHANGE_MANIFEST_CHARS}-character review bound"
        )
    }

    #[cfg(test)]
    pub(crate) fn entries(&self) -> &[ChangeManifestEntry] {
        &self.entries
    }

    pub(crate) fn entry_for_path(&self, path: &str) -> Option<&ChangeManifestEntry> {
        self.entries.iter().find(|entry| {
            entry.path == path
                || entry
                    .previous_path
                    .as_deref()
                    .is_some_and(|old| old == path)
        })
    }
}

fn present_content(content: Option<&PathContent>) -> Option<&PathContent> {
    content.filter(|content| content.kind != "missing")
}

fn content_kind(
    content: Option<&PathContent>,
    absolute: PathBuf,
    status: ChangeStatus,
) -> Result<ReviewContentKind> {
    let Some(content) = content else {
        return Ok(ReviewContentKind::Missing);
    };
    match content.kind.as_str() {
        "file" if status == ChangeStatus::Deleted => Ok(ReviewContentKind::Unknown),
        "file" => {
            let bytes = read_bounded_bytes(&absolute, MAX_INSPECTION_FILE_BYTES, "review path")?;
            Ok(if bytes.contains(&0) {
                ReviewContentKind::Binary
            } else {
                ReviewContentKind::Text
            })
        }
        "symlink" => Ok(ReviewContentKind::Symlink),
        "directory" => Ok(ReviewContentKind::Directory),
        "missing" => Ok(ReviewContentKind::Missing),
        _ => Ok(ReviewContentKind::Other),
    }
}

fn reviewable_text_file(path: &Path) -> Result<bool> {
    if !path.is_file() {
        return Ok(false);
    }
    let bytes = read_bounded_bytes(path, MAX_INSPECTION_FILE_BYTES, "review path")?;
    Ok(!bytes.contains(&0))
}

#[derive(Debug, Clone, Serialize)]
struct CheckEvidenceBrief {
    check_id: String,
    current: bool,
    success: bool,
    timed_out: bool,
    duration_ms: u64,
    executor: String,
    output_sha256: String,
    output_excerpt: String,
}

#[derive(Debug, Clone, Serialize)]
struct CheckEvidenceBriefSet {
    details: Vec<CheckEvidenceBrief>,
    omitted_detail_ids: Vec<String>,
}

pub(crate) fn selected_check_evidence_brief(
    graph: &WorkspaceGraph,
    ledger: &CheckEvidenceLedger,
    selected_checks: &[String],
    workspace_root: &Path,
) -> Result<String> {
    let mut briefs = Vec::new();
    for check_id in selected_checks {
        let Some(evidence) = ledger.get(check_id) else {
            continue;
        };
        briefs.push(CheckEvidenceBrief {
            check_id: check_id.clone(),
            current: check_evidence_is_current(workspace_root, graph, ledger, check_id)?,
            success: evidence.success,
            timed_out: evidence.timed_out,
            duration_ms: evidence.duration_ms,
            executor: evidence.executor.clone(),
            output_sha256: format!("{:x}", Sha256::digest(evidence.output.as_bytes())),
            output_excerpt: truncate_chars(&evidence.output, MAX_CHECK_EXCERPT_CHARS),
        });
    }
    let mut set = CheckEvidenceBriefSet {
        details: briefs,
        omitted_detail_ids: Vec::new(),
    };
    loop {
        let rendered = serde_json::to_string_pretty(&set)
            .context("failed to serialize selected check evidence brief")?;
        if rendered.chars().count() <= MAX_CHECK_BRIEF_CHARS {
            return Ok(rendered);
        }
        let Some(omitted) = set.details.pop() else {
            return Ok(agent_context::bound_tool_result_for_prompt(
                &rendered,
                MAX_CHECK_BRIEF_CHARS,
            )
            .content);
        };
        set.omitted_detail_ids.insert(0, omitted.check_id);
    }
}

pub(crate) fn inspect_change(
    path: &str,
    repository: &RepositoryContext,
    graph: &WorkspaceGraph,
    ledger: &CheckEvidenceLedger,
    workspace_root: &Path,
    max_chars: usize,
) -> Result<(String, Option<String>)> {
    if max_chars < MIN_INSPECTION_CHARS {
        bail!(
            "focused inspection budget {max_chars} is too small; at least {MIN_INSPECTION_CHARS} characters are required"
        );
    }
    let current = ContentSnapshot::capture(workspace_root)?;
    let manifest = ChangeManifest::build(repository, workspace_root, &current.fingerprint)?;
    let entry = manifest
        .entry_for_path(path)
        .with_context(|| format!("inspect_change path '{path}' is not in the task delta"))?;
    let inspected_path = entry.path.clone();
    let current_path = workspace_root.join(&inspected_path);
    let (current_bytes, current_lines, current_sha256, reviewable_text) = if current_path.is_file()
    {
        let bytes = read_bounded_bytes(
            &current_path,
            MAX_INSPECTION_FILE_BYTES,
            "changed review path",
        )
        .with_context(|| format!("failed to read changed path {inspected_path}"))?;
        let lines = (!bytes.contains(&0)).then(|| String::from_utf8_lossy(&bytes).lines().count());
        let sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
        (Some(bytes.len()), lines, sha256, !bytes.contains(&0))
    } else {
        (None, None, None, false)
    };

    let header = format!(
        "inspect_change v1\npath={inspected_path}\nprevious_path={}\nstatus={}\ncontent_kind={}\nchecked_content_fingerprint={}\nHarness current content fingerprint: {}\ncurrent_bytes={}\ncurrent_lines={}\ncurrent_sha256={}\n",
        entry.previous_path.as_deref().unwrap_or("none"),
        change_status_label(entry.status),
        content_kind_label(entry.content_kind),
        current.fingerprint,
        current.fingerprint,
        current_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
        current_lines.map_or_else(|| "none".to_string(), |value| value.to_string()),
        current_sha256.as_deref().unwrap_or("none"),
    );
    let section_overhead = "\nCURRENT_CONTEXT\n\n\nDIFF_HUNKS\n\n\nRELEVANT_CHECK_EVIDENCE\n"
        .chars()
        .count();
    let payload_budget = max_chars
        .saturating_sub(header.chars().count())
        .saturating_sub(section_overhead);
    let context_budget = payload_budget.saturating_mul(2) / 5;
    let diff_budget = payload_budget.saturating_mul(2) / 5;
    let checks_budget = payload_budget
        .saturating_sub(context_budget)
        .saturating_sub(diff_budget);

    let diff = render_path_diff(workspace_root, entry)?;
    let ranges = parse_new_hunk_ranges(&diff);
    let context = if reviewable_text {
        let bytes = read_bounded_bytes(
            &current_path,
            MAX_INSPECTION_FILE_BYTES,
            "changed review text",
        )
        .with_context(|| format!("failed to read changed text path {inspected_path}"))?;
        let text = String::from_utf8_lossy(&bytes);
        render_hunk_context(&text, &ranges, context_budget)
    } else {
        format!(
            "[current text context unavailable for {:?}]",
            entry.content_kind
        )
    };
    let bounded_diff = agent_context::bound_tool_result_for_prompt(&diff, diff_budget.max(512));
    let check_ids = plan_checks_for_paths(graph, vec![inspected_path.clone()])?.checks;
    let checks = selected_check_evidence_brief(graph, ledger, &check_ids, workspace_root)?;
    let bounded_checks =
        agent_context::bound_tool_result_for_prompt(&checks, checks_budget.max(512));
    let rendered = format!(
        "{header}\nCURRENT_CONTEXT\n{context}\n\nDIFF_HUNKS\n{}\n\nRELEVANT_CHECK_EVIDENCE\n{}",
        bounded_diff.content, bounded_checks.content
    );
    let bounded = agent_context::bound_tool_result_for_prompt(&rendered, max_chars);
    let bounded = bound_utf8_prefix_bytes(&bounded.content, max_chars);
    let earned_read = reviewable_text.then_some(inspected_path);
    Ok((bounded, earned_read))
}

fn bound_utf8_prefix_bytes(input: &str, max_bytes: usize) -> String {
    if input.len() <= max_bytes {
        return input.to_string();
    }
    let marker = format!("\n[inspection omitted to fit {max_bytes}-byte bound]");
    let prefix_bytes = max_bytes.saturating_sub(marker.len());
    let mut end = prefix_bytes.min(input.len());
    while end > 0 && !input.is_char_boundary(end) {
        end -= 1;
    }
    let mut output = input[..end].to_string();
    output.push_str(&marker);
    output
}

const fn change_status_label(status: ChangeStatus) -> &'static str {
    match status {
        ChangeStatus::Added => "added",
        ChangeStatus::Modified => "modified",
        ChangeStatus::Deleted => "deleted",
        ChangeStatus::Renamed => "renamed",
    }
}

const fn content_kind_label(kind: ReviewContentKind) -> &'static str {
    match kind {
        ReviewContentKind::Text => "text",
        ReviewContentKind::Binary => "binary",
        ReviewContentKind::Symlink => "symlink",
        ReviewContentKind::Directory => "directory",
        ReviewContentKind::Other => "other",
        ReviewContentKind::Missing => "missing",
        ReviewContentKind::Unknown => "unknown",
    }
}

fn render_path_diff(workspace_root: &Path, entry: &ChangeManifestEntry) -> Result<String> {
    let mut command = Command::new("git");
    command
        .args([
            "diff",
            "--no-ext-diff",
            "--find-renames",
            "--unified=3",
            "--",
        ])
        .current_dir(workspace_root);
    if let Some(previous) = &entry.previous_path {
        command.arg(previous);
    }
    command.arg(&entry.path);
    let output = run_bounded_command(command, "focused change diff")?;
    if !output.status.success() {
        bail!(
            "failed to render focused change diff for '{}': {}",
            entry.path,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let diff = String::from_utf8_lossy(&output.stdout).to_string();
    if !diff.is_empty() {
        return Ok(diff);
    }
    Ok(match entry.status {
        ChangeStatus::Added => {
            "[new path; no Git-index diff is available, inspect bounded current context below]"
                .to_string()
        }
        ChangeStatus::Deleted => "[deleted path; no textual Git diff is available]".to_string(),
        ChangeStatus::Renamed => format!(
            "[renamed without a textual hunk: {} -> {}]",
            entry.previous_path.as_deref().unwrap_or("<unknown>"),
            entry.path
        ),
        ChangeStatus::Modified => {
            "[content differs from the task baseline; no Git-index hunk is available]".to_string()
        }
    })
}

fn parse_new_hunk_ranges(diff: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    for line in diff.lines().filter(|line| line.starts_with("@@ ")) {
        let Some(plus) = line.split_whitespace().find(|part| part.starts_with('+')) else {
            continue;
        };
        let mut values = plus.trim_start_matches('+').split(',');
        let Some(start) = values.next().and_then(|value| value.parse::<usize>().ok()) else {
            continue;
        };
        let count = values
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        if count > 0 {
            ranges.push((
                start.saturating_sub(3).max(1),
                start.saturating_add(count + 2),
            ));
        }
    }
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for range in ranges {
        if let Some(last) = merged.last_mut()
            && range.0 <= last.1.saturating_add(1)
        {
            last.1 = last.1.max(range.1);
        } else {
            merged.push(range);
        }
    }
    merged
}

fn render_hunk_context(text: &str, ranges: &[(usize, usize)], max_chars: usize) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    let ranges = if ranges.is_empty() {
        vec![(1, lines.len())]
    } else {
        ranges.to_vec()
    };
    let mut output = String::new();
    let mut emitted = BTreeSet::new();
    for (start, end) in ranges {
        for line_number in start..=end.min(lines.len()) {
            if !emitted.insert(line_number) {
                continue;
            }
            let rendered = format!("{line_number}: {}\n", lines[line_number - 1]);
            if output
                .chars()
                .count()
                .saturating_add(rendered.chars().count())
                > max_chars
            {
                output.push_str(&format!(
                    "[current context bounded; next_unshown_line={line_number}; total_lines={}]",
                    lines.len()
                ));
                return output;
            }
            output.push_str(&rendered);
        }
    }
    if output.is_empty() {
        "[no current text lines]".to_string()
    } else {
        output
    }
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        value.to_string()
    } else {
        format!(
            "{}…",
            value
                .chars()
                .take(max_chars.saturating_sub(1))
                .collect::<String>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::{
        CheckTrigger, Executor, ExecutorKind, WORKSPACE_CONFIG_VERSION, WorkspaceCheck,
        WorkspaceComponent, WorkspaceTask,
    };

    fn init_repo() -> tempfile::TempDir {
        let temp = tempfile::tempdir().unwrap();
        Command::new("git")
            .args(["init", "-q"])
            .current_dir(temp.path())
            .status()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "tests@example.invalid"])
            .current_dir(temp.path())
            .status()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "pb tests"])
            .current_dir(temp.path())
            .status()
            .unwrap();
        std::fs::write(temp.path().join("seed.txt"), "seed\n").unwrap();
        Command::new("git")
            .args(["add", "seed.txt"])
            .current_dir(temp.path())
            .status()
            .unwrap();
        Command::new("git")
            .args(["commit", "-q", "-m", "test: seed"])
            .current_dir(temp.path())
            .status()
            .unwrap();
        temp
    }

    #[test]
    fn rb1_large_polyglot_repository_brief_is_stable_and_bounded() {
        let repo = init_repo();
        std::fs::write(repo.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        std::fs::write(repo.path().join("package.json"), "{}\n").unwrap();
        std::fs::write(repo.path().join("AGENTS.md"), "instruction ".repeat(500)).unwrap();
        std::fs::create_dir(repo.path().join(".pb")).unwrap();
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        let executor = Executor {
            id: "local".to_string(),
            kind: ExecutorKind::Local,
            environment: None,
        };
        let mut graph = WorkspaceGraph {
            version: WORKSPACE_CONFIG_VERSION,
            executors: BTreeMap::from([(executor.id.clone(), executor)]),
            components: BTreeMap::new(),
            checks: BTreeMap::new(),
            tasks: BTreeMap::new(),
            cargo_workspaces: BTreeMap::new(),
            discovery_warnings: Vec::new(),
            source: WorkspaceGraphSource::Discovered,
        };
        for index in 0..180 {
            let id = format!("component-{index:03}");
            graph.components.insert(
                id.clone(),
                WorkspaceComponent {
                    id: id.clone(),
                    root: ".".to_string(),
                    include: vec!["**".to_string()],
                    exclude: Vec::new(),
                    executor: "local".to_string(),
                    depends_on: Vec::new(),
                },
            );
            let check_id = format!("check-{index:03}");
            graph.checks.insert(
                check_id.clone(),
                WorkspaceCheck {
                    id: check_id,
                    label: format!("check component {index}"),
                    command: "true".to_string(),
                    cwd: ".".to_string(),
                    executor: "local".to_string(),
                    components: vec![id],
                    trigger: CheckTrigger::Changed,
                    inputs: vec!["**".to_string()],
                    outputs: Vec::new(),
                    depends_on: Vec::new(),
                    timeout_seconds: 60,
                },
            );
            let task_id = format!("task-{index:03}");
            graph.tasks.insert(
                task_id.clone(),
                WorkspaceTask {
                    id: task_id,
                    label: format!("task {index}"),
                    command: "true".to_string(),
                    cwd: ".".to_string(),
                    executor: "local".to_string(),
                    allowed_changes: Vec::new(),
                    timeout_seconds: 60,
                },
            );
        }
        let first = RepositoryBrief::build(&graph, &repository, repo.path(), "graph-hash")
            .unwrap()
            .to_pretty_json()
            .unwrap();
        let second = RepositoryBrief::build(&graph, &repository, repo.path(), "graph-hash")
            .unwrap()
            .to_pretty_json()
            .unwrap();
        assert_eq!(first, second);
        assert!(first.chars().count() <= MAX_REPOSITORY_BRIEF_CHARS);
        assert!(first.contains("graph-hash"));
        assert!(first.contains("Cargo.toml"));
        assert!(first.contains("project_instructions"));
        assert!(first.contains("\"checks\""));
        assert!(first.contains("component-000"));
        assert!(first.contains("check-000"));
        assert!(!first.contains("\"components\": 0"));
        assert!(!first.contains("\".pb\""));
    }

    #[test]
    fn rv2_manifest_distinguishes_new_deleted_renamed_and_binary_paths() {
        let repo = init_repo();
        std::fs::write(repo.path().join("delete.txt"), "delete\n").unwrap();
        std::fs::write(repo.path().join("rename.txt"), "rename\n").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(repo.path())
            .status()
            .unwrap();
        Command::new("git")
            .args(["commit", "-q", "-m", "test: fixtures"])
            .current_dir(repo.path())
            .status()
            .unwrap();
        let repository = RepositoryContext::capture(repo.path(), repo.path()).unwrap();
        std::fs::remove_file(repo.path().join("delete.txt")).unwrap();
        std::fs::rename(
            repo.path().join("rename.txt"),
            repo.path().join("renamed.txt"),
        )
        .unwrap();
        std::fs::write(repo.path().join("new.txt"), "new\n").unwrap();
        std::fs::write(repo.path().join("image.bin"), [0, 1, 2, 3]).unwrap();
        let fingerprint = ContentSnapshot::capture(repo.path()).unwrap().fingerprint;
        let manifest = ChangeManifest::build(&repository, repo.path(), &fingerprint).unwrap();
        assert!(
            manifest.entries().iter().any(|entry| {
                entry.path == "delete.txt" && entry.status == ChangeStatus::Deleted
            })
        );
        assert!(manifest.entries().iter().any(|entry| {
            entry.path == "renamed.txt"
                && entry.previous_path.as_deref() == Some("rename.txt")
                && entry.status == ChangeStatus::Renamed
        }));
        assert!(manifest.entries().iter().any(|entry| {
            entry.path == "new.txt"
                && entry.status == ChangeStatus::Added
                && entry.content_kind == ReviewContentKind::Text
        }));
        assert!(manifest.entries().iter().any(|entry| {
            entry.path == "image.bin" && entry.content_kind == ReviewContentKind::Binary
        }));
        let graph = WorkspaceGraph::legacy(&[]);
        let ledger = CheckEvidenceLedger::default();
        let (deleted, deleted_read) = inspect_change(
            "delete.txt",
            &repository,
            &graph,
            &ledger,
            repo.path(),
            8_000,
        )
        .unwrap();
        assert!(deleted.contains("status=deleted"));
        assert!(deleted.contains("CURRENT_CONTEXT"));
        assert!(deleted_read.is_none());
        let (new_file, new_read) =
            inspect_change("new.txt", &repository, &graph, &ledger, repo.path(), 8_000).unwrap();
        assert!(new_file.contains("status=added"));
        assert!(new_file.contains("1: new"));
        assert_eq!(new_read.as_deref(), Some("new.txt"));
        let (renamed, renamed_read) = inspect_change(
            "renamed.txt",
            &repository,
            &graph,
            &ledger,
            repo.path(),
            8_000,
        )
        .unwrap();
        assert!(renamed.contains("previous_path=rename.txt"));
        assert_eq!(renamed_read.as_deref(), Some("renamed.txt"));
        let (binary, binary_read) = inspect_change(
            "image.bin",
            &repository,
            &graph,
            &ledger,
            repo.path(),
            8_000,
        )
        .unwrap();
        assert!(binary.contains("content_kind=binary"));
        assert!(!binary.contains('\0'));
        assert!(binary_read.is_none());
        assert!(
            inspect_change(
                "new.txt",
                &repository,
                &graph,
                &ledger,
                repo.path(),
                MIN_INSPECTION_CHARS - 1,
            )
            .is_err()
        );
    }

    #[test]
    fn change_manifest_prompt_representation_is_strictly_bounded() {
        let entries = (0..40)
            .map(|index| ChangeManifestEntry {
                path: format!("{}{index}.rs", "nested/".repeat(35)),
                previous_path: None,
                status: ChangeStatus::Modified,
                content_kind: ReviewContentKind::Text,
                inspection: "inspect_change_required",
            })
            .collect();
        let manifest = ChangeManifest {
            schema_version: 1,
            checked_content_fingerprint: "f".repeat(64),
            entries,
        };
        let rendered = manifest.to_prompt_json().unwrap();
        assert!(rendered.chars().count() <= MAX_CHANGE_MANIFEST_CHARS);

        let mut oversized = manifest;
        for entry in &mut oversized.entries {
            entry.path = format!("{}oversized.rs", "path-segment/".repeat(80));
        }
        assert!(oversized.to_prompt_json().is_err());

        let utf8 = bound_utf8_prefix_bytes(&"🙂".repeat(1_000), MIN_INSPECTION_CHARS);
        assert!(utf8.len() <= MIN_INSPECTION_CHARS);
        assert!(utf8.contains("inspection omitted"));
    }
}
