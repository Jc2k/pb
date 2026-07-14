use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use globset::{Glob, GlobSet, GlobSetBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use crate::agent_core::{CheckCommandOutput, CommandBackend, EventSink};
use crate::environment::{EnvironmentBackend, EnvironmentConfig};
use crate::events::AgentEvent;
use crate::session_store::now_millis;
use crate::workspace::{
    CheckTrigger, ContentSnapshot, ExecutorKind, RepositoryContext, WorkspaceCheck, WorkspaceGraph,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSource {
    AgentTool,
    ExactGuardCommand,
    CommitTool,
    Handoff,
}

impl EvidenceSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AgentTool => "agent_tool",
            Self::ExactGuardCommand => "exact_guard_command",
            Self::CommitTool => "commit_tool",
            Self::Handoff => "handoff",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "agent_tool" => Some(Self::AgentTool),
            "exact_guard_command" => Some(Self::ExactGuardCommand),
            "commit_tool" => Some(Self::CommitTool),
            "handoff" => Some(Self::Handoff),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckEvidence {
    pub check_id: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub cwd: String,
    pub command_fingerprint: String,
    pub input_fingerprint: String,
    #[serde(default)]
    pub dependency_outputs: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_fingerprint: Option<String>,
    pub exit_status: i32,
    pub success: bool,
    pub timed_out: bool,
    pub duration_ms: u64,
    pub executor: String,
    pub source: EvidenceSource,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckEvidenceLedger {
    #[serde(default)]
    evidence: BTreeMap<String, CheckEvidence>,
}

impl CheckEvidenceLedger {
    pub fn record(&mut self, evidence: CheckEvidence) {
        self.evidence.insert(evidence.check_id.clone(), evidence);
    }

    pub fn get(&self, check_id: &str) -> Option<&CheckEvidence> {
        self.evidence.get(check_id)
    }

    pub fn from_events(events: &[AgentEvent]) -> Self {
        let mut ledger = Self::default();
        for event in events {
            let AgentEvent::CheckResult {
                check_id,
                exit_status,
                success,
                timed_out,
                duration_ms,
                fingerprint,
                command,
                cwd,
                executor: Some(executor),
                source: Some(source),
                command_fingerprint: Some(command_fingerprint),
                dependency_outputs,
                output_fingerprint,
                skip_reason: None,
                ..
            } = event
            else {
                continue;
            };
            let Some(source) = EvidenceSource::parse(source) else {
                continue;
            };
            ledger.record(CheckEvidence {
                check_id: check_id.clone(),
                command: command.clone().unwrap_or_default(),
                cwd: cwd.clone().unwrap_or_default(),
                command_fingerprint: command_fingerprint.clone(),
                input_fingerprint: fingerprint.clone(),
                dependency_outputs: dependency_outputs.clone(),
                output_fingerprint: output_fingerprint.clone(),
                exit_status: *exit_status,
                success: *success,
                timed_out: *timed_out,
                duration_ms: *duration_ms,
                executor: executor.clone(),
                source,
            });
        }
        ledger
    }

    fn current(
        &self,
        check: &WorkspaceCheck,
        input_fingerprint: &str,
        dependency_outputs: &BTreeMap<String, String>,
        current_output: &Option<String>,
    ) -> Option<&CheckEvidence> {
        let evidence = self.get(&check.id)?;
        (evidence.success
            && !evidence.timed_out
            && evidence.command_fingerprint == command_fingerprint(check)
            && evidence.input_fingerprint == input_fingerprint
            && &evidence.dependency_outputs == dependency_outputs
            && &evidence.output_fingerprint == current_output)
            .then_some(evidence)
    }
}

pub fn check_evidence_is_current(
    repo_root: &Path,
    graph: &WorkspaceGraph,
    ledger: &CheckEvidenceLedger,
    check_id: &str,
) -> Result<bool> {
    fn current(
        repo_root: &Path,
        graph: &WorkspaceGraph,
        ledger: &CheckEvidenceLedger,
        check_id: &str,
        visiting: &mut HashSet<String>,
    ) -> Result<bool> {
        if !visiting.insert(check_id.to_string()) {
            bail!("check dependency cycle includes '{check_id}'");
        }
        let check = graph
            .checks
            .get(check_id)
            .with_context(|| format!("workspace has no check named '{check_id}'"))?;
        for dependency in &check.depends_on {
            if !current(repo_root, graph, ledger, dependency, visiting)? {
                visiting.remove(check_id);
                return Ok(false);
            }
        }
        let dependency_outputs = check
            .depends_on
            .iter()
            .filter_map(|dependency| {
                ledger.get(dependency).map(|evidence| {
                    (
                        dependency.clone(),
                        evidence
                            .output_fingerprint
                            .clone()
                            .unwrap_or_else(|| evidence.input_fingerprint.clone()),
                    )
                })
            })
            .collect::<BTreeMap<_, _>>();
        let input = input_fingerprint(repo_root, &check.inputs)?;
        let output = output_fingerprint(repo_root, &check.outputs)?;
        let is_current = ledger
            .current(check, &input, &dependency_outputs, &output)
            .is_some();
        visiting.remove(check_id);
        Ok(is_current)
    }

    current(repo_root, graph, ledger, check_id, &mut HashSet::new())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckPlan {
    pub changed_paths: Vec<String>,
    pub affected_components: Vec<String>,
    pub checks: Vec<String>,
    #[serde(default)]
    pub reasons: BTreeMap<String, Vec<String>>,
}

impl CheckPlan {
    pub fn is_no_change(&self) -> bool {
        self.changed_paths.is_empty()
    }
}

pub fn plan_checks(graph: &WorkspaceGraph, repository: &RepositoryContext) -> Result<CheckPlan> {
    let changed_paths = repository.task_changed_paths()?;
    plan_checks_for_paths(graph, changed_paths)
}

pub fn plan_checks_for_paths(
    graph: &WorkspaceGraph,
    mut changed_paths: Vec<String>,
) -> Result<CheckPlan> {
    changed_paths.sort();
    changed_paths.dedup();
    let mut affected = BTreeSet::new();
    let mut has_unowned_path = false;
    for path in &changed_paths {
        let owners = graph
            .components
            .values()
            .filter(|component| component_owns_path(component, path))
            .map(|component| component.id.clone())
            .collect::<Vec<_>>();
        if owners.is_empty() {
            has_unowned_path = true;
        } else {
            affected.extend(owners);
        }
    }
    if has_unowned_path {
        affected.extend(graph.components.keys().cloned());
    }

    loop {
        let mut added = false;
        for component in graph.components.values() {
            if !affected.contains(&component.id)
                && component
                    .depends_on
                    .iter()
                    .any(|dependency| affected.contains(dependency))
            {
                affected.insert(component.id.clone());
                added = true;
            }
        }
        if !added {
            break;
        }
    }

    let mut selected = BTreeSet::new();
    let mut reasons = BTreeMap::<String, Vec<String>>::new();
    for check in graph.checks.values() {
        let affected_component = check
            .components
            .iter()
            .find(|component| affected.contains(*component))
            .cloned();
        let changed_input = changed_paths
            .iter()
            .find(|path| patterns_match_path(&check.inputs, path))
            .cloned();
        let always = check.trigger == CheckTrigger::Always;
        if always || affected_component.is_some() || changed_input.is_some() {
            selected.insert(check.id.clone());
            let check_reasons = reasons.entry(check.id.clone()).or_default();
            if always {
                check_reasons.push("configured to run always".to_string());
            }
            if let Some(component) = affected_component {
                check_reasons.push(format!("affected component {component}"));
            }
            if let Some(path) = changed_input {
                check_reasons.push(format!("changed input {path}"));
            }
        }
    }
    add_check_dependencies(graph, &mut selected)?;
    let checks = stable_topological_checks(graph, &selected)?;
    for check in &checks {
        reasons
            .entry(check.clone())
            .or_insert_with(|| vec!["dependency of another selected check".to_string()]);
    }
    Ok(CheckPlan {
        changed_paths,
        affected_components: affected.into_iter().collect(),
        checks,
        reasons,
    })
}

fn add_check_dependencies(graph: &WorkspaceGraph, selected: &mut BTreeSet<String>) -> Result<()> {
    let mut pending = selected.iter().cloned().collect::<Vec<_>>();
    while let Some(id) = pending.pop() {
        let check = graph
            .checks
            .get(&id)
            .with_context(|| format!("check plan references unknown check '{id}'"))?;
        for dependency in &check.depends_on {
            if selected.insert(dependency.clone()) {
                pending.push(dependency.clone());
            }
        }
    }
    Ok(())
}

fn stable_topological_checks(
    graph: &WorkspaceGraph,
    selected: &BTreeSet<String>,
) -> Result<Vec<String>> {
    let mut ordered = Vec::with_capacity(selected.len());
    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    fn visit(
        id: &str,
        graph: &WorkspaceGraph,
        selected: &BTreeSet<String>,
        visiting: &mut HashSet<String>,
        visited: &mut HashSet<String>,
        ordered: &mut Vec<String>,
    ) -> Result<()> {
        if visited.contains(id) {
            return Ok(());
        }
        if !visiting.insert(id.to_string()) {
            bail!("check dependency cycle reached '{id}' while planning");
        }
        let check = graph
            .checks
            .get(id)
            .with_context(|| format!("check plan references unknown check '{id}'"))?;
        for dependency in &check.depends_on {
            if selected.contains(dependency) {
                visit(dependency, graph, selected, visiting, visited, ordered)?;
            }
        }
        visiting.remove(id);
        visited.insert(id.to_string());
        ordered.push(id.to_string());
        Ok(())
    }
    for id in selected {
        visit(
            id,
            graph,
            selected,
            &mut visiting,
            &mut visited,
            &mut ordered,
        )?;
    }
    Ok(ordered)
}

fn component_owns_path(component: &crate::workspace::WorkspaceComponent, path: &str) -> bool {
    let included = if component.include.is_empty() {
        path == component.root || path.starts_with(&format!("{}/", component.root))
    } else {
        patterns_match_path(&component.include, path)
    };
    included && !patterns_match_path(&component.exclude, path)
}

fn patterns_match_path(patterns: &[String], path: &str) -> bool {
    compile_patterns(patterns).is_ok_and(|matcher| matcher.is_match(path))
}

fn compile_patterns(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(
            Glob::new(pattern).with_context(|| format!("invalid workspace pattern '{pattern}'"))?,
        );
    }
    builder
        .build()
        .context("failed to compile workspace patterns")
}

pub fn input_fingerprint(repo_root: &Path, patterns: &[String]) -> Result<String> {
    let snapshot = ContentSnapshot::capture(repo_root)?;
    let matcher = compile_patterns(patterns)?;
    let mut digest = Sha256::new();
    for pattern in patterns {
        digest.update(pattern.as_bytes());
        digest.update([0]);
    }
    for (path, content) in snapshot
        .paths
        .iter()
        .filter(|(path, _)| matcher.is_match(path.as_str()))
    {
        digest.update(path.as_bytes());
        digest.update([0]);
        digest.update(content.kind.as_bytes());
        digest.update([0]);
        digest.update(content.fingerprint.as_bytes());
        digest.update([0]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn output_fingerprint(repo_root: &Path, patterns: &[String]) -> Result<Option<String>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let matcher = compile_patterns(patterns)?;
    let roots = pattern_roots(repo_root, patterns);
    let mut matched = BTreeMap::new();
    for root in roots {
        if root.is_file() || root.is_symlink() {
            capture_output_path(repo_root, &root, &matcher, &mut matched)?;
            continue;
        }
        if !root.exists() {
            continue;
        }
        for entry in WalkDir::new(&root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|entry| entry.file_name() != ".git")
        {
            let entry = entry.with_context(|| {
                format!("failed to inspect check output below {}", root.display())
            })?;
            if entry.file_type().is_file() || entry.file_type().is_symlink() {
                capture_output_path(repo_root, entry.path(), &matcher, &mut matched)?;
            }
        }
    }
    if matched.is_empty() {
        return Ok(None);
    }
    let mut digest = Sha256::new();
    for (path, fingerprint) in matched {
        digest.update(path.as_bytes());
        digest.update([0]);
        digest.update(fingerprint.as_bytes());
        digest.update([0]);
    }
    Ok(Some(format!("{:x}", digest.finalize())))
}

fn pattern_roots(repo_root: &Path, patterns: &[String]) -> BTreeSet<PathBuf> {
    patterns
        .iter()
        .map(|pattern| {
            let prefix = pattern
                .split('/')
                .take_while(|part| !part.chars().any(|ch| matches!(ch, '*' | '?' | '[' | '{')))
                .collect::<Vec<_>>()
                .join("/");
            if prefix.is_empty() {
                repo_root.to_path_buf()
            } else {
                repo_root.join(prefix)
            }
        })
        .collect()
}

fn capture_output_path(
    repo_root: &Path,
    path: &Path,
    matcher: &GlobSet,
    matched: &mut BTreeMap<String, String>,
) -> Result<()> {
    let relative = path
        .strip_prefix(repo_root)
        .with_context(|| format!("check output {} escapes repository", path.display()))?
        .to_string_lossy()
        .replace('\\', "/");
    if !matcher.is_match(&relative) {
        return Ok(());
    }
    let mut digest = Sha256::new();
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect check output {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        digest.update(b"symlink");
        digest.update(
            std::fs::read_link(path)
                .with_context(|| format!("failed to read output symlink {}", path.display()))?
                .to_string_lossy()
                .as_bytes(),
        );
    } else {
        digest.update(b"file");
        digest.update(
            std::fs::read(path)
                .with_context(|| format!("failed to read check output {}", path.display()))?,
        );
    }
    matched.insert(relative, format!("{:x}", digest.finalize()));
    Ok(())
}

pub fn command_fingerprint(check: &WorkspaceCheck) -> String {
    let mut digest = Sha256::new();
    for value in [
        check.command.as_str(),
        check.cwd.as_str(),
        check.executor.as_str(),
    ] {
        digest.update(value.as_bytes());
        digest.update([0]);
    }
    digest.update(check.timeout_seconds.to_le_bytes());
    format!("{:x}", digest.finalize())
}

struct ExecutorRegistry<'a> {
    repo_root: &'a Path,
    graph: &'a WorkspaceGraph,
    project_environment: Option<&'a EnvironmentConfig>,
    project_backend: Option<&'a CommandBackend>,
    owned: BTreeMap<String, CommandBackend>,
    started: BTreeSet<String>,
}

impl<'a> ExecutorRegistry<'a> {
    fn new(
        repo_root: &'a Path,
        graph: &'a WorkspaceGraph,
        project_environment: Option<&'a EnvironmentConfig>,
        project_backend: Option<&'a CommandBackend>,
    ) -> Self {
        Self {
            repo_root,
            graph,
            project_environment,
            project_backend,
            owned: BTreeMap::new(),
            started: BTreeSet::new(),
        }
    }

    fn execute(
        &mut self,
        check: &WorkspaceCheck,
        sink: &mut dyn EventSink,
    ) -> Result<CheckCommandOutput> {
        self.ensure_started(&check.executor, sink)?;
        let backend = if self.graph.executors[&check.executor].kind == ExecutorKind::Project {
            self.project_backend
                .or_else(|| self.owned.get(&check.executor))
        } else {
            self.owned.get(&check.executor)
        }
        .with_context(|| {
            format!(
                "executor '{}' did not produce a command backend",
                check.executor
            )
        })?;
        backend.exec_workspace_check(check, self.repo_root)
    }

    fn ensure_started(&mut self, executor_id: &str, sink: &mut dyn EventSink) -> Result<()> {
        if self.started.contains(executor_id) {
            return Ok(());
        }
        let executor = self
            .graph
            .executors
            .get(executor_id)
            .with_context(|| format!("workspace graph has no executor '{executor_id}'"))?;
        let kind = executor_kind_name(executor.kind).to_string();
        let start_result = match executor.kind {
            ExecutorKind::Project if self.project_backend.is_some() => Ok(None),
            ExecutorKind::Project => {
                if let Some(environment) = self.project_environment {
                    CommandBackend::start(environment, self.repo_root).map(Some)
                } else {
                    Ok(Some(CommandBackend::Local {
                        workspace_root: self.repo_root.to_path_buf(),
                    }))
                }
            }
            ExecutorKind::Local => Ok(Some(CommandBackend::Local {
                workspace_root: self.repo_root.to_path_buf(),
            })),
            ExecutorKind::Container => self
                .executor_environment(executor.environment.as_deref())
                .and_then(|environment| {
                    if environment.backend != EnvironmentBackend::AppleContainers {
                        bail!(
                            "executor '{executor_id}' requires a container environment, but its backend is {:?}",
                            environment.backend
                        );
                    }
                    CommandBackend::start(&environment, self.repo_root).map(Some)
                }),
        };
        match start_result {
            Ok(backend) => {
                if let Some(backend) = backend {
                    self.owned.insert(executor_id.to_string(), backend);
                }
                self.started.insert(executor_id.to_string());
                sink.emit(AgentEvent::ExecutorStarted {
                    executor_id: executor_id.to_string(),
                    kind,
                    success: true,
                    detail: String::new(),
                    timestamp_ms: Some(now_millis()),
                });
                Ok(())
            }
            Err(error) => {
                sink.emit(AgentEvent::ExecutorStarted {
                    executor_id: executor_id.to_string(),
                    kind,
                    success: false,
                    detail: format!("{error:#}"),
                    timestamp_ms: Some(now_millis()),
                });
                Err(error)
            }
        }
    }

    fn executor_environment(&self, configured: Option<&str>) -> Result<EnvironmentConfig> {
        if let Some(configured) = configured {
            let path = if configured.ends_with(".toml") || configured.contains('/') {
                self.repo_root.join(configured)
            } else {
                self.repo_root
                    .join(".pb")
                    .join("environments")
                    .join(format!("{configured}.toml"))
            };
            return EnvironmentConfig::load_path(&path);
        }
        self.project_environment
            .cloned()
            .context("container executor has no environment configuration")
    }
}

fn executor_kind_name(kind: ExecutorKind) -> &'static str {
    match kind {
        ExecutorKind::Project => "project",
        ExecutorKind::Local => "local",
        ExecutorKind::Container => "container",
    }
}

pub struct WorkspaceCheckRuntime<'a> {
    repo_root: &'a Path,
    graph: &'a WorkspaceGraph,
    registry: ExecutorRegistry<'a>,
    ledger: CheckEvidenceLedger,
}

impl<'a> WorkspaceCheckRuntime<'a> {
    pub(crate) fn new(
        repo_root: &'a Path,
        graph: &'a WorkspaceGraph,
        project_environment: Option<&'a EnvironmentConfig>,
        project_backend: Option<&'a CommandBackend>,
        ledger: CheckEvidenceLedger,
    ) -> Self {
        Self {
            repo_root,
            graph,
            registry: ExecutorRegistry::new(repo_root, graph, project_environment, project_backend),
            ledger,
        }
    }

    pub fn ledger(&self) -> &CheckEvidenceLedger {
        &self.ledger
    }

    pub fn run_plan(
        &mut self,
        plan: &CheckPlan,
        source: EvidenceSource,
        nesting_depth: usize,
        sink: &mut dyn EventSink,
    ) -> Result<CheckRunSummary> {
        let mut summary = CheckRunSummary::default();
        let mut failed = BTreeSet::new();
        for check_id in &plan.checks {
            let check = self
                .graph
                .checks
                .get(check_id)
                .with_context(|| format!("check plan references unknown check '{check_id}'"))?;
            let failed_dependencies = check
                .depends_on
                .iter()
                .filter(|dependency| failed.contains(*dependency))
                .cloned()
                .collect::<Vec<_>>();
            if !failed_dependencies.is_empty() {
                failed.insert(check.id.clone());
                summary.skipped.push(check.id.clone());
                summary.failures.push(CheckFailureSummary {
                    check_id: check.id.clone(),
                    exit_status: 125,
                    timed_out: false,
                    output: String::new(),
                    skip_reason: Some(format!(
                        "dependency failure: {}",
                        failed_dependencies.join(", ")
                    )),
                });
                sink.emit(AgentEvent::CheckResult {
                    check_id: check.id.clone(),
                    exit_status: 125,
                    success: false,
                    timed_out: false,
                    output: String::new(),
                    truncated: false,
                    duration_ms: 0,
                    fingerprint: String::new(),
                    command: Some(check.command.clone()),
                    cwd: Some(check.cwd.clone()),
                    executor: Some(check.executor.clone()),
                    source: Some(source.as_str().to_string()),
                    command_fingerprint: Some(command_fingerprint(check)),
                    dependency_outputs: BTreeMap::new(),
                    output_fingerprint: None,
                    reused: false,
                    skip_reason: Some(format!(
                        "dependency failure: {}",
                        failed_dependencies.join(", ")
                    )),
                    nesting_depth: (nesting_depth > 0).then_some(nesting_depth),
                    timestamp_ms: Some(now_millis()),
                });
                continue;
            }
            let dependency_outputs = check
                .depends_on
                .iter()
                .filter_map(|dependency| {
                    self.ledger.get(dependency).map(|evidence| {
                        (
                            dependency.clone(),
                            evidence
                                .output_fingerprint
                                .clone()
                                .unwrap_or_else(|| evidence.input_fingerprint.clone()),
                        )
                    })
                })
                .collect::<BTreeMap<_, _>>();
            let input = input_fingerprint(self.repo_root, &check.inputs)?;
            let current_output = output_fingerprint(self.repo_root, &check.outputs)?;
            if let Some(evidence) =
                self.ledger
                    .current(check, &input, &dependency_outputs, &current_output)
            {
                summary.reused.push(check.id.clone());
                sink.emit(check_result_event(
                    evidence,
                    "current evidence reused".to_string(),
                    false,
                    true,
                    nesting_depth,
                ));
                continue;
            }

            let output = self.registry.execute(check, sink)?;
            let output_fingerprint = output_fingerprint(self.repo_root, &check.outputs)?;
            let missing_output = !check.outputs.is_empty() && output_fingerprint.is_none();
            let success = output.exit_status == 0 && !output.timed_out && !missing_output;
            let output_text = if missing_output {
                format!(
                    "{}\n\nrequired output was not produced: {}",
                    output.output,
                    check.outputs.join(", ")
                )
            } else {
                output.output.clone()
            };
            let evidence = CheckEvidence {
                check_id: check.id.clone(),
                command: check.command.clone(),
                cwd: check.cwd.clone(),
                command_fingerprint: command_fingerprint(check),
                input_fingerprint: input,
                dependency_outputs,
                output_fingerprint,
                exit_status: output.exit_status,
                success,
                timed_out: output.timed_out,
                duration_ms: output.duration_ms,
                executor: check.executor.clone(),
                source,
            };
            self.ledger.record(evidence.clone());
            summary.executed.push(check.id.clone());
            if !success {
                failed.insert(check.id.clone());
                summary.failed.push(check.id.clone());
                summary.failures.push(CheckFailureSummary {
                    check_id: check.id.clone(),
                    exit_status: output.exit_status,
                    timed_out: output.timed_out,
                    output: output_text.clone(),
                    skip_reason: None,
                });
            }
            sink.emit(check_result_event(
                &evidence,
                output_text,
                output.truncated,
                false,
                nesting_depth,
            ));
        }
        Ok(summary)
    }

    pub fn run_named(
        &mut self,
        check_id: &str,
        source: EvidenceSource,
        nesting_depth: usize,
        sink: &mut dyn EventSink,
    ) -> Result<CheckRunSummary> {
        if !self.graph.checks.contains_key(check_id) {
            bail!("workspace has no check named '{check_id}'");
        }
        let mut selected = BTreeSet::from([check_id.to_string()]);
        add_check_dependencies(self.graph, &mut selected)?;
        let plan = CheckPlan {
            changed_paths: Vec::new(),
            affected_components: Vec::new(),
            checks: stable_topological_checks(self.graph, &selected)?,
            reasons: BTreeMap::new(),
        };
        self.run_plan(&plan, source, nesting_depth, sink)
    }
}

fn check_result_event(
    evidence: &CheckEvidence,
    output: String,
    truncated: bool,
    reused: bool,
    nesting_depth: usize,
) -> AgentEvent {
    AgentEvent::CheckResult {
        check_id: evidence.check_id.clone(),
        exit_status: evidence.exit_status,
        success: evidence.success,
        timed_out: evidence.timed_out,
        output,
        truncated,
        duration_ms: if reused { 0 } else { evidence.duration_ms },
        fingerprint: evidence.input_fingerprint.clone(),
        command: Some(evidence.command.clone()),
        cwd: Some(evidence.cwd.clone()),
        executor: Some(evidence.executor.clone()),
        source: Some(evidence.source.as_str().to_string()),
        command_fingerprint: Some(evidence.command_fingerprint.clone()),
        dependency_outputs: evidence.dependency_outputs.clone(),
        output_fingerprint: evidence.output_fingerprint.clone(),
        reused,
        skip_reason: None,
        nesting_depth: (nesting_depth > 0).then_some(nesting_depth),
        timestamp_ms: Some(now_millis()),
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckRunSummary {
    pub executed: Vec<String>,
    pub reused: Vec<String>,
    pub failed: Vec<String>,
    pub skipped: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failures: Vec<CheckFailureSummary>,
}

impl CheckRunSummary {
    pub fn all_succeeded(&self) -> bool {
        self.failed.is_empty() && self.skipped.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckFailureSummary {
    pub check_id: String,
    pub exit_status: i32,
    pub timed_out: bool,
    pub output: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::{Executor, WorkspaceBaseline, WorkspaceComponent, WorkspaceGraphSource};
    use std::process::Command;

    fn init_repo() -> tempfile::TempDir {
        let repo = tempfile::tempdir().unwrap();
        for args in [
            &["init", "--initial-branch=main"][..],
            &["config", "user.name", "pb checks test"][..],
            &["config", "user.email", "checks@pb.local"][..],
        ] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(repo.path())
                    .status()
                    .unwrap()
                    .success()
            );
        }
        repo
    }

    fn write(root: &Path, relative: &str, content: &str) {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    fn commit_all(root: &Path) {
        assert!(
            Command::new("git")
                .args(["add", "-A"])
                .current_dir(root)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .args(["commit", "-m", "test: fixture"])
                .current_dir(root)
                .status()
                .unwrap()
                .success()
        );
    }

    fn graph() -> WorkspaceGraph {
        let executor = Executor {
            id: "local".to_string(),
            kind: ExecutorKind::Local,
            environment: None,
        };
        let executor_id = executor.id.clone();
        let shared = WorkspaceComponent {
            id: "shared".to_string(),
            root: "shared".to_string(),
            include: vec!["shared/**".to_string()],
            exclude: Vec::new(),
            executor: executor_id.clone(),
            depends_on: Vec::new(),
        };
        let api = WorkspaceComponent {
            id: "api".to_string(),
            root: "api".to_string(),
            include: vec!["api/**".to_string()],
            exclude: Vec::new(),
            executor: executor_id.clone(),
            depends_on: vec![shared.id.clone()],
        };
        let worker = WorkspaceComponent {
            id: "worker".to_string(),
            root: "worker".to_string(),
            include: vec!["worker/**".to_string()],
            exclude: Vec::new(),
            executor: executor_id.clone(),
            depends_on: Vec::new(),
        };
        let check = |id: &str, component: &str, command: &str| WorkspaceCheck {
            id: id.to_string(),
            label: id.to_string(),
            command: command.to_string(),
            cwd: ".".to_string(),
            executor: executor_id.clone(),
            components: vec![component.to_string()],
            trigger: CheckTrigger::Changed,
            inputs: vec![format!("{component}/**")],
            outputs: Vec::new(),
            depends_on: Vec::new(),
            timeout_seconds: 10,
        };
        WorkspaceGraph {
            version: 1,
            executors: BTreeMap::from([(executor_id.clone(), executor)]),
            components: BTreeMap::from([
                (shared.id.clone(), shared),
                (api.id.clone(), api),
                (worker.id.clone(), worker),
            ]),
            checks: BTreeMap::from([
                (
                    "shared-test".to_string(),
                    check("shared-test", "shared", "true"),
                ),
                ("api-test".to_string(), check("api-test", "api", "true")),
                (
                    "worker-test".to_string(),
                    check("worker-test", "worker", "true"),
                ),
            ]),
            tasks: BTreeMap::new(),
            cargo_workspaces: BTreeMap::new(),
            discovery_warnings: Vec::new(),
            source: WorkspaceGraphSource::Explicit,
        }
    }

    #[test]
    fn affected_component_closure_selects_dependants_but_not_unrelated_services() {
        let plan = plan_checks_for_paths(&graph(), vec!["shared/src/lib.rs".to_string()]).unwrap();
        assert_eq!(plan.affected_components, vec!["api", "shared"]);
        assert_eq!(plan.checks, vec!["api-test", "shared-test"]);
        assert!(!plan.checks.contains(&"worker-test".to_string()));
    }

    #[test]
    fn no_change_selects_only_explicit_always_checks() {
        let mut graph = graph();
        graph.checks.get_mut("worker-test").unwrap().trigger = CheckTrigger::Always;
        let plan = plan_checks_for_paths(&graph, Vec::new()).unwrap();
        assert!(plan.is_no_change());
        assert_eq!(plan.checks, vec!["worker-test"]);
    }

    #[test]
    fn current_evidence_is_reused_and_relevant_mutation_makes_it_stale() {
        let repo = init_repo();
        write(repo.path(), "api/input.txt", "one\n");
        commit_all(repo.path());
        let mut graph = graph();
        graph.checks.get_mut("api-test").unwrap().command = "true".to_string();
        let backend = CommandBackend::Local {
            workspace_root: repo.path().to_path_buf(),
        };
        let mut runtime = WorkspaceCheckRuntime::new(
            repo.path(),
            &graph,
            None,
            Some(&backend),
            CheckEvidenceLedger::default(),
        );
        let plan = CheckPlan {
            changed_paths: vec!["api/input.txt".to_string()],
            affected_components: vec!["api".to_string()],
            checks: vec!["api-test".to_string()],
            reasons: BTreeMap::new(),
        };
        let mut events = Vec::new();
        let first = runtime
            .run_plan(&plan, EvidenceSource::Handoff, 0, &mut |event| {
                events.push(event)
            })
            .unwrap();
        assert_eq!(first.executed, vec!["api-test"]);
        let second = runtime
            .run_plan(&plan, EvidenceSource::Handoff, 0, &mut |event| {
                events.push(event)
            })
            .unwrap();
        assert_eq!(second.reused, vec!["api-test"]);

        write(repo.path(), "api/input.txt", "two\n");
        let third = runtime
            .run_plan(&plan, EvidenceSource::Handoff, 0, &mut |event| {
                events.push(event)
            })
            .unwrap();
        assert_eq!(third.executed, vec!["api-test"]);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::ExecutorStarted { .. }))
                .count(),
            1,
            "the local executor must be started lazily and reused"
        );
    }

    #[test]
    fn only_executors_needed_by_the_selected_plan_are_started() {
        let repo = init_repo();
        write(repo.path(), "shared/input.txt", "one\n");
        commit_all(repo.path());
        let mut graph = graph();
        graph.executors.insert(
            "worker-local".to_string(),
            Executor {
                id: "worker-local".to_string(),
                kind: ExecutorKind::Local,
                environment: None,
            },
        );
        graph.components.get_mut("worker").unwrap().executor = "worker-local".to_string();
        graph.checks.get_mut("worker-test").unwrap().executor = "worker-local".to_string();
        let plan = plan_checks_for_paths(&graph, vec!["shared/input.txt".to_string()]).unwrap();
        let mut runtime = WorkspaceCheckRuntime::new(
            repo.path(),
            &graph,
            None,
            None,
            CheckEvidenceLedger::default(),
        );
        let mut started = Vec::new();
        let summary = runtime
            .run_plan(&plan, EvidenceSource::Handoff, 0, &mut |event| {
                if let AgentEvent::ExecutorStarted { executor_id, .. } = event {
                    started.push(executor_id);
                }
            })
            .unwrap();
        assert!(summary.all_succeeded());
        assert_eq!(started, vec!["local"]);
    }

    #[test]
    fn failed_dependency_skips_consumers_without_starting_their_executor() {
        let repo = init_repo();
        write(repo.path(), "api/input.txt", "one\n");
        commit_all(repo.path());
        let mut graph = graph();
        graph.executors.insert(
            "consumer-local".to_string(),
            Executor {
                id: "consumer-local".to_string(),
                kind: ExecutorKind::Local,
                environment: None,
            },
        );
        let producer = graph.checks.get_mut("shared-test").unwrap();
        producer.command = "false".to_string();
        let consumer = graph.checks.get_mut("api-test").unwrap();
        consumer.command = "touch consumer-ran".to_string();
        consumer.executor = "consumer-local".to_string();
        consumer.depends_on = vec!["shared-test".to_string()];
        let plan = CheckPlan {
            changed_paths: vec!["api/input.txt".to_string()],
            affected_components: vec!["api".to_string()],
            checks: vec!["shared-test".to_string(), "api-test".to_string()],
            reasons: BTreeMap::new(),
        };
        let mut runtime = WorkspaceCheckRuntime::new(
            repo.path(),
            &graph,
            None,
            None,
            CheckEvidenceLedger::default(),
        );
        let mut started = Vec::new();
        let summary = runtime
            .run_plan(&plan, EvidenceSource::Handoff, 0, &mut |event| {
                if let AgentEvent::ExecutorStarted { executor_id, .. } = event {
                    started.push(executor_id);
                }
            })
            .unwrap();
        assert_eq!(summary.failed, vec!["shared-test"]);
        assert_eq!(summary.skipped, vec!["api-test"]);
        assert_eq!(started, vec!["local"]);
        assert!(!repo.path().join("consumer-ran").exists());
    }

    #[test]
    fn persisted_check_events_restore_reusable_evidence() {
        let repo = init_repo();
        write(repo.path(), "api/input.txt", "one\n");
        commit_all(repo.path());
        let graph = graph();
        let plan = CheckPlan {
            changed_paths: vec!["api/input.txt".to_string()],
            affected_components: vec!["api".to_string()],
            checks: vec!["api-test".to_string()],
            reasons: BTreeMap::new(),
        };
        let mut events = Vec::new();
        let mut first = WorkspaceCheckRuntime::new(
            repo.path(),
            &graph,
            None,
            None,
            CheckEvidenceLedger::default(),
        );
        first
            .run_plan(&plan, EvidenceSource::AgentTool, 0, &mut |event| {
                events.push(event)
            })
            .unwrap();
        let ledger = CheckEvidenceLedger::from_events(&events);
        let mut resumed = WorkspaceCheckRuntime::new(repo.path(), &graph, None, None, ledger);
        let summary = resumed
            .run_plan(&plan, EvidenceSource::Handoff, 0, &mut |_| {})
            .unwrap();
        assert_eq!(summary.reused, vec!["api-test"]);
        assert!(summary.executed.is_empty());

        write(repo.path(), "api/input.txt", "two\n");
        let mut stale_resume = WorkspaceCheckRuntime::new(
            repo.path(),
            &graph,
            None,
            None,
            CheckEvidenceLedger::from_events(&events),
        );
        let stale_summary = stale_resume
            .run_plan(&plan, EvidenceSource::Handoff, 0, &mut |_| {})
            .unwrap();
        assert_eq!(stale_summary.executed, vec!["api-test"]);
        assert!(stale_summary.reused.is_empty());
    }

    #[test]
    fn generated_output_is_required_and_part_of_consumer_dependency_key() {
        let repo = init_repo();
        write(repo.path(), "web/source.txt", "source\n");
        write(repo.path(), "app/main.txt", "app\n");
        commit_all(repo.path());
        let executor = Executor {
            id: "local".to_string(),
            kind: ExecutorKind::Local,
            environment: None,
        };
        let producer = WorkspaceCheck {
            id: "bundle".to_string(),
            label: "bundle".to_string(),
            command: "mkdir -p generated && cp web/source.txt generated/bundle.txt".to_string(),
            cwd: ".".to_string(),
            executor: executor.id.clone(),
            components: vec!["web".to_string()],
            trigger: CheckTrigger::Needed,
            inputs: vec!["web/**".to_string()],
            outputs: vec!["generated/**".to_string()],
            depends_on: Vec::new(),
            timeout_seconds: 10,
        };
        let consumer = WorkspaceCheck {
            id: "app-test".to_string(),
            label: "app".to_string(),
            command: "test -f generated/bundle.txt".to_string(),
            cwd: ".".to_string(),
            executor: executor.id.clone(),
            components: vec!["app".to_string()],
            trigger: CheckTrigger::Changed,
            inputs: vec!["app/**".to_string()],
            outputs: Vec::new(),
            depends_on: vec![producer.id.clone()],
            timeout_seconds: 10,
        };
        let graph = WorkspaceGraph {
            version: 1,
            executors: BTreeMap::from([(executor.id.clone(), executor)]),
            components: BTreeMap::from([
                (
                    "web".to_string(),
                    WorkspaceComponent {
                        id: "web".to_string(),
                        root: "web".to_string(),
                        include: vec!["web/**".to_string()],
                        exclude: Vec::new(),
                        executor: "local".to_string(),
                        depends_on: Vec::new(),
                    },
                ),
                (
                    "app".to_string(),
                    WorkspaceComponent {
                        id: "app".to_string(),
                        root: "app".to_string(),
                        include: vec!["app/**".to_string()],
                        exclude: Vec::new(),
                        executor: "local".to_string(),
                        depends_on: vec!["web".to_string()],
                    },
                ),
            ]),
            checks: BTreeMap::from([
                (producer.id.clone(), producer),
                (consumer.id.clone(), consumer),
            ]),
            tasks: BTreeMap::new(),
            cargo_workspaces: BTreeMap::new(),
            discovery_warnings: Vec::new(),
            source: WorkspaceGraphSource::Explicit,
        };
        let backend = CommandBackend::Local {
            workspace_root: repo.path().to_path_buf(),
        };
        let mut runtime = WorkspaceCheckRuntime::new(
            repo.path(),
            &graph,
            None,
            Some(&backend),
            CheckEvidenceLedger::default(),
        );
        let plan = CheckPlan {
            changed_paths: vec!["web/source.txt".to_string()],
            affected_components: vec!["app".to_string(), "web".to_string()],
            checks: vec!["bundle".to_string(), "app-test".to_string()],
            reasons: BTreeMap::new(),
        };
        let mut events = Vec::new();
        let summary = runtime
            .run_plan(&plan, EvidenceSource::Handoff, 0, &mut |event| {
                events.push(event)
            })
            .unwrap();
        assert!(summary.all_succeeded());
        assert!(
            runtime.ledger()["app-test"]
                .dependency_outputs
                .contains_key("bundle")
        );

        write(repo.path(), "app/main.txt", "app changed\n");
        let app_plan = plan_checks_for_paths(&graph, vec!["app/main.txt".to_string()]).unwrap();
        assert_eq!(app_plan.checks, vec!["bundle", "app-test"]);
        let current_bundle = runtime
            .run_plan(&app_plan, EvidenceSource::Handoff, 0, &mut |event| {
                events.push(event)
            })
            .unwrap();
        assert_eq!(current_bundle.executed, vec!["app-test"]);
        assert_eq!(current_bundle.reused, vec!["bundle"]);

        std::fs::remove_file(repo.path().join("generated/bundle.txt")).unwrap();
        let mut resumed = WorkspaceCheckRuntime::new(
            repo.path(),
            &graph,
            None,
            Some(&backend),
            CheckEvidenceLedger::from_events(&events),
        );
        let rerun = resumed
            .run_plan(&app_plan, EvidenceSource::Handoff, 0, &mut |_| {})
            .unwrap();
        assert_eq!(rerun.executed, vec!["bundle"]);
        assert_eq!(rerun.reused, vec!["app-test"]);
    }

    impl std::ops::Index<&str> for CheckEvidenceLedger {
        type Output = CheckEvidence;

        fn index(&self, index: &str) -> &Self::Output {
            self.get(index).unwrap()
        }
    }

    #[test]
    fn repository_plan_uses_task_baseline_across_invocations() {
        let repo = init_repo();
        write(repo.path(), "shared/value.txt", "one\n");
        commit_all(repo.path());
        let task_baseline = WorkspaceBaseline::capture(repo.path()).unwrap();
        write(repo.path(), "shared/value.txt", "two\n");
        let repository =
            RepositoryContext::resume(repo.path(), repo.path(), task_baseline).unwrap();
        let plan = plan_checks(&graph(), &repository).unwrap();
        assert!(plan.checks.contains(&"shared-test".to_string()));
        assert!(plan.checks.contains(&"api-test".to_string()));
    }
}
