//! Controller-owned semantic analysis worlds and bounded LSP provider orchestration.
//!
//! This module owns I/O, deadlines, provider processes, and live-workspace revalidation. Portable
//! identities, diagnostic debt, and verdict types remain in `pb-control-collar`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use pb_control_collar::analysis::{
    AnalyzerCapability, BaselineCompleteness, ClosureVerdict, DefiniteErrorClass, DiagnosticDelta,
    DiagnosticIdentity, ProviderVerdict, SemanticDiagnosticSnapshot, SemanticFileBinding,
    SemanticWorldSnapshot, UnknownReason, Viability, diagnostic_delta,
};
use pb_control_collar::mutation::LogicalPath;
use pb_control_collar::mutation::{
    WorkspaceSnapshot, prepare_create, prepare_patch, prepare_replace,
};
use pb_control_collar::receipt::Digest;
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::lsp::{
    LspOverlayDiagnosticReport, LspOverlayDiagnosticTarget, LspOverlayDocument, LspToolRegistry,
};

const SEMANTIC_PROVIDER_TIMEOUT: Duration = Duration::from_secs(8);
const SEMANTIC_BOUNDARY_CACHE_ENTRIES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticOverlayMutation {
    pub path: String,
    pub base: Option<String>,
    pub result: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderSemanticResult {
    pub world: SemanticWorldSnapshot,
    pub baseline: SemanticDiagnosticSnapshot,
    pub candidate: SemanticDiagnosticSnapshot,
    pub delta: DiagnosticDelta,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticGateReport {
    pub workspace_fingerprint: String,
    pub providers: Vec<ProviderSemanticResult>,
    pub verdict: ProviderVerdict,
}

pub struct SemanticProviderBroker<'a> {
    registry: &'a LspToolRegistry,
    workspace_root: &'a Path,
    timeout: Duration,
}

/// Controller-owned bridge between an immutable mutation snapshot and blocking semantic
/// providers. The sampler sees only portable verdicts; filesystem and provider authority remain
/// here, outside `pb-control-collar`.
struct LspSemanticBoundaryProvider {
    registry: LspToolRegistry,
    workspace_root: PathBuf,
    workspace: WorkspaceSnapshot,
    expected_workspace_fingerprint: String,
    cache: Mutex<BTreeMap<String, ProviderVerdict>>,
}

pub fn semantic_boundary_control(
    registry: &LspToolRegistry,
    workspace_root: &Path,
    workspace: &WorkspaceSnapshot,
) -> Result<Option<crate::inference::SemanticBoundaryControl>> {
    if !registry.servers.values().any(|config| {
        !config.disabled
            && config.semantic_enforcement != crate::lsp::LspSemanticEnforcement::Disabled
    }) {
        return Ok(None);
    }
    let before = crate::workspace::ContentSnapshot::capture(workspace_root)?;
    for entry in workspace.entries() {
        let path = workspace_root.join(entry.path.as_str());
        let live = std::fs::read(&path).with_context(|| {
            format!(
                "failed to bind semantic boundary base {}",
                entry.path.as_str()
            )
        })?;
        if live != entry.bytes {
            bail!(
                "semantic boundary snapshot drifted before generation for {}",
                entry.path.as_str()
            );
        }
    }
    let after = crate::workspace::ContentSnapshot::capture(workspace_root)?;
    if before.fingerprint != after.fingerprint {
        bail!("workspace changed while binding the semantic generation world");
    }
    Ok(Some(crate::inference::SemanticBoundaryControl::new(
        LspSemanticBoundaryProvider {
            registry: registry.clone(),
            workspace_root: workspace_root.to_path_buf(),
            workspace: workspace.clone(),
            expected_workspace_fingerprint: after.fingerprint,
            cache: Mutex::new(BTreeMap::new()),
        },
    )))
}

impl crate::inference::SemanticBoundaryProvider for LspSemanticBoundaryProvider {
    fn probe(&self, tool: &str, arguments: &Value) -> ProviderVerdict {
        let key = semantic_probe_identity(tool, arguments);
        if let Ok(cache) = self.cache.lock()
            && let Some(verdict) = cache.get(&key)
        {
            return verdict.clone();
        }
        let mutations = match semantic_mutations_from_call(&self.workspace, tool, arguments) {
            Ok(mutations) => mutations,
            Err(_) => {
                return semantic_unknown(UnknownReason::UnsupportedConstruct, true);
            }
        };
        let required = transaction_has_required_provider(&self.registry, &mutations);
        let verdict = match crate::workspace::ContentSnapshot::capture(&self.workspace_root) {
            Ok(current) if current.fingerprint == self.expected_workspace_fingerprint => {
                match SemanticProviderBroker::new(&self.registry, &self.workspace_root)
                    .analyze_transaction(&mutations)
                {
                    Ok(report) => boundary_enforcement_verdict(report.verdict, required),
                    Err(_) => semantic_unknown(UnknownReason::ProviderUnavailable, required),
                }
            }
            Ok(_) => semantic_unknown(UnknownReason::ConfigurationChanged, required),
            Err(_) => semantic_unknown(UnknownReason::ProviderUnavailable, required),
        };
        if let Ok(mut cache) = self.cache.lock() {
            if cache.len() >= SEMANTIC_BOUNDARY_CACHE_ENTRIES
                && let Some(oldest) = cache.keys().next().cloned()
            {
                cache.remove(&oldest);
            }
            cache.insert(key, verdict.clone());
        }
        verdict
    }
}

fn boundary_enforcement_verdict(mut verdict: ProviderVerdict, required: bool) -> ProviderVerdict {
    if required && verdict.closure != ClosureVerdict::Allow {
        verdict.closure = ClosureVerdict::Reject;
    } else if !required && verdict.closure == ClosureVerdict::Reject {
        verdict.closure = ClosureVerdict::Defer;
    }
    verdict
}

impl<'a> SemanticProviderBroker<'a> {
    pub fn new(registry: &'a LspToolRegistry, workspace_root: &'a Path) -> Self {
        Self {
            registry,
            workspace_root,
            timeout: SEMANTIC_PROVIDER_TIMEOUT,
        }
    }

    #[cfg(test)]
    fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn analyze_transaction(
        &self,
        mutations: &[SemanticOverlayMutation],
    ) -> Result<SemanticGateReport> {
        if mutations.is_empty() {
            bail!("semantic transaction requires at least one mutation");
        }
        let content = crate::workspace::ContentSnapshot::capture(self.workspace_root)?;
        let mut paths = BTreeSet::new();
        let mut base_documents = Vec::new();
        let mut result_documents = Vec::new();
        let mut deleted = false;
        let mut bindings = Vec::new();
        for mutation in mutations {
            let path = LogicalPath::parse(mutation.path.clone())?;
            if !paths.insert(path.clone()) {
                bail!("semantic transaction repeats path {}", mutation.path);
            }
            if let Some(base) = &mutation.base {
                bindings.push(SemanticFileBinding {
                    path: path.clone(),
                    sha256: Digest::of(base.as_bytes()),
                });
                base_documents.push(LspOverlayDocument {
                    path: mutation.path.clone(),
                    text: base.clone(),
                });
            }
            if let Some(result) = &mutation.result {
                result_documents.push(LspOverlayDocument {
                    path: mutation.path.clone(),
                    text: result.clone(),
                });
            } else {
                deleted = true;
            }
        }

        let baseline_report = if base_documents.is_empty() {
            None
        } else {
            Some(crate::lsp::overlay_diagnostics(
                self.registry,
                self.workspace_root,
                &base_documents,
                &content.fingerprint,
                self.timeout,
            )?)
        };
        let candidate_report = if result_documents.is_empty() {
            None
        } else {
            Some(crate::lsp::overlay_diagnostics(
                self.registry,
                self.workspace_root,
                &result_documents,
                &content.fingerprint,
                self.timeout,
            )?)
        };

        let mut unknown_reasons = BTreeSet::new();
        if let Some(report) = &baseline_report {
            unknown_reasons.extend(report.unknown_reasons.iter().copied());
        }
        if let Some(report) = &candidate_report {
            unknown_reasons.extend(report.unknown_reasons.iter().copied());
        }
        if deleted {
            // Pull diagnostics for a removed document cannot prove that dependants still resolve.
            unknown_reasons.insert(UnknownReason::UnsupportedConstruct);
        }

        let dependency_sha256 = dependency_identity(&content);
        let mut providers = Vec::new();
        for (server, config) in &self.registry.servers {
            let baseline_targets = targets_for_server(baseline_report.as_ref(), server);
            let candidate_targets = targets_for_server(candidate_report.as_ref(), server);
            if baseline_targets.is_empty() && candidate_targets.is_empty() {
                continue;
            }
            let (provider_version, pinned) = provider_identity(config);
            let configuration_sha256 = format!(
                "{:x}",
                Sha256::digest(
                    serde_json::to_vec(config)
                        .context("failed to serialize semantic provider configuration")?
                )
            );
            let language = provider_language(config, mutations);
            let capabilities = capabilities_for_language(&language);
            let complete = pinned
                && baseline_report
                    .as_ref()
                    .is_none_or(|report| report.complete)
                && candidate_report
                    .as_ref()
                    .is_none_or(|report| report.complete)
                && !deleted;
            let world = SemanticWorldSnapshot::new(
                server,
                provider_version,
                language,
                content.fingerprint.clone(),
                configuration_sha256,
                dependency_sha256.clone(),
                capabilities,
                bindings.clone(),
                if complete {
                    BaselineCompleteness::Complete
                } else {
                    BaselineCompleteness::Incomplete
                },
            )?;
            let mut provider_unknown = unknown_reasons.clone();
            if !pinned {
                provider_unknown.insert(UnknownReason::ProviderUnavailable);
            }
            let baseline = semantic_snapshot(
                &world,
                baseline_targets,
                if complete {
                    BaselineCompleteness::Complete
                } else {
                    BaselineCompleteness::Incomplete
                },
                &provider_unknown,
            )?;
            let candidate = semantic_snapshot(
                &world,
                candidate_targets,
                if complete {
                    BaselineCompleteness::Complete
                } else {
                    BaselineCompleteness::Incomplete
                },
                &provider_unknown,
            )?;
            let delta = diagnostic_delta(&baseline, &candidate)?;
            providers.push(ProviderSemanticResult {
                world,
                baseline,
                candidate,
                delta,
            });
        }
        providers.sort_by(|left, right| left.world.id.provider.cmp(&right.world.id.provider));

        if providers.is_empty() {
            unknown_reasons.insert(UnknownReason::ProviderUnavailable);
        }
        for provider in &providers {
            unknown_reasons.extend(provider.delta.unknown_reasons.iter().copied());
        }
        let definite_errors = providers
            .iter()
            .flat_map(|provider| {
                provider
                    .delta
                    .introduced
                    .iter()
                    .map(|diagnostic| diagnostic.class)
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let authoritative = !providers.is_empty()
            && providers
                .iter()
                .all(|provider| provider.delta.authoritative)
            && unknown_reasons.is_empty();
        let verdict = if !definite_errors.is_empty() {
            ProviderVerdict {
                viability: Viability::Impossible,
                closure: ClosureVerdict::Reject,
                definite_errors,
                unknown_reasons: unknown_reasons.into_iter().collect(),
                obligations: Vec::new(),
                biases: Vec::new(),
            }
        } else if authoritative {
            ProviderVerdict {
                viability: Viability::Valid,
                closure: ClosureVerdict::Allow,
                definite_errors: Vec::new(),
                unknown_reasons: Vec::new(),
                obligations: Vec::new(),
                biases: Vec::new(),
            }
        } else {
            ProviderVerdict {
                viability: Viability::Unknown,
                closure: ClosureVerdict::Defer,
                definite_errors: Vec::new(),
                unknown_reasons: unknown_reasons.into_iter().collect(),
                obligations: Vec::new(),
                biases: Vec::new(),
            }
        };
        Ok(SemanticGateReport {
            workspace_fingerprint: content.fingerprint,
            providers,
            verdict,
        })
    }
}

/// Run the configured semantic transaction gate. Disabled providers are ignored. Advisory
/// providers return evidence without blocking publication. If any participating provider is
/// required, only an authoritative `Allow` verdict permits publication.
pub fn enforce_configured_transaction(
    registry: &LspToolRegistry,
    workspace_root: &Path,
    mutations: &[SemanticOverlayMutation],
) -> Result<Option<SemanticGateReport>> {
    let has_configured_provider = transaction_has_configured_provider(registry, mutations);
    if !has_configured_provider {
        return Ok(None);
    }
    let report =
        SemanticProviderBroker::new(registry, workspace_root).analyze_transaction(mutations)?;
    let required = transaction_has_required_provider(registry, mutations);
    if required && report.verdict.closure != ClosureVerdict::Allow {
        let errors = report
            .verdict
            .definite_errors
            .iter()
            .map(|error| format!("{error:?}").to_ascii_lowercase())
            .collect::<Vec<_>>();
        let unknown = report
            .verdict
            .unknown_reasons
            .iter()
            .map(|reason| format!("{reason:?}").to_ascii_lowercase())
            .collect::<Vec<_>>();
        bail!(
            "required semantic gate did not authorize publication (definite_errors=[{}], unknown_reasons=[{}])",
            errors.join(","),
            unknown.join(",")
        );
    }
    Ok(Some(report))
}

fn transaction_has_configured_provider(
    registry: &LspToolRegistry,
    mutations: &[SemanticOverlayMutation],
) -> bool {
    transaction_has_provider(registry, mutations, |mode| {
        mode != crate::lsp::LspSemanticEnforcement::Disabled
    })
}

fn transaction_has_required_provider(
    registry: &LspToolRegistry,
    mutations: &[SemanticOverlayMutation],
) -> bool {
    transaction_has_provider(registry, mutations, |mode| {
        mode == crate::lsp::LspSemanticEnforcement::Required
    })
}

fn transaction_has_provider(
    registry: &LspToolRegistry,
    mutations: &[SemanticOverlayMutation],
    accepts: impl Fn(crate::lsp::LspSemanticEnforcement) -> bool,
) -> bool {
    mutations.iter().any(|mutation| {
        let extension = Path::new(&mutation.path)
            .extension()
            .and_then(|extension| extension.to_str());
        registry.servers.values().any(|config| {
            !config.disabled
                && accepts(config.semantic_enforcement)
                && extension.is_some_and(|extension| {
                    config
                        .language_ids
                        .iter()
                        .any(|language| language_extension_matches(language, extension))
                })
        })
    })
}

fn semantic_mutations_from_call(
    workspace: &WorkspaceSnapshot,
    tool: &str,
    arguments: &Value,
) -> Result<Vec<SemanticOverlayMutation>> {
    let string = |name: &str| {
        arguments
            .get(name)
            .and_then(Value::as_str)
            .with_context(|| format!("{tool} requires string argument {name}"))
    };
    match tool {
        "write_file" => {
            let path = string("path")?;
            let content = string("content")?;
            let prepared = prepare_create(
                workspace,
                LogicalPath::parse(path.to_string())?,
                content.as_bytes().to_vec(),
            )?;
            Ok(vec![SemanticOverlayMutation {
                path: path.to_string(),
                base: None,
                result: prepared
                    .result_bytes()
                    .map(std::str::from_utf8)
                    .transpose()?
                    .map(str::to_string),
            }])
        }
        "replace_file" => {
            let path = string("path")?;
            let logical = LogicalPath::parse(path.to_string())?;
            let content = string("content")?;
            let prepared =
                prepare_replace(workspace, logical.clone(), content.as_bytes().to_vec())?;
            let base = workspace
                .get(&logical)
                .context("semantic replace base is absent")?;
            Ok(vec![SemanticOverlayMutation {
                path: path.to_string(),
                base: Some(std::str::from_utf8(&base.bytes)?.to_string()),
                result: prepared
                    .result_bytes()
                    .map(std::str::from_utf8)
                    .transpose()?
                    .map(str::to_string),
            }])
        }
        "edit_file" => {
            let path = string("path")?;
            let logical = LogicalPath::parse(path.to_string())?;
            let old_text = string("old_text")?;
            let new_text = string("new_text")?;
            if old_text.is_empty() {
                bail!("semantic edit old_text is empty");
            }
            let base = workspace
                .get(&logical)
                .context("semantic edit base is absent")?;
            let base_text = std::str::from_utf8(&base.bytes)?;
            if base_text.matches(old_text).take(2).count() != 1 {
                bail!("semantic edit old_text is absent or ambiguous");
            }
            let result = base_text.replacen(old_text, new_text, 1);
            let prepared = prepare_replace(workspace, logical, result.as_bytes().to_vec())?;
            Ok(vec![SemanticOverlayMutation {
                path: path.to_string(),
                base: Some(base_text.to_string()),
                result: prepared
                    .result_bytes()
                    .map(std::str::from_utf8)
                    .transpose()?
                    .map(str::to_string),
            }])
        }
        "apply_patch" => {
            let patch = string("patch")?;
            let prepared = prepare_patch(workspace, patch, 32, 256)?;
            prepared
                .files()
                .iter()
                .map(|file| {
                    let base = workspace
                        .get(file.path())
                        .map(|entry| std::str::from_utf8(&entry.bytes).map(str::to_string))
                        .transpose()?;
                    let result = file
                        .result_bytes()
                        .map(std::str::from_utf8)
                        .transpose()?
                        .map(str::to_string);
                    Ok(SemanticOverlayMutation {
                        path: file.path().as_str().to_string(),
                        base,
                        result,
                    })
                })
                .collect()
        }
        _ => bail!("semantic boundary does not cover tool {tool}"),
    }
}

fn semantic_probe_identity(tool: &str, arguments: &Value) -> String {
    let mut digest = Sha256::new();
    digest.update(tool.as_bytes());
    digest.update([0]);
    if let Ok(bytes) = serde_json::to_vec(arguments) {
        digest.update(bytes);
    }
    format!("{:x}", digest.finalize())
}

fn semantic_unknown(reason: UnknownReason, reject: bool) -> ProviderVerdict {
    ProviderVerdict {
        viability: Viability::Unknown,
        closure: if reject {
            ClosureVerdict::Reject
        } else {
            ClosureVerdict::Defer
        },
        definite_errors: Vec::new(),
        unknown_reasons: vec![reason],
        obligations: Vec::new(),
        biases: Vec::new(),
    }
}

fn language_extension_matches(language: &str, extension: &str) -> bool {
    match language.to_ascii_lowercase().as_str() {
        "rust" => extension == "rs",
        "typescript" => extension == "ts",
        "typescriptreact" => extension == "tsx",
        "javascript" => matches!(extension, "js" | "mjs" | "cjs"),
        "javascriptreact" => extension == "jsx",
        "python" => matches!(extension, "py" | "pyi"),
        _ => false,
    }
}

fn targets_for_server<'a>(
    report: Option<&'a LspOverlayDiagnosticReport>,
    server: &str,
) -> Vec<&'a LspOverlayDiagnosticTarget> {
    report
        .into_iter()
        .flat_map(|report| &report.targets)
        .filter(|target| target.server == server)
        .collect()
}

fn semantic_snapshot(
    world: &SemanticWorldSnapshot,
    targets: Vec<&LspOverlayDiagnosticTarget>,
    completeness: BaselineCompleteness,
    unknown_reasons: &BTreeSet<UnknownReason>,
) -> Result<SemanticDiagnosticSnapshot> {
    let mut document_versions = BTreeMap::new();
    let mut document_sha256 = BTreeMap::new();
    let mut diagnostics = BTreeSet::new();
    let mut unclassified = false;
    for target in targets {
        let path = LogicalPath::parse(target.path.clone())?;
        document_versions.insert(path.clone(), target.version);
        document_sha256.insert(path.clone(), target.text_sha256);
        let items = target
            .diagnostics
            .as_array()
            .context("semantic LSP diagnostic result is not an array")?;
        for item in items {
            if item.get("severity").and_then(Value::as_u64) != Some(1) {
                continue;
            }
            let Some(identity) = diagnostic_identity(&target.server, &path, item)? else {
                unclassified = true;
                continue;
            };
            diagnostics.insert(identity);
        }
    }
    let mut unknown_reasons = unknown_reasons.clone();
    if unclassified {
        unknown_reasons.insert(UnknownReason::UnclassifiedDiagnostic);
    }
    let completeness = if unknown_reasons.is_empty() {
        completeness
    } else {
        BaselineCompleteness::Incomplete
    };
    let snapshot = SemanticDiagnosticSnapshot {
        world: world.id.clone(),
        document_versions,
        document_sha256,
        completeness,
        diagnostics,
        unknown_reasons,
    };
    snapshot.validate()?;
    Ok(snapshot)
}

fn diagnostic_identity(
    server: &str,
    path: &LogicalPath,
    diagnostic: &Value,
) -> Result<Option<DiagnosticIdentity>> {
    let Some(start) = diagnostic.pointer("/range/start") else {
        return Ok(None);
    };
    let Some(end) = diagnostic.pointer("/range/end") else {
        return Ok(None);
    };
    let Some(start_line) = start.get("line").and_then(Value::as_u64) else {
        return Ok(None);
    };
    let Some(start_character) = start.get("character").and_then(Value::as_u64) else {
        return Ok(None);
    };
    let Some(end_line) = end.get("line").and_then(Value::as_u64) else {
        return Ok(None);
    };
    let Some(end_character) = end.get("character").and_then(Value::as_u64) else {
        return Ok(None);
    };
    let code = diagnostic
        .get("code")
        .map(|value| match value {
            Value::String(value) => value.clone(),
            value => value.to_string(),
        })
        .unwrap_or_default();
    let source = diagnostic
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let class = classify_diagnostic(server, &code);
    let mut provenance = Sha256::new();
    provenance.update(server.as_bytes());
    provenance.update([0]);
    provenance.update(source.as_bytes());
    provenance.update([0]);
    provenance.update(code.as_bytes());
    Ok(Some(DiagnosticIdentity {
        path: path.clone(),
        start_line: u32::try_from(start_line).context("diagnostic line exceeds u32")?,
        start_character: u32::try_from(start_character)
            .context("diagnostic character exceeds u32")?,
        end_line: u32::try_from(end_line).context("diagnostic line exceeds u32")?,
        end_character: u32::try_from(end_character).context("diagnostic character exceeds u32")?,
        class,
        provenance_sha256: format!("{:x}", provenance.finalize()),
    }))
}

fn classify_diagnostic(server: &str, code: &str) -> DefiniteErrorClass {
    let normalized_server = server.to_ascii_lowercase();
    let normalized = code.to_ascii_lowercase();
    if normalized_server.contains("rust") {
        match normalized.as_str() {
            "e0308" | "type-mismatch" => DefiniteErrorClass::TypeMismatch,
            "e0425" | "e0433" | "unresolved-ident" | "unresolved-macro-call" => {
                DefiniteErrorClass::UnresolvedName
            }
            "e0432" | "unresolved-import" => DefiniteErrorClass::UnresolvedImport,
            "e0599" | "unresolved-method" => DefiniteErrorClass::MissingMethod,
            "e0609" | "unresolved-field" => DefiniteErrorClass::MissingField,
            "e0603" | "private-assoc-item" | "private-field" => DefiniteErrorClass::Privacy,
            "e0382" | "e0505" | "e0507" => DefiniteErrorClass::Ownership,
            "e0596" => DefiniteErrorClass::Mutability,
            _ => DefiniteErrorClass::Other,
        }
    } else if normalized_server.contains("typescript") || normalized_server.contains("tsserver") {
        match normalized.as_str() {
            "2304" => DefiniteErrorClass::UnresolvedName,
            "2307" => DefiniteErrorClass::UnresolvedImport,
            "2339" => DefiniteErrorClass::MissingField,
            "2322" | "2362" | "2363" => DefiniteErrorClass::TypeMismatch,
            "2345" | "2554" => DefiniteErrorClass::InvalidCall,
            _ => DefiniteErrorClass::Other,
        }
    } else if normalized_server.contains("pyright") {
        match normalized.as_str() {
            "reportundefinedvariable" => DefiniteErrorClass::UnresolvedName,
            "reportmissingimports" | "reportmissingmodulesource" => {
                DefiniteErrorClass::UnresolvedImport
            }
            "reportattributeaccessissue" => DefiniteErrorClass::MissingField,
            "reportargumenttype" | "reportcallissue" => DefiniteErrorClass::InvalidCall,
            "reportassignmenttype"
            | "reportoperatortype"
            | "reportoperatorissue"
            | "reportreturntype" => DefiniteErrorClass::TypeMismatch,
            _ => DefiniteErrorClass::Other,
        }
    } else {
        DefiniteErrorClass::Other
    }
}

fn provider_identity(config: &crate::lsp::LspServerConfig) -> (String, bool) {
    if let Some(image) = config.container_image.as_deref() {
        if let Some(digest) = config
            .verified_manifest_digest
            .as_deref()
            .filter(|digest| digest.starts_with("sha256:") && digest.len() == 71)
        {
            return (digest.to_string(), true);
        }
        if let Some((_, digest)) = image.rsplit_once('@')
            && digest.starts_with("sha256:")
            && digest.len() == 71
        {
            return (digest.to_string(), true);
        }
    }
    let digest = serde_json::to_vec(config)
        .map(|bytes| format!("config:{:x}", Sha256::digest(bytes)))
        .unwrap_or_else(|_| "config:unavailable".to_string());
    (digest, false)
}

fn provider_language(
    config: &crate::lsp::LspServerConfig,
    mutations: &[SemanticOverlayMutation],
) -> String {
    mutations
        .iter()
        .find_map(|mutation| {
            Path::new(&mutation.path)
                .extension()
                .and_then(|extension| extension.to_str())
                .and_then(|extension| match extension {
                    "rs" => Some("rust"),
                    "ts" | "tsx" => Some("typescript"),
                    "js" | "jsx" | "mjs" | "cjs" => Some("javascript"),
                    "py" | "pyi" => Some("python"),
                    _ => None,
                })
        })
        .map(str::to_string)
        .or_else(|| config.language_ids.first().cloned())
        .unwrap_or_else(|| "unknown".to_string())
}

fn capabilities_for_language(language: &str) -> BTreeSet<AnalyzerCapability> {
    let mut capabilities = BTreeSet::from([
        AnalyzerCapability::SymbolResolution,
        AnalyzerCapability::TypeChecking,
        AnalyzerCapability::DependencyResolution,
    ]);
    if language == "rust" {
        capabilities.insert(AnalyzerCapability::OwnershipChecking);
    }
    capabilities
}

fn dependency_identity(snapshot: &crate::workspace::ContentSnapshot) -> String {
    const DEPENDENCY_FILES: &[&str] = &[
        "Cargo.toml",
        "Cargo.lock",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "deno.json",
        "deno.jsonc",
        "deno.lock",
        "pyproject.toml",
        "uv.lock",
        "poetry.lock",
        "requirements.txt",
    ];
    let mut digest = Sha256::new();
    for path in DEPENDENCY_FILES {
        if let Some(content) = snapshot.paths.get(*path) {
            digest.update((path.len() as u64).to_le_bytes());
            digest.update(path.as_bytes());
            digest.update(content.fingerprint.as_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(closure: ClosureVerdict) -> ProviderVerdict {
        ProviderVerdict {
            viability: Viability::Unknown,
            closure,
            definite_errors: Vec::new(),
            unknown_reasons: Vec::new(),
            obligations: Vec::new(),
            biases: Vec::new(),
        }
    }

    #[test]
    fn boundary_policy_blocks_only_required_non_allow_results() {
        assert_eq!(
            boundary_enforcement_verdict(verdict(ClosureVerdict::Reject), false).closure,
            ClosureVerdict::Defer
        );
        assert_eq!(
            boundary_enforcement_verdict(verdict(ClosureVerdict::Defer), true).closure,
            ClosureVerdict::Reject
        );
        assert_eq!(
            boundary_enforcement_verdict(verdict(ClosureVerdict::Allow), true).closure,
            ClosureVerdict::Allow
        );
    }

    #[test]
    fn diagnostic_codes_map_without_inspecting_provider_messages() {
        assert_eq!(
            classify_diagnostic("rust-analyzer", "E0308"),
            DefiniteErrorClass::TypeMismatch
        );
        assert_eq!(
            classify_diagnostic("typescript", "2307"),
            DefiniteErrorClass::UnresolvedImport
        );
        assert_eq!(
            classify_diagnostic("pyright", "reportOperatorIssue"),
            DefiniteErrorClass::TypeMismatch
        );
        assert_eq!(
            classify_diagnostic("unknown", "mystery"),
            DefiniteErrorClass::Other
        );
    }

    #[test]
    fn host_commands_are_not_misrepresented_as_version_pinned() {
        let config = crate::lsp::LspServerConfig {
            command: Some("rust-analyzer".to_string()),
            ..Default::default()
        };
        let (_, pinned) = provider_identity(&config);
        assert!(!pinned);
    }

    #[test]
    fn broker_without_a_matching_provider_returns_unknown() {
        let directory = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(directory.path())
            .status()
            .unwrap();
        let registry = LspToolRegistry::default();
        let broker = SemanticProviderBroker::new(&registry, directory.path())
            .with_timeout(Duration::from_millis(100));
        let report = broker
            .analyze_transaction(&[SemanticOverlayMutation {
                path: "src/lib.rs".to_string(),
                base: None,
                result: Some("pub fn value() {}\n".to_string()),
            }])
            .unwrap();
        assert_eq!(report.verdict.viability, Viability::Unknown);
        assert!(
            report
                .verdict
                .unknown_reasons
                .contains(&UnknownReason::ProviderUnavailable)
        );
    }

    #[test]
    fn pull_provider_rejects_new_type_diagnostic_on_exact_candidate_overlay() {
        let directory = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(directory.path())
            .status()
            .unwrap();
        std::fs::create_dir(directory.path().join("src")).unwrap();
        let base = "pub fn value() -> i32 { 1 }\n";
        std::fs::write(directory.path().join("src/lib.rs"), base).unwrap();
        let log_directory = tempfile::tempdir().unwrap();
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_lsp.py");
        let registry = crate::lsp::discover_tools(
            BTreeMap::from([(
                "rust-analyzer".to_string(),
                crate::lsp::LspServerConfig {
                    command: Some("python3".to_string()),
                    args: vec![
                        fixture.to_string_lossy().into_owned(),
                        log_directory
                            .path()
                            .join("semantic.jsonl")
                            .to_string_lossy()
                            .into_owned(),
                    ],
                    language_ids: vec!["rust".to_string()],
                    semantic_enforcement: crate::lsp::LspSemanticEnforcement::Required,
                    ..Default::default()
                },
            )]),
            directory.path(),
        );
        let candidate = "pub fn value() -> i32 { TYPE_ERROR }\n";
        let report = SemanticProviderBroker::new(&registry, directory.path())
            .analyze_transaction(&[SemanticOverlayMutation {
                path: "src/lib.rs".to_string(),
                base: Some(base.to_string()),
                result: Some(candidate.to_string()),
            }])
            .unwrap();

        assert_eq!(report.verdict.closure, ClosureVerdict::Reject);
        assert_eq!(
            report.verdict.definite_errors,
            vec![DefiniteErrorClass::TypeMismatch]
        );
        assert_eq!(report.providers.len(), 1);
        let provider = &report.providers[0];
        provider.world.validate().unwrap();
        assert!(!provider.delta.authoritative);
        assert_eq!(provider.delta.introduced.len(), 1);
        assert_eq!(
            provider.baseline.document_sha256[&LogicalPath::parse("src/lib.rs").unwrap()],
            Digest::of(base.as_bytes())
        );
        assert_eq!(
            provider.candidate.document_sha256[&LogicalPath::parse("src/lib.rs").unwrap()],
            Digest::of(candidate.as_bytes())
        );
    }
}
