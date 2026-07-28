use std::collections::{BTreeMap, BTreeSet};

use pb_control_collar::{
    CollarError, CollarResult,
    analysis::{
        Analysis, AnalysisBoundary, AnalyzerCheckpoint, AnalyzerLayerDescriptor, ClosureVerdict,
        IncrementalAnalyzer, LayerReadiness, LayerReadinessReceipt, ProgramSnapshot, RepairIntent,
        SemanticObligation, SourceEvent, SourceOrigin, Viability,
    },
    mutation::{LogicalPath, MutationKind},
};
use tree_sitter::{InputEdit, Node, Parser, Point, Tree};

use crate::{
    RustCallableShape, RustImportResolution, RustProjectRequestWorld, RustRequestWorld,
    RustSemanticCertainty, RustSymbolKind, RustSymbolShape, RustTypeShape, RustUnknownReason,
};

pub type RustLayerCheckpoint = AnalyzerCheckpoint;

/// Project-wide request layer. It binds each streamed path to rust-analyzer's existing
/// file-to-module mapping and otherwise uses a package scope only when every plausible target has
/// the same dependency surface. Ambiguity yields Unknown; it never guesses and rejects.
pub struct RustWorkspaceStreamingLayer {
    project: RustProjectRequestWorld,
    active: WorkspaceActive,
    // One append-only stream per target/path keeps rollback checkpoints source-length-only. This
    // avoids copying a large known file once per accepted token when a patch edits near its tail.
    layers: Vec<(ra_ap_hir::Crate, LogicalPath, Box<RustStreamingLayer>)>,
    snapshots: Vec<WorkspaceStateSnapshot>,
    epoch: u64,
    max_source_bytes: usize,
    max_checkpoints: usize,
    last_analysis: Analysis,
}

#[derive(Clone, Copy)]
enum WorkspaceActive {
    None,
    Target(usize),
    Unknown,
}

struct WorkspaceStateSnapshot {
    active: WorkspaceActive,
    layer_checkpoints: Vec<AnalyzerCheckpoint>,
    layer_count: usize,
    last_analysis: Analysis,
}

impl RustWorkspaceStreamingLayer {
    pub fn new(
        project: RustProjectRequestWorld,
        max_source_bytes: usize,
        max_checkpoints: usize,
    ) -> CollarResult<Self> {
        if max_source_bytes == 0 || max_checkpoints == 0 {
            return Err(CollarError::Analysis(
                "Rust workspace streaming limits must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            project,
            active: WorkspaceActive::None,
            layers: Vec::new(),
            snapshots: Vec::new(),
            epoch: 0,
            max_source_bytes,
            max_checkpoints,
            last_analysis: repairable_analysis(),
        })
    }

    fn begin_file(
        &mut self,
        path: &LogicalPath,
        language: &pb_control_collar::analysis::LanguageId,
        mutation: MutationKind,
    ) -> CollarResult<Analysis> {
        let Some(target) = self.project.target_for_path(path) else {
            self.active = WorkspaceActive::Unknown;
            self.last_analysis = unknown_target_analysis(AnalysisBoundary::File);
            return Ok(self.last_analysis.clone());
        };
        let index = if let Some(index) =
            self.layers
                .iter()
                .position(|(candidate, candidate_path, _)| {
                    *candidate == target && candidate_path == path
                }) {
            index
        } else {
            let mut layer = RustStreamingLayer::new(
                self.project.request_for_target(target),
                self.max_source_bytes,
                self.max_checkpoints,
            )?;
            layer.begin(ProgramSnapshot::default())?;
            self.layers.push((target, path.clone(), Box::new(layer)));
            self.layers.len().saturating_sub(1)
        };
        let layer = &mut self.layers[index].2;
        let analysis = layer.apply(SourceEvent::BeginFile {
            path,
            language,
            mutation,
        })?;
        self.active = WorkspaceActive::Target(index);
        self.last_analysis = analysis.clone();
        Ok(analysis)
    }
}

impl IncrementalAnalyzer for RustWorkspaceStreamingLayer {
    fn descriptor(&self) -> &AnalyzerLayerDescriptor {
        self.project.descriptor()
    }

    fn readiness(&self) -> LayerReadiness {
        LayerReadiness::Ready
    }

    fn readiness_receipt(&self) -> Option<&LayerReadinessReceipt> {
        Some(self.project.readiness_receipt())
    }

    fn begin(&mut self, snapshot: ProgramSnapshot) -> CollarResult<()> {
        let rust_bytes = snapshot
            .files
            .iter()
            .filter(|file| file.language == self.descriptor().language)
            .try_fold(0usize, |total, file| total.checked_add(file.bytes.len()))
            .ok_or_else(|| {
                CollarError::Analysis("Rust program snapshot byte count overflowed".to_string())
            })?;
        if rust_bytes > self.max_source_bytes {
            return Err(CollarError::Analysis(format!(
                "Rust program snapshot exceeds the {}-byte request limit",
                self.max_source_bytes
            )));
        }
        self.active = WorkspaceActive::None;
        self.layers.clear();
        self.snapshots.clear();
        self.epoch = self.epoch.wrapping_add(1);
        self.last_analysis = repairable_analysis();
        Ok(())
    }

    fn checkpoint(&mut self) -> CollarResult<AnalyzerCheckpoint> {
        if self.snapshots.len() >= self.max_checkpoints {
            return Err(CollarError::Analysis(format!(
                "Rust workspace stream exceeds the {}-checkpoint limit",
                self.max_checkpoints
            )));
        }
        let layer_checkpoints = self
            .layers
            .iter_mut()
            .map(|(_, _, layer)| layer.checkpoint())
            .collect::<CollarResult<Vec<_>>>()?;
        self.snapshots.push(WorkspaceStateSnapshot {
            active: self.active,
            layer_count: self.layers.len(),
            layer_checkpoints,
            last_analysis: self.last_analysis.clone(),
        });
        Ok(AnalyzerCheckpoint {
            epoch: self.epoch,
            revision: u64::try_from(self.snapshots.len().saturating_sub(1)).unwrap_or(u64::MAX),
        })
    }

    fn apply(&mut self, event: SourceEvent<'_>) -> CollarResult<Analysis> {
        if let SourceEvent::BeginFile {
            path,
            language,
            mutation,
        } = event
        {
            if language != &self.descriptor().language {
                return Err(CollarError::Analysis(
                    "Rust workspace layer received a non-Rust file".to_string(),
                ));
            }
            return self.begin_file(path, language, mutation);
        }
        let analysis = match self.active {
            WorkspaceActive::Target(index) => self
                .layers
                .get_mut(index)
                .ok_or_else(|| {
                    CollarError::Analysis("active Rust target layer is missing".to_string())
                })?
                .2
                .apply(event)?,
            WorkspaceActive::Unknown => match event {
                SourceEvent::Boundary(boundary) => unknown_target_analysis(boundary),
                SourceEvent::EndFile => unknown_target_analysis(AnalysisBoundary::File),
                SourceEvent::Bytes { .. } | SourceEvent::DeleteKnownBytes(_) => {
                    self.last_analysis.clone()
                }
                SourceEvent::BeginFile { .. } => unreachable!(),
            },
            WorkspaceActive::None => {
                return Err(CollarError::Analysis(
                    "Rust workspace event arrived before BeginFile".to_string(),
                ));
            }
        };
        self.last_analysis = analysis.clone();
        Ok(analysis)
    }

    fn rollback(&mut self, checkpoint: AnalyzerCheckpoint) -> CollarResult<()> {
        if checkpoint.epoch != self.epoch {
            return Err(CollarError::Analysis(
                "Rust workspace checkpoint belongs to another request".to_string(),
            ));
        }
        let revision = usize::try_from(checkpoint.revision).map_err(|_| {
            CollarError::Analysis("Rust workspace checkpoint does not fit usize".to_string())
        })?;
        let snapshot = self.snapshots.get(revision).ok_or_else(|| {
            CollarError::Analysis("Rust workspace checkpoint is not in this stream".to_string())
        })?;
        let active = snapshot.active;
        let layer_count = snapshot.layer_count;
        let layer_checkpoints = snapshot.layer_checkpoints.clone();
        let last_analysis = snapshot.last_analysis.clone();
        if layer_count > self.layers.len() || layer_checkpoints.len() != layer_count {
            return Err(CollarError::Analysis(
                "Rust workspace checkpoint has an invalid target-layer count".to_string(),
            ));
        }
        for ((_, _, layer), inner) in self.layers[..layer_count].iter_mut().zip(layer_checkpoints) {
            layer.rollback(inner)?;
        }
        if matches!(active, WorkspaceActive::Target(index) if index >= layer_count) {
            return Err(CollarError::Analysis(
                "Rust workspace checkpoint has an invalid active target".to_string(),
            ));
        }
        self.layers.truncate(layer_count);
        self.active = active;
        self.last_analysis = last_analysis;
        self.snapshots.truncate(revision.saturating_add(1));
        Ok(())
    }

    fn finalize(&mut self) -> CollarResult<Analysis> {
        let analysis = match self.active {
            WorkspaceActive::Target(index) => self
                .layers
                .get_mut(index)
                .ok_or_else(|| {
                    CollarError::Analysis("active Rust target layer is missing".to_string())
                })?
                .2
                .finalize()?,
            WorkspaceActive::Unknown => unknown_target_analysis(AnalysisBoundary::ToolCall),
            WorkspaceActive::None => repairable_analysis(),
        };
        self.last_analysis = analysis.clone();
        Ok(analysis)
    }
}

/// Request-local forward parser layered over one warm rust-analyzer world. The generated prefix is
/// parsed here; rust-analyzer remains an immutable HIR resolver for the coherent base project.
pub struct RustStreamingLayer {
    world: RustRequestWorld,
    parser: Parser,
    active: Option<RustFileState>,
    snapshots: Vec<RustStateSnapshot>,
    epoch: u64,
    max_source_bytes: usize,
    max_checkpoints: usize,
    last_analysis: Analysis,
}

impl RustStreamingLayer {
    pub fn new(
        world: RustRequestWorld,
        max_source_bytes: usize,
        max_checkpoints: usize,
    ) -> CollarResult<Self> {
        if max_source_bytes == 0 || max_checkpoints == 0 {
            return Err(CollarError::Analysis(
                "Rust streaming limits must be non-zero".to_string(),
            ));
        }
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .map_err(|error| {
                CollarError::Analysis(format!("failed to load pinned Rust grammar: {error}"))
            })?;
        Ok(Self {
            world,
            parser,
            active: None,
            snapshots: Vec::new(),
            epoch: 0,
            max_source_bytes,
            max_checkpoints,
            last_analysis: repairable_analysis(),
        })
    }

    fn begin_file(&mut self, path: &LogicalPath) -> CollarResult<Analysis> {
        let tree = self.parser.parse(&[] as &[u8], None).ok_or_else(|| {
            CollarError::Analysis("pinned Rust parser returned no initial tree".to_string())
        })?;
        self.active = Some(RustFileState {
            path: path.clone(),
            source: Vec::new(),
            generated_ranges: Vec::new(),
            generated_points: Vec::new(),
            tree,
            checked_uses: BTreeSet::new(),
            checked_expressions: BTreeSet::new(),
            imports: BTreeMap::new(),
            violations: BTreeSet::new(),
            unknowns: BTreeSet::new(),
        });
        self.last_analysis = repairable_analysis();
        Ok(self.last_analysis.clone())
    }

    fn push_bytes(&mut self, origin: SourceOrigin, bytes: &[u8]) -> CollarResult<Analysis> {
        let state = self.active.as_mut().ok_or_else(|| {
            CollarError::Analysis("Rust bytes arrived before BeginFile".to_string())
        })?;
        let old_len = state.source.len();
        let new_len = old_len
            .checked_add(bytes.len())
            .ok_or_else(|| CollarError::Analysis("Rust source length overflowed".to_string()))?;
        if new_len > self.max_source_bytes {
            return Err(CollarError::Analysis(format!(
                "Rust source exceeds the {}-byte limit",
                self.max_source_bytes
            )));
        }
        state.source.extend_from_slice(bytes);
        if origin == SourceOrigin::Generated && old_len != new_len {
            if let Some((_, end)) = state.generated_ranges.last_mut()
                && *end == old_len
            {
                *end = new_len;
            } else {
                state.generated_ranges.push((old_len, new_len));
            }
        }
        let mut edited = state.tree.clone();
        let old_point = end_point(&state.source[..old_len]);
        edited.edit(&InputEdit {
            start_byte: old_len,
            old_end_byte: old_len,
            new_end_byte: new_len,
            start_position: old_point,
            old_end_position: old_point,
            new_end_position: end_point(&state.source),
        });
        state.tree = self
            .parser
            .parse(&state.source, Some(&edited))
            .ok_or_else(|| {
                CollarError::Analysis("pinned Rust parser returned no updated tree".to_string())
            })?;
        self.last_analysis = repairable_analysis();
        Ok(self.last_analysis.clone())
    }

    fn analyze_boundary(&mut self, boundary: AnalysisBoundary) -> CollarResult<Analysis> {
        let state = self.active.as_mut().ok_or_else(|| {
            CollarError::Analysis("Rust boundary arrived before BeginFile".to_string())
        })?;
        let pending = collect_complete_uses(
            &state.tree,
            &state.source,
            &state.generated_ranges,
            &state.generated_points,
            &state.checked_uses,
        );
        for use_declaration in pending {
            state.checked_uses.insert(use_declaration.range);
            let paths = match use_declaration.paths {
                Ok(paths) => paths,
                Err(()) => {
                    state.unknowns.insert(RustStreamUnknown::UnsupportedImport);
                    continue;
                }
            };
            for path in paths {
                let borrowed = path.segments.iter().map(String::as_str).collect::<Vec<_>>();
                match self.world.resolve_import(&borrowed) {
                    RustImportResolution::Resolved(shape) => {
                        if let Some(binding) = path.binding {
                            state.imports.insert(binding, shape);
                        }
                    }
                    RustImportResolution::Absent => {
                        state.violations.insert(RustViolation::InvalidImport);
                    }
                    RustImportResolution::Unknown(reason) => {
                        state.unknowns.insert(match reason {
                            RustUnknownReason::PartialScope => {
                                RustStreamUnknown::PartialImportScope
                            }
                            RustUnknownReason::UnsupportedPath => {
                                RustStreamUnknown::UnsupportedImport
                            }
                        });
                    }
                }
            }
        }
        let expressions = evaluate_complete_expressions(
            &self.world,
            &state.tree,
            &state.source,
            &state.imports,
            &state.generated_ranges,
            &state.generated_points,
            &state.checked_expressions,
        );
        for (range, verdict) in expressions {
            state.checked_expressions.insert(range);
            match verdict {
                ExpressionVerdict::Valid => {}
                ExpressionVerdict::Violation(violation) => {
                    state.violations.insert(violation);
                }
                ExpressionVerdict::Unknown => {
                    state.unknowns.insert(RustStreamUnknown::UnknownTypeShape);
                }
            }
        }
        self.last_analysis = state.analysis(boundary);
        Ok(self.last_analysis.clone())
    }
}

impl IncrementalAnalyzer for RustStreamingLayer {
    fn descriptor(&self) -> &AnalyzerLayerDescriptor {
        self.world.descriptor()
    }

    fn readiness(&self) -> LayerReadiness {
        LayerReadiness::Ready
    }

    fn readiness_receipt(&self) -> Option<&LayerReadinessReceipt> {
        Some(self.world.readiness_receipt())
    }

    fn begin(&mut self, snapshot: ProgramSnapshot) -> CollarResult<()> {
        let rust_bytes = snapshot
            .files
            .iter()
            .filter(|file| file.language == self.descriptor().language)
            .try_fold(0usize, |total, file| total.checked_add(file.bytes.len()))
            .ok_or_else(|| {
                CollarError::Analysis("Rust program snapshot byte count overflowed".to_string())
            })?;
        if rust_bytes > self.max_source_bytes {
            return Err(CollarError::Analysis(format!(
                "Rust program snapshot exceeds the {}-byte request limit",
                self.max_source_bytes
            )));
        }
        self.active = None;
        self.snapshots.clear();
        self.epoch = self.epoch.wrapping_add(1);
        self.last_analysis = repairable_analysis();
        Ok(())
    }

    fn checkpoint(&mut self) -> CollarResult<AnalyzerCheckpoint> {
        if self.snapshots.len() >= self.max_checkpoints {
            return Err(CollarError::Analysis(format!(
                "Rust stream exceeds the {}-checkpoint limit",
                self.max_checkpoints
            )));
        }
        let state = self.active.as_ref().map(RustFileState::snapshot);
        self.snapshots.push(RustStateSnapshot {
            state,
            last_analysis: self.last_analysis.clone(),
        });
        Ok(AnalyzerCheckpoint {
            epoch: self.epoch,
            revision: u64::try_from(self.snapshots.len().saturating_sub(1)).unwrap_or(u64::MAX),
        })
    }

    fn apply(&mut self, event: SourceEvent<'_>) -> CollarResult<Analysis> {
        match event {
            SourceEvent::BeginFile {
                path,
                language,
                mutation: _,
            } => {
                if language != &self.descriptor().language {
                    return Err(CollarError::Analysis(
                        "Rust layer received a non-Rust file".to_string(),
                    ));
                }
                self.begin_file(path)
            }
            SourceEvent::Bytes { origin, bytes } => self.push_bytes(origin, bytes),
            SourceEvent::DeleteKnownBytes(bytes) => {
                let state = self.active.as_mut().ok_or_else(|| {
                    CollarError::Analysis("Rust deletion arrived before BeginFile".to_string())
                })?;
                if !bytes.is_empty()
                    && state.generated_points.last().copied() != Some(state.source.len())
                {
                    state.generated_points.push(state.source.len());
                }
                Ok(self.last_analysis.clone())
            }
            SourceEvent::Boundary(boundary) => self.analyze_boundary(boundary),
            SourceEvent::EndFile => self.analyze_boundary(AnalysisBoundary::File),
        }
    }

    fn rollback(&mut self, checkpoint: AnalyzerCheckpoint) -> CollarResult<()> {
        if checkpoint.epoch != self.epoch {
            return Err(CollarError::Analysis(
                "Rust checkpoint belongs to another file epoch".to_string(),
            ));
        }
        let revision = usize::try_from(checkpoint.revision).map_err(|_| {
            CollarError::Analysis("Rust checkpoint revision does not fit usize".to_string())
        })?;
        let snapshot = self.snapshots.get(revision).ok_or_else(|| {
            CollarError::Analysis("Rust checkpoint is not part of this stream".to_string())
        })?;
        match (&mut self.active, &snapshot.state) {
            (Some(state), Some(snapshot)) => state.restore(snapshot)?,
            (active, None) => *active = None,
            (None, Some(_)) => {
                return Err(CollarError::Analysis(
                    "Rust checkpoint lost its active file state".to_string(),
                ));
            }
        }
        self.last_analysis = snapshot.last_analysis.clone();
        self.snapshots.truncate(revision.saturating_add(1));
        Ok(())
    }

    fn finalize(&mut self) -> CollarResult<Analysis> {
        self.analyze_boundary(AnalysisBoundary::ToolCall)
    }
}

struct RustFileState {
    #[allow(dead_code)]
    path: LogicalPath,
    source: Vec<u8>,
    generated_ranges: Vec<(usize, usize)>,
    generated_points: Vec<usize>,
    tree: Tree,
    checked_uses: BTreeSet<(usize, usize)>,
    checked_expressions: BTreeSet<(usize, usize)>,
    imports: BTreeMap<String, RustSymbolShape>,
    violations: BTreeSet<RustViolation>,
    unknowns: BTreeSet<RustStreamUnknown>,
}

impl RustFileState {
    fn snapshot(&self) -> RustFileSnapshot {
        RustFileSnapshot {
            path: self.path.clone(),
            source_len: self.source.len(),
            generated_ranges: self.generated_ranges.clone(),
            generated_points: self.generated_points.clone(),
            tree: self.tree.clone(),
            checked_uses: self.checked_uses.clone(),
            checked_expressions: self.checked_expressions.clone(),
            imports: self.imports.clone(),
            violations: self.violations.clone(),
            unknowns: self.unknowns.clone(),
        }
    }

    fn restore(&mut self, snapshot: &RustFileSnapshot) -> CollarResult<()> {
        if self.path != snapshot.path || snapshot.source_len > self.source.len() {
            return Err(CollarError::Analysis(
                "Rust checkpoint does not match the active append-only source".to_string(),
            ));
        }
        self.source.truncate(snapshot.source_len);
        self.generated_ranges.clone_from(&snapshot.generated_ranges);
        self.generated_points.clone_from(&snapshot.generated_points);
        self.tree = snapshot.tree.clone();
        self.checked_uses.clone_from(&snapshot.checked_uses);
        self.checked_expressions
            .clone_from(&snapshot.checked_expressions);
        self.imports.clone_from(&snapshot.imports);
        self.violations.clone_from(&snapshot.violations);
        self.unknowns.clone_from(&snapshot.unknowns);
        Ok(())
    }

    fn analysis(&self, boundary: AnalysisBoundary) -> Analysis {
        if !self.violations.is_empty() {
            return Analysis {
                viability: Viability::Impossible,
                closure: ClosureVerdict::Reject,
                obligations: self
                    .violations
                    .iter()
                    .map(|violation| SemanticObligation {
                        kind: violation.kind().to_string(),
                        boundary,
                    })
                    .collect(),
                biases: Vec::new(),
            };
        }
        if !self.unknowns.is_empty() {
            return Analysis {
                viability: Viability::Unknown,
                closure: ClosureVerdict::Defer,
                obligations: self
                    .unknowns
                    .iter()
                    .map(|unknown| SemanticObligation {
                        kind: unknown.kind().to_string(),
                        boundary,
                    })
                    .collect(),
                biases: Vec::new(),
            };
        }
        Analysis {
            viability: Viability::Valid,
            closure: ClosureVerdict::Allow,
            obligations: Vec::new(),
            biases: Vec::new(),
        }
    }
}

struct RustFileSnapshot {
    path: LogicalPath,
    // Source bytes remain in the owning append-only stream. Descendant candidate branches can be
    // rolled back by truncation because the checkpoint cache discards non-ancestor branches.
    source_len: usize,
    generated_ranges: Vec<(usize, usize)>,
    generated_points: Vec<usize>,
    tree: Tree,
    checked_uses: BTreeSet<(usize, usize)>,
    checked_expressions: BTreeSet<(usize, usize)>,
    imports: BTreeMap<String, RustSymbolShape>,
    violations: BTreeSet<RustViolation>,
    unknowns: BTreeSet<RustStreamUnknown>,
}

struct RustStateSnapshot {
    state: Option<RustFileSnapshot>,
    last_analysis: Analysis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RustViolation {
    InvalidImport,
    InvalidCall,
    TypeMismatch,
}

impl RustViolation {
    fn kind(self) -> &'static str {
        match self {
            Self::InvalidImport => "rust_invalid_import",
            Self::InvalidCall => "rust_invalid_call",
            Self::TypeMismatch => "rust_type_mismatch",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RustStreamUnknown {
    PartialImportScope,
    UnsupportedImport,
    UnknownTypeShape,
}

impl RustStreamUnknown {
    fn kind(self) -> &'static str {
        match self {
            Self::PartialImportScope => "rust_partial_import_scope",
            Self::UnsupportedImport => "rust_unsupported_import",
            Self::UnknownTypeShape => "rust_unknown_type_shape",
        }
    }
}

struct ParsedUse {
    range: (usize, usize),
    paths: Result<Vec<UsePath>, ()>,
}

struct UsePath {
    segments: Vec<String>,
    binding: Option<String>,
}

fn collect_complete_uses(
    tree: &Tree,
    source: &[u8],
    generated_ranges: &[(usize, usize)],
    generated_points: &[usize],
    checked: &BTreeSet<(usize, usize)>,
) -> Vec<ParsedUse> {
    let mut pending = Vec::new();
    visit_nodes(tree.root_node(), &mut |node| {
        if node.kind() != "use_declaration" || node.has_error() {
            return;
        }
        let range = (node.start_byte(), node.end_byte());
        if checked.contains(&range)
            || !range_was_generated(range, generated_ranges, generated_points)
            || !source
                .get(range.0..range.1)
                .is_some_and(|bytes| bytes.trim_ascii_end().ends_with(b";"))
        {
            return;
        }
        let paths = node
            .child_by_field_name("argument")
            .ok_or(())
            .and_then(|argument| collect_use_argument(argument, &[], source));
        pending.push(ParsedUse { range, paths });
    });
    pending
}

fn visit_nodes(node: Node<'_>, visit: &mut impl FnMut(Node<'_>)) {
    visit(node);
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        visit_nodes(child, visit);
    }
}

fn collect_use_argument(
    node: Node<'_>,
    prefix: &[String],
    source: &[u8],
) -> Result<Vec<UsePath>, ()> {
    match node.kind() {
        "identifier" | "crate" | "self" | "super" => {
            let segment = node_text(node, source)?;
            if segment == "self" && !prefix.is_empty() {
                return Ok(vec![UsePath {
                    segments: prefix.to_vec(),
                    binding: prefix.last().cloned(),
                }]);
            }
            let mut segments = prefix.to_vec();
            segments.push(segment.to_string());
            Ok(vec![UsePath {
                binding: segments.last().cloned(),
                segments,
            }])
        }
        "scoped_identifier" => {
            let mut segments = prefix.to_vec();
            segments.extend(simple_path(node, source)?);
            Ok(vec![UsePath {
                binding: segments.last().cloned(),
                segments,
            }])
        }
        "use_as_clause" => {
            let path = node.child_by_field_name("path").ok_or(())?;
            let alias = node_text(node.child_by_field_name("alias").ok_or(())?, source)?;
            let mut paths = collect_use_argument(path, prefix, source)?;
            for path in &mut paths {
                path.binding = (alias != "_").then(|| alias.to_string());
            }
            Ok(paths)
        }
        "scoped_use_list" => {
            let mut nested_prefix = prefix.to_vec();
            if let Some(path) = node.child_by_field_name("path") {
                nested_prefix.extend(simple_path(path, source)?);
            }
            collect_use_argument(
                node.child_by_field_name("list").ok_or(())?,
                &nested_prefix,
                source,
            )
        }
        "use_list" => {
            let mut paths = Vec::new();
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                paths.extend(collect_use_argument(child, prefix, source)?);
            }
            Ok(paths)
        }
        "use_wildcard" => {
            let mut segments = prefix.to_vec();
            let mut cursor = node.walk();
            if let Some(path) = node.named_children(&mut cursor).next() {
                segments.extend(simple_path(path, source)?);
            }
            if segments.is_empty() {
                return Err(());
            }
            Ok(vec![UsePath {
                segments,
                binding: None,
            }])
        }
        _ => Err(()),
    }
}

fn simple_path(node: Node<'_>, source: &[u8]) -> Result<Vec<String>, ()> {
    match node.kind() {
        "identifier" | "crate" | "self" | "super" => Ok(vec![node_text(node, source)?.to_string()]),
        "scoped_identifier" => {
            let mut segments = Vec::new();
            if let Some(path) = node.child_by_field_name("path") {
                segments.extend(simple_path(path, source)?);
            }
            segments.extend(simple_path(
                node.child_by_field_name("name").ok_or(())?,
                source,
            )?);
            Ok(segments)
        }
        _ => Err(()),
    }
}

fn node_text<'a>(node: Node<'_>, source: &'a [u8]) -> Result<&'a str, ()> {
    std::str::from_utf8(source.get(node.byte_range()).ok_or(())?).map_err(|_| ())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpressionVerdict {
    Valid,
    Violation(RustViolation),
    Unknown,
}

fn evaluate_complete_expressions(
    world: &RustRequestWorld,
    tree: &Tree,
    source: &[u8],
    imports: &BTreeMap<String, RustSymbolShape>,
    generated_ranges: &[(usize, usize)],
    generated_points: &[usize],
    checked: &BTreeSet<(usize, usize)>,
) -> Vec<((usize, usize), ExpressionVerdict)> {
    let mut evaluated = Vec::new();
    visit_nodes(tree.root_node(), &mut |node| {
        if !matches!(node.kind(), "call_expression" | "binary_expression") || node.has_error() {
            return;
        }
        let range = (node.start_byte(), node.end_byte());
        if checked.contains(&range)
            || !range_was_generated(range, generated_ranges, generated_points)
        {
            return;
        }
        evaluated.push((range, evaluate_expression(world, source, imports, node)));
    });
    evaluated
}

fn range_was_generated(
    range: (usize, usize),
    generated_ranges: &[(usize, usize)],
    generated_points: &[usize],
) -> bool {
    generated_ranges
        .iter()
        .any(|generated| generated.0 < range.1 && range.0 < generated.1)
        || generated_points
            .iter()
            .any(|point| range.0 <= *point && *point < range.1)
}

fn evaluate_expression(
    world: &RustRequestWorld,
    source: &[u8],
    imports: &BTreeMap<String, RustSymbolShape>,
    node: Node<'_>,
) -> ExpressionVerdict {
    match node.kind() {
        "call_expression" => evaluate_call(world, source, imports, node),
        "binary_expression" => evaluate_binary(source, node),
        _ => ExpressionVerdict::Unknown,
    }
}

fn evaluate_call(
    world: &RustRequestWorld,
    source: &[u8],
    imports: &BTreeMap<String, RustSymbolShape>,
    node: Node<'_>,
) -> ExpressionVerdict {
    let Some(callee) = node.child_by_field_name("function") else {
        return ExpressionVerdict::Unknown;
    };
    let shape = match callee.kind() {
        "identifier" => node_text(callee, source)
            .ok()
            .and_then(|name| imports.get(name).cloned()),
        "scoped_identifier" => simple_path(callee, source).ok().and_then(|segments| {
            let borrowed = segments.iter().map(String::as_str).collect::<Vec<_>>();
            match world.resolve_import(&borrowed) {
                RustImportResolution::Resolved(shape) => Some(shape),
                RustImportResolution::Absent | RustImportResolution::Unknown(_) => None,
            }
        }),
        _ => None,
    };
    let Some(shape) = shape else {
        return ExpressionVerdict::Unknown;
    };
    if shape.callables.is_empty() {
        return if shape.certainty == RustSemanticCertainty::Exact
            && shape
                .kinds
                .iter()
                .all(|kind| *kind == RustSymbolKind::Module)
        {
            ExpressionVerdict::Violation(RustViolation::InvalidCall)
        } else {
            ExpressionVerdict::Unknown
        };
    }
    let Some(arguments) = node.child_by_field_name("arguments") else {
        return ExpressionVerdict::Unknown;
    };
    let mut cursor = arguments.walk();
    let actual = arguments
        .named_children(&mut cursor)
        .map(|argument| expression_type(source, argument, 0))
        .collect::<Vec<_>>();

    let mut saw_unknown = false;
    let mut saw_type_mismatch = false;
    for callable in &shape.callables {
        match call_compatibility(callable, &actual) {
            CallCompatibility::Compatible => {
                return if shape.certainty == RustSemanticCertainty::Exact {
                    ExpressionVerdict::Valid
                } else {
                    ExpressionVerdict::Unknown
                };
            }
            CallCompatibility::Unknown => saw_unknown = true,
            CallCompatibility::ArityMismatch => {}
            CallCompatibility::TypeMismatch => saw_type_mismatch = true,
        }
    }
    if saw_unknown {
        ExpressionVerdict::Unknown
    } else if shape.certainty == RustSemanticCertainty::Partial {
        ExpressionVerdict::Unknown
    } else if saw_type_mismatch {
        ExpressionVerdict::Violation(RustViolation::TypeMismatch)
    } else {
        ExpressionVerdict::Violation(RustViolation::InvalidCall)
    }
}

enum CallCompatibility {
    Compatible,
    Unknown,
    ArityMismatch,
    TypeMismatch,
}

fn call_compatibility(callable: &RustCallableShape, actual: &[RustTypeShape]) -> CallCompatibility {
    if actual.len() < callable.parameters.len() {
        return CallCompatibility::ArityMismatch;
    }
    if actual.len() > callable.parameters.len() {
        match callable.accepts_extra_arguments {
            Some(false) => return CallCompatibility::ArityMismatch,
            None => return CallCompatibility::Unknown,
            Some(true) => {}
        }
    }
    let mut unknown = false;
    for (actual, expected) in actual.iter().zip(&callable.parameters) {
        if *actual == RustTypeShape::Unknown || *expected == RustTypeShape::Unknown {
            unknown = true;
        } else if actual != expected {
            return CallCompatibility::TypeMismatch;
        }
    }
    if unknown {
        CallCompatibility::Unknown
    } else {
        CallCompatibility::Compatible
    }
}

fn evaluate_binary(source: &[u8], node: Node<'_>) -> ExpressionVerdict {
    let Some(operator) = node.child_by_field_name("operator") else {
        return ExpressionVerdict::Unknown;
    };
    if node_text(operator, source) != Ok("+") {
        return ExpressionVerdict::Unknown;
    }
    let (Some(left), Some(right)) = (
        node.child_by_field_name("left"),
        node.child_by_field_name("right"),
    ) else {
        return ExpressionVerdict::Unknown;
    };
    let left = expression_type(source, left, 0);
    let right = expression_type(source, right, 0);
    match (left, right) {
        (RustTypeShape::Unknown, _) | (_, RustTypeShape::Unknown) => ExpressionVerdict::Unknown,
        (RustTypeShape::Integer, RustTypeShape::Integer)
        | (RustTypeShape::Float, RustTypeShape::Float) => ExpressionVerdict::Valid,
        _ => ExpressionVerdict::Violation(RustViolation::TypeMismatch),
    }
}

fn expression_type(source: &[u8], node: Node<'_>, depth: usize) -> RustTypeShape {
    if depth >= 16 {
        return RustTypeShape::Unknown;
    }
    match node.kind() {
        "integer_literal" => RustTypeShape::Integer,
        "float_literal" => RustTypeShape::Float,
        "boolean_literal" => RustTypeShape::Boolean,
        "string_literal" | "raw_string_literal" => RustTypeShape::StringSlice,
        "unit_expression" => RustTypeShape::Unit,
        "parenthesized_expression" => {
            let mut cursor = node.walk();
            node.named_children(&mut cursor)
                .next()
                .map_or(RustTypeShape::Unknown, |child| {
                    expression_type(source, child, depth.saturating_add(1))
                })
        }
        _ => {
            let _ = source;
            RustTypeShape::Unknown
        }
    }
}

fn repairable_analysis() -> Analysis {
    Analysis {
        viability: Viability::Repairable,
        closure: ClosureVerdict::Defer,
        obligations: Vec::new(),
        biases: Vec::<RepairIntent>::new(),
    }
}

fn unknown_target_analysis(boundary: AnalysisBoundary) -> Analysis {
    Analysis {
        viability: Viability::Unknown,
        closure: ClosureVerdict::Defer,
        obligations: vec![SemanticObligation {
            kind: "rust_ambiguous_target".to_string(),
            boundary,
        }],
        biases: Vec::new(),
    }
}

fn end_point(source: &[u8]) -> Point {
    let row = source.iter().filter(|byte| **byte == b'\n').count();
    let column = source
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(source.len(), |last| source.len().saturating_sub(last + 1));
    Point::new(row, column)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_use_trees_expand_to_resolvable_paths() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .unwrap();
        let source = b"use std::{collections::{BTreeMap as Map, BTreeSet}, fmt::*};\n";
        let tree = parser.parse(source, None).unwrap();
        let parsed =
            collect_complete_uses(&tree, source, &[(0, source.len())], &[], &BTreeSet::new());
        assert_eq!(parsed.len(), 1);
        let paths = parsed.into_iter().next().unwrap().paths.unwrap();
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0].segments, ["std", "collections", "BTreeMap"]);
        assert_eq!(paths[0].binding.as_deref(), Some("Map"));
        assert_eq!(paths[1].segments, ["std", "collections", "BTreeSet"]);
        assert_eq!(paths[2].segments, ["std", "fmt"]);
        assert_eq!(paths[2].binding, None);
    }

    #[test]
    fn rust_literal_addition_rejects_known_incompatible_operands() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .unwrap();
        let source = b"fn value() { let _ = \"text\" + 1; }\n";
        let tree = parser.parse(source, None).unwrap();
        let mut verdicts = Vec::new();
        visit_nodes(tree.root_node(), &mut |node| {
            if node.kind() == "binary_expression" {
                verdicts.push(evaluate_binary(source, node));
            }
        });
        assert_eq!(
            verdicts,
            vec![ExpressionVerdict::Violation(RustViolation::TypeMismatch)]
        );
    }

    #[test]
    fn deletion_touch_points_cover_only_affected_half_open_nodes() {
        assert!(range_was_generated((10, 20), &[], &[10]));
        assert!(range_was_generated((10, 20), &[], &[19]));
        assert!(!range_was_generated((10, 20), &[], &[20]));
        assert!(!range_was_generated((10, 20), &[], &[9]));
    }
}
