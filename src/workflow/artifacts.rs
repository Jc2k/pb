use std::collections::{BTreeSet, HashSet};
use std::path::{Component, Path};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

use crate::workspace::{ContentSnapshot, WorkspaceGraph};

const MAX_ARTIFACT_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactEnvelope<T> {
    pub id: String,
    pub sha256: String,
    pub artifact: T,
}

impl<T> ArtifactEnvelope<T>
where
    T: Serialize,
{
    pub fn new(id: impl Into<String>, artifact: T) -> Result<Self> {
        let id = id.into();
        validate_id("artifact", &id)?;
        let bytes = artifact_bytes(&artifact)?;
        Ok(Self {
            id,
            sha256: format!("{:x}", Sha256::digest(bytes)),
            artifact,
        })
    }

    pub fn validate_digest(&self) -> Result<()> {
        validate_id("artifact", &self.id)?;
        let expected = format!("{:x}", Sha256::digest(artifact_bytes(&self.artifact)?));
        if self.sha256 != expected {
            bail!(
                "artifact '{}' digest mismatch: expected {}, got {}",
                self.id,
                expected,
                self.sha256
            );
        }
        Ok(())
    }
}

fn artifact_bytes<T: Serialize>(artifact: &T) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec(artifact).context("failed to serialize workflow artifact")?;
    if bytes.len() > MAX_ARTIFACT_BYTES {
        bail!(
            "workflow artifact is {} bytes; maximum is {}",
            bytes.len(),
            MAX_ARTIFACT_BYTES
        );
    }
    Ok(bytes)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanArtifact {
    pub summary: String,
    pub requirements: Vec<PlanRequirement>,
    pub steps: Vec<PlanStep>,
    pub acceptance: Vec<PlanAcceptance>,
    #[serde(default)]
    pub risks: Vec<PlanRisk>,
    #[serde(default)]
    pub assumptions: Vec<String>,
    #[serde(default)]
    pub open_questions: Vec<String>,
    #[serde(default)]
    pub resolved_challenge_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanRequirement {
    pub id: String,
    pub description: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanStep {
    pub id: String,
    pub requirement_ids: Vec<String>,
    #[serde(default)]
    pub component_ids: Vec<String>,
    pub paths: Vec<PlanPath>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanPath {
    pub path: String,
    pub change: PlannedChange,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannedChange {
    Create,
    Modify,
    Delete,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanAcceptance {
    pub id: String,
    pub requirement_ids: Vec<String>,
    #[serde(default)]
    pub check_ids: Vec<String>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanRisk {
    pub id: String,
    pub description: String,
    pub mitigation: String,
}

impl PlanArtifact {
    pub fn validate(&self, graph: &WorkspaceGraph, snapshot: &ContentSnapshot) -> Result<()> {
        non_empty("plan summary", &self.summary)?;
        if self.requirements.is_empty() || self.steps.is_empty() || self.acceptance.is_empty() {
            bail!("plan requires non-empty requirements, steps, and acceptance facts");
        }
        if !self.open_questions.is_empty() {
            bail!("plan has unresolved open questions and cannot be accepted");
        }
        let requirement_ids = unique_ids(
            "requirement",
            self.requirements.iter().map(|item| item.id.as_str()),
        )?;
        let step_ids = unique_ids("plan step", self.steps.iter().map(|item| item.id.as_str()))?;
        let acceptance_ids = unique_ids(
            "acceptance",
            self.acceptance.iter().map(|item| item.id.as_str()),
        )?;
        unique_ids("risk", self.risks.iter().map(|item| item.id.as_str()))?;
        unique_ids(
            "resolved challenge",
            self.resolved_challenge_ids.iter().map(String::as_str),
        )?;
        debug_assert_eq!(step_ids.len(), self.steps.len());
        debug_assert_eq!(acceptance_ids.len(), self.acceptance.len());

        let mut requirements_with_steps = HashSet::new();
        let mut planned_paths = snapshot.paths.keys().cloned().collect::<HashSet<_>>();
        for step in &self.steps {
            non_empty("plan step description", &step.description)?;
            if step.requirement_ids.is_empty() {
                bail!("plan step '{}' must reference a requirement", step.id);
            }
            validate_references(
                "plan step requirement",
                &step.id,
                &step.requirement_ids,
                &requirement_ids,
            )?;
            requirements_with_steps.extend(step.requirement_ids.iter().cloned());
            for component in &step.component_ids {
                if !graph.components.contains_key(component) {
                    bail!(
                        "plan step '{}' references unknown component '{}'",
                        step.id,
                        component
                    );
                }
            }
            for planned in &step.paths {
                validate_repository_path("plan path", &planned.path)?;
                let exists = planned_paths.contains(&planned.path);
                match planned.change {
                    PlannedChange::Create if exists => bail!(
                        "plan step '{}' marks path '{}' as create after it already exists in the ordered plan state",
                        step.id,
                        planned.path
                    ),
                    PlannedChange::Modify | PlannedChange::Delete if !exists => bail!(
                        "plan step '{}' marks path '{}' as {:?} before it exists in the ordered plan state",
                        step.id,
                        planned.path,
                        planned.change
                    ),
                    _ => {}
                }
                match planned.change {
                    PlannedChange::Create => {
                        planned_paths.insert(planned.path.clone());
                    }
                    PlannedChange::Delete => {
                        planned_paths.remove(&planned.path);
                    }
                    PlannedChange::Modify => {}
                }
            }
        }

        let mut requirements_with_acceptance = HashSet::new();
        for acceptance in &self.acceptance {
            non_empty("acceptance description", &acceptance.description)?;
            if acceptance.requirement_ids.is_empty() {
                bail!(
                    "acceptance fact '{}' must reference a requirement",
                    acceptance.id
                );
            }
            validate_references(
                "acceptance requirement",
                &acceptance.id,
                &acceptance.requirement_ids,
                &requirement_ids,
            )?;
            requirements_with_acceptance.extend(acceptance.requirement_ids.iter().cloned());
            for check in &acceptance.check_ids {
                if !graph.checks.contains_key(check) {
                    bail!(
                        "acceptance fact '{}' references unknown check '{}'",
                        acceptance.id,
                        check
                    );
                }
            }
        }

        let missing_steps = requirement_ids
            .iter()
            .filter(|id| !requirements_with_steps.contains(**id))
            .map(|id| (*id).to_string())
            .collect::<Vec<_>>();
        let missing_acceptance = requirement_ids
            .iter()
            .filter(|id| !requirements_with_acceptance.contains(**id))
            .map(|id| (*id).to_string())
            .collect::<Vec<_>>();
        if !missing_steps.is_empty() || !missing_acceptance.is_empty() {
            bail!(
                "plan requirement coverage is incomplete; missing steps for [{}], missing acceptance for [{}]",
                missing_steps.join(", "),
                missing_acceptance.join(", ")
            );
        }
        artifact_bytes(self)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PlanAssessmentKind {
    RequirementCoverage,
    Architecture,
    ComponentImpact,
    TestStrategy,
    FailureModes,
    Assumptions,
}

pub const REQUIRED_PLAN_ASSESSMENTS: [PlanAssessmentKind; 6] = [
    PlanAssessmentKind::RequirementCoverage,
    PlanAssessmentKind::Architecture,
    PlanAssessmentKind::ComponentImpact,
    PlanAssessmentKind::TestStrategy,
    PlanAssessmentKind::FailureModes,
    PlanAssessmentKind::Assumptions,
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanReviewArtifact {
    pub plan_id: String,
    pub plan_sha256: String,
    pub assessments: Vec<PlanAssessment>,
    #[serde(default)]
    pub challenges: Vec<ReviewChallenge>,
    pub verdict: ReviewVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanAssessment {
    pub kind: PlanAssessmentKind,
    pub status: AssessmentStatus,
    #[serde(default)]
    pub evidence: Vec<EvidenceReference>,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReviewChallenge {
    pub id: String,
    pub severity: ReviewSeverity,
    #[serde(default)]
    pub requirement_ids: Vec<String>,
    pub description: String,
    #[serde(default)]
    pub evidence: Vec<EvidenceReference>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentStatus {
    Pass,
    Concern,
    Fail,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReviewVerdict {
    Pass,
    Revise,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReviewSeverity {
    P0,
    P1,
    P2,
    P3,
}

impl ReviewSeverity {
    pub const fn is_blocking(self) -> bool {
        matches!(self, Self::P0 | Self::P1)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceReference {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check_id: Option<String>,
    pub description: String,
}

impl PlanReviewArtifact {
    pub fn validate(&self, plan: &ArtifactEnvelope<PlanArtifact>) -> Result<()> {
        plan.validate_digest()?;
        if self.plan_id != plan.id || self.plan_sha256 != plan.sha256 {
            bail!("plan review does not reference the current accepted plan");
        }
        validate_assessment_set(
            self.assessments.iter().map(|item| item.kind),
            &REQUIRED_PLAN_ASSESSMENTS,
            "plan review",
        )?;
        unique_ids(
            "plan challenge",
            self.challenges.iter().map(|item| item.id.as_str()),
        )?;
        for assessment in &self.assessments {
            non_empty("plan assessment explanation", &assessment.explanation)?;
            validate_evidence(&assessment.evidence)?;
        }
        for challenge in &self.challenges {
            non_empty("plan challenge description", &challenge.description)?;
            validate_evidence(&challenge.evidence)?;
            validate_references(
                "plan challenge requirement",
                &challenge.id,
                &challenge.requirement_ids,
                &plan
                    .artifact
                    .requirements
                    .iter()
                    .map(|requirement| requirement.id.as_str())
                    .collect(),
            )?;
        }
        let blocking = self
            .challenges
            .iter()
            .any(|challenge| challenge.severity.is_blocking());
        match (self.verdict, blocking) {
            (ReviewVerdict::Pass, true) => {
                bail!("passing plan review contains a blocking challenge")
            }
            (ReviewVerdict::Revise, false) => {
                bail!("plan review requests revision without a blocking challenge")
            }
            _ => {}
        }
        artifact_bytes(self)?;
        Ok(())
    }

    pub fn validate_observed_evidence(
        &self,
        graph: &WorkspaceGraph,
        read_paths: &HashSet<String>,
    ) -> Result<()> {
        for evidence in self
            .assessments
            .iter()
            .flat_map(|assessment| assessment.evidence.iter())
            .chain(
                self.challenges
                    .iter()
                    .flat_map(|challenge| challenge.evidence.iter()),
            )
        {
            validate_observed_evidence_reference(evidence, graph, read_paths)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ImplementationArtifact {
    pub plan_id: String,
    pub plan_sha256: String,
    pub content_fingerprint: String,
    pub steps: Vec<ImplementationStep>,
    pub summary: String,
    pub no_change: bool,
    pub semantic_commit_subject: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ImplementationStep {
    pub step_id: String,
    pub status: ImplementationStepStatus,
    #[serde(default)]
    pub touched_paths: Vec<String>,
    pub summary: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImplementationStepStatus {
    Completed,
    NoChange,
    Incomplete,
}

impl ImplementationArtifact {
    pub fn validate(&self, plan: &ArtifactEnvelope<PlanArtifact>) -> Result<()> {
        if self.plan_id != plan.id || self.plan_sha256 != plan.sha256 {
            bail!("implementation does not reference the current accepted plan");
        }
        non_empty("implementation summary", &self.summary)?;
        non_empty("content fingerprint", &self.content_fingerprint)?;
        if !self.no_change {
            validate_semantic_subject(&self.semantic_commit_subject)?;
        }
        let expected = plan
            .artifact
            .steps
            .iter()
            .map(|step| step.id.as_str())
            .collect::<BTreeSet<_>>();
        let actual = unique_ids(
            "implementation step",
            self.steps.iter().map(|step| step.step_id.as_str()),
        )?;
        if actual != expected {
            bail!("implementation must account for every accepted plan step exactly once");
        }
        if self
            .steps
            .iter()
            .any(|step| step.status == ImplementationStepStatus::Incomplete)
        {
            bail!("implementation contains an incomplete plan step");
        }
        for step in &self.steps {
            non_empty("implementation step summary", &step.summary)?;
            for path in &step.touched_paths {
                validate_repository_path("implementation touched path", path)?;
            }
        }
        if self.no_change && self.steps.iter().any(|step| !step.touched_paths.is_empty()) {
            bail!("no-change implementation reports touched paths");
        }
        artifact_bytes(self)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum CodeAssessmentKind {
    Correctness,
    Requirements,
    Architecture,
    Tests,
    Regressions,
    Maintainability,
}

pub const REQUIRED_CODE_ASSESSMENTS: [CodeAssessmentKind; 6] = [
    CodeAssessmentKind::Correctness,
    CodeAssessmentKind::Requirements,
    CodeAssessmentKind::Architecture,
    CodeAssessmentKind::Tests,
    CodeAssessmentKind::Regressions,
    CodeAssessmentKind::Maintainability,
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CodeReviewArtifact {
    pub content_fingerprint: String,
    pub assessments: Vec<CodeAssessment>,
    #[serde(default)]
    pub findings: Vec<CodeFinding>,
    pub verdict: ReviewVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CodeAssessment {
    pub kind: CodeAssessmentKind,
    pub status: AssessmentStatus,
    #[serde(default)]
    pub evidence: Vec<EvidenceReference>,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CodeFinding {
    pub id: String,
    pub severity: ReviewSeverity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(default)]
    pub requirement_ids: Vec<String>,
    #[serde(default)]
    pub plan_step_ids: Vec<String>,
    #[serde(default)]
    pub evidence: Vec<EvidenceReference>,
    pub explanation: String,
}

impl CodeReviewArtifact {
    pub fn validate(&self, content_fingerprint: &str) -> Result<()> {
        if self.content_fingerprint != content_fingerprint {
            bail!("code review does not reference the current content fingerprint");
        }
        validate_assessment_set(
            self.assessments.iter().map(|item| item.kind),
            &REQUIRED_CODE_ASSESSMENTS,
            "code review",
        )?;
        unique_ids(
            "code finding",
            self.findings.iter().map(|item| item.id.as_str()),
        )?;
        for assessment in &self.assessments {
            non_empty("code assessment explanation", &assessment.explanation)?;
            validate_evidence(&assessment.evidence)?;
        }
        for finding in &self.findings {
            non_empty("code finding explanation", &finding.explanation)?;
            if let Some(path) = &finding.path {
                validate_repository_path("code finding path", path)?;
            }
            if finding.line == Some(0) || (finding.line.is_some() && finding.path.is_none()) {
                bail!("code finding line must be positive and paired with a repository path");
            }
            if finding.requirement_ids.is_empty() || finding.plan_step_ids.is_empty() {
                bail!(
                    "code finding '{}' must reference an affected requirement and plan step",
                    finding.id
                );
            }
            validate_evidence(&finding.evidence)?;
        }
        let blocking = self
            .findings
            .iter()
            .any(|finding| finding.severity.is_blocking());
        match (self.verdict, blocking) {
            (ReviewVerdict::Pass, true) => {
                bail!("passing code review contains a blocking finding")
            }
            (ReviewVerdict::Revise, false) => {
                bail!("code review requests revision without a blocking finding")
            }
            _ => {}
        }
        artifact_bytes(self)?;
        Ok(())
    }
}

fn validate_assessment_set<T>(
    actual: impl Iterator<Item = T>,
    expected: &[T],
    kind: &str,
) -> Result<()>
where
    T: Copy + Eq + std::hash::Hash + std::fmt::Debug,
{
    let actual = actual.collect::<HashSet<_>>();
    if actual.len() != expected.len() || expected.iter().any(|item| !actual.contains(item)) {
        bail!("{kind} must contain exactly one assessment for every required dimension");
    }
    Ok(())
}

fn validate_evidence(evidence: &[EvidenceReference]) -> Result<()> {
    for item in evidence {
        if item.path.is_none() && item.check_id.is_none() {
            bail!("evidence reference must name a repository path or check id");
        }
        if let Some(path) = &item.path {
            validate_repository_path("evidence path", path)?;
        }
        if item.line == Some(0) || (item.line.is_some() && item.path.is_none()) {
            bail!("evidence line must be positive and paired with a repository path");
        }
        non_empty("evidence description", &item.description)?;
    }
    Ok(())
}

fn validate_observed_evidence_reference(
    evidence: &EvidenceReference,
    graph: &WorkspaceGraph,
    read_paths: &HashSet<String>,
) -> Result<()> {
    if let Some(path) = &evidence.path
        && !read_paths.contains(path)
    {
        bail!("review cites repository path '{path}' without reading it in this fresh context");
    }
    if let Some(check_id) = &evidence.check_id
        && !graph.checks.contains_key(check_id)
    {
        bail!("review cites unknown workspace check '{check_id}'");
    }
    Ok(())
}

fn unique_ids<'a>(kind: &str, ids: impl Iterator<Item = &'a str>) -> Result<BTreeSet<&'a str>> {
    let mut result = BTreeSet::new();
    for id in ids {
        validate_id(kind, id)?;
        if !result.insert(id) {
            bail!("duplicate {kind} id '{id}'");
        }
    }
    Ok(result)
}

fn validate_references(
    kind: &str,
    owner: &str,
    references: &[String],
    known: &BTreeSet<&str>,
) -> Result<()> {
    for reference in references {
        if !known.contains(reference.as_str()) {
            bail!("{kind} on '{owner}' references unknown id '{reference}'");
        }
    }
    Ok(())
}

fn validate_id(kind: &str, id: &str) -> Result<()> {
    if id.is_empty()
        || !id.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
    {
        bail!("{kind} id '{id}' contains unsupported characters");
    }
    Ok(())
}

fn validate_repository_path(field: &str, raw: &str) -> Result<()> {
    let path = Path::new(raw.trim());
    if path.as_os_str().is_empty() || path.is_absolute() {
        bail!("{field} must be a non-empty repository-relative path");
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        bail!("{field} must stay inside the repository");
    }
    Ok(())
}

fn validate_semantic_subject(subject: &str) -> Result<()> {
    let subject = subject.trim();
    let Some((kind, description)) = subject.split_once(':') else {
        bail!("semantic commit subject must use '<type>: <description>'");
    };
    if !matches!(
        kind,
        "feat" | "fix" | "docs" | "test" | "refactor" | "chore" | "perf" | "build" | "ci"
    ) || description.trim().is_empty()
    {
        bail!("semantic commit subject has an unsupported type or empty description");
    }
    Ok(())
}

fn non_empty(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field} must not be empty");
    }
    Ok(())
}

pub fn deserialize_artifact<T: DeserializeOwned>(value: serde_json::Value) -> Result<T> {
    serde_json::from_value(value).context("failed to deserialize workflow artifact")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::workspace::{PathContent, WorkspaceGraph};

    fn graph() -> WorkspaceGraph {
        WorkspaceGraph::legacy(&["cargo test".to_string()])
    }

    fn snapshot() -> ContentSnapshot {
        ContentSnapshot {
            fingerprint: "before".to_string(),
            paths: BTreeMap::from([(
                "src/lib.rs".to_string(),
                PathContent {
                    kind: "file".to_string(),
                    fingerprint: "path".to_string(),
                },
            )]),
        }
    }

    fn plan() -> PlanArtifact {
        let check = graph().checks.keys().next().unwrap().clone();
        PlanArtifact {
            summary: "Implement the change".to_string(),
            requirements: vec![PlanRequirement {
                id: "r1".to_string(),
                description: "change behavior".to_string(),
                source: "user".to_string(),
            }],
            steps: vec![PlanStep {
                id: "s1".to_string(),
                requirement_ids: vec!["r1".to_string()],
                component_ids: vec!["repository".to_string()],
                paths: vec![PlanPath {
                    path: "src/lib.rs".to_string(),
                    change: PlannedChange::Modify,
                }],
                description: "edit source".to_string(),
            }],
            acceptance: vec![PlanAcceptance {
                id: "a1".to_string(),
                requirement_ids: vec!["r1".to_string()],
                check_ids: vec![check],
                description: "tests pass".to_string(),
            }],
            risks: Vec::new(),
            assumptions: Vec::new(),
            open_questions: Vec::new(),
            resolved_challenge_ids: Vec::new(),
        }
    }

    #[test]
    fn plan_validation_enforces_structural_coverage_and_paths() {
        let plan = plan();
        plan.validate(&graph(), &snapshot()).unwrap();

        let mut unresolved = plan.clone();
        unresolved.open_questions.push("which API?".to_string());
        assert!(unresolved.validate(&graph(), &snapshot()).is_err());

        let mut missing = plan;
        missing.steps[0].requirement_ids.clear();
        assert!(missing.validate(&graph(), &snapshot()).is_err());
    }

    #[test]
    fn plan_path_validation_follows_ordered_create_modify_delete_state() {
        let mut ordered = plan();
        ordered.steps = vec![
            PlanStep {
                id: "s1".to_string(),
                requirement_ids: vec!["r1".to_string()],
                component_ids: vec!["repository".to_string()],
                paths: vec![PlanPath {
                    path: "new.js".to_string(),
                    change: PlannedChange::Create,
                }],
                description: "create the module".to_string(),
            },
            PlanStep {
                id: "s2".to_string(),
                requirement_ids: vec!["r1".to_string()],
                component_ids: vec!["repository".to_string()],
                paths: vec![PlanPath {
                    path: "new.js".to_string(),
                    change: PlannedChange::Modify,
                }],
                description: "add the behavior".to_string(),
            },
            PlanStep {
                id: "s3".to_string(),
                requirement_ids: vec!["r1".to_string()],
                component_ids: vec!["repository".to_string()],
                paths: vec![PlanPath {
                    path: "new.js".to_string(),
                    change: PlannedChange::Delete,
                }],
                description: "remove the temporary module".to_string(),
            },
        ];
        ordered.validate(&graph(), &snapshot()).unwrap();

        let mut modify_before_create = ordered.clone();
        modify_before_create.steps.swap(0, 1);
        assert!(
            modify_before_create
                .validate(&graph(), &snapshot())
                .unwrap_err()
                .to_string()
                .contains("before it exists in the ordered plan state")
        );

        let mut modify_after_delete = ordered;
        modify_after_delete.steps.push(PlanStep {
            id: "s4".to_string(),
            requirement_ids: vec!["r1".to_string()],
            component_ids: vec!["repository".to_string()],
            paths: vec![PlanPath {
                path: "new.js".to_string(),
                change: PlannedChange::Modify,
            }],
            description: "invalid late edit".to_string(),
        });
        assert!(
            modify_after_delete
                .validate(&graph(), &snapshot())
                .unwrap_err()
                .to_string()
                .contains("before it exists in the ordered plan state")
        );
    }

    #[test]
    fn artifact_digest_detects_tampering() {
        let mut envelope = ArtifactEnvelope::new("plan-1", plan()).unwrap();
        envelope.artifact.summary = "tampered".to_string();
        assert!(envelope.validate_digest().is_err());
    }

    #[test]
    fn plan_review_requires_exact_hash_and_all_dimensions() {
        let envelope = ArtifactEnvelope::new("plan-1", plan()).unwrap();
        let assessments = REQUIRED_PLAN_ASSESSMENTS
            .into_iter()
            .map(|kind| PlanAssessment {
                kind,
                status: AssessmentStatus::Pass,
                evidence: vec![EvidenceReference {
                    path: Some("src/lib.rs".to_string()),
                    line: None,
                    check_id: None,
                    description: "inspected".to_string(),
                }],
                explanation: "covered".to_string(),
            })
            .collect();
        let mut review = PlanReviewArtifact {
            plan_id: envelope.id.clone(),
            plan_sha256: envelope.sha256.clone(),
            assessments,
            challenges: Vec::new(),
            verdict: ReviewVerdict::Pass,
        };
        review.validate(&envelope).unwrap();
        review.plan_sha256 = "wrong".to_string();
        assert!(review.validate(&envelope).is_err());
    }

    #[test]
    fn plan_review_cannot_claim_unread_or_unknown_evidence() {
        let envelope = ArtifactEnvelope::new("plan-1", plan()).unwrap();
        let mut review = PlanReviewArtifact {
            plan_id: envelope.id.clone(),
            plan_sha256: envelope.sha256.clone(),
            assessments: REQUIRED_PLAN_ASSESSMENTS
                .into_iter()
                .map(|kind| PlanAssessment {
                    kind,
                    status: AssessmentStatus::Pass,
                    evidence: Vec::new(),
                    explanation: "assessed".to_string(),
                })
                .collect(),
            challenges: Vec::new(),
            verdict: ReviewVerdict::Pass,
        };
        review.assessments[0].evidence.push(EvidenceReference {
            path: Some("src/lib.rs".to_string()),
            line: Some(1),
            check_id: None,
            description: "inspected source".to_string(),
        });

        assert!(
            review
                .validate_observed_evidence(&graph(), &HashSet::new())
                .unwrap_err()
                .to_string()
                .contains("without reading")
        );
        let reads = HashSet::from(["src/lib.rs".to_string()]);
        review.validate_observed_evidence(&graph(), &reads).unwrap();

        review.assessments[0].evidence[0].check_id = Some("unknown-check".to_string());
        assert!(
            review
                .validate_observed_evidence(&graph(), &reads)
                .unwrap_err()
                .to_string()
                .contains("unknown workspace check")
        );
    }

    #[test]
    fn code_review_rejects_blockers_in_a_pass() {
        let assessments = REQUIRED_CODE_ASSESSMENTS
            .into_iter()
            .map(|kind| CodeAssessment {
                kind,
                status: AssessmentStatus::Pass,
                evidence: Vec::new(),
                explanation: "checked".to_string(),
            })
            .collect();
        let review = CodeReviewArtifact {
            content_fingerprint: "content".to_string(),
            assessments,
            findings: vec![CodeFinding {
                id: "f1".to_string(),
                severity: ReviewSeverity::P1,
                path: Some("src/lib.rs".to_string()),
                line: Some(1),
                requirement_ids: vec!["r1".to_string()],
                plan_step_ids: vec!["s1".to_string()],
                evidence: Vec::new(),
                explanation: "bug".to_string(),
            }],
            verdict: ReviewVerdict::Pass,
        };
        assert!(review.validate("content").is_err());
    }
}
