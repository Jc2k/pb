use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use super::{PlanArtifact, PlannedChange};
use crate::workspace::ContentSnapshot;

pub const WORK_UNIT_LEDGER_VERSION: u32 = 2;
pub const MAX_WORK_UNIT_PROGRESS_CREDITS: usize = 4;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkUnitState {
    EvidenceNeeded,
    MutationReady,
    StructurallyComplete,
    DiagnosticFailed,
    DiagnosticRepairReady,
    BlockedForReplan,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkUnit {
    pub id: String,
    pub plan_step_id: String,
    pub operation: PlannedChange,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_path_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invocation_path_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_path_fingerprint: Option<String>,
    #[serde(default)]
    pub adopted: bool,
    pub state: WorkUnitState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkUnitLedger {
    pub version: u32,
    #[serde(default)]
    pub plan_id: String,
    #[serde(default)]
    pub plan_sha256: String,
    #[serde(default)]
    pub baseline_fingerprint: String,
    #[serde(default)]
    pub invocation_fingerprint: String,
    #[serde(default)]
    pub observed_fingerprint: String,
    #[serde(default)]
    pub units: Vec<WorkUnit>,
    #[serde(default)]
    pub progress_credited_units: BTreeSet<String>,
    #[serde(default)]
    pub diagnostic_failures: BTreeMap<String, String>,
}

impl Default for WorkUnitLedger {
    fn default() -> Self {
        Self {
            version: WORK_UNIT_LEDGER_VERSION,
            plan_id: String::new(),
            plan_sha256: String::new(),
            baseline_fingerprint: String::new(),
            invocation_fingerprint: String::new(),
            observed_fingerprint: String::new(),
            units: Vec::new(),
            progress_credited_units: BTreeSet::new(),
            diagnostic_failures: BTreeMap::new(),
        }
    }
}

impl WorkUnitLedger {
    pub fn no_change(
        plan_id: &str,
        plan_sha256: &str,
        baseline: &ContentSnapshot,
        invocation: &ContentSnapshot,
        current: &ContentSnapshot,
    ) -> Result<Self> {
        let ledger = Self {
            version: WORK_UNIT_LEDGER_VERSION,
            plan_id: plan_id.to_string(),
            plan_sha256: plan_sha256.to_string(),
            baseline_fingerprint: baseline.fingerprint.clone(),
            invocation_fingerprint: invocation.fingerprint.clone(),
            observed_fingerprint: current.fingerprint.clone(),
            units: Vec::new(),
            progress_credited_units: BTreeSet::new(),
            diagnostic_failures: BTreeMap::new(),
        };
        ledger.validate()?;
        Ok(ledger)
    }

    pub fn from_plan(
        plan_id: &str,
        plan_sha256: &str,
        plan: &PlanArtifact,
        baseline: &ContentSnapshot,
        invocation: &ContentSnapshot,
        current: &ContentSnapshot,
        exact_evidence_paths: &BTreeSet<String>,
    ) -> Result<Self> {
        let mut units = Vec::new();
        for step in &plan.steps {
            for (path_index, planned) in step.paths.iter().enumerate() {
                units.push(WorkUnit {
                    id: format!("{}:{path_index}", step.id),
                    plan_step_id: step.id.clone(),
                    operation: planned.change,
                    path: planned.path.clone(),
                    baseline_path_fingerprint: baseline
                        .paths
                        .get(&planned.path)
                        .map(|entry| entry.fingerprint.clone()),
                    invocation_path_fingerprint: invocation
                        .paths
                        .get(&planned.path)
                        .map(|entry| entry.fingerprint.clone()),
                    current_path_fingerprint: None,
                    adopted: baseline.paths.get(&planned.path)
                        != invocation.paths.get(&planned.path),
                    state: WorkUnitState::EvidenceNeeded,
                });
            }
        }
        let mut ledger = Self {
            version: WORK_UNIT_LEDGER_VERSION,
            plan_id: plan_id.to_string(),
            plan_sha256: plan_sha256.to_string(),
            baseline_fingerprint: baseline.fingerprint.clone(),
            invocation_fingerprint: invocation.fingerprint.clone(),
            observed_fingerprint: String::new(),
            units,
            progress_credited_units: BTreeSet::new(),
            diagnostic_failures: BTreeMap::new(),
        };
        ledger.reconcile(current, exact_evidence_paths)?;
        ledger.validate()?;
        Ok(ledger)
    }

    pub fn is_initialized(&self) -> bool {
        !self.plan_id.is_empty()
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != WORK_UNIT_LEDGER_VERSION {
            bail!(
                "unsupported work-unit ledger schema {}; expected {}",
                self.version,
                WORK_UNIT_LEDGER_VERSION
            );
        }
        if !self.is_initialized() {
            if self.plan_sha256.is_empty()
                && self.baseline_fingerprint.is_empty()
                && self.invocation_fingerprint.is_empty()
                && self.observed_fingerprint.is_empty()
                && self.units.is_empty()
                && self.progress_credited_units.is_empty()
                && self.diagnostic_failures.is_empty()
            {
                return Ok(());
            }
            bail!("partially initialized work-unit ledger");
        }
        if self.plan_sha256.is_empty()
            || self.baseline_fingerprint.is_empty()
            || self.invocation_fingerprint.is_empty()
            || self.observed_fingerprint.is_empty()
        {
            bail!("initialized work-unit ledger has incomplete authority fingerprints");
        }
        let mut ids = BTreeSet::new();
        for unit in &self.units {
            if unit.id.trim().is_empty()
                || unit.plan_step_id.trim().is_empty()
                || unit.path.trim().is_empty()
                || unit.path.starts_with('/')
                || unit
                    .path
                    .split('/')
                    .any(|part| part.is_empty() || part == "." || part == "..")
            {
                bail!("invalid work-unit identity or path");
            }
            if !ids.insert(unit.id.as_str()) {
                bail!("work-unit ledger repeats id {}", unit.id);
            }
        }
        if self
            .progress_credited_units
            .iter()
            .any(|id| !ids.contains(id.as_str()))
        {
            bail!("work-unit progress credit names an unknown unit");
        }
        let paths = self
            .units
            .iter()
            .map(|unit| unit.path.as_str())
            .collect::<BTreeSet<_>>();
        if self
            .diagnostic_failures
            .keys()
            .any(|path| !paths.contains(path.as_str()))
        {
            bail!("work-unit diagnostic failure names an unknown path");
        }
        Ok(())
    }

    pub fn validate_plan(&self, plan_id: &str, plan_sha256: &str) -> Result<()> {
        self.validate()?;
        if self.is_initialized() && (self.plan_id != plan_id || self.plan_sha256 != plan_sha256) {
            bail!("work-unit ledger authority does not match the accepted plan");
        }
        Ok(())
    }

    pub fn reconcile(
        &mut self,
        current: &ContentSnapshot,
        exact_evidence_paths: &BTreeSet<String>,
    ) -> Result<bool> {
        self.validate_authority_shape()?;
        let before = self.clone();
        self.observed_fingerprint = current.fingerprint.clone();
        self.diagnostic_failures.retain(|path, fingerprint| {
            current
                .paths
                .get(path)
                .map(|entry| entry.fingerprint.as_str())
                .unwrap_or("<missing>")
                == fingerprint.as_str()
        });
        for unit in &mut self.units {
            let now = current.paths.get(&unit.path);
            unit.current_path_fingerprint = now.map(|entry| entry.fingerprint.clone());
            let task_transition_complete = match unit.operation {
                PlannedChange::Create => unit.baseline_path_fingerprint.is_none() && now.is_some(),
                PlannedChange::Modify => {
                    now.is_some() && unit.current_path_fingerprint != unit.baseline_path_fingerprint
                }
                PlannedChange::Delete => unit.baseline_path_fingerprint.is_some() && now.is_none(),
            };
            let invocation_transition_complete = match unit.operation {
                PlannedChange::Create => {
                    unit.invocation_path_fingerprint.is_none() && now.is_some()
                }
                PlannedChange::Modify => {
                    now.is_some()
                        && unit.current_path_fingerprint != unit.invocation_path_fingerprint
                }
                PlannedChange::Delete => {
                    unit.invocation_path_fingerprint.is_some() && now.is_none()
                }
            };
            if self.diagnostic_failures.contains_key(&unit.path) {
                unit.state = if now.is_none() {
                    WorkUnitState::BlockedForReplan
                } else if exact_evidence_paths.contains(&unit.path) {
                    WorkUnitState::DiagnosticRepairReady
                } else {
                    WorkUnitState::DiagnosticFailed
                };
                continue;
            }
            if unit.adopted && task_transition_complete
                || invocation_transition_complete
                || unit.state == WorkUnitState::StructurallyComplete && task_transition_complete
            {
                unit.state = WorkUnitState::StructurallyComplete;
                continue;
            }
            unit.state = match unit.operation {
                PlannedChange::Create if now.is_none() => WorkUnitState::MutationReady,
                PlannedChange::Create => WorkUnitState::BlockedForReplan,
                PlannedChange::Modify | PlannedChange::Delete if now.is_none() => {
                    WorkUnitState::BlockedForReplan
                }
                PlannedChange::Modify | PlannedChange::Delete
                    if exact_evidence_paths.contains(&unit.path) =>
                {
                    WorkUnitState::MutationReady
                }
                PlannedChange::Modify | PlannedChange::Delete => WorkUnitState::EvidenceNeeded,
            };
        }
        self.validate()?;
        Ok(*self != before)
    }

    pub fn active(&self) -> Option<&WorkUnit> {
        self.units
            .iter()
            .find(|unit| unit.state != WorkUnitState::StructurallyComplete)
    }

    pub fn structurally_complete(&self) -> bool {
        self.is_initialized()
            && self
                .units
                .iter()
                .all(|unit| unit.state == WorkUnitState::StructurallyComplete)
    }

    pub fn actual_task_paths(&self) -> BTreeSet<String> {
        self.units
            .iter()
            .filter(|unit| unit.state == WorkUnitState::StructurallyComplete)
            .map(|unit| unit.path.clone())
            .collect()
    }

    pub fn credit_progress(&mut self, unit_id: &str) -> Result<bool> {
        self.validate()?;
        if self.progress_credited_units.len() >= MAX_WORK_UNIT_PROGRESS_CREDITS {
            return Ok(false);
        }
        if !self.units.iter().any(|unit| unit.id == unit_id) {
            bail!("cannot credit unknown work unit {unit_id}");
        }
        Ok(self.progress_credited_units.insert(unit_id.to_string()))
    }

    pub fn mark_diagnostic_failed(
        &mut self,
        paths: impl IntoIterator<Item = String>,
        current: &ContentSnapshot,
    ) -> Result<()> {
        let known = self
            .units
            .iter()
            .map(|unit| unit.path.as_str())
            .collect::<BTreeSet<_>>();
        for path in paths {
            if !known.contains(path.as_str()) {
                continue;
            }
            let fingerprint = current
                .paths
                .get(&path)
                .map(|entry| entry.fingerprint.clone())
                .unwrap_or_else(|| "<missing>".to_string());
            self.diagnostic_failures.insert(path, fingerprint);
        }
        self.validate()
    }

    fn validate_authority_shape(&self) -> Result<()> {
        if self.version != WORK_UNIT_LEDGER_VERSION || !self.is_initialized() {
            bail!("cannot reconcile an uninitialized work-unit ledger");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::{PlanAcceptance, PlanPath, PlanRequirement, PlanStep};
    use crate::workspace::PathContent;
    use std::collections::BTreeMap;

    fn snapshot(fingerprint: &str, paths: &[(&str, &str)]) -> ContentSnapshot {
        ContentSnapshot {
            fingerprint: fingerprint.to_string(),
            paths: paths
                .iter()
                .map(|(path, hash)| {
                    (
                        (*path).to_string(),
                        PathContent {
                            kind: "file".to_string(),
                            fingerprint: (*hash).to_string(),
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>(),
        }
    }

    fn plan(paths: &[(&str, PlannedChange)]) -> PlanArtifact {
        PlanArtifact {
            summary: "work".to_string(),
            requirements: vec![PlanRequirement {
                id: "r1".to_string(),
                description: "work".to_string(),
                source: "user".to_string(),
            }],
            steps: paths
                .iter()
                .enumerate()
                .map(|(index, (path, change))| PlanStep {
                    id: format!("s{}", index + 1),
                    requirement_ids: vec!["r1".to_string()],
                    component_ids: Vec::new(),
                    paths: vec![PlanPath {
                        path: (*path).to_string(),
                        change: *change,
                    }],
                    description: "work".to_string(),
                })
                .collect(),
            acceptance: vec![PlanAcceptance {
                id: "a1".to_string(),
                requirement_ids: vec!["r1".to_string()],
                check_ids: Vec::new(),
                description: "done".to_string(),
            }],
            risks: Vec::new(),
            assumptions: Vec::new(),
            open_questions: Vec::new(),
            resolved_challenge_ids: Vec::new(),
        }
    }

    #[test]
    fn orders_create_modify_delete_and_requires_exact_evidence() {
        let baseline = snapshot("base", &[("modify.txt", "old"), ("delete.txt", "old")]);
        let current = baseline.clone();
        let mut ledger = WorkUnitLedger::from_plan(
            "plan-1",
            "sha",
            &plan(&[
                ("create.txt", PlannedChange::Create),
                ("modify.txt", PlannedChange::Modify),
                ("delete.txt", PlannedChange::Delete),
            ]),
            &baseline,
            &baseline,
            &current,
            &BTreeSet::new(),
        )
        .unwrap();
        assert_eq!(ledger.active().unwrap().path, "create.txt");
        assert_eq!(ledger.active().unwrap().state, WorkUnitState::MutationReady);

        let created = snapshot(
            "created",
            &[
                ("create.txt", "new"),
                ("modify.txt", "old"),
                ("delete.txt", "old"),
            ],
        );
        ledger.reconcile(&created, &BTreeSet::new()).unwrap();
        assert_eq!(ledger.active().unwrap().path, "modify.txt");
        assert_eq!(
            ledger.active().unwrap().state,
            WorkUnitState::EvidenceNeeded
        );
        ledger
            .reconcile(&created, &BTreeSet::from(["modify.txt".to_string()]))
            .unwrap();
        assert_eq!(ledger.active().unwrap().state, WorkUnitState::MutationReady);
    }

    #[test]
    fn adopted_resume_delta_is_structurally_complete_without_false_authorship() {
        let baseline = snapshot("base", &[("formatter.mjs", "old")]);
        let invocation = snapshot("invocation", &[("formatter.mjs", "adopted")]);
        let ledger = WorkUnitLedger::from_plan(
            "plan-1",
            "sha",
            &plan(&[("formatter.mjs", PlannedChange::Modify)]),
            &baseline,
            &invocation,
            &invocation,
            &BTreeSet::new(),
        )
        .unwrap();
        assert!(ledger.units[0].adopted);
        assert_eq!(ledger.units[0].state, WorkUnitState::StructurallyComplete);
    }

    #[test]
    fn checkpoint_round_trip_preserves_active_unit_and_fingerprints() {
        let baseline = snapshot("base", &[("modify.txt", "old")]);
        let ledger = WorkUnitLedger::from_plan(
            "plan-1",
            "sha",
            &plan(&[("modify.txt", PlannedChange::Modify)]),
            &baseline,
            &baseline,
            &baseline,
            &BTreeSet::new(),
        )
        .unwrap();
        let encoded = serde_json::to_vec(&ledger).unwrap();
        let restored: WorkUnitLedger = serde_json::from_slice(&encoded).unwrap();
        restored.validate_plan("plan-1", "sha").unwrap();
        assert_eq!(restored, ledger);
    }

    #[test]
    fn failed_create_diagnostic_reopens_as_a_bounded_repair_after_fresh_evidence() {
        let baseline = snapshot("base", &[]);
        let current = snapshot("created", &[("new.txt", "v1")]);
        let artifact = plan(&[("new.txt", PlannedChange::Create)]);
        let mut ledger = WorkUnitLedger::from_plan(
            "plan",
            "digest",
            &artifact,
            &baseline,
            &baseline,
            &current,
            &BTreeSet::new(),
        )
        .unwrap();
        assert!(ledger.structurally_complete());

        ledger
            .mark_diagnostic_failed(["new.txt".to_string()], &current)
            .unwrap();
        ledger.reconcile(&current, &BTreeSet::new()).unwrap();
        assert_eq!(
            ledger.active().unwrap().state,
            WorkUnitState::DiagnosticFailed
        );

        ledger
            .reconcile(&current, &BTreeSet::from(["new.txt".to_string()]))
            .unwrap();
        assert_eq!(
            ledger.active().unwrap().state,
            WorkUnitState::DiagnosticRepairReady
        );
    }
}
