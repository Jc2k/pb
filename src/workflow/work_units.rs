use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use super::{PlanArtifact, PlannedChange};
use crate::workspace::{ContentSnapshot, PathContent};

pub const WORK_UNIT_LEDGER_VERSION: u32 = 2;
pub const MAX_WORK_UNIT_PROGRESS_CREDITS: usize = 4;

fn present_path<'a>(snapshot: &'a ContentSnapshot, path: &str) -> Option<&'a PathContent> {
    snapshot
        .paths
        .get(path)
        .filter(|content| content.kind != "missing")
}

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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contributing_step_ids: Vec<String>,
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

impl WorkUnit {
    pub fn contributes_to(&self, step_id: &str) -> bool {
        self.plan_step_id == step_id
            || self
                .contributing_step_ids
                .iter()
                .any(|candidate| candidate == step_id)
    }
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
        let mut compiled = BTreeMap::<String, (usize, String, Vec<String>)>::new();
        let mut occurrence = 0_usize;
        for step in &plan.steps {
            for planned in &step.paths {
                let entry = compiled.entry(planned.path.clone()).or_insert_with(|| {
                    let first = occurrence;
                    occurrence += 1;
                    (first, step.id.clone(), Vec::new())
                });
                if !entry.2.contains(&step.id) {
                    entry.2.push(step.id.clone());
                }
            }
        }
        let mut compiled = compiled.into_iter().collect::<Vec<_>>();
        compiled.sort_by_key(|(_, (first, _, _))| *first);
        let mut units = Vec::with_capacity(compiled.len());
        for (path, (first, primary_step_id, contributing_step_ids)) in compiled {
            let baseline_path = present_path(baseline, &path);
            let invocation_path = present_path(invocation, &path);
            let final_present = plan
                .steps
                .iter()
                .flat_map(|step| &step.paths)
                .filter(|planned| planned.path == path)
                .next_back()
                .is_some_and(|planned| planned.change != PlannedChange::Delete);
            let operation = match (baseline_path.is_some(), final_present) {
                (false, true) => PlannedChange::Create,
                (true, true) => PlannedChange::Modify,
                (true, false) => PlannedChange::Delete,
                (false, false) => {
                    bail!("plan path '{path}' has no durable transition from the task baseline")
                }
            };
            units.push(WorkUnit {
                id: format!("path:{first}"),
                plan_step_id: primary_step_id,
                contributing_step_ids,
                operation,
                path,
                baseline_path_fingerprint: baseline_path.map(|entry| entry.fingerprint.clone()),
                invocation_path_fingerprint: invocation_path.map(|entry| entry.fingerprint.clone()),
                current_path_fingerprint: None,
                adopted: baseline_path != invocation_path,
                state: WorkUnitState::EvidenceNeeded,
            });
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
        let mut paths = BTreeSet::new();
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
            if !paths.insert(unit.path.as_str()) {
                bail!(
                    "legacy work-unit ledger repeats path {}; the accepted plan must be replanned",
                    unit.path
                );
            }
            let mut step_ids = BTreeSet::new();
            for step_id in &unit.contributing_step_ids {
                if step_id.trim().is_empty() || !step_ids.insert(step_id) {
                    bail!("work-unit ledger has invalid contributing plan step ids");
                }
            }
        }
        if self
            .progress_credited_units
            .iter()
            .any(|id| !ids.contains(id.as_str()))
        {
            bail!("work-unit progress credit names an unknown unit");
        }
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
            present_path(current, path)
                .map(|entry| entry.fingerprint.as_str())
                .unwrap_or("<missing>")
                == fingerprint.as_str()
        });
        for unit in &mut self.units {
            let now = present_path(current, &unit.path);
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
            let fingerprint = present_path(current, &path)
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
    fn legacy_unique_path_checkpoint_defaults_contributing_steps() {
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
        let mut encoded = serde_json::to_value(&ledger).unwrap();
        encoded["units"][0]
            .as_object_mut()
            .unwrap()
            .remove("contributing_step_ids");

        let restored: WorkUnitLedger = serde_json::from_value(encoded).unwrap();

        restored.validate_plan("plan-1", "sha").unwrap();
        assert!(restored.units[0].contributes_to("s1"));
        assert!(restored.units[0].contributing_step_ids.is_empty());
    }

    #[test]
    fn repeated_path_steps_compile_to_one_verifiable_transition() {
        let baseline = snapshot("base", &[]);
        let artifact = plan(&[
            ("new.txt", PlannedChange::Create),
            ("new.txt", PlannedChange::Modify),
        ]);
        let mut ledger = WorkUnitLedger::from_plan(
            "plan",
            "digest",
            &artifact,
            &baseline,
            &baseline,
            &baseline,
            &BTreeSet::new(),
        )
        .unwrap();

        assert_eq!(ledger.units.len(), 1);
        assert_eq!(ledger.units[0].operation, PlannedChange::Create);
        assert_eq!(ledger.units[0].plan_step_id, "s1");
        assert_eq!(ledger.units[0].contributing_step_ids, ["s1", "s2"]);
        assert!(ledger.units[0].contributes_to("s1"));
        assert!(ledger.units[0].contributes_to("s2"));

        ledger
            .reconcile(
                &snapshot("current", &[("new.txt", "new")]),
                &BTreeSet::new(),
            )
            .unwrap();
        assert!(ledger.structurally_complete());
    }

    #[test]
    fn delete_then_create_compiles_to_modification_of_existing_path() {
        let baseline = snapshot("base", &[("existing.txt", "old")]);
        let artifact = plan(&[
            ("existing.txt", PlannedChange::Delete),
            ("existing.txt", PlannedChange::Create),
        ]);
        let ledger = WorkUnitLedger::from_plan(
            "plan",
            "digest",
            &artifact,
            &baseline,
            &baseline,
            &snapshot("current", &[("existing.txt", "replacement")]),
            &BTreeSet::new(),
        )
        .unwrap();

        assert_eq!(ledger.units.len(), 1);
        assert_eq!(ledger.units[0].operation, PlannedChange::Modify);
        assert!(ledger.structurally_complete());
    }

    #[test]
    fn legacy_repeated_path_ledger_requires_replan() {
        let baseline = snapshot("base", &[("same.txt", "old")]);
        let mut ledger = WorkUnitLedger::from_plan(
            "plan",
            "digest",
            &plan(&[("same.txt", PlannedChange::Modify)]),
            &baseline,
            &baseline,
            &baseline,
            &BTreeSet::new(),
        )
        .unwrap();
        let mut duplicate = ledger.units[0].clone();
        duplicate.id = "legacy-second-occurrence".to_string();
        ledger.units.push(duplicate);

        assert!(
            ledger
                .validate()
                .unwrap_err()
                .to_string()
                .contains("replanned")
        );
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

    #[test]
    fn tracked_missing_delete_completes_and_advances_to_the_next_unit() {
        let baseline = snapshot(
            "base",
            &[("delete.txt", "old-delete"), ("modify.txt", "old-modify")],
        );
        let artifact = plan(&[
            ("delete.txt", PlannedChange::Delete),
            ("modify.txt", PlannedChange::Modify),
        ]);
        let evidence = BTreeSet::from(["delete.txt".to_string(), "modify.txt".to_string()]);
        let mut ledger = WorkUnitLedger::from_plan(
            "plan", "digest", &artifact, &baseline, &baseline, &baseline, &evidence,
        )
        .unwrap();
        assert_eq!(ledger.active().unwrap().path, "delete.txt");
        assert_eq!(ledger.active().unwrap().state, WorkUnitState::MutationReady);

        let mut deleted = snapshot("deleted", &[("modify.txt", "old-modify")]);
        deleted.paths.insert(
            "delete.txt".to_string(),
            PathContent {
                kind: "missing".to_string(),
                fingerprint: "tracked-missing".to_string(),
            },
        );
        ledger.reconcile(&deleted, &evidence).unwrap();

        assert_eq!(ledger.units[0].state, WorkUnitState::StructurallyComplete);
        assert_eq!(ledger.active().unwrap().path, "modify.txt");
        assert_eq!(ledger.active().unwrap().state, WorkUnitState::MutationReady);
    }
}
