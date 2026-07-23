use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::{TaskPlannerQualification, task_planner_protocol_sha256, task_planner_template_sha256};

pub const TASK_QUALIFICATION_CATALOG_VERSION: u32 = 1;

const EMBEDDED_TASK_QUALIFICATIONS: &str =
    include_str!("../../fixtures/task-decomposition/qualifications.json");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlannerQualificationCatalog {
    pub version: u32,
    pub entries: Vec<TaskPlannerQualificationEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskPlannerQualificationEntry {
    pub model: String,
    pub backend: String,
    pub qualification: TaskPlannerQualification,
}

impl TaskPlannerQualificationCatalog {
    pub fn embedded() -> Result<Self> {
        let catalog: Self = serde_json::from_str(EMBEDDED_TASK_QUALIFICATIONS)
            .context("embedded Task planner qualification catalog is invalid")?;
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != TASK_QUALIFICATION_CATALOG_VERSION {
            bail!(
                "unsupported Task qualification catalog version {}; expected {}",
                self.version,
                TASK_QUALIFICATION_CATALOG_VERSION
            );
        }
        let template = task_planner_template_sha256();
        let protocol = task_planner_protocol_sha256();
        let mut models = HashSet::new();
        let mut selectors = HashSet::new();
        for entry in &self.entries {
            if entry.model.trim().is_empty()
                || !matches!(entry.backend.as_str(), "llama_cpp" | "flashmoe")
            {
                bail!("Task qualification entry has an invalid model/backend selector");
            }
            if !selectors.insert((entry.model.as_str(), entry.backend.as_str())) {
                bail!("Task qualification catalog contains a duplicate model/backend selector");
            }
            let qualification = &entry.qualification;
            qualification.validate()?;
            if qualification.template_sha256 != template
                || qualification.protocol_sha256 != protocol
            {
                bail!("Task qualification targets a different planner template or protocol");
            }
            if !models.insert(qualification.model_sha256.as_str()) {
                bail!("Task qualification catalog contains a duplicate model digest");
            }
        }
        Ok(())
    }

    pub fn candidate(&self, model: &str, backend: &str) -> Option<&TaskPlannerQualification> {
        self.entries
            .iter()
            .find(|entry| entry.model == model && entry.backend == backend)
            .map(|entry| &entry.qualification)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_catalog_is_valid_and_fail_closed_without_promotions() {
        let catalog = TaskPlannerQualificationCatalog::embedded().unwrap();
        assert!(catalog.entries.is_empty());
    }

    #[test]
    fn catalog_rejects_duplicate_model_or_wrong_protocol() {
        let qualification = TaskPlannerQualification::new(
            "a".repeat(64),
            task_planner_template_sha256(),
            task_planner_protocol_sha256(),
            "b".repeat(64),
            true,
            false,
        )
        .unwrap();
        let duplicate = TaskPlannerQualificationCatalog {
            version: TASK_QUALIFICATION_CATALOG_VERSION,
            entries: vec![
                TaskPlannerQualificationEntry {
                    model: "model".to_string(),
                    backend: "llama_cpp".to_string(),
                    qualification: qualification.clone(),
                },
                TaskPlannerQualificationEntry {
                    model: "other-model".to_string(),
                    backend: "llama_cpp".to_string(),
                    qualification,
                },
            ],
        };
        assert!(duplicate.validate().is_err());

        let wrong = TaskPlannerQualificationCatalog {
            version: TASK_QUALIFICATION_CATALOG_VERSION,
            entries: vec![TaskPlannerQualificationEntry {
                model: "model".to_string(),
                backend: "llama_cpp".to_string(),
                qualification: TaskPlannerQualification::new(
                    "c".repeat(64),
                    task_planner_template_sha256(),
                    "d".repeat(64),
                    "e".repeat(64),
                    true,
                    false,
                )
                .unwrap(),
            }],
        };
        assert!(wrong.validate().is_err());
    }
}
