use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::MultiTaskRun;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultiTaskCheckpoint {
    pub sha256: String,
    pub run: MultiTaskRun,
}

impl MultiTaskCheckpoint {
    pub fn new(run: MultiTaskRun) -> Result<Self> {
        run.validate()?;
        let sha256 = multi_task_digest(&run)?;
        Ok(Self { sha256, run })
    }

    pub fn validate(&self) -> Result<()> {
        let expected = multi_task_digest(&self.run)?;
        if self.sha256 != expected {
            bail!(
                "multi-Task checkpoint digest mismatch: expected {}, got {}",
                expected,
                self.sha256
            );
        }
        self.run.validate()
    }
}

fn multi_task_digest(run: &MultiTaskRun) -> Result<String> {
    let bytes = serde_json::to_vec(run).context("failed to serialize multi-Task checkpoint")?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
