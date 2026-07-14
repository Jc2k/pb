#[cfg(test)]
use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use super::ReadyEvidenceBundle;

pub const PUBLICATION_REQUEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PublicationRequest {
    pub version: u32,
    pub idempotency_key: String,
    pub evidence_sha256: String,
    pub evidence: ReadyEvidenceBundle,
}

impl PublicationRequest {
    pub fn new(evidence: ReadyEvidenceBundle) -> Result<Self> {
        evidence.validate()?;
        let evidence_sha256 = evidence.sha256()?;
        Ok(Self {
            version: PUBLICATION_REQUEST_VERSION,
            idempotency_key: format!("ready:{}:{}", evidence.workflow_id, evidence.commit_oid),
            evidence_sha256,
            evidence,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != PUBLICATION_REQUEST_VERSION {
            bail!(
                "unsupported publication request version {}; expected {}",
                self.version,
                PUBLICATION_REQUEST_VERSION
            );
        }
        self.evidence.validate()?;
        let expected_key = format!(
            "ready:{}:{}",
            self.evidence.workflow_id, self.evidence.commit_oid
        );
        if self.idempotency_key != expected_key {
            bail!("publication idempotency key is not bound to workflow and commit");
        }
        if self.evidence_sha256 != self.evidence.sha256()? {
            bail!("publication request evidence digest mismatch");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PublicationDisposition {
    NotConfigured,
    Accepted,
    Reused,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PublicationReceipt {
    pub idempotency_key: String,
    pub evidence_sha256: String,
    pub disposition: PublicationDisposition,
}

pub trait ReadyEvidencePublisher {
    fn publish(&mut self, request: &PublicationRequest) -> Result<PublicationReceipt>;
}

/// Deliberately performs no provider or network operation. It makes the absence
/// of publication configuration explicit without weakening local Ready status.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopReadyEvidencePublisher;

impl ReadyEvidencePublisher for NoopReadyEvidencePublisher {
    fn publish(&mut self, request: &PublicationRequest) -> Result<PublicationReceipt> {
        request.validate()?;
        Ok(PublicationReceipt {
            idempotency_key: request.idempotency_key.clone(),
            evidence_sha256: request.evidence_sha256.clone(),
            disposition: PublicationDisposition::NotConfigured,
        })
    }
}

#[cfg(test)]
#[derive(Debug, Default)]
struct MockReadyEvidencePublisher {
    accepted: BTreeMap<String, String>,
    provider_mutations: usize,
}

#[cfg(test)]
impl ReadyEvidencePublisher for MockReadyEvidencePublisher {
    fn publish(&mut self, request: &PublicationRequest) -> Result<PublicationReceipt> {
        request.validate()?;
        let disposition = match self.accepted.get(&request.idempotency_key) {
            Some(existing) if existing == &request.evidence_sha256 => {
                PublicationDisposition::Reused
            }
            Some(_) => bail!("publication idempotency key was reused for different evidence"),
            None => {
                self.accepted.insert(
                    request.idempotency_key.clone(),
                    request.evidence_sha256.clone(),
                );
                self.provider_mutations += 1;
                PublicationDisposition::Accepted
            }
        };
        Ok(PublicationReceipt {
            idempotency_key: request.idempotency_key.clone(),
            evidence_sha256: request.evidence_sha256.clone(),
            disposition,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(commit_oid: &str) -> ReadyEvidenceBundle {
        ReadyEvidenceBundle {
            workflow_id: "workflow-1".to_string(),
            commit_oid: commit_oid.to_string(),
            plan_sha256: "a".repeat(64),
            review_sha256: "b".repeat(64),
            check_evidence_ids: vec!["check:test".to_string()],
            repository_remote: Some("git@example.test:team/project.git".to_string()),
        }
    }

    #[test]
    fn noop_publisher_is_deterministic_and_has_no_stateful_side_effect() {
        let request = PublicationRequest::new(evidence("commit-1")).unwrap();
        let mut publisher = NoopReadyEvidencePublisher;
        let first = publisher.publish(&request).unwrap();
        let second = publisher.publish(&request).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.disposition, PublicationDisposition::NotConfigured);
    }

    #[test]
    fn mock_publisher_accepts_an_idempotency_key_only_once() {
        let request = PublicationRequest::new(evidence("commit-1")).unwrap();
        let mut publisher = MockReadyEvidencePublisher::default();
        assert_eq!(
            publisher.publish(&request).unwrap().disposition,
            PublicationDisposition::Accepted
        );
        assert_eq!(
            publisher.publish(&request).unwrap().disposition,
            PublicationDisposition::Reused
        );
        assert_eq!(publisher.provider_mutations, 1);
    }

    #[test]
    fn mock_publisher_rejects_same_key_with_changed_evidence() {
        let request = PublicationRequest::new(evidence("commit-1")).unwrap();
        let mut changed = request.clone();
        changed.evidence.review_sha256 = "c".repeat(64);
        changed.evidence_sha256 = changed.evidence.sha256().unwrap();
        let mut publisher = MockReadyEvidencePublisher::default();
        publisher.publish(&request).unwrap();
        assert!(publisher.publish(&changed).is_err());
        assert_eq!(publisher.provider_mutations, 1);
    }

    #[test]
    fn publication_request_rejects_untrusted_or_malformed_evidence() {
        let mut unsafe_remote = evidence("commit-1");
        unsafe_remote.repository_remote =
            Some("https://token:secret@example.test/team/project.git".to_string());
        assert!(PublicationRequest::new(unsafe_remote).is_err());

        let mut malformed_digest = evidence("commit-1");
        malformed_digest.review_sha256 = "not-a-sha".to_string();
        assert!(PublicationRequest::new(malformed_digest).is_err());
    }
}
