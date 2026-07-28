use std::fmt;

use crate::{CollarError, CollarResult};

use super::{
    Analysis, IncrementalAnalyzer, LayerReadiness, ProgramSnapshot, SourceEvent, Viability,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnalyzerCheckpoint {
    pub epoch: u64,
    pub revision: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerStackCheckpoint {
    active: Vec<usize>,
    participating: Vec<usize>,
    layers: Vec<(usize, AnalyzerCheckpoint)>,
}

/// Controller-composed stack of independently implemented language layers. Layer construction and
/// warm-up happen outside the collar; this object owns only request-local snapshots used during
/// inference.
pub struct LanguageLayerStack {
    layers: Vec<Box<dyn IncrementalAnalyzer + Send>>,
    active: Vec<usize>,
    // Final validation is transaction-wide. A mixed-language patch must finalize every layer that
    // saw a file, not merely the layer matching the last file streamed.
    participating: Vec<usize>,
}

impl LanguageLayerStack {
    pub fn new(
        mut layers: Vec<Box<dyn IncrementalAnalyzer + Send>>,
        snapshot: ProgramSnapshot,
    ) -> CollarResult<Self> {
        for layer in &mut layers {
            if layer.readiness() != LayerReadiness::Ready || layer.readiness_receipt().is_none() {
                return Err(CollarError::Analysis(format!(
                    "language layer {:?} is not ready before inference",
                    layer.descriptor().id
                )));
            }
            layer.begin(snapshot.clone())?;
        }
        Ok(Self {
            layers,
            active: Vec::new(),
            participating: Vec::new(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn checkpoint(&mut self) -> CollarResult<LayerStackCheckpoint> {
        let mut checkpoints = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter_mut().enumerate() {
            checkpoints.push((index, layer.checkpoint()?));
        }
        Ok(LayerStackCheckpoint {
            active: self.active.clone(),
            participating: self.participating.clone(),
            layers: checkpoints,
        })
    }

    pub fn apply(&mut self, event: SourceEvent<'_>) -> CollarResult<Analysis> {
        if let SourceEvent::BeginFile { language, .. } = event {
            self.active = self
                .layers
                .iter()
                .enumerate()
                .filter_map(|(index, layer)| {
                    (&layer.descriptor().language == language).then_some(index)
                })
                .collect();
            for index in &self.active {
                if !self.participating.contains(index) {
                    self.participating.push(*index);
                }
            }
            self.participating.sort_unstable();
        }
        let mut analyses = Vec::with_capacity(self.active.len());
        for index in &self.active {
            analyses.push(self.layers[*index].apply(event)?);
        }
        if analyses.is_empty() {
            Ok(Analysis {
                viability: Viability::Valid,
                closure: super::ClosureVerdict::Allow,
                obligations: Vec::new(),
                biases: Vec::new(),
            })
        } else {
            Ok(Analysis::compose(analyses))
        }
    }

    pub fn rollback(&mut self, checkpoint: LayerStackCheckpoint) -> CollarResult<()> {
        for (index, layer_checkpoint) in checkpoint.layers {
            let layer = self.layers.get_mut(index).ok_or_else(|| {
                CollarError::Analysis(
                    "language-layer checkpoint references a missing layer".to_string(),
                )
            })?;
            layer.rollback(layer_checkpoint)?;
        }
        self.active = checkpoint.active;
        self.participating = checkpoint.participating;
        Ok(())
    }

    pub fn finalize(&mut self) -> CollarResult<Analysis> {
        let mut analyses = Vec::with_capacity(self.participating.len());
        for index in &self.participating {
            analyses.push(self.layers[*index].finalize()?);
        }
        Ok(Analysis::compose(analyses))
    }
}

impl fmt::Debug for LanguageLayerStack {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LanguageLayerStack")
            .field("layers", &self.layers.len())
            .field("active", &self.active)
            .field("participating", &self.participating)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::{
            AnalysisBoundary, AnalyzerLayerDescriptor, ClosureVerdict, LanguageId,
            LayerReadinessReceipt, ReadinessOrigin, SemanticCompleteness, SemanticObligation,
            SemanticWorldId,
        },
        mutation::LogicalPath,
    };

    struct FinalLayer {
        descriptor: AnalyzerLayerDescriptor,
        receipt: LayerReadinessReceipt,
        final_analysis: Analysis,
    }

    impl FinalLayer {
        fn new(language: &str, closure: ClosureVerdict) -> Self {
            let world = SemanticWorldId {
                provider: language.to_string(),
                provider_version: "test".to_string(),
                world_sha256: "1".repeat(64),
                configuration_sha256: "2".repeat(64),
                dependency_sha256: "3".repeat(64),
            };
            Self {
                descriptor: AnalyzerLayerDescriptor {
                    id: format!("{language}-test"),
                    language: LanguageId(language.to_string()),
                    world: world.clone(),
                    capabilities: Vec::new(),
                },
                receipt: LayerReadinessReceipt {
                    world,
                    origin: ReadinessOrigin::ColdBuild,
                    completeness: SemanticCompleteness::Partial,
                    load_millis: 0,
                    prime_millis: 0,
                    primed_queries: 0,
                },
                final_analysis: Analysis {
                    viability: if closure == ClosureVerdict::Reject {
                        Viability::Impossible
                    } else {
                        Viability::Valid
                    },
                    closure,
                    obligations: (closure == ClosureVerdict::Reject)
                        .then(|| SemanticObligation {
                            kind: format!("{language}_final_rejection"),
                            boundary: AnalysisBoundary::ToolCall,
                        })
                        .into_iter()
                        .collect(),
                    biases: Vec::new(),
                },
            }
        }
    }

    impl IncrementalAnalyzer for FinalLayer {
        fn descriptor(&self) -> &AnalyzerLayerDescriptor {
            &self.descriptor
        }

        fn readiness(&self) -> LayerReadiness {
            LayerReadiness::Ready
        }

        fn readiness_receipt(&self) -> Option<&LayerReadinessReceipt> {
            Some(&self.receipt)
        }

        fn begin(&mut self, _snapshot: ProgramSnapshot) -> CollarResult<()> {
            Ok(())
        }

        fn checkpoint(&mut self) -> CollarResult<AnalyzerCheckpoint> {
            Ok(AnalyzerCheckpoint {
                epoch: 0,
                revision: 0,
            })
        }

        fn apply(&mut self, _event: SourceEvent<'_>) -> CollarResult<Analysis> {
            Ok(Analysis {
                viability: Viability::Repairable,
                closure: ClosureVerdict::Defer,
                obligations: Vec::new(),
                biases: Vec::new(),
            })
        }

        fn rollback(&mut self, _checkpoint: AnalyzerCheckpoint) -> CollarResult<()> {
            Ok(())
        }

        fn finalize(&mut self) -> CollarResult<Analysis> {
            Ok(self.final_analysis.clone())
        }
    }

    #[test]
    fn mixed_language_finalization_includes_every_participating_layer() {
        let mut stack = LanguageLayerStack::new(
            vec![
                Box::new(FinalLayer::new("rust", ClosureVerdict::Reject)),
                Box::new(FinalLayer::new("python", ClosureVerdict::Allow)),
            ],
            ProgramSnapshot::default(),
        )
        .unwrap();
        let rust_path = LogicalPath::parse("src/lib.rs").unwrap();
        let rust = LanguageId("rust".to_string());
        stack
            .apply(SourceEvent::BeginFile {
                path: &rust_path,
                language: &rust,
                mutation: crate::mutation::MutationKind::Modify,
            })
            .unwrap();
        stack.apply(SourceEvent::EndFile).unwrap();

        let python_path = LogicalPath::parse("main.py").unwrap();
        let python = LanguageId("python".to_string());
        stack
            .apply(SourceEvent::BeginFile {
                path: &python_path,
                language: &python,
                mutation: crate::mutation::MutationKind::Modify,
            })
            .unwrap();
        stack.apply(SourceEvent::EndFile).unwrap();

        let final_analysis = stack.finalize().unwrap();
        assert_eq!(final_analysis.viability, Viability::Impossible);
        assert_eq!(final_analysis.closure, ClosureVerdict::Reject);
        assert_eq!(final_analysis.obligations[0].kind, "rust_final_rejection");
    }
}
