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
    layers: Vec<(usize, AnalyzerCheckpoint)>,
}

/// Controller-composed stack of independently implemented language layers. Layer construction and
/// warm-up happen outside the collar; this object owns only request-local snapshots used during
/// inference.
pub struct LanguageLayerStack {
    layers: Vec<Box<dyn IncrementalAnalyzer + Send>>,
    active: Vec<usize>,
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
        Ok(())
    }

    pub fn finalize(&mut self) -> CollarResult<Analysis> {
        let mut analyses = Vec::with_capacity(self.active.len());
        for index in &self.active {
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
            .finish()
    }
}
