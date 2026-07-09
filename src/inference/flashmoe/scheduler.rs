use super::capabilities::{
    FlashMoeCapabilityPlan, FlashMoeGraphStage, FlashMoeStageCapability, FlashMoeStagePlacement,
    FlashMoeUnsupportedCapability,
};
use super::math::softmax_in_place;
use anyhow::{Result, bail};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeScheduledGraph {
    stages: Vec<FlashMoeStageCapability>,
}

impl FlashMoeScheduledGraph {
    pub fn from_capabilities(
        capabilities: &FlashMoeCapabilityPlan,
    ) -> Result<Self, FlashMoeUnsupportedCapability> {
        capabilities.validate_complete()?;
        let stages = FlashMoeGraphStage::ALL
            .iter()
            .map(|stage| {
                capabilities.stage(*stage).copied().ok_or_else(|| {
                    FlashMoeUnsupportedCapability::new(
                        capabilities.family,
                        *stage,
                        "graph stage has no declared implementation",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { stages })
    }

    pub fn stages(&self) -> &[FlashMoeStageCapability] {
        &self.stages
    }

    pub fn stage(&self, stage: FlashMoeGraphStage) -> &FlashMoeStageCapability {
        self.stages
            .iter()
            .find(|capability| capability.stage == stage)
            .expect("scheduled graph contains every FlashMoe stage")
    }

    pub fn active_expert_reads(&self) -> &FlashMoeStageCapability {
        self.stage(FlashMoeGraphStage::ActiveExpertReads)
    }

    pub fn cmd_sequence(&self) -> [&FlashMoeStageCapability; 3] {
        [
            self.stage(FlashMoeGraphStage::DeferredPreviousCmd3),
            self.stage(FlashMoeGraphStage::Cmd1AttentionProjections),
            self.stage(FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection),
        ]
    }

    pub fn declares_scheduler_owned_expert_reads(&self) -> bool {
        self.active_expert_reads().placement == FlashMoeStagePlacement::SchedulerIo
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertRoute {
    pub expert: usize,
    pub score: f32,
}

impl ExpertRoute {
    pub fn from_pair((expert, score): (usize, f32)) -> Self {
        Self { expert, score }
    }

    pub fn from_scores(routes: &[(usize, f32)]) -> Result<Vec<Self>> {
        routes
            .iter()
            .copied()
            .map(|route| {
                let route = Self::from_pair(route);
                route.validate()?;
                Ok(route)
            })
            .collect()
    }

    fn validate(&self) -> Result<()> {
        if !self.score.is_finite() {
            bail!(
                "expert route score for expert {} is not finite: {}",
                self.expert,
                self.score
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledExpertRoutes {
    pub layer: usize,
    pub routes: Vec<ExpertRoute>,
    pub weights: Vec<f32>,
}

impl ScheduledExpertRoutes {
    pub fn from_routes(
        layer: usize,
        routes: Vec<ExpertRoute>,
        routed_expert_scale: f32,
    ) -> Result<Self> {
        if !(routed_expert_scale.is_finite() && routed_expert_scale > 0.0) {
            bail!("routed expert scale must be positive and finite");
        }
        for route in &routes {
            route.validate()?;
        }
        let mut weights: Vec<f32> = routes.iter().map(|route| route.score).collect();
        softmax_in_place(&mut weights);
        for weight in &mut weights {
            *weight *= routed_expert_scale;
        }
        Ok(Self {
            layer,
            routes,
            weights,
        })
    }

    pub fn from_scores(
        layer: usize,
        routes: &[(usize, f32)],
        routed_expert_scale: f32,
    ) -> Result<Self> {
        Self::from_routes(
            layer,
            ExpertRoute::from_scores(routes)?,
            routed_expert_scale,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledExpertBatch<T> {
    pub layer: usize,
    pub routes: Vec<ExpertRoute>,
    pub weights: Vec<f32>,
    pub experts: Arc<[T]>,
}

impl<T> ScheduledExpertBatch<T> {
    pub fn from_parts(
        routes: ScheduledExpertRoutes,
        experts: Vec<T>,
    ) -> Result<ScheduledExpertBatch<T>> {
        if experts.len() != routes.routes.len() {
            bail!(
                "scheduled expert batch has {} experts for {} routes on layer {}",
                experts.len(),
                routes.routes.len(),
                routes.layer
            );
        }
        if routes.weights.len() != routes.routes.len() {
            bail!(
                "scheduled expert batch has {} weights for {} routes on layer {}",
                routes.weights.len(),
                routes.routes.len(),
                routes.layer
            );
        }
        Ok(Self {
            layer: routes.layer,
            routes: routes.routes,
            weights: routes.weights,
            experts: Arc::from(experts),
        })
    }

    pub fn len(&self) -> usize {
        self.experts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ExpertSchedulerMetrics {
    issued_reads: u64,
    positioned_reads: u64,
    read_failures: u64,
    total_queue_latency: Duration,
    max_queue_latency: Duration,
    total_read_latency: Duration,
    max_read_latency: Duration,
    bytes_read: u64,
    warm_reads: u64,
    total_warm_read_latency: Duration,
    max_warm_read_latency: Duration,
    warm_bytes_read: u64,
}

impl ExpertSchedulerMetrics {
    pub(crate) fn record_issued_read(&mut self) {
        self.issued_reads = self.issued_reads.saturating_add(1);
    }

    pub(crate) fn record_positioned_read(&mut self) {
        self.positioned_reads = self.positioned_reads.saturating_add(1);
    }

    pub(crate) fn record_read_failure(&mut self) {
        self.read_failures = self.read_failures.saturating_add(1);
    }

    pub(crate) fn record_queue_latency(&mut self, latency: Duration) {
        self.total_queue_latency += latency;
        self.max_queue_latency = self.max_queue_latency.max(latency);
    }

    pub(crate) fn record_read_latency(&mut self, latency: Duration) {
        self.total_read_latency += latency;
        self.max_read_latency = self.max_read_latency.max(latency);
    }

    pub(crate) fn record_bytes_read(&mut self, bytes: u64) {
        self.bytes_read = self.bytes_read.saturating_add(bytes);
    }

    pub(crate) fn record_warm_read(&mut self, latency: Duration, bytes: u64) {
        self.warm_reads = self.warm_reads.saturating_add(1);
        self.total_warm_read_latency += latency;
        self.max_warm_read_latency = self.max_warm_read_latency.max(latency);
        self.warm_bytes_read = self.warm_bytes_read.saturating_add(bytes);
    }

    pub(crate) fn snapshot(&self) -> ExpertSchedulerSnapshot {
        ExpertSchedulerSnapshot {
            issued_reads: self.issued_reads,
            positioned_reads: self.positioned_reads,
            read_failures: self.read_failures,
            total_queue_latency: self.total_queue_latency,
            max_queue_latency: self.max_queue_latency,
            total_read_latency: self.total_read_latency,
            max_read_latency: self.max_read_latency,
            bytes_read: self.bytes_read,
            warm_reads: self.warm_reads,
            total_warm_read_latency: self.total_warm_read_latency,
            max_warm_read_latency: self.max_warm_read_latency,
            warm_bytes_read: self.warm_bytes_read,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertSchedulerSnapshot {
    pub issued_reads: u64,
    pub positioned_reads: u64,
    pub read_failures: u64,
    pub total_queue_latency: Duration,
    pub max_queue_latency: Duration,
    pub total_read_latency: Duration,
    pub max_read_latency: Duration,
    pub bytes_read: u64,
    pub warm_reads: u64,
    pub total_warm_read_latency: Duration,
    pub max_warm_read_latency: Duration,
    pub warm_bytes_read: u64,
}

impl ExpertSchedulerSnapshot {
    pub fn saturating_delta(self, before: Self) -> Self {
        Self {
            issued_reads: self.issued_reads.saturating_sub(before.issued_reads),
            positioned_reads: self
                .positioned_reads
                .saturating_sub(before.positioned_reads),
            read_failures: self.read_failures.saturating_sub(before.read_failures),
            total_queue_latency: self
                .total_queue_latency
                .saturating_sub(before.total_queue_latency),
            max_queue_latency: self.max_queue_latency,
            total_read_latency: self
                .total_read_latency
                .saturating_sub(before.total_read_latency),
            max_read_latency: self.max_read_latency,
            bytes_read: self.bytes_read.saturating_sub(before.bytes_read),
            warm_reads: self.warm_reads.saturating_sub(before.warm_reads),
            total_warm_read_latency: self
                .total_warm_read_latency
                .saturating_sub(before.total_warm_read_latency),
            max_warm_read_latency: self.max_warm_read_latency,
            warm_bytes_read: self.warm_bytes_read.saturating_sub(before.warm_bytes_read),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::flashmoe::{QWEN35_MODEL, QwenModelConfig, QwenMoeModelLayout};

    fn qwen35_layout() -> QwenMoeModelLayout {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{
  "model_type": "qwen3_5_moe",
  "architectures": ["Qwen3_5MoeForCausalLM"],
  "num_hidden_layers": 60,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 2,
  "vocab_size": 248320,
  "rope_theta": 10000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "moe_intermediate_size": 1024,
  "num_shared_experts": 1,
  "shared_expert_intermediate_size": 1024
}"#,
        )
        .unwrap();
        QwenMoeModelLayout::from_config(QWEN35_MODEL, &config).unwrap()
    }

    #[test]
    fn scheduled_graph_preserves_the_declared_stage_order() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

        let stages: Vec<_> = graph.stages().iter().map(|stage| stage.stage).collect();

        assert_eq!(stages, FlashMoeGraphStage::ALL);
        assert!(graph.declares_scheduler_owned_expert_reads());
    }

    #[test]
    fn scheduled_graph_exposes_the_upstream_command_sequence() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd_sequence: Vec<_> = graph
            .cmd_sequence()
            .iter()
            .map(|stage| stage.stage)
            .collect();

        assert_eq!(
            cmd_sequence,
            vec![
                FlashMoeGraphStage::DeferredPreviousCmd3,
                FlashMoeGraphStage::Cmd1AttentionProjections,
                FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection,
            ]
        );
        assert_eq!(
            graph
                .stage(FlashMoeGraphStage::RoutingSoftmaxTopK)
                .placement,
            FlashMoeStagePlacement::CpuDeclared
        );
    }

    #[test]
    fn scheduled_expert_routes_normalize_and_scale_scores() {
        let scheduled =
            ScheduledExpertRoutes::from_scores(12, &[(7, 1.0), (3, 2.0), (9, -1.0)], 0.25).unwrap();
        let mut expected = vec![1.0, 2.0, -1.0];
        softmax_in_place(&mut expected);
        for weight in &mut expected {
            *weight *= 0.25;
        }

        assert_eq!(scheduled.layer, 12);
        assert_eq!(
            scheduled.routes,
            vec![
                ExpertRoute {
                    expert: 7,
                    score: 1.0,
                },
                ExpertRoute {
                    expert: 3,
                    score: 2.0,
                },
                ExpertRoute {
                    expert: 9,
                    score: -1.0,
                },
            ]
        );
        assert_eq!(scheduled.weights.len(), expected.len());
        for (actual, expected) in scheduled.weights.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn scheduled_expert_routes_reject_non_finite_scores() {
        let err = ScheduledExpertRoutes::from_scores(0, &[(2, f32::NAN)], 1.0).unwrap_err();

        assert!(
            err.to_string()
                .contains("expert route score for expert 2 is not finite"),
            "{err:#}"
        );
    }

    #[test]
    fn scheduled_expert_batch_validates_route_weight_and_expert_counts() {
        let routes = ScheduledExpertRoutes::from_scores(3, &[(8, 2.0), (4, 1.0)], 1.0).unwrap();
        let batch = ScheduledExpertBatch::from_parts(routes, vec!["expert-8", "expert-4"]).unwrap();

        assert_eq!(batch.layer, 3);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert_eq!(batch.experts.as_ref(), ["expert-8", "expert-4"]);

        let routes = ScheduledExpertRoutes::from_scores(3, &[(8, 2.0), (4, 1.0)], 1.0).unwrap();
        let err = ScheduledExpertBatch::from_parts(routes, vec!["expert-8"]).unwrap_err();
        assert!(
            err.to_string()
                .contains("scheduled expert batch has 1 experts for 2 routes"),
            "{err:#}"
        );
    }

    #[test]
    fn expert_scheduler_metrics_snapshot_reports_saturating_delta() {
        let mut metrics = ExpertSchedulerMetrics::default();
        metrics.record_issued_read();
        metrics.record_positioned_read();
        metrics.record_queue_latency(Duration::from_millis(7));
        metrics.record_read_latency(Duration::from_millis(11));
        metrics.record_bytes_read(128);
        let before = metrics.snapshot();

        metrics.record_issued_read();
        metrics.record_read_failure();
        metrics.record_queue_latency(Duration::from_millis(3));
        metrics.record_read_latency(Duration::from_millis(5));
        metrics.record_bytes_read(32);
        metrics.record_warm_read(Duration::from_millis(5), 32);

        let delta = metrics.snapshot().saturating_delta(before);
        assert_eq!(delta.issued_reads, 1);
        assert_eq!(delta.positioned_reads, 0);
        assert_eq!(delta.read_failures, 1);
        assert_eq!(delta.total_queue_latency, Duration::from_millis(3));
        assert_eq!(delta.total_read_latency, Duration::from_millis(5));
        assert_eq!(delta.bytes_read, 32);
        assert_eq!(delta.warm_reads, 1);
        assert_eq!(delta.warm_bytes_read, 32);
    }
}
