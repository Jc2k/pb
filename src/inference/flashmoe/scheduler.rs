use super::capabilities::{
    FlashMoeCapabilityPlan, FlashMoeGraphStage, FlashMoeStageCapability, FlashMoeStagePlacement,
    FlashMoeUnsupportedCapability,
};
use super::experts::{
    ExpertRawRead, ExpertRawReadResponse, ExpertReadPath, ExpertSlotDescriptor,
    FLASHMOE_EXPERT_IO_POLICY,
};
use super::math::softmax_in_place;
use super::model_family::QwenMoeFamily;
use anyhow::{Result, bail};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeScheduledGraph {
    family: QwenMoeFamily,
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
        Ok(Self {
            family: capabilities.family,
            stages,
        })
    }

    pub fn family(&self) -> QwenMoeFamily {
        self.family
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

    pub fn build_cmd2_post_attention(
        &self,
        layer: usize,
        active_experts: usize,
        attention: ScheduledCmd2AttentionSource,
        residual: ScheduledCmd2ResidualSource,
    ) -> Result<ScheduledCmd2PostAttention, FlashMoeUnsupportedCapability> {
        let stage = *self.stage(FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection);
        if stage.placement != FlashMoeStagePlacement::Metal {
            return Err(FlashMoeUnsupportedCapability::new(
                self.family,
                stage.stage,
                "CMD2 post-attention stage must be implemented as a declared Metal command",
            ));
        }
        Ok(ScheduledCmd2PostAttention {
            stage,
            layer,
            active_experts,
            attention,
            residual,
        })
    }

    pub fn build_cmd3_expert_phase(
        &self,
        layer: usize,
        expert_count: usize,
        input: ScheduledCmd3InputSource,
        shared: ScheduledSharedExpertSource,
        next_norm: ScheduledNextNormSource,
    ) -> Result<ScheduledCmd3ExpertPhase, FlashMoeUnsupportedCapability> {
        let stage = *self.stage(FlashMoeGraphStage::Cmd3ExpertAndSharedCombine);
        if stage.placement != FlashMoeStagePlacement::Metal {
            return Err(FlashMoeUnsupportedCapability::new(
                self.family,
                stage.stage,
                "CMD3 expert/shared combine must be implemented as a declared Metal command",
            ));
        }
        Ok(ScheduledCmd3ExpertPhase {
            stage,
            layer,
            expert_count,
            input,
            shared,
            next_norm,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledCmd2AttentionSource {
    CpuAttentionValues,
    MetalAttentionValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledCmd2ResidualSource {
    CpuHidden,
    MetalBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledCmd2PostAttention {
    pub stage: FlashMoeStageCapability,
    pub layer: usize,
    pub active_experts: usize,
    pub attention: ScheduledCmd2AttentionSource,
    pub residual: ScheduledCmd2ResidualSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledCmd3InputSource {
    MetalPostAttentionPrep,
    CpuNormedResidualUpload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledSharedExpertSource {
    None,
    DenseCpuWeights,
    ResidentQ4Projections,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledNextNormSource {
    None,
    CpuVisibleWeights,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledCmd3ExpertPhase {
    pub stage: FlashMoeStageCapability,
    pub layer: usize,
    pub expert_count: usize,
    pub input: ScheduledCmd3InputSource,
    pub shared: ScheduledSharedExpertSource,
    pub next_norm: ScheduledNextNormSource,
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

pub type ScheduledExpertSet<T> = ScheduledExpertBatch<T>;

pub(crate) struct PendingScheduledRead<T> {
    id: u64,
    rx: mpsc::Receiver<T>,
}

impl<T> fmt::Debug for PendingScheduledRead<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingScheduledRead")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<T> PendingScheduledRead<T> {
    pub(crate) fn new(id: u64, rx: mpsc::Receiver<T>) -> Self {
        Self { id, rx }
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn recv(self) -> Result<T, mpsc::RecvError> {
        self.rx.recv()
    }
}

pub(crate) struct PendingScheduledExpertSet<T> {
    layer: usize,
    routes: Vec<ExpertRoute>,
    reads: Vec<PendingScheduledRead<T>>,
}

impl<T> fmt::Debug for PendingScheduledExpertSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingScheduledExpertSet")
            .field("layer", &self.layer)
            .field("routes", &self.routes)
            .field("read_count", &self.reads.len())
            .finish()
    }
}

impl<T> PendingScheduledExpertSet<T> {
    pub(crate) fn new(
        layer: usize,
        routes: Vec<ExpertRoute>,
        reads: Vec<PendingScheduledRead<T>>,
    ) -> Self {
        Self {
            layer,
            routes,
            reads,
        }
    }

    pub(crate) fn into_parts(self) -> (usize, Vec<ExpertRoute>, Vec<PendingScheduledRead<T>>) {
        (self.layer, self.routes, self.reads)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ExpertReadKey {
    pub(crate) layer: usize,
    pub(crate) expert: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledExpertReadIssue {
    pub(crate) id: u64,
    pub(crate) key: ExpertReadKey,
    pub(crate) warm: bool,
    pub(crate) issued_at: Instant,
}

#[derive(Debug)]
pub(crate) struct ScheduledExpertReadResponse<T> {
    pub(crate) id: u64,
    pub(crate) queue_latency: Duration,
    pub(crate) read_path: ExpertReadPath,
    pub(crate) read_latency: Duration,
    pub(crate) bytes_read: u64,
    pub(crate) warm: bool,
    pub(crate) result: Result<T>,
}

#[derive(Debug)]
pub(crate) struct ScheduledExpertSlot {
    raw: ExpertRawRead,
}

impl ScheduledExpertSlot {
    fn from_raw(raw: ExpertRawRead) -> Self {
        Self { raw }
    }

    pub(crate) fn layer(&self) -> usize {
        self.raw.layer
    }

    pub(crate) fn expert(&self) -> usize {
        self.raw.expert
    }

    pub(crate) fn descriptor(&self) -> ExpertSlotDescriptor {
        self.raw.slot
    }

    pub(crate) fn into_raw(self) -> ExpertRawRead {
        self.raw
    }
}

#[derive(Debug)]
pub(crate) struct ActiveExpertReadScheduler {
    metrics: ExpertSchedulerMetrics,
    seen_reads: BTreeSet<ExpertReadKey>,
    next_read_id: u64,
    routed_expert_scale: f32,
}

impl ActiveExpertReadScheduler {
    pub(crate) fn new(routed_expert_scale: f32) -> Self {
        assert_eq!(
            FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
            ExpertReadPath::PositionedRead,
            "expert files must be read with positioned reads"
        );
        assert!(
            routed_expert_scale.is_finite() && routed_expert_scale > 0.0,
            "routed expert scale must be positive and finite"
        );
        assert!(
            !FLASHMOE_EXPERT_IO_POLICY.application_expert_cache,
            "do not add an application-level expert cache; trust the OS page cache"
        );
        assert!(
            !FLASHMOE_EXPERT_IO_POLICY.lz4_expert_compression,
            "do not add LZ4 expert compression"
        );
        assert!(
            !FLASHMOE_EXPERT_IO_POLICY.speculative_routing,
            "do not add speculative expert routing"
        );
        assert!(
            !FLASHMOE_EXPERT_IO_POLICY.broad_ssd_gpu_overlap,
            "do not broadly overlap SSD expert reads with GPU compute"
        );
        Self {
            metrics: ExpertSchedulerMetrics::default(),
            seen_reads: BTreeSet::new(),
            next_read_id: 0,
            routed_expert_scale,
        }
    }

    pub(crate) fn issue_read(&mut self, layer: usize, expert: usize) -> ScheduledExpertReadIssue {
        let key = ExpertReadKey { layer, expert };
        let warm = !self.seen_reads.insert(key);
        self.metrics.record_issued_read();
        let id = self.next_read_id;
        self.next_read_id = self.next_read_id.wrapping_add(1);
        ScheduledExpertReadIssue {
            id,
            key,
            warm,
            issued_at: Instant::now(),
        }
    }

    pub(crate) fn finish_read<T>(
        &mut self,
        pending_id: u64,
        response: ScheduledExpertReadResponse<T>,
    ) -> Result<T> {
        if response.id != pending_id {
            self.metrics.record_read_failure();
            bail!(
                "expert I/O worker returned response {} for pending read {}",
                response.id,
                pending_id
            );
        }
        self.metrics.record_queue_latency(response.queue_latency);
        match response.read_path {
            ExpertReadPath::PositionedRead => {
                self.metrics.record_positioned_read();
            }
        }
        self.metrics.record_read_latency(response.read_latency);
        self.metrics.record_bytes_read(response.bytes_read);
        if response.warm {
            self.metrics
                .record_warm_read(response.read_latency, response.bytes_read);
        }
        response.result.inspect_err(|_| {
            self.metrics.record_read_failure();
        })
    }

    pub(crate) fn finish_slot_read(
        &mut self,
        pending_id: u64,
        response: ExpertRawReadResponse,
    ) -> Result<ScheduledExpertSlot> {
        self.finish_read(
            pending_id,
            ScheduledExpertReadResponse {
                id: response.id,
                queue_latency: response.queue_latency,
                read_path: response.read_path,
                read_latency: response.read_latency,
                bytes_read: response.bytes_read,
                warm: response.warm,
                result: response.result.map(ScheduledExpertSlot::from_raw),
            },
        )
    }

    pub(crate) fn finish_routes<T>(
        &mut self,
        layer: usize,
        routes: Vec<ExpertRoute>,
        experts: Vec<T>,
        mut identify: impl FnMut(&T) -> (usize, usize),
    ) -> Result<ScheduledExpertSet<T>> {
        let scheduled_routes =
            ScheduledExpertRoutes::from_routes(layer, routes, self.routed_expert_scale)?;
        if experts.len() != scheduled_routes.routes.len() {
            bail!(
                "expert scheduler returned {} experts for {} routed entries on layer {}",
                experts.len(),
                scheduled_routes.routes.len(),
                scheduled_routes.layer
            );
        }
        for (route, expert) in scheduled_routes.routes.iter().zip(experts.iter()) {
            let (expert_layer, expert_id) = identify(expert);
            if expert_layer != scheduled_routes.layer || expert_id != route.expert {
                bail!(
                    "expert scheduler returned layer {} expert {} for routed layer {} expert {}",
                    expert_layer,
                    expert_id,
                    scheduled_routes.layer,
                    route.expert
                );
            }
        }
        ScheduledExpertSet::from_parts(scheduled_routes, experts)
    }

    pub(crate) fn snapshot(&self) -> ExpertSchedulerSnapshot {
        self.metrics.snapshot()
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
    use crate::inference::flashmoe::experts::{
        ExpertPackMetadata, ExpertRawPayload, FixedQ4ExpertSlotSpec,
    };
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

    fn raw_pbq4_read(layer: usize, expert: usize, payload: Vec<u8>) -> ExpertRawRead {
        let layout = qwen35_layout();
        ExpertRawRead {
            layer,
            expert,
            slot: ExpertSlotDescriptor {
                layer,
                expert,
                slot_offset: 1024,
                slot_capacity: payload.len(),
                payload_len: payload.len(),
            },
            metadata: ExpertPackMetadata {
                layer,
                expert,
                packed_bytes: payload.len() as u64,
                records: Vec::new(),
            },
            fixed_q4: FixedQ4ExpertSlotSpec::new(
                layout.q4_expert_layout,
                layout.hidden_size,
                layout.moe_intermediate_size,
            )
            .unwrap(),
            recycle_pool: None,
            payload: ExpertRawPayload::Pbq4(payload),
            read_latency: Duration::from_millis(7),
            read_path: ExpertReadPath::PositionedRead,
        }
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
    fn scheduled_graph_builds_explicit_cmd2_and_cmd3_descriptors() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

        let cmd2 = graph
            .build_cmd2_post_attention(
                14,
                4,
                ScheduledCmd2AttentionSource::MetalAttentionValues,
                ScheduledCmd2ResidualSource::MetalBuffer,
            )
            .unwrap();
        let cmd3 = graph
            .build_cmd3_expert_phase(
                14,
                4,
                ScheduledCmd3InputSource::MetalPostAttentionPrep,
                ScheduledSharedExpertSource::ResidentQ4Projections,
                ScheduledNextNormSource::CpuVisibleWeights,
            )
            .unwrap();

        assert_eq!(
            cmd2.stage.stage,
            FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection
        );
        assert_eq!(cmd2.stage.placement, FlashMoeStagePlacement::Metal);
        assert_eq!(cmd2.layer, 14);
        assert_eq!(cmd2.active_experts, 4);
        assert_eq!(
            cmd2.attention,
            ScheduledCmd2AttentionSource::MetalAttentionValues
        );
        assert_eq!(cmd2.residual, ScheduledCmd2ResidualSource::MetalBuffer);
        assert_eq!(
            cmd3.stage.stage,
            FlashMoeGraphStage::Cmd3ExpertAndSharedCombine
        );
        assert_eq!(cmd3.stage.placement, FlashMoeStagePlacement::Metal);
        assert_eq!(cmd3.layer, 14);
        assert_eq!(cmd3.expert_count, 4);
        assert_eq!(cmd3.input, ScheduledCmd3InputSource::MetalPostAttentionPrep);
        assert_eq!(
            cmd3.shared,
            ScheduledSharedExpertSource::ResidentQ4Projections
        );
        assert_eq!(cmd3.next_norm, ScheduledNextNormSource::CpuVisibleWeights);
    }

    #[test]
    fn scheduled_graph_rejects_non_metal_cmd3_builder() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let mut graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd3 = graph
            .stages
            .iter_mut()
            .find(|stage| stage.stage == FlashMoeGraphStage::Cmd3ExpertAndSharedCombine)
            .unwrap();
        cmd3.placement = FlashMoeStagePlacement::CpuDeclared;

        let err = graph
            .build_cmd3_expert_phase(
                0,
                4,
                ScheduledCmd3InputSource::CpuNormedResidualUpload,
                ScheduledSharedExpertSource::DenseCpuWeights,
                ScheduledNextNormSource::None,
            )
            .unwrap_err();

        assert_eq!(err.family, graph.family());
        assert_eq!(err.stage, FlashMoeGraphStage::Cmd3ExpertAndSharedCombine);
        assert!(
            err.to_string()
                .contains("CMD3 expert/shared combine must be implemented"),
            "{err:#}"
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
    fn pending_scheduled_expert_set_owns_read_receivers_and_routes() {
        let (tx, rx) = mpsc::channel();
        let read = PendingScheduledRead::new(77, rx);
        assert_eq!(read.id(), 77);
        let pending = PendingScheduledExpertSet::new(
            5,
            vec![ExpertRoute {
                expert: 9,
                score: 1.25,
            }],
            vec![read],
        );

        tx.send("expert-9").unwrap();
        let (layer, routes, reads) = pending.into_parts();

        assert_eq!(layer, 5);
        assert_eq!(
            routes,
            vec![ExpertRoute {
                expert: 9,
                score: 1.25
            }]
        );
        assert_eq!(reads.len(), 1);
        assert_eq!(
            reads.into_iter().next().unwrap().recv().unwrap(),
            "expert-9"
        );
    }

    #[test]
    fn active_expert_scheduler_issues_ids_and_marks_repeated_reads_warm() {
        let mut scheduler = ActiveExpertReadScheduler::new(1.0);

        let cold = scheduler.issue_read(4, 7);
        let warm = scheduler.issue_read(4, 7);

        assert_eq!(cold.id, 0);
        assert_eq!(
            cold.key,
            ExpertReadKey {
                layer: 4,
                expert: 7
            }
        );
        assert!(!cold.warm);
        assert_eq!(warm.id, 1);
        assert_eq!(warm.key, cold.key);
        assert!(warm.warm);
        assert_eq!(scheduler.snapshot().issued_reads, 2);
    }

    #[test]
    fn active_expert_scheduler_finishes_responses_and_records_failures() {
        let mut scheduler = ActiveExpertReadScheduler::new(0.5);
        let first = scheduler.issue_read(2, 9);
        let value = scheduler
            .finish_read(
                first.id,
                ScheduledExpertReadResponse {
                    id: first.id,
                    queue_latency: Duration::from_millis(2),
                    read_path: ExpertReadPath::PositionedRead,
                    read_latency: Duration::from_millis(5),
                    bytes_read: 128,
                    warm: first.warm,
                    result: Ok("expert-9"),
                },
            )
            .unwrap();

        assert_eq!(value, "expert-9");
        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.positioned_reads, 1);
        assert_eq!(snapshot.bytes_read, 128);
        assert_eq!(snapshot.read_failures, 0);

        let second = scheduler.issue_read(2, 10);
        let err = scheduler
            .finish_read(
                second.id,
                ScheduledExpertReadResponse {
                    id: second.id + 1,
                    queue_latency: Duration::ZERO,
                    read_path: ExpertReadPath::PositionedRead,
                    read_latency: Duration::ZERO,
                    bytes_read: 0,
                    warm: false,
                    result: Ok("wrong-id"),
                },
            )
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("returned response 2 for pending read 1"),
            "{err:#}"
        );
        assert_eq!(scheduler.snapshot().read_failures, 1);
    }

    #[test]
    fn active_expert_scheduler_finishes_raw_reads_as_scheduled_slots() {
        let mut scheduler = ActiveExpertReadScheduler::new(1.0);
        let issue = scheduler.issue_read(3, 8);

        let slot = scheduler
            .finish_slot_read(
                issue.id,
                ExpertRawReadResponse {
                    id: issue.id,
                    queue_latency: Duration::from_millis(1),
                    read_path: ExpertReadPath::PositionedRead,
                    read_latency: Duration::from_millis(7),
                    bytes_read: 3,
                    warm: issue.warm,
                    result: Ok(raw_pbq4_read(3, 8, vec![1, 2, 3])),
                },
            )
            .unwrap();

        assert_eq!(slot.layer(), 3);
        assert_eq!(slot.expert(), 8);
        assert_eq!(
            slot.descriptor(),
            ExpertSlotDescriptor {
                layer: 3,
                expert: 8,
                slot_offset: 1024,
                slot_capacity: 3,
                payload_len: 3,
            }
        );
        let raw = slot.into_raw();
        assert!(matches!(raw.payload, ExpertRawPayload::Pbq4(_)));
        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.issued_reads, 1);
        assert_eq!(snapshot.positioned_reads, 1);
        assert_eq!(snapshot.bytes_read, 3);
        assert_eq!(snapshot.read_failures, 0);
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
