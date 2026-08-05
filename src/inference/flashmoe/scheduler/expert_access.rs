#![cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]

use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ExpertRoute {
    pub(crate) expert: usize,
    pub(crate) score: f32,
}

impl ExpertRoute {
    pub(crate) fn from_pair((expert, score): (usize, f32)) -> Self {
        Self { expert, score }
    }

    pub(crate) fn from_scores(routes: &[(usize, f32)]) -> Result<InlineExpertRoutes> {
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
struct ValidatedRouteWeights {
    routes: InlineExpertRoutes,
    weights: InlineRouteWeights,
}

impl ValidatedRouteWeights {
    fn new(routes: InlineExpertRoutes, weights: InlineRouteWeights) -> Self {
        debug_assert_eq!(routes.len(), weights.len());
        Self { routes, weights }
    }

    fn routes(&self) -> &[ExpertRoute] {
        &self.routes
    }

    fn weights(&self) -> &[f32] {
        &self.weights
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScheduledExpertRoutes {
    pub(crate) layer: usize,
    route_weights: ValidatedRouteWeights,
}

impl ScheduledExpertRoutes {
    #[cfg(test)]
    pub(crate) fn from_routes(
        layer: usize,
        routes: impl Into<InlineExpertRoutes>,
        routed_expert_scale: f32,
    ) -> Result<Self> {
        Self::from_routes_with_policy(
            layer,
            routes,
            QwenMoeRoutingWeightNormalization::RenormalizeSelected,
            routed_expert_scale,
        )
    }

    pub(crate) fn from_routes_with_policy(
        layer: usize,
        routes: impl Into<InlineExpertRoutes>,
        normalization: QwenMoeRoutingWeightNormalization,
        routed_expert_scale: f32,
    ) -> Result<Self> {
        let routes = routes.into();
        if !(routed_expert_scale.is_finite() && routed_expert_scale > 0.0) {
            bail!("routed expert scale must be positive and finite");
        }
        for route in &routes {
            route.validate()?;
        }
        let mut weights: InlineRouteWeights = routes.iter().map(|route| route.score).collect();
        match normalization {
            QwenMoeRoutingWeightNormalization::RenormalizeSelected => {
                let sum = weights.iter().sum::<f32>();
                if !(sum.is_finite() && sum > 0.0) {
                    bail!("selected expert probabilities must have a positive finite sum");
                }
                let inverse_sum = sum.recip();
                for weight in &mut weights {
                    if *weight < 0.0 {
                        bail!("selected expert probabilities must be non-negative");
                    }
                    *weight *= inverse_sum;
                }
            }
            QwenMoeRoutingWeightNormalization::DeepSeekRenormalizeSelectedWithFloor => {
                const DEEPSEEK_SELECTED_SUM_FLOOR: f32 = 6.103515625e-5;
                let sum = weights.iter().sum::<f32>();
                if !(sum.is_finite() && sum >= 0.0) {
                    bail!(
                        "DeepSeek selected expert probabilities must have a finite non-negative sum"
                    );
                }
                let inverse_sum = sum.max(DEEPSEEK_SELECTED_SUM_FLOOR).recip();
                for weight in &mut weights {
                    if *weight < 0.0 {
                        bail!("DeepSeek selected expert probabilities must be non-negative");
                    }
                    *weight *= inverse_sum;
                }
            }
            QwenMoeRoutingWeightNormalization::PreserveFullSoftmax => bail!(
                "FlashMoe unsupported routing weights: preserving probabilities from the full expert softmax requires a declared scheduler implementation"
            ),
        }
        for weight in &mut weights {
            *weight *= routed_expert_scale;
        }
        Ok(Self {
            layer,
            route_weights: ValidatedRouteWeights::new(routes, weights),
        })
    }

    #[cfg(test)]
    pub(crate) fn from_scores(
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

    #[cfg(test)]
    pub(crate) fn from_routing_command(
        command: &ScheduledRoutingCommand,
        routed_expert_scale: f32,
    ) -> Result<Self> {
        command.validate_for_active_expert_issue()?;
        Self::from_scores(command.layer, &command.routes, routed_expert_scale)
    }

    pub(crate) fn from_routing_command_with_policy(
        command: &ScheduledRoutingCommand,
        normalization: QwenMoeRoutingWeightNormalization,
        routed_expert_scale: f32,
    ) -> Result<Self> {
        command.validate_for_active_expert_issue()?;
        Self::from_routes_with_policy(
            command.layer,
            ExpertRoute::from_scores(&command.routes)?,
            normalization,
            routed_expert_scale,
        )
    }

    #[cfg(test)]
    pub(crate) fn expert_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.routes().iter().map(|route| route.expert)
    }

    pub(crate) fn routes(&self) -> &[ExpertRoute] {
        self.route_weights.routes()
    }

    pub(crate) fn weights(&self) -> &[f32] {
        self.route_weights.weights()
    }

    #[cfg(test)]
    pub(crate) fn uses_inline_storage(&self) -> bool {
        !self.route_weights.routes.spilled() && !self.route_weights.weights.spilled()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScheduledExpertBatch<T> {
    pub(crate) layer: usize,
    route_weights: ValidatedRouteWeights,
    pub(crate) experts: Arc<[T]>,
}

impl<T> ScheduledExpertBatch<T> {
    pub(crate) fn from_parts(
        routes: ScheduledExpertRoutes,
        experts: Vec<T>,
    ) -> Result<ScheduledExpertBatch<T>> {
        if experts.len() != routes.routes().len() {
            bail!(
                "scheduled expert batch has {} experts for {} routes on layer {}",
                experts.len(),
                routes.routes().len(),
                routes.layer
            );
        }
        Ok(Self {
            layer: routes.layer,
            route_weights: routes.route_weights,
            experts: Arc::from(experts),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.experts.len()
    }

    pub(crate) fn routes(&self) -> &[ExpertRoute] {
        self.route_weights.routes()
    }

    pub(crate) fn weights(&self) -> &[f32] {
        self.route_weights.weights()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }
}

impl<T> ScheduledExpertBatch<T>
where
    T: ScheduledCmd3ExpertPayload,
{
    pub(crate) fn cmd3_expert_phase_payloads(
        &self,
        width: usize,
    ) -> Result<Vec<ScheduledExpertPhaseMlpPayload<'_>>> {
        self.experts
            .iter()
            .map(|expert| expert.scheduled_cmd3_expert_phase_payload(width))
            .collect()
    }
}

pub(crate) type ScheduledExpertSet<T> = ScheduledExpertBatch<T>;

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
    routes: ScheduledExpertRoutes,
    reads: Vec<PendingScheduledRead<T>>,
}

#[derive(Debug)]
pub(crate) enum PendingScheduledExpertAccess {
    Streamed(PendingScheduledExpertSet<ExpertRawReadResponse>),
    Resident(ScheduledExpertSet<Arc<ScheduledExpertSlot>>),
}

impl<T> fmt::Debug for PendingScheduledExpertSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingScheduledExpertSet")
            .field("layer", &self.routes.layer)
            .field("routes", &self.routes)
            .field("read_count", &self.reads.len())
            .finish()
    }
}

impl<T> PendingScheduledExpertSet<T> {
    pub(crate) fn new(routes: ScheduledExpertRoutes, reads: Vec<PendingScheduledRead<T>>) -> Self {
        Self { routes, reads }
    }

    pub(crate) fn into_parts(self) -> (ScheduledExpertRoutes, Vec<PendingScheduledRead<T>>) {
        (self.routes, self.reads)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ScheduledExpertReadSet {
    routes: ScheduledExpertRoutes,
    issues: Vec<ScheduledExpertReadIssue>,
}

impl ScheduledExpertReadSet {
    pub(crate) fn layer(&self) -> usize {
        self.routes.layer
    }

    pub(crate) fn len(&self) -> usize {
        self.issues.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.issues.is_empty()
    }

    pub(crate) fn issues(&self) -> &[ScheduledExpertReadIssue] {
        &self.issues
    }

    pub(crate) fn into_routes(self) -> ScheduledExpertRoutes {
        self.routes
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
    pub(crate) raw: ExpertRawRead,
}

impl ScheduledExpertSlot {
    pub(crate) fn from_raw(raw: ExpertRawRead) -> Self {
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

    pub(crate) fn mix_hash(&self) -> u64 {
        let mut hash = ((self.layer() as u64) << 32) ^ self.expert() as u64;
        let prefix = match &self.raw.payload {
            ExpertRawPayload::Pbq4(bytes) => bytes.as_slice(),
            ExpertRawPayload::FixedQ4(fixed_q4) => fixed_q4.bytes.as_slice(),
            ExpertRawPayload::FixedDense(fixed_dense) => fixed_dense.bytes.as_slice(),
            ExpertRawPayload::FixedDeepSeekGguf(deepseek) => deepseek.bytes.as_slice(),
        };
        for byte in prefix.iter().take(4096) {
            hash = hash.rotate_left(5) ^ u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash
    }

    fn resident_backing(&self) -> Result<&ReusableExpertBytes> {
        if self.raw.read_path != ExpertReadPath::ResidentMapped {
            bail!(
                "FlashMoe resident expert table received non-resident layer {} expert {} payload",
                self.layer(),
                self.expert()
            );
        }
        match &self.raw.payload {
            ExpertRawPayload::FixedQ4(payload) => Ok(&payload.bytes),
            ExpertRawPayload::FixedDense(payload) => Ok(&payload.bytes),
            ExpertRawPayload::Pbq4(_) => {
                bail!("FlashMoe resident expert table cannot retain PBQ4 compatibility payloads")
            }
            ExpertRawPayload::FixedDeepSeekGguf(_) => bail!(
                "FlashMoe resident expert table is not a declared DeepSeek graph implementation"
            ),
        }
    }
}

impl ScheduledCmd3Expert for ScheduledExpertSlot {
    fn scheduled_expert_layer(&self) -> usize {
        self.layer()
    }

    fn scheduled_expert_id(&self) -> usize {
        self.expert()
    }

    fn scheduled_expert_slot_descriptor(&self) -> ExpertSlotDescriptor {
        self.descriptor()
    }
}

impl ScheduledCmd3ExpertPayload for ScheduledExpertSlot {
    fn scheduled_cmd3_expert_phase_payload(
        &self,
        width: usize,
    ) -> Result<ScheduledExpertPhaseMlpPayload<'_>> {
        match &self.raw.payload {
            ExpertRawPayload::FixedQ4(fixed_q4) => {
                let gate = fixed_q4.matvec_payload(
                    ExpertMlpProjection::Gate,
                    width,
                    fixed_q4.spec.intermediate_size,
                );
                let up = fixed_q4.matvec_payload(
                    ExpertMlpProjection::Up,
                    width,
                    fixed_q4.spec.intermediate_size,
                );
                let Some((gate, up)) = gate.zip(up) else {
                    bail!(
                        "FlashMoe unsupported active expert CMD3 path: scheduler-owned fixed-Q4 slot layer {} expert {} does not provide gate/up payloads for width {width}",
                        self.layer(),
                        self.expert()
                    );
                };
                let Some(down) =
                    fixed_q4.matvec_payload(ExpertMlpProjection::Down, gate.rows, width)
                else {
                    bail!(
                        "FlashMoe unsupported active expert CMD3 path: scheduler-owned fixed-Q4 slot layer {} expert {} does not provide down payload for width {width}",
                        self.layer(),
                        self.expert()
                    );
                };
                Ok(ScheduledExpertPhaseMlpPayload::Q4(
                    ScheduledQ4ExpertPhaseMlpPayload::new(
                        self.layer(),
                        self.expert(),
                        width,
                        gate,
                        up,
                        down,
                    )?,
                ))
            }
            ExpertRawPayload::FixedDense(fixed_dense) => {
                let intermediate = fixed_dense.spec.intermediate_size;
                let gate =
                    fixed_dense.matvec_payload(ExpertMlpProjection::Gate, width, intermediate)?;
                let up =
                    fixed_dense.matvec_payload(ExpertMlpProjection::Up, width, intermediate)?;
                let down =
                    fixed_dense.matvec_payload(ExpertMlpProjection::Down, intermediate, width)?;
                Ok(ScheduledExpertPhaseMlpPayload::Dense(
                    ScheduledDenseExpertPhaseMlpPayload::new(
                        self.layer(),
                        self.expert(),
                        width,
                        gate,
                        up,
                        down,
                    )?,
                ))
            }
            ExpertRawPayload::FixedDeepSeekGguf(deepseek) => {
                Ok(ScheduledExpertPhaseMlpPayload::DeepSeekGguf(
                    ScheduledDeepSeekGgufExpertPhaseMlpPayload::new(
                        self.layer(),
                        self.expert(),
                        deepseek.spec,
                        &deepseek.bytes,
                        width,
                    )?,
                ))
            }
            ExpertRawPayload::Pbq4(_) => {
                bail!(
                    "FlashMoe unsupported active expert CMD3 path: scheduler-owned layer {} expert {} slot contains PBQ4/component import data instead of a resolved whole-expert payload",
                    self.layer(),
                    self.expert()
                )
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct ActiveExpertReadScheduler {
    metrics: ExpertSchedulerMetrics,
    seen_reads: BTreeSet<ExpertReadKey>,
    next_read_id: u64,
    routing_weight_normalization: QwenMoeRoutingWeightNormalization,
    routed_expert_scale: f32,
}

impl ActiveExpertReadScheduler {
    #[cfg(test)]
    pub(crate) fn new(routed_expert_scale: f32) -> Self {
        Self::new_with_routing_policy(
            QwenMoeRoutingWeightNormalization::RenormalizeSelected,
            routed_expert_scale,
        )
    }

    pub(crate) fn new_with_routing_policy(
        routing_weight_normalization: QwenMoeRoutingWeightNormalization,
        routed_expert_scale: f32,
    ) -> Self {
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
        assert!(
            FLASHMOE_EXPERT_IO_POLICY.layer_ahead_request_staging,
            "the resolved saturated-batch graph requires one-layer-ahead request staging"
        );
        Self {
            metrics: ExpertSchedulerMetrics::default(),
            seen_reads: BTreeSet::new(),
            next_read_id: 0,
            routing_weight_normalization,
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
            ExpertReadPath::ResidentMapped => {}
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

    pub(crate) fn scheduled_routes_from_command(
        &self,
        command: &ScheduledRoutingCommand,
    ) -> Result<ScheduledExpertRoutes> {
        ScheduledExpertRoutes::from_routing_command_with_policy(
            command,
            self.routing_weight_normalization,
            self.routed_expert_scale,
        )
    }

    pub(crate) fn issue_routed_reads(
        &mut self,
        command: &ScheduledRoutingCommand,
    ) -> Result<ScheduledExpertReadSet> {
        let routes = self.scheduled_routes_from_command(command)?;
        let issues = routes
            .routes()
            .iter()
            .map(|route| self.issue_read(routes.layer, route.expert))
            .collect();
        Ok(ScheduledExpertReadSet { routes, issues })
    }

    pub(crate) fn finish_routes<T>(
        &mut self,
        scheduled_routes: ScheduledExpertRoutes,
        experts: Vec<T>,
        mut identify: impl FnMut(&T) -> (usize, usize),
    ) -> Result<ScheduledExpertSet<T>> {
        if experts.len() != scheduled_routes.routes().len() {
            bail!(
                "expert scheduler returned {} experts for {} routed entries on layer {}",
                experts.len(),
                scheduled_routes.routes().len(),
                scheduled_routes.layer
            );
        }
        for (route, expert) in scheduled_routes.routes().iter().zip(experts.iter()) {
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

#[derive(Debug)]
pub(crate) struct ScheduledResidentExpertTable {
    first_expert_layer: usize,
    experts_per_layer: usize,
    slots: Vec<Arc<ScheduledExpertSlot>>,
    core: ActiveExpertReadScheduler,
}

impl ScheduledResidentExpertTable {
    pub(crate) fn new(
        graph: &FlashMoeScheduledGraph,
        store: ExpertSlotStore,
        bind_resident: &mut impl FnMut(&ReusableExpertBytes) -> Result<()>,
    ) -> Result<Self> {
        if graph.expert_storage == ExpertStorageLayout::FixedDeepSeekGguf {
            bail!("DeepSeek expert storage has no resident mapped graph implementation");
        }
        let raw_slots = store.map_resident_slots(
            graph.layers(),
            graph.first_expert_layer(),
            graph.experts_per_layer(),
        )?;
        let expected_slots = graph
            .layers()
            .checked_sub(graph.first_expert_layer())
            .context("resident first expert layer exceeds scheduled graph layer count")?
            .checked_mul(graph.experts_per_layer())
            .context("resident scheduled expert slot count overflow")?;
        if raw_slots.len() != expected_slots {
            bail!(
                "resident expert table prepared {} slots, expected {expected_slots}",
                raw_slots.len()
            );
        }
        let mut slots = Vec::with_capacity(raw_slots.len());
        for raw in raw_slots {
            let slot = ScheduledExpertSlot::from_raw(raw);
            bind_resident(slot.resident_backing()?)?;
            slots.push(Arc::new(slot));
        }
        Ok(Self {
            first_expert_layer: graph.first_expert_layer(),
            experts_per_layer: graph.experts_per_layer(),
            slots,
            core: ActiveExpertReadScheduler::new_with_routing_policy(
                graph.routing_weight_normalization(),
                graph.routed_expert_scale(),
            ),
        })
    }

    fn slot(&self, layer: usize, expert: usize) -> Result<Arc<ScheduledExpertSlot>> {
        let relative_layer = layer
            .checked_sub(self.first_expert_layer)
            .with_context(|| {
                format!(
                    "resident expert layer {layer} precedes first sparse layer {}",
                    self.first_expert_layer
                )
            })?;
        if expert >= self.experts_per_layer {
            bail!(
                "resident expert {expert} is outside resolved count {} for layer {layer}",
                self.experts_per_layer
            );
        }
        let index = relative_layer
            .checked_mul(self.experts_per_layer)
            .and_then(|base| base.checked_add(expert))
            .context("resident expert table index overflow")?;
        self.slots
            .get(index)
            .cloned()
            .with_context(|| format!("resident expert table has no layer {layer} expert {expert}"))
    }

    fn acquire(
        &mut self,
        command: &ScheduledRoutingCommand,
    ) -> Result<ScheduledExpertSet<Arc<ScheduledExpertSlot>>> {
        let routes = self.core.scheduled_routes_from_command(command)?;
        let experts = routes
            .routes()
            .iter()
            .map(|route| self.slot(routes.layer, route.expert))
            .collect::<Result<Vec<_>>>()?;
        self.core
            .finish_routes(routes, experts, |expert| (expert.layer(), expert.expert()))
    }

    fn snapshot(&self) -> ExpertSchedulerSnapshot {
        self.core.snapshot()
    }
}

#[derive(Debug)]
pub(crate) enum ScheduledExpertAccessCoordinator {
    Streamed(ScheduledExpertReadCoordinator),
    Resident(ScheduledResidentExpertTable),
}

impl ScheduledExpertAccessCoordinator {
    pub(crate) fn normalize_routes(
        &self,
        command: &ScheduledRoutingCommand,
    ) -> Result<ScheduledExpertRoutes> {
        match self {
            Self::Streamed(coordinator) => coordinator.core.scheduled_routes_from_command(command),
            Self::Resident(table) => table.core.scheduled_routes_from_command(command),
        }
    }

    pub(crate) fn acquire_unique(
        &mut self,
        layer: usize,
        experts: &[usize],
    ) -> Result<Vec<Arc<ScheduledExpertSlot>>> {
        match self {
            Self::Streamed(coordinator) => coordinator.acquire_unique(layer, experts),
            Self::Resident(table) => experts
                .iter()
                .map(|expert| table.slot(layer, *expert))
                .collect(),
        }
    }

    pub(crate) fn issue_routing_command(
        &mut self,
        command: &ScheduledRoutingCommand,
    ) -> Result<PendingScheduledExpertAccess> {
        match self {
            Self::Streamed(coordinator) => Ok(PendingScheduledExpertAccess::Streamed(
                coordinator.issue_routing_command(command)?,
            )),
            Self::Resident(table) => Ok(PendingScheduledExpertAccess::Resident(
                table.acquire(command)?,
            )),
        }
    }

    pub(crate) fn finish_routes(
        &mut self,
        pending: PendingScheduledExpertAccess,
    ) -> Result<ScheduledExpertSet<Arc<ScheduledExpertSlot>>> {
        match (self, pending) {
            (Self::Streamed(coordinator), PendingScheduledExpertAccess::Streamed(pending)) => {
                coordinator.finish_routes(pending)
            }
            (Self::Resident(_), PendingScheduledExpertAccess::Resident(scheduled)) => Ok(scheduled),
            (Self::Streamed(_), PendingScheduledExpertAccess::Resident(_)) => {
                bail!("resident expert result reached the streamed graph implementation")
            }
            (Self::Resident(_), PendingScheduledExpertAccess::Streamed(_)) => {
                bail!("streamed expert result reached the resident graph implementation")
            }
        }
    }

    pub(crate) fn read_experts_into(
        &mut self,
        layer: usize,
        experts: &[usize],
        destination: &mut [u8],
        slot_stride: usize,
        workers: usize,
    ) -> Result<DirectExpertReadSummary> {
        match self {
            Self::Streamed(coordinator) => {
                coordinator.read_experts_into(layer, experts, destination, slot_stride, workers)
            }
            Self::Resident(_) => bail!(
                "request-scoped batch expert staging is not declared by the resident expert graph"
            ),
        }
    }

    /// # Safety
    ///
    /// `destination` must remain allocated and exclusively borrowed until
    /// the returned guard is finished or dropped.
    pub(crate) unsafe fn issue_layer_prepare_into<'a>(
        &self,
        layer: usize,
        destination: &'a mut [u8],
        workers: usize,
    ) -> Result<PendingExpertLayerPrepare<'a>> {
        match self {
            Self::Streamed(coordinator) => unsafe {
                coordinator.issue_layer_prepare_into(layer, destination, workers)
            },
            Self::Resident(_) => {
                bail!("whole-layer expert preparation is not declared by the resident expert graph")
            }
        }
    }

    pub(crate) fn finish_layer_prepare(
        &self,
        pending: PendingExpertLayerPrepare<'_>,
    ) -> Result<ExpertLayerPrepareSummary> {
        match self {
            Self::Streamed(coordinator) => coordinator.finish_layer_prepare(pending),
            Self::Resident(_) => {
                bail!("whole-layer expert preparation completion reached the resident expert graph")
            }
        }
    }

    pub(crate) fn snapshot(&self) -> ExpertSchedulerSnapshot {
        match self {
            Self::Streamed(coordinator) => coordinator.snapshot(),
            Self::Resident(table) => table.snapshot(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ScheduledExpertReadCoordinator {
    store: ExpertSlotStore,
    pool: ExpertReadWorkerPool,
    core: ActiveExpertReadScheduler,
}

impl ScheduledExpertReadCoordinator {
    #[cfg(test)]
    pub(crate) fn new(store: ExpertSlotStore) -> Self {
        Self::new_with_routed_expert_scale(store, 1.0)
    }

    #[cfg(test)]
    pub(crate) fn new_with_routed_expert_scale(
        store: ExpertSlotStore,
        routed_expert_scale: f32,
    ) -> Self {
        Self::new_with_routing_policy(
            store,
            QwenMoeRoutingWeightNormalization::RenormalizeSelected,
            routed_expert_scale,
        )
    }

    pub(crate) fn new_with_routing_policy(
        store: ExpertSlotStore,
        routing_weight_normalization: QwenMoeRoutingWeightNormalization,
        routed_expert_scale: f32,
    ) -> Self {
        Self {
            store,
            pool: ExpertReadWorkerPool::default(),
            core: ActiveExpertReadScheduler::new_with_routing_policy(
                routing_weight_normalization,
                routed_expert_scale,
            ),
        }
    }

    pub(crate) fn issue_experts(
        &mut self,
        layer: usize,
        experts: &[usize],
    ) -> Result<Vec<PendingScheduledRead<ExpertRawReadResponse>>> {
        if experts.is_empty() {
            return Ok(Vec::new());
        }
        self.pool
            .ensure_workers(experts.len().min(BATCH_EXPERT_READ_WORKERS).max(1));
        let reader = self.store.layer_reader(layer)?;
        let mut pending = Vec::with_capacity(experts.len());
        for expert in experts {
            let plan = reader.prepare_read(*expert)?;
            let issue = self.core.issue_read(layer, *expert);
            let rx = self.pool.submit_read(
                issue.id,
                issue.key.expert,
                Arc::clone(&reader),
                plan,
                issue.warm,
                issue.issued_at,
            )?;
            pending.push(PendingScheduledRead::new(issue.id, rx));
        }
        Ok(pending)
    }

    fn acquire_unique(
        &mut self,
        layer: usize,
        experts: &[usize],
    ) -> Result<Vec<Arc<ScheduledExpertSlot>>> {
        let pending = self.issue_experts(layer, experts)?;
        self.finish(pending)
    }

    pub(crate) fn issue_routing_command(
        &mut self,
        command: &ScheduledRoutingCommand,
    ) -> Result<PendingScheduledExpertSet<ExpertRawReadResponse>> {
        let issued = self.core.issue_routed_reads(command)?;
        let reads = self.submit_issued_reads(&issued)?;
        let routes = issued.into_routes();
        Ok(PendingScheduledExpertSet::new(routes, reads))
    }

    fn submit_issued_reads(
        &mut self,
        issued: &ScheduledExpertReadSet,
    ) -> Result<Vec<PendingScheduledRead<ExpertRawReadResponse>>> {
        if issued.is_empty() {
            return Ok(Vec::new());
        }
        self.pool.ensure_workers(issued.len().max(1));
        let reader = self.store.layer_reader(issued.layer())?;
        let mut pending = Vec::with_capacity(issued.len());
        // Submit positioned reads directly into reusable whole-expert slots; the OS page cache
        // remains the cache policy for this stage.
        for issue in issued.issues() {
            let plan = reader.prepare_read(issue.key.expert)?;
            let rx = self.pool.submit_read(
                issue.id,
                issue.key.expert,
                Arc::clone(&reader),
                plan,
                issue.warm,
                issue.issued_at,
            )?;
            pending.push(PendingScheduledRead::new(issue.id, rx));
        }
        Ok(pending)
    }

    pub(crate) fn finish(
        &mut self,
        pending: Vec<PendingScheduledRead<ExpertRawReadResponse>>,
    ) -> Result<Vec<Arc<ScheduledExpertSlot>>> {
        let mut out = Vec::with_capacity(pending.len());
        for pending in pending {
            let pending_id = pending.id();
            let response = pending
                .recv()
                .context("expert I/O worker dropped response channel")?;
            let slot = self.core.finish_slot_read(pending_id, response)?;
            out.push(Arc::new(slot));
        }
        Ok(out)
    }

    pub(crate) fn finish_routes(
        &mut self,
        pending: PendingScheduledExpertSet<ExpertRawReadResponse>,
    ) -> Result<ScheduledExpertSet<Arc<ScheduledExpertSlot>>> {
        let (routes, reads) = pending.into_parts();
        let experts = self.finish(reads)?;
        self.core
            .finish_routes(routes, experts, |expert| (expert.layer(), expert.expert()))
    }

    pub(crate) fn read_experts_into(
        &mut self,
        layer: usize,
        experts: &[usize],
        destination: &mut [u8],
        slot_stride: usize,
        workers: usize,
    ) -> Result<DirectExpertReadSummary> {
        let issues = experts
            .iter()
            .map(|expert| self.core.issue_read(layer, *expert))
            .collect::<Vec<_>>();
        let read_started = Instant::now();
        let summary =
            match self
                .store
                .read_unique_into(layer, experts, destination, slot_stride, workers)
            {
                Ok(summary) => summary,
                Err(error) => {
                    let latency = read_started.elapsed();
                    let message = format!("{error:#}");
                    for issue in issues {
                        let recorded = self.core.finish_read(
                            issue.id,
                            ScheduledExpertReadResponse::<()> {
                                id: issue.id,
                                queue_latency: Duration::ZERO,
                                read_path: FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
                                read_latency: latency,
                                bytes_read: 0,
                                warm: issue.warm,
                                result: Err(anyhow::anyhow!(message.clone())),
                            },
                        );
                        debug_assert!(
                            recorded.is_err(),
                            "synthetic failed expert read unexpectedly succeeded"
                        );
                    }
                    return Err(error);
                }
            };
        if summary.read_latencies.len() != issues.len() {
            bail!(
                "direct batch expert read returned {} timings for {} issued experts",
                summary.read_latencies.len(),
                issues.len()
            );
        }
        for (issue, read_latency) in issues.into_iter().zip(&summary.read_latencies) {
            self.core.finish_read(
                issue.id,
                ScheduledExpertReadResponse {
                    id: issue.id,
                    queue_latency: Duration::ZERO,
                    read_path: FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
                    read_latency: *read_latency,
                    bytes_read: u64::try_from(slot_stride)?,
                    warm: issue.warm,
                    result: Ok(()),
                },
            )?;
        }
        Ok(summary)
    }

    /// # Safety
    ///
    /// `destination` must remain allocated and exclusively borrowed until
    /// the returned guard is finished or dropped.
    pub(crate) unsafe fn issue_layer_prepare_into<'a>(
        &self,
        layer: usize,
        destination: &'a mut [u8],
        workers: usize,
    ) -> Result<PendingExpertLayerPrepare<'a>> {
        unsafe {
            self.store
                .issue_layer_prepare_into(layer, destination, workers)
        }
    }

    pub(crate) fn finish_layer_prepare(
        &self,
        pending: PendingExpertLayerPrepare<'_>,
    ) -> Result<ExpertLayerPrepareSummary> {
        pending.finish()
    }

    pub(crate) fn snapshot(&self) -> ExpertSchedulerSnapshot {
        self.core.snapshot()
    }

    #[cfg(test)]
    pub(crate) fn worker_count(&self) -> usize {
        self.pool.worker_count()
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
pub(crate) struct ExpertSchedulerSnapshot {
    pub(crate) issued_reads: u64,
    pub(crate) positioned_reads: u64,
    pub(crate) read_failures: u64,
    pub(crate) total_queue_latency: Duration,
    pub(crate) max_queue_latency: Duration,
    pub(crate) total_read_latency: Duration,
    pub(crate) max_read_latency: Duration,
    pub(crate) bytes_read: u64,
    pub(crate) warm_reads: u64,
    pub(crate) total_warm_read_latency: Duration,
    pub(crate) max_warm_read_latency: Duration,
    pub(crate) warm_bytes_read: u64,
}

impl ExpertSchedulerSnapshot {
    pub(crate) fn saturating_delta(self, before: Self) -> Self {
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
