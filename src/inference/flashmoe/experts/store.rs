use super::*;

#[derive(Debug)]
pub(crate) struct ExpertLayerReader {
    path: PathBuf,
    file: fs::File,
    metadata: ExpertLayerPackMetadata,
    slot_spec: ExpertSlotSpec,
    buffer_pool: ReusableExpertBytePool,
}

#[derive(Debug, Clone)]
pub(crate) struct ExpertSlotStore {
    root: PathBuf,
    slot_spec: ExpertSlotSpec,
    buffer_pool: ReusableExpertBytePool,
    layers: Arc<Mutex<BTreeMap<usize, Arc<ExpertLayerReader>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertStorageLayout {
    FixedQ4,
    FixedMxfp4,
    FixedBf16,
    FixedF16,
    FixedDeepSeekGguf,
}

impl ExpertStorageLayout {
    pub(crate) fn quantization(self) -> ExpertQuantization {
        match self {
            Self::FixedQ4 | Self::FixedMxfp4 | Self::FixedDeepSeekGguf => {
                ExpertQuantization::FourBitProduction
            }
            Self::FixedBf16 => ExpertQuantization::Bf16,
            Self::FixedF16 => ExpertQuantization::F16,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::FixedQ4 => "fixed-Q4",
            Self::FixedMxfp4 => "fixed-MXFP4",
            Self::FixedBf16 => "fixed-BF16",
            Self::FixedF16 => "fixed-F16",
            Self::FixedDeepSeekGguf => "fixed-DeepSeek-IQ2_XXS/Q2_K",
        }
    }
}

pub(crate) fn validate_requested_expert_storage(
    root: &Path,
    resolved_layout: ExpertStorageLayout,
    requested_quantization: ExpertQuantization,
) -> Result<()> {
    if resolved_layout.quantization() != requested_quantization {
        bail!(
            "FlashMoe unsupported expert storage policy: requested {} but cache metadata in {} resolves {} expert slots",
            requested_quantization.as_str(),
            root.display(),
            resolved_layout.as_str()
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpertStoreExecutionDescriptor {
    pub(crate) layout: ExpertStorageLayout,
    pub(crate) slot_spec: ExpertSlotSpec,
    pub(crate) layers: usize,
    pub(crate) first_expert_layer: usize,
    pub(crate) experts_per_layer: usize,
}

impl ExpertStoreExecutionDescriptor {
    pub(crate) fn total_expert_bytes(self) -> Result<usize> {
        self.layers
            .checked_sub(self.first_expert_layer)
            .context("first expert layer exceeds resolved layer count")?
            .checked_mul(self.experts_per_layer)
            .and_then(|slots| slots.checked_mul(self.slot_spec.expert_bytes()))
            .context("resolved expert corpus byte length overflow")
    }
}

#[derive(Debug)]
pub(crate) struct ResolvedExpertSlotStore {
    pub(crate) store: ExpertSlotStore,
    pub(crate) descriptor: ExpertStoreExecutionDescriptor,
    pub(crate) upgraded_pbq4_layers: usize,
}

#[derive(Default)]
pub(crate) struct ExpertReadWorkerPool {
    workers: Vec<thread::JoinHandle<()>>,
    senders: Vec<mpsc::Sender<ExpertReadJob>>,
    next_worker: usize,
}

impl std::fmt::Debug for ExpertReadWorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertReadWorkerPool")
            .field("workers", &self.workers.len())
            .field("next_worker", &self.next_worker)
            .finish()
    }
}

impl ExpertReadWorkerPool {
    pub(crate) fn submit_read(
        &mut self,
        id: u64,
        expert: usize,
        reader: Arc<ExpertLayerReader>,
        plan: ExpertReadPlan,
        warm: bool,
        issued_at: Instant,
    ) -> Result<mpsc::Receiver<ExpertRawReadResponse>> {
        if self.senders.is_empty() {
            self.ensure_workers(1);
        }
        let (tx, rx) = mpsc::channel();
        let worker = self.next_worker % self.senders.len();
        self.next_worker = self.next_worker.wrapping_add(1);
        self.senders[worker]
            .send(ExpertReadJob {
                id,
                expert,
                reader,
                plan,
                warm,
                issued_at,
                tx,
            })
            .context("failed to submit expert read to I/O worker")?;
        Ok(rx)
    }

    pub(crate) fn ensure_workers(&mut self, workers: usize) {
        while self.workers.len() < workers {
            let (tx, rx) = mpsc::channel::<ExpertReadJob>();
            let handle = thread::spawn(move || {
                let mut scratch = ReusableExpertBuffer::default();
                while let Ok(job) = rx.recv() {
                    let started_at = Instant::now();
                    let queue_latency = started_at.saturating_duration_since(job.issued_at);
                    let bytes_read = job.plan.packed_len as u64;
                    let result = job
                        .reader
                        .read_prepared_into(job.expert, job.plan, &mut scratch);
                    let (result, read_latency, read_path) = match result {
                        Ok(raw) => {
                            let read_latency = raw.read_latency;
                            let read_path = raw.read_path;
                            (Ok(raw), read_latency, read_path)
                        }
                        Err(error) => (
                            Err(error),
                            started_at.elapsed(),
                            FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
                        ),
                    };
                    let _ = job.tx.send(ExpertRawReadResponse {
                        id: job.id,
                        queue_latency,
                        read_path,
                        read_latency,
                        bytes_read,
                        warm: job.warm,
                        result,
                    });
                }
            });
            self.senders.push(tx);
            self.workers.push(handle);
        }
    }

    #[cfg(test)]
    pub(crate) fn worker_count(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for ExpertReadWorkerPool {
    fn drop(&mut self) {
        self.senders.clear();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

struct ExpertReadJob {
    id: u64,
    expert: usize,
    reader: Arc<ExpertLayerReader>,
    plan: ExpertReadPlan,
    warm: bool,
    issued_at: Instant,
    tx: mpsc::Sender<ExpertRawReadResponse>,
}

#[derive(Debug)]
pub(crate) struct ExpertRawReadResponse {
    pub(crate) id: u64,
    pub(crate) queue_latency: Duration,
    pub(crate) read_path: ExpertReadPath,
    pub(crate) read_latency: Duration,
    pub(crate) bytes_read: u64,
    pub(crate) warm: bool,
    pub(crate) result: Result<ExpertRawRead>,
}

#[derive(Debug)]
pub(crate) struct DirectExpertReadSummary {
    pub(crate) read_latencies: Vec<Duration>,
    #[cfg(test)]
    pub(crate) positioned_runs: usize,
    #[cfg(test)]
    pub(crate) bytes_read: u64,
}

#[derive(Debug)]
pub(crate) struct PendingExpertLayerPrepare {
    pub(crate) layer: usize,
    pub(crate) bytes: u64,
    pub(crate) workers: Vec<thread::JoinHandle<Result<ExpertLayerPrepareWorkerSummary>>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertLayerPrepareWorkerSummary {
    pub(crate) bytes_read: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertLayerPrepareSummary {
    pub(crate) layer: usize,
    #[cfg(test)]
    pub(crate) bytes_read: u64,
}

impl PendingExpertLayerPrepare {
    pub(crate) fn layer(&self) -> usize {
        self.layer
    }

    pub(crate) fn finish(mut self) -> Result<ExpertLayerPrepareSummary> {
        let mut bytes_read = 0u64;
        let mut first_error = None;
        for worker in self.workers.drain(..) {
            match worker.join() {
                Ok(Ok(summary)) => {
                    bytes_read = bytes_read
                        .checked_add(summary.bytes_read)
                        .context("prepared expert layer byte count overflow")?;
                }
                Ok(Err(error)) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
                Err(_) => {
                    if first_error.is_none() {
                        first_error = Some(anyhow::anyhow!(
                            "expert layer {} preparation worker panicked",
                            self.layer
                        ));
                    }
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        if bytes_read != self.bytes {
            bail!(
                "prepared expert layer {} read {bytes_read} bytes, expected {}",
                self.layer,
                self.bytes
            );
        }
        Ok(ExpertLayerPrepareSummary {
            layer: self.layer,
            #[cfg(test)]
            bytes_read,
        })
    }
}

impl Drop for PendingExpertLayerPrepare {
    fn drop(&mut self) {
        // Direct request staging is caller-owned. A primary execution error
        // may unwind before the explicit finish point, so never let a worker
        // outlive the Metal allocation it is filling.
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[derive(Clone, Copy)]
struct DirectExpertWritePtr(*mut u8);

// SAFETY: this pointer is only created from a caller-owned destination that is
// kept alive and inaccessible until every worker joins. Each sent copy is used
// for a validated, disjoint byte range; `slice_at` remains unsafe so callers
// must uphold those range and lifetime invariants.
unsafe impl Send for DirectExpertWritePtr {}

impl DirectExpertWritePtr {
    unsafe fn slice_at<'a>(self, offset: usize, len: usize) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(offset), len) }
    }
}

#[derive(Debug, Clone, Copy)]
struct DirectExpertReadRun {
    first_index: usize,
    end_index: usize,
    file_offset: u64,
    destination_offset: usize,
    len: usize,
}

fn resolve_fixed_dense_metadata_dtype(
    metadata: &ExpertLayerPackMetadata,
) -> Result<DenseExpertDtype> {
    let mut resolved: Option<DenseExpertDtype> = None;
    for record in metadata.packs.iter().flat_map(|pack| &pack.records) {
        let dtype = DenseExpertDtype::from_metadata_dtype(&record.dtype).with_context(|| {
            format!(
                "FlashMoe unsupported fixed dense expert dtype {} in layer {} tensor {}",
                record.dtype, metadata.layer, record.tensor
            )
        })?;
        if let Some(existing) = resolved
            && existing != dtype
        {
            bail!(
                "FlashMoe unsupported fixed dense expert storage at layer {}: metadata mixes {} and {} tensors",
                metadata.layer,
                existing.as_str(),
                dtype.as_str()
            );
        }
        resolved = Some(dtype);
    }
    resolved.with_context(|| {
        format!(
            "FlashMoe unsupported fixed dense expert storage at layer {}: metadata declares no BF16/F16 tensor records",
            metadata.layer
        )
    })
}

impl ExpertSlotStore {
    #[cfg(test)]
    pub(crate) fn open(root: PathBuf) -> Result<Self> {
        Self::open_with_fixed_q4(root, FixedQ4ExpertSlotSpec::qwen35_a17b()?)
    }

    pub(crate) fn resolve_from_metadata(
        root: PathBuf,
        layout: &QwenMoeModelLayout,
        requested_quantization: ExpertQuantization,
    ) -> Result<ResolvedExpertSlotStore> {
        if !root.is_dir() {
            bail!("expert store {} does not exist", root.display());
        }
        let first_expert_layer = layout.first_sparse_layer;
        let metadata = read_expert_layer_pack_metadata(&root, first_expert_layer)?.with_context(|| {
            format!(
                "FlashMoe unsupported expert storage: first sparse layer {first_expert_layer} metadata is missing from {}",
                root.display()
            )
        })?;
        let resolved_layout = match metadata.format.as_str() {
            FIXED_Q4_EXPERT_LAYER_FORMAT_V1
            | PBQ4_EXPERT_LAYER_FORMAT_V1
            | PBQ4_EXPERT_LAYER_FORMAT_V2 => ExpertStorageLayout::FixedQ4,
            FIXED_MXFP4_EXPERT_LAYER_FORMAT_V1 => ExpertStorageLayout::FixedMxfp4,
            FIXED_DENSE_EXPERT_LAYER_FORMAT_V1 => {
                match resolve_fixed_dense_metadata_dtype(&metadata)? {
                    DenseExpertDtype::Bf16 => ExpertStorageLayout::FixedBf16,
                    DenseExpertDtype::F16 => ExpertStorageLayout::FixedF16,
                }
            }
            FIXED_DEEPSEEK_GGUF_EXPERT_LAYER_FORMAT_V1 => ExpertStorageLayout::FixedDeepSeekGguf,
            format => {
                bail!("FlashMoe unsupported expert storage format {format} in layer 0 metadata")
            }
        };
        validate_requested_expert_storage(&root, resolved_layout, requested_quantization)?;
        let (slot_spec, upgraded_pbq4_layers) = match metadata.format.as_str() {
            FIXED_Q4_EXPERT_LAYER_FORMAT_V1 | FIXED_MXFP4_EXPERT_LAYER_FORMAT_V1 => (
                ExpertSlotSpec::from_model_layout(layout, resolved_layout)?,
                0,
            ),
            FIXED_DENSE_EXPERT_LAYER_FORMAT_V1 => (
                ExpertSlotSpec::from_model_layout(layout, resolved_layout)?,
                0,
            ),
            FIXED_DEEPSEEK_GGUF_EXPERT_LAYER_FORMAT_V1 => {
                let spec = DeepSeekGgufExpertSlotSpec::from_model_layout(layout)?;
                spec.validate_metadata(&metadata)?;
                (ExpertSlotSpec::FixedDeepSeekGguf(spec), 0)
            }
            PBQ4_EXPERT_LAYER_FORMAT_V1 | PBQ4_EXPERT_LAYER_FORMAT_V2 => {
                let slot_spec =
                    ExpertSlotSpec::from_model_layout(layout, ExpertStorageLayout::FixedQ4)?;
                let spec = slot_spec
                    .fixed_q4()
                    .expect("fixed-Q4 storage resolves Q4 spec");
                let mut upgraded = 0usize;
                for layer in first_expert_layer..layout.layers {
                    if rewrite_pbq4_layer_to_fixed_q4(&root, layer, layout.experts_per_layer, spec)
                        .with_context(|| {
                            format!("failed to upgrade layer {layer} expert cache to fixed Q4")
                        })?
                    {
                        upgraded += 1;
                    }
                }
                (slot_spec, upgraded)
            }
            _ => unreachable!("expert storage format was validated before slot resolution"),
        };
        let store = Self::open_with_slot_spec(root, slot_spec)?;
        let descriptor = store.resolve_execution_descriptor_from(
            layout.layers,
            layout.experts_per_layer,
            first_expert_layer,
        )?;
        Ok(ResolvedExpertSlotStore {
            store,
            descriptor,
            upgraded_pbq4_layers,
        })
    }

    #[cfg(test)]
    pub(crate) fn open_with_fixed_q4(
        root: PathBuf,
        fixed_q4: FixedQ4ExpertSlotSpec,
    ) -> Result<Self> {
        Self::open_with_slot_spec(root, ExpertSlotSpec::FixedQ4(fixed_q4))
    }

    #[cfg(test)]
    pub(crate) fn open_with_fixed_dense(
        root: PathBuf,
        fixed_dense: FixedDenseExpertSlotSpec,
    ) -> Result<Self> {
        Self::open_with_slot_spec(root, ExpertSlotSpec::FixedDense(fixed_dense))
    }

    pub(crate) fn open_with_slot_spec(root: PathBuf, slot_spec: ExpertSlotSpec) -> Result<Self> {
        if !root.is_dir() {
            bail!("expert store {} does not exist", root.display());
        }
        Ok(Self {
            root,
            slot_spec,
            buffer_pool: Arc::new(Mutex::new(Vec::new())),
            layers: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    #[cfg(test)]
    pub(crate) fn resolve_execution_descriptor(
        &self,
        layers: usize,
        experts_per_layer: usize,
    ) -> Result<ExpertStoreExecutionDescriptor> {
        self.resolve_execution_descriptor_from(layers, experts_per_layer, 0)
    }

    pub(crate) fn resolve_execution_descriptor_from(
        &self,
        layers: usize,
        experts_per_layer: usize,
        first_expert_layer: usize,
    ) -> Result<ExpertStoreExecutionDescriptor> {
        if layers == 0 || experts_per_layer == 0 {
            bail!(
                "FlashMoe expert storage resolution requires non-zero layers and experts, layers={layers}, experts_per_layer={experts_per_layer}"
            );
        }
        if first_expert_layer >= layers {
            bail!(
                "FlashMoe first expert layer {first_expert_layer} must be within {layers} model layers"
            );
        }
        let slot_bytes = self.slot_spec.expert_bytes();
        let storage_layout = self.slot_spec.storage_layout();
        let expected_format = self.slot_spec.metadata_format();
        let expected_layer_bytes = (slot_bytes as u64)
            .checked_mul(experts_per_layer as u64)
            .context("fixed expert layer byte length overflow")?;
        for layer in first_expert_layer..layers {
            let metadata = read_expert_layer_pack_metadata(&self.root, layer)?.with_context(|| {
                format!(
                    "FlashMoe unsupported {storage_layout:?} expert storage: layer {layer} metadata is missing"
                )
            })?;
            if metadata.format != expected_format {
                bail!(
                    "FlashMoe unsupported expert storage at layer {layer}: format {} is import compatibility only; expected {}",
                    metadata.format,
                    expected_format
                );
            }
            if let Some(expected) = self.slot_spec.fixed_dense() {
                let actual = resolve_fixed_dense_metadata_dtype(&metadata)?;
                if actual != expected.dtype {
                    bail!(
                        "FlashMoe unsupported fixed dense expert storage at layer {layer}: metadata declares {}, resolved graph requires {}",
                        actual.as_str(),
                        expected.dtype.as_str()
                    );
                }
            }
            if let Some(expected) = self.slot_spec.fixed_deepseek_gguf() {
                expected.validate_metadata(&metadata)?;
            }
            if metadata.expert_size != slot_bytes as u64 {
                bail!(
                    "FlashMoe {storage_layout:?} expert storage layer {layer} has slot size {}, expected {}",
                    metadata.expert_size,
                    slot_bytes
                );
            }
            if metadata.experts != experts_per_layer || metadata.packs.len() != experts_per_layer {
                bail!(
                    "FlashMoe {storage_layout:?} expert storage layer {layer} declares {} experts and {} records, expected {experts_per_layer}",
                    metadata.experts,
                    metadata.packs.len()
                );
            }
            for expert in 0..experts_per_layer {
                let pack = metadata.pack_for(expert).with_context(|| {
                    format!(
                        "FlashMoe {storage_layout:?} expert storage layer {layer} is missing expert {expert}"
                    )
                })?;
                if pack.packed_bytes != metadata.expert_size {
                    bail!(
                        "FlashMoe {storage_layout:?} expert storage layer {layer} expert {expert} has {} bytes, expected whole-slot payload {}",
                        pack.packed_bytes,
                        metadata.expert_size
                    );
                }
            }
            let path = expert_layer_path(&self.root, layer);
            let actual_layer_bytes = fs::metadata(&path)
                .with_context(|| format!("failed to stat expert layer {}", path.display()))?
                .len();
            if actual_layer_bytes != expected_layer_bytes {
                bail!(
                    "FlashMoe {storage_layout:?} expert storage layer {layer} has file size {actual_layer_bytes}, expected {expected_layer_bytes}"
                );
            }
        }
        Ok(ExpertStoreExecutionDescriptor {
            layout: storage_layout,
            slot_spec: self.slot_spec,
            layers,
            first_expert_layer,
            experts_per_layer,
        })
    }

    #[cfg(test)]
    pub(crate) fn read_many_raw(
        &self,
        layer: usize,
        experts: &[usize],
    ) -> Result<Vec<ExpertRawRead>> {
        let reader = self.layer_reader(layer)?;
        let mut scratch = ReusableExpertBuffer::default();
        let mut out = Vec::with_capacity(experts.len());
        for &expert in experts {
            let plan = reader.prepare_read(expert)?;
            out.push(reader.read_prepared_into(expert, plan, &mut scratch)?);
        }
        Ok(out)
    }

    pub(crate) fn layer_reader(&self, layer: usize) -> Result<Arc<ExpertLayerReader>> {
        if let Some(reader) = self
            .layers
            .lock()
            .expect("expert layer cache poisoned")
            .get(&layer)
            .cloned()
        {
            return Ok(reader);
        }

        let reader = Arc::new(ExpertLayerReader::open(
            &self.root,
            layer,
            self.slot_spec,
            Arc::clone(&self.buffer_pool),
        )?);
        let mut layers = self.layers.lock().expect("expert layer cache poisoned");
        Ok(layers.entry(layer).or_insert_with(|| reader).clone())
    }

    pub(crate) fn read_unique_into(
        &self,
        layer: usize,
        experts: &[usize],
        destination: &mut [u8],
        slot_stride: usize,
        workers: usize,
    ) -> Result<DirectExpertReadSummary> {
        self.layer_reader(layer)?
            .read_unique_into(experts, destination, slot_stride, workers)
    }

    pub(crate) fn map_resident_slots(
        &self,
        layers: usize,
        first_expert_layer: usize,
        experts_per_layer: usize,
    ) -> Result<Vec<ExpertRawRead>> {
        let slot_count = layers
            .checked_sub(first_expert_layer)
            .context("resident first expert layer exceeds layer count")?
            .checked_mul(experts_per_layer)
            .context("resident expert slot count overflow")?;
        let mut slots = Vec::with_capacity(slot_count);
        for layer in first_expert_layer..layers {
            let reader = self.layer_reader(layer)?;
            for expert in 0..experts_per_layer {
                let plan = reader.prepare_read(expert)?;
                slots.push(reader.map_prepared_resident(expert, plan)?);
            }
        }
        Ok(slots)
    }

    pub(crate) unsafe fn issue_layer_prepare_into(
        &self,
        layer: usize,
        destination: &mut [u8],
        workers: usize,
    ) -> Result<PendingExpertLayerPrepare> {
        unsafe {
            ExpertLayerReader::issue_layer_prepare_into(
                self.layer_reader(layer)?,
                destination,
                workers,
            )
        }
    }
}

impl ExpertLayerReader {
    /// Start a fixed whole-layer positioned read directly into caller-owned
    /// request staging. The caller must keep the destination alive and avoid
    /// all access until the returned handle is finished.
    unsafe fn issue_layer_prepare_into(
        reader: Arc<Self>,
        destination: &mut [u8],
        workers: usize,
    ) -> Result<PendingExpertLayerPrepare> {
        let (slot_bytes, layer_bytes) = reader.validate_layer_prepare()?;
        if destination.len() != layer_bytes {
            bail!(
                "expert layer {} preparation destination has {} bytes, expected complete fixed layer {layer_bytes}",
                reader.metadata.layer,
                destination.len()
            );
        }

        let worker_count = workers.max(1).min(reader.metadata.experts);
        let experts_per_worker = reader.metadata.experts.div_ceil(worker_count);
        let target = DirectExpertWritePtr(destination.as_mut_ptr());
        let ranges = (0..worker_count)
            .map(|worker| {
                let first_expert = worker
                    .checked_mul(experts_per_worker)
                    .context("expert layer staging worker range overflow")?;
                if first_expert >= reader.metadata.experts {
                    return Ok(None);
                }
                let end_expert = first_expert
                    .saturating_add(experts_per_worker)
                    .min(reader.metadata.experts);
                let offset = first_expert
                    .checked_mul(slot_bytes)
                    .context("expert layer staging offset overflow")?;
                let len = (end_expert - first_expert)
                    .checked_mul(slot_bytes)
                    .context("expert layer staging range overflow")?;
                Ok(Some((offset, len)))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let mut handles = Vec::with_capacity(ranges.len());
        for (offset, len) in ranges {
            let worker_reader = Arc::clone(&reader);
            handles.push(thread::spawn(move || {
                let destination = unsafe { target.slice_at(offset, len) };
                read_exact_at_positioned(&worker_reader.file, destination, u64::try_from(offset)?)
                    .with_context(|| {
                        format!(
                            "failed to prepare expert layer {} directly into request staging from {}",
                            worker_reader.metadata.layer,
                            worker_reader.path.display()
                        )
                    })?;
                Ok(ExpertLayerPrepareWorkerSummary {
                    bytes_read: u64::try_from(len)?,
                })
            }));
        }
        Ok(PendingExpertLayerPrepare {
            layer: reader.metadata.layer,
            bytes: u64::try_from(layer_bytes)?,
            workers: handles,
        })
    }

    fn validate_layer_prepare(&self) -> Result<(usize, usize)> {
        let slot_bytes = usize::try_from(self.metadata.expert_size)
            .context("expert layer preparation slot size does not fit usize")?;
        if slot_bytes == 0 || self.metadata.experts == 0 {
            bail!(
                "expert layer {} preparation requires non-empty fixed whole slots",
                self.metadata.layer
            );
        }
        for expert in 0..self.metadata.experts {
            let pack = self.metadata.pack_for(expert).with_context(|| {
                format!(
                    "expert layer {} preparation is missing expert {expert}",
                    self.metadata.layer
                )
            })?;
            if usize::try_from(pack.packed_bytes)? != slot_bytes {
                bail!(
                    "expert layer {} preparation requires whole slots, but expert {expert} has {} of {slot_bytes} bytes",
                    self.metadata.layer,
                    pack.packed_bytes
                );
            }
        }
        let layer_bytes = slot_bytes
            .checked_mul(self.metadata.experts)
            .context("expert layer preparation byte count overflow")?;
        let actual_bytes = usize::try_from(
            self.file
                .metadata()
                .with_context(|| {
                    format!(
                        "failed to stat expert layer {} before preparation",
                        self.path.display()
                    )
                })?
                .len(),
        )
        .context("expert layer preparation file length does not fit usize")?;
        if actual_bytes != layer_bytes {
            bail!(
                "expert layer {} preparation file has {actual_bytes} bytes, expected {layer_bytes}",
                self.metadata.layer
            );
        }
        Ok((slot_bytes, layer_bytes))
    }

    pub(crate) fn open(
        root: &Path,
        layer: usize,
        slot_spec: ExpertSlotSpec,
        buffer_pool: ReusableExpertBytePool,
    ) -> Result<Self> {
        let path = expert_layer_path(root, layer);
        let metadata = read_expert_layer_pack_metadata(root, layer)?.with_context(|| {
            format!(
                "failed to read expert layer metadata {}",
                expert_layer_metadata_path(root, layer).display()
            )
        })?;
        let file = fs::File::open(&path)
            .with_context(|| format!("failed to open expert layer {}", path.display()))?;
        Ok(Self {
            path,
            file,
            metadata,
            slot_spec,
            buffer_pool,
        })
    }

    pub(crate) fn prepare_read(&self, expert: usize) -> Result<ExpertReadPlan> {
        let metadata = self.metadata.pack_for(expert).cloned().with_context(|| {
            format!(
                "expert layer {} has no metadata for expert {expert}",
                self.metadata.layer
            )
        })?;
        if metadata.packed_bytes > self.metadata.expert_size {
            bail!(
                "expert layer {} expert {expert} metadata length {} exceeds slot size {}",
                self.metadata.layer,
                metadata.packed_bytes,
                self.metadata.expert_size
            );
        }
        let offset = expert_slot_offset(expert, self.metadata.expert_size)?;
        let packed_len = usize::try_from(metadata.packed_bytes)
            .context("expert pack length does not fit usize")?;
        let slot_capacity = usize::try_from(self.metadata.expert_size)
            .context("expert layer slot size does not fit usize")?;
        Ok(ExpertReadPlan {
            #[cfg(test)]
            metadata,
            offset,
            packed_len,
            slot_capacity,
        })
    }

    pub(crate) fn read_prepared_into(
        &self,
        expert: usize,
        plan: ExpertReadPlan,
        scratch: &mut ReusableExpertBuffer,
    ) -> Result<ExpertRawRead> {
        if scratch.capacity() < plan.slot_capacity
            && let Some(bytes) = take_reusable_expert_bytes(&self.buffer_pool, plan.slot_capacity)
        {
            let previous = scratch.adopt_buffer(bytes);
            recycle_reusable_expert_bytes(
                &self.buffer_pool,
                previous,
                self.slot_spec.expert_bytes(),
            );
        }
        let payload = scratch.prepare_payload(plan.slot_capacity, plan.packed_len)?;
        let read_started = Instant::now();
        read_exact_at_positioned(&self.file, payload, plan.offset).with_context(|| {
            format!(
                "failed to read expert {expert} from {}",
                self.path.display()
            )
        })?;
        let read_latency = read_started.elapsed();
        self.finish_whole_slot(
            expert,
            &plan,
            scratch.take_payload(),
            Some(Arc::clone(&self.buffer_pool)),
            read_latency,
            FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
        )
    }

    fn map_prepared_resident(&self, expert: usize, plan: ExpertReadPlan) -> Result<ExpertRawRead> {
        if plan.packed_len != plan.slot_capacity {
            bail!(
                "resident expert {}/{} requires one complete fixed slot, packed={} capacity={}",
                self.metadata.layer,
                expert,
                plan.packed_len,
                plan.slot_capacity
            );
        }
        let started = Instant::now();
        let bytes =
            ReusableExpertBytes::resident_file_slot(&self.file, plan.offset, plan.slot_capacity)
                .with_context(|| {
                    format!(
                        "failed to prepare resident expert {}/{} from {}",
                        self.metadata.layer,
                        expert,
                        self.path.display()
                    )
                })?;
        self.finish_whole_slot(
            expert,
            &plan,
            bytes,
            None,
            started.elapsed(),
            ExpertReadPath::ResidentMapped,
        )
    }

    fn finish_whole_slot(
        &self,
        expert: usize,
        plan: &ExpertReadPlan,
        bytes: ReusableExpertBytes,
        recycle_pool: Option<ReusableExpertBytePool>,
        read_latency: Duration,
        read_path: ExpertReadPath,
    ) -> Result<ExpertRawRead> {
        let slot = ExpertSlotView::new(
            self.metadata.layer,
            expert,
            plan.offset,
            plan.slot_capacity,
            bytes.as_slice(),
        )?;
        let descriptor = slot.descriptor();
        let payload = if slot.payload().starts_with(PBQ4_EXPERT_MAGIC) {
            if read_path == ExpertReadPath::ResidentMapped {
                bail!(
                    "resident expert {}/{} contains PBQ4 compatibility data instead of a resolved fixed slot",
                    self.metadata.layer,
                    expert
                );
            }
            ExpertRawPayload::Pbq4(bytes.into_vec())
        } else {
            match self.slot_spec {
                ExpertSlotSpec::FixedQ4(spec) => {
                    FixedQ4ExpertSlotView::new(slot, spec.layout).with_context(|| {
                        format!(
                            "expert {} is neither a PBQ4 pack nor a fixed Q4 slot matching the model layout",
                            self.path.display()
                        )
                    })?;
                    ExpertRawPayload::FixedQ4(FixedQ4ExpertPayload::from_reusable_whole_slot(
                        spec,
                        bytes,
                        recycle_pool,
                    )?)
                }
                ExpertSlotSpec::FixedDense(spec) => ExpertRawPayload::FixedDense(
                    FixedDenseExpertPayload::from_reusable_whole_slot(spec, bytes, recycle_pool)?,
                ),
                ExpertSlotSpec::FixedDeepSeekGguf(spec) => ExpertRawPayload::FixedDeepSeekGguf(
                    DeepSeekGgufExpertPayload::from_reusable_whole_slot(spec, bytes, recycle_pool)?,
                ),
            }
        };
        Ok(ExpertRawRead {
            layer: self.metadata.layer,
            expert,
            slot: descriptor,
            #[cfg(test)]
            metadata: plan.metadata.clone(),
            payload,
            read_latency,
            read_path,
        })
    }

    fn read_unique_into(
        &self,
        experts: &[usize],
        destination: &mut [u8],
        slot_stride: usize,
        workers: usize,
    ) -> Result<DirectExpertReadSummary> {
        if experts.is_empty() {
            bail!("direct batch expert read requires at least one expert");
        }
        let expected_stride = usize::try_from(self.metadata.expert_size)
            .context("expert slot size does not fit usize")?;
        if slot_stride != expected_stride {
            bail!(
                "direct batch expert destination stride {slot_stride} does not match resolved whole-slot size {expected_stride}"
            );
        }
        let expected_len = self
            .metadata
            .experts
            .checked_mul(slot_stride)
            .context("direct batch expert destination size overflow")?;
        if destination.len() != expected_len {
            bail!(
                "direct batch expert destination has {} bytes, expected graph-declared whole-layer staging size {expected_len}",
                destination.len()
            );
        }
        if experts.windows(2).any(|pair| pair[0] >= pair[1]) {
            bail!("direct batch expert ids must be sorted and unique");
        }

        let mut plans = Vec::with_capacity(experts.len());
        for &expert in experts {
            let plan = self.prepare_read(expert)?;
            if plan.packed_len != slot_stride || plan.slot_capacity != slot_stride {
                bail!(
                    "direct batch expert {}/{} is not one complete resolved whole slot: packed={} capacity={} stride={slot_stride}",
                    self.metadata.layer,
                    expert,
                    plan.packed_len,
                    plan.slot_capacity
                );
            }
            plans.push(plan);
        }

        let mut runs = Vec::new();
        let mut first_index = 0usize;
        for index in 1..=experts.len() {
            let continues =
                index < experts.len() && experts[index] == experts[index - 1].saturating_add(1);
            if continues {
                continue;
            }
            let first_expert = experts[first_index];
            let expert_count = index - first_index;
            runs.push(DirectExpertReadRun {
                first_index,
                end_index: index,
                file_offset: plans[first_index].offset,
                destination_offset: first_expert
                    .checked_mul(slot_stride)
                    .context("direct batch expert destination offset overflow")?,
                len: expert_count
                    .checked_mul(slot_stride)
                    .context("direct batch expert run size overflow")?,
            });
            first_index = index;
        }

        let worker_count = workers.max(1).min(runs.len());
        let target = DirectExpertWritePtr(destination.as_mut_ptr());
        let latencies = Mutex::new(vec![None; experts.len()]);
        let error = Mutex::new(None);
        thread::scope(|scope| {
            for worker in 0..worker_count {
                let runs = &runs;
                let latencies = &latencies;
                let error = &error;
                scope.spawn(move || {
                    for run_index in (worker..runs.len()).step_by(worker_count) {
                        if error.lock().expect("direct expert read error lock poisoned").is_some() {
                            return;
                        }
                        let run = runs[run_index];
                        let started = Instant::now();
                        // The scheduler validates a strictly increasing expert set, and each run
                        // therefore owns a disjoint whole-slot destination range for the lifetime
                        // of this scoped worker.
                        let result = read_exact_at_positioned(
                            &self.file,
                            unsafe { target.slice_at(run.destination_offset, run.len) },
                            run.file_offset,
                        )
                        .with_context(|| {
                            format!(
                                "failed direct batch expert run for layer {} experts {}..{} from {}",
                                self.metadata.layer,
                                experts[run.first_index],
                                experts[run.end_index - 1],
                                self.path.display()
                            )
                        });
                        let elapsed = started.elapsed();
                        if let Err(read_error) = result {
                            let mut slot = error
                                .lock()
                                .expect("direct expert read error lock poisoned");
                            if slot.is_none() {
                                *slot = Some(read_error);
                            }
                            return;
                        }
                        let mut values = latencies
                            .lock()
                            .expect("direct expert read latency lock poisoned");
                        for value in &mut values[run.first_index..run.end_index] {
                            *value = Some(elapsed);
                        }
                    }
                });
            }
        });
        if let Some(error) = error
            .into_inner()
            .expect("direct expert read error lock poisoned")
        {
            return Err(error);
        }
        let read_latencies = latencies
            .into_inner()
            .expect("direct expert read latency lock poisoned")
            .into_iter()
            .enumerate()
            .map(|(index, latency)| {
                latency.with_context(|| {
                    format!(
                        "direct batch expert {}/{} did not produce read timing",
                        self.metadata.layer, experts[index]
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DirectExpertReadSummary {
            read_latencies,
            #[cfg(test)]
            positioned_runs: runs.len(),
            #[cfg(test)]
            bytes_read: u64::try_from(experts.len().saturating_mul(slot_stride))?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ExpertReadPlan {
    #[cfg(test)]
    pub(crate) metadata: ExpertPackMetadata,
    pub(crate) offset: u64,
    pub(crate) packed_len: usize,
    pub(crate) slot_capacity: usize,
}

#[derive(Debug)]
pub(crate) struct ExpertRawRead {
    pub(crate) layer: usize,
    pub(crate) expert: usize,
    pub(crate) slot: ExpertSlotDescriptor,
    #[cfg(test)]
    pub(crate) metadata: ExpertPackMetadata,
    pub(crate) payload: ExpertRawPayload,
    pub(crate) read_latency: Duration,
    pub(crate) read_path: ExpertReadPath,
}

#[derive(Debug)]
pub(crate) enum ExpertRawPayload {
    Pbq4(Vec<u8>),
    FixedQ4(FixedQ4ExpertPayload),
    FixedDense(FixedDenseExpertPayload),
    FixedDeepSeekGguf(DeepSeekGgufExpertPayload),
}

pub(crate) fn expert_slot_offset(expert: usize, expert_size: u64) -> Result<u64> {
    (expert as u64)
        .checked_mul(expert_size)
        .context("expert slot offset overflow")
}

pub(crate) fn expert_slot_end(expert: usize, expert_size: u64, packed_bytes: u64) -> Result<u64> {
    expert_slot_offset(expert, expert_size)?
        .checked_add(packed_bytes)
        .context("expert slot end overflow")
}

pub(crate) fn read_exact_at_positioned(file: &fs::File, buf: &mut [u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        let mut read_total = 0usize;
        while read_total < buf.len() {
            let n = file.read_at(
                &mut buf[read_total..],
                offset
                    .checked_add(read_total as u64)
                    .context("positioned read offset overflow")?,
            )?;
            if n == 0 {
                bail!("short positioned read");
            }
            read_total += n;
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let mut file = file.try_clone().context("failed to clone file for read")?;
        file.seek(std::io::SeekFrom::Start(offset))?;
        file.read_exact(buf)?;
        Ok(())
    }
}
