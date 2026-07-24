use super::*;

fn validate_lm_head_matvec_shape(
    entry: &RuntimeTensorEntry,
    canonical_name: &str,
    vocab_size: usize,
    input_len: usize,
) -> Result<(usize, usize)> {
    let expected_shape = [vocab_size, input_len];
    match entry.shape.as_slice() {
        [rows, cols] if *rows >= vocab_size && *cols == input_len => Ok((*rows, *cols)),
        _ => bail!(
            "Flash-MoE dense tensor {canonical_name} shape mismatch: expected at least {:?}, actual shape {:?}, input length {input_len}",
            expected_shape,
            entry.shape
        ),
    }
}

#[derive(Debug, Clone)]
pub struct DenseStore {
    #[cfg(test)]
    manifest_path: PathBuf,
    pub(in crate::inference::flashmoe) len: u64,
    pub(in crate::inference::flashmoe) mmap: Arc<memmap2::Mmap>,
    registry: TensorRegistry,
    pub(in crate::inference::flashmoe) resident: Arc<std::sync::Mutex<DenseTensorCache>>,
    pub(in crate::inference::flashmoe) norm_weights:
        Arc<std::sync::Mutex<BTreeMap<DenseNormWeightKey, Arc<Vec<f32>>>>>,
    q4_mmap_projections:
        Arc<std::sync::Mutex<BTreeMap<DenseQ4ProjectionKey, Arc<DenseQ4MmapMatvecProjection>>>>,
    pub(in crate::inference::flashmoe) decoded_tiles: Arc<std::sync::Mutex<DenseTensorTileCache>>,
    #[cfg(test)]
    #[allow(dead_code)]
    raw_tiles: Arc<std::sync::Mutex<DenseRawTensorTileCache>>,
    #[cfg(test)]
    decoded_full_tensors: Arc<std::sync::atomic::AtomicUsize>,
    #[cfg(test)]
    pub(in crate::inference::flashmoe) decoded_tensor_tiles: Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Debug, Default)]
pub(in crate::inference::flashmoe) struct DenseTensorCache {
    tensors: BTreeMap<String, Arc<Vec<f32>>>,
    pub(in crate::inference::flashmoe) bytes: usize,
    max_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DenseTensorTileKey {
    name: String,
    start_row: usize,
    row_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::inference::flashmoe) struct DenseNormWeightKey {
    pub(in crate::inference::flashmoe) name: String,
    pub(in crate::inference::flashmoe) width: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(in crate::inference::flashmoe) struct DenseTileReadTiming {
    pub(in crate::inference::flashmoe) total: Duration,
    pub(in crate::inference::flashmoe) read_range: Duration,
    pub(in crate::inference::flashmoe) decode: Duration,
    pub(in crate::inference::flashmoe) cache_insert: Duration,
    pub(in crate::inference::flashmoe) cache_evict: Duration,
    pub(in crate::inference::flashmoe) cache_hits: u64,
    pub(in crate::inference::flashmoe) cache_misses: u64,
    pub(in crate::inference::flashmoe) cache_inserts: u64,
    pub(in crate::inference::flashmoe) cache_evictions: u64,
    pub(in crate::inference::flashmoe) bytes_read: u64,
    pub(in crate::inference::flashmoe) decoded_bytes: u64,
}

impl DenseTileReadTiming {
    fn add(&mut self, other: Self) {
        self.total += other.total;
        self.read_range += other.read_range;
        self.decode += other.decode;
        self.cache_insert += other.cache_insert;
        self.cache_evict += other.cache_evict;
        self.cache_hits = self.cache_hits.saturating_add(other.cache_hits);
        self.cache_misses = self.cache_misses.saturating_add(other.cache_misses);
        self.cache_inserts = self.cache_inserts.saturating_add(other.cache_inserts);
        self.cache_evictions = self.cache_evictions.saturating_add(other.cache_evictions);
        self.bytes_read = self.bytes_read.saturating_add(other.bytes_read);
        self.decoded_bytes = self.decoded_bytes.saturating_add(other.decoded_bytes);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct DenseTensorTileCacheInsertStats {
    inserts: u64,
    evictions: u64,
    insert_time: Duration,
    evict_time: Duration,
}

#[derive(Debug, Default)]
pub(in crate::inference::flashmoe) struct DenseTensorTileCache {
    tiles: BTreeMap<DenseTensorTileKey, Arc<Vec<f32>>>,
    pub(in crate::inference::flashmoe) bytes: usize,
    max_bytes: usize,
}

#[derive(Debug, Default)]
#[cfg(test)]
#[allow(dead_code)]
struct DenseRawTensorTileCache {
    tiles: BTreeMap<DenseTensorTileKey, Arc<Vec<u8>>>,
    bytes: usize,
    max_bytes: usize,
}

impl DenseTensorTileCache {
    fn with_budget(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            ..Self::default()
        }
    }

    fn get(&self, key: &DenseTensorTileKey) -> Option<Arc<Vec<f32>>> {
        self.tiles.get(key).cloned()
    }

    fn insert(
        &mut self,
        key: DenseTensorTileKey,
        tile: Arc<Vec<f32>>,
    ) -> DenseTensorTileCacheInsertStats {
        let bytes = tile.len() * std::mem::size_of::<f32>();
        if bytes == 0 || bytes > self.max_bytes {
            return DenseTensorTileCacheInsertStats::default();
        }
        if let Some(previous) = self.tiles.remove(&key) {
            self.bytes = self
                .bytes
                .saturating_sub(previous.len() * std::mem::size_of::<f32>());
        }
        let evict_started = Instant::now();
        let mut evictions = 0u64;
        while self.bytes.saturating_add(bytes) > self.max_bytes && !self.tiles.is_empty() {
            let Some(victim) = self.tiles.keys().next().cloned() else {
                break;
            };
            if let Some(previous) = self.tiles.remove(&victim) {
                self.bytes = self
                    .bytes
                    .saturating_sub(previous.len() * std::mem::size_of::<f32>());
                evictions = evictions.saturating_add(1);
            }
        }
        let evict_time = if evictions > 0 {
            evict_started.elapsed()
        } else {
            Duration::ZERO
        };

        let insert_started = Instant::now();
        self.tiles.insert(key, tile);
        self.bytes = self.bytes.saturating_add(bytes);
        DenseTensorTileCacheInsertStats {
            inserts: 1,
            evictions,
            insert_time: insert_started.elapsed(),
            evict_time,
        }
    }
}

#[cfg(test)]
#[allow(dead_code)]
impl DenseRawTensorTileCache {
    fn with_budget(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            ..Self::default()
        }
    }

    fn get(&self, key: &DenseTensorTileKey) -> Option<Arc<Vec<u8>>> {
        self.tiles.get(key).cloned()
    }

    fn insert(
        &mut self,
        key: DenseTensorTileKey,
        tile: Arc<Vec<u8>>,
    ) -> DenseTensorTileCacheInsertStats {
        let bytes = tile.len();
        if bytes == 0 || bytes > self.max_bytes {
            return DenseTensorTileCacheInsertStats::default();
        }
        if let Some(previous) = self.tiles.remove(&key) {
            self.bytes = self.bytes.saturating_sub(previous.len());
        }
        let evict_started = Instant::now();
        let mut evictions = 0u64;
        while self.bytes.saturating_add(bytes) > self.max_bytes && !self.tiles.is_empty() {
            let Some(victim) = self.tiles.keys().next().cloned() else {
                break;
            };
            if let Some(previous) = self.tiles.remove(&victim) {
                self.bytes = self.bytes.saturating_sub(previous.len());
                evictions = evictions.saturating_add(1);
            }
        }
        let evict_time = if evictions > 0 {
            evict_started.elapsed()
        } else {
            Duration::ZERO
        };

        let insert_started = Instant::now();
        self.tiles.insert(key, tile);
        self.bytes = self.bytes.saturating_add(bytes);
        DenseTensorTileCacheInsertStats {
            inserts: 1,
            evictions,
            insert_time: insert_started.elapsed(),
            evict_time,
        }
    }
}

impl DenseTensorCache {
    fn with_budget(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            ..Self::default()
        }
    }

    fn get(&self, name: &str) -> Option<Arc<Vec<f32>>> {
        self.tensors.get(name).cloned()
    }

    fn insert(&mut self, name: String, tensor: Arc<Vec<f32>>) {
        let bytes = tensor.len() * std::mem::size_of::<f32>();
        if bytes > self.max_bytes {
            return;
        }
        while self.bytes.saturating_add(bytes) > self.max_bytes && !self.tensors.is_empty() {
            let Some(victim) = self.tensors.keys().next().cloned() else {
                break;
            };
            if let Some(previous) = self.tensors.remove(&victim) {
                self.bytes = self
                    .bytes
                    .saturating_sub(previous.len() * std::mem::size_of::<f32>());
            }
        }
        if let Some(previous) = self.tensors.insert(name, tensor) {
            self.bytes = self
                .bytes
                .saturating_sub(previous.len() * std::mem::size_of::<f32>());
        }
        self.bytes = self.bytes.saturating_add(bytes);
    }
}

impl DenseStore {
    pub fn open(path: PathBuf, manifest_path: PathBuf) -> Result<Self> {
        let file = fs::File::open(&path)
            .with_context(|| format!("failed to open dense store {}", path.display()))?;
        let len = file
            .metadata()
            .with_context(|| format!("failed to stat dense store {}", path.display()))?
            .len();
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .with_context(|| format!("failed to memory-map dense store {}", path.display()))?
        };
        let registry = TensorRegistry::load(&manifest_path)?;
        Ok(Self {
            #[cfg(test)]
            manifest_path,
            len,
            mmap: Arc::new(mmap),
            registry,
            resident: Arc::new(std::sync::Mutex::new(DenseTensorCache::with_budget(
                512 * 1024 * 1024,
            ))),
            norm_weights: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            q4_mmap_projections: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            decoded_tiles: Arc::new(std::sync::Mutex::new(DenseTensorTileCache::with_budget(
                DENSE_DECODED_TILE_CACHE_BYTES,
            ))),
            #[cfg(test)]
            raw_tiles: Arc::new(std::sync::Mutex::new(DenseRawTensorTileCache::with_budget(
                DENSE_DECODED_TILE_CACHE_BYTES,
            ))),
            #[cfg(test)]
            decoded_full_tensors: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            #[cfg(test)]
            decoded_tensor_tiles: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    pub fn registry(&self) -> &TensorRegistry {
        &self.registry
    }

    #[cfg(test)]
    pub(in crate::inference::flashmoe) fn q4_mmap_projection_cache_len(&self) -> usize {
        self.q4_mmap_projections
            .lock()
            .expect("dense q4 projection cache poisoned")
            .len()
    }

    pub(in crate::inference::flashmoe) fn seed(
        &self,
        position: usize,
        previous: u32,
    ) -> Result<u64> {
        Ok(self
            .read_u64(position as u64)?
            .wrapping_add(u64::from(previous)))
    }

    pub(in crate::inference::flashmoe) fn embedding(
        &self,
        token: u32,
        width: usize,
    ) -> Result<Vec<f32>> {
        if let Some(row) =
            self.read_tensor_row_f32("model.embed_tokens.weight", token as usize, width)?
        {
            return Ok(row);
        }
        bail!(
            "Flash-MoE dense tensor registry cannot provide model.embed_tokens.weight row for token {token}; refusing synthetic embeddings"
        )
    }

    #[cfg(test)]
    pub(in crate::inference::flashmoe) fn project(
        &self,
        layer: usize,
        name: &str,
        input: &[f32],
        width: usize,
    ) -> Result<Vec<f32>> {
        let tensor_name = attention_tensor_name(layer, name);
        if let Some(projected) = self.matvec_tensor_prefix(&tensor_name, input, width)? {
            return Ok(projected);
        }
        let salt = self.tensor_seed(&tensor_name, stable_hash(name) ^ ((layer as u64) << 32));
        let mut out = vec![0.0f32; width];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (col, value) in input.iter().enumerate() {
                let bits = self.read_u64(salt ^ ((row as u64) << 20) ^ col as u64)?;
                let weight = ((bits >> 40) as f32 / ((1u64 << 24) as f32)) * 2.0 - 1.0;
                acc = value.mul_add(weight, acc);
            }
            *slot = acc / (input.len().max(1) as f32).sqrt();
        }
        Ok(out)
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(in crate::inference::flashmoe) fn project_with_metal(
        &self,
        metal: Option<&MetalExecutionFacade>,
        layer: usize,
        name: &str,
        input: &[f32],
        width: usize,
    ) -> Result<Vec<f32>> {
        let tensor_name = attention_tensor_name(layer, name);
        if let Some(entry) = self.registry.tensor(&tensor_name) {
            let (rows, cols) =
                validate_dense_matvec_shape(entry, &tensor_name, width, input.len())?;
            if let Some(metal) = metal {
                return self.metal_matvec_tiled(metal, &tensor_name, input, rows, cols, width);
            }
            if let TensorQuantization::Q4 { .. } = entry.quantization {
                return self.q4_matvec_tiled(&tensor_name, input, rows, cols, width);
            }
        }
        self.project(layer, name, input, width)
    }
    pub(in crate::inference::flashmoe) fn project_resident_tensors_from_cpu_input(
        &self,
        metal: &MetalExecutionFacade,
        specs: &[DenseProjectionRequest<'_>],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        if specs.is_empty() {
            bail!("FlashMoe scheduled resident projection batch has no projections");
        }
        metal.require_resident_dense_weights()?;
        let mut projections = Vec::with_capacity(specs.len());
        for spec in specs {
            let projection = self
                .resident_mmap_projection(spec.tensor_name, spec.output_width, input.len())?
                .with_context(|| {
                    format!(
                        "FlashMoe unsupported scheduled resident projection batch: missing projection {}",
                        spec.tensor_name
                    )
                })?;
            projections.push(projection);
        }
        let (outputs, _, _) = metal.resident_mmap_matvec_batch(&projections, input)?;
        Ok(outputs)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::inference::flashmoe) fn glm_mla_input_projections_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        layer: usize,
        layout: MlaAttentionLayout,
        input: MetalBatchProjectionInput<'_>,
        q_norm_weight: &[f32],
        kv_norm_weight: &[f32],
        norm_epsilon: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let q_a_name = attention_tensor_name(layer, "q_a_proj");
        let kv_a_name = attention_tensor_name(layer, "kv_a_proj_with_mqa");
        let q_b_name = attention_tensor_name(layer, "q_b_proj");
        let q_a = self
            .resident_mmap_projection(&q_a_name, layout.q_lora_rank, input.len())?
            .with_context(|| format!("missing resident GLM MLA projection {q_a_name}"))?;
        let kv_a = self
            .resident_mmap_projection(&kv_a_name, layout.kv_a_width, input.len())?
            .with_context(|| format!("missing resident GLM MLA projection {kv_a_name}"))?;
        let q_b = self
            .resident_mmap_projection(&q_b_name, layout.q_width, layout.q_lora_rank)?
            .with_context(|| format!("missing resident GLM MLA projection {q_b_name}"))?;
        metal.resident_glm_mla_input_projection_chain(
            &q_a,
            &kv_a,
            &q_b,
            input,
            q_norm_weight,
            kv_norm_weight,
            layout.kv_lora_rank,
            norm_epsilon,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::inference::flashmoe) fn glm_mla_fused_attention_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        layer: usize,
        layout: MlaAttentionLayout,
        input: MetalBatchProjectionInput<'_>,
        q_norm_weight: &[f32],
        kv_norm_weight: &[f32],
        norm_epsilon: f32,
        previous_records: &[(&[f32], &[f32])],
        rope_cos: &[f32],
        rope_sin: &[f32],
        post_attention: Option<GlmMlaPostAttentionRequest<'_>>,
    ) -> Result<MetalGlmMlaFusedAttentionOutput> {
        if layout.kv_projection != MlaKvProjectionLayout::AbsorbedMultiLinear {
            bail!(
                "MLA layer {layer} fused Metal execution requires pre-absorbed embed_q/unembed_out weights"
            );
        }
        let q_a_name = attention_tensor_name(layer, "q_a_proj");
        let kv_a_name = attention_tensor_name(layer, "kv_a_proj_with_mqa");
        let q_b_name = attention_tensor_name(layer, "q_b_proj");
        let embed_q_name = attention_tensor_name(layer, "embed_q");
        let unembed_out_name = attention_tensor_name(layer, "unembed_out");
        let q_a = self
            .resident_mmap_projection(&q_a_name, layout.q_lora_rank, input.len())?
            .with_context(|| format!("missing resident GLM MLA projection {q_a_name}"))?;
        let kv_a = self
            .resident_mmap_projection(&kv_a_name, layout.kv_a_width, input.len())?
            .with_context(|| format!("missing resident GLM MLA projection {kv_a_name}"))?;
        let q_b = self
            .resident_mmap_projection(&q_b_name, layout.q_width, layout.q_lora_rank)?
            .with_context(|| format!("missing resident GLM MLA projection {q_b_name}"))?;
        let embed_q = self
            .dense_q4_mmap_multilinear_projection(
                &embed_q_name,
                layout.num_heads,
                layout.kv_lora_rank,
                layout.qk_nope_head_dim,
            )?
            .with_context(|| {
                format!(
                    "GLM MLA fused Metal execution requires resident Q4 projection {embed_q_name}"
                )
            })?;
        let unembed_out = self
            .dense_q4_mmap_multilinear_projection(
                &unembed_out_name,
                layout.num_heads,
                layout.v_head_dim,
                layout.kv_lora_rank,
            )?
            .with_context(|| {
                format!(
                    "GLM MLA fused Metal execution requires resident Q4 projection {unembed_out_name}"
                )
            })?;
        let post_projections = post_attention
            .map(|post| {
                let out_proj_name = attention_tensor_name(layer, "o_proj");
                build_required_cmd2_resident_post_attention_prep_projections(
                    layer,
                    post.experts,
                    &out_proj_name,
                    layout.attention_output_width,
                    post.residual.len(),
                    post.active_experts,
                    |tensor_name, output_width, input_len| {
                        self.resident_mmap_projection(tensor_name, output_width, input_len)
                    },
                )
            })
            .transpose()?;

        let mut previous_record_latents = Vec::with_capacity(
            previous_records
                .len()
                .checked_mul(layout.kv_lora_rank)
                .context("MLA fused previous latent size overflow")?,
        );
        let mut previous_record_rotary = Vec::with_capacity(
            previous_records
                .len()
                .checked_mul(layout.qk_rope_head_dim)
                .context("MLA fused previous rotary size overflow")?,
        );
        for (latent, rotary) in previous_records {
            if latent.len() != layout.kv_lora_rank || rotary.len() != layout.qk_rope_head_dim {
                bail!(
                    "MLA layer {layer} previous cache record has latent/rotary widths {}/{}, expected {}/{}",
                    latent.len(),
                    rotary.len(),
                    layout.kv_lora_rank,
                    layout.qk_rope_head_dim,
                );
            }
            previous_record_latents.extend_from_slice(latent);
            previous_record_rotary.extend_from_slice(rotary);
        }
        let scale = (layout.qk_head_dim as f32).sqrt().recip();
        metal.resident_glm_mla_fused_attention(
            &q_a,
            &kv_a,
            &q_b,
            &embed_q,
            &unembed_out,
            MetalGlmMlaFusedAttentionInput {
                input,
                heads: layout.num_heads,
                latent_rank: layout.kv_lora_rank,
                nope_dim: layout.qk_nope_head_dim,
                rope_dim: layout.qk_rope_head_dim,
                previous_record_latents: &previous_record_latents,
                previous_record_rotary: &previous_record_rotary,
                rope_cos,
                rope_sin,
                scale,
                post_attention: post_attention.zip(post_projections.as_ref()).map(
                    |(post, projections)| MetalGlmMlaPostAttentionInput {
                        projections,
                        residual: post.residual,
                        post_norm_weight: post.post_norm_weight,
                        router_correction_bias: post.router_correction_bias,
                    },
                ),
            },
            q_norm_weight,
            kv_norm_weight,
            norm_epsilon,
        )
    }

    fn required_resident_static_tensor(
        &self,
        layer: usize,
        tensor_name: &str,
        expected_values: usize,
        allowed_dtypes: &[ResidentStaticDtype],
    ) -> Result<ResidentStaticTensorRef> {
        let entry = self.registry.require(tensor_name).with_context(|| {
            format!(
                "FlashMoe unsupported scheduled linear-attention static-weight path at layer {layer}: missing tensor {tensor_name}"
            )
        })?;
        ResidentStaticTensorRef::from_entry(
            tensor_name,
            entry,
            self.len,
            expected_values,
            allowed_dtypes,
        )?
        .with_context(|| {
            format!(
                "FlashMoe unsupported scheduled linear-attention static-weight path at layer {layer}: tensor {tensor_name} does not resolve {} values as {}",
                expected_values,
                allowed_dtypes
                    .iter()
                    .map(ResidentStaticDtype::as_str)
                    .collect::<Vec<_>>()
                    .join("/")
            )
        })
    }

    fn required_linear_attention_resident_bindings(
        &self,
        layer: usize,
        layout: LinearAttentionLayout,
        hidden_width: usize,
        experts: usize,
    ) -> Result<LinearAttentionResidentBindings> {
        let input_requests = linear_attention_input_projection_requests(
            layer,
            layout.conv_dim,
            layout.total_value_width,
            layout.num_value_heads,
        )?;
        let mut input_projections = Vec::with_capacity(4);
        for spec in input_requests.requests() {
            input_projections.push(
                self.resident_mmap_projection(spec.tensor_name, spec.output_width, hidden_width)?
                    .with_context(|| {
                        format!(
                            "FlashMoe unsupported scheduled linear-attention CMD1 path at layer {layer}: missing resident projection {}",
                            spec.tensor_name
                        )
                    })?,
            );
        }
        let input_projections = input_projections.try_into().map_err(|values: Vec<_>| {
            anyhow::anyhow!(
                "FlashMoe unsupported scheduled linear-attention CMD1 path at layer {layer}: expected 4 resident projections, resolved {}",
                values.len()
            )
        })?;

        let conv_name = linear_attention_tensor_name(layer, "conv1d");
        let a_log_name = linear_attention_scalar_tensor_name(layer, "A_log");
        let dt_bias_name = linear_attention_scalar_tensor_name(layer, "dt_bias");
        let norm_name = linear_attention_tensor_name(layer, "norm");
        let static_tensors = LinearAttentionStaticBindings {
            conv_weight: self.required_resident_static_tensor(
                layer,
                &conv_name,
                layout.conv_dim * layout.conv_kernel_size,
                &[
                    ResidentStaticDtype::Bf16,
                    ResidentStaticDtype::F16,
                    ResidentStaticDtype::F32,
                ],
            )?,
            a_log: self.required_resident_static_tensor(
                layer,
                &a_log_name,
                layout.num_value_heads,
                &[ResidentStaticDtype::F32],
            )?,
            dt_bias: self.required_resident_static_tensor(
                layer,
                &dt_bias_name,
                layout.num_value_heads,
                &[
                    ResidentStaticDtype::Bf16,
                    ResidentStaticDtype::F16,
                    ResidentStaticDtype::F32,
                ],
            )?,
            norm_weight: self.required_resident_static_tensor(
                layer,
                &norm_name,
                layout.value_dim,
                &[
                    ResidentStaticDtype::Bf16,
                    ResidentStaticDtype::F16,
                    ResidentStaticDtype::F32,
                ],
            )?,
        };
        let out_proj_name = linear_attention_tensor_name(layer, "out_proj");
        let out_proj = self
            .resident_mmap_projection(&out_proj_name, hidden_width, layout.total_value_width)?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled linear-attention CMD2 path at layer {layer}: missing resident output projection {out_proj_name}"
                )
            })?;
        let router_name = router_tensor_name(layer);
        let router = self
            .resident_mmap_projection(&router_name, experts, hidden_width)?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled linear-attention CMD2 path at layer {layer}: missing resident router projection {router_name}"
                )
            })?;
        Ok(LinearAttentionResidentBindings {
            layer,
            input_projections,
            static_tensors,
            out_proj,
            router,
        })
    }

    pub(in crate::inference::flashmoe) fn resolve_linear_attention_weight_table(
        &self,
        layouts: &[Option<LinearAttentionLayout>],
        hidden_width: usize,
        experts: usize,
    ) -> Result<LinearAttentionWeightTable> {
        let layers = layouts
            .iter()
            .copied()
            .enumerate()
            .map(|(layer, layout)| {
                layout
                    .map(|layout| {
                        self.required_linear_attention_resident_bindings(
                            layer,
                            layout,
                            hidden_width,
                            experts,
                        )
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(LinearAttentionWeightTable { layers })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::inference::flashmoe) fn linear_attention_post_attention_prep_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        layout: LinearAttentionLayout,
        bindings: &LinearAttentionResidentBindings,
        input: MetalBatchProjectionInput<'_>,
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        active_experts: usize,
    ) -> Result<MetalPostAttentionPrep> {
        let layer = bindings.layer;
        let residual_len = residual.len();
        metal.require_resident_dense_weights()?;
        if residual_len != post_norm_weight.len() {
            bail!(
                "FlashMoe unsupported scheduled Qwen3.5 linear-attention CMD1/CMD2 path at layer {layer}: residual/norm widths {residual_len}/{} do not match",
                post_norm_weight.len()
            );
        }
        metal.linear_attention_post_attention_prep(
            layout,
            bindings,
            input,
            residual,
            post_norm_weight,
            active_experts,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(in crate::inference::flashmoe) fn project_resident_tensors_from_metal_input(
        &self,
        metal: &MetalExecutionFacade,
        specs: &[DenseProjectionRequest<'_>],
        input_buffer: ObjcId,
        input_len: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if specs.is_empty() {
            bail!("FlashMoe scheduled resident projection batch has no projections");
        }
        metal.require_resident_dense_weights()?;
        let mut projections = Vec::with_capacity(specs.len());
        for spec in specs {
            let projection = self
                .resident_mmap_projection(spec.tensor_name, spec.output_width, input_len)?
                .with_context(|| {
                    format!(
                        "FlashMoe unsupported scheduled resident projection batch: missing projection {}",
                        spec.tensor_name
                    )
                })?;
            projections.push(projection);
        }
        let (outputs, _, _) = metal.resident_mmap_matvec_batch_with_input_buffer(
            &projections,
            input_buffer,
            input_len,
        )?;
        Ok(outputs)
    }

    /// Project using a fully-qualified canonical tensor name (e.g. for shared
    /// experts or any non-attention projection).  Falls back to a zero-vector
    /// when the tensor is absent (tensor not present in this checkpoint means
    /// the feature is disabled for this model variant).
    #[cfg(test)]
    pub(in crate::inference::flashmoe) fn project_dense_tensor_with_metal(
        &self,
        metal: Option<&MetalExecutionFacade>,
        tensor_name: &str,
        input: &[f32],
        output_width: usize,
    ) -> Result<Option<Vec<f32>>> {
        let entry = match self.registry.tensor(tensor_name) {
            Some(e) => e,
            None => return Ok(None),
        };
        let (rows, cols) =
            validate_dense_matvec_shape(entry, tensor_name, output_width, input.len())?;
        if let Some(metal) = metal {
            return self
                .metal_matvec_tiled(metal, tensor_name, input, rows, cols, output_width)
                .map(Some);
        }
        if let TensorQuantization::Q4 { .. } = entry.quantization {
            return self
                .q4_matvec_tiled(tensor_name, input, rows, cols, output_width)
                .map(Some);
        }
        if let Some(projected) = self.matvec_tensor_prefix(tensor_name, input, output_width)? {
            return Ok(Some(projected));
        }
        Ok(None)
    }

    #[cfg(test)]
    pub(in crate::inference::flashmoe) fn rms_norm(
        &self,
        canonical_name: &str,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let mut out = input.to_vec();
        let weight = self.norm_weight(canonical_name, input.len())?;
        rms_norm_with_weight_in_place(&mut out, weight.as_deref());
        Ok(out)
    }

    pub(in crate::inference::flashmoe) fn norm_weight(
        &self,
        canonical_name: &str,
        width: usize,
    ) -> Result<Option<Vec<f32>>> {
        let key = DenseNormWeightKey {
            name: canonical_name.to_string(),
            width,
        };
        if let Some(weight) = self
            .norm_weights
            .lock()
            .expect("dense norm weight cache poisoned")
            .get(&key)
            .cloned()
        {
            return Ok(Some((*weight).clone()));
        }
        let Some(weight) = self.read_tensor_row_f32(canonical_name, 0, width)? else {
            return Ok(None);
        };
        self.norm_weights
            .lock()
            .expect("dense norm weight cache poisoned")
            .insert(key, Arc::new(weight.clone()));
        Ok(Some(weight))
    }

    pub(in crate::inference::flashmoe) fn declared_router_projection(
        &self,
        tensor_name: &str,
        expert: usize,
        hidden: &[f32],
    ) -> Result<f32> {
        if let Some(row) = self.read_tensor_row_f32(tensor_name, expert, hidden.len())? {
            let acc = row
                .iter()
                .zip(hidden)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
            return Ok(acc);
        }
        bail!(
            "FlashMoe declared router projection {tensor_name} cannot provide row {expert}; refusing synthetic router fallback"
        )
    }

    pub(in crate::inference::flashmoe) fn router_scores(
        &self,
        score_plan: RouterScoreProjectionScorePlan<'_>,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let experts = score_plan.experts;
        if score_plan.source == RouterScoreProjectionScoreSource::ResidentDenseFullTensor
            && let Some(scores) = self.router_scores_with_accelerate(score_plan, hidden)?
        {
            return Ok(scores);
        }
        let tensor_name = score_plan.tensor_name.to_string();
        let mut router_scores = vec![0.0f32; experts];
        for (expert, score) in router_scores.iter_mut().enumerate() {
            *score = self.declared_router_projection(&tensor_name, expert, hidden)?;
        }
        Ok(router_scores)
    }

    pub(in crate::inference::flashmoe) fn router_score_projection_descriptor(
        &self,
        layer: usize,
        experts: usize,
        hidden_width: usize,
    ) -> Result<Option<RouterScoreProjectionDescriptor>> {
        build_router_score_projection_descriptor(
            layer,
            experts,
            hidden_width,
            self.len,
            |tensor_name| self.registry.tensor(tensor_name),
        )
    }

    pub(in crate::inference::flashmoe) fn dense_q4_mmap_projection(
        &self,
        tensor_name: &str,
        output_width: usize,
        input_len: usize,
    ) -> Result<Option<DenseQ4MmapMatvecProjection>> {
        let key = DenseQ4ProjectionKey::new(tensor_name, output_width, input_len);
        if let Some(projection) = self
            .q4_mmap_projections
            .lock()
            .expect("dense q4 projection cache poisoned")
            .get(&key)
            .cloned()
        {
            return Ok(Some((*projection).clone()));
        }
        let Some(projection) = build_dense_q4_mmap_projection(
            tensor_name,
            output_width,
            input_len,
            self.len,
            |name| self.registry.tensor(name),
        )?
        else {
            return Ok(None);
        };
        let projection = Arc::new(projection);
        let mut cache = self
            .q4_mmap_projections
            .lock()
            .expect("dense q4 projection cache poisoned");
        if let Some(existing) = cache.get(&key).cloned() {
            Ok(Some((*existing).clone()))
        } else {
            cache.insert(key, projection.clone());
            Ok(Some((*projection).clone()))
        }
    }

    fn dense_q4_mmap_multilinear_projection(
        &self,
        tensor_name: &str,
        heads: usize,
        output_width_per_head: usize,
        input_len: usize,
    ) -> Result<Option<DenseQ4MmapMatvecProjection>> {
        let output_width = heads
            .checked_mul(output_width_per_head)
            .context("dense Q4 multilinear output width overflow")?;
        let key = DenseQ4ProjectionKey::new(tensor_name, output_width, input_len);
        if let Some(projection) = self
            .q4_mmap_projections
            .lock()
            .expect("dense q4 projection cache poisoned")
            .get(&key)
            .cloned()
        {
            return Ok(Some((*projection).clone()));
        }
        let Some(entry) = self.registry.tensor(tensor_name) else {
            return Ok(None);
        };
        let Some(projection) = DenseQ4MmapMatvecProjection::from_multilinear_entry(
            tensor_name,
            entry,
            self.len,
            heads,
            output_width_per_head,
            input_len,
        )?
        else {
            return Ok(None);
        };
        let projection = Arc::new(projection);
        let mut cache = self
            .q4_mmap_projections
            .lock()
            .expect("dense q4 projection cache poisoned");
        if let Some(existing) = cache.get(&key).cloned() {
            Ok(Some((*existing).clone()))
        } else {
            cache.insert(key, projection.clone());
            Ok(Some((*projection).clone()))
        }
    }

    pub(in crate::inference::flashmoe) fn resident_mmap_projection(
        &self,
        tensor_name: &str,
        output_width: usize,
        input_len: usize,
    ) -> Result<Option<ResidentMmapMatvecProjection>> {
        let Some(entry) = self.registry.tensor(tensor_name) else {
            return Ok(None);
        };
        if matches!(&entry.quantization, TensorQuantization::Q4 { .. }) {
            return self
                .dense_q4_mmap_projection(tensor_name, output_width, input_len)
                .map(|projection| projection.map(ResidentMmapMatvecProjection::Q4));
        }
        ResidentMmapMatvecProjection::from_entry(
            tensor_name,
            entry,
            self.len,
            output_width,
            input_len,
        )
        .map(Some)
    }

    fn q4_affine_scalar(&self, byte_offset: u64, dtype: &str, index: usize) -> Result<f32> {
        let element_size = expert_scale_bias_dtype_size(dtype)
            .with_context(|| format!("unsupported Q4 scale/bias dtype {dtype}"))?;
        let start = usize::try_from(byte_offset)
            .context("Q4 scale/bias offset exceeds usize")?
            .checked_add(
                index
                    .checked_mul(element_size)
                    .context("Q4 scalar offset overflow")?,
            )
            .context("Q4 scalar offset overflow")?;
        let end = start
            .checked_add(element_size)
            .context("Q4 scalar range overflow")?;
        let bytes = self
            .mmap
            .get(start..end)
            .with_context(|| format!("Q4 scalar range {start}..{end} exceeds dense mmap"))?;
        if dtype.eq_ignore_ascii_case("F32")
            || dtype.eq_ignore_ascii_case("FLOAT32")
            || dtype.eq_ignore_ascii_case("FP32")
        {
            Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
        } else if dtype.eq_ignore_ascii_case("BF16") || dtype.eq_ignore_ascii_case("BFLOAT16") {
            let bits = u16::from_le_bytes(bytes.try_into().unwrap()) as u32;
            Ok(f32::from_bits(bits << 16))
        } else if dtype.eq_ignore_ascii_case("F16")
            || dtype.eq_ignore_ascii_case("FLOAT16")
            || dtype.eq_ignore_ascii_case("FP16")
        {
            Ok(f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
        } else {
            bail!("unsupported Q4 scale/bias dtype {dtype}")
        }
    }

    fn q4_row_add_scaled(
        &self,
        projection: &DenseQ4MmapMatvecProjection,
        row: usize,
        coefficient: f32,
        output: &mut [f32],
    ) -> Result<()> {
        if row >= projection.rows || output.len() != projection.cols {
            bail!(
                "Q4 row accumulation for {} has row {row}/{} and output width {}/{}",
                projection.tensor_name,
                projection.rows,
                output.len(),
                projection.cols
            );
        }
        let packed_start = usize::try_from(projection.packed_byte_offset)
            .context("Q4 packed offset exceeds usize")?
            .checked_add(
                row.checked_mul(projection.row_packed_bytes)
                    .context("Q4 packed row offset overflow")?,
            )
            .context("Q4 packed row offset overflow")?;
        let packed = self
            .mmap
            .get(packed_start..packed_start + projection.row_packed_bytes)
            .with_context(|| format!("Q4 packed row {row} exceeds dense mmap"))?;
        for group in 0..projection.groups_per_row {
            let scalar_index = row * projection.groups_per_row + group;
            let scale = self.q4_affine_scalar(
                projection.scales_byte_offset,
                &projection.scale_bias_dtype,
                scalar_index,
            )?;
            let bias = self.q4_affine_scalar(
                projection.biases_byte_offset,
                &projection.scale_bias_dtype,
                scalar_index,
            )?;
            let start = group * projection.group_size;
            let end = (start + projection.group_size).min(projection.cols);
            for col in start..end {
                let byte = packed[col / 2];
                let q = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                output[col] += coefficient * q.mul_add(scale, bias);
            }
        }
        Ok(())
    }

    fn q4_row_dot(
        &self,
        projection: &DenseQ4MmapMatvecProjection,
        row: usize,
        input: &[f32],
    ) -> Result<f32> {
        if row >= projection.rows || input.len() != projection.cols {
            bail!(
                "Q4 row dot for {} has row {row}/{} and input width {}/{}",
                projection.tensor_name,
                projection.rows,
                input.len(),
                projection.cols
            );
        }
        let packed_start = usize::try_from(projection.packed_byte_offset)
            .context("Q4 packed offset exceeds usize")?
            .checked_add(
                row.checked_mul(projection.row_packed_bytes)
                    .context("Q4 packed row offset overflow")?,
            )
            .context("Q4 packed row offset overflow")?;
        let packed = self
            .mmap
            .get(packed_start..packed_start + projection.row_packed_bytes)
            .with_context(|| format!("Q4 packed row {row} exceeds dense mmap"))?;
        let mut sum = 0.0;
        for group in 0..projection.groups_per_row {
            let scalar_index = row * projection.groups_per_row + group;
            let scale = self.q4_affine_scalar(
                projection.scales_byte_offset,
                &projection.scale_bias_dtype,
                scalar_index,
            )?;
            let bias = self.q4_affine_scalar(
                projection.biases_byte_offset,
                &projection.scale_bias_dtype,
                scalar_index,
            )?;
            let start = group * projection.group_size;
            let end = (start + projection.group_size).min(projection.cols);
            for col in start..end {
                let byte = packed[col / 2];
                let q = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                sum += input[col] * q.mul_add(scale, bias);
            }
        }
        Ok(sum)
    }

    pub(in crate::inference::flashmoe) fn mla_absorbed_attention(
        &self,
        layer: usize,
        layout: MlaAttentionLayout,
        query: &[f32],
        records: &[(&[f32], &[f32])],
    ) -> Result<Vec<f32>> {
        if query.len() != layout.q_width || records.is_empty() {
            bail!(
                "MLA layer {layer} requires query width {} and a non-empty KV cache, got {} and {} records",
                layout.q_width,
                query.len(),
                records.len()
            );
        }
        for (latent, rotary) in records {
            if latent.len() != layout.kv_lora_rank || rotary.len() != layout.qk_rope_head_dim {
                bail!(
                    "MLA layer {layer} cache record has latent/rotary widths {}/{}, expected {}/{}",
                    latent.len(),
                    rotary.len(),
                    layout.kv_lora_rank,
                    layout.qk_rope_head_dim
                );
            }
        }
        enum AbsorbedWeights {
            Fused(DenseQ4MmapMatvecProjection),
            MultiLinear {
                embed_q: DenseQ4MmapMatvecProjection,
                unembed_out: DenseQ4MmapMatvecProjection,
            },
        }
        let weights = match layout.kv_projection {
            MlaKvProjectionLayout::FusedKvB => {
                let tensor_name = attention_tensor_name(layer, "kv_b_proj");
                let projection = self
                    .dense_q4_mmap_projection(
                        &tensor_name,
                        layout.kv_b_width,
                        layout.kv_lora_rank,
                    )?
                    .with_context(|| {
                        format!(
                            "GLM MLA weight absorption requires resident Q4 projection {tensor_name}"
                        )
                    })?;
                AbsorbedWeights::Fused(projection)
            }
            MlaKvProjectionLayout::AbsorbedMultiLinear => {
                let embed_q_name = attention_tensor_name(layer, "embed_q");
                let embed_q = self
                    .dense_q4_mmap_multilinear_projection(
                        &embed_q_name,
                        layout.num_heads,
                        layout.kv_lora_rank,
                        layout.qk_nope_head_dim,
                    )?
                    .with_context(|| {
                        format!(
                            "GLM MLA weight absorption requires resident Q4 projection {embed_q_name}"
                        )
                    })?;
                let unembed_out_name = attention_tensor_name(layer, "unembed_out");
                let unembed_out = self
                    .dense_q4_mmap_multilinear_projection(
                        &unembed_out_name,
                        layout.num_heads,
                        layout.v_head_dim,
                        layout.kv_lora_rank,
                    )?
                    .with_context(|| {
                        format!(
                            "GLM MLA weight absorption requires resident Q4 projection {unembed_out_name}"
                        )
                    })?;
                AbsorbedWeights::MultiLinear {
                    embed_q,
                    unembed_out,
                }
            }
        };
        let mut output = vec![0.0; layout.attention_output_width];
        let scale = (layout.qk_head_dim as f32).sqrt().recip();
        let kv_b_head_width = layout.qk_nope_head_dim + layout.v_head_dim;
        for head in 0..layout.num_heads {
            let query_head = &query[head * layout.qk_head_dim..(head + 1) * layout.qk_head_dim];
            let (query_nope, query_rope) = query_head.split_at(layout.qk_nope_head_dim);
            let absorbed_query = match &weights {
                AbsorbedWeights::Fused(projection) => {
                    let row_base = head * kv_b_head_width;
                    let mut absorbed_query = vec![0.0; layout.kv_lora_rank];
                    for (dimension, coefficient) in query_nope.iter().copied().enumerate() {
                        self.q4_row_add_scaled(
                            projection,
                            row_base + dimension,
                            coefficient,
                            &mut absorbed_query,
                        )?;
                    }
                    absorbed_query
                }
                AbsorbedWeights::MultiLinear { embed_q, .. } => (0..layout.kv_lora_rank)
                    .map(|dimension| {
                        self.q4_row_dot(embed_q, head * layout.kv_lora_rank + dimension, query_nope)
                    })
                    .collect::<Result<Vec<_>>>()?,
            };
            let mut scores = records
                .iter()
                .map(|(latent, rotary)| {
                    let latent_score = absorbed_query
                        .iter()
                        .zip(*latent)
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    let rotary_score = query_rope
                        .iter()
                        .zip(*rotary)
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    (latent_score + rotary_score) * scale
                })
                .collect::<Vec<_>>();
            softmax_in_place(&mut scores);
            let mut context = vec![0.0; layout.kv_lora_rank];
            for (weight, (latent, _)) in scores.iter().zip(records) {
                for (slot, value) in context.iter_mut().zip(*latent) {
                    *slot += *weight * value;
                }
            }
            let head_output = &mut output[head * layout.v_head_dim..(head + 1) * layout.v_head_dim];
            for (dimension, slot) in head_output.iter_mut().enumerate() {
                *slot = match &weights {
                    AbsorbedWeights::Fused(projection) => self.q4_row_dot(
                        projection,
                        head * kv_b_head_width + layout.qk_nope_head_dim + dimension,
                        &context,
                    )?,
                    AbsorbedWeights::MultiLinear { unembed_out, .. } => self.q4_row_dot(
                        unembed_out,
                        head * layout.v_head_dim + dimension,
                        &context,
                    )?,
                };
            }
        }
        Ok(output)
    }

    pub(in crate::inference::flashmoe) fn mla_absorbed_attention_metal(
        &self,
        metal: &MetalExecutionFacade,
        layer: usize,
        layout: MlaAttentionLayout,
        query: &[f32],
        records: &[(&[f32], &[f32])],
    ) -> Result<Vec<f32>> {
        if query.len() != layout.q_width || records.is_empty() {
            bail!(
                "MLA layer {layer} requires query width {} and a non-empty KV cache, got {} and {} records",
                layout.q_width,
                query.len(),
                records.len()
            );
        }
        for (latent, rotary) in records {
            if latent.len() != layout.kv_lora_rank || rotary.len() != layout.qk_rope_head_dim {
                bail!(
                    "MLA layer {layer} cache record has latent/rotary widths {}/{}, expected {}/{}",
                    latent.len(),
                    rotary.len(),
                    layout.kv_lora_rank,
                    layout.qk_rope_head_dim
                );
            }
        }
        if layout.kv_projection != MlaKvProjectionLayout::AbsorbedMultiLinear {
            bail!(
                "MLA layer {layer} Metal multilinear execution requires pre-absorbed embed_q/unembed_out weights"
            );
        }
        let embed_q_name = attention_tensor_name(layer, "embed_q");
        let embed_q = self
            .dense_q4_mmap_multilinear_projection(
                &embed_q_name,
                layout.num_heads,
                layout.kv_lora_rank,
                layout.qk_nope_head_dim,
            )?
            .with_context(|| {
                format!(
                    "GLM MLA Metal weight absorption requires resident Q4 projection {embed_q_name}"
                )
            })?;
        let unembed_out_name = attention_tensor_name(layer, "unembed_out");
        let unembed_out = self
            .dense_q4_mmap_multilinear_projection(
                &unembed_out_name,
                layout.num_heads,
                layout.v_head_dim,
                layout.kv_lora_rank,
            )?
            .with_context(|| {
                format!(
                    "GLM MLA Metal weight absorption requires resident Q4 projection {unembed_out_name}"
                )
            })?;

        let mut query_nope = Vec::with_capacity(
            layout
                .num_heads
                .checked_mul(layout.qk_nope_head_dim)
                .context("MLA no-PE query size overflow")?,
        );
        let mut query_rope = Vec::with_capacity(
            layout
                .num_heads
                .checked_mul(layout.qk_rope_head_dim)
                .context("MLA rotary query size overflow")?,
        );
        for head in 0..layout.num_heads {
            let start = head * layout.qk_head_dim;
            query_nope.extend_from_slice(&query[start..start + layout.qk_nope_head_dim]);
            query_rope.extend_from_slice(
                &query[start + layout.qk_nope_head_dim..start + layout.qk_head_dim],
            );
        }
        let mut record_latents = Vec::with_capacity(
            records
                .len()
                .checked_mul(layout.kv_lora_rank)
                .context("MLA latent record size overflow")?,
        );
        let mut record_rotary = Vec::with_capacity(
            records
                .len()
                .checked_mul(layout.qk_rope_head_dim)
                .context("MLA rotary record size overflow")?,
        );
        for (latent, rotary) in records {
            record_latents.extend_from_slice(latent);
            record_rotary.extend_from_slice(rotary);
        }
        let scale = (layout.qk_head_dim as f32).sqrt().recip();
        metal.resident_glm_mla_absorbed_attention(
            &embed_q,
            &unembed_out,
            MetalGlmMlaAbsorbedAttentionInput {
                heads: layout.num_heads,
                latent_rank: layout.kv_lora_rank,
                query_nope: &query_nope,
                query_rope: &query_rope,
                record_latents: &record_latents,
                record_rotary: &record_rotary,
                sequence: records.len(),
                rope_dim: layout.qk_rope_head_dim,
                scale,
            },
        )
    }

    #[cfg(test)]
    pub(in crate::inference::flashmoe) fn resolve_shared_expert_weight_table(
        &self,
        layer_count: usize,
        width: usize,
        shared_experts: usize,
        intermediate: usize,
    ) -> Result<SharedExpertWeightTable> {
        self.resolve_shared_expert_weight_table_from(
            layer_count,
            width,
            shared_experts,
            intermediate,
            0,
            true,
        )
    }

    pub(in crate::inference::flashmoe) fn resolve_shared_expert_weight_table_from(
        &self,
        layer_count: usize,
        width: usize,
        shared_experts: usize,
        intermediate: usize,
        first_sparse_layer: usize,
        requires_router: bool,
    ) -> Result<SharedExpertWeightTable> {
        let layers = (0..layer_count)
            .map(|layer| {
                if layer < first_sparse_layer {
                    return Ok(SharedExpertLayerWeights::None);
                }
                build_required_shared_expert_resident_phase_projections_with_router(
                    layer,
                    width,
                    shared_experts,
                    intermediate,
                    requires_router,
                    |tensor_name, output_width, input_len| {
                        self.resident_mmap_projection(tensor_name, output_width, input_len)
                    },
                )
                .map(|weights| match weights {
                    Some(weights) => SharedExpertLayerWeights::Resident(weights),
                    None => SharedExpertLayerWeights::None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SharedExpertWeightTable { layers })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(in crate::inference::flashmoe) fn layer_major_post_attention_projections(
        &self,
        layer: usize,
        experts: usize,
        out_proj_name: &str,
        attention_width: usize,
        residual_width: usize,
        active_experts: usize,
    ) -> Result<Cmd2ResidentPostAttentionPrepProjections> {
        build_required_cmd2_resident_post_attention_prep_projections(
            layer,
            experts,
            out_proj_name,
            attention_width,
            residual_width,
            active_experts,
            |tensor_name, output_width, input_len| {
                self.resident_mmap_projection(tensor_name, output_width, input_len)
            },
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(in crate::inference::flashmoe) fn post_attention_prep_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        layer: usize,
        experts: usize,
        out_proj_name: &str,
        attention_output: &[f32],
        residual: MetalBatchProjectionInput<'_>,
        post_norm_weight: &[f32],
        active_experts: usize,
        router_correction_bias: Option<&[f32]>,
    ) -> Result<MetalPostAttentionPrep> {
        metal.require_resident_dense_weights()?;
        let residual_len = residual.len();
        let projections = build_required_cmd2_resident_post_attention_prep_projections(
            layer,
            experts,
            out_proj_name,
            attention_output.len(),
            residual_len,
            active_experts,
            |tensor_name, output_width, input_len| {
                self.resident_mmap_projection(tensor_name, output_width, input_len)
            },
        )?;
        metal.resident_post_attention_prep_topk(
            &projections,
            attention_output,
            residual,
            post_norm_weight,
            router_correction_bias,
        )
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(in crate::inference::flashmoe) fn router_topk_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        layer: usize,
        experts: usize,
        hidden: &[f32],
        active_experts: usize,
    ) -> Result<Option<Vec<(usize, f32)>>> {
        if active_experts == 0 || !metal.has_resident_dense_weights() {
            return Ok(None);
        }
        let Some(descriptor) = build_router_score_projection_descriptor(
            layer,
            experts,
            hidden.len(),
            self.len,
            |tensor_name| self.registry.tensor(tensor_name),
        )?
        else {
            return Ok(None);
        };
        let plan = descriptor.topk_plan(hidden.len(), active_experts)?;
        metal.router_score_top_candidates(&plan, hidden)
    }

    pub(in crate::inference::flashmoe) fn router_scores_with_accelerate(
        &self,
        score_plan: RouterScoreProjectionScorePlan<'_>,
        hidden: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        if score_plan.source != RouterScoreProjectionScoreSource::ResidentDenseFullTensor {
            return Ok(None);
        }
        if score_plan.hidden_width != hidden.len() {
            return Ok(None);
        }
        let weights =
            self.read_tensor_rows_f32_cached(score_plan.tensor_name, 0, score_plan.experts)?;
        dense_f32_matvec_rows(
            weights.as_slice(),
            hidden,
            score_plan.experts,
            score_plan.hidden_width,
        )
    }

    pub(in crate::inference::flashmoe) fn lm_head_logits_with_metal(
        &self,
        metal: Option<&MetalExecutionFacade>,
        hidden: &[f32],
        vocab_size: usize,
    ) -> Result<Vec<f32>> {
        let lm_head_name = self.lm_head_tensor_name()?;
        if let Some(metal) = metal
            && let Some(entry) = self.registry.tensor(lm_head_name)
        {
            let (rows, cols) =
                validate_lm_head_matvec_shape(entry, lm_head_name, vocab_size, hidden.len())?;
            let mut logits = vec![f32::NEG_INFINITY; vocab_size];
            let projected =
                self.metal_matvec_tiled(metal, lm_head_name, hidden, rows, cols, rows)?;
            for (token, value) in projected.into_iter().take(vocab_size).enumerate() {
                logits[token] = value;
            }
            return Ok(logits);
        }

        self.lm_head_logits(lm_head_name, hidden, vocab_size)
    }

    pub(in crate::inference::flashmoe) fn lm_head_raw_top_candidates_with_metal(
        &self,
        metal: &MetalExecutionFacade,
        hidden: &[f32],
        vocab_size: usize,
        candidate_count: usize,
    ) -> Result<Vec<(usize, f32)>> {
        metal.require_resident_dense_weights()?;
        let lm_head_name = self.lm_head_tensor_name()?;
        let entry = self.registry.require(lm_head_name)?;
        let (rows, cols) =
            validate_lm_head_matvec_shape(entry, lm_head_name, vocab_size, hidden.len())?;

        let candidate_count = candidate_count.min(vocab_size).max(1);
        let projection = self
            .resident_mmap_projection(lm_head_name, rows, cols)?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported resolved LM-head path: missing resident projection {lm_head_name}"
                )
            })?;
        metal.resident_top_candidates(&projection, hidden, vocab_size, candidate_count)
    }

    pub(in crate::inference::flashmoe) fn lm_head_raw_top_candidates_with_metal_masked(
        &self,
        metal: &MetalExecutionFacade,
        hidden: &[f32],
        vocab_size: usize,
        candidate_count: usize,
        allowed_tokens: &[u32],
    ) -> Result<Vec<(usize, f32)>> {
        metal.require_resident_dense_weights()?;
        let lm_head_name = self.lm_head_tensor_name()?;
        let entry = self.registry.require(lm_head_name)?;
        let (rows, cols) =
            validate_lm_head_matvec_shape(entry, lm_head_name, vocab_size, hidden.len())?;

        let candidate_count = candidate_count.min(vocab_size).max(1);
        let projection = self
            .resident_mmap_projection(lm_head_name, rows, cols)?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported resolved LM-head path: missing resident projection {lm_head_name}"
                )
            })?;
        metal.resident_top_candidates_masked(
            &projection,
            hidden,
            vocab_size,
            candidate_count,
            allowed_tokens,
        )
    }

    pub(in crate::inference::flashmoe) fn lm_head_logits(
        &self,
        lm_head_name: &str,
        hidden: &[f32],
        vocab_size: usize,
    ) -> Result<Vec<f32>> {
        let entry = self.registry.require(lm_head_name)?;
        validate_lm_head_matvec_shape(entry, lm_head_name, vocab_size, hidden.len())?;
        let mut logits = vec![f32::NEG_INFINITY; vocab_size];
        for idx in 0..vocab_size {
            let Some(row) = self.read_tensor_row_f32(lm_head_name, idx, hidden.len())? else {
                bail!(
                    "Flash-MoE LM head tensor {lm_head_name} cannot provide row for token {idx}; refusing synthetic logits"
                );
            };
            logits[idx] = row
                .iter()
                .zip(hidden)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
        }
        Ok(logits)
    }

    pub(in crate::inference::flashmoe) fn lm_head_tensor_name(&self) -> Result<&'static str> {
        if self.registry.tensor("lm_head.weight").is_some() {
            Ok("lm_head.weight")
        } else if self.registry.tensor("model.embed_tokens.weight").is_some() {
            Ok("model.embed_tokens.weight")
        } else {
            bail!(
                "Flash-MoE dense tensor registry is missing lm_head.weight and tied model.embed_tokens.weight"
            )
        }
    }

    pub(in crate::inference::flashmoe) fn matvec_tensor_prefix(
        &self,
        canonical_name: &str,
        input: &[f32],
        width: usize,
    ) -> Result<Option<Vec<f32>>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            return Ok(None);
        };
        if entry.quantization != TensorQuantization::None {
            bail!("dense q4 tensor {canonical_name} cannot be read as a full f32 tensor");
        }
        let (rows, cols) = validate_dense_matvec_shape(entry, canonical_name, width, input.len())?;
        if let Some(tensor) = self.dense_tensor_f32(canonical_name)? {
            let expected_len = rows
                .checked_mul(cols)
                .context("dense resident tensor value count overflow")?;
            if tensor.len() != expected_len {
                bail!(
                    "Flash-MoE dense tensor {canonical_name} has {} decoded values; expected {expected_len} for shape {:?} and input length {}",
                    tensor.len(),
                    entry.shape,
                    input.len()
                );
            }
            let mut out = vec![0.0f32; width];
            for (row, slot) in out.iter_mut().take(rows).enumerate() {
                let start = row
                    .checked_mul(cols)
                    .context("dense resident tensor row offset overflow")?;
                let end = start
                    .checked_add(cols)
                    .context("dense resident tensor row length overflow")?;
                let weights = &tensor[start..end];
                let acc = weights
                    .iter()
                    .zip(input.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>();
                *slot = acc;
            }
            return Ok(Some(out));
        }
        let mut out = vec![0.0f32; width];
        for (row, slot) in out.iter_mut().take(rows).enumerate() {
            let weights = self.read_tensor_row_f32(canonical_name, row, cols)?;
            let Some(weights) = weights else {
                return Ok(None);
            };
            let acc = weights
                .iter()
                .zip(input.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
            *slot = acc;
        }
        Ok(Some(out))
    }

    pub(in crate::inference::flashmoe) fn metal_matvec_tiled(
        &self,
        metal: &MetalExecutionFacade,
        canonical_name: &str,
        input: &[f32],
        rows: usize,
        cols: usize,
        output_width: usize,
    ) -> Result<Vec<f32>> {
        let entry = self.registry.tensor(canonical_name).with_context(|| {
            format!("Flash-MoE dense tensor registry is missing {canonical_name}")
        })?;
        validate_dense_matvec_shape(entry, canonical_name, output_width, input.len())?;
        if rows != output_width || cols != input.len() {
            bail!(
                "FlashMoe scheduled Q4 projection {canonical_name} dimensions do not match output/input widths"
            );
        }
        let projection = self
            .dense_q4_mmap_projection(canonical_name, output_width, input.len())?
            .with_context(|| {
                format!(
                    "FlashMoe unsupported scheduled Q4 projection: missing resident descriptor for {canonical_name}"
                )
            })?;
        let projection = ResidentMmapMatvecProjection::Q4(projection);
        let (mut outputs, _, _) =
            metal.resident_mmap_matvec_batch(std::slice::from_ref(&projection), input)?;
        outputs
            .pop()
            .with_context(|| format!("Metal Q4 projection {canonical_name} returned no output"))
    }

    #[cfg(test)]
    pub(in crate::inference::flashmoe) fn q4_matvec_tiled(
        &self,
        canonical_name: &str,
        input: &[f32],
        rows: usize,
        cols: usize,
        output_width: usize,
    ) -> Result<Vec<f32>> {
        let entry = self.registry.tensor(canonical_name).with_context(|| {
            format!("Flash-MoE dense tensor registry is missing {canonical_name}")
        })?;
        let TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } = &entry.quantization
        else {
            bail!("Flash-MoE dense tensor {canonical_name} is not q4-quantized");
        };
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&entry.shape, *group_size, scale_bias_dtype)?;
        if rows != layout.rows || cols != layout.cols {
            bail!(
                "Flash-MoE dense q4 tensor {canonical_name} matvec dimensions mismatch: layout rows={}, cols={}, requested rows={rows}, cols={cols}",
                layout.rows,
                layout.cols
            );
        }
        let mut output = vec![0.0f32; output_width];
        let tile_rows = dense_projection_tile_rows(cols, rows);
        for start in (0..rows).step_by(tile_rows) {
            let end = (start + tile_rows).min(rows);
            let rows = end - start;
            let (packed, scales, biases, _) =
                self.read_dense_q4_rows(entry, start, rows, *group_size)?;
            let projected = q4_fma_matvec_with_group_size(
                &packed,
                input,
                &scales,
                &biases,
                rows,
                cols,
                *group_size,
            )?;
            output[start..end].copy_from_slice(&projected);
        }
        Ok(output)
    }

    pub(in crate::inference::flashmoe) fn dense_tensor_f32(
        &self,
        canonical_name: &str,
    ) -> Result<Option<Arc<Vec<f32>>>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            return Ok(None);
        };
        if let Some(tensor) = self
            .resident
            .lock()
            .expect("dense tensor cache poisoned")
            .get(canonical_name)
        {
            return Ok(Some(tensor));
        }
        if let TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } = &entry.quantization
        {
            let layout =
                dense_q4_layout_with_scale_bias_dtype(&entry.shape, *group_size, scale_bias_dtype)?;
            let decoded_bytes = layout
                .rows
                .checked_mul(layout.cols)
                .and_then(|items| items.checked_mul(std::mem::size_of::<f32>()))
                .context("dense q4 full tensor decoded byte length overflow")?;
            if decoded_bytes > DENSE_Q4_FULL_DECODE_MAX_BYTES {
                bail!(
                    "dense q4 tensor {canonical_name} would decode to {decoded_bytes} bytes, over full decode limit {DENSE_Q4_FULL_DECODE_MAX_BYTES}"
                );
            }
            let (packed, scales, biases, _) =
                self.read_dense_q4_rows(entry, 0, layout.rows, *group_size)?;
            let tensor = Arc::new(q4_dequantize_rows_with_group_size(
                &packed,
                &scales,
                &biases,
                layout.rows,
                layout.cols,
                *group_size,
            )?);
            #[cfg(test)]
            self.decoded_full_tensors
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.resident
                .lock()
                .expect("dense tensor cache poisoned")
                .insert(canonical_name.to_string(), tensor.clone());
            return Ok(Some(tensor));
        }
        let bytes = self.read_range(entry.byte_offset, entry.byte_len as usize)?;
        let tensor = Arc::new(decode_dense_tensor_f32(&entry.dtype, &bytes)?);
        #[cfg(test)]
        self.decoded_full_tensors
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.resident
            .lock()
            .expect("dense tensor cache poisoned")
            .insert(canonical_name.to_string(), tensor.clone());
        Ok(Some(tensor))
    }

    pub(in crate::inference::flashmoe) fn read_tensor_rows_f32_cached(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<Arc<Vec<f32>>> {
        let (tile, _) =
            self.read_tensor_rows_f32_cached_profiled(canonical_name, start_row, row_count)?;
        Ok(tile)
    }

    pub(in crate::inference::flashmoe) fn read_tensor_rows_f32_cached_profiled(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<(Arc<Vec<f32>>, DenseTileReadTiming)> {
        let started = Instant::now();
        let key = DenseTensorTileKey {
            name: canonical_name.to_string(),
            start_row,
            row_count,
        };
        if let Some(tile) = self
            .decoded_tiles
            .lock()
            .expect("dense decoded tile cache poisoned")
            .get(&key)
        {
            let mut timing = DenseTileReadTiming {
                cache_hits: 1,
                ..DenseTileReadTiming::default()
            };
            timing.total = started.elapsed();
            return Ok((tile, timing));
        }
        let mut timing = DenseTileReadTiming {
            cache_misses: 1,
            ..DenseTileReadTiming::default()
        };
        let (decoded, uncached_timing) =
            self.read_tensor_rows_f32_profiled(canonical_name, start_row, row_count)?;
        timing.add(uncached_timing);
        let tile = Arc::new(decoded);
        let stats = self
            .decoded_tiles
            .lock()
            .expect("dense decoded tile cache poisoned")
            .insert(key, tile.clone());
        timing.cache_inserts = timing.cache_inserts.saturating_add(stats.inserts);
        timing.cache_evictions = timing.cache_evictions.saturating_add(stats.evictions);
        timing.cache_insert += stats.insert_time;
        timing.cache_evict += stats.evict_time;
        timing.total = started.elapsed();
        Ok((tile, timing))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(in crate::inference::flashmoe) fn read_tensor_rows_raw_cached_profiled(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<(Arc<Vec<u8>>, String, DenseTileReadTiming)> {
        let started = Instant::now();
        let key = DenseTensorTileKey {
            name: canonical_name.to_string(),
            start_row,
            row_count,
        };
        if let Some(tile) = self
            .raw_tiles
            .lock()
            .expect("dense raw tile cache poisoned")
            .get(&key)
        {
            let mut timing = DenseTileReadTiming {
                cache_hits: 1,
                ..DenseTileReadTiming::default()
            };
            timing.total = started.elapsed();
            let dtype = self
                .registry
                .tensor(canonical_name)
                .map(|entry| entry.dtype.clone())
                .with_context(|| {
                    format!("Flash-MoE dense tensor registry is missing {canonical_name}")
                })?;
            return Ok((tile, dtype, timing));
        }

        let mut timing = DenseTileReadTiming {
            cache_misses: 1,
            ..DenseTileReadTiming::default()
        };
        let (bytes, dtype, uncached_timing) =
            self.read_tensor_rows_raw_profiled(canonical_name, start_row, row_count)?;
        timing.add(uncached_timing);
        let tile = Arc::new(bytes);
        let stats = self
            .raw_tiles
            .lock()
            .expect("dense raw tile cache poisoned")
            .insert(key, tile.clone());
        timing.cache_inserts = timing.cache_inserts.saturating_add(stats.inserts);
        timing.cache_evictions = timing.cache_evictions.saturating_add(stats.evictions);
        timing.cache_insert += stats.insert_time;
        timing.cache_evict += stats.evict_time;
        timing.total = started.elapsed();
        Ok((tile, dtype, timing))
    }

    pub(in crate::inference::flashmoe) fn read_dense_q4_rows(
        &self,
        entry: &RuntimeTensorEntry,
        start_row: usize,
        row_count: usize,
        group_size: usize,
    ) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>, DenseTileReadTiming)> {
        let started = Instant::now();
        let mut timing = DenseTileReadTiming::default();
        let TensorQuantization::Q4 {
            scale_bias_dtype, ..
        } = &entry.quantization
        else {
            bail!("dense tensor {} is not q4-quantized", entry.name);
        };
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&entry.shape, group_size, scale_bias_dtype)?;
        if entry.byte_len as usize != layout.total_bytes {
            bail!(
                "dense q4 tensor {} byte length {} does not match computed layout {}",
                entry.name,
                entry.byte_len,
                layout.total_bytes
            );
        }
        let end_row = start_row
            .checked_add(row_count)
            .context("dense q4 tile row range overflow")?;
        if end_row > layout.rows {
            bail!(
                "dense q4 tensor {} rows {}..{} exceed row count {}",
                entry.name,
                start_row,
                end_row,
                layout.rows
            );
        }
        if row_count == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new(), timing));
        }
        let packed_offset = start_row
            .checked_mul(layout.row_packed_bytes)
            .context("dense q4 packed tile offset overflow")?;
        let packed_len = row_count
            .checked_mul(layout.row_packed_bytes)
            .context("dense q4 packed tile length overflow")?;
        let groups_offset = start_row
            .checked_mul(layout.groups_per_row)
            .and_then(|groups| groups.checked_mul(layout.scale_bias_bytes))
            .context("dense q4 groups tile offset overflow")?;
        let groups_len = row_count
            .checked_mul(layout.groups_per_row)
            .and_then(|groups| groups.checked_mul(layout.scale_bias_bytes))
            .context("dense q4 groups tile byte length overflow")?;

        let (packed, read_packed) =
            self.read_range_profiled(entry.byte_offset + packed_offset as u64, packed_len)?;
        let (scale_bytes, read_scales) = self.read_range_profiled(
            entry.byte_offset + layout.packed_bytes as u64 + groups_offset as u64,
            groups_len,
        )?;
        let (bias_bytes, read_biases) = self.read_range_profiled(
            entry.byte_offset
                + layout.packed_bytes as u64
                + layout.scales_bytes as u64
                + groups_offset as u64,
            groups_len,
        )?;
        timing.read_range += read_packed + read_scales + read_biases;
        timing.bytes_read = timing
            .bytes_read
            .saturating_add((packed_len + groups_len + groups_len) as u64);
        let decode_started = Instant::now();
        let scales = decode_dense_tensor_f32(scale_bias_dtype, &scale_bytes)?;
        let biases = decode_dense_tensor_f32(scale_bias_dtype, &bias_bytes)?;
        timing.decode += decode_started.elapsed();
        timing.decoded_bytes = timing
            .decoded_bytes
            .saturating_add(((scales.len() + biases.len()) * std::mem::size_of::<f32>()) as u64);
        timing.total = started.elapsed();
        Ok((packed, scales, biases, timing))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(in crate::inference::flashmoe) fn read_tensor_rows_raw_profiled(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<(Vec<u8>, String, DenseTileReadTiming)> {
        let started = Instant::now();
        let mut timing = DenseTileReadTiming::default();
        let Some(entry) = self.registry.tensor(canonical_name) else {
            bail!("Flash-MoE dense tensor registry is missing {canonical_name}");
        };
        if entry.quantization != TensorQuantization::None {
            bail!("dense q4 tensor {canonical_name} cannot be read as raw dense rows");
        }
        let Some(element_size) = dtype_size(&entry.dtype) else {
            bail!(
                "Flash-MoE dense tensor {} has unsupported dtype {}",
                entry.name,
                entry.dtype
            );
        };
        let cols = entry.shape.last().copied().unwrap_or(0);
        if entry.shape.is_empty() || cols == 0 || row_count == 0 {
            return Ok((Vec::new(), entry.dtype.clone(), timing));
        }
        let rows = entry
            .shape
            .iter()
            .take(entry.shape.len() - 1)
            .product::<usize>()
            .max(1);
        let end_row = start_row
            .checked_add(row_count)
            .context("dense tensor raw tile row range overflow")?;
        if end_row > rows {
            bail!(
                "Flash-MoE dense tensor {} raw tile rows {}..{} exceed row count {}",
                entry.name,
                start_row,
                end_row,
                rows
            );
        }
        let row_bytes = cols
            .checked_mul(element_size)
            .context("dense tensor raw tile row byte length overflow")?;
        let byte_offset = start_row
            .checked_mul(row_bytes)
            .context("dense tensor raw tile byte offset overflow")?;
        let byte_len = row_count
            .checked_mul(row_bytes)
            .context("dense tensor raw tile byte length overflow")?;
        let (bytes, read_range) =
            self.read_range_profiled(entry.byte_offset + byte_offset as u64, byte_len)?;
        timing.read_range += read_range;
        timing.bytes_read = timing.bytes_read.saturating_add(byte_len as u64);
        timing.total = started.elapsed();
        Ok((bytes, entry.dtype.clone(), timing))
    }

    #[cfg(test)]
    pub(in crate::inference::flashmoe) fn read_tensor_rows_f32(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        let (tensor, _) =
            self.read_tensor_rows_f32_profiled(canonical_name, start_row, row_count)?;
        Ok(tensor)
    }

    pub(in crate::inference::flashmoe) fn read_tensor_rows_f32_profiled(
        &self,
        canonical_name: &str,
        start_row: usize,
        row_count: usize,
    ) -> Result<(Vec<f32>, DenseTileReadTiming)> {
        let started = Instant::now();
        let mut timing = DenseTileReadTiming::default();
        let Some(entry) = self.registry.tensor(canonical_name) else {
            bail!("Flash-MoE dense tensor registry is missing {canonical_name}");
        };
        if entry.quantization != TensorQuantization::None {
            bail!("dense q4 tensor {canonical_name} cannot be decoded as f32 rows");
        }
        let Some(element_size) = dtype_size(&entry.dtype) else {
            bail!(
                "Flash-MoE dense tensor {} has unsupported dtype {}",
                entry.name,
                entry.dtype
            );
        };
        let cols = entry.shape.last().copied().unwrap_or(0);
        if entry.shape.is_empty() || cols == 0 || row_count == 0 {
            return Ok((Vec::new(), timing));
        }
        let rows = entry
            .shape
            .iter()
            .take(entry.shape.len() - 1)
            .product::<usize>()
            .max(1);
        let end_row = start_row
            .checked_add(row_count)
            .context("dense tensor tile row range overflow")?;
        if end_row > rows {
            bail!(
                "Flash-MoE dense tensor {} tile rows {}..{} exceed row count {}",
                entry.name,
                start_row,
                end_row,
                rows
            );
        }
        let row_bytes = cols
            .checked_mul(element_size)
            .context("dense tensor tile row byte length overflow")?;
        let byte_offset = start_row
            .checked_mul(row_bytes)
            .context("dense tensor tile byte offset overflow")?;
        let byte_len = row_count
            .checked_mul(row_bytes)
            .context("dense tensor tile byte length overflow")?;
        let (bytes, read_range) =
            self.read_range_profiled(entry.byte_offset + byte_offset as u64, byte_len)?;
        timing.read_range += read_range;
        timing.bytes_read = timing.bytes_read.saturating_add(byte_len as u64);
        #[cfg(test)]
        self.decoded_tensor_tiles
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let decode_started = Instant::now();
        let tensor = decode_dense_tensor_f32(&entry.dtype, &bytes)?;
        timing.decode += decode_started.elapsed();
        timing.decoded_bytes = timing
            .decoded_bytes
            .saturating_add((tensor.len() * std::mem::size_of::<f32>()) as u64);
        timing.total = started.elapsed();
        Ok((tensor, timing))
    }

    pub(in crate::inference::flashmoe) fn read_tensor_row_f32(
        &self,
        canonical_name: &str,
        row: usize,
        requested_cols: usize,
    ) -> Result<Option<Vec<f32>>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            return Ok(None);
        };
        if let TensorQuantization::Q4 { group_size, .. } = entry.quantization {
            let cols = entry.shape.last().copied().unwrap_or(0);
            if entry.shape.is_empty() || requested_cols == 0 || cols == 0 {
                return Ok(None);
            }
            let rows = entry
                .shape
                .iter()
                .take(entry.shape.len() - 1)
                .product::<usize>()
                .max(1);
            if row >= rows {
                return Ok(None);
            }
            let (packed, scales, biases, _) = self.read_dense_q4_rows(entry, row, 1, group_size)?;
            let mut decoded =
                q4_dequantize_rows_with_group_size(&packed, &scales, &biases, 1, cols, group_size)?;
            decoded.truncate(requested_cols.min(cols));
            return Ok(Some(decoded));
        }
        let Some(element_size) = dtype_size(&entry.dtype) else {
            bail!(
                "Flash-MoE dense tensor {} has unsupported dtype {}",
                entry.name,
                entry.dtype
            );
        };
        if entry.shape.is_empty() || requested_cols == 0 {
            return Ok(None);
        }
        let cols = entry.shape.last().copied().unwrap_or(0);
        if cols == 0 {
            return Ok(None);
        }
        let rows = entry
            .shape
            .iter()
            .take(entry.shape.len() - 1)
            .product::<usize>()
            .max(1);
        if row >= rows {
            return Ok(None);
        }
        let used_cols = requested_cols.min(cols);
        let row_offset = row
            .checked_mul(cols)
            .and_then(|items| items.checked_mul(element_size))
            .context("dense tensor row offset overflow")? as u64;
        let byte_len = used_cols
            .checked_mul(element_size)
            .context("dense tensor row byte length overflow")?;
        let bytes = self.read_range(entry.byte_offset + row_offset, byte_len)?;
        Ok(Some(decode_dense_tensor_f32(&entry.dtype, &bytes)?))
    }

    pub(in crate::inference::flashmoe) fn read_range(
        &self,
        offset: u64,
        byte_len: usize,
    ) -> Result<Vec<u8>> {
        let (bytes, _) = self.read_range_profiled(offset, byte_len)?;
        Ok(bytes)
    }

    pub(in crate::inference::flashmoe) fn read_range_profiled(
        &self,
        offset: u64,
        byte_len: usize,
    ) -> Result<(Vec<u8>, Duration)> {
        if offset.saturating_add(byte_len as u64) > self.len {
            bail!(
                "dense tensor read {}..{} exceeds store length {}",
                offset,
                offset.saturating_add(byte_len as u64),
                self.len
            );
        }
        let started = Instant::now();
        let bytes = self.mmap[offset as usize..offset as usize + byte_len].to_vec();
        Ok((bytes, started.elapsed()))
    }

    #[cfg(test)]
    pub(in crate::inference::flashmoe) fn tensor_seed(
        &self,
        canonical_name: &str,
        fallback: u64,
    ) -> u64 {
        if let Some(tensor) = self.registry.tensor(canonical_name) {
            stable_hash(&tensor.name)
                ^ stable_hash(&tensor.dtype)
                ^ tensor.byte_offset
                ^ tensor.byte_len.rotate_left(7)
                ^ ((tensor.shape.iter().copied().product::<usize>() as u64) << 11)
        } else {
            tracing::trace!(
                tensor = canonical_name,
                manifest = %self.manifest_path.display(),
                "Flash-MoE tensor registry missing canonical tensor; using deterministic fallback seed"
            );
            fallback
        }
    }

    pub(in crate::inference::flashmoe) fn read_u64(&self, offset_hint: u64) -> Result<u64> {
        if self.len == 0 {
            return Ok(offset_hint.rotate_left(13) ^ 0x9e37_79b9_7f4a_7c15);
        }
        let offset = offset_hint % self.len;
        let mut out = [0u8; 8];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.mmap[((offset as usize) + i) % self.mmap.len()];
        }
        Ok(u64::from_le_bytes(out) ^ offset_hint.rotate_left(7))
    }

    /// Read a full 1-D or 2-D F32/BF16 tensor into a `Vec<f32>`.
    ///
    /// Returns `Ok(None)` when the tensor name is absent from the manifest.
    pub(in crate::inference::flashmoe) fn read_full_tensor_f32(
        &self,
        canonical_name: &str,
    ) -> Result<Option<Vec<f32>>> {
        let Some(entry) = self.registry.tensor(canonical_name) else {
            return Ok(None);
        };
        if entry.quantization != TensorQuantization::None {
            bail!("dense q4 tensor {canonical_name} cannot be read as a full f32 tensor");
        }
        let Some(_element_size) = dtype_size(&entry.dtype) else {
            bail!(
                "Flash-MoE dense tensor {} has unsupported dtype {}",
                entry.name,
                entry.dtype
            );
        };
        let byte_len = entry.byte_len as usize;
        let bytes = self.read_range(entry.byte_offset, byte_len)?;
        Ok(Some(decode_dense_tensor_f32(&entry.dtype, &bytes)?))
    }

    #[cfg(test)]
    pub(in crate::inference::flashmoe) fn read_full_tensor_f32_cached(
        &self,
        canonical_name: &str,
    ) -> Result<Option<Arc<Vec<f32>>>> {
        self.dense_tensor_f32(canonical_name)
    }

    #[cfg(test)]
    pub(in crate::inference::flashmoe) fn decoded_full_tensor_count(&self) -> usize {
        self.decoded_full_tensors
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(in crate::inference::flashmoe) fn decoded_tensor_tile_count(&self) -> usize {
        self.decoded_tensor_tiles
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

pub(in crate::inference::flashmoe) fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype.to_ascii_uppercase().as_str() {
        "F32" | "FLOAT32" | "FP32" => Some(4),
        "BF16" | "BFLOAT16" | "F16" | "FLOAT16" | "FP16" => Some(2),
        "U8" | "I8" => Some(1),
        _ => None,
    }
}

pub(in crate::inference::flashmoe) fn decode_dense_tensor_f32(
    dtype: &str,
    bytes: &[u8],
) -> Result<Vec<f32>> {
    match dtype.to_ascii_uppercase().as_str() {
        "F32" | "FLOAT32" | "FP32" => {
            if !bytes.len().is_multiple_of(4) {
                bail!(
                    "F32 tensor byte length {} is not divisible by 4",
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        "BF16" | "BFLOAT16" => {
            if !bytes.len().is_multiple_of(2) {
                bail!(
                    "BF16 tensor byte length {} is not divisible by 2",
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let hi = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                    f32::from_bits(hi << 16)
                })
                .collect())
        }
        "F16" | "FLOAT16" | "FP16" => {
            if !bytes.len().is_multiple_of(2) {
                bail!(
                    "F16 tensor byte length {} is not divisible by 2",
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect())
        }
        "U8" => Ok(bytes.iter().map(|value| *value as f32).collect()),
        "I8" => Ok(bytes.iter().map(|value| (*value as i8) as f32).collect()),
        other => bail!("unsupported dense tensor dtype {other}"),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = (bits & 0x03ff) as u32;
    let value = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut frac = frac;
                let mut exp = -14i32;
                while (frac & 0x0400) == 0 {
                    frac <<= 1;
                    exp -= 1;
                }
                frac &= 0x03ff;
                sign | (((exp + 127) as u32) << 23) | (frac << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13),
    };
    f32::from_bits(value)
}
