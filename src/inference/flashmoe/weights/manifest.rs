use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentDenseLayout {
    Q4,
    Bf16,
    F16,
    F32,
}

impl ResidentDenseLayout {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Q4 => "resident Q4",
            Self::Bf16 => "resident BF16",
            Self::F16 => "resident F16",
            Self::F32 => "resident F32",
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FlashMoeManifest {
    pub model: String,
    pub cache_version: String,
    pub dense_shards: Vec<String>,
    pub expert_tensors: Vec<ExpertTensorRef>,
    pub dense_tensors: Vec<DenseTensorRef>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpertTensorRef {
    pub tensor: String,
    pub shard: String,
    pub layer: Option<usize>,
    pub expert: Option<usize>,
    pub dtype: Option<String>,
    pub shape: Vec<usize>,
    pub source_offsets: Option<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q4_sources: Option<DenseQ4SourceRefs>,
}

impl AggregateExpertTensor for ExpertTensorRef {
    fn aggregate_tensor_name(&self) -> &str {
        &self.tensor
    }

    fn aggregate_tensor_shape(&self) -> &[usize] {
        &self.shape
    }

    fn aggregate_tensor_has_native_q4(&self) -> bool {
        self.q4_sources.is_some()
    }

    fn aggregate_tensor_is_mxfp4(&self) -> bool {
        self.q4_sources
            .as_ref()
            .is_some_and(|source| source.source_format == DenseQ4SourceFormat::MlxMxfp4)
    }
}

impl ExpertSourceTensor for ExpertTensorRef {
    fn expert_source_offsets(&self) -> Option<[u64; 2]> {
        self.source_offsets
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DenseTensorRef {
    pub tensor: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub source_offsets: [u64; 2],
    pub runtime_offset: u64,
    pub byte_len: u64,
    #[serde(default)]
    pub quantization: TensorQuantization,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q4_sources: Option<DenseQ4SourceRefs>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ4SourceFormat {
    MlxAffine,
    MlxAffine8,
    ColibriInt4,
    ColibriInt8,
    MlxMxfp4,
}

impl Default for DenseQ4SourceFormat {
    fn default() -> Self {
        Self::MlxAffine
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DenseQ4SourceRefs {
    pub scales_shard: String,
    pub scales_offsets: [u64; 2],
    pub biases_shard: String,
    pub biases_offsets: [u64; 2],
    pub scale_bias_dtype: String,
    #[serde(default)]
    pub source_format: DenseQ4SourceFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_group_size: Option<usize>,
    /// Optional cache-build row permutation for a combined source projection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_row_order: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TensorQuantization {
    None,
    Q4 {
        group_size: usize,
        format: String,
        #[serde(default = "default_dense_q4_scale_bias_dtype")]
        scale_bias_dtype: String,
    },
    /// Source GGUF blocks are preserved verbatim in the canonical resident
    /// store. The graph binder resolves the matching Metal implementation at
    /// load time from these typed fields; inference never probes encodings.
    Gguf {
        tensor_type: u32,
        block_elements: u64,
        block_bytes: u64,
    },
}

impl Default for TensorQuantization {
    fn default() -> Self {
        Self::None
    }
}

fn default_dense_q4_scale_bias_dtype() -> String {
    "F32".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeTensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub alignment: u64,
    pub quantization: TensorQuantization,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorRegistry {
    pub(super) tensors: BTreeMap<String, RuntimeTensorEntry>,
}

impl TensorRegistry {
    pub fn load(manifest_path: &Path) -> Result<Self> {
        let manifest: FlashMoeManifest =
            serde_json::from_slice(&fs::read(manifest_path).with_context(|| {
                format!(
                    "failed to read Flash-MoE tensor manifest {}",
                    manifest_path.display()
                )
            })?)
            .with_context(|| {
                format!(
                    "failed to parse Flash-MoE tensor manifest {}",
                    manifest_path.display()
                )
            })?;
        Ok(Self::from_manifest(&manifest))
    }

    pub(crate) fn from_manifest(manifest: &FlashMoeManifest) -> Self {
        let mut tensors = BTreeMap::new();
        for tensor in &manifest.dense_tensors {
            insert_tensor_entry_with_aliases(
                &mut tensors,
                &tensor.tensor,
                RuntimeTensorEntry {
                    name: tensor.tensor.clone(),
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    byte_offset: tensor.runtime_offset,
                    byte_len: tensor.byte_len,
                    alignment: TENSOR_ALIGNMENT,
                    quantization: tensor.quantization.clone(),
                },
            );
        }
        for tensor in &manifest.expert_tensors {
            if let Some([start, end]) = tensor.source_offsets {
                insert_tensor_entry_with_aliases(
                    &mut tensors,
                    &tensor.tensor,
                    RuntimeTensorEntry {
                        name: tensor.tensor.clone(),
                        dtype: tensor
                            .dtype
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string()),
                        shape: tensor.shape.clone(),
                        byte_offset: start,
                        byte_len: end.saturating_sub(start),
                        alignment: TENSOR_ALIGNMENT,
                        quantization: TensorQuantization::Q4 {
                            group_size: GROUP_SIZE,
                            format: ExpertQuantization::FourBitProduction.as_str().to_string(),
                            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                        },
                    },
                );
            }
        }
        Self { tensors }
    }

    pub fn tensor(&self, canonical_name: &str) -> Option<&RuntimeTensorEntry> {
        self.tensors.get(canonical_name)
    }

    pub(crate) fn has_tensor_with_prefix(&self, prefix: &str) -> bool {
        self.tensors.keys().any(|name| name.starts_with(prefix))
    }

    pub fn require(&self, canonical_name: &str) -> Result<&RuntimeTensorEntry> {
        self.tensor(canonical_name)
            .with_context(|| format!("Flash-MoE tensor registry is missing {canonical_name}"))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub(crate) fn resolve_resident_dense_layout(&self) -> Result<ResidentDenseLayout> {
        let matrix_tensors = self
            .tensors
            .values()
            .filter(|tensor| tensor.shape.len() >= 2)
            .filter(|tensor| !is_routed_expert_tensor_name(&tensor.name));
        let mut found_matrix = false;
        let mut found_q4 = false;
        let mut unquantized_layout: Option<ResidentDenseLayout> = None;

        for tensor in matrix_tensors {
            found_matrix = true;
            match tensor.quantization {
                TensorQuantization::Q4 { .. } => found_q4 = true,
                TensorQuantization::Gguf { .. } => bail!(
                    "FlashMoe generic dense layout resolver cannot bind native GGUF tensor {}; a model-family graph binder is required",
                    tensor.name
                ),
                TensorQuantization::None => {
                    let layout =
                        resident_dense_layout_for_dtype(&tensor.dtype).with_context(|| {
                            format!(
                                "FlashMoe dense tensor {} has unsupported resident dtype {}",
                                tensor.name, tensor.dtype
                            )
                        })?;
                    if let Some(existing) = unquantized_layout
                        && existing != layout
                    {
                        bail!(
                            "FlashMoe dense manifest mixes resident matrix layouts {} and {}",
                            existing.as_str(),
                            layout.as_str()
                        );
                    }
                    unquantized_layout = Some(layout);
                }
            }
        }

        if !found_matrix {
            bail!("FlashMoe dense manifest contains no matrix tensors");
        }
        if found_q4 {
            return Ok(ResidentDenseLayout::Q4);
        }
        unquantized_layout
            .context("FlashMoe dense manifest has no resolvable resident matrix layout")
    }
}

fn resident_dense_layout_for_dtype(dtype: &str) -> Option<ResidentDenseLayout> {
    match dtype.to_ascii_uppercase().as_str() {
        "BF16" | "BFLOAT16" => Some(ResidentDenseLayout::Bf16),
        "F16" | "FLOAT16" | "FP16" => Some(ResidentDenseLayout::F16),
        "F32" | "FLOAT32" | "FP32" => Some(ResidentDenseLayout::F32),
        _ => None,
    }
}

fn is_routed_expert_tensor_name(name: &str) -> bool {
    name.contains(".mlp.experts.") || name.contains(".switch_mlp.")
}

fn insert_tensor_entry_with_aliases(
    tensors: &mut BTreeMap<String, RuntimeTensorEntry>,
    name: &str,
    entry: RuntimeTensorEntry,
) {
    tensors
        .entry(name.to_string())
        .or_insert_with(|| entry.clone());
    let canonical_name = canonical_hf_tensor_name(name);
    if canonical_name != name {
        tensors.entry(canonical_name).or_insert(entry);
    }
}

pub(crate) fn canonical_hf_tensor_name(name: &str) -> String {
    let canonical = if let Some(rest) = name.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else if let Some(rest) = name.strip_prefix("language_model.") {
        rest.to_string()
    } else if let Some(rest) = name.strip_prefix("model.visual.") {
        format!("visual.{rest}")
    } else if let Some(rest) = name.strip_prefix("vision_tower.") {
        format!("visual.{rest}")
    } else {
        name.to_string()
    };
    if canonical.starts_with("visual.") {
        canonical
            .replace(".mlp.linear_fc1.", ".mlp.fc1.")
            .replace(".mlp.linear_fc2.", ".mlp.fc2.")
    } else {
        canonical.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseQ4Layout {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) group_size: usize,
    pub(crate) row_packed_bytes: usize,
    pub(crate) groups_per_row: usize,
    pub(crate) packed_bytes: usize,
    pub(crate) scales_bytes: usize,
    pub(crate) scale_bias_bytes: usize,
    pub(crate) total_bytes: usize,
}

#[cfg(test)]
pub(crate) fn dense_q4_layout(shape: &[usize], group_size: usize) -> Result<DenseQ4Layout> {
    dense_q4_layout_with_scale_bias_dtype(shape, group_size, EXPERT_SCALE_BIAS_DTYPE_F32)
}

pub(crate) fn dense_q4_layout_with_scale_bias_dtype(
    shape: &[usize],
    group_size: usize,
    scale_bias_dtype: &str,
) -> Result<DenseQ4Layout> {
    if group_size == 0 {
        bail!("dense q4 group_size must be positive");
    }
    let cols = shape.last().copied().unwrap_or(0);
    if shape.len() < 2 || cols == 0 {
        bail!(
            "dense q4 tensor shape {:?} is not a non-empty matrix",
            shape
        );
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .context("dense q4 tensor row count overflow")
        })?;
    let row_packed_bytes = cols.div_ceil(2);
    let groups_per_row = cols.div_ceil(group_size);
    let packed_bytes = rows
        .checked_mul(row_packed_bytes)
        .context("dense q4 packed byte length overflow")?;
    let groups = rows
        .checked_mul(groups_per_row)
        .context("dense q4 group count overflow")?;
    let scale_bias_bytes = expert_scale_bias_dtype_size(scale_bias_dtype)
        .with_context(|| format!("unsupported dense q4 scale/bias dtype {scale_bias_dtype}"))?;
    let scales_bytes = groups
        .checked_mul(scale_bias_bytes)
        .context("dense q4 scale byte length overflow")?;
    let total_bytes = packed_bytes
        .checked_add(scales_bytes)
        .and_then(|value| value.checked_add(scales_bytes))
        .context("dense q4 total byte length overflow")?;
    Ok(DenseQ4Layout {
        rows,
        cols,
        group_size,
        row_packed_bytes,
        groups_per_row,
        packed_bytes,
        scales_bytes,
        scale_bias_bytes,
        total_bytes,
    })
}

pub(crate) fn validate_dense_matvec_shape(
    entry: &RuntimeTensorEntry,
    canonical_name: &str,
    expected_rows: usize,
    input_len: usize,
) -> Result<(usize, usize)> {
    let expected_shape = [expected_rows, input_len];
    match entry.shape.as_slice() {
        [rows, cols] if *rows == expected_rows && *cols == input_len => Ok((*rows, *cols)),
        _ => bail!(
            "Flash-MoE dense tensor {canonical_name} shape mismatch: expected shape {:?}, actual shape {:?}, input length {input_len}",
            expected_shape,
            entry.shape
        ),
    }
}
