use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use super::experts::{
    AggregateExpertTensor, EXPERT_SCALE_BIAS_DTYPE_F32, ExpertSourceTensor,
    expert_scale_bias_dtype_size,
};
use super::state::{FlashMoeRoutingOutputSource, FlashMoeRoutingOutputState};
use super::types::{ExpertQuantization, GROUP_SIZE};
use anyhow::{Context, Result, bail};

pub(crate) const TENSOR_ALIGNMENT: u64 = 4096;

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
pub struct DenseQ4SourceRefs {
    pub scales_shard: String,
    pub scales_offsets: [u64; 2],
    pub biases_shard: String,
    pub biases_offsets: [u64; 2],
    pub scale_bias_dtype: String,
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
    tensors: BTreeMap<String, RuntimeTensorEntry>,
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
    if let Some(rest) = name.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else if let Some(rest) = name.strip_prefix("language_model.") {
        rest.to_string()
    } else if let Some(rest) = name.strip_prefix("model.visual.") {
        format!("visual.{rest}")
    } else {
        name.to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseQ4Layout {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RouterScoreProjectionBinding {
    ResidentDense(DenseMmapMatvecProjection),
    ResidentQ4(DenseQ4MmapMatvecProjection),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RouterScoreProjectionDescriptor {
    pub(crate) layer: usize,
    pub(crate) tensor_name: String,
    pub(crate) experts: usize,
    pub(crate) hidden_width: usize,
    pub(crate) binding: RouterScoreProjectionBinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterScoreProjectionExecutionKind {
    ResidentDense,
    ResidentQ4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RouterScoreProjectionExecution<'a> {
    pub(crate) layer: usize,
    pub(crate) tensor_name: &'a str,
    pub(crate) experts: usize,
    pub(crate) hidden_width: usize,
    pub(crate) kind: RouterScoreProjectionExecutionKind,
}

impl RouterScoreProjectionDescriptor {
    pub(crate) fn from_entry(
        layer: usize,
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        experts: usize,
        hidden_width: usize,
    ) -> Result<Self> {
        match &entry.quantization {
            TensorQuantization::None => {
                let Some(element_size) = dense_dtype_size(&entry.dtype) else {
                    bail!(
                        "Flash-MoE router tensor {} has unsupported dtype {}",
                        tensor_name,
                        entry.dtype
                    );
                };
                let projection = DenseMmapMatvecProjection::from_entry(
                    tensor_name,
                    entry,
                    store_len,
                    experts,
                    hidden_width,
                    element_size,
                )?;
                Ok(Self {
                    layer,
                    tensor_name: tensor_name.to_string(),
                    experts,
                    hidden_width,
                    binding: RouterScoreProjectionBinding::ResidentDense(projection),
                })
            }
            TensorQuantization::Q4 { .. } => {
                let Some(projection) = DenseQ4MmapMatvecProjection::from_entry(
                    tensor_name,
                    entry,
                    store_len,
                    experts,
                    hidden_width,
                )?
                else {
                    bail!(
                        "Flash-MoE router tensor {tensor_name} cannot resolve a resident Q4 projection descriptor for shape [{experts}, {hidden_width}]"
                    );
                };
                Ok(Self {
                    layer,
                    tensor_name: tensor_name.to_string(),
                    experts,
                    hidden_width,
                    binding: RouterScoreProjectionBinding::ResidentQ4(projection),
                })
            }
        }
    }

    pub(crate) fn execution(
        &self,
        layer: usize,
        experts: usize,
        hidden_width: usize,
    ) -> Result<RouterScoreProjectionExecution<'_>> {
        if self.layer != layer {
            bail!(
                "Flash-MoE router score projection execution layer {} does not match scheduled layer {}",
                self.layer,
                layer
            );
        }
        if self.experts != experts {
            bail!(
                "Flash-MoE router score projection execution experts {} does not match scheduled experts {}",
                self.experts,
                experts
            );
        }
        if self.hidden_width != hidden_width {
            bail!(
                "Flash-MoE router score projection execution hidden width {} does not match scheduled hidden width {}",
                self.hidden_width,
                hidden_width
            );
        }
        let kind = match self.binding {
            RouterScoreProjectionBinding::ResidentDense(_) => {
                RouterScoreProjectionExecutionKind::ResidentDense
            }
            RouterScoreProjectionBinding::ResidentQ4(_) => {
                RouterScoreProjectionExecutionKind::ResidentQ4
            }
        };
        Ok(RouterScoreProjectionExecution {
            layer: self.layer,
            tensor_name: &self.tensor_name,
            experts: self.experts,
            hidden_width: self.hidden_width,
            kind,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RouterScoreBatch {
    state: FlashMoeRoutingOutputState,
    pub(crate) projection: Option<RouterScoreProjectionDescriptor>,
    pub(crate) scores: Vec<f32>,
}

impl RouterScoreBatch {
    pub(crate) fn new(
        state: FlashMoeRoutingOutputState,
        projection: Option<RouterScoreProjectionDescriptor>,
        scores: Vec<f32>,
    ) -> Result<Self> {
        if !state.is_declared_graph_state() {
            bail!("FlashMoe router score batch is not declared graph state");
        }
        if state.source() != FlashMoeRoutingOutputSource::CpuRouterScores {
            bail!(
                "FlashMoe router score batch source {:?} is not CPU router scores",
                state.source()
            );
        }
        if scores.len() != state.experts() {
            bail!(
                "FlashMoe router score batch has {} scores for {} declared experts",
                scores.len(),
                state.experts()
            );
        }
        if let Some(projection) = projection.as_ref() {
            if projection.layer != state.layer() {
                bail!(
                    "FlashMoe router score batch layer {} does not match projection layer {}",
                    state.layer(),
                    projection.layer
                );
            }
            if projection.experts != state.experts() {
                bail!(
                    "FlashMoe router score batch expert count {} does not match projection experts {}",
                    state.experts(),
                    projection.experts
                );
            }
        }
        Ok(Self {
            state,
            projection,
            scores,
        })
    }

    pub(crate) fn state(&self) -> FlashMoeRoutingOutputState {
        self.state
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuVisibleNextNormWeights<'a> {
    pub(crate) tensor_name: &'a str,
    pub(crate) width: usize,
    values: &'a [f32],
}

impl<'a> CpuVisibleNextNormWeights<'a> {
    pub(crate) fn new(tensor_name: &'a str, values: &'a [f32], width: usize) -> Result<Self> {
        if tensor_name.is_empty() {
            bail!("FlashMoe scheduled next-norm weights require a tensor name");
        }
        if width == 0 {
            bail!("FlashMoe scheduled next-norm weights require non-zero width");
        }
        if values.len() < width {
            bail!(
                "FlashMoe scheduled next-norm weight tensor {tensor_name} length {} is smaller than width {width}",
                values.len()
            );
        }
        Ok(Self {
            tensor_name,
            width,
            values,
        })
    }

    pub(crate) fn values(self) -> &'a [f32] {
        &self.values[..self.width]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScheduledNextNormWeights<'a> {
    None,
    CpuVisible(CpuVisibleNextNormWeights<'a>),
}

impl<'a> ScheduledNextNormWeights<'a> {
    pub(crate) fn none() -> Self {
        Self::None
    }

    pub(crate) fn cpu_visible(
        tensor_name: &'a str,
        values: &'a [f32],
        width: usize,
    ) -> Result<Self> {
        Ok(Self::CpuVisible(CpuVisibleNextNormWeights::new(
            tensor_name,
            values,
            width,
        )?))
    }

    pub(crate) fn is_none(self) -> bool {
        matches!(self, Self::None)
    }

    pub(crate) fn is_cpu_visible(self) -> bool {
        matches!(self, Self::CpuVisible(_))
    }

    pub(crate) fn width(self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::CpuVisible(weights) => Some(weights.width),
        }
    }

    pub(crate) fn values(self) -> Option<&'a [f32]> {
        match self {
            Self::None => None,
            Self::CpuVisible(weights) => Some(weights.values()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedScheduledNextNormWeights {
    tensor_name: Option<String>,
    values: Option<Vec<f32>>,
    width: usize,
}

impl PreparedScheduledNextNormWeights {
    pub(crate) fn none() -> Self {
        Self {
            tensor_name: None,
            values: None,
            width: 0,
        }
    }

    pub(crate) fn cpu_visible(tensor_name: String, values: Vec<f32>, width: usize) -> Result<Self> {
        CpuVisibleNextNormWeights::new(&tensor_name, &values, width)?;
        Ok(Self {
            tensor_name: Some(tensor_name),
            values: Some(values),
            width,
        })
    }

    pub(crate) fn scheduled(&self) -> Result<ScheduledNextNormWeights<'_>> {
        match (self.tensor_name.as_deref(), self.values.as_deref()) {
            (Some(tensor_name), Some(values)) => {
                ScheduledNextNormWeights::cpu_visible(tensor_name, values, self.width)
            }
            (None, None) => Ok(ScheduledNextNormWeights::none()),
            _ => bail!("FlashMoe scheduled next-norm weights have incomplete descriptor state"),
        }
    }
}

pub(crate) fn layer_norm_tensor_name(layer: usize, name: &str) -> String {
    format!("model.layers.{layer}.{name}.weight")
}

pub(crate) fn router_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.gate.weight")
}

pub(crate) fn shared_expert_tensor_name(layer: usize, projection: &str) -> String {
    format!("model.layers.{layer}.mlp.shared_expert.{projection}.weight")
}

pub(crate) fn shared_expert_gate_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.shared_expert_gate.weight")
}

pub(crate) fn prepare_scheduled_next_norm_weights<F>(
    layer: usize,
    total_layers: usize,
    width: usize,
    needs_next_layer_norm: bool,
    mut lookup: F,
) -> Result<PreparedScheduledNextNormWeights>
where
    F: FnMut(&str, usize) -> Result<Option<Vec<f32>>>,
{
    if !needs_next_layer_norm || layer + 1 >= total_layers {
        return Ok(PreparedScheduledNextNormWeights::none());
    }

    let name = layer_norm_tensor_name(layer + 1, "input_layernorm");
    let values = lookup(&name, width)?.with_context(|| {
        format!(
            "FlashMoe unsupported scheduled CMD3 path: missing next-layer norm weight {name} for layer {layer}"
        )
    })?;
    PreparedScheduledNextNormWeights::cpu_visible(name, values, width)
}

pub(crate) fn qwen3next_norm_uses_offset(norm_offsets_enabled: bool, canonical_name: &str) -> bool {
    norm_offsets_enabled
        && (canonical_name == "model.norm.weight"
            || canonical_name.contains(".input_layernorm.weight")
            || canonical_name.contains(".post_attention_layernorm.weight")
            || canonical_name.contains(".self_attn.q_norm.weight")
            || canonical_name.contains(".self_attn.k_norm.weight"))
}

pub(crate) fn qwen3next_norm_weight_needs_offset(weight: &[f32]) -> bool {
    if weight.is_empty() {
        return false;
    }

    let mut sum = 0.0f32;
    let mut has_negative = false;
    for value in weight {
        sum += *value;
        has_negative |= *value < 0.0;
    }

    has_negative || sum / (weight.len() as f32) < 0.75
}

pub(crate) fn apply_qwen3next_norm_offset_if_needed(
    norm_offsets_enabled: bool,
    canonical_name: &str,
    weight: &mut [f32],
) {
    if qwen3next_norm_uses_offset(norm_offsets_enabled, canonical_name)
        && qwen3next_norm_weight_needs_offset(weight)
    {
        for value in weight {
            *value += 1.0;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResidentStaticTensorRef {
    pub(crate) tensor_name: String,
    pub(crate) byte_offset: u64,
    pub(crate) dtype: String,
    pub(crate) values: usize,
    pub(crate) element_size: usize,
}

impl ResidentStaticTensorRef {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        expected_values: usize,
        allowed_dtypes: &[&str],
    ) -> Result<Option<Self>> {
        if entry.quantization != TensorQuantization::None {
            return Ok(None);
        }
        if !allowed_dtypes
            .iter()
            .any(|allowed| entry.dtype.eq_ignore_ascii_case(allowed))
        {
            return Ok(None);
        }
        let Some(element_size) = dense_dtype_size(&entry.dtype) else {
            return Ok(None);
        };
        let expected_bytes = expected_values
            .checked_mul(element_size)
            .context("resident static tensor byte length overflow")?;
        if entry.byte_len as usize != expected_bytes {
            return Ok(None);
        }
        if entry
            .byte_offset
            .checked_add(entry.byte_len)
            .map_or(true, |end| end > store_len)
        {
            return Ok(None);
        }
        if entry.byte_offset % element_size as u64 != 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            tensor_name: tensor_name.to_string(),
            byte_offset: entry.byte_offset,
            dtype: entry.dtype.clone(),
            values: expected_values,
            element_size,
        }))
    }
}

fn dense_dtype_size(dtype: &str) -> Option<usize> {
    match dtype.to_ascii_uppercase().as_str() {
        "BF16" | "BFLOAT16" | "F16" | "FLOAT16" | "FP16" => Some(2),
        "F32" | "FLOAT32" | "FP32" => Some(4),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseMmapMatvecProjection {
    pub(crate) tensor_name: String,
    pub(crate) byte_offset: u64,
    pub(crate) dtype: String,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) output_width: usize,
}

impl DenseMmapMatvecProjection {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        output_width: usize,
        input_len: usize,
        element_size: usize,
    ) -> Result<Self> {
        let (rows, cols) =
            validate_dense_matvec_shape(entry, tensor_name, output_width, input_len)?;
        let row_bytes = cols
            .checked_mul(element_size)
            .context("dense tensor resident row byte length overflow")?;
        let byte_len = rows
            .checked_mul(row_bytes)
            .context("dense tensor resident byte length overflow")?;
        if entry
            .byte_offset
            .checked_add(byte_len as u64)
            .map_or(true, |end| end > store_len)
        {
            bail!(
                "Flash-MoE dense tensor {} byte range {}..{} exceeds dense store length {}",
                tensor_name,
                entry.byte_offset,
                entry.byte_offset.saturating_add(byte_len as u64),
                store_len
            );
        }
        Ok(Self {
            tensor_name: tensor_name.to_string(),
            byte_offset: entry.byte_offset,
            dtype: entry.dtype.clone(),
            rows,
            cols,
            output_width,
        })
    }

    pub(crate) fn stride(&self) -> usize {
        self.cols
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseQ4MmapMatvecProjection {
    pub(crate) tensor_name: String,
    pub(crate) packed_byte_offset: u64,
    pub(crate) scales_byte_offset: u64,
    pub(crate) biases_byte_offset: u64,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) output_width: usize,
    pub(crate) row_packed_bytes: usize,
    pub(crate) groups_per_row: usize,
    pub(crate) group_size: usize,
    pub(crate) scale_bias_dtype: String,
}

impl DenseQ4MmapMatvecProjection {
    pub(crate) fn from_entry(
        tensor_name: &str,
        entry: &RuntimeTensorEntry,
        store_len: u64,
        output_width: usize,
        input_len: usize,
    ) -> Result<Option<Self>> {
        let TensorQuantization::Q4 {
            group_size,
            scale_bias_dtype,
            ..
        } = &entry.quantization
        else {
            return Ok(None);
        };
        let (rows, cols) =
            validate_dense_matvec_shape(entry, tensor_name, output_width, input_len)?;
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&entry.shape, *group_size, scale_bias_dtype)?;
        if entry.byte_len as usize != layout.total_bytes
            || rows != layout.rows
            || cols != layout.cols
        {
            return Ok(None);
        }
        if entry
            .byte_offset
            .checked_add(entry.byte_len)
            .map_or(true, |end| end > store_len)
        {
            return Ok(None);
        }
        let packed_byte_offset = entry.byte_offset;
        let scales_byte_offset = entry
            .byte_offset
            .checked_add(layout.packed_bytes as u64)
            .context("dense q4 projection scales offset overflow")?;
        let biases_byte_offset = scales_byte_offset
            .checked_add(layout.scales_bytes as u64)
            .context("dense q4 projection biases offset overflow")?;
        Ok(Some(Self {
            tensor_name: tensor_name.to_string(),
            packed_byte_offset,
            scales_byte_offset,
            biases_byte_offset,
            rows,
            cols,
            output_width,
            row_packed_bytes: layout.row_packed_bytes,
            groups_per_row: layout.groups_per_row,
            group_size: *group_size,
            scale_bias_dtype: scale_bias_dtype.clone(),
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SharedExpertPhaseShape {
    pub(crate) width: usize,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) total_intermediate: usize,
}

impl SharedExpertPhaseShape {
    pub(crate) fn new(width: usize, shared_experts: usize, intermediate: usize) -> Result<Self> {
        if width == 0 || shared_experts == 0 || intermediate == 0 {
            bail!(
                "shared expert graph shape requires non-zero width, shared expert count, and intermediate width"
            );
        }
        let total_intermediate = shared_experts
            .checked_mul(intermediate)
            .context("shared expert intermediate width overflow")?;
        Ok(Self {
            width,
            shared_experts,
            intermediate,
            total_intermediate,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SharedExpertPhaseWeights {
    pub(crate) gate: Arc<Vec<f32>>,
    pub(crate) up: Arc<Vec<f32>>,
    pub(crate) down: Arc<Vec<f32>>,
    pub(crate) router: Arc<Vec<f32>>,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) width: usize,
}

impl SharedExpertPhaseWeights {
    pub(crate) fn new(
        gate: Arc<Vec<f32>>,
        up: Arc<Vec<f32>>,
        down: Arc<Vec<f32>>,
        router: Arc<Vec<f32>>,
        shared_experts: usize,
        intermediate: usize,
        width: usize,
    ) -> Result<Self> {
        let weights = Self {
            gate,
            up,
            down,
            router,
            shared_experts,
            intermediate,
            width,
        };
        weights.validated_shape()?;
        Ok(weights)
    }

    pub(crate) fn validated_shape(&self) -> Result<SharedExpertPhaseShape> {
        let shape =
            SharedExpertPhaseShape::new(self.width, self.shared_experts, self.intermediate)?;
        let dense_len = shape
            .total_intermediate
            .checked_mul(shape.width)
            .context("shared expert dense projection width overflow")?;
        let router_len = shape
            .shared_experts
            .checked_mul(shape.width)
            .context("shared expert router projection width overflow")?;
        if self.gate.len() != dense_len
            || self.up.len() != dense_len
            || self.down.len() != dense_len
            || self.router.len() != router_len
        {
            bail!(
                "FlashMoe scheduled shared dense expert shape is invalid: width={} shared_experts={} intermediate={} gate={} up={} down={} router={}",
                self.width,
                self.shared_experts,
                self.intermediate,
                self.gate.len(),
                self.up.len(),
                self.down.len(),
                self.router.len()
            );
        }
        Ok(shape)
    }
}

pub(crate) fn build_shared_expert_phase_weights<F>(
    layer: usize,
    width: usize,
    shared_experts: usize,
    intermediate: usize,
    mut lookup: F,
) -> Result<Option<SharedExpertPhaseWeights>>
where
    F: FnMut(&str) -> Result<Option<Arc<Vec<f32>>>>,
{
    if shared_experts == 0 || intermediate == 0 {
        return Ok(None);
    }
    SharedExpertPhaseShape::new(width, shared_experts, intermediate)?;

    let gate_name = shared_expert_tensor_name(layer, "gate_proj");
    let up_name = shared_expert_tensor_name(layer, "up_proj");
    let down_name = shared_expert_tensor_name(layer, "down_proj");
    let shared_gate_name = shared_expert_gate_tensor_name(layer);
    let gate = lookup(&gate_name)?
        .with_context(|| format!("missing configured shared expert tensor {gate_name}"))?;
    let up = lookup(&up_name)?
        .with_context(|| format!("missing configured shared expert tensor {up_name}"))?;
    let down = lookup(&down_name)?
        .with_context(|| format!("missing configured shared expert tensor {down_name}"))?;
    let router = lookup(&shared_gate_name)?.with_context(|| {
        format!("missing configured shared expert gate tensor {shared_gate_name}")
    })?;

    SharedExpertPhaseWeights::new(gate, up, down, router, shared_experts, intermediate, width)
        .map(Some)
}

#[derive(Debug, Clone)]
pub(crate) struct SharedExpertPhaseQ4Projections {
    pub(crate) gate: DenseQ4MmapMatvecProjection,
    pub(crate) up: DenseQ4MmapMatvecProjection,
    pub(crate) down: DenseQ4MmapMatvecProjection,
    pub(crate) router: DenseQ4MmapMatvecProjection,
    pub(crate) shared_experts: usize,
    pub(crate) intermediate: usize,
    pub(crate) width: usize,
}

impl SharedExpertPhaseQ4Projections {
    pub(crate) fn validated_shape(&self) -> Result<SharedExpertPhaseShape> {
        let shape =
            SharedExpertPhaseShape::new(self.width, self.shared_experts, self.intermediate)?;
        if self.gate.cols != shape.width
            || self.up.cols != shape.width
            || self.router.cols != shape.width
            || self.down.cols != shape.total_intermediate
            || self.gate.output_width != shape.total_intermediate
            || self.up.output_width != shape.total_intermediate
            || self.down.output_width != shape.width
            || self.router.output_width != shape.shared_experts
        {
            bail!(
                "FlashMoe scheduled shared Q4 expert shape is invalid: width={} shared_experts={} intermediate={} gate=({},{}) up=({},{}) down=({},{}) router=({},{})",
                self.width,
                self.shared_experts,
                self.intermediate,
                self.gate.output_width,
                self.gate.cols,
                self.up.output_width,
                self.up.cols,
                self.down.output_width,
                self.down.cols,
                self.router.output_width,
                self.router.cols
            );
        }
        Ok(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_quantization_defaults_to_unquantized_dense() {
        assert_eq!(TensorQuantization::default(), TensorQuantization::None);
    }

    #[test]
    fn tensor_quantization_q4_defaults_scale_bias_dtype_for_legacy_manifests() {
        let quantization: TensorQuantization =
            serde_json::from_str(r#"{"Q4":{"group_size":16,"format":"dense-q4"}}"#).unwrap();

        assert_eq!(
            quantization,
            TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "F32".to_string(),
            }
        );
    }

    #[test]
    fn dense_tensor_ref_preserves_runtime_binding_offsets() {
        let tensor = DenseTensorRef {
            tensor: "model.embed_tokens.weight".to_string(),
            shard: "model-00001.safetensors".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![8, 4],
            source_offsets: [128, 192],
            runtime_offset: 4096,
            byte_len: 64,
            quantization: TensorQuantization::None,
            q4_sources: None,
        };

        assert_eq!(tensor.runtime_offset, 4096);
        assert_eq!(tensor.byte_len, 64);
        assert_eq!(tensor.quantization, TensorQuantization::None);
    }

    #[test]
    fn tensor_registry_builds_dense_aliases_from_manifest() {
        let manifest = FlashMoeManifest {
            model: "fixture".to_string(),
            cache_version: "test".to_string(),
            dense_shards: Vec::new(),
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: "model.language_model.layers.7.self_attn.q_proj.weight".to_string(),
                shard: "model.safetensors".to_string(),
                dtype: "BF16".to_string(),
                shape: vec![4, 8],
                source_offsets: [0, 64],
                runtime_offset: 4096,
                byte_len: 64,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };

        let registry = TensorRegistry::from_manifest(&manifest);
        let alias = registry
            .tensor("model.layers.7.self_attn.q_proj.weight")
            .unwrap();

        assert_eq!(
            alias.name,
            "model.language_model.layers.7.self_attn.q_proj.weight"
        );
        assert_eq!(alias.byte_offset, 4096);
        assert_eq!(alias.alignment, TENSOR_ALIGNMENT);
        assert!(registry.has_tensor_with_prefix("model.layers.7"));
    }

    #[test]
    fn tensor_registry_keeps_expert_manifest_refs_as_import_compatibility() {
        let manifest = FlashMoeManifest {
            model: "fixture".to_string(),
            cache_version: "test".to_string(),
            dense_shards: Vec::new(),
            dense_tensors: Vec::new(),
            expert_tensors: vec![ExpertTensorRef {
                tensor: "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
                shard: "model.safetensors".to_string(),
                layer: Some(0),
                expert: Some(1),
                dtype: Some("F32".to_string()),
                shape: vec![2, 4],
                source_offsets: Some([128, 256]),
                q4_sources: None,
            }],
        };

        let registry = TensorRegistry::from_manifest(&manifest);
        let tensor = registry
            .tensor("model.layers.0.mlp.experts.1.gate_proj.weight")
            .unwrap();

        assert_eq!(tensor.byte_offset, 128);
        assert_eq!(tensor.byte_len, 128);
        assert_eq!(
            tensor.quantization,
            TensorQuantization::Q4 {
                group_size: GROUP_SIZE,
                format: ExpertQuantization::FourBitProduction.as_str().to_string(),
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
            }
        );
    }

    #[test]
    fn dense_mmap_projection_stride_uses_runtime_cols() {
        let projection = DenseMmapMatvecProjection {
            tensor_name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            byte_offset: 4096,
            dtype: "BF16".to_string(),
            rows: 16,
            cols: 32,
            output_width: 64,
        };

        assert_eq!(projection.stride(), 32);
    }

    #[test]
    fn dense_mmap_projection_descriptor_resolves_entry_bounds() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![4, 8],
            byte_offset: 64,
            byte_len: 64,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let projection = DenseMmapMatvecProjection::from_entry(
            "model.layers.0.self_attn.q_proj.weight",
            &entry,
            256,
            4,
            8,
            2,
        )
        .unwrap();

        assert_eq!(projection.byte_offset, 64);
        assert_eq!(projection.rows, 4);
        assert_eq!(projection.cols, 8);
        assert_eq!(projection.output_width, 4);
    }

    #[test]
    fn router_score_projection_descriptor_resolves_dense_binding() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 4],
            byte_offset: 64,
            byte_len: 32,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let descriptor =
            RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 128, 2, 4).unwrap();

        assert_eq!(descriptor.layer, 3);
        assert_eq!(descriptor.experts, 2);
        assert_eq!(descriptor.hidden_width, 4);
        match descriptor.binding {
            RouterScoreProjectionBinding::ResidentDense(projection) => {
                assert_eq!(projection.tensor_name, entry.name);
                assert_eq!(projection.byte_offset, 64);
                assert_eq!(projection.output_width, 2);
            }
            RouterScoreProjectionBinding::ResidentQ4(_) => panic!("expected dense binding"),
        }
    }

    #[test]
    fn router_score_projection_descriptor_resolves_q4_binding() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            byte_offset: 128,
            byte_len: 12,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "BF16".to_string(),
            },
        };

        let descriptor =
            RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 256, 2, 4).unwrap();

        match descriptor.binding {
            RouterScoreProjectionBinding::ResidentQ4(projection) => {
                assert_eq!(projection.packed_byte_offset, 128);
                assert_eq!(projection.output_width, 2);
                assert_eq!(projection.cols, 4);
            }
            RouterScoreProjectionBinding::ResidentDense(_) => panic!("expected q4 binding"),
        }
    }

    #[test]
    fn router_score_projection_execution_declares_binding_without_fallback() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 4],
            byte_offset: 64,
            byte_len: 32,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };
        let descriptor =
            RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 128, 2, 4).unwrap();

        let execution = descriptor.execution(3, 2, 4).unwrap();
        assert_eq!(execution.layer, 3);
        assert_eq!(execution.tensor_name, entry.name);
        assert_eq!(execution.experts, 2);
        assert_eq!(execution.hidden_width, 4);
        assert_eq!(
            execution.kind,
            RouterScoreProjectionExecutionKind::ResidentDense
        );

        let err = descriptor.execution(3, 3, 4).unwrap_err();
        assert!(
            err.to_string()
                .contains("experts 2 does not match scheduled experts 3"),
            "{err:#}"
        );
    }

    #[test]
    fn router_score_projection_descriptor_rejects_wrong_shape_without_fallback() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![3, 4],
            byte_offset: 0,
            byte_len: 48,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let err = RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 64, 2, 4)
            .unwrap_err();

        assert!(err.to_string().contains("shape mismatch"), "{err:#}");
    }

    #[test]
    fn router_score_batch_keeps_projection_with_scores() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.3.mlp.gate.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 4],
            byte_offset: 64,
            byte_len: 32,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };
        let projection =
            RouterScoreProjectionDescriptor::from_entry(3, &entry.name, &entry, 128, 2, 4).unwrap();

        let state = FlashMoeRoutingOutputState::cpu_router_scores(3, 2, 1);
        let batch = RouterScoreBatch::new(state, Some(projection), vec![1.0, -2.0]).unwrap();

        assert_eq!(batch.state(), state);
        assert_eq!(batch.scores, vec![1.0, -2.0]);
        assert_eq!(batch.projection.as_ref().unwrap().layer, 3);
        assert_eq!(batch.projection.as_ref().unwrap().experts, 2);
    }

    #[test]
    fn router_score_batch_rejects_scores_outside_declared_state() {
        let err = RouterScoreBatch::new(
            FlashMoeRoutingOutputState::cpu_router_scores(3, 2, 1),
            None,
            vec![1.0],
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("1 scores for 2 declared experts"),
            "{err:#}"
        );
    }

    #[test]
    fn resident_static_tensor_descriptor_resolves_offsets_and_dtype() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.self_attn.conv1d.weight".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![8],
            byte_offset: 16,
            byte_len: 16,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        let resident =
            ResidentStaticTensorRef::from_entry(&entry.name, &entry, 64, 8, &["BF16", "BFLOAT16"])
                .unwrap()
                .unwrap();

        assert_eq!(resident.tensor_name, entry.name);
        assert_eq!(resident.byte_offset, 16);
        assert_eq!(resident.values, 8);
        assert_eq!(resident.element_size, 2);
    }

    #[test]
    fn resident_static_tensor_descriptor_rejects_wrong_layout_without_fallback() {
        let mut entry = RuntimeTensorEntry {
            name: "model.layers.0.self_attn.A_log".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4],
            byte_offset: 4,
            byte_len: 16,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::None,
        };

        assert!(
            ResidentStaticTensorRef::from_entry(&entry.name, &entry, 32, 4, &["F32"])
                .unwrap()
                .is_some()
        );

        entry.byte_len = 12;
        assert!(
            ResidentStaticTensorRef::from_entry(&entry.name, &entry, 32, 4, &["F32"])
                .unwrap()
                .is_none()
        );

        entry.byte_len = 16;
        entry.quantization = TensorQuantization::Q4 {
            group_size: GROUP_SIZE,
            format: "mlx".to_string(),
            scale_bias_dtype: "F32".to_string(),
        };
        assert!(
            ResidentStaticTensorRef::from_entry(&entry.name, &entry, 32, 4, &["F32"])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn dense_q4_projection_descriptor_carries_one_binding_shape() {
        let projection = DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            packed_byte_offset: 128,
            scales_byte_offset: 256,
            biases_byte_offset: 512,
            rows: 16,
            cols: 32,
            output_width: 16,
            row_packed_bytes: 16,
            groups_per_row: 2,
            group_size: 16,
            scale_bias_dtype: "BF16".to_string(),
        };

        assert_eq!(projection.row_packed_bytes, projection.cols.div_ceil(2));
        assert_eq!(projection.groups_per_row, 2);
        assert_eq!(projection.output_width, projection.rows);
    }

    #[test]
    fn shared_expert_dense_descriptor_groups_projection_weights() {
        let shared = SharedExpertPhaseWeights::new(
            Arc::new(vec![1.0, 2.0]),
            Arc::new(vec![3.0, 4.0]),
            Arc::new(vec![5.0, 6.0]),
            Arc::new(vec![7.0]),
            1,
            2,
            1,
        )
        .unwrap();

        assert_eq!(shared.shared_experts, 1);
        assert_eq!(shared.intermediate, 2);
        assert_eq!(shared.width, 1);
        assert_eq!(shared.gate.as_slice(), &[1.0, 2.0]);
        assert_eq!(shared.router.as_slice(), &[7.0]);
        assert_eq!(
            shared.validated_shape().unwrap(),
            SharedExpertPhaseShape::new(1, 1, 2).unwrap()
        );
    }

    #[test]
    fn shared_expert_weight_builder_loads_named_dense_tensors() {
        let mut tensors = BTreeMap::<String, Arc<Vec<f32>>>::new();
        tensors.insert(
            shared_expert_tensor_name(3, "gate_proj"),
            Arc::new(vec![1.0, 2.0, 3.0, 4.0]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "up_proj"),
            Arc::new(vec![5.0, 6.0, 7.0, 8.0]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "down_proj"),
            Arc::new(vec![9.0, 10.0, 11.0, 12.0]),
        );
        tensors.insert(
            shared_expert_gate_tensor_name(3),
            Arc::new(vec![13.0, 14.0]),
        );

        let shared =
            build_shared_expert_phase_weights(3, 2, 1, 2, |name| Ok(tensors.get(name).cloned()))
                .unwrap()
                .unwrap();

        assert_eq!(shared.width, 2);
        assert_eq!(shared.shared_experts, 1);
        assert_eq!(shared.intermediate, 2);
        assert_eq!(shared.gate.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shared.router.as_slice(), &[13.0, 14.0]);
    }

    #[test]
    fn shared_expert_weight_builder_skips_disabled_shared_experts() {
        let none = build_shared_expert_phase_weights(3, 2, 0, 2, |_| {
            panic!("disabled shared experts must not request tensors")
        })
        .unwrap();
        assert!(none.is_none());

        let none = build_shared_expert_phase_weights(3, 2, 1, 0, |_| {
            panic!("zero intermediate shared experts must not request tensors")
        })
        .unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn shared_expert_weight_builder_rejects_missing_and_mismatched_tensors() {
        let missing = build_shared_expert_phase_weights(3, 2, 1, 2, |_| Ok(None)).unwrap_err();
        assert!(missing.to_string().contains(
            "missing configured shared expert tensor model.layers.3.mlp.shared_expert.gate_proj.weight"
        ));

        let mut tensors = BTreeMap::<String, Arc<Vec<f32>>>::new();
        tensors.insert(
            shared_expert_tensor_name(3, "gate_proj"),
            Arc::new(vec![1.0]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "up_proj"),
            Arc::new(vec![2.0; 4]),
        );
        tensors.insert(
            shared_expert_tensor_name(3, "down_proj"),
            Arc::new(vec![3.0; 4]),
        );
        tensors.insert(shared_expert_gate_tensor_name(3), Arc::new(vec![4.0; 2]));

        let mismatch =
            build_shared_expert_phase_weights(3, 2, 1, 2, |name| Ok(tensors.get(name).cloned()))
                .unwrap_err();
        assert!(mismatch.to_string().contains("shape is invalid"));
    }

    #[test]
    fn shared_expert_q4_descriptor_groups_resident_projection_bindings() {
        let gate = DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.0.mlp.shared_expert.gate_proj.weight".to_string(),
            packed_byte_offset: 128,
            scales_byte_offset: 256,
            biases_byte_offset: 512,
            rows: 16,
            cols: 32,
            output_width: 16,
            row_packed_bytes: 16,
            groups_per_row: 2,
            group_size: 16,
            scale_bias_dtype: "BF16".to_string(),
        };
        let up = DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.0.mlp.shared_expert.up_proj.weight".to_string(),
            ..gate.clone()
        };
        let down = DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.0.mlp.shared_expert.down_proj.weight".to_string(),
            rows: 32,
            cols: 16,
            output_width: 32,
            row_packed_bytes: 8,
            groups_per_row: 1,
            ..gate.clone()
        };
        let router = DenseQ4MmapMatvecProjection {
            tensor_name: "model.layers.0.mlp.shared_expert_gate.weight".to_string(),
            rows: 1,
            output_width: 1,
            ..gate.clone()
        };
        let shared = SharedExpertPhaseQ4Projections {
            gate,
            up,
            down,
            router,
            shared_experts: 1,
            intermediate: 16,
            width: 32,
        };

        assert_eq!(shared.gate.packed_byte_offset, 128);
        assert_eq!(shared.down.output_width, 32);
        assert_eq!(shared.router.cols, 32);
        assert_eq!(shared.shared_experts, 1);
        assert_eq!(shared.intermediate, 16);
        assert_eq!(shared.width, 32);
        assert_eq!(
            shared.validated_shape().unwrap(),
            SharedExpertPhaseShape::new(32, 1, 16).unwrap()
        );
    }

    #[test]
    fn shared_expert_descriptors_reject_mismatched_graph_shape() {
        let shared = SharedExpertPhaseWeights {
            gate: Arc::new(vec![1.0, 2.0]),
            up: Arc::new(vec![3.0, 4.0]),
            down: Arc::new(vec![5.0, 6.0]),
            router: Arc::new(vec![7.0]),
            shared_experts: 1,
            intermediate: 2,
            width: 2,
        };

        let err = shared.validated_shape().unwrap_err();
        assert!(err.to_string().contains("shape is invalid"));
    }

    #[test]
    fn scheduled_next_norm_weights_declare_cpu_visible_width() {
        let values = [1.0, 0.5, 0.25, 0.125];
        let weights = ScheduledNextNormWeights::cpu_visible(
            "model.layers.1.input_layernorm.weight",
            &values,
            3,
        )
        .unwrap();

        assert!(weights.is_cpu_visible());
        assert_eq!(weights.width(), Some(3));
        assert_eq!(weights.values().unwrap(), &[1.0, 0.5, 0.25]);
        assert!(ScheduledNextNormWeights::none().is_none());

        let empty_name_err = ScheduledNextNormWeights::cpu_visible("", &values, 3).unwrap_err();
        assert!(empty_name_err.to_string().contains("require a tensor name"));

        let short_err = ScheduledNextNormWeights::cpu_visible(
            "model.layers.1.input_layernorm.weight",
            &values[..2],
            3,
        )
        .unwrap_err();
        assert!(short_err.to_string().contains("smaller than width 3"));
    }

    #[test]
    fn prepared_next_norm_weights_resolve_declared_cmd3_descriptor() {
        let prepared = PreparedScheduledNextNormWeights::cpu_visible(
            "model.layers.1.input_layernorm.weight".to_string(),
            vec![1.0, 0.5, 0.25, 0.125],
            3,
        )
        .unwrap();
        let scheduled = prepared.scheduled().unwrap();

        assert!(scheduled.is_cpu_visible());
        assert_eq!(scheduled.width(), Some(3));
        assert_eq!(scheduled.values().unwrap(), &[1.0, 0.5, 0.25]);
    }

    #[test]
    fn prepare_next_norm_weights_declares_only_non_terminal_cmd3_layers() {
        let prepared = prepare_scheduled_next_norm_weights(0, 2, 4, true, |name, width| {
            assert_eq!(name, "model.layers.1.input_layernorm.weight");
            assert_eq!(width, 4);
            Ok(Some(vec![1.0, 1.1, 1.2, 1.3]))
        })
        .unwrap();
        assert!(prepared.scheduled().unwrap().is_cpu_visible());

        let terminal = prepare_scheduled_next_norm_weights(1, 2, 4, true, |_, _| {
            panic!("terminal layer must not request next-layer norm weights")
        })
        .unwrap();
        assert!(terminal.scheduled().unwrap().is_none());

        let disabled = prepare_scheduled_next_norm_weights(0, 2, 4, false, |_, _| {
            panic!("disabled next-layer norm must not request weights")
        })
        .unwrap();
        assert!(disabled.scheduled().unwrap().is_none());
    }

    #[test]
    fn prepare_next_norm_weights_reports_missing_scheduled_cmd3_weight() {
        let err = prepare_scheduled_next_norm_weights(2, 4, 8, true, |_, _| Ok(None)).unwrap_err();

        assert!(err.to_string().contains(
            "missing next-layer norm weight model.layers.3.input_layernorm.weight for layer 2"
        ));
    }

    #[test]
    fn qwen_moe_weight_tensor_names_are_canonical_hf_paths() {
        assert_eq!(
            layer_norm_tensor_name(7, "post_attention_layernorm"),
            "model.layers.7.post_attention_layernorm.weight"
        );
        assert_eq!(router_tensor_name(7), "model.layers.7.mlp.gate.weight");
        assert_eq!(
            shared_expert_tensor_name(7, "gate_proj"),
            "model.layers.7.mlp.shared_expert.gate_proj.weight"
        );
        assert_eq!(
            shared_expert_gate_tensor_name(7),
            "model.layers.7.mlp.shared_expert_gate.weight"
        );
    }

    #[test]
    fn qwen3next_plain_rms_norm_offset_policy_matches_reference_names() {
        for name in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.3.self_attn.q_norm.weight",
            "model.layers.3.self_attn.k_norm.weight",
        ] {
            assert!(
                qwen3next_norm_uses_offset(true, name),
                "{name} should use Qwen3Next 1+weight RMSNorm semantics"
            );
        }

        for name in [
            "model.layers.0.linear_attn.norm.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
        ] {
            assert!(
                !qwen3next_norm_uses_offset(true, name),
                "{name} is not a plain Qwen3NextRMSNorm weight"
            );
        }

        assert!(!qwen3next_norm_uses_offset(false, "model.norm.weight"));
    }

    #[test]
    fn qwen3next_norm_offset_is_applied_only_to_offset_style_weights() {
        assert!(qwen3next_norm_weight_needs_offset(&[
            -0.0498, -0.0654, -0.0209, 0.0547
        ]));
        assert!(qwen3next_norm_weight_needs_offset(&[
            0.6679, 0.7187, 0.7265, 0.7031
        ]));
        assert!(!qwen3next_norm_weight_needs_offset(&[
            0.9492, 0.9335, 0.9804, 0.9609
        ]));
        assert!(!qwen3next_norm_weight_needs_offset(&[
            1.6718, 1.7187, 1.7265, 1.7031
        ]));

        let mut offset = vec![0.6679, 0.7187, 0.7265, 0.7031];
        apply_qwen3next_norm_offset_if_needed(
            true,
            "model.layers.0.input_layernorm.weight",
            &mut offset,
        );
        for (actual, expected) in offset.iter().zip([1.6679, 1.7187, 1.7265, 1.7031]) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let mut disabled = vec![0.6679, 0.7187, 0.7265, 0.7031];
        apply_qwen3next_norm_offset_if_needed(false, "model.norm.weight", &mut disabled);
        assert_eq!(disabled, vec![0.6679, 0.7187, 0.7265, 0.7031]);
    }

    #[test]
    fn dense_q4_layout_accounts_for_scale_bias_dtype() {
        let layout = dense_q4_layout_with_scale_bias_dtype(&[2, 4], 16, "BF16").unwrap();

        assert_eq!(layout.rows, 2);
        assert_eq!(layout.cols, 4);
        assert_eq!(layout.row_packed_bytes, 2);
        assert_eq!(layout.groups_per_row, 1);
        assert_eq!(layout.packed_bytes, 4);
        assert_eq!(layout.scales_bytes, 4);
        assert_eq!(layout.scale_bias_bytes, 2);
        assert_eq!(layout.total_bytes, 12);
    }

    #[test]
    fn dense_q4_projection_descriptor_resolves_offsets_from_entry() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            byte_offset: 128,
            byte_len: 12,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "BF16".to_string(),
            },
        };

        let projection = DenseQ4MmapMatvecProjection::from_entry(
            "model.layers.0.mlp.gate_proj.weight",
            &entry,
            256,
            2,
            4,
        )
        .unwrap()
        .unwrap();

        assert_eq!(projection.packed_byte_offset, 128);
        assert_eq!(projection.scales_byte_offset, 132);
        assert_eq!(projection.biases_byte_offset, 136);
        assert_eq!(projection.rows, 2);
        assert_eq!(projection.cols, 4);
        assert_eq!(projection.scale_bias_dtype, "BF16");
    }

    #[test]
    fn dense_q4_projection_descriptor_rejects_missing_capacity() {
        let entry = RuntimeTensorEntry {
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            byte_offset: 128,
            byte_len: 12,
            alignment: TENSOR_ALIGNMENT,
            quantization: TensorQuantization::Q4 {
                group_size: 16,
                format: "dense-q4".to_string(),
                scale_bias_dtype: "BF16".to_string(),
            },
        };

        let projection = DenseQ4MmapMatvecProjection::from_entry(
            "model.layers.0.mlp.gate_proj.weight",
            &entry,
            139,
            2,
            4,
        )
        .unwrap();

        assert_eq!(projection, None);
    }
}
