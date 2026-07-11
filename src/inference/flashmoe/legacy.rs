//! Flash-MoE inspired inference backend for Qwen3.5-397B-A17B on Apple Silicon.
//!
//! The upstream flash-moe design is very different from llama.cpp: non-expert
//! tensors are mmap'd, routed expert tensors stay on SSD, and each token reads
//! only the active MoE experts with parallel `pread` before dispatching fused
//! Metal kernels.  This module captures that runtime contract in pb instead of
//! pretending a GGUF file is required for Qwen3.5.

#![allow(
    dead_code,
    clippy::assertions_on_constants,
    clippy::collapsible_if,
    clippy::default_constructed_unit_structs,
    clippy::derivable_impls,
    clippy::enum_variant_names,
    clippy::explicit_counter_loop,
    clippy::manual_checked_ops,
    clippy::manual_inspect,
    clippy::manual_is_multiple_of,
    clippy::manual_saturating_arithmetic,
    clippy::manual_slice_size_calculation,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::needless_option_as_deref,
    clippy::ptr_arg,
    clippy::type_complexity,
    clippy::unnecessary_get_then_check,
    clippy::unnecessary_map_or,
    clippy::useless_format,
    clippy::useless_vec
)]

#[cfg(target_os = "macos")]
use std::ffi::c_int;
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::path::Path;
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use anyhow::{Context, Result, bail};
#[cfg(test)]
use std::collections::BTreeMap;

#[cfg(test)]
use super::cache::*;
#[cfg(test)]
use super::capabilities::FlashMoeGraphStage;
#[cfg(test)]
use super::capabilities::{
    FlashMoeCapabilityPlan, FlashMoeStageCapability, FlashMoeStageImplementation,
    FlashMoeStagePlacement,
};
#[cfg(test)]
use super::experts::*;
use super::experts::{
    ExpertMlpProjection, ExpertRawPayload, ExpertRawRead, ExpertSlotDescriptor,
    FixedQ4ExpertPayload, PackedExpertTensor, Q4MatvecPayload, fixed_q4_payload_from_pbq4_records,
    parse_pbq4_expert_pack,
};
#[cfg(test)]
use super::math::*;
#[cfg(test)]
use super::metal::MetalBatchProjectionInput;
#[cfg(test)]
use super::metal::MetalExecutionContext;
#[cfg(test)]
use super::model_family::QwenMoeExpertComponentKind;
#[cfg(test)]
use super::model_family::QwenMoeModelLayout;
#[cfg(test)]
use super::model_family::QwenMoeQ4ExpertLayout;
#[cfg(test)]
use super::model_family::is_qwen35_or_legacy_alias;
use super::model_family::{QwenModelConfig, QwenMoeFamily};
#[cfg(test)]
use super::planning::*;
#[cfg(test)]
use super::runtime::MetalExecutionFacade;
#[cfg(test)]
use super::scheduler::FlashMoeScheduledGraph;
#[cfg(test)]
use super::scheduler::ScheduledExpertPhaseMlpPayload;
#[cfg(test)]
use super::scheduler::ScheduledExpertReadCoordinator as ExpertScheduler;
#[cfg(test)]
use super::scheduler::ScheduledQ4ExpertPhaseMlpPayload;
#[cfg(test)]
use super::scheduler::ScheduledRoutingCandidateSource;
#[cfg(test)]
use super::scheduler::ScheduledRoutingCommand;
#[cfg(test)]
use super::scheduler::ScheduledRoutingTopK;
#[cfg(test)]
use super::scheduler::{ScheduledCmd3Expert, ScheduledCmd3ExpertPayload};
#[cfg(test)]
use super::state::KvCache;
use super::state::LinearAttentionLayout;
#[cfg(test)]
use super::state::{
    FlashMoeExpertPhaseOutput, FlashMoeRecurrentLayerState, FlashMoeSessionState,
    FlashMoeStatePlacement, reusable_session_prefix_len, stable_session_cache_tokens,
    take_reusable_session_cache_entry,
};
#[cfg(test)]
use super::text::*;
use super::types::*;
#[cfg(test)]
use super::vision::ImagePreprocessor;
#[cfg(test)]
use super::vision::MropePosition;
#[cfg(test)]
use super::vision::block_major_patch_coords;
#[cfg(test)]
use super::weights::*;
#[cfg(test)]
const DENSE_Q4_GROUP_SIZE: usize = 16;
#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: c_int = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: c_int = 111;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sscal(n: c_int, alpha: f32, x: *mut f32, inc_x: c_int);
    fn cblas_sgemv(
        order: c_int,
        trans_a: c_int,
        m: c_int,
        n: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        x: *const f32,
        inc_x: c_int,
        beta: f32,
        y: *mut f32,
        inc_y: c_int,
    );
    fn cblas_sger(
        order: c_int,
        m: c_int,
        n: c_int,
        alpha: f32,
        x: *const f32,
        inc_x: c_int,
        y: *const f32,
        inc_y: c_int,
        a: *mut f32,
        lda: c_int,
    );
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ImagePlaceholderSpec {
    token_count: usize,
    grid_h: usize,
    grid_w: usize,
}

#[cfg(test)]
impl ImagePlaceholderSpec {
    fn validate(self, image_index: usize) -> Result<()> {
        if self.token_count == 0 {
            bail!("image {image_index} produced zero visual tokens");
        }
        if self.grid_h == 0 || self.grid_w == 0 {
            bail!(
                "image {image_index} has invalid merged grid {}x{}; both dimensions must be positive",
                self.grid_h,
                self.grid_w
            );
        }
        let expected = self.grid_h.saturating_mul(self.grid_w);
        if self.token_count != expected {
            bail!(
                "image {image_index} visual token count {} does not match merged grid {}x{} ({expected} tokens)",
                self.token_count,
                self.grid_h,
                self.grid_w
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisualModality {
    Image,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VisualTokenSpan {
    modality: VisualModality,
    start: usize,
    end: usize,
    grid_h: usize,
    grid_w: usize,
}

#[cfg(test)]
impl VisualTokenSpan {
    fn image(start: usize, end: usize, grid_h: usize, grid_w: usize) -> Self {
        Self {
            modality: VisualModality::Image,
            start,
            end,
            grid_h,
            grid_w,
        }
    }

    fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    fn expected_token_count(self) -> usize {
        match self.modality {
            VisualModality::Image => self.grid_h.saturating_mul(self.grid_w),
        }
    }

    fn position_advance(self) -> usize {
        match self.modality {
            VisualModality::Image => self.grid_h.max(self.grid_w),
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpandedVisionPrompt {
    tokens: Vec<u32>,
    visual_spans: Vec<VisualTokenSpan>,
}

#[cfg(test)]
fn qwen3vl_multimodal_mrope_positions(
    prompt_tokens: &[u32],
    image_pad_token: u32,
    visual_spans: &[VisualTokenSpan],
) -> Result<(Vec<MropePosition>, usize)> {
    let actual_image_tokens = prompt_tokens
        .iter()
        .filter(|&&token| token == image_pad_token)
        .count();
    let expected_image_tokens = visual_spans
        .iter()
        .map(|span| span.end.saturating_sub(span.start))
        .sum::<usize>();
    if actual_image_tokens != expected_image_tokens {
        bail!(
            "image placeholder count {actual_image_tokens} does not match expected visual token count {expected_image_tokens}"
        );
    }

    let mut positions = Vec::with_capacity(prompt_tokens.len());
    let mut current_pos = 0usize;
    let mut i = 0usize;
    let mut span_index = 0usize;
    while i < prompt_tokens.len() {
        if span_index < visual_spans.len() && i == visual_spans[span_index].start {
            let span = visual_spans[span_index];
            if span.end < span.start || span.end > prompt_tokens.len() {
                bail!(
                    "image span {span_index} has invalid bounds {}..{} for prompt length {}",
                    span.start,
                    span.end,
                    prompt_tokens.len()
                );
            }
            if span.grid_h == 0 || span.grid_w == 0 {
                bail!(
                    "image span {span_index} has invalid merged grid {}x{}; both dimensions must be positive",
                    span.grid_h,
                    span.grid_w
                );
            }
            let expected_span_tokens = span.expected_token_count();
            let actual_span_tokens = span.len();
            if actual_span_tokens != expected_span_tokens {
                bail!(
                    "image span {} has {actual_span_tokens} placeholder tokens but grid {}x{} requires {expected_span_tokens}",
                    span_index,
                    span.grid_h,
                    span.grid_w
                );
            }
            let start_position = current_pos;
            let mut image_idx = 0usize;
            while i < span.end {
                if prompt_tokens.get(i).copied() != Some(image_pad_token) {
                    bail!("image span {span_index} contains a non-placeholder token");
                }
                let row = if span.grid_w > 0 {
                    image_idx / span.grid_w
                } else {
                    0
                };
                let col = if span.grid_w > 0 {
                    image_idx % span.grid_w
                } else {
                    0
                };
                positions.push(MropePosition {
                    temporal: start_position,
                    height: start_position + row,
                    width: start_position + col,
                });
                image_idx += 1;
                i += 1;
            }
            current_pos += span.position_advance();
            span_index += 1;
        } else if prompt_tokens[i] == image_pad_token {
            bail!("image placeholder at token {i} is not part of a visual span");
        } else {
            positions.push(MropePosition::text(current_pos));
            current_pos += 1;
            i += 1;
        }
    }
    if span_index != visual_spans.len() {
        bail!(
            "only matched {span_index} visual spans in prompt but {} were expected",
            visual_spans.len()
        );
    }
    Ok((positions, current_pos))
}

#[cfg(test)]
fn qwen3vl_single_image_mrope_positions(
    prompt_tokens: &[u32],
    image_pad_token: u32,
    image_grid_h: usize,
    image_grid_w: usize,
) -> Result<(Vec<MropePosition>, usize)> {
    let (run_start, run_end, _) = single_token_run_bounds(prompt_tokens, image_pad_token)
        .context("single-image prompt contains no image placeholder run")?;
    qwen3vl_multimodal_mrope_positions(
        prompt_tokens,
        image_pad_token,
        &[VisualTokenSpan::image(
            run_start,
            run_end,
            image_grid_h,
            image_grid_w,
        )],
    )
}

#[cfg(test)]
fn expand_multimodal_image_placeholders(
    prompt_tokens: Vec<u32>,
    vision_start_token: u32,
    vision_end_token: u32,
    image_pad_token: u32,
    image_specs: &[ImagePlaceholderSpec],
) -> Result<ExpandedVisionPrompt> {
    let image_runs = token_run_bounds(&prompt_tokens, image_pad_token);
    if image_runs.len() != image_specs.len() {
        bail!(
            "prompt contains {} image placeholder runs but {} images were provided",
            image_runs.len(),
            image_specs.len()
        );
    }

    let mut expanded = Vec::with_capacity(prompt_tokens.len());
    let mut visual_spans = Vec::with_capacity(image_specs.len());
    let mut cursor = 0usize;
    for (image_index, ((run_start, run_end, image_pad_count), spec)) in image_runs
        .into_iter()
        .zip(image_specs.iter().copied())
        .enumerate()
    {
        spec.validate(image_index)?;
        if image_pad_count != 1 && image_pad_count != spec.token_count {
            bail!(
                "image {image_index} placeholder span contains {image_pad_count} <|image_pad|> tokens but the encoded image produced {} visual tokens; use one placeholder for implicit expansion or exactly one per visual token",
                spec.token_count
            );
        }
        let has_start = run_start > 0 && prompt_tokens[run_start - 1] == vision_start_token;
        let has_end = run_end < prompt_tokens.len() && prompt_tokens[run_end] == vision_end_token;
        if has_start != has_end {
            bail!(
                "image {image_index} placeholders at token range {run_start}..{run_end} must be wrapped by both <|vision_start|> and <|vision_end|>"
            );
        }

        expanded.extend_from_slice(&prompt_tokens[cursor..run_start]);
        if !has_start {
            expanded.push(vision_start_token);
        }
        let span_start = expanded.len();
        expanded.extend(std::iter::repeat_n(image_pad_token, spec.token_count));
        let span_end = expanded.len();
        visual_spans.push(VisualTokenSpan::image(
            span_start,
            span_end,
            spec.grid_h,
            spec.grid_w,
        ));
        if !has_end {
            expanded.push(vision_end_token);
        }
        cursor = run_end;
    }
    expanded.extend_from_slice(&prompt_tokens[cursor..]);
    Ok(ExpandedVisionPrompt {
        tokens: expanded,
        visual_spans,
    })
}

#[cfg(test)]
fn expand_single_image_placeholders(
    prompt_tokens: Vec<u32>,
    vision_start_token: u32,
    vision_end_token: u32,
    image_pad_token: u32,
    expected_image_tokens: usize,
) -> Result<Vec<u32>> {
    Ok(expand_multimodal_image_placeholders(
        prompt_tokens,
        vision_start_token,
        vision_end_token,
        image_pad_token,
        &[ImagePlaceholderSpec {
            token_count: expected_image_tokens,
            grid_h: 1,
            grid_w: expected_image_tokens,
        }],
    )?
    .tokens)
}

#[cfg(test)]
fn token_run_bounds(tokens: &[u32], needle: u32) -> Vec<(usize, usize, usize)> {
    let mut runs = Vec::new();
    let mut start = None;
    let mut count = 0usize;
    for (idx, &token) in tokens.iter().enumerate() {
        if token == needle {
            if start.is_none() {
                start = Some(idx);
            }
            count += 1;
        } else if let Some(run_start) = start.take() {
            runs.push((run_start, idx, count));
            count = 0;
        }
    }
    if let Some(run_start) = start {
        runs.push((run_start, tokens.len(), count));
    }
    runs
}

#[cfg(test)]
fn single_token_run_bounds(tokens: &[u32], needle: u32) -> Option<(usize, usize, usize)> {
    let mut start = None;
    let mut end = None;
    let mut count = 0usize;
    let mut in_run = false;
    let mut runs = 0usize;
    for (idx, &token) in tokens.iter().enumerate() {
        if token == needle {
            count += 1;
            if !in_run {
                runs += 1;
                if runs > 1 {
                    return None;
                }
                start = Some(idx);
                in_run = true;
            }
            end = Some(idx + 1);
        } else {
            in_run = false;
        }
    }
    Some((start?, end?, count))
}

#[cfg(test)]
fn compute_expert_phase_cpu<E: AsRef<ExpertWeights>>(
    experts: &[E],
    weights: &[f32],
    normed: &[f32],
    residual: &[f32],
    shared: Option<&SharedExpertPhaseWeights>,
    next_norm_weight: Option<&[f32]>,
) -> Result<FlashMoeExpertPhaseOutput> {
    let width = residual.len();
    if weights.len() != experts.len() {
        bail!(
            "expert phase got {} expert weights for {} active experts",
            weights.len(),
            experts.len()
        );
    }
    if normed.len() < width {
        bail!(
            "expert phase normalized input has len {}; expected at least {width}",
            normed.len()
        );
    }
    let mut moe = vec![0.0f32; width];
    if let Some(shared) = shared {
        let total_intermediate = shared
            .shared_experts
            .checked_mul(shared.intermediate)
            .context("shared expert intermediate width overflow")?;
        if shared.width != width
            || shared.gate.len() != total_intermediate * width
            || shared.up.len() != total_intermediate * width
            || shared.down.len() != width * total_intermediate
            || shared.router.len() != shared.shared_experts * width
        {
            bail!("shared expert tensors do not match the deferred expert phase dimensions");
        }
        let gate = cpu_dense_matvec(shared.gate.as_slice(), normed, total_intermediate, width);
        let up = cpu_dense_matvec(shared.up.as_slice(), normed, total_intermediate, width);
        let router = cpu_dense_matvec(
            shared.router.as_slice(),
            normed,
            shared.shared_experts,
            width,
        );
        let mut activated = vec![0.0f32; total_intermediate];
        for idx in 0..total_intermediate {
            let shared_idx = idx / shared.intermediate.max(1);
            let shared_weight = sigmoid(router.get(shared_idx).copied().unwrap_or(0.0));
            activated[idx] = silu(gate[idx]) * up[idx] * shared_weight;
        }
        let shared_out = cpu_dense_matvec(
            shared.down.as_slice(),
            &activated,
            width,
            total_intermediate,
        );
        add_in_place(&mut moe, &shared_out);
    }
    for (expert, weight) in experts.iter().zip(weights.iter().copied()) {
        let contribution = expert.as_ref().mlp(normed, width)?;
        add_scaled_in_place(&mut moe, &contribution, weight);
    }
    let mut hidden = residual.to_vec();
    add_in_place(&mut hidden, &moe);
    let next_normed = next_norm_weight.map(|weight| {
        let mut normed = hidden.clone();
        rms_norm_with_weight_in_place(&mut normed, Some(weight));
        normed
    });
    Ok(FlashMoeExpertPhaseOutput::new(hidden, next_normed))
}

#[cfg(test)]
pub(super) fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

#[cfg(test)]
pub(super) fn ensure_synthetic_runtime_allowed(tensor_name: &str) -> Result<()> {
    if cfg!(test) {
        Ok(())
    } else {
        bail!(
            "Flash-MoE tensor {tensor_name} is unavailable; synthetic runtime fallback is disabled outside tests"
        )
    }
}

fn is_full_attention_layer(layer: usize) -> bool {
    // Compatibility helper for Qwen3.5-397B-A17B's Flash-MoE schedule.
    // Runtime layer type inference must use the tensor manifest instead.
    //
    // Flash-MoE schedules full attention every 4th layer when counted from 1.
    // In 0-indexed coordinates that is layers 3, 7, 11, ...
    (layer + 1).is_multiple_of(FULL_ATTN_INTERVAL)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertWeights {
    pub layer: usize,
    pub expert: usize,
    pub slot: ExpertSlotDescriptor,
    pub packed: Vec<u8>,
    pub records: Vec<PackedExpertTensor>,
    fixed_q4: Option<FixedQ4ExpertPayload>,
}

impl AsRef<ExpertWeights> for ExpertWeights {
    fn as_ref(&self) -> &ExpertWeights {
        self
    }
}

#[cfg(test)]
impl ScheduledCmd3Expert for ExpertWeights {
    fn scheduled_expert_layer(&self) -> usize {
        self.layer
    }

    fn scheduled_expert_id(&self) -> usize {
        self.expert
    }

    fn scheduled_expert_slot_descriptor(&self) -> ExpertSlotDescriptor {
        self.slot
    }
}

#[cfg(test)]
impl ScheduledCmd3ExpertPayload for ExpertWeights {
    fn scheduled_cmd3_expert_phase_payload(
        &self,
        width: usize,
    ) -> Result<ScheduledExpertPhaseMlpPayload<'_>> {
        let fixed_q4 = self.fixed_q4_required()?;
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
                "FlashMoe unsupported active expert CMD3 path: fixed Q4 expert layer {} expert {} does not provide gate/up payloads for width {width}",
                self.layer,
                self.expert
            );
        };
        let Some(down) = fixed_q4.matvec_payload(ExpertMlpProjection::Down, gate.rows, width)
        else {
            bail!(
                "FlashMoe unsupported active expert CMD3 path: fixed Q4 expert layer {} expert {} does not provide down payload for width {width}",
                self.layer,
                self.expert
            );
        };
        Ok(ScheduledExpertPhaseMlpPayload::Q4(
            ScheduledQ4ExpertPhaseMlpPayload::new(self.layer, self.expert, width, gate, up, down)?,
        ))
    }
}

impl ExpertWeights {
    fn from_raw_read(raw: ExpertRawRead) -> Result<Self> {
        let (records, fixed_q4, packed_prefix) = match raw.payload {
            ExpertRawPayload::Pbq4(bytes) => {
                let fixed_q4_spec = raw.slot_spec.fixed_q4().with_context(|| {
                    format!(
                        "PBQ4 import payload at layer {} expert {} cannot populate non-Q4 execution storage",
                        raw.layer, raw.expert
                    )
                })?;
                let records =
                    parse_pbq4_expert_pack(&bytes, Some(&raw.metadata)).with_context(|| {
                        format!(
                            "failed to parse expert pack layer {} expert {}",
                            raw.layer, raw.expert
                        )
                    })?;
                match fixed_q4_payload_from_pbq4_records(
                    raw.layer,
                    raw.expert,
                    fixed_q4_spec,
                    &records,
                    raw.recycle_pool,
                ) {
                    Ok(fixed_q4) => (Vec::new(), Some(fixed_q4), Vec::new()),
                    Err(error) => {
                        tracing::trace!(
                            layer = raw.layer,
                            expert = raw.expert,
                            error = %error,
                            "PBQ4 expert pack is not compatible with the fixed Q4 slot layout"
                        );
                        (records, None, bytes[..bytes.len().min(4096)].to_vec())
                    }
                }
            }
            ExpertRawPayload::FixedQ4(fixed_q4) => (Vec::new(), Some(fixed_q4), Vec::new()),
            ExpertRawPayload::FixedDense(fixed_dense) => {
                bail!(
                    "legacy ExpertWeights adapter does not implement fixed {} expert payloads",
                    fixed_dense.spec.dtype.as_str()
                )
            }
        };
        Ok(Self {
            layer: raw.layer,
            expert: raw.expert,
            slot: raw.slot,
            packed: packed_prefix,
            records,
            fixed_q4,
        })
    }

    #[cfg(test)]
    pub fn q4_fma_matvec(
        &self,
        input: &[f32],
        scales: &[f32],
        biases: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        q4_fma_matvec(&self.packed, input, scales, biases, rows, cols)
    }

    pub(super) fn project(&self, hidden: &[f32], width: usize) -> Result<Vec<f32>> {
        let fixed_q4 = self.fixed_q4_required()?;
        fixed_q4
            .project_cpu(ExpertMlpProjection::Down, hidden, width)
            .with_context(|| {
                format!(
                    "fixed Q4 expert layer {} expert {} has no compatible down projection",
                    self.layer, self.expert
                )
            })
    }

    fn mlp(&self, hidden: &[f32], width: usize) -> Result<Vec<f32>> {
        self.fixed_q4_mlp(self.fixed_q4_required()?, hidden, width)
    }

    fn fixed_q4_required(&self) -> Result<&FixedQ4ExpertPayload> {
        self.fixed_q4.as_ref().with_context(|| {
            format!(
                "FlashMoe unsupported active expert execution for layer {} expert {}: scheduler-owned fixed-Q4 whole-expert slot is required; PBQ4/component records are import compatibility only",
                self.layer, self.expert
            )
        })
    }

    fn fixed_q4_mlp(
        &self,
        fixed_q4: &FixedQ4ExpertPayload,
        hidden: &[f32],
        width: usize,
    ) -> Result<Vec<f32>> {
        let gate = fixed_q4
            .project_cpu(
                ExpertMlpProjection::Gate,
                hidden,
                fixed_q4.spec.intermediate_size,
            )
            .with_context(|| {
                format!(
                    "fixed Q4 expert layer {} expert {} has no compatible gate projection",
                    self.layer, self.expert
                )
            })?;
        let up = fixed_q4
            .project_cpu(
                ExpertMlpProjection::Up,
                hidden,
                fixed_q4.spec.intermediate_size,
            )
            .with_context(|| {
                format!(
                    "fixed Q4 expert layer {} expert {} has no compatible up projection",
                    self.layer, self.expert
                )
            })?;
        let intermediate: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(gate, up)| silu(*gate) * up)
            .collect();
        fixed_q4
            .project_cpu(ExpertMlpProjection::Down, &intermediate, width)
            .with_context(|| {
                format!(
                    "fixed Q4 expert layer {} expert {} has no compatible down projection",
                    self.layer, self.expert
                )
            })
    }

    #[cfg(test)]
    fn record_suffix(&self, suffix: &str) -> Option<&PackedExpertTensor> {
        self.records
            .iter()
            .find(|record| record.name.ends_with(suffix))
    }

    #[cfg(test)]
    fn project_record(
        &self,
        tensor: &PackedExpertTensor,
        input: &[f32],
        width: usize,
    ) -> Result<Option<Vec<f32>>> {
        let Some(payload) = tensor.matvec_payload(
            input,
            width.max(tensor.shape.first().copied().unwrap_or(width)),
        ) else {
            return Ok(None);
        };
        let projected = q4_fma_matvec_with_group_size(
            payload.packed,
            &input[..payload.cols],
            payload.scales,
            payload.biases,
            payload.rows,
            payload.cols,
            payload.group_size,
        )
        .with_context(|| {
            format!(
                "failed to run q4 matvec for expert tensor {} (layer {}, expert {})",
                tensor.name, self.layer, self.expert
            )
        })?;
        Ok(Some(projected))
    }

    fn mix_hash(&self) -> u64 {
        let mut hash = ((self.layer as u64) << 32) ^ self.expert as u64;
        for byte in self.packed.iter().take(4096) {
            hash = hash.rotate_left(5) ^ u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash
    }

    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    fn primary_matvec_payload(&self, hidden: &[f32], width: usize) -> Option<Q4MatvecPayload<'_>> {
        if let Some(fixed_q4) = &self.fixed_q4 {
            return [
                ExpertMlpProjection::Gate,
                ExpertMlpProjection::Up,
                ExpertMlpProjection::Down,
            ]
            .into_iter()
            .filter_map(|projection| fixed_q4.matvec_payload(projection, hidden.len(), width))
            .max_by_key(|payload| payload.rows.saturating_mul(payload.cols));
        }
        None
    }
}

#[cfg(test)]
pub(super) fn read_expert_weights_many(
    store: &ExpertSlotStore,
    layer: usize,
    experts: &[usize],
) -> Result<Vec<ExpertWeights>> {
    store
        .read_many_raw(layer, experts)?
        .into_iter()
        .map(ExpertWeights::from_raw_read)
        .collect()
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn conv1d_step(
    conv_state: &[f32],
    new_input: &[f32],
    weight: &[f32],
    out: &mut [f32],
    channels: usize,
    kernel_size: usize,
) {
    debug_assert_eq!(out.len(), channels);
    for c in 0..channels {
        let mut acc = 0.0f32;
        for k in 0..kernel_size.saturating_sub(1) {
            let w_idx = c
                .checked_mul(kernel_size)
                .and_then(|idx| idx.checked_add(k))
                .unwrap_or(0);
            let s_idx = k
                .checked_mul(channels)
                .and_then(|idx| idx.checked_add(c))
                .unwrap_or(0);
            if let (Some(w), Some(state)) = (weight.get(w_idx), conv_state.get(s_idx)) {
                acc = state.mul_add(*w, acc);
            }
        }
        let tail_w = c
            .checked_mul(kernel_size)
            .and_then(|idx| idx.checked_add(kernel_size.saturating_sub(1)))
            .and_then(|idx| weight.get(idx).copied())
            .unwrap_or(0.0);
        let input = new_input.get(c).copied().unwrap_or(0.0);
        out[c] = silu(input.mul_add(tail_w, acc));
    }
}

fn apply_gated_delta_recurrence(
    layout: LinearAttentionLayout,
    ssm_state: &mut [f32],
    lin_q: &[f32],
    lin_k: &[f32],
    lin_v: &[f32],
    alpha: &[f32],
    beta: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    out_values: &mut [f32],
) {
    let mut kv_mem = vec![0.0f32; layout.value_dim];
    let mut delta = vec![0.0f32; layout.value_dim];
    apply_gated_delta_recurrence_with_scratch(
        layout,
        ssm_state,
        lin_q,
        lin_k,
        lin_v,
        alpha,
        beta,
        a_log,
        dt_bias,
        &mut kv_mem,
        &mut delta,
        out_values,
    );
}

fn apply_gated_delta_recurrence_with_scratch(
    layout: LinearAttentionLayout,
    ssm_state: &mut [f32],
    lin_q: &[f32],
    lin_k: &[f32],
    lin_v: &[f32],
    alpha: &[f32],
    beta: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    kv_mem: &mut [f32],
    delta: &mut [f32],
    out_values: &mut [f32],
) {
    let heads_per_key = layout.value_heads_per_key_head();
    let matrix_len = layout.value_dim * layout.key_dim;
    for vh in 0..layout.num_value_heads {
        let kh = vh / heads_per_key;
        let a_val = alpha.get(vh).copied().unwrap_or(0.0);
        let dt_b = dt_bias.get(vh).copied().unwrap_or(0.0);
        let a_weight = a_log.get(vh).copied().unwrap_or(0.0).exp();
        let softplus = (1.0 + (a_val + dt_b).exp()).ln();
        let decay = (-a_weight * softplus).exp();
        let beta_gate = 1.0 / (1.0 + (-beta.get(vh).copied().unwrap_or(0.0)).exp());

        let state_base = vh * matrix_len;
        let v_base = vh * layout.value_dim;
        let k_base = kh * layout.key_dim;
        let q_base = kh * layout.key_dim;
        let state = &mut ssm_state[state_base..state_base + matrix_len];
        let value = &lin_v[v_base..v_base + layout.value_dim];
        let key = &lin_k[k_base..k_base + layout.key_dim];
        let query = &lin_q[q_base..q_base + layout.key_dim];
        let out = &mut out_values[v_base..v_base + layout.value_dim];

        #[cfg(target_os = "macos")]
        if gated_delta_head_step_accelerate(
            layout.value_dim,
            layout.key_dim,
            state,
            key,
            query,
            value,
            decay,
            beta_gate,
            kv_mem,
            delta,
            out,
        ) {
            continue;
        }

        gated_delta_head_step_scalar(
            layout.value_dim,
            layout.key_dim,
            state,
            key,
            query,
            value,
            decay,
            beta_gate,
            out,
        );
    }
}

#[cfg(target_os = "macos")]
fn gated_delta_head_step_accelerate(
    value_dim: usize,
    key_dim: usize,
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    value: &[f32],
    decay: f32,
    beta_gate: f32,
    kv_mem: &mut [f32],
    delta: &mut [f32],
    out: &mut [f32],
) -> bool {
    let Ok(m) = c_int::try_from(value_dim) else {
        return false;
    };
    let Ok(n) = c_int::try_from(key_dim) else {
        return false;
    };
    let Ok(items) = c_int::try_from(value_dim.saturating_mul(key_dim)) else {
        return false;
    };
    if state.len() != value_dim * key_dim
        || key.len() != key_dim
        || query.len() != key_dim
        || value.len() != value_dim
        || kv_mem.len() != value_dim
        || delta.len() != value_dim
        || out.len() != value_dim
    {
        return false;
    }

    unsafe {
        cblas_sscal(items, decay, state.as_mut_ptr(), 1);
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            m,
            n,
            1.0,
            state.as_ptr(),
            n,
            key.as_ptr(),
            1,
            0.0,
            kv_mem.as_mut_ptr(),
            1,
        );
    }
    for vi in 0..value_dim {
        delta[vi] = (value[vi] - kv_mem[vi]) * beta_gate;
    }
    unsafe {
        cblas_sger(
            CBLAS_ROW_MAJOR,
            m,
            n,
            1.0,
            delta.as_ptr(),
            1,
            key.as_ptr(),
            1,
            state.as_mut_ptr(),
            n,
        );
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            m,
            n,
            1.0,
            state.as_ptr(),
            n,
            query.as_ptr(),
            1,
            0.0,
            out.as_mut_ptr(),
            1,
        );
    }
    true
}

fn gated_delta_head_step_scalar(
    value_dim: usize,
    key_dim: usize,
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    value: &[f32],
    decay: f32,
    beta_gate: f32,
    out: &mut [f32],
) {
    for vi in 0..value_dim {
        let row_base = vi * key_dim;
        let row = &mut state[row_base..row_base + key_dim];
        for slot in row.iter_mut() {
            *slot *= decay;
        }
        let mut kv_mem = 0.0f32;
        for ki in 0..key_dim {
            kv_mem = row[ki].mul_add(key[ki], kv_mem);
        }
        let delta = (value[vi] - kv_mem) * beta_gate;
        for ki in 0..key_dim {
            row[ki] = key[ki].mul_add(delta, row[ki]);
        }
        let mut sum = 0.0f32;
        for ki in 0..key_dim {
            sum = row[ki].mul_add(query[ki], sum);
        }
        out[vi] = sum;
    }
}

#[cfg(test)]
fn read_one_expert(root: &Path, layer: usize, expert: usize) -> Result<ExpertWeights> {
    let store = ExpertSlotStore::open(root.to_path_buf())?;
    let mut experts = read_expert_weights_many(&store, layer, &[expert])?;
    experts
        .pop()
        .with_context(|| format!("expert layer {layer} returned no expert {expert}"))
}

#[cfg(test)]
mod tests {
    use super::super::experts::FixedQ4ExpertPayloadDecoded;
    use super::*;

    fn make_safetensors(tensors: &[(&str, &[u8])]) -> Vec<u8> {
        let typed: Vec<(&str, &str, Vec<usize>, &[u8])> = tensors
            .iter()
            .map(|(name, bytes)| (*name, "U8", vec![bytes.len()], *bytes))
            .collect();
        make_typed_safetensors(&typed)
    }

    fn make_typed_safetensors(tensors: &[(&str, &str, Vec<usize>, &[u8])]) -> Vec<u8> {
        let mut offset = 0usize;
        let mut entries = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let end = offset + bytes.len();
            entries.insert(
                (*name).to_string(),
                serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,end]}),
            );
            data.extend_from_slice(bytes);
            offset = end;
        }
        let header = serde_json::Value::Object(entries).to_string().into_bytes();
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        out
    }

    fn f32_tensor_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn bf16_tensor_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
        for value in values {
            let bits = (value.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        bytes
    }

    fn u32_tensor_bytes(values: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn test_expert_triplet(
        layer: usize,
        expert: usize,
    ) -> Vec<(String, String, Vec<usize>, Vec<u8>)> {
        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        vec![
            (
                format!("{prefix}.gate_proj.weight"),
                "U8".to_string(),
                vec![16, 8],
                vec![1; 16 * 8],
            ),
            (
                format!("{prefix}.up_proj.weight"),
                "U8".to_string(),
                vec![16, 8],
                vec![2; 16 * 8],
            ),
            (
                format!("{prefix}.down_proj.weight"),
                "U8".to_string(),
                vec![8, 16],
                vec![3; 8 * 16],
            ),
        ]
    }

    fn typed_fixture_refs(
        tensors: &[(String, String, Vec<usize>, Vec<u8>)],
    ) -> Vec<(&str, &str, Vec<usize>, &[u8])> {
        tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    name.as_str(),
                    dtype.as_str(),
                    shape.clone(),
                    bytes.as_slice(),
                )
            })
            .collect()
    }

    fn expert_triplet_weight_map(layer: usize, expert: usize) -> String {
        format!(
            r#"{{"weight_map":{{"model.layers.0.self_attn.q_proj.weight":"dense.safetensors","model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight":"expert.safetensors","model.layers.{layer}.mlp.experts.{expert}.up_proj.weight":"expert.safetensors","model.layers.{layer}.mlp.experts.{expert}.down_proj.weight":"expert.safetensors"}}}}"#
        )
    }

    fn write_test_config(snapshot: &Path) {
        std::fs::write(
            snapshot.join("config.json"),
            br#"{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
    }

    fn test_tokenizer_json() -> &'static [u8] {
        br#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 100, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 101, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 102, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0,
      "h": 1,
      "i": 2,
      "hi": 3,
      "hello": 4,
      "user": 5,
      "assistant": 6,
      "<|im_start|>": 100,
      "<|im_end|>": 101,
      "<|endoftext|>": 102
    },
    "unk_token": "<unk>"
  }
}"#
    }

    fn test_tokenizer_config_json() -> &'static [u8] {
        br##"{
  "bos_token": null,
  "eos_token": "<|im_end|>",
  "pad_token": "<|endoftext|>",
  "add_bos_token": false,
  "added_tokens_decoder": {
    "100": {"content": "<|im_start|>", "special": true},
    "101": {"content": "<|im_end|>", "special": true},
    "102": {"content": "<|endoftext|>", "special": true}
  },
  "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
  "split_special_tokens": false,
  "model_max_length": 32768,
  "chat_template": "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
}"##
    }

    fn test_default_tokenizer_config_json() -> &'static [u8] {
        br#"{
  "eos_token": "<|im_end|>",
  "add_bos_token": false,
  "split_special_tokens": false,
  "chat_template": "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
}"#
    }

    fn test_tokenizer_config_json_with_template(template: &str) -> Vec<u8> {
        serde_json::json!({
            "bos_token": null,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "add_bos_token": false,
            "added_tokens_decoder": {
                "100": {"content": "<|im_start|>", "special": true},
                "101": {"content": "<|im_end|>", "special": true},
                "102": {"content": "<|endoftext|>", "special": true}
            },
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            "split_special_tokens": false,
            "model_max_length": 32768u64,
            "chat_template": template
        })
        .to_string()
        .into_bytes()
    }

    #[test]
    fn non_flashmoe_models_still_select_llamacpp_backend() {
        for model in [
            "hf://unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M.gguf",
            "qwen-vision.gguf",
            "/models/local-model.gguf",
        ] {
            assert_eq!(select_backend(model), BackendSelection::LlamaCpp);
            assert!(plan(model, Path::new("/models")).is_none());
        }
    }

    #[test]
    fn flashmoe_tokenizer_loads_metadata_from_active_model_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
        std::fs::write(
            snapshot.join("tokenizer_config.json"),
            test_tokenizer_config_json(),
        )
        .unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        let tokenizer = QwenTokenizer::from_files(&plan.tokenizer, &plan.tokenizer_config).unwrap();
        assert_eq!(tokenizer.eos_token_id(), 101);
        assert_eq!(tokenizer.encode("<|im_end|>").unwrap(), vec![101]);
    }

    fn test_qwen3_tool_tokenizer_config_json() -> &'static [u8] {
        Box::leak(
            test_tokenizer_config_json_with_template(
                r#"{%- if tools %}
{{- '<|im_start|>system\n' }}
{%- if messages and messages[0].role == 'system' %}{{- messages[0].content + '\n\n' }}{%- endif %}
{{- '<tools>\n' }}
{%- for tool in tools %}{{- tool | tojson }}{{- '\n' }}{%- endfor %}
{{- '</tools><|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
{%- if not (tools and loop.first and message.role == 'system') %}
{%- if message.role == 'tool' %}
{{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
{%- else %}
{{- '<|im_start|>' + message.role + '\n' }}{{- message.content }}
{%- for tool_call in message.tool_calls %}
{%- if message.content and loop.first %}{{- '\n' }}{%- endif %}
{{- '<tool_call>\n{"name": ' }}{{- tool_call.name | tojson }}{{- ', "arguments": ' }}{{- tool_call.arguments | tojson }}{{- '}\n</tool_call>\n' }}
{%- endfor %}
{{- '<|im_end|>\n' }}
{%- endif %}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"#,
            )
            .into_boxed_slice(),
        )
    }

    fn test_qwen3vl_tool_tokenizer_config_json() -> &'static [u8] {
        Box::leak(
            test_tokenizer_config_json_with_template(
                r#"{%- macro render_content(content) %}
{%- if content and (content[0].type is defined or content[0].image is defined or content[0].image_url is defined or content[0].text is defined) %}
{%- for item in content %}
{%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
{{- '<|vision_start|><|image_pad|><|vision_end|>' }}
{%- elif 'text' in item %}
{{- item.text }}
{%- endif %}
{%- endfor %}
{%- else %}
{{- content }}
{%- endif %}
{%- endmacro %}
{%- if tools %}
{{- '<|im_start|>system\n<tools>\n' }}
{%- for tool in tools %}{{- tool | tojson }}{{- '\n' }}{%- endfor %}
{{- '</tools><|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
{%- if message.role == 'tool' %}
{{- '<|im_start|>user\n<tool_response>\n' }}{{- render_content(message.content) }}{{- '\n</tool_response><|im_end|>\n' }}
{%- else %}
{{- '<|im_start|>' + message.role + '\n' }}{{- render_content(message.content) }}
{%- for tool_call in message.tool_calls %}
{%- if message.content and loop.first %}{{- '\n' }}{%- endif %}
{{- '<tool_call>\n{"name": ' }}{{- tool_call.name | tojson }}{{- ', "arguments": ' }}{{- tool_call.arguments | tojson }}{{- '}\n</tool_call>\n' }}
{%- endfor %}
{{- '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"#,
            )
            .into_boxed_slice(),
        )
    }

    fn test_byte_bpe_tokenizer_json() -> &'static [u8] {
        br#"{
  "version": "1.0",
  "added_tokens": [
    {"id": 100, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 101, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 102, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
  "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "h": 1,
      "e": 2,
      "l": 3,
      "o": 4,
      "he": 5,
      "hel": 6,
      "hell": 7,
      "hello": 8,
      "\u0120": 9,
      "w": 10,
      "r": 11,
      "d": 12,
      "wo": 13,
      "wor": 14,
      "worl": 15,
      "world": 16,
      "<|im_start|>": 100,
      "<|im_end|>": 101,
      "<|endoftext|>": 102
    },
    "merges": ["h e", "he l", "hel l", "hell o", "w o", "wo r", "wor l", "worl d"],
    "unk_token": "<unk>"
  }
}"#
    }

    fn test_qwen3vl_tokenizer_json() -> &'static [u8] {
        br#"{
  "version": "1.0",
  "added_tokens": [
    {"id": 100, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 101, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 102, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 200, "content": "<|vision_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 201, "content": "<|vision_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 202, "content": "<|image_pad|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "pre_tokenizer": {"type": "Whitespace"},
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0,
      "user": 5,
      "assistant": 6,
      "describe": 7,
      "now": 8,
      "<|im_start|>": 100,
      "<|im_end|>": 101,
      "<|endoftext|>": 102,
      "<|vision_start|>": 200,
      "<|vision_end|>": 201,
      "<|image_pad|>": 202
    },
    "unk_token": "<unk>"
  }
}"#
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "{actual:.8} != {expected:.8}"
        );
    }

    fn assert_close_with_tolerance(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual:.8} != {expected:.8} within {tolerance:.8}"
        );
    }

    #[derive(Debug, Clone, Copy)]
    enum FlashMoeFixtureFamily {
        Qwen35FlashMoe,
        Qwen3Moe,
        Qwen3VlMoe,
    }

    impl FlashMoeFixtureFamily {
        fn model(self) -> &'static str {
            match self {
                Self::Qwen35FlashMoe => QWEN35_MODEL,
                Self::Qwen3Moe => "hf://Qwen/Qwen3-30B-A3B",
                Self::Qwen3VlMoe => QWEN3_VL_MODEL,
            }
        }

        fn config_json(self) -> &'static [u8] {
            match self {
                Self::Qwen35FlashMoe => {
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
}"#
                }
                Self::Qwen3Moe => {
                    br#"{
  "model_type": "qwen3_moe",
  "architectures": ["Qwen3MoeForCausalLM"],
  "num_hidden_layers": 2,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 151936,
  "rope_theta": 1000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 512,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 1536
}"#
                }
                Self::Qwen3VlMoe => {
                    br#"{
  "model_type": "qwen3_vl_moe",
  "architectures": ["Qwen3VLMoeForConditionalGeneration"],
  "num_hidden_layers": 2,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 248320,
  "rope_theta": 1000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 512,
  "num_experts_per_tok": 3,
  "moe_intermediate_size": 1536,
  "rope_scaling": {"mrope_section": [24, 20, 20]},
  "vision_config": {
    "depth": 1,
    "hidden_size": 64,
    "num_heads": 4,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "out_hidden_size": 4096
  }
}"#
                }
            }
        }

        fn config(self) -> QwenModelConfig {
            serde_json::from_slice(self.config_json()).unwrap()
        }
    }

    fn tiny_q4_expert_pack() -> (Vec<u8>, ExpertPackMetadata) {
        let prefix = "model.layers.0.mlp.experts.1";
        build_expert_pack(
            0,
            1,
            vec![
                ExpertRecordInput {
                    tensor: format!("{prefix}.gate_proj.weight"),
                    dtype: "F32".to_string(),
                    shape: vec![2, 2],
                    source_offsets: [0, 16],
                    source_hash: Some("fixture-gate".to_string()),
                    values: vec![0.0, 15.0, 15.0, 0.0],
                },
                ExpertRecordInput {
                    tensor: format!("{prefix}.up_proj.weight"),
                    dtype: "F32".to_string(),
                    shape: vec![2, 2],
                    source_offsets: [16, 32],
                    source_hash: Some("fixture-up".to_string()),
                    values: vec![15.0, 15.0, 15.0, 15.0],
                },
                ExpertRecordInput {
                    tensor: format!("{prefix}.down_proj.weight"),
                    dtype: "F32".to_string(),
                    shape: vec![2, 2],
                    source_offsets: [32, 48],
                    source_hash: Some("fixture-down".to_string()),
                    values: vec![15.0, 0.0, 0.0, 15.0],
                },
            ],
        )
        .unwrap()
    }

    fn fixed_q4_test_layout(
        hidden_size: usize,
        intermediate_size: usize,
        group_size: usize,
    ) -> QwenMoeQ4ExpertLayout {
        use crate::inference::flashmoe::QwenMoeExpertComponentLayout;
        use QwenMoeExpertComponentKind::*;

        let packed_gate_up = intermediate_size * hidden_size.div_ceil(2);
        let gate_up_scale_bias = intermediate_size * hidden_size.div_ceil(group_size) * 2;
        let packed_down = hidden_size * intermediate_size.div_ceil(2);
        let down_scale_bias = hidden_size * intermediate_size.div_ceil(group_size) * 2;
        let mut offset = 0usize;
        let mut component = |kind, bytes| {
            let layout = QwenMoeExpertComponentLayout {
                kind,
                offset,
                bytes,
            };
            offset += bytes;
            layout
        };
        let components = [
            component(GateWeight, packed_gate_up),
            component(GateScale, gate_up_scale_bias),
            component(GateBias, gate_up_scale_bias),
            component(UpWeight, packed_gate_up),
            component(UpScale, gate_up_scale_bias),
            component(UpBias, gate_up_scale_bias),
            component(DownWeight, packed_down),
            component(DownScale, down_scale_bias),
            component(DownBias, down_scale_bias),
        ];
        QwenMoeQ4ExpertLayout {
            expert_bytes: offset,
            group_size,
            components,
        }
    }

    #[test]
    fn flashmoe_parity_tokenizer_chat_template_and_routing_goldens() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_tokenizer_config_json()),
        )
        .unwrap();

        let rendered = tokenizer
            .apply_chat_template_to_messages(&[ChatMessage::text(ChatRole::User, "hi")], &[], true)
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(tokenizer.encode("hi<|im_end|>").unwrap(), vec![3, 101]);
        assert_eq!(tokenizer.decode(&[3, 101, 4]).unwrap(), "hi");

        let routed = top_k(&[0.0, 2.0, 2.0, -1.0, 1.0], 3);
        assert_eq!(routed, vec![(1, 2.0), (2, 2.0), (4, 1.0)]);
        let mut weights: Vec<f32> = routed.iter().map(|(_, score)| *score).collect();
        softmax_in_place(&mut weights);
        for (actual, expected) in weights.iter().zip([0.42231882, 0.42231882, 0.15536241]) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn flashmoe_parity_q4_expert_pack_and_mlp_goldens() {
        let (pack, metadata) = tiny_q4_expert_pack();
        let records = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].shape, vec![2, 2]);
        assert_eq!(records[0].group_size, GROUP_SIZE);
        assert_eq!(records[0].packed, vec![0xf0, 0x0f]);
        assert_eq!(records[0].scales, vec![1.0, 1.0]);
        assert_eq!(records[0].biases, vec![0.0, 0.0]);
        assert_eq!(records[1].packed, vec![0x00, 0x00]);
        assert_eq!(records[1].biases, vec![15.0, 15.0]);

        let parsed_expert = ExpertWeights {
            layer: 0,
            expert: 1,
            slot: ExpertSlotDescriptor {
                layer: 0,
                expert: 1,
                slot_offset: 0,
                slot_capacity: metadata.packed_bytes as usize,
                payload_len: metadata.packed_bytes as usize,
            },
            packed: pack,
            records,
            fixed_q4: None,
        };
        assert_eq!(parsed_expert.slot.layer, parsed_expert.layer);
        assert_eq!(parsed_expert.slot.expert, parsed_expert.expert);
        let hidden = [1.0, 2.0];
        let gate = parsed_expert
            .project_record(
                parsed_expert.record_suffix("gate_proj.weight").unwrap(),
                &hidden,
                2,
            )
            .unwrap()
            .unwrap();
        let up = parsed_expert
            .project_record(
                parsed_expert.record_suffix("up_proj.weight").unwrap(),
                &hidden,
                2,
            )
            .unwrap()
            .unwrap();
        assert_eq!(gate, vec![30.0, 15.0]);
        assert_eq!(up, vec![45.0, 45.0]);

        let err = parsed_expert.mlp(&hidden, 2).unwrap_err();
        assert!(
            err.to_string()
                .contains("PBQ4/component records are import compatibility only"),
            "{err:#}"
        );
        let err = parsed_expert
            .scheduled_cmd3_expert_phase_payload(2)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("scheduler-owned fixed-Q4 whole-expert slot is required"),
            "{err:#}"
        );

        let spec = FixedQ4ExpertSlotSpec {
            layout: fixed_q4_test_layout(2, 2, GROUP_SIZE),
            hidden_size: 2,
            intermediate_size: 2,
        };
        let fixed_q4 = fixed_q4_payload_from_pbq4_records(
            parsed_expert.layer,
            parsed_expert.expert,
            spec,
            &parsed_expert.records,
            None,
        )
        .unwrap();
        let fixed_expert = ExpertWeights {
            layer: parsed_expert.layer,
            expert: parsed_expert.expert,
            slot: ExpertSlotDescriptor {
                layer: parsed_expert.layer,
                expert: parsed_expert.expert,
                slot_offset: 0,
                slot_capacity: spec.layout.expert_bytes,
                payload_len: spec.layout.expert_bytes,
            },
            packed: Vec::new(),
            records: Vec::new(),
            fixed_q4: Some(fixed_q4),
        };
        let intermediate = [silu(gate[0]) * up[0], silu(gate[1]) * up[1]];
        let out = fixed_expert.mlp(&hidden, 2).unwrap();
        assert_close(out[0], 15.0 * intermediate[0]);
        assert_close(out[1], 15.0 * intermediate[1]);
        let err = fixed_expert
            .scheduled_cmd3_expert_phase_payload(2)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("offsets are outside or misaligned"),
            "{err:#}"
        );
    }

    #[test]
    fn pbq4_records_are_adapted_to_fixed_q4_payload() {
        use crate::inference::flashmoe::QwenMoeExpertComponentLayout;
        use QwenMoeExpertComponentKind::*;

        fn bf16_values(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| f32_to_bf16_bits(*value).to_le_bytes())
                .collect()
        }

        fn native_record(
            name: &str,
            shape: Vec<usize>,
            packed: Vec<u8>,
            scales: Vec<u8>,
            biases: Vec<u8>,
            groups: usize,
        ) -> NativeQ4ExpertRecordInput {
            NativeQ4ExpertRecordInput {
                tensor: name.to_string(),
                dtype: "q4".to_string(),
                shape,
                source_offsets: [0, 0],
                source_hash: Some(format!("hash-{name}")),
                packed,
                scale_bytes: scales,
                bias_bytes: biases,
                groups,
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            }
        }

        let layout = QwenMoeQ4ExpertLayout {
            expert_bytes: 464,
            group_size: GROUP_SIZE,
            components: [
                QwenMoeExpertComponentLayout {
                    kind: GateWeight,
                    offset: 0,
                    bytes: 64,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateScale,
                    offset: 64,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateBias,
                    offset: 68,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpWeight,
                    offset: 72,
                    bytes: 64,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpScale,
                    offset: 136,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpBias,
                    offset: 140,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownWeight,
                    offset: 144,
                    bytes: 64,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownScale,
                    offset: 208,
                    bytes: 128,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownBias,
                    offset: 336,
                    bytes: 128,
                },
            ],
        };
        let layer = 5;
        let expert = 7;
        let gate_name = format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight");
        let up_name = format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight");
        let down_name = format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight");
        let gate_packed = vec![0x10; 64];
        let up_packed = vec![0x54; 64];
        let down_packed = vec![0x98; 64];
        let gate_scales = bf16_values(&[0.5, 0.25]);
        let gate_biases = bf16_values(&[1.0, -1.0]);
        let up_scales = bf16_values(&[0.75, 0.125]);
        let up_biases = bf16_values(&[0.0, 0.5]);
        let down_scales = bf16_values(&vec![0.25; 64]);
        let down_biases = bf16_values(&vec![-0.5; 64]);
        let (pack, metadata) = build_native_q4_expert_pack(
            layer,
            expert,
            vec![
                native_record(
                    &gate_name,
                    vec![2, 64],
                    gate_packed.clone(),
                    gate_scales.clone(),
                    gate_biases.clone(),
                    2,
                ),
                native_record(
                    &up_name,
                    vec![2, 64],
                    up_packed.clone(),
                    up_scales.clone(),
                    up_biases.clone(),
                    2,
                ),
                native_record(
                    &down_name,
                    vec![64, 2],
                    down_packed.clone(),
                    down_scales.clone(),
                    down_biases.clone(),
                    64,
                ),
            ],
        )
        .unwrap();
        let records = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap();
        let spec = FixedQ4ExpertSlotSpec {
            layout,
            hidden_size: 64,
            intermediate_size: 2,
        };
        let fixed =
            fixed_q4_payload_from_pbq4_records(layer, expert, spec, &records, None).unwrap();

        assert!(!fixed.bytes.starts_with(PBQ4_EXPERT_MAGIC));
        assert_eq!(&fixed.bytes[0..64], gate_packed.as_slice());
        assert_eq!(&fixed.bytes[64..68], gate_scales.as_slice());
        assert_eq!(&fixed.bytes[68..72], gate_biases.as_slice());
        assert_eq!(&fixed.bytes[72..136], up_packed.as_slice());
        assert_eq!(&fixed.bytes[136..140], up_scales.as_slice());
        assert_eq!(&fixed.bytes[140..144], up_biases.as_slice());
        assert_eq!(&fixed.bytes[144..208], down_packed.as_slice());
        assert_eq!(&fixed.bytes[208..336], down_scales.as_slice());
        assert_eq!(&fixed.bytes[336..464], down_biases.as_slice());

        let parsed_expert = ExpertWeights {
            layer,
            expert,
            slot: ExpertSlotDescriptor {
                layer,
                expert,
                slot_offset: 0,
                slot_capacity: metadata.packed_bytes as usize,
                payload_len: metadata.packed_bytes as usize,
            },
            packed: pack,
            records,
            fixed_q4: None,
        };
        let fixed_expert = ExpertWeights {
            layer,
            expert,
            slot: ExpertSlotDescriptor {
                layer,
                expert,
                slot_offset: 0,
                slot_capacity: layout.expert_bytes,
                payload_len: layout.expert_bytes,
            },
            packed: Vec::new(),
            records: Vec::new(),
            fixed_q4: Some(fixed),
        };
        let hidden: Vec<f32> = (0..64).map(|value| value as f32 / 8.0 - 4.0).collect();
        let err = parsed_expert.mlp(&hidden, 64).unwrap_err();
        assert!(
            err.to_string()
                .contains("PBQ4/component records are import compatibility only"),
            "{err:#}"
        );
        let fixed = fixed_expert.mlp(&hidden, 64).unwrap();
        assert_eq!(fixed.len(), 64);
        assert!(fixed.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn pbq4_layer_cache_rewrites_to_fixed_q4_slots() {
        use crate::inference::flashmoe::QwenMoeExpertComponentLayout;
        use QwenMoeExpertComponentKind::*;

        fn bf16_values(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| f32_to_bf16_bits(*value).to_le_bytes())
                .collect()
        }

        fn native_record(
            name: &str,
            shape: Vec<usize>,
            packed: Vec<u8>,
            scales: Vec<u8>,
            biases: Vec<u8>,
            groups: usize,
        ) -> NativeQ4ExpertRecordInput {
            NativeQ4ExpertRecordInput {
                tensor: name.to_string(),
                dtype: "q4".to_string(),
                shape,
                source_offsets: [11, 22],
                source_hash: Some(format!("hash-{name}")),
                packed,
                scale_bytes: scales,
                bias_bytes: biases,
                groups,
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            }
        }

        let layout = QwenMoeQ4ExpertLayout {
            expert_bytes: 464,
            group_size: GROUP_SIZE,
            components: [
                QwenMoeExpertComponentLayout {
                    kind: GateWeight,
                    offset: 0,
                    bytes: 64,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateScale,
                    offset: 64,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateBias,
                    offset: 68,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpWeight,
                    offset: 72,
                    bytes: 64,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpScale,
                    offset: 136,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpBias,
                    offset: 140,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownWeight,
                    offset: 144,
                    bytes: 64,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownScale,
                    offset: 208,
                    bytes: 128,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownBias,
                    offset: 336,
                    bytes: 128,
                },
            ],
        };
        let layer = 0;
        let expert = 0;
        let gate_name = format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight");
        let up_name = format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight");
        let down_name = format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight");
        let (pack, metadata) = build_native_q4_expert_pack(
            layer,
            expert,
            vec![
                native_record(
                    &gate_name,
                    vec![2, 64],
                    vec![0x10; 64],
                    bf16_values(&[0.5, 0.25]),
                    bf16_values(&[1.0, -1.0]),
                    2,
                ),
                native_record(
                    &up_name,
                    vec![2, 64],
                    vec![0x54; 64],
                    bf16_values(&[0.75, 0.125]),
                    bf16_values(&[0.0, 0.5]),
                    2,
                ),
                native_record(
                    &down_name,
                    vec![64, 2],
                    vec![0x98; 64],
                    bf16_values(&vec![0.25; 64]),
                    bf16_values(&vec![-0.5; 64]),
                    64,
                ),
            ],
        )
        .unwrap();

        let tmp = tempfile::tempdir().unwrap();
        write_test_expert_layer(tmp.path(), layer, vec![(expert, pack, metadata)], 1).unwrap();
        let spec = FixedQ4ExpertSlotSpec {
            layout,
            hidden_size: 64,
            intermediate_size: 2,
        };
        assert!(rewrite_pbq4_layer_to_fixed_q4(tmp.path(), layer, 1, spec).unwrap());
        assert!(!rewrite_pbq4_layer_to_fixed_q4(tmp.path(), layer, 1, spec).unwrap());

        let metadata = read_expert_layer_pack_metadata(tmp.path(), layer)
            .unwrap()
            .unwrap();
        assert_eq!(metadata.format, FIXED_Q4_EXPERT_LAYER_FORMAT_V1);
        assert_eq!(metadata.expert_size, layout.expert_bytes as u64);
        assert_eq!(metadata.packs[0].packed_bytes, layout.expert_bytes as u64);
        assert_eq!(metadata.packs[0].records[0].record_offset, 0);
        assert_eq!(metadata.packs[0].records[1].record_offset, 72);
        assert_eq!(metadata.packs[0].records[2].record_offset, 144);

        let mut prefix = vec![0u8; PBQ4_EXPERT_MAGIC.len()];
        let file = fs::File::open(expert_layer_path(tmp.path(), layer)).unwrap();
        read_exact_at_positioned(&file, &mut prefix, 0).unwrap();
        assert_ne!(prefix, PBQ4_EXPERT_MAGIC);

        let store = ExpertSlotStore::open_with_fixed_q4(tmp.path().to_path_buf(), spec).unwrap();
        let expert = read_expert_weights_many(&store, layer, &[expert])
            .unwrap()
            .pop()
            .unwrap();
        assert!(expert.fixed_q4.is_some());
        assert!(expert.records.is_empty());
        assert_eq!(
            expert.fixed_q4.as_ref().unwrap().bytes.len(),
            layout.expert_bytes
        );
    }

    #[test]
    fn fixed_q4_expert_phase_payload_borrows_whole_slot_bytes() {
        use crate::inference::flashmoe::{ExpertSlotView, QwenMoeExpertComponentLayout};
        use QwenMoeExpertComponentKind::*;
        let layout = QwenMoeQ4ExpertLayout {
            expert_bytes: 48,
            group_size: 2,
            components: [
                QwenMoeExpertComponentLayout {
                    kind: GateWeight,
                    offset: 0,
                    bytes: 8,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateScale,
                    offset: 8,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: GateBias,
                    offset: 12,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpWeight,
                    offset: 16,
                    bytes: 8,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpScale,
                    offset: 24,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: UpBias,
                    offset: 28,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownWeight,
                    offset: 32,
                    bytes: 8,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownScale,
                    offset: 40,
                    bytes: 4,
                },
                QwenMoeExpertComponentLayout {
                    kind: DownBias,
                    offset: 44,
                    bytes: 4,
                },
            ],
        };
        let mut payload: Vec<u8> = (0..48).collect();
        for range in [8..12, 12..16, 24..28, 28..32, 40..44, 44..48] {
            payload[range].copy_from_slice(
                &[
                    f32_to_bf16_bits(1.0).to_le_bytes(),
                    f32_to_bf16_bits(0.0).to_le_bytes(),
                ]
                .concat(),
            );
        }
        let slot = ExpertSlotView::new(3, 4, 2048, 48, &payload).unwrap();
        let view = FixedQ4ExpertSlotView::new(slot, layout).unwrap();
        let spec = FixedQ4ExpertSlotSpec {
            layout,
            hidden_size: 2,
            intermediate_size: 2,
        };
        let decoded = FixedQ4ExpertPayloadDecoded::from_slot(&view, spec).unwrap();
        let expert = ExpertWeights {
            layer: 3,
            expert: 4,
            slot: slot.descriptor(),
            packed: payload[..payload.len().min(4096)].to_vec(),
            records: Vec::new(),
            fixed_q4: Some(FixedQ4ExpertPayload {
                spec,
                bytes: payload,
                decoded: Some(decoded),
                recycle_pool: None,
            }),
        };

        let phase = expert.scheduled_cmd3_expert_phase_payload(2).unwrap();
        let phase = phase.q4();
        let fixed = expert.fixed_q4.as_ref().unwrap();
        let base = fixed.bytes.as_ptr() as usize;
        let end = base + fixed.bytes.len();
        for component in [phase.gate.packed, phase.up.packed, phase.down.packed] {
            let ptr = component.as_ptr() as usize;
            assert!(ptr >= base && ptr < end);
        }
        assert_eq!(phase.gate.scale_bytes, &fixed.bytes[8..12]);
        assert_eq!(phase.up.bias_bytes, &fixed.bytes[28..32]);
        assert_eq!(phase.down.scale_bytes, &fixed.bytes[40..44]);
        let gate_source = phase.gate.source.unwrap();
        let up_source = phase.up.source.unwrap();
        let down_source = phase.down.source.unwrap();
        assert!(gate_source.same_buffer(up_source));
        assert!(gate_source.same_buffer(down_source));
        assert_eq!(gate_source.packed_offset, 0);
        assert_eq!(gate_source.scale_offset, 8);
        assert_eq!(gate_source.bias_offset, 12);
        assert_eq!(up_source.packed_offset, 16);
        assert_eq!(up_source.scale_offset, 24);
        assert_eq!(up_source.bias_offset, 28);
        assert_eq!(down_source.packed_offset, 32);
        assert_eq!(down_source.scale_offset, 40);
        assert_eq!(down_source.bias_offset, 44);

        let lazy_expert = ExpertWeights {
            layer: 3,
            expert: 4,
            slot: expert.slot,
            packed: fixed.payload_prefix(4096).to_vec(),
            records: Vec::new(),
            fixed_q4: Some(FixedQ4ExpertPayload {
                spec,
                bytes: fixed.bytes.clone(),
                decoded: None,
                recycle_pool: None,
            }),
        };
        let hidden = [0.25, -0.5];
        let eager_out = expert.mlp(&hidden, 2).unwrap();
        let lazy_out = lazy_expert.mlp(&hidden, 2).unwrap();
        for (actual, expected) in lazy_out.iter().zip(eager_out.iter()) {
            assert_close(*actual, *expected);
        }
    }
    #[test]
    fn fixed_q4_expert_payload_recycles_whole_slot_bytes_to_pool() {
        let spec = FixedQ4ExpertSlotSpec {
            layout: QwenMoeQ4ExpertLayout::qwen35_a17b(),
            hidden_size: HIDDEN_DIM,
            intermediate_size: 1024,
        };
        let pool = Arc::new(Mutex::new(Vec::new()));
        let mut bytes = Vec::with_capacity(spec.layout.expert_bytes);
        bytes.resize(spec.layout.expert_bytes, 0);

        {
            let _payload = FixedQ4ExpertPayload {
                spec,
                bytes,
                decoded: None,
                recycle_pool: Some(Arc::clone(&pool)),
            };
        }

        assert_eq!(pool.lock().unwrap().len(), 1);
        let returned = take_reusable_expert_bytes(&pool, spec.layout.expert_bytes).unwrap();
        assert!(returned.capacity() >= spec.layout.expert_bytes);
        let mut scratch = ReusableExpertBuffer::default();
        let previous = scratch.adopt_buffer(returned);
        assert_eq!(previous.capacity(), 0);
        assert!(scratch.capacity() >= spec.layout.expert_bytes);
    }

    #[test]
    fn pbq4_metadata_parser_matches_generic_parser() {
        let (pack, metadata) = tiny_q4_expert_pack();
        let generic = parse_pbq4_expert_pack_generic(&pack, Some(&metadata)).unwrap();
        let metadata_fast = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap();

        assert_eq!(metadata_fast, generic);
    }

    #[test]
    fn pbq4_metadata_parser_rejects_record_offset_drift() {
        let (pack, mut metadata) = tiny_q4_expert_pack();
        metadata.records[1].record_offset += 1;

        let err = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap_err();

        assert!(
            err.to_string().contains("metadata offset mismatch"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn build_expert_pack_writes_bf16_scale_bias_metadata_and_stays_projectable() {
        let input_values: Vec<f32> = (0..64).map(|idx| (idx as f32 - 32.0) * 0.125).collect();
        let (pack, metadata) = build_expert_pack(
            0,
            0,
            vec![ExpertRecordInput {
                tensor: "model.layers.0.mlp.experts.0.down_proj.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![1, 64],
                source_offsets: [0, 256],
                source_hash: Some("fixture".to_string()),
                values: input_values,
            }],
        )
        .unwrap();
        let record = &metadata.records[0];
        assert_eq!(record.scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
        assert_eq!(record.groups, 1);
        assert_eq!(
            pack.len(),
            PBQ4_EXPERT_MAGIC.len()
                + 4
                + record.tensor.len()
                + 8
                + 8
                + 2
                + 2
                + record.packed_bytes as usize
        );

        let parsed = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].packed.len(), 32);
        assert_eq!(parsed[0].scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
        assert_eq!(parsed[0].scale_bytes.len(), 2);
        assert_eq!(parsed[0].bias_bytes.len(), 2);
        let scale_offset = record.record_offset as usize + 4 + record.tensor.len() + 8 + 8;
        assert_eq!(parsed[0].scale_bytes, pack[scale_offset..scale_offset + 2]);
        let out = q4_fma_matvec_with_group_size(
            &parsed[0].packed,
            &[1.0; 64],
            &parsed[0].scales,
            &parsed[0].biases,
            1,
            64,
            GROUP_SIZE,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
    }

    #[test]
    fn flashmoe_parity_attention_layout_and_prefix_reuse_goldens() {
        let (mut config, mut manifest) =
            tiny_attention_manifest(&[AttentionLayerType::Full, AttentionLayerType::Linear]);
        config.num_attention_heads = 2;
        config.num_key_value_heads = Some(1);

        let registry = TensorRegistry::from_manifest(&manifest);
        let runtime = DenseTransformerRuntime::from_registry(&config, &registry).unwrap();
        let full = runtime.full_attention_layout(0).unwrap();
        assert_eq!(full.q_layout, FullAttentionQLayout::Standard);
        assert_eq!(full.q_projection_width, 8);
        assert_eq!(full.q_width, 8);
        assert_eq!(full.kv_width, 4);
        assert_eq!(full.head_dim, 4);
        assert_eq!(full.rotary_dim, 4);
        assert_eq!(full.num_q_heads, 2);
        assert_eq!(full.kv_heads, 1);

        let linear = runtime.linear_attention_layout(1).unwrap();
        assert_eq!(
            linear,
            LinearAttentionLayout {
                num_value_heads: 2,
                num_key_heads: 1,
                key_dim: 4,
                value_dim: 2,
                total_key_width: 4,
                total_value_width: 4,
                conv_dim: 12,
                conv_kernel_size: 3,
            }
        );
        assert_eq!(linear.conv_state_len(), 24);
        assert_eq!(linear.ssm_state_len(), 16);

        let q_name = attention_tensor_name(0, "q_proj");
        manifest
            .dense_tensors
            .iter_mut()
            .find(|tensor| tensor.tensor == q_name)
            .unwrap()
            .shape = vec![16, 8];
        let gated = DenseTransformerRuntime::from_registry(
            &config,
            &TensorRegistry::from_manifest(&manifest),
        )
        .unwrap()
        .full_attention_layout(0)
        .unwrap();
        assert_eq!(gated.q_layout, FullAttentionQLayout::Gated);
        assert_eq!(gated.q_projection_width, 16);
        assert_eq!(gated.rotary_dim, 2);

        assert_eq!(
            reusable_session_prefix_len(&[10, 20, 30], &[10, 20, 30, 40]),
            Some(3)
        );
        assert_eq!(reusable_session_prefix_len(&[10, 20, 30], &[10, 20]), None);
        assert_eq!(
            reusable_session_prefix_len(&[10, 20, 30], &[10, 20, 99, 40]),
            None
        );
    }

    #[test]
    fn qwen3vl_parity_multimodal_prompt_image_tokens_and_mrope_goldens() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_qwen3vl_tokenizer_json(),
            Some(test_qwen3vl_tool_tokenizer_config_json()),
        )
        .unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[ChatMessage {
                    role: ChatRole::User,
                    content: ChatMessageContent::Parts(vec![ChatContentPart::Image {
                        image: Some("fixture.png".to_string()),
                        placeholder_tokens: None,
                    }]),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                }],
                &[],
                true,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        );

        let temp = tempfile::tempdir().unwrap();
        let image_file = temp.path().join("qwen3vl_fixture.png");
        let image = image::RgbImage::from_fn(84, 56, |x, y| {
            image::Rgb([(x % 251) as u8, (y % 251) as u8, ((x + y) % 251) as u8])
        });
        image.save(&image_file).unwrap();

        let preprocessor = ImagePreprocessor::default_qwen3_vl();
        let (patch_grid_h, patch_grid_w, patches) = preprocessor.preprocess(&image_file).unwrap();
        assert_eq!((patch_grid_h, patch_grid_w), (4, 6));
        assert_eq!(
            patches.len(),
            patch_grid_h * patch_grid_w * preprocessor.patch_flat_dim()
        );
        let visual_grid_h = patch_grid_h / preprocessor.merge_size;
        let visual_grid_w = patch_grid_w / preprocessor.merge_size;
        let visual_tokens = visual_grid_h * visual_grid_w;
        assert_eq!((visual_grid_h, visual_grid_w, visual_tokens), (2, 3, 6));

        let vision_start = tokenizer.token_id("<|vision_start|>").unwrap();
        let vision_end = tokenizer.token_id("<|vision_end|>").unwrap();
        let image_pad = tokenizer.token_id("<|image_pad|>").unwrap();
        let prompt_tokens = tokenizer.encode(&rendered).unwrap();
        assert_eq!(token_run_bounds(&prompt_tokens, image_pad), vec![(3, 4, 1)]);

        let expanded = expand_multimodal_image_placeholders(
            prompt_tokens,
            vision_start,
            vision_end,
            image_pad,
            &[ImagePlaceholderSpec {
                token_count: visual_tokens,
                grid_h: visual_grid_h,
                grid_w: visual_grid_w,
            }],
        )
        .unwrap();
        assert_eq!(
            expanded.tokens,
            vec![100, 5, 200, 202, 202, 202, 202, 202, 202, 201, 101, 100, 6]
        );
        assert_eq!(
            expanded.visual_spans,
            vec![VisualTokenSpan::image(3, 9, 2, 3)]
        );

        let (positions, next_position) =
            qwen3vl_multimodal_mrope_positions(&expanded.tokens, image_pad, &expanded.visual_spans)
                .unwrap();
        assert_eq!(
            &positions[..3],
            &[
                MropePosition::text(0),
                MropePosition::text(1),
                MropePosition::text(2)
            ]
        );
        assert_eq!(
            &positions[3..9],
            &[
                MropePosition {
                    temporal: 3,
                    height: 3,
                    width: 3,
                },
                MropePosition {
                    temporal: 3,
                    height: 3,
                    width: 4,
                },
                MropePosition {
                    temporal: 3,
                    height: 3,
                    width: 5,
                },
                MropePosition {
                    temporal: 3,
                    height: 4,
                    width: 3,
                },
                MropePosition {
                    temporal: 3,
                    height: 4,
                    width: 4,
                },
                MropePosition {
                    temporal: 3,
                    height: 4,
                    width: 5,
                },
            ]
        );
        assert_eq!(
            &positions[9..],
            &[
                MropePosition::text(6),
                MropePosition::text(7),
                MropePosition::text(8),
                MropePosition::text(9)
            ]
        );
        assert_eq!(next_position, 10);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    mod arm_macos_integration {
        use super::*;

        fn tiny_dense_store(root: &Path) -> DenseStore {
            let dense_path = root.join("model_weights.bin");
            let manifest_path = root.join("model_weights.json");
            std::fs::write(&dense_path, [0u8]).unwrap();
            std::fs::write(
                &manifest_path,
                serde_json::to_vec(&FlashMoeManifest {
                    model: QWEN35_MODEL.to_string(),
                    cache_version: CACHE_VERSION.to_string(),
                    dense_shards: Vec::new(),
                    expert_tensors: Vec::new(),
                    dense_tensors: Vec::new(),
                })
                .unwrap(),
            )
            .unwrap();
            DenseStore::open(dense_path, manifest_path).unwrap()
        }

        #[test]
        #[ignore = "requires Apple Silicon Metal; run on ARM macOS with `cargo test --all-targets -- --ignored`"]
        fn arm_macos_compiles_flashmoe_metal_kernels() {
            let temp = tempfile::tempdir().unwrap();
            let config: QwenModelConfig = serde_json::from_slice(
                br#"{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
            )
            .unwrap();
            let runtime = DenseTransformerRuntime::new(&config);
            let dense = tiny_dense_store(temp.path());
            let _executor = MetalExecutionContext::compile(
                dense.mmap.clone(),
                dense.len,
                &runtime.linear_attention,
            )
            .unwrap();
        }
    }

    #[test]
    fn legacy_qwen_coder_alias_maps_to_qwen35_flashmoe_model() {
        assert_eq!(
            canonical_model("hf://unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M.gguf"),
            QWEN35_MODEL
        );
    }

    #[test]
    fn explicit_qwen35_bf16_model_is_not_rewritten_to_mlx() {
        assert_eq!(canonical_model(QWEN35_BF16_MODEL), QWEN35_BF16_MODEL);
    }

    #[test]
    fn qwen35_and_legacy_alias_are_flashmoe_names() {
        assert!(is_qwen35_or_legacy_alias("hf://Qwen/Qwen3.5-397B-A17B"));
        assert!(is_qwen35_or_legacy_alias("qwen3-coder-next"));
        assert!(!is_qwen35_or_legacy_alias("qwen-vision.gguf"));
        assert_eq!(
            select_backend("qwen-vision.gguf"),
            BackendSelection::LlamaCpp
        );
    }

    #[test]
    fn qwen3_moe_hf_repos_are_flashmoe_pull_candidates() {
        assert!(is_flashmoe_hf_model("hf://Qwen/Qwen3-30B-A3B"));
        assert!(is_flashmoe_hf_model("hf://Qwen/Qwen3-235B-A22B-Instruct"));
        assert!(is_flashmoe_hf_model("hf://Qwen/Qwen3-VL-MoE-Instruct"));
        assert!(!is_flashmoe_hf_model("hf://Qwen/Qwen3-8B"));
        assert!(!is_flashmoe_hf_model("qwen3-30b-a3b"));
    }

    #[test]
    fn plan_uses_flashmoe_cache_layout() {
        let plan = plan_unchecked(QWEN35_MODEL, Path::new("/models"));
        assert!(plan.runtime_dir.ends_with(CACHE_VERSION));
        assert!(plan.non_expert_weights.ends_with("model_weights.bin"));
        assert!(plan.experts_dir.ends_with("packed_experts"));
        assert!(plan.uses_metal);
        assert!(plan.streams_experts_from_nand);
        assert_eq!(plan.quantization, ExpertQuantization::FourBitProduction);
        assert!(plan.describe().contains("397B"));
    }

    #[test]
    fn explicit_bf16_qwen35_uses_bf16_cache_layout() {
        assert_eq!(
            cache_version_for_model(QWEN35_MODEL),
            CACHE_VERSION,
            "default Qwen3.5 FlashMoe model should stay on the MLX Q4 cache"
        );
        assert_eq!(
            cache_version_for_model(QWEN35_BF16_MODEL),
            QWEN35_BF16_CACHE_VERSION,
            "explicit BF16 source model should use the existing BF16 cache"
        );

        let plan = plan_unchecked(QWEN35_BF16_MODEL, Path::new("/models"));
        assert!(plan.runtime_dir.ends_with(QWEN35_BF16_CACHE_VERSION));
        assert_eq!(plan.model, QWEN35_BF16_MODEL);
    }

    #[test]
    fn cache_cleanup_preserves_active_runtime_and_dry_run() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_BF16_MODEL, tmp.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        fs::write(plan.runtime_dir.join("model_weights.bin"), b"active").unwrap();

        let stale_runtime = plan.model_cache_dir.join("flashmoe-v2-denseq4");
        fs::create_dir_all(stale_runtime.join("packed_experts")).unwrap();
        fs::write(stale_runtime.join("model_weights.bin"), b"stale").unwrap();
        fs::write(
            plan.model_cache_dir
                .join("model.safetensors-00001-of-00094.safetensors"),
            b"src",
        )
        .unwrap();

        let dry_run = clean_cache(&plan, false, false).unwrap();

        assert!(!dry_run.deleted);
        assert_eq!(dry_run.candidates.len(), 1);
        assert_eq!(
            dry_run.candidates[0].kind,
            FlashMoeCacheCleanupKind::StaleRuntimeDir
        );
        assert_eq!(dry_run.candidates[0].path, stale_runtime);
        assert!(plan.runtime_dir.join("model_weights.bin").is_file());
        assert!(stale_runtime.is_dir());
    }

    #[test]
    fn cache_cleanup_deletes_stale_runtimes_and_source_shards_only_when_requested() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_BF16_MODEL, tmp.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        fs::write(plan.runtime_dir.join("model_weights.bin"), b"active").unwrap();

        let stale_runtime = plan.model_cache_dir.join("flashmoe-v1");
        let source_shard = plan
            .model_cache_dir
            .join("model.safetensors-00002-of-00094.safetensors");
        let unrelated_file = plan.model_cache_dir.join("config.json");
        fs::create_dir_all(&stale_runtime).unwrap();
        fs::write(stale_runtime.join("model_weights.bin"), b"stale").unwrap();
        fs::write(&source_shard, b"source").unwrap();
        fs::write(&unrelated_file, b"{}").unwrap();

        let runtimes_only = clean_cache(&plan, false, true).unwrap();

        assert!(runtimes_only.deleted);
        assert_eq!(runtimes_only.candidates.len(), 1);
        assert!(!stale_runtime.exists());
        assert!(source_shard.is_file());
        assert!(plan.runtime_dir.join("model_weights.bin").is_file());

        let with_sources = clean_cache(&plan, true, true).unwrap();

        assert_eq!(with_sources.candidates.len(), 1);
        assert_eq!(
            with_sources.candidates[0].kind,
            FlashMoeCacheCleanupKind::SourceShard
        );
        assert!(!source_shard.exists());
        assert!(unrelated_file.is_file());
        assert!(plan.runtime_dir.join("model_weights.bin").is_file());
    }

    #[test]
    fn source_shard_cleanup_does_not_delete_runtime_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        fs::write(plan.runtime_dir.join("model_weights.bin"), b"active").unwrap();

        let stale_runtime = plan.model_cache_dir.join("flashmoe-v1");
        let source_shard = plan
            .model_cache_dir
            .join("model.safetensors-00001-of-00002.safetensors");
        fs::create_dir_all(&stale_runtime).unwrap();
        fs::write(stale_runtime.join("model_weights.bin"), b"stale").unwrap();
        fs::write(&source_shard, b"source").unwrap();

        let report = clean_source_shards(&plan, true).unwrap();

        assert!(report.deleted);
        assert_eq!(report.candidates.len(), 1);
        assert_eq!(
            report.candidates[0].kind,
            FlashMoeCacheCleanupKind::SourceShard
        );
        assert!(!source_shard.exists());
        assert!(stale_runtime.is_dir());
        assert!(plan.runtime_dir.join("model_weights.bin").is_file());
    }

    #[test]
    fn source_shard_cleanup_matches_mlx_model_shard_names() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        fs::write(plan.runtime_dir.join("model_weights.bin"), b"active").unwrap();

        let mlx_source_shard = plan
            .model_cache_dir
            .join("model-00001-of-00046.safetensors");
        fs::write(&mlx_source_shard, b"source").unwrap();

        let report = clean_source_shards(&plan, true).unwrap();

        assert!(report.deleted);
        assert_eq!(report.candidates.len(), 1);
        assert_eq!(
            report.candidates[0].kind,
            FlashMoeCacheCleanupKind::SourceShard
        );
        assert!(!mlx_source_shard.exists());
        assert!(plan.runtime_dir.join("model_weights.bin").is_file());
    }

    #[test]
    fn cache_status_reports_missing_runtime_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        let status = plan.cache_status().unwrap();
        assert!(!status.ready);
        assert!(
            status
                .missing
                .iter()
                .any(|p| p.ends_with("model_weights.bin"))
        );
        assert_eq!(status.expert_files, 0);
    }

    #[test]
    fn cache_status_rejects_partial_expert_layer_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        fs::create_dir_all(&plan.experts_dir).unwrap();
        fs::write(&plan.non_expert_weights, b"dense").unwrap();
        fs::write(
            &plan.tensor_manifest,
            br#"{"model":"","cache_version":"","dense_shards":[],"expert_tensors":[],"dense_tensors":[]}"#,
        )
        .unwrap();
        fs::write(
            &plan.model_config,
            br#"{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
        fs::write(&plan.tokenizer, test_tokenizer_json()).unwrap();
        let packs: Vec<_> = (0..8)
            .map(|expert| {
                let tensor = format!("model.layers.0.mlp.experts.{expert}.down_proj.weight");
                let pack = test_expert_pack(&tensor);
                let metadata = test_expert_pack_metadata(0, expert, &tensor, pack.len());
                (expert, pack, metadata)
            })
            .collect();
        write_test_expert_layer(&plan.experts_dir, 0, packs, 8).unwrap();

        let status = plan.cache_status().unwrap();
        assert!(!status.ready);
        assert!(
            status
                .missing
                .iter()
                .any(|path| path.ends_with("layer_01.bin"))
        );
    }

    #[test]
    fn cleanup_deletes_stale_expert_temp_files_only() {
        let tmp = tempfile::tempdir().unwrap();
        let experts_dir = tmp.path();
        let final_bin = experts_dir.join("layer_00.bin");
        let temp_bin = experts_dir.join("layer_00.bin.tmp-123-ThreadId(1)");
        let temp_json = experts_dir.join("layer_00.json.tmp-123-ThreadId(1)");

        fs::write(&final_bin, b"PBQ4EXPERT ").unwrap();
        fs::write(&temp_bin, b"partial").unwrap();
        fs::write(&temp_json, b"partial").unwrap();

        let deleted = cleanup_stale_expert_temp_files(experts_dir).unwrap();

        assert_eq!(deleted, 2);
        assert!(final_bin.is_file());
        assert!(!temp_bin.exists());
        assert!(!temp_json.exists());
    }

    #[test]
    fn routing_top_k_is_stable_and_softmax_normalizes() {
        let selected = top_k(&[0.1, 0.9, 0.9, -1.0], 2);
        assert_eq!(selected, vec![(1, 0.9), (2, 0.9)]);
        let mut weights: Vec<f32> = selected.iter().map(|(_, score)| *score).collect();
        softmax_in_place(&mut weights);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn optional_qk_norm_absent_leaves_projection_unchanged() {
        let mut values = vec![3.0, 4.0, 5.0, 12.0];
        let original = values.clone();
        apply_optional_per_head_rms_norm(&mut values, 2, 2, None).unwrap();
        assert_eq!(values, original);

        let mut normed = original.clone();
        apply_optional_per_head_rms_norm(&mut normed, 2, 2, Some(&[1.0, 1.0])).unwrap();
        assert_ne!(normed, original);
    }

    #[test]
    fn fused_qk_norm_rope_matches_separate_reference_steps() {
        let layout = FullAttentionLayout {
            q_layout: FullAttentionQLayout::Standard,
            q_projection_width: 16,
            q_width: 16,
            kv_width: 8,
            head_dim: 8,
            rotary_dim: 8,
            num_q_heads: 2,
            kv_heads: 1,
            rotary_pairing: RotaryPairing::SplitHalf,
        };
        let q_weight = vec![0.5, 1.0, 1.5, 2.0, 0.75, 1.25, 1.75, 2.25];
        let k_weight = vec![1.25, 0.75, 1.5, 0.5, 2.0, 1.0, 0.875, 1.125];
        let mut expected_q: Vec<f32> = (0..layout.q_width)
            .map(|idx| ((idx as f32) * 0.31).sin() + 0.125)
            .collect();
        let mut expected_k: Vec<f32> = (0..layout.kv_width)
            .map(|idx| ((idx as f32) * 0.19).cos() - 0.25)
            .collect();
        let mut actual_q = expected_q.clone();
        let mut actual_k = expected_k.clone();

        apply_optional_per_head_rms_norm(
            &mut expected_q,
            layout.num_q_heads,
            layout.head_dim,
            Some(&q_weight),
        )
        .unwrap();
        apply_optional_per_head_rms_norm(
            &mut expected_k,
            layout.kv_heads,
            layout.head_dim,
            Some(&k_weight),
        )
        .unwrap();
        apply_rotary_for_layout(
            &mut expected_q,
            &mut expected_k,
            MropePosition {
                temporal: 7,
                height: 3,
                width: 5,
            },
            1_000_000.0,
            layout,
            Some([1, 1, 1]),
        );

        apply_full_attention_qk_norm_and_rotary(
            &mut actual_q,
            &mut actual_k,
            layout,
            MropePosition {
                temporal: 7,
                height: 3,
                width: 5,
            },
            1_000_000.0,
            Some([1, 1, 1]),
            Some(&q_weight),
            Some(&k_weight),
        )
        .unwrap();

        for (idx, (actual, expected)) in actual_q.iter().zip(expected_q.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 1e-6,
                "q element {idx} diverged: actual={actual}, expected={expected}"
            );
        }
        for (idx, (actual, expected)) in actual_k.iter().zip(expected_k.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 1e-6,
                "k element {idx} diverged: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn qwen3next_plain_rms_norm_offsets_match_reference_module_types() {
        let qwen35: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_5_moe","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":2,"vocab_size":248320,"rope_theta":10000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();
        let legacy_qwen3: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":2}"#,
        )
        .unwrap();

        for name in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.3.self_attn.q_norm.weight",
            "model.layers.3.self_attn.k_norm.weight",
        ] {
            assert!(
                qwen3next_norm_uses_offset(qwen35.uses_qwen3next_norm_offsets(), name),
                "{name} should use Qwen3Next 1+weight RMSNorm semantics"
            );
        }

        for name in [
            "model.layers.0.linear_attn.norm.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
        ] {
            assert!(
                !qwen3next_norm_uses_offset(qwen35.uses_qwen3next_norm_offsets(), name),
                "{name} is not a plain Qwen3NextRMSNorm weight"
            );
        }

        assert!(!qwen3next_norm_uses_offset(
            legacy_qwen3.uses_qwen3next_norm_offsets(),
            "model.norm.weight"
        ));
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
    }

    #[test]
    fn split_gated_q_projection_matches_reference_head_chunks() {
        let layout = FullAttentionLayout {
            q_layout: FullAttentionQLayout::Gated,
            q_projection_width: 12,
            q_width: 6,
            kv_width: 6,
            head_dim: 3,
            rotary_dim: 2,
            num_q_heads: 2,
            kv_heads: 1,
            rotary_pairing: RotaryPairing::SplitHalf,
        };

        let projected = vec![
            1.0, 2.0, 3.0, // head 0 query
            10.0, 20.0, 30.0, // head 0 gate
            4.0, 5.0, 6.0, // head 1 query
            40.0, 50.0, 60.0, // head 1 gate
        ];

        let (query, gate) = split_q_projection(projected, layout).unwrap();

        assert_eq!(query, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(gate.unwrap(), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn causal_attention_gqa_matches_independent_reference() {
        let query = vec![
            0.2, -0.1, // q head 0 -> kv head 0
            0.4, 0.3, // q head 1 -> kv head 0
            -0.5, 0.7, // q head 2 -> kv head 1
            0.6, -0.2, // q head 3 -> kv head 1
        ];
        let k0 = vec![0.1, 0.3, -0.2, 0.4];
        let v0 = vec![1.0, 2.0, 3.0, 4.0];
        let k1 = vec![0.5, -0.4, 0.6, 0.2];
        let v1 = vec![-1.0, 0.5, 2.0, -0.5];
        let keys_values = vec![(&k0[..], &v0[..]), (&k1[..], &v1[..])];

        let got = causal_attention(&query, &keys_values, 4, 2, 2);
        let mut expected = vec![0.0f32; query.len()];
        let scale = (2.0f32).sqrt().recip();
        for q_head in 0..4 {
            let kv_head = q_head / 2;
            let q = &query[q_head * 2..q_head * 2 + 2];
            let mut scores = keys_values
                .iter()
                .map(|(key, _)| {
                    let k = &key[kv_head * 2..kv_head * 2 + 2];
                    (q[0] * k[0] + q[1] * k[1]) * scale
                })
                .collect::<Vec<_>>();
            softmax_in_place(&mut scores);
            for (score, (_, value)) in scores.iter().zip(keys_values.iter()) {
                let v = &value[kv_head * 2..kv_head * 2 + 2];
                expected[q_head * 2] += score * v[0];
                expected[q_head * 2 + 1] += score * v[1];
            }
        }

        for (got, expected) in got.iter().zip(expected.iter()) {
            assert!((got - expected).abs() < 1e-6, "{got} != {expected}");
        }
    }

    #[test]
    fn token_sampler_supports_deterministic_and_seeded_sampling() {
        let logits = vec![0.1, 3.0, 2.9, 0.0];
        let mut deterministic = TokenSampler::new(0.0, 1, 123);
        assert_eq!(deterministic.sample(&logits, &[], &[]).unwrap(), 1);

        let mut seeded_a = TokenSampler::new(0.7, 3, 42);
        let mut seeded_b = TokenSampler::new(0.7, 3, 42);
        let first = seeded_a.sample(&logits, &[], &[]).unwrap();
        let second = seeded_b.sample(&logits, &[], &[]).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn token_sampler_applies_repeat_penalty_before_sampling() {
        let logits = vec![0.0, 2.0, 1.95];
        let sampler = TokenSampler::new(0.7, 3, 7);
        let repeated = sampler.repeated_tokens(&[], &[1]);
        let processed: Vec<f32> = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(token, logit)| sampler.process_logit(token, logit, &repeated))
            .collect();
        assert!(processed[1] < logits[1]);
        assert_eq!(processed[2], logits[2]);
    }

    #[test]
    fn shared_repeat_penalty_matches_sampler_for_cached_lm_head_topk() {
        let sampler = TokenSampler::new(0.7, 4, 7);
        let repeated = sampler.repeated_tokens(&[2], &[1]);
        let logits = [0.0, 2.1, -2.0, 1.8];

        for (token, logit) in logits.iter().copied().enumerate() {
            assert_eq!(
                process_sample_logit(token, logit, sampler.repeat_penalty, &repeated),
                sampler.process_logit(token, logit, &repeated)
            );
        }
        assert!(process_sample_logit(1, logits[1], sampler.repeat_penalty, &repeated) < logits[1]);
        assert!(process_sample_logit(2, logits[2], sampler.repeat_penalty, &repeated) < logits[2]);
        assert_eq!(
            process_sample_logit(3, logits[3], sampler.repeat_penalty, &repeated),
            logits[3]
        );
    }

    #[test]
    fn token_sampler_sampling_from_candidates_matches_full_logits() {
        let logits = vec![0.1, 3.0, 2.9, 0.0, -0.5, 2.0];
        let prompt = vec![5];
        let generated = vec![1, 4];

        let mut full = TokenSampler::new(0.7, 4, 99);
        let mut candidate = TokenSampler::new(0.7, 4, 99);
        let candidates = candidate.top_candidates(&logits, &prompt, &generated);

        assert_eq!(
            full.sample(&logits, &prompt, &generated).unwrap(),
            candidate.sample_candidates(candidates).unwrap()
        );
    }

    #[test]
    fn resident_lm_head_candidate_superset_preserves_repeat_penalized_top_k() {
        let logits = vec![10.0, 9.99, 9.98, 9.97, 9.96, 9.0, 8.0];
        let sampler = TokenSampler::new(0.7, 2, 99);
        let prompt = vec![0, 1, 2];
        let repeated = sampler.repeated_tokens(&prompt, &[]);
        let raw_count = sampler.top_k + repeated.len();
        let raw_candidates = top_k(&logits, raw_count);

        let reranked = rerank_resident_lm_head_candidates(
            &raw_candidates,
            sampler.top_k,
            sampler.repeat_penalty,
            &repeated,
        );

        assert_eq!(reranked, sampler.top_candidates(&logits, &prompt, &[]));
        assert_eq!(
            reranked.iter().map(|(token, _)| *token).collect::<Vec<_>>(),
            vec![3, 4]
        );
    }

    #[test]
    fn top_k_candidates_matches_full_top_k_across_tiles() {
        let scores = [0.2, 1.0, 0.9, -1.0, 3.0, 2.0, 3.0];
        let mut candidates = TopKCandidates::new(3);
        for (offset, chunk) in scores.chunks(2).enumerate() {
            for (inner, score) in chunk.iter().copied().enumerate() {
                candidates.push(offset * 2 + inner, score);
            }
        }
        assert_eq!(candidates.into_sorted_vec(), top_k(&scores, 3));
    }

    #[test]
    fn cpu_routing_topk_and_softmax_support_non_four_k() {
        let scores = [0.2, 1.0, 0.9, -1.0, 3.0, 2.0, 3.0, 1.5];
        let active = top_k(&scores, 5);
        let active_ids: Vec<_> = active.iter().map(|(expert, _)| *expert).collect();
        assert_eq!(active_ids, vec![4, 6, 5, 7, 1]);

        let mut weights: Vec<f32> = active.iter().map(|(_, score)| *score).collect();
        softmax_in_place(&mut weights);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(
            weights
                .iter()
                .all(|weight| weight.is_finite() && *weight > 0.0)
        );
    }

    #[test]
    fn routing_weights_match_flashmoe_softmax_then_topk_reference() {
        let router_scores = [0.25, -1.0, 3.5, 3.5, 0.0, 2.25, -0.5, 1.75];
        let k = 4;

        let active = top_k(&router_scores, k);
        let mut pb_weights: Vec<f32> = active.iter().map(|(_, score)| *score).collect();
        softmax_in_place(&mut pb_weights);

        let mut reference_scores = router_scores;
        softmax_in_place(&mut reference_scores);
        let reference_active = top_k(&reference_scores, k);
        let reference_sum: f32 = reference_active.iter().map(|(_, score)| *score).sum();
        let reference_weights: Vec<f32> = reference_active
            .iter()
            .map(|(_, score)| *score / reference_sum)
            .collect();

        let pb_ids: Vec<_> = active.iter().map(|(expert, _)| *expert).collect();
        let reference_ids: Vec<_> = reference_active.iter().map(|(expert, _)| *expert).collect();
        assert_eq!(pb_ids, reference_ids);
        for (idx, (actual, expected)) in pb_weights.iter().zip(reference_weights.iter()).enumerate()
        {
            assert!(
                (*actual - *expected).abs() <= 1e-6,
                "routing weight {idx} diverged: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn build_cache_writes_runtime_metadata_and_metal_kernels() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
        std::fs::write(
            snapshot.join("tokenizer_config.json"),
            test_tokenizer_config_json(),
        )
        .unwrap();
        write_test_config(&snapshot);
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            b"{\"weight_map\":{}}",
        )
        .unwrap();
        let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot).unwrap();
        assert!(plan.runtime_dir.join("kernels.metal").is_file());
        assert!(plan.tokenizer.is_file());
        assert!(plan.tokenizer_config.is_file());
        assert!(plan.tensor_manifest.is_file());
    }

    #[test]
    fn qwen3vl_cache_status_requires_vision_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN3_VL_MODEL, tmp.path());
        std::fs::create_dir_all(&plan.runtime_dir).unwrap();
        std::fs::create_dir_all(&plan.experts_dir).unwrap();
        std::fs::write(&plan.non_expert_weights, b"").unwrap();
        std::fs::write(
            &plan.tensor_manifest,
            br#"{"model":"hf://Qwen/Qwen3-VL-MoE-Instruct","cache_version":"flashmoe-v3","dense_shards":[],"expert_tensors":[],"dense_tensors":[]}"#,
        )
        .unwrap();
        std::fs::write(
            &plan.model_config,
            br#"{
                "model_type": "qwen3_vl",
                "text_config": {
                    "hidden_size": 8,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 1,
                    "vocab_size": 16,
                    "num_experts": 1,
                    "num_experts_per_tok": 1,
                    "moe_intermediate_size": 4
                },
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 4,
                    "num_heads": 1
                }
            }"#,
        )
        .unwrap();
        std::fs::write(&plan.tokenizer, b"{}").unwrap();

        let status = plan.cache_status().unwrap();
        assert!(
            status
                .missing
                .contains(plan.vision_weights.as_ref().unwrap())
        );
        assert!(
            status
                .missing
                .contains(plan.vision_manifest.as_ref().unwrap())
        );
        assert!(
            status
                .missing
                .contains(plan.vision_config_path.as_ref().unwrap())
        );
    }

    #[test]
    fn resident_dense_mmap_projection_uses_full_row_dispatch() {
        assert_eq!(dense_projection_tile_rows(8192, 4096), 2048);
        assert_eq!(
            dense_projection_tile_rows_for_metal("BF16", 8192, 4096, true),
            4096
        );
        assert_eq!(
            dense_projection_tile_rows_for_metal("BF16", 8192, 4096, false),
            2048
        );
        assert_eq!(
            dense_projection_tile_rows_for_metal("U8", 8192, 4096, true),
            2048
        );
    }

    #[test]
    fn expert_phase_cpu_combines_shared_experts_and_next_norm() {
        let shared = SharedExpertPhaseWeights {
            // Two shared experts, one intermediate channel each, hidden width two.
            gate: Arc::new(vec![1.0, 0.0, 0.0, 1.0]),
            up: Arc::new(vec![0.0, 2.0, 3.0, 0.0]),
            down: Arc::new(vec![1.0, 2.0, -1.0, 0.5]),
            router: Arc::new(vec![1.0, 0.0, 0.0, -1.0]),
            shared_experts: 2,
            intermediate: 1,
            width: 2,
        };
        let residual = vec![0.5, -1.0];
        let normed = vec![2.0, 4.0];
        let experts: &[ExpertWeights] = &[];
        let out = compute_expert_phase_cpu(
            experts,
            &[],
            &normed,
            &residual,
            Some(&shared),
            Some(&[1.0, 0.5]),
        )
        .unwrap();
        let shared_gate: [f32; 2] = [2.0, 4.0];
        let shared_up: [f32; 2] = [8.0, 6.0];
        let shared_router: [f32; 2] = [2.0, -4.0];
        let activated = [
            silu(shared_gate[0]) * shared_up[0] * sigmoid(shared_router[0]),
            silu(shared_gate[1]) * shared_up[1] * sigmoid(shared_router[1]),
        ];
        let expected_hidden = vec![
            residual[0] + activated[0] + 2.0 * activated[1],
            residual[1] - activated[0] + 0.5 * activated[1],
        ];
        let (hidden, next_normed) = out.into_hidden_and_next_normed();
        for (actual, expected) in hidden.iter().zip(expected_hidden.iter()) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        let mut expected_normed = expected_hidden;
        rms_norm_with_weight_in_place(&mut expected_normed, Some(&[1.0, 0.5]));
        for (actual, expected) in next_normed.unwrap().iter().zip(expected_normed.iter()) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn build_cache_parses_safetensors_index_into_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
        write_test_config(&snapshot);
        std::fs::write(
            snapshot.join("dense.safetensors"),
            make_safetensors(&[("model.layers.0.self_attn.q_proj.weight", b"dense")]),
        )
        .unwrap();
        std::fs::write(
            snapshot.join("expert.safetensors"),
            make_typed_safetensors(&typed_fixture_refs(&test_expert_triplet(2, 7))),
        )
        .unwrap();
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            expert_triplet_weight_map(2, 7),
        )
        .unwrap();
        let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot).unwrap();
        let manifest: FlashMoeManifest =
            serde_json::from_slice(&std::fs::read(&plan.tensor_manifest).unwrap()).unwrap();
        assert_eq!(manifest.dense_shards, vec!["dense.safetensors"]);
        assert_eq!(manifest.dense_tensors[0].dtype, "U8");
        assert_eq!(manifest.dense_tensors[0].shape, vec![5]);
        assert_eq!(manifest.dense_tensors[0].runtime_offset, 0);
        assert_eq!(std::fs::read(&plan.non_expert_weights).unwrap(), b"dense");
        assert_eq!(manifest.expert_tensors[0].layer, Some(2));
        assert_eq!(manifest.expert_tensors[0].expert, Some(7));
        assert!(plan.non_expert_weights.is_file());
        let expert_pack = expert_layer_path(&plan.experts_dir, 2);
        assert!(expert_pack.is_file());
        assert!(std::fs::metadata(&expert_pack).unwrap().len() > 0);
        let metadata = read_expert_pack_metadata(&plan.experts_dir, 2, 7)
            .unwrap()
            .unwrap();
        assert_eq!(metadata.expert, 7);

        let registry = TensorRegistry::load(&plan.tensor_manifest).unwrap();
        let dense = registry
            .require("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(dense.dtype, "U8");
        assert_eq!(dense.shape, vec![5]);
        assert_eq!(dense.byte_offset, 0);
        assert_eq!(dense.byte_len, 5);
        assert_eq!(dense.quantization, TensorQuantization::None);
        let expert = registry
            .require("model.layers.2.mlp.experts.7.gate_proj.weight")
            .unwrap();
        assert!(matches!(
            expert.quantization,
            TensorQuantization::Q4 {
                group_size: GROUP_SIZE,
                ..
            }
        ));
    }

    #[test]
    fn expert_tensor_classifier_ignores_mtp_speculative_layers() {
        assert!(is_expert_tensor_name(
            "model.layers.0.mlp.experts.gate_up_proj"
        ));
        assert!(is_expert_tensor_name(
            "model.layers.0.mlp.experts.7.gate_proj.weight"
        ));
        assert!(is_expert_tensor_name(
            "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        ));
        assert!(!is_expert_tensor_name(
            "mtp.layers.0.mlp.experts.7.gate_proj.weight"
        ));
    }

    #[test]
    fn expert_cache_requires_complete_qwen_expert_mlp_triplet() {
        let tensor = ExpertTensorRef {
            tensor: "model.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
            shard: "expert.safetensors".to_string(),
            layer: Some(0),
            expert: Some(0),
            dtype: Some("BF16".to_string()),
            shape: vec![16, 8],
            source_offsets: Some([0, 16]),
            q4_sources: None,
        };
        let err = validate_expert_tensor_group(0, 0, &[&tensor], None).unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required tensor up_proj.weight"),
            "{err:#}"
        );
    }

    #[test]
    fn qwen_config_validates_runtime_dimensions() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":4}"#,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.kv_heads(), 8);
        assert_eq!(config.experts(), 512);
        assert_eq!(config.config_active_experts(), 4);
    }

    #[test]
    fn qwen_config_accepts_arbitrary_num_experts_per_tok() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.config_active_experts(), 10);
    }

    #[test]
    fn routing_policy_defaults_qwen35_flashmoe_profile_to_k4() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_5_moe","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":2,"vocab_size":248320,"rope_theta":10000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();

        let policy = FlashMoeRoutingPolicy::default()
            .resolve(QWEN35_MODEL, &config)
            .unwrap();

        assert_eq!(policy.active_experts, 4);
        assert_eq!(policy.source, ActiveExpertsSource::Qwen35FlashMoeProfile);
    }

    #[test]
    fn routing_policy_defaults_other_qwen_moe_to_model_config_k() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":2}"#,
        )
        .unwrap();

        let policy = FlashMoeRoutingPolicy::default()
            .resolve("hf://Qwen/Qwen3-30B-A3B", &config)
            .unwrap();

        assert_eq!(policy.active_experts, 2);
        assert_eq!(policy.source, ActiveExpertsSource::ModelConfig);
    }

    #[test]
    fn flashmoe_parity_routing_defaults_are_model_family_aware() {
        let qwen35 = FlashMoeRoutingPolicy::default()
            .resolve(
                FlashMoeFixtureFamily::Qwen35FlashMoe.model(),
                &FlashMoeFixtureFamily::Qwen35FlashMoe.config(),
            )
            .unwrap();
        assert_eq!(qwen35.active_experts, 4);
        assert_eq!(qwen35.source, ActiveExpertsSource::Qwen35FlashMoeProfile);

        for (family, expected_k) in [
            (FlashMoeFixtureFamily::Qwen3Moe, 2),
            (FlashMoeFixtureFamily::Qwen3VlMoe, 3),
        ] {
            let policy = FlashMoeRoutingPolicy::default()
                .resolve(family.model(), &family.config())
                .unwrap();
            assert_eq!(policy.active_experts, expected_k, "{family:?}");
            assert_eq!(policy.source, ActiveExpertsSource::ModelConfig);
        }
    }

    #[test]
    fn routing_policy_honors_explicit_active_expert_override() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":2}"#,
        )
        .unwrap();

        let policy = FlashMoeRoutingPolicy::new(Some(6), false)
            .resolve("hf://Qwen/Qwen3-30B-A3B", &config)
            .unwrap();

        assert_eq!(policy.active_experts, 6);
        assert_eq!(policy.source, ActiveExpertsSource::UserOverride);
    }

    #[test]
    fn routing_policy_guards_qwen35_k_below_four_unless_forced() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_5_moe","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":2,"vocab_size":248320,"rope_theta":10000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();

        let err = FlashMoeRoutingPolicy::new(Some(3), false)
            .resolve(QWEN35_MODEL, &config)
            .unwrap_err();
        assert!(err.to_string().contains("requires K >= 4"), "{err:#}");

        let forced = FlashMoeRoutingPolicy::new(Some(3), true)
            .resolve(QWEN35_MODEL, &config)
            .unwrap();
        assert_eq!(forced.active_experts, 3);
        assert!(forced.force_active_experts);
    }

    #[test]
    fn dense_registry_validation_rejects_missing_lm_head_and_transformer_tensors() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":128,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: Vec::new(),
            expert_tensors: Vec::new(),
            dense_tensors: Vec::new(),
        };
        let registry = TensorRegistry::from_manifest(&manifest);
        let err = validate_required_tensor_manifest(&config, &registry).unwrap_err();
        assert!(
            err.to_string().contains("model.embed_tokens.weight"),
            "{err:#}"
        );
    }

    /// Build a `FlashMoeManifest` containing every dense tensor required by `validate_required_tensor_manifest`
    /// for a tiny 1-layer, 8-hidden-dim, 2-head, 1-kv-head, 128-vocab, 4-expert model.
    fn minimal_dense_manifest(with_lm_head: bool) -> (QwenModelConfig, FlashMoeManifest) {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":128,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
        // kv_width = num_key_value_heads(1) * (hidden_size / num_attention_heads) = 1 * (8/2) = 4
        let mut tensors = vec![
            ("model.embed_tokens.weight", vec![128usize, 8]),
            ("model.norm.weight", vec![8]),
            ("model.layers.0.self_attn.q_proj.weight", vec![8, 8]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 8]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 8]),
            ("model.layers.0.self_attn.o_proj.weight", vec![8, 8]),
            ("model.layers.0.self_attn.q_norm.weight", vec![4]),
            ("model.layers.0.self_attn.k_norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![8]),
            ("model.layers.0.post_attention_layernorm.weight", vec![8]),
            ("model.layers.0.mlp.gate.weight", vec![4, 8]),
        ];
        if with_lm_head {
            tensors.push(("lm_head.weight", vec![128, 8]));
        }
        let dense_tensors = tensors
            .iter()
            .enumerate()
            .map(|(i, (name, shape))| {
                let byte_len: u64 = shape.iter().product::<usize>() as u64 * 2; // BF16 = 2 bytes/elem
                DenseTensorRef {
                    tensor: name.to_string(),
                    shard: "shard.safetensors".to_string(),
                    dtype: "BF16".to_string(),
                    shape: shape.clone(),
                    source_offsets: [0, byte_len],
                    runtime_offset: i as u64 * 4096,
                    byte_len,
                    quantization: TensorQuantization::None,
                    q4_sources: None,
                }
            })
            .collect();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["shard.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors,
        };
        (config, manifest)
    }

    fn make_dense_ref(tensor: &str, shape: Vec<usize>, slot: usize) -> DenseTensorRef {
        let byte_len: u64 = shape.iter().product::<usize>() as u64 * 2;
        DenseTensorRef {
            tensor: tensor.to_string(),
            shard: "hybrid.safetensors".to_string(),
            dtype: "BF16".to_string(),
            shape,
            source_offsets: [0, byte_len],
            runtime_offset: slot as u64 * 4096,
            byte_len,
            quantization: TensorQuantization::None,
            q4_sources: None,
        }
    }

    fn tiny_attention_manifest(
        layer_types: &[AttentionLayerType],
    ) -> (QwenModelConfig, FlashMoeManifest) {
        let (mut config, _) = minimal_dense_manifest(true);
        config.num_hidden_layers = layer_types.len();
        let mut slot = 0usize;
        let mut tensors = Vec::new();
        let mut push = |name: String, shape: Vec<usize>| {
            tensors.push(make_dense_ref(&name, shape, slot));
            slot += 1;
        };
        push(
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        );
        push("model.norm.weight".to_string(), vec![config.hidden_size]);
        push(
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        );

        for (layer, layer_type) in layer_types.iter().copied().enumerate() {
            push(
                layer_norm_tensor_name(layer, "input_layernorm"),
                vec![config.hidden_size],
            );
            push(
                layer_norm_tensor_name(layer, "post_attention_layernorm"),
                vec![config.hidden_size],
            );
            push(
                router_tensor_name(layer),
                vec![config.experts(), config.hidden_size],
            );
            match layer_type {
                AttentionLayerType::Full => {
                    push(
                        attention_tensor_name(layer, "q_proj"),
                        vec![config.hidden_size, config.hidden_size],
                    );
                    push(
                        attention_tensor_name(layer, "k_proj"),
                        vec![4, config.hidden_size],
                    );
                    push(
                        attention_tensor_name(layer, "v_proj"),
                        vec![4, config.hidden_size],
                    );
                    push(
                        attention_tensor_name(layer, "o_proj"),
                        vec![config.hidden_size, config.hidden_size],
                    );
                    push(layer_norm_tensor_name(layer, "self_attn.q_norm"), vec![4]);
                    push(layer_norm_tensor_name(layer, "self_attn.k_norm"), vec![4]);
                }
                AttentionLayerType::Linear => {
                    push(
                        linear_attention_tensor_name(layer, "in_proj_qkv"),
                        vec![12, config.hidden_size],
                    );
                    push(
                        linear_attention_tensor_name(layer, "in_proj_z"),
                        vec![4, config.hidden_size],
                    );
                    push(
                        linear_attention_tensor_name(layer, "in_proj_b"),
                        vec![2, config.hidden_size],
                    );
                    push(
                        linear_attention_tensor_name(layer, "in_proj_a"),
                        vec![2, config.hidden_size],
                    );
                    push(linear_attention_tensor_name(layer, "conv1d"), vec![12, 3]);
                    push(linear_attention_scalar_tensor_name(layer, "A_log"), vec![2]);
                    push(
                        linear_attention_scalar_tensor_name(layer, "dt_bias"),
                        vec![2],
                    );
                    push(linear_attention_tensor_name(layer, "norm"), vec![2]);
                    push(
                        linear_attention_tensor_name(layer, "out_proj"),
                        vec![config.hidden_size, 4],
                    );
                }
            }
        }

        let manifest = FlashMoeManifest {
            model: "hf://example/tiny-attention".to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["tiny.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: tensors,
        };
        (config, manifest)
    }

    fn assert_manifest_attention_kinds(layer_types: &[AttentionLayerType]) {
        let (config, manifest) = tiny_attention_manifest(layer_types);
        let registry = TensorRegistry::from_manifest(&manifest);
        validate_required_tensor_manifest(&config, &registry)
            .expect("manifest-driven attention schedule should validate");
        let runtime = DenseTransformerRuntime::from_registry(&config, &registry)
            .expect("manifest-driven attention schedule should build runtime layouts");

        for (layer, layer_type) in layer_types.iter().copied().enumerate() {
            match layer_type {
                AttentionLayerType::Full => {
                    assert_eq!(runtime.layer_kind(layer), FlashMoeLayerKind::FullAttention);
                    runtime
                        .full_attention_layout(layer)
                        .expect("full-attention layer should have full layout");
                    assert!(
                        runtime.linear_attention_layout(layer).is_err(),
                        "full-attention layer {layer} should not have a linear layout"
                    );
                }
                AttentionLayerType::Linear => {
                    assert_eq!(
                        runtime.layer_kind(layer),
                        FlashMoeLayerKind::LinearAttention
                    );
                    runtime
                        .linear_attention_layout(layer)
                        .expect("linear-attention layer should have linear layout");
                    assert!(
                        runtime.full_attention_layout(layer).is_err(),
                        "linear-attention layer {layer} should not have a full layout"
                    );
                }
            }
        }
    }

    #[test]
    fn full_attention_manifest_requires_qk_norm_bindings() {
        let (config, mut manifest) = minimal_dense_manifest(true);
        manifest
            .dense_tensors
            .retain(|tensor| tensor.tensor != "model.layers.0.self_attn.k_norm.weight");
        let registry = TensorRegistry::from_manifest(&manifest);

        let error = validate_required_tensor_manifest(&config, &registry).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("model.layers.0.self_attn.k_norm.weight"),
            "{error:#}"
        );
    }

    #[test]
    fn validate_rejects_configured_shared_expert_without_gate() {
        let (mut config, mut manifest) = minimal_dense_manifest(true);
        config.num_shared_experts = Some(1);
        config.shared_expert_intermediate_size = Some(16);
        let mut slot = manifest.dense_tensors.len();
        for (name, shape) in [
            (
                shared_expert_tensor_name(0, "gate_proj"),
                vec![16, config.hidden_size],
            ),
            (
                shared_expert_tensor_name(0, "up_proj"),
                vec![16, config.hidden_size],
            ),
            (
                shared_expert_tensor_name(0, "down_proj"),
                vec![config.hidden_size, 16],
            ),
        ] {
            manifest
                .dense_tensors
                .push(make_dense_ref(&name, shape, slot));
            slot += 1;
        }

        let registry = TensorRegistry::from_manifest(&manifest);
        let err = validate_required_tensor_manifest(&config, &registry).unwrap_err();
        assert!(
            err.to_string().contains("shared_expert_gate.weight"),
            "{err:#}"
        );
    }

    #[test]
    fn hybrid_attention_schedule_matches_flashmoe() {
        assert!(!is_full_attention_layer(0));
        assert!(!is_full_attention_layer(1));
        assert!(!is_full_attention_layer(2));
        assert!(is_full_attention_layer(3));
        assert!(!is_full_attention_layer(4));
        assert!(!is_full_attention_layer(5));
        assert!(!is_full_attention_layer(6));
        assert!(is_full_attention_layer(7));
    }

    #[test]
    fn manifest_attention_detection_accepts_all_full_attention() {
        assert_manifest_attention_kinds(&[
            AttentionLayerType::Full,
            AttentionLayerType::Full,
            AttentionLayerType::Full,
            AttentionLayerType::Full,
        ]);
    }

    #[test]
    fn manifest_attention_detection_accepts_qwen35_mixed_schedule() {
        let layer_types: Vec<_> = (0..8)
            .map(|layer| {
                if is_full_attention_layer(layer) {
                    AttentionLayerType::Full
                } else {
                    AttentionLayerType::Linear
                }
            })
            .collect();

        assert_manifest_attention_kinds(&layer_types);
    }

    #[test]
    fn manifest_attention_detection_accepts_non_every_fourth_mixed_schedule() {
        assert_manifest_attention_kinds(&[
            AttentionLayerType::Full,
            AttentionLayerType::Linear,
            AttentionLayerType::Full,
            AttentionLayerType::Linear,
        ]);
    }

    #[test]
    fn manifest_attention_detection_rejects_conflicting_layer_layouts() {
        let (config, mut manifest) = tiny_attention_manifest(&[AttentionLayerType::Full]);
        let slot = manifest.dense_tensors.len();
        manifest.dense_tensors.push(make_dense_ref(
            &linear_attention_tensor_name(0, "in_proj_qkv"),
            vec![12, config.hidden_size],
            slot,
        ));
        let registry = TensorRegistry::from_manifest(&manifest);
        let err = validate_required_tensor_manifest(&config, &registry).unwrap_err();

        assert!(err.to_string().contains("both linear-attention"), "{err:#}");
        assert!(err.to_string().contains("full-attention"), "{err:#}");
    }

    #[test]
    fn qwen_q4_graph_binding_rejects_projection_outside_resident_store() {
        let shape = vec![4, 4];
        let group_size = 2;
        let layout =
            dense_q4_layout_with_scale_bias_dtype(&shape, group_size, EXPERT_SCALE_BIAS_DTYPE_BF16)
                .unwrap();
        let tensor_name = "model.layers.0.self_attn.q_proj.weight";
        let byte_offset = 64u64;
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: tensor_name.to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: "U32".to_string(),
                shape,
                source_offsets: [0, layout.total_bytes as u64],
                runtime_offset: byte_offset,
                byte_len: layout.total_bytes as u64,
                quantization: TensorQuantization::Q4 {
                    group_size,
                    format: DENSE_Q4_FORMAT.to_string(),
                    scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                },
                q4_sources: None,
            }],
        };
        let registry = TensorRegistry::from_manifest(&manifest);
        let required_len = byte_offset + layout.total_bytes as u64;

        require_resident_q4_graph_projection(
            QwenMoeFamily::Qwen35A17B,
            &registry,
            required_len,
            "CMD1 full-attention projection",
            tensor_name,
            4,
            4,
        )
        .unwrap();

        let err = require_resident_q4_graph_projection(
            QwenMoeFamily::Qwen35A17B,
            &registry,
            required_len - 1,
            "CMD1 full-attention projection",
            tensor_name,
            4,
            4,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("unsupported resolved Qwen35A17B Q4 CMD1 full-attention projection"),
            "{err:#}"
        );
    }

    fn session_cache_reuse_requires_entire_cached_token_prefix() {
        assert_eq!(
            reusable_session_prefix_len(&[1, 2, 3], &[1, 2, 3, 4, 5]),
            Some(3)
        );
        assert_eq!(reusable_session_prefix_len(&[1, 2, 3], &[1, 2, 9]), None);
        assert_eq!(reusable_session_prefix_len(&[1, 2, 3], &[1, 2]), None);
    }

    fn byte_tokens(text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    fn weather_tool() -> ChatTool {
        ChatTool {
            name: "get_weather".to_string(),
            description: Some("Get weather.".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
        }
    }

    fn assistant_weather_tool_call(content: &str) -> ChatMessage {
        let mut assistant = ChatMessage::text(ChatRole::Assistant, content);
        assistant.tool_calls.push(ChatToolCall {
            id: None,
            name: "get_weather".to_string(),
            arguments: serde_json::json!({"city": "London"}),
        });
        assistant
    }

    fn weather_tool_result() -> ChatMessage {
        ChatMessage {
            role: ChatRole::Tool,
            content: ChatMessageContent::Text("{\"temp\":12}".to_string()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: Some("get_weather".to_string()),
        }
    }

    fn rendered_tool_prompt_pair(assistant: ChatMessage) -> (String, String) {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_qwen3_tool_tokenizer_config_json()),
        )
        .unwrap();
        let tool = weather_tool();
        let initial_messages = vec![
            ChatMessage::text(ChatRole::System, "be precise"),
            ChatMessage::text(ChatRole::User, "weather?"),
        ];
        let first_prompt = tokenizer
            .apply_chat_template_to_messages(&initial_messages, std::slice::from_ref(&tool), true)
            .unwrap();
        let mut next_messages = initial_messages;
        next_messages.push(assistant);
        next_messages.push(weather_tool_result());
        let next_prompt = tokenizer
            .apply_chat_template_to_messages(&next_messages, &[tool], true)
            .unwrap();
        (first_prompt, next_prompt)
    }

    #[test]
    fn session_cache_reuses_prompt_prefix_after_json_compat_tool_call() {
        let (first_prompt, next_prompt) =
            rendered_tool_prompt_pair(assistant_weather_tool_call(""));
        let first_prompt_tokens = byte_tokens(&first_prompt);
        let next_prompt_tokens = byte_tokens(&next_prompt);
        let mut old_cached_tokens = first_prompt_tokens.clone();
        old_cached_tokens.extend(byte_tokens(
            r#"{"type":"tool_call","tool":"get_weather","arguments":{"city":"London"},"thinking":"checking"}"#,
        ));

        assert_eq!(
            reusable_session_prefix_len(&old_cached_tokens, &next_prompt_tokens),
            None
        );
        let stable_cached_tokens = stable_session_cache_tokens(&first_prompt_tokens);
        assert_eq!(
            reusable_session_prefix_len(&stable_cached_tokens, &next_prompt_tokens),
            Some(first_prompt_tokens.len())
        );
    }

    #[test]
    fn session_cache_reuses_prompt_prefix_after_native_tool_call_rerender() {
        let (first_prompt, next_prompt) =
            rendered_tool_prompt_pair(assistant_weather_tool_call("checking"));
        let first_prompt_tokens = byte_tokens(&first_prompt);
        let next_prompt_tokens = byte_tokens(&next_prompt);
        let mut old_cached_tokens = first_prompt_tokens.clone();
        old_cached_tokens.extend(byte_tokens(
            "checking\n<tool_call>\n{\"arguments\":{\"city\":\"London\"},\"name\":\"get_weather\"}\n</tool_call>\n",
        ));

        assert_eq!(
            reusable_session_prefix_len(&old_cached_tokens, &next_prompt_tokens),
            None
        );
        let stable_cached_tokens = stable_session_cache_tokens(&first_prompt_tokens);
        assert_eq!(
            reusable_session_prefix_len(&stable_cached_tokens, &next_prompt_tokens),
            Some(first_prompt_tokens.len())
        );
    }

    #[test]
    fn session_cache_reuse_moves_state_and_shallow_snapshots_cpu_kv_cache() {
        let cached_tokens = vec![10, 20];
        let mut cache = KvCache::new(2, 2);
        for (position, token) in cached_tokens.iter().copied().enumerate() {
            cache.record_prompt_token(position, token).unwrap();
        }
        cache
            .record_kv(0, 0, vec![1.0, 1.1], vec![2.0, 2.1])
            .unwrap();
        cache
            .record_kv(1, 0, vec![3.0, 3.1], vec![4.0, 4.1])
            .unwrap();

        let mut sessions = BTreeMap::new();
        sessions.insert(
            "chat".to_string(),
            FlashMoeSessionState {
                tokens: cached_tokens,
                kv_cache: cache,
                last_hidden: vec![9.0, 9.1],
            },
        );

        let next_prompt = [10, 20, 30];
        let (prefix_len, state) =
            take_reusable_session_cache_entry(&mut sessions, "chat", &next_prompt).unwrap();
        assert!(sessions.is_empty());
        let FlashMoeSessionState {
            tokens,
            mut kv_cache,
            last_hidden,
        } = state;
        assert_eq!(prefix_len, tokens.len());
        assert_eq!(last_hidden, vec![9.0, 9.1]);

        kv_cache.resize_capacity(next_prompt.len());
        let snapshot = kv_cache.shallow_snapshot();
        assert_eq!(snapshot.keys_values(1, 0).unwrap().len(), 2);
        kv_cache
            .record_kv(2, 0, vec![5.0, 5.1], vec![6.0, 6.1])
            .unwrap();
        assert_eq!(snapshot.keys_values(2, 0).unwrap().len(), 2);
    }

    #[test]
    fn recurrent_layer_state_recording_rejects_gpu_placement_without_fallback() {
        let mut cache = KvCache::new(2, 2);

        cache
            .record_recurrent_layer_state(FlashMoeRecurrentLayerState::cpu_visible(1, 0, 99))
            .unwrap();

        let err = cache
            .record_recurrent_layer_state(FlashMoeRecurrentLayerState::new(
                1,
                0,
                99,
                FlashMoeStatePlacement::GpuResident,
            ))
            .unwrap_err();
        assert!(
            err.to_string().contains("requires CpuVisible placement"),
            "{err:#}"
        );
    }

    #[test]
    fn session_prefix_reuse_preserves_cpu_kv_boundaries() {
        let cached_tokens = vec![10, 20];
        let mut cache = KvCache::new(2, 2);
        for (position, token) in cached_tokens.iter().copied().enumerate() {
            cache.record_prompt_token(position, token).unwrap();
        }
        cache
            .record_kv(0, 0, vec![1.0, 1.1], vec![2.0, 2.1])
            .unwrap();
        cache
            .record_kv(1, 0, vec![3.0, 3.1], vec![4.0, 4.1])
            .unwrap();

        let session_state = FlashMoeSessionState {
            tokens: cached_tokens,
            kv_cache: cache,
            last_hidden: vec![9.0, 9.1],
        };
        let next_prompt = [10, 20, 30];
        let prefix_len = reusable_session_prefix_len(&session_state.tokens, &next_prompt).unwrap();
        assert_eq!(prefix_len, session_state.tokens.len());

        let mut reused = session_state.kv_cache.shallow_snapshot();
        reused.resize_capacity(next_prompt.len());
        assert_eq!(reused.keys_values(1, 0).unwrap().len(), prefix_len);
        assert_eq!(reused.keys_values(2, 0).unwrap().len(), prefix_len);
        assert_eq!(session_state.last_hidden, vec![9.0, 9.1]);

        assert_eq!(
            reusable_session_prefix_len(&session_state.tokens, &[10]),
            None
        );
        assert_eq!(
            reusable_session_prefix_len(&session_state.tokens, &[10, 99, 30]),
            None
        );
    }

    #[test]
    fn validate_accepts_hybrid_gated_deltanet_manifest() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":4,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":4}"#,
        )
        .unwrap();
        let mut slot = 0usize;
        let mut tensors = Vec::new();
        let mut push = |name: String, shape: Vec<usize>| {
            tensors.push(make_dense_ref(&name, shape, slot));
            slot += 1;
        };
        push(
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        );
        push("model.norm.weight".to_string(), vec![config.hidden_size]);
        push(
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        );
        let head_dim = config.hidden_size / config.num_attention_heads;
        let kv_width = config.kv_heads() * head_dim;
        for layer in 0..config.num_hidden_layers {
            push(
                layer_norm_tensor_name(layer, "input_layernorm"),
                vec![config.hidden_size],
            );
            push(
                layer_norm_tensor_name(layer, "post_attention_layernorm"),
                vec![config.hidden_size],
            );
            push(
                router_tensor_name(layer),
                vec![config.experts(), config.hidden_size],
            );
            if is_full_attention_layer(layer) {
                push(
                    attention_tensor_name(layer, "q_proj"),
                    vec![config.hidden_size, config.hidden_size],
                );
                push(
                    attention_tensor_name(layer, "k_proj"),
                    vec![kv_width, config.hidden_size],
                );
                push(
                    attention_tensor_name(layer, "v_proj"),
                    vec![kv_width, config.hidden_size],
                );
                push(
                    attention_tensor_name(layer, "o_proj"),
                    vec![config.hidden_size, config.hidden_size],
                );
                push(
                    layer_norm_tensor_name(layer, "self_attn.q_norm"),
                    vec![head_dim],
                );
                push(
                    layer_norm_tensor_name(layer, "self_attn.k_norm"),
                    vec![head_dim],
                );
            } else {
                push(
                    linear_attention_tensor_name(layer, "in_proj_qkv"),
                    vec![LINEAR_CONV_DIM, config.hidden_size],
                );
                push(
                    linear_attention_tensor_name(layer, "in_proj_z"),
                    vec![LINEAR_TOTAL_VALUE, config.hidden_size],
                );
                push(
                    linear_attention_tensor_name(layer, "in_proj_b"),
                    vec![LINEAR_NUM_V_HEADS, config.hidden_size],
                );
                push(
                    linear_attention_tensor_name(layer, "in_proj_a"),
                    vec![LINEAR_NUM_V_HEADS, config.hidden_size],
                );
                push(
                    linear_attention_tensor_name(layer, "conv1d"),
                    vec![LINEAR_CONV_DIM, CONV_KERNEL_SIZE],
                );
                push(
                    linear_attention_scalar_tensor_name(layer, "A_log"),
                    vec![LINEAR_NUM_V_HEADS],
                );
                push(
                    linear_attention_scalar_tensor_name(layer, "dt_bias"),
                    vec![LINEAR_NUM_V_HEADS],
                );
                push(
                    linear_attention_tensor_name(layer, "norm"),
                    vec![LINEAR_VALUE_DIM],
                );
                push(
                    linear_attention_tensor_name(layer, "out_proj"),
                    vec![config.hidden_size, LINEAR_TOTAL_VALUE],
                );
            }
        }
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["hybrid.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: tensors,
        };
        let registry = TensorRegistry::from_manifest(&manifest);
        validate_required_tensor_manifest(&config, &registry)
            .expect("hybrid manifest should validate for GatedDeltaNet/full-attn layer mix");
    }

    #[test]
    fn validate_accepts_hf_conv1d_singleton_axis_shape() {
        let tensor_name = linear_attention_tensor_name(0, "conv1d");
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["hybrid.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: tensor_name.clone(),
                shard: "hybrid.safetensors".to_string(),
                dtype: "BF16".to_string(),
                shape: vec![LINEAR_CONV_DIM, 1, CONV_KERNEL_SIZE],
                source_offsets: [0, 0],
                runtime_offset: 0,
                byte_len: 0,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };
        let registry = TensorRegistry::from_manifest(&manifest);

        assert_eq!(
            require_conv1d_tensor_shape(&registry, &tensor_name)
                .expect("HF conv1d [channels, 1, kernel] shape should validate"),
            (LINEAR_CONV_DIM, CONV_KERNEL_SIZE)
        );
    }

    #[test]
    fn linear_attention_layout_infers_non_qwen35_dimensions() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":128,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();
        let mut slot = 0usize;
        let mut tensors = Vec::new();
        let mut push = |name: String, shape: Vec<usize>| {
            tensors.push(make_dense_ref(&name, shape, slot));
            slot += 1;
        };
        push(
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        );
        push("model.norm.weight".to_string(), vec![config.hidden_size]);
        push(
            layer_norm_tensor_name(0, "input_layernorm"),
            vec![config.hidden_size],
        );
        push(
            layer_norm_tensor_name(0, "post_attention_layernorm"),
            vec![config.hidden_size],
        );
        push(
            router_tensor_name(0),
            vec![config.experts(), config.hidden_size],
        );
        push(
            linear_attention_tensor_name(0, "in_proj_qkv"),
            vec![12, config.hidden_size],
        );
        push(
            linear_attention_tensor_name(0, "in_proj_z"),
            vec![4, config.hidden_size],
        );
        push(
            linear_attention_tensor_name(0, "in_proj_b"),
            vec![2, config.hidden_size],
        );
        push(
            linear_attention_tensor_name(0, "in_proj_a"),
            vec![2, config.hidden_size],
        );
        push(linear_attention_tensor_name(0, "conv1d"), vec![12, 3]);
        push(linear_attention_scalar_tensor_name(0, "A_log"), vec![2]);
        push(linear_attention_scalar_tensor_name(0, "dt_bias"), vec![2]);
        push(linear_attention_tensor_name(0, "norm"), vec![2]);
        push(
            linear_attention_tensor_name(0, "out_proj"),
            vec![config.hidden_size, 4],
        );

        let manifest = FlashMoeManifest {
            model: "hf://example/tiny-linear".to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["tiny.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: tensors,
        };
        let registry = TensorRegistry::from_manifest(&manifest);
        validate_required_tensor_manifest(&config, &registry)
            .expect("variable linear attention layout should validate");
        let runtime = DenseTransformerRuntime::from_registry(&config, &registry).unwrap();
        let layout = runtime.linear_attention_layout(0).unwrap();

        assert_eq!(layout.num_value_heads, 2);
        assert_eq!(layout.num_key_heads, 1);
        assert_eq!(layout.key_dim, 4);
        assert_eq!(layout.value_dim, 2);
        assert_eq!(layout.conv_dim, 12);
        assert_eq!(layout.conv_kernel_size, 3);
        assert_eq!(layout.conv_state_len(), 24);
        assert_eq!(layout.ssm_state_len(), 16);
    }

    #[test]
    fn linear_attention_key_dim_uses_qwen35_default_only_for_known_shape() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":4096,"num_attention_heads":31,"num_key_value_heads":1,"vocab_size":128,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":16}"#,
        )
        .unwrap();

        let key_dim = infer_linear_attention_key_dim(
            &config,
            LINEAR_TOTAL_KEY,
            LINEAR_TOTAL_VALUE,
            LINEAR_VALUE_DIM,
        )
        .expect("exact Qwen3.5 linear-attention shape should allow the default key dim");
        assert_eq!(key_dim, LINEAR_KEY_DIM);

        let err = infer_linear_attention_key_dim(
            &config,
            LINEAR_TOTAL_KEY,
            LINEAR_TOTAL_VALUE,
            LINEAR_VALUE_DIM * 32,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("not divisible by config head_dim"),
            "{err:#}"
        );
    }

    #[test]
    fn qwen35_linear_attention_keeps_direct_qkv_projection_order() {
        let qwen35: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_5_moe_text","num_hidden_layers":1,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":2,"vocab_size":248320,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();
        assert!(!qwen35.linear_attention_qkv_projection_requires_reorder());

        let qwen_next: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_next","num_hidden_layers":1,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":2,"vocab_size":248320,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10}"#,
        )
        .unwrap();
        assert!(qwen_next.linear_attention_qkv_projection_requires_reorder());
    }

    #[test]
    fn linear_attention_qk_normalization_matches_qwen35_reference_scaling() {
        let layout = LinearAttentionLayout {
            num_value_heads: 4,
            num_key_heads: 2,
            key_dim: 4,
            value_dim: 3,
            total_key_width: 8,
            total_value_width: 12,
            conv_dim: 28,
            conv_kernel_size: 4,
        };
        let mut q = vec![1.0, 2.0, -3.0, 4.0, -1e-6, 2e-6, -3e-6, 4e-6];
        let mut k = vec![0.5, -1.5, 2.5, -3.5, 4e-6, -2e-6, 1e-6, 0.5e-6];
        let mut expected_q = q.clone();
        let mut expected_k = k.clone();

        for head in 0..layout.num_key_heads {
            let start = head * layout.key_dim;
            let end = start + layout.key_dim;
            let q_sum_sq = expected_q[start..end]
                .iter()
                .map(|value| value * value)
                .sum::<f32>();
            let q_inv_rms = (q_sum_sq / layout.key_dim as f32 + 1e-6).sqrt().recip();
            let k_sum_sq = expected_k[start..end]
                .iter()
                .map(|value| value * value)
                .sum::<f32>();
            let k_inv_rms = (k_sum_sq / layout.key_dim as f32 + 1e-6).sqrt().recip();
            let inv_scale = 1.0f32 / (layout.key_dim as f32).sqrt();
            for value in &mut expected_q[start..end] {
                *value *= q_inv_rms * inv_scale * inv_scale;
            }
            for value in &mut expected_k[start..end] {
                *value *= k_inv_rms * inv_scale;
            }
        }

        normalize_linear_attention_qk_in_place(layout, &mut q, &mut k).unwrap();

        for (actual, expected) in q.iter().zip(expected_q.iter()) {
            assert_close(*actual, *expected);
        }
        for (actual, expected) in k.iter().zip(expected_k.iter()) {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn linear_attention_qkv_projection_reorders_key_head_groups_for_conv() {
        let layout = LinearAttentionLayout {
            num_value_heads: 4,
            num_key_heads: 2,
            key_dim: 2,
            value_dim: 3,
            total_key_width: 4,
            total_value_width: 12,
            conv_dim: 20,
            conv_kernel_size: 4,
        };
        let mut qkv = vec![
            10.0, 11.0, // head 0 q
            20.0, 21.0, // head 0 k
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, // head 0 value heads
            40.0, 41.0, // head 1 q
            50.0, 51.0, // head 1 k
            60.0, 61.0, 62.0, 63.0, 64.0, 65.0, // head 1 value heads
        ];

        reorder_grouped_linear_qkv_projection(&mut qkv, layout).unwrap();

        assert_eq!(
            qkv,
            vec![
                10.0, 11.0, 40.0, 41.0, // all q
                20.0, 21.0, 50.0, 51.0, // all k
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0,
            ]
        );
    }

    #[test]
    fn conv1d_step_matches_reference_causal_conv1d_state_order() {
        let channels = 2usize;
        let kernel_size = 4usize;
        // State is chronological: [oldest token channels, ..., newest token channels].
        let conv_state = vec![
            1.0, 10.0, // t - 3
            2.0, 20.0, // t - 2
            3.0, 30.0, // t - 1
        ];
        let new_input = vec![4.0, 40.0];
        // Per-channel PyTorch Conv1d cross-correlation weights. With left causal
        // padding, weight[0] multiplies the oldest context and weight[K-1] the
        // current input for the current output position.
        let weight = vec![
            0.1, 0.2, 0.3, 0.4, // channel 0
            -0.2, 0.05, 0.15, -0.1, // channel 1
        ];
        let mut got = vec![0.0; channels];

        conv1d_step(
            &conv_state,
            &new_input,
            &weight,
            &mut got,
            channels,
            kernel_size,
        );

        let expected0 = silu(1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3 + 4.0 * 0.4);
        let expected1 = silu(10.0 * -0.2 + 20.0 * 0.05 + 30.0 * 0.15 + 40.0 * -0.1);
        assert!(
            (got[0] - expected0).abs() < 1e-6,
            "{} != {expected0}",
            got[0]
        );
        assert!(
            (got[1] - expected1).abs() < 1e-6,
            "{} != {expected1}",
            got[1]
        );
    }

    #[test]
    fn gated_delta_recurrence_matches_qwen3next_recurrent_reference() {
        let layout = LinearAttentionLayout {
            num_value_heads: 4,
            num_key_heads: 2,
            key_dim: 2,
            value_dim: 3,
            total_key_width: 4,
            total_value_width: 12,
            conv_dim: 10,
            conv_kernel_size: 2,
        };
        let mut reference_state_key_major = vec![
            vec![0.10, 0.20, 0.30, 0.40, 0.50, 0.60], // value head 0: [key_dim, value_dim]
            vec![-0.10, 0.05, 0.15, 0.25, -0.20, 0.35],
            vec![0.30, -0.40, 0.10, 0.05, 0.07, -0.02],
            vec![-0.15, 0.25, -0.35, 0.45, -0.55, 0.65],
        ];
        let mut state = vec![0.0f32; layout.ssm_state_len()];
        for vh in 0..layout.num_value_heads {
            let base = vh * layout.value_dim * layout.key_dim;
            for key_idx in 0..layout.key_dim {
                for value_idx in 0..layout.value_dim {
                    state[base + value_idx * layout.key_dim + key_idx] =
                        reference_state_key_major[vh][key_idx * layout.value_dim + value_idx];
                }
            }
        }

        let lin_q = vec![0.7, -0.2, 0.3, 0.6];
        let lin_k = vec![0.4, 0.1, -0.5, 0.8];
        let lin_v = vec![
            0.2, -0.1, 0.6, 0.05, 0.4, -0.3, 0.9, 0.8, -0.2, -0.6, 0.3, 0.7,
        ];
        let alpha = vec![0.1f32, -0.3, 0.7, -0.2];
        let beta = vec![0.2f32, 0.5, -0.4, 0.8];
        let a_log = vec![-0.2f32, 0.4, 0.1, -0.5];
        let dt_bias = vec![0.05f32, -0.15, 0.2, -0.1];
        let mut out = vec![0.0; layout.total_value_width];
        let mut expected_out = vec![0.0; layout.total_value_width];
        let heads_per_key = layout.value_heads_per_key_head();

        for vh in 0..layout.num_value_heads {
            let key_head = vh / heads_per_key;
            let q = &lin_q[key_head * layout.key_dim..key_head * layout.key_dim + layout.key_dim];
            let k = &lin_k[key_head * layout.key_dim..key_head * layout.key_dim + layout.key_dim];
            let v = &lin_v[vh * layout.value_dim..vh * layout.value_dim + layout.value_dim];
            let a_weight = a_log[vh].exp();
            let softplus = (1.0 + (alpha[vh] + dt_bias[vh]).exp()).ln();
            let decay = (-a_weight * softplus).exp();
            let beta_gate = 1.0 / (1.0 + (-beta[vh]).exp());
            let state = &mut reference_state_key_major[vh];
            for key_idx in 0..layout.key_dim {
                for value_idx in 0..layout.value_dim {
                    state[key_idx * layout.value_dim + value_idx] *= decay;
                }
            }
            let mut kv_mem = vec![0.0f32; layout.value_dim];
            for value_idx in 0..layout.value_dim {
                for key_idx in 0..layout.key_dim {
                    kv_mem[value_idx] += state[key_idx * layout.value_dim + value_idx] * k[key_idx];
                }
            }
            for key_idx in 0..layout.key_dim {
                for value_idx in 0..layout.value_dim {
                    let delta = (v[value_idx] - kv_mem[value_idx]) * beta_gate;
                    state[key_idx * layout.value_dim + value_idx] += k[key_idx] * delta;
                }
            }
            for value_idx in 0..layout.value_dim {
                for key_idx in 0..layout.key_dim {
                    expected_out[vh * layout.value_dim + value_idx] +=
                        state[key_idx * layout.value_dim + value_idx] * q[key_idx];
                }
            }
        }

        apply_gated_delta_recurrence(
            layout, &mut state, &lin_q, &lin_k, &lin_v, &alpha, &beta, &a_log, &dt_bias, &mut out,
        );

        for (got, expected) in out.iter().zip(expected_out.iter()) {
            assert!((got - expected).abs() < 1e-5, "{got} != {expected}");
        }
        for vh in 0..layout.num_value_heads {
            let base = vh * layout.value_dim * layout.key_dim;
            for key_idx in 0..layout.key_dim {
                for value_idx in 0..layout.value_dim {
                    let got = state[base + value_idx * layout.key_dim + key_idx];
                    let expected =
                        reference_state_key_major[vh][key_idx * layout.value_dim + value_idx];
                    assert!((got - expected).abs() < 1e-5, "{got} != {expected}");
                }
            }
        }
    }

    #[test]
    fn validate_accepts_tied_lm_head() {
        // lm_head.weight absent → tied embeddings; validator should pass.
        let (config, manifest) = minimal_dense_manifest(false);
        let registry = TensorRegistry::from_manifest(&manifest);
        validate_required_tensor_manifest(&config, &registry)
            .expect("tied-embedding manifest should pass validation");
    }

    #[test]
    fn validate_accepts_separate_lm_head() {
        // lm_head.weight present with correct shape → should pass.
        let (config, manifest) = minimal_dense_manifest(true);
        let registry = TensorRegistry::from_manifest(&manifest);
        validate_required_tensor_manifest(&config, &registry)
            .expect("manifest with separate lm_head should pass validation");
    }

    #[test]
    fn dense_registry_validation_accepts_native_mlx_q4_dense_tensors() {
        let (config, mut manifest) = minimal_dense_manifest(true);
        let embed = manifest
            .dense_tensors
            .iter_mut()
            .find(|tensor| tensor.tensor == "model.embed_tokens.weight")
            .expect("minimal manifest should include embeddings");
        embed.dtype = "U32".to_string();
        embed.quantization = TensorQuantization::Q4 {
            group_size: GROUP_SIZE,
            format: DENSE_Q4_MLX_FORMAT.to_string(),
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
        };
        let layout = dense_q4_layout_with_scale_bias_dtype(
            &embed.shape,
            GROUP_SIZE,
            EXPERT_SCALE_BIAS_DTYPE_BF16,
        )
        .unwrap();
        embed.byte_len = layout.total_bytes as u64;

        let registry = TensorRegistry::from_manifest(&manifest);

        validate_required_tensor_manifest(&config, &registry)
            .expect("native MLX q4 dense tensors should validate by quantization metadata");
    }

    #[test]
    fn validate_rejects_misshapen_lm_head() {
        let (config, mut manifest) = minimal_dense_manifest(true);
        // Corrupt the lm_head shape so it has wrong dimensions.
        for t in &mut manifest.dense_tensors {
            if t.tensor == "lm_head.weight" {
                t.shape = vec![128, 16]; // should be [128, 8]
            }
        }
        let registry = TensorRegistry::from_manifest(&manifest);
        let err = validate_required_tensor_manifest(&config, &registry).unwrap_err();
        assert!(
            err.to_string().contains("lm_head.weight"),
            "expected lm_head shape error, got: {err:#}"
        );
        assert!(
            err.to_string().contains("expected"),
            "expected shape mismatch message, got: {err:#}"
        );
    }

    #[test]
    fn validate_accepts_mlx_conv1d_trailing_singleton_axis_shape() {
        let tensor_name = linear_attention_tensor_name(0, "conv1d");
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["hybrid.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: tensor_name.clone(),
                shard: "hybrid.safetensors".to_string(),
                dtype: "BF16".to_string(),
                shape: vec![LINEAR_CONV_DIM, CONV_KERNEL_SIZE, 1],
                source_offsets: [0, 0],
                runtime_offset: 0,
                byte_len: 0,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };
        let registry = TensorRegistry::from_manifest(&manifest);

        assert_eq!(
            require_conv1d_tensor_shape(&registry, &tensor_name)
                .expect("MLX conv1d [channels, kernel, 1] shape should validate"),
            (LINEAR_CONV_DIM, CONV_KERNEL_SIZE)
        );
    }

    #[test]
    fn validate_accepts_expert_tensors_absent_from_registry() {
        // Expert tensors are packed into ExpertSlotStore files and need not all appear in the
        // dense registry.  The validator must not reject a registry that has no expert entries.
        let (config, manifest) = minimal_dense_manifest(false);
        assert!(manifest.expert_tensors.is_empty());
        let registry = TensorRegistry::from_manifest(&manifest);
        validate_required_tensor_manifest(&config, &registry)
            .expect("registry without expert tensors should still pass dense validation");
    }

    #[test]
    fn tensor_registry_aliases_qwen35_language_model_prefix() {
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: "model.language_model.embed_tokens.weight".to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: "BF16".to_string(),
                shape: vec![248320, 4096],
                source_offsets: [0, 0],
                runtime_offset: 0,
                byte_len: 0,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };
        let registry = TensorRegistry::from_manifest(&manifest);

        assert!(
            registry
                .tensor("model.language_model.embed_tokens.weight")
                .is_some()
        );
        assert!(registry.tensor("model.embed_tokens.weight").is_some());
    }

    #[test]
    fn qwen35_hf_tensor_names_are_canonicalized_for_runtime() {
        assert_eq!(
            canonical_hf_tensor_name("model.language_model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            canonical_hf_tensor_name("model.language_model.layers.7.self_attn.q_proj.weight"),
            "model.layers.7.self_attn.q_proj.weight"
        );
        assert_eq!(
            canonical_hf_tensor_name("language_model.model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            canonical_hf_tensor_name("language_model.model.layers.3.self_attn.q_proj.weight"),
            "model.layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            canonical_hf_tensor_name("language_model.lm_head.weight"),
            "lm_head.weight"
        );
        assert_eq!(
            canonical_hf_tensor_name("model.visual.patch_embed.proj.weight"),
            "visual.patch_embed.proj.weight"
        );
        assert_eq!(canonical_hf_tensor_name("lm_head.weight"), "lm_head.weight");
    }

    #[test]
    fn qwen_config_deserializes_qwen3_moe_extra_fields() {
        // Real Qwen3 MoE checkpoints include additional config fields that should be parsed
        // without error and reflected in the struct.
        let json = br#"{
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_hidden_layers": 60,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "torch_dtype": "bfloat16",
            "num_experts": 512,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1536,
            "tie_word_embeddings": false,
            "num_shared_experts": 1,
            "shared_expert_intermediate_size": 1536
        }"#;
        let config: QwenModelConfig = serde_json::from_slice(json).unwrap();
        assert_eq!(config.tie_word_embeddings, Some(false));
        assert_eq!(config.num_shared_experts, Some(1));
        assert_eq!(config.shared_expert_intermediate_size, Some(1536));
        assert_eq!(config.experts(), 512);
        config.validate().unwrap();
    }

    #[test]
    fn qwen_config_deserializes_qwen35_nested_text_and_vision_fields() {
        let json = br#"{
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "image_token_id": 248056,
            "model_type": "qwen3_5_moe",
            "text_config": {
                "dtype": "bfloat16",
                "head_dim": 256,
                "hidden_size": 4096,
                "max_position_embeddings": 262144,
                "model_type": "qwen3_5_moe_text",
                "moe_intermediate_size": 1024,
                "num_attention_heads": 32,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "num_hidden_layers": 60,
                "num_key_value_heads": 2,
                "shared_expert_intermediate_size": 1024,
                "vocab_size": 248320,
                "rope_parameters": {
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25
                }
            },
            "tie_word_embeddings": false,
            "vision_config": {
                "depth": 27,
                "deepstack_visual_indexes": [5, 11, 17],
                "hidden_size": 1152,
                "in_channels": 3,
                "intermediate_size": 4304,
                "num_heads": 16,
                "out_hidden_size": 4096,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            },
            "vision_end_token_id": 248054,
            "vision_start_token_id": 248053
        }"#;

        let config: QwenModelConfig = serde_json::from_slice(json).unwrap();
        assert_eq!(config.num_hidden_layers, 60);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.head_dim, Some(256));
        assert_eq!(config.full_attention_head_dim(), 256);
        assert_eq!(config.derived_attention_head_dim(), 128);
        assert_eq!(config.num_key_value_heads, Some(2));
        assert_eq!(config.vocab_size, 248320);
        assert_eq!(config.rope_theta, Some(10000000.0));
        assert_eq!(config.partial_rotary_factor, Some(0.25));
        assert_eq!(config.torch_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(config.num_experts_per_tok, Some(10));
        assert_eq!(config.tie_word_embeddings, Some(false));

        let vision = config.vision_config.as_ref().unwrap();
        assert_eq!(vision.depth, 27);
        assert_eq!(vision.embed_dim, 1152);
        assert_eq!(vision.num_heads, 16);
        assert_eq!(vision.patch_size, 16);
        assert_eq!(vision.merge_size, 2);
        assert_eq!(vision.temporal_patch_size, 2);
        assert_eq!(vision.in_chans, 3);
        assert_eq!(vision.deepstack_visual_indexes, vec![5, 11, 17]);
        assert_eq!(vision.out_hidden_size, Some(4096));
        assert_eq!(vision.patch_flat_dim(), 3 * 2 * 16 * 16);
        assert_eq!(vision.mlp_hidden_size(), 4304);

        config.validate().unwrap();
    }

    #[test]
    fn qwen_config_deserializes_mrope_section_from_rope_scaling() {
        let json = br#"{
            "model_type": "qwen3_vl",
            "text_config": {
                "hidden_size": 128,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "vocab_size": 1024,
                "rope_scaling": {
                    "rope_theta": 1000000.0,
                    "mrope_section": [24, 20, 20]
                }
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 64,
                "num_heads": 4
            }
        }"#;

        let config: QwenModelConfig = serde_json::from_slice(json).unwrap();
        assert_eq!(config.rope_theta, Some(1_000_000.0));
        assert_eq!(config.mrope_section, Some(DEFAULT_MROPE_SECTION));
        assert_eq!(config.text_mrope_section(), Some(DEFAULT_MROPE_SECTION));
        config.validate().unwrap();
    }

    #[test]
    fn qwen3vl_config_rejects_out_of_range_deepstack_index() {
        let json = br#"{
            "model_type": "qwen3_vl",
            "text_config": {
                "hidden_size": 128,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 1024
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 64,
                "num_heads": 4,
                "deepstack_visual_indexes": [0, 2]
            }
        }"#;

        let config: QwenModelConfig = serde_json::from_slice(json).unwrap();
        let err = config.validate().unwrap_err();
        assert!(
            err.to_string().contains("deepstack_visual_indexes"),
            "expected deepstack bounds error, got: {err:#}"
        );
    }

    #[test]
    fn qwen3vl_single_image_placeholder_is_expanded_in_place() {
        assert_eq!(
            expand_single_image_placeholders(vec![1, 9, 2], 7, 8, 9, 4).unwrap(),
            vec![1, 7, 9, 9, 9, 9, 8, 2]
        );
        assert_eq!(
            expand_single_image_placeholders(vec![1, 7, 9, 9, 8, 2], 7, 8, 9, 2).unwrap(),
            vec![1, 7, 9, 9, 8, 2]
        );
        assert_eq!(
            expand_single_image_placeholders(vec![1, 9, 9, 2], 7, 8, 9, 2).unwrap(),
            vec![1, 7, 9, 9, 8, 2]
        );
        assert!(expand_single_image_placeholders(vec![1, 2], 7, 8, 9, 2).is_err());
        assert!(expand_single_image_placeholders(vec![1, 9, 2, 9], 7, 8, 9, 2).is_err());
        assert!(expand_single_image_placeholders(vec![1, 7, 9, 2], 7, 8, 9, 2).is_err());
        assert!(qwen3vl_single_image_mrope_positions(&[1, 9, 2, 9], 9, 1, 2).is_err());
    }

    #[test]
    fn qwen3vl_placeholder_expansion_handles_explicit_and_implicit_spans() {
        let expanded = expand_multimodal_image_placeholders(
            vec![1, 7, 9, 9, 9, 9, 8, 2, 9, 3],
            7,
            8,
            9,
            &[
                ImagePlaceholderSpec {
                    token_count: 4,
                    grid_h: 2,
                    grid_w: 2,
                },
                ImagePlaceholderSpec {
                    token_count: 2,
                    grid_h: 1,
                    grid_w: 2,
                },
            ],
        )
        .unwrap();

        assert_eq!(expanded.tokens, vec![1, 7, 9, 9, 9, 9, 8, 2, 7, 9, 9, 8, 3]);
        assert_eq!(
            expanded.visual_spans,
            vec![
                VisualTokenSpan::image(2, 6, 2, 2),
                VisualTokenSpan::image(9, 11, 1, 2),
            ]
        );
    }

    #[test]
    fn qwen3vl_placeholder_expansion_rejects_clear_mismatches() {
        let err = expand_multimodal_image_placeholders(
            vec![1, 9, 2],
            7,
            8,
            9,
            &[ImagePlaceholderSpec {
                token_count: 5,
                grid_h: 2,
                grid_w: 3,
            }],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("image 0 visual token count 5 does not match merged grid 2x3 (6 tokens)"),
            "{err:#}"
        );

        let err = expand_multimodal_image_placeholders(
            vec![1, 7, 9, 9, 9, 8, 2],
            7,
            8,
            9,
            &[ImagePlaceholderSpec {
                token_count: 4,
                grid_h: 2,
                grid_w: 2,
            }],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("image 0 placeholder span contains 3 <|image_pad|> tokens but the encoded image produced 4 visual tokens; use one placeholder for implicit expansion or exactly one per visual token"),
            "{err:#}"
        );

        let err = expand_multimodal_image_placeholders(
            vec![1, 7, 9, 2],
            7,
            8,
            9,
            &[ImagePlaceholderSpec {
                token_count: 2,
                grid_h: 1,
                grid_w: 2,
            }],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("must be wrapped by both <|vision_start|> and <|vision_end|>"),
            "{err:#}"
        );

        let err = qwen3vl_multimodal_mrope_positions(
            &[9, 9, 9],
            9,
            &[VisualTokenSpan::image(0, 3, 2, 2)],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("image span 0 has 3 placeholder tokens but grid 2x2 requires 4"),
            "{err:#}"
        );

        let err =
            qwen3vl_multimodal_mrope_positions(&[9, 1], 9, &[VisualTokenSpan::image(0, 2, 1, 2)])
                .unwrap_err();
        assert!(
            err.to_string()
                .contains("image placeholder count 1 does not match expected visual token count 2"),
            "{err:#}"
        );
    }

    fn expand_and_position_for_test(
        tokens: Vec<u32>,
        image_specs: &[ImagePlaceholderSpec],
    ) -> (ExpandedVisionPrompt, Vec<MropePosition>, usize) {
        let expanded = expand_multimodal_image_placeholders(tokens, 7, 8, 9, image_specs).unwrap();
        let (positions, next_position) =
            qwen3vl_multimodal_mrope_positions(&expanded.tokens, 9, &expanded.visual_spans)
                .unwrap();
        (expanded, positions, next_position)
    }

    #[test]
    fn qwen3vl_text_before_image_gets_own_visual_span() {
        let (expanded, positions, next_position) = expand_and_position_for_test(
            vec![1, 9],
            &[ImagePlaceholderSpec {
                token_count: 4,
                grid_h: 2,
                grid_w: 2,
            }],
        );

        assert_eq!(expanded.tokens, vec![1, 7, 9, 9, 9, 9, 8]);
        assert_eq!(
            expanded.visual_spans,
            vec![VisualTokenSpan::image(2, 6, 2, 2)]
        );
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(positions[1], MropePosition::text(1));
        assert_eq!(
            &positions[2..6],
            &[
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 2,
                },
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 3,
                },
                MropePosition {
                    temporal: 2,
                    height: 3,
                    width: 2,
                },
                MropePosition {
                    temporal: 2,
                    height: 3,
                    width: 3,
                },
            ]
        );
        assert_eq!(positions[6], MropePosition::text(4));
        assert_eq!(next_position, 5);
    }

    #[test]
    fn qwen3vl_image_before_text_gets_own_visual_span() {
        let (expanded, positions, next_position) = expand_and_position_for_test(
            vec![9, 2],
            &[ImagePlaceholderSpec {
                token_count: 2,
                grid_h: 1,
                grid_w: 2,
            }],
        );

        assert_eq!(expanded.tokens, vec![7, 9, 9, 8, 2]);
        assert_eq!(
            expanded.visual_spans,
            vec![VisualTokenSpan::image(1, 3, 1, 2)]
        );
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(
            &positions[1..3],
            &[
                MropePosition {
                    temporal: 1,
                    height: 1,
                    width: 1,
                },
                MropePosition {
                    temporal: 1,
                    height: 1,
                    width: 2,
                },
            ]
        );
        assert_eq!(positions[3], MropePosition::text(3));
        assert_eq!(positions[4], MropePosition::text(4));
        assert_eq!(next_position, 5);
    }

    #[test]
    fn qwen3vl_text_image_text_advances_after_visual_grid() {
        let (expanded, positions, next_position) = expand_and_position_for_test(
            vec![1, 9, 2],
            &[ImagePlaceholderSpec {
                token_count: 4,
                grid_h: 2,
                grid_w: 2,
            }],
        );

        assert_eq!(expanded.tokens, vec![1, 7, 9, 9, 9, 9, 8, 2]);
        assert_eq!(
            expanded.visual_spans,
            vec![VisualTokenSpan::image(2, 6, 2, 2)]
        );
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(positions[1], MropePosition::text(1));
        assert_eq!(positions[6], MropePosition::text(4));
        assert_eq!(positions[7], MropePosition::text(5));
        assert_eq!(next_position, 6);
    }

    #[test]
    fn qwen3vl_two_images_get_separate_visual_spans() {
        let (expanded, positions, next_position) = expand_and_position_for_test(
            vec![1, 9, 2, 9, 3],
            &[
                ImagePlaceholderSpec {
                    token_count: 2,
                    grid_h: 1,
                    grid_w: 2,
                },
                ImagePlaceholderSpec {
                    token_count: 2,
                    grid_h: 2,
                    grid_w: 1,
                },
            ],
        );

        assert_eq!(expanded.tokens, vec![1, 7, 9, 9, 8, 2, 7, 9, 9, 8, 3]);
        assert_eq!(
            expanded.visual_spans,
            vec![
                VisualTokenSpan::image(2, 4, 1, 2),
                VisualTokenSpan::image(7, 9, 2, 1),
            ]
        );
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(positions[1], MropePosition::text(1));
        assert_eq!(
            &positions[2..4],
            &[
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 2,
                },
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 3,
                },
            ]
        );
        assert_eq!(positions[4], MropePosition::text(4));
        assert_eq!(positions[5], MropePosition::text(5));
        assert_eq!(positions[6], MropePosition::text(6));
        assert_eq!(
            &positions[7..9],
            &[
                MropePosition {
                    temporal: 7,
                    height: 7,
                    width: 7,
                },
                MropePosition {
                    temporal: 7,
                    height: 8,
                    width: 7,
                },
            ]
        );
        assert_eq!(positions[9], MropePosition::text(9));
        assert_eq!(positions[10], MropePosition::text(10));
        assert_eq!(next_position, 11);
    }

    #[test]
    fn qwen3vl_multiple_image_grids_with_different_dimensions_are_positioned() {
        let (expanded, positions, next_position) = expand_and_position_for_test(
            vec![1, 9, 2, 9, 3],
            &[
                ImagePlaceholderSpec {
                    token_count: 6,
                    grid_h: 2,
                    grid_w: 3,
                },
                ImagePlaceholderSpec {
                    token_count: 4,
                    grid_h: 1,
                    grid_w: 4,
                },
            ],
        );

        assert_eq!(
            expanded.tokens,
            vec![1, 7, 9, 9, 9, 9, 9, 9, 8, 2, 7, 9, 9, 9, 9, 8, 3]
        );
        assert_eq!(
            expanded.visual_spans,
            vec![
                VisualTokenSpan::image(2, 8, 2, 3),
                VisualTokenSpan::image(11, 15, 1, 4),
            ]
        );
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(positions[1], MropePosition::text(1));
        assert_eq!(
            &positions[2..8],
            &[
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 2,
                },
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 3,
                },
                MropePosition {
                    temporal: 2,
                    height: 2,
                    width: 4,
                },
                MropePosition {
                    temporal: 2,
                    height: 3,
                    width: 2,
                },
                MropePosition {
                    temporal: 2,
                    height: 3,
                    width: 3,
                },
                MropePosition {
                    temporal: 2,
                    height: 3,
                    width: 4,
                },
            ]
        );
        assert_eq!(positions[8], MropePosition::text(5));
        assert_eq!(positions[9], MropePosition::text(6));
        assert_eq!(positions[10], MropePosition::text(7));
        assert_eq!(
            &positions[11..15],
            &[
                MropePosition {
                    temporal: 8,
                    height: 8,
                    width: 8,
                },
                MropePosition {
                    temporal: 8,
                    height: 8,
                    width: 9,
                },
                MropePosition {
                    temporal: 8,
                    height: 8,
                    width: 10,
                },
                MropePosition {
                    temporal: 8,
                    height: 8,
                    width: 11,
                },
            ]
        );
        assert_eq!(positions[15], MropePosition::text(12));
        assert_eq!(positions[16], MropePosition::text(13));
        assert_eq!(next_position, 14);
    }

    #[test]
    fn qwen3vl_parity_multiple_images_render_expand_and_position() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_qwen3vl_tokenizer_json(),
            Some(test_qwen3vl_tool_tokenizer_config_json()),
        )
        .unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[ChatMessage {
                    role: ChatRole::User,
                    content: ChatMessageContent::Parts(vec![
                        ChatContentPart::Text {
                            text: "describe ".to_string(),
                        },
                        ChatContentPart::Image {
                            image: Some("first.png".to_string()),
                            placeholder_tokens: None,
                        },
                        ChatContentPart::Text {
                            text: " now ".to_string(),
                        },
                        ChatContentPart::Image {
                            image: Some("second.png".to_string()),
                            placeholder_tokens: None,
                        },
                    ]),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                }],
                &[],
                true,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\ndescribe <|vision_start|><|image_pad|><|vision_end|> now <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        );

        let vision_start = tokenizer.token_id("<|vision_start|>").unwrap();
        let vision_end = tokenizer.token_id("<|vision_end|>").unwrap();
        let image_pad = tokenizer.token_id("<|image_pad|>").unwrap();
        let prompt_tokens = tokenizer.encode(&rendered).unwrap();
        assert_eq!(
            token_run_bounds(&prompt_tokens, image_pad),
            vec![(4, 5, 1), (8, 9, 1)]
        );

        let expanded = expand_multimodal_image_placeholders(
            prompt_tokens,
            vision_start,
            vision_end,
            image_pad,
            &[
                ImagePlaceholderSpec {
                    token_count: 4,
                    grid_h: 2,
                    grid_w: 2,
                },
                ImagePlaceholderSpec {
                    token_count: 2,
                    grid_h: 1,
                    grid_w: 2,
                },
            ],
        )
        .unwrap();
        assert_eq!(
            expanded.tokens,
            vec![
                100, 5, 7, 200, 202, 202, 202, 202, 201, 8, 200, 202, 202, 201, 101, 100, 6
            ]
        );
        assert_eq!(
            expanded.visual_spans,
            vec![
                VisualTokenSpan::image(4, 8, 2, 2),
                VisualTokenSpan::image(11, 13, 1, 2),
            ]
        );

        let (positions, next_position) =
            qwen3vl_multimodal_mrope_positions(&expanded.tokens, image_pad, &expanded.visual_spans)
                .unwrap();
        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(positions[3], MropePosition::text(3));
        assert_eq!(
            &positions[4..8],
            &[
                MropePosition {
                    temporal: 4,
                    height: 4,
                    width: 4,
                },
                MropePosition {
                    temporal: 4,
                    height: 4,
                    width: 5,
                },
                MropePosition {
                    temporal: 4,
                    height: 5,
                    width: 4,
                },
                MropePosition {
                    temporal: 4,
                    height: 5,
                    width: 5,
                },
            ]
        );
        assert_eq!(positions[8], MropePosition::text(6));
        assert_eq!(positions[10], MropePosition::text(8));
        assert_eq!(
            &positions[11..13],
            &[
                MropePosition {
                    temporal: 9,
                    height: 9,
                    width: 9,
                },
                MropePosition {
                    temporal: 9,
                    height: 9,
                    width: 10,
                },
            ]
        );
        assert_eq!(positions[16], MropePosition::text(14));
        assert_eq!(next_position, 15);
    }

    #[test]
    fn qwen3vl_smart_resize_obeys_pixel_budget_after_rounding() {
        let preprocessor = ImagePreprocessor::default_qwen3_vl();
        let (h, w) = preprocessor.smart_resize(10_000, 10_000);
        assert_eq!(h % VIT_SPATIAL_MERGE_SIZE as u32, 0);
        assert_eq!(w % VIT_SPATIAL_MERGE_SIZE as u32, 0);
        assert!((h as usize) * (w as usize) <= preprocessor.max_pixels);

        let (small_h, small_w) = preprocessor.smart_resize(1, 1);
        assert!((small_h as usize) * (small_w as usize) >= preprocessor.min_pixels);
    }

    #[test]
    fn qwen3vl_vision_patch_coords_are_block_major() {
        assert_eq!(
            block_major_patch_coords(4, 4, 2),
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 1),
                (2, 2),
                (2, 3),
                (3, 2),
                (3, 3),
            ]
        );
    }

    #[test]
    fn build_cache_accepts_qwen3_style_index_with_qknorm_and_shared_expert() {
        // Fixture derived from the Qwen3 MoE architecture:
        //   - q_norm / k_norm per attention layer (Qwen3 QK-norm)
        //   - shared_expert MLP that is always active and gated by shared_expert_gate
        //   - separate lm_head.weight (tie_word_embeddings=false)
        //   - 4 routable experts per layer
        // All of these tensors should be classified correctly (dense vs expert) and the
        // validator should accept the resulting manifest.
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();

        // config.json with Qwen3-style extra fields
        std::fs::write(
            snapshot.join("config.json"),
            br#"{
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_hidden_layers": 1,
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 300,
                "rope_theta": 1000000.0,
                "torch_dtype": "bfloat16",
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
                "tie_word_embeddings": false,
                "num_shared_experts": 1,
                "shared_expert_intermediate_size": 16
            }"#,
        )
        .unwrap();

        // Dense shard: all non-expert tensors including Qwen3-specific q_norm/k_norm and
        // shared_expert projections.  Shapes are consistent with the config above.
        // kv_width = num_key_value_heads(1) * (hidden_size / num_attention_heads) = 1 * (8/2) = 4
        let dense_shard = make_typed_safetensors(&[
            (
                "model.embed_tokens.weight",
                "BF16",
                vec![300, 8],
                &vec![0u8; 300 * 8 * 2],
            ),
            (
                "lm_head.weight",
                "BF16",
                vec![300, 8],
                &vec![0u8; 300 * 8 * 2],
            ),
            ("model.norm.weight", "BF16", vec![8], &vec![0u8; 8 * 2]),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "BF16",
                vec![8, 8],
                &vec![0u8; 8 * 8 * 2],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "BF16",
                vec![4, 8],
                &vec![0u8; 4 * 8 * 2],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                "BF16",
                vec![4, 8],
                &vec![0u8; 4 * 8 * 2],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "BF16",
                vec![8, 8],
                &vec![0u8; 8 * 8 * 2],
            ),
            // QK-norm tensors present in Qwen3 MoE checkpoints
            (
                "model.layers.0.self_attn.q_norm.weight",
                "BF16",
                vec![4],
                &vec![0u8; 4 * 2],
            ),
            (
                "model.layers.0.self_attn.k_norm.weight",
                "BF16",
                vec![4],
                &vec![0u8; 4 * 2],
            ),
            (
                "model.layers.0.input_layernorm.weight",
                "BF16",
                vec![8],
                &vec![0u8; 8 * 2],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "BF16",
                vec![8],
                &vec![0u8; 8 * 2],
            ),
            (
                "model.layers.0.mlp.gate.weight",
                "BF16",
                vec![4, 8],
                &vec![0u8; 4 * 8 * 2],
            ),
            // Shared expert (always active, not gated): treated as dense, not packed
            (
                "model.layers.0.mlp.shared_expert.gate_proj.weight",
                "BF16",
                vec![16, 8],
                &vec![0u8; 16 * 8 * 2],
            ),
            (
                "model.layers.0.mlp.shared_expert.up_proj.weight",
                "BF16",
                vec![16, 8],
                &vec![0u8; 16 * 8 * 2],
            ),
            (
                "model.layers.0.mlp.shared_expert.down_proj.weight",
                "BF16",
                vec![8, 16],
                &vec![0u8; 8 * 16 * 2],
            ),
            (
                "model.layers.0.mlp.shared_expert_gate.weight",
                "BF16",
                vec![1, 8],
                &vec![0u8; 8 * 2],
            ),
        ]);
        std::fs::write(snapshot.join("dense.safetensors"), dense_shard).unwrap();

        // Expert shard: 4 routed experts, each with gate/up/down projections.
        let mut expert_entries: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = Vec::new();
        let gate_bytes = vec![0u8; 16 * 8 * 2];
        let down_bytes = vec![0u8; 8 * 16 * 2];
        let names: Vec<(String, String, String)> = (0..4)
            .flat_map(|e| {
                let pfx = format!("model.layers.0.mlp.experts.{e}");
                [
                    (
                        format!("{pfx}.gate_proj.weight"),
                        "gate".to_string(),
                        format!("{e}-gate"),
                    ),
                    (
                        format!("{pfx}.up_proj.weight"),
                        "up".to_string(),
                        format!("{e}-up"),
                    ),
                    (
                        format!("{pfx}.down_proj.weight"),
                        "down".to_string(),
                        format!("{e}-down"),
                    ),
                ]
            })
            .collect();
        for (name, proj, _) in &names {
            let (shape, data): (Vec<usize>, &[u8]) = if proj == "down" {
                (vec![8, 16], &down_bytes)
            } else {
                (vec![16, 8], &gate_bytes)
            };
            expert_entries.push((name.as_str(), "BF16", shape, data.to_vec()));
        }
        std::fs::write(
            snapshot.join("expert.safetensors"),
            make_typed_safetensors(
                &expert_entries
                    .iter()
                    .map(|(n, d, s, b)| (*n, *d, s.clone(), b.as_slice()))
                    .collect::<Vec<_>>(),
            ),
        )
        .unwrap();

        // Build weight_map: all tensors → their shard file
        let mut weight_map = serde_json::Map::new();
        // dense tensors
        for name in [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
        ] {
            weight_map.insert(
                name.to_string(),
                serde_json::Value::String("dense.safetensors".to_string()),
            );
        }
        // expert tensors
        for (name, _, _) in &names {
            weight_map.insert(
                name.clone(),
                serde_json::Value::String("expert.safetensors".to_string()),
            );
        }
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::to_string(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();

        let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot)
            .expect("build should succeed for Qwen3-style snapshot with qknorm and shared_expert");

        // Validate: manifest should classify shared_expert and q/k_norm as dense, not expert.
        let manifest: FlashMoeManifest =
            serde_json::from_slice(&std::fs::read(&plan.tensor_manifest).unwrap()).unwrap();
        assert!(
            manifest
                .dense_tensors
                .iter()
                .any(|t| t.tensor.contains("q_norm")),
            "q_norm should be a dense tensor"
        );
        assert!(
            manifest
                .dense_tensors
                .iter()
                .any(|t| t.tensor.contains("k_norm")),
            "k_norm should be a dense tensor"
        );
        assert!(
            manifest
                .dense_tensors
                .iter()
                .any(|t| t.tensor.contains("shared_expert")),
            "shared_expert should be a dense tensor"
        );
        // 4 experts × 3 projections = 12 expert tensor entries
        assert_eq!(manifest.expert_tensors.len(), 12);

        // The validator must accept the resulting registry.
        let config = QwenModelConfig::from_file(&plan.model_config).unwrap();
        let registry = TensorRegistry::load(&plan.tensor_manifest).unwrap();
        validate_required_tensor_manifest(&config, &registry)
            .expect("Qwen3-style manifest should pass validation");
    }

    #[test]
    fn packer_splits_qwen35_aggregate_expert_tensors() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        fs::create_dir_all(&snapshot).unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        fs::create_dir_all(&plan.experts_dir).unwrap();

        let gate_up_bytes: Vec<u8> = (0u8..16).collect();
        let down_bytes: Vec<u8> = (16u8..24).collect();
        fs::write(
            snapshot.join("expert.safetensors"),
            make_typed_safetensors(&[
                (
                    "model.layers.1.mlp.experts.gate_up_proj",
                    "U8",
                    vec![2, 4, 2],
                    &gate_up_bytes,
                ),
                (
                    "model.layers.1.mlp.experts.down_proj",
                    "U8",
                    vec![2, 2, 2],
                    &down_bytes,
                ),
            ]),
        )
        .unwrap();

        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":2,"num_attention_heads":1,"vocab_size":16,"torch_dtype":"bfloat16","num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":2}"#,
        )
        .unwrap();
        let tensors = vec![
            ExpertTensorRef {
                tensor: "model.layers.1.mlp.experts.gate_up_proj".to_string(),
                shard: "expert.safetensors".to_string(),
                layer: Some(1),
                expert: None,
                dtype: Some("U8".to_string()),
                shape: vec![2, 4, 2],
                source_offsets: Some([0, 16]),
                q4_sources: None,
            },
            ExpertTensorRef {
                tensor: "model.layers.1.mlp.experts.down_proj".to_string(),
                shard: "expert.safetensors".to_string(),
                layer: Some(1),
                expert: None,
                dtype: Some("U8".to_string()),
                shape: vec![2, 2, 2],
                source_offsets: Some([16, 24]),
                q4_sources: None,
            },
        ];

        pack_expert_tensors(
            &snapshot,
            ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
            &tensors,
            Some(&config),
        )
        .unwrap();

        let layer_path = expert_layer_path(&plan.experts_dir, 1);
        let metadata = read_expert_layer_pack_metadata(&plan.experts_dir, 1)
            .unwrap()
            .unwrap();
        assert!(layer_path.is_file());
        assert_eq!(
            fs::metadata(&layer_path).unwrap().len(),
            metadata.expert_size * metadata.experts as u64
        );
        assert_eq!(metadata.experts, 2);
        assert_eq!(metadata.packs.len(), 2);
        assert!(metadata.pack_for(0).is_some());
        assert!(metadata.pack_for(1).is_some());

        let expert0 = read_one_expert(&plan.experts_dir, 1, 0).unwrap();
        let expert1 = read_one_expert(&plan.experts_dir, 1, 1).unwrap();
        assert!(expert_pack_is_complete(&plan.experts_dir, 1, 0));
        assert!(expert_pack_is_complete(&plan.experts_dir, 1, 1));
        for expert in [&expert0, &expert1] {
            assert_eq!(expert.records.len(), 3);
            assert!(expert.record_suffix("gate_proj.weight").is_some());
            assert!(expert.record_suffix("up_proj.weight").is_some());
            assert!(expert.record_suffix("down_proj.weight").is_some());
        }
        let input = [1.0, 1.0];
        let expert0_gate = expert0
            .project_record(
                expert0.record_suffix("gate_proj.weight").unwrap(),
                &input,
                2,
            )
            .unwrap()
            .unwrap();
        let expert0_up = expert0
            .project_record(expert0.record_suffix("up_proj.weight").unwrap(), &input, 2)
            .unwrap()
            .unwrap();
        let expert1_gate = expert1
            .project_record(
                expert1.record_suffix("gate_proj.weight").unwrap(),
                &input,
                2,
            )
            .unwrap()
            .unwrap();
        let expert1_down = expert1
            .project_record(
                expert1.record_suffix("down_proj.weight").unwrap(),
                &input,
                2,
            )
            .unwrap()
            .unwrap();
        for (actual, expected) in expert0_gate.iter().zip([1.0, 5.0]) {
            assert_close_with_tolerance(*actual, expected, 0.01);
        }
        for (actual, expected) in expert0_up.iter().zip([9.0, 13.0]) {
            assert_close_with_tolerance(*actual, expected, 0.01);
        }
        for (actual, expected) in expert1_gate.iter().zip([17.0, 21.0]) {
            assert_close_with_tolerance(*actual, expected, 0.01);
        }
        for (actual, expected) in expert1_down.iter().zip([41.0, 45.0]) {
            assert_close_with_tolerance(*actual, expected, 0.01);
        }
    }

    #[test]
    fn packer_splits_mlx_switch_mlp_aggregate_expert_tensors() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        fs::create_dir_all(&snapshot).unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        fs::create_dir_all(&plan.experts_dir).unwrap();

        let gate_bytes: Vec<u8> = (0u8..8).collect();
        let up_bytes: Vec<u8> = (8u8..16).collect();
        let down_bytes: Vec<u8> = (16u8..24).collect();
        fs::write(
            snapshot.join("expert.safetensors"),
            make_typed_safetensors(&[
                (
                    "model.layers.1.mlp.switch_mlp.gate_proj.weight",
                    "U8",
                    vec![2, 2, 2],
                    &gate_bytes,
                ),
                (
                    "model.layers.1.mlp.switch_mlp.up_proj.weight",
                    "U8",
                    vec![2, 2, 2],
                    &up_bytes,
                ),
                (
                    "model.layers.1.mlp.switch_mlp.down_proj.weight",
                    "U8",
                    vec![2, 2, 2],
                    &down_bytes,
                ),
            ]),
        )
        .unwrap();

        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":2,"num_attention_heads":1,"vocab_size":16,"torch_dtype":"bfloat16","num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":2}"#,
        )
        .unwrap();
        let tensors = vec![
            ExpertTensorRef {
                tensor: "model.layers.1.mlp.switch_mlp.gate_proj.weight".to_string(),
                shard: "expert.safetensors".to_string(),
                layer: Some(1),
                expert: None,
                dtype: Some("U8".to_string()),
                shape: vec![2, 2, 2],
                source_offsets: Some([0, 8]),
                q4_sources: None,
            },
            ExpertTensorRef {
                tensor: "model.layers.1.mlp.switch_mlp.up_proj.weight".to_string(),
                shard: "expert.safetensors".to_string(),
                layer: Some(1),
                expert: None,
                dtype: Some("U8".to_string()),
                shape: vec![2, 2, 2],
                source_offsets: Some([8, 16]),
                q4_sources: None,
            },
            ExpertTensorRef {
                tensor: "model.layers.1.mlp.switch_mlp.down_proj.weight".to_string(),
                shard: "expert.safetensors".to_string(),
                layer: Some(1),
                expert: None,
                dtype: Some("U8".to_string()),
                shape: vec![2, 2, 2],
                source_offsets: Some([16, 24]),
                q4_sources: None,
            },
        ];

        pack_expert_tensors(
            &snapshot,
            ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
            &tensors,
            Some(&config),
        )
        .unwrap();

        let metadata = read_expert_layer_pack_metadata(&plan.experts_dir, 1)
            .unwrap()
            .unwrap();
        assert_eq!(metadata.experts, 2);
        assert_eq!(metadata.packs.len(), 2);
        let expert0 = read_one_expert(&plan.experts_dir, 1, 0).unwrap();
        let expert1 = read_one_expert(&plan.experts_dir, 1, 1).unwrap();
        for expert in [&expert0, &expert1] {
            assert_eq!(expert.records.len(), 3);
            assert!(expert.record_suffix("gate_proj.weight").is_some());
            assert!(expert.record_suffix("up_proj.weight").is_some());
            assert!(expert.record_suffix("down_proj.weight").is_some());
        }
        let input = [1.0, 1.0];
        let expert0_gate = expert0
            .project_record(
                expert0.record_suffix("gate_proj.weight").unwrap(),
                &input,
                2,
            )
            .unwrap()
            .unwrap();
        let expert0_up = expert0
            .project_record(expert0.record_suffix("up_proj.weight").unwrap(), &input, 2)
            .unwrap()
            .unwrap();
        let expert1_gate = expert1
            .project_record(
                expert1.record_suffix("gate_proj.weight").unwrap(),
                &input,
                2,
            )
            .unwrap()
            .unwrap();
        let expert1_down = expert1
            .project_record(
                expert1.record_suffix("down_proj.weight").unwrap(),
                &input,
                2,
            )
            .unwrap()
            .unwrap();
        for (actual, expected) in expert0_gate.iter().zip([1.0, 5.0]) {
            assert_close_with_tolerance(*actual, expected, 0.01);
        }
        for (actual, expected) in expert0_up.iter().zip([17.0, 21.0]) {
            assert_close_with_tolerance(*actual, expected, 0.01);
        }
        for (actual, expected) in expert1_gate.iter().zip([9.0, 13.0]) {
            assert_close_with_tolerance(*actual, expected, 0.01);
        }
        for (actual, expected) in expert1_down.iter().zip([41.0, 45.0]) {
            assert_close_with_tolerance(*actual, expected, 0.01);
        }
    }

    #[test]
    fn packer_copies_native_mlx_q4_switch_mlp_experts_without_requantizing() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        fs::create_dir_all(&snapshot).unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        fs::create_dir_all(&plan.experts_dir).unwrap();

        let packed_words: Vec<u32> = (0..16).map(|_| 0x7654_3210).collect();
        let gate_packed = u32_tensor_bytes(&packed_words);
        let up_words: Vec<u32> = (0..16)
            .map(|row| 0x0123_4567u32.wrapping_add(row))
            .collect();
        let up_packed = u32_tensor_bytes(&up_words);
        let down_words: Vec<u32> = (0..16)
            .map(|row| 0x89ab_cdefu32.wrapping_add(row))
            .collect();
        let down_packed = u32_tensor_bytes(&down_words);
        let gate_scales = bf16_tensor_bytes(&[0.5; 16]);
        let gate_biases = bf16_tensor_bytes(&[1.0; 16]);
        let up_scales = bf16_tensor_bytes(&[0.25; 16]);
        let up_biases = bf16_tensor_bytes(&[2.0; 16]);
        let down_scales = bf16_tensor_bytes(&[0.125; 16]);
        let down_biases = bf16_tensor_bytes(&[3.0; 16]);
        let tensors = vec![
            (
                "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight".to_string(),
                "U32".to_string(),
                vec![2, 8, 1],
                gate_packed.clone(),
            ),
            (
                "language_model.model.layers.1.mlp.switch_mlp.gate_proj.scales".to_string(),
                EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                vec![2, 8, 1],
                gate_scales.clone(),
            ),
            (
                "language_model.model.layers.1.mlp.switch_mlp.gate_proj.biases".to_string(),
                EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                vec![2, 8, 1],
                gate_biases.clone(),
            ),
            (
                "language_model.model.layers.1.mlp.switch_mlp.up_proj.weight".to_string(),
                "U32".to_string(),
                vec![2, 8, 1],
                up_packed.clone(),
            ),
            (
                "language_model.model.layers.1.mlp.switch_mlp.up_proj.scales".to_string(),
                EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                vec![2, 8, 1],
                up_scales.clone(),
            ),
            (
                "language_model.model.layers.1.mlp.switch_mlp.up_proj.biases".to_string(),
                EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                vec![2, 8, 1],
                up_biases.clone(),
            ),
            (
                "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight".to_string(),
                "U32".to_string(),
                vec![2, 8, 1],
                down_packed.clone(),
            ),
            (
                "language_model.model.layers.1.mlp.switch_mlp.down_proj.scales".to_string(),
                EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                vec![2, 8, 1],
                down_scales.clone(),
            ),
            (
                "language_model.model.layers.1.mlp.switch_mlp.down_proj.biases".to_string(),
                EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                vec![2, 8, 1],
                down_biases.clone(),
            ),
        ];
        let fixture_refs = typed_fixture_refs(&tensors);
        fs::write(
            snapshot.join("experts.safetensors"),
            make_typed_safetensors(&fixture_refs),
        )
        .unwrap();
        let mut weight_map = serde_json::Map::new();
        for (name, _, _, _) in &tensors {
            weight_map.insert(
                name.clone(),
                serde_json::Value::String("experts.safetensors".to_string()),
            );
        }
        let index = serde_json::Value::Object(serde_json::Map::from_iter([(
            "weight_map".to_string(),
            serde_json::Value::Object(weight_map),
        )]));
        let index_path = snapshot.join("model.safetensors.index.json");
        fs::write(&index_path, index.to_string()).unwrap();

        let (manifest, visual_refs) = build_manifest(QWEN35_MODEL, &snapshot, &index_path).unwrap();
        assert!(visual_refs.is_empty());
        assert!(manifest.dense_tensors.is_empty());
        assert_eq!(manifest.expert_tensors.len(), 3);
        assert!(manifest.expert_tensors.iter().all(|tensor| {
            tensor.q4_sources.is_some()
                && tensor.shape == vec![2, 8, 8]
                && tensor.dtype.as_deref() == Some("U32")
        }));

        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":8,"num_attention_heads":1,"vocab_size":16,"torch_dtype":"bfloat16","num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":8}"#,
        )
        .unwrap();
        pack_expert_tensors(
            &snapshot,
            ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
            &manifest.expert_tensors,
            Some(&config),
        )
        .unwrap();

        let expert0 = read_one_expert(&plan.experts_dir, 1, 0).unwrap();
        let gate0 = expert0.record_suffix("gate_proj.weight").unwrap();
        assert_eq!(gate0.dtype, "U32");
        assert_eq!(gate0.shape, vec![8, 8]);
        assert_eq!(gate0.scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
        assert_eq!(gate0.packed, gate_packed[..32]);
        assert_eq!(gate0.scale_bytes, gate_scales[..16]);
        assert_eq!(gate0.bias_bytes, gate_biases[..16]);
        let input = [1.0; 8];
        let projected = expert0.project_record(gate0, &input, 8).unwrap().unwrap();
        assert_eq!(projected, vec![22.0; 8]);

        let expert1 = read_one_expert(&plan.experts_dir, 1, 1).unwrap();
        let up1 = expert1.record_suffix("up_proj.weight").unwrap();
        assert_eq!(up1.packed, up_packed[32..]);
        assert_eq!(up1.scale_bytes, up_scales[16..]);
        assert_eq!(up1.bias_bytes, up_biases[16..]);
        let down1 = expert1.record_suffix("down_proj.weight").unwrap();
        assert_eq!(down1.packed, down_packed[32..]);
        assert_eq!(down1.scale_bytes, down_scales[16..]);
        assert_eq!(down1.bias_bytes, down_biases[16..]);
    }

    #[test]
    fn native_q4_qwen35_expert_pack_uses_fixed_slot_layout() {
        let fixed = QwenMoeQ4ExpertLayout::qwen35_a17b();
        let native_input = |tensor: &str,
                            shape: Vec<usize>,
                            weight_kind: QwenMoeExpertComponentKind,
                            scale_kind: QwenMoeExpertComponentKind,
                            bias_kind: QwenMoeExpertComponentKind,
                            packed_byte: u8,
                            scale_byte: u8,
                            bias_byte: u8| {
            NativeQ4ExpertRecordInput {
                tensor: tensor.to_string(),
                dtype: "U32".to_string(),
                shape,
                source_offsets: [0, 1],
                source_hash: Some(format!("{tensor}:hash")),
                packed: vec![packed_byte; fixed.component(weight_kind).bytes],
                scale_bytes: vec![scale_byte; fixed.component(scale_kind).bytes],
                bias_bytes: vec![bias_byte; fixed.component(bias_kind).bytes],
                groups: fixed.component(scale_kind).bytes / 2,
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            }
        };

        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_5_moe_text","num_hidden_layers":60,"hidden_size":4096,"num_attention_heads":32,"head_dim":256,"num_key_value_heads":2,"vocab_size":248320,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":1024,"shared_expert_intermediate_size":1024}"#,
        )
        .unwrap();
        let layout = AggregateExpertLayout::new(
            config.experts(),
            config.hidden_size,
            config.moe_intermediate_size.unwrap(),
        )
        .unwrap();
        let (packed, metadata) = build_fixed_native_q4_expert_pack(
            0,
            7,
            fixed,
            vec![
                native_input(
                    "model.layers.0.mlp.experts.7.gate_proj.weight",
                    vec![1024, 4096],
                    QwenMoeExpertComponentKind::GateWeight,
                    QwenMoeExpertComponentKind::GateScale,
                    QwenMoeExpertComponentKind::GateBias,
                    0x11,
                    0x22,
                    0x33,
                ),
                native_input(
                    "model.layers.0.mlp.experts.7.up_proj.weight",
                    vec![1024, 4096],
                    QwenMoeExpertComponentKind::UpWeight,
                    QwenMoeExpertComponentKind::UpScale,
                    QwenMoeExpertComponentKind::UpBias,
                    0x44,
                    0x55,
                    0x66,
                ),
                native_input(
                    "model.layers.0.mlp.experts.7.down_proj.weight",
                    vec![4096, 1024],
                    QwenMoeExpertComponentKind::DownWeight,
                    QwenMoeExpertComponentKind::DownScale,
                    QwenMoeExpertComponentKind::DownBias,
                    0x77,
                    0x88,
                    0x99,
                ),
            ],
        )
        .unwrap();

        assert_eq!(layout.hidden, 4096);
        assert_eq!(layout.intermediate, 1024);
        assert_eq!(packed.len(), fixed.expert_bytes);
        assert!(!packed.starts_with(PBQ4_EXPERT_MAGIC));
        assert_eq!(metadata.layer, 0);
        assert_eq!(metadata.expert, 7);
        assert_eq!(metadata.packed_bytes, fixed.expert_bytes as u64);
        assert_eq!(metadata.records.len(), 3);
        assert_eq!(
            &packed[fixed
                .component(QwenMoeExpertComponentKind::GateWeight)
                .offset
                ..fixed
                    .component(QwenMoeExpertComponentKind::GateWeight)
                    .offset
                    + 4],
            &[0x11; 4]
        );
        assert_eq!(
            &packed[fixed.component(QwenMoeExpertComponentKind::UpScale).offset
                ..fixed.component(QwenMoeExpertComponentKind::UpScale).offset + 4],
            &[0x55; 4]
        );
        assert_eq!(
            &packed[fixed.component(QwenMoeExpertComponentKind::DownBias).offset
                ..fixed.component(QwenMoeExpertComponentKind::DownBias).offset + 4],
            &[0x99; 4]
        );
    }

    #[test]
    fn aggregate_expert_reuse_rejects_changed_source_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        fs::create_dir_all(&snapshot).unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        fs::create_dir_all(&plan.experts_dir).unwrap();

        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":2,"num_attention_heads":1,"vocab_size":16,"torch_dtype":"bfloat16","num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":2}"#,
        )
        .unwrap();
        let tensors = vec![
            ExpertTensorRef {
                tensor: "model.layers.1.mlp.experts.gate_up_proj".to_string(),
                shard: "expert.safetensors".to_string(),
                layer: Some(1),
                expert: None,
                dtype: Some("U8".to_string()),
                shape: vec![2, 4, 2],
                source_offsets: Some([0, 16]),
                q4_sources: None,
            },
            ExpertTensorRef {
                tensor: "model.layers.1.mlp.experts.down_proj".to_string(),
                shard: "expert.safetensors".to_string(),
                layer: Some(1),
                expert: None,
                dtype: Some("U8".to_string()),
                shape: vec![2, 2, 2],
                source_offsets: Some([16, 24]),
                q4_sources: None,
            },
        ];

        let write_expert_shard = |gate_up_bytes: Vec<u8>, down_bytes: Vec<u8>| {
            fs::write(
                snapshot.join("expert.safetensors"),
                make_typed_safetensors(&[
                    (
                        "model.layers.1.mlp.experts.gate_up_proj",
                        "U8",
                        vec![2, 4, 2],
                        gate_up_bytes.as_slice(),
                    ),
                    (
                        "model.layers.1.mlp.experts.down_proj",
                        "U8",
                        vec![2, 2, 2],
                        down_bytes.as_slice(),
                    ),
                ]),
            )
            .unwrap();
        };

        write_expert_shard((0u8..16).collect(), (16u8..24).collect());
        pack_expert_tensors(
            &snapshot,
            ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
            &tensors,
            Some(&config),
        )
        .unwrap();
        let before = read_expert_pack_metadata(&plan.experts_dir, 1, 0)
            .unwrap()
            .unwrap()
            .records[0]
            .source_hash
            .clone()
            .unwrap();

        write_expert_shard((100u8..116).collect(), (200u8..208).collect());
        pack_expert_tensors(
            &snapshot,
            ExpertPackingPolicy::new(&plan.model, &plan.experts_dir, plan.quantization),
            &tensors,
            Some(&config),
        )
        .unwrap();
        let after = read_expert_pack_metadata(&plan.experts_dir, 1, 0)
            .unwrap()
            .unwrap()
            .records[0]
            .source_hash
            .clone()
            .unwrap();

        assert_ne!(before, after);
    }

    #[test]
    fn dense_store_reads_registered_tensor_rows_by_dtype() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("model_weights.bin");
        let manifest_path = tmp.path().join("model_weights.json");
        let mut bytes = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: "model.layers.0.self_attn.q_proj.weight".to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: "F32".to_string(),
                shape: vec![2, 2],
                source_offsets: [0, bytes.len() as u64],
                runtime_offset: 0,
                byte_len: bytes.len() as u64,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let row = store
            .read_tensor_row_f32("model.layers.0.self_attn.q_proj.weight", 1, 2)
            .unwrap()
            .unwrap();
        assert_eq!(row, vec![3.0, 4.0]);
        let tile = store
            .read_tensor_rows_f32("model.layers.0.self_attn.q_proj.weight", 0, 2)
            .unwrap();
        assert_eq!(tile, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            store
                .resident
                .lock()
                .expect("dense tensor cache poisoned")
                .bytes,
            0
        );
        let projected = store
            .project(0, "q_proj", &[1.0, 1.0], 2)
            .expect("registered dense projection should decode F32 weights");
        assert_eq!(projected.len(), 2);
        assert!(projected[1] > projected[0]);
    }

    #[test]
    fn router_scores_use_cached_full_tensor_matvec() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("model_weights.bin");
        let manifest_path = tmp.path().join("model_weights.json");
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: router_tensor_name(0),
                shard: "dense.safetensors".to_string(),
                dtype: "F32".to_string(),
                shape: vec![2, 3],
                source_offsets: [0, bytes.len() as u64],
                runtime_offset: 0,
                byte_len: bytes.len() as u64,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let config = QwenModelConfig {
            model_type: Some("qwen".to_string()),
            architectures: None,
            num_hidden_layers: 1,
            hidden_size: 3,
            num_attention_heads: 1,
            head_dim: None,
            num_key_value_heads: Some(1),
            vocab_size: 16,
            rope_theta: None,
            partial_rotary_factor: None,
            torch_dtype: Some("float32".to_string()),
            num_experts: Some(2),
            num_experts_per_tok: Some(1),
            norm_topk_prob: None,
            moe_intermediate_size: Some(4),
            intermediate_size: None,
            max_position_embeddings: Some(8),
            mrope_section: None,
            tie_word_embeddings: None,
            num_shared_experts: None,
            shared_expert_intermediate_size: None,
            vision_config: None,
        };
        let mut graph_layout = QwenMoeModelLayout::from_config(QWEN35_MODEL, &config).unwrap();
        graph_layout.hidden_size = GROUP_SIZE;
        graph_layout.moe_intermediate_size = GROUP_SIZE;
        let capability_plan = FlashMoeCapabilityPlan::for_model_layout(&graph_layout).unwrap();
        let scheduled_graph = FlashMoeScheduledGraph::from_capabilities(&capability_plan).unwrap();
        let scheduled_routing = scheduled_graph
            .build_routing_topk(0, 2, 1, ScheduledRoutingCandidateSource::CpuRouterScores)
            .unwrap();
        let projection = store.router_score_projection_descriptor(0, 2, 3).unwrap();
        let projection_ref = projection
            .as_ref()
            .expect("registered router tensor should produce a typed projection descriptor");
        assert_eq!(projection_ref.layer, 0);
        assert_eq!(projection_ref.experts, 2);
        assert_eq!(projection_ref.hidden_width, 3);
        assert_eq!(projection_ref.tensor_name, router_tensor_name(0));
        let command = scheduled_routing
            .build_score_projection_command(projection, 3)
            .unwrap();

        let routing_command = store
            .router_command_with_metal(None, command, &[0.5, -1.0, 2.0])
            .unwrap();

        assert_eq!(
            routing_command.source,
            ScheduledRoutingCandidateSource::CpuRouterScores
        );
        assert_eq!(routing_command.layer, 0);
        assert_eq!(routing_command.active_experts, 1);
        assert_eq!(routing_command.routes, vec![(1, 9.0)]);
        assert_eq!(
            store
                .decoded_tensor_tiles
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn dense_store_caches_small_norm_weights() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("model_weights.bin");
        let manifest_path = tmp.path().join("model_weights.json");
        let mut bytes = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: "model.layers.0.input_layernorm.weight".to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: "F32".to_string(),
                shape: vec![4],
                source_offsets: [0, bytes.len() as u64],
                runtime_offset: 0,
                byte_len: bytes.len() as u64,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();

        let first = store
            .norm_weight("model.layers.0.input_layernorm.weight", 4)
            .unwrap()
            .unwrap();
        let second = store
            .norm_weight("model.layers.0.input_layernorm.weight", 4)
            .unwrap()
            .unwrap();

        assert_eq!(first, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(second, first);
        assert_eq!(
            store
                .norm_weights
                .lock()
                .expect("dense norm weight cache poisoned")
                .len(),
            1
        );
    }

    #[test]
    fn dense_store_rms_norm_uses_small_weight_cache_without_decoded_tile_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("model_weights.bin");
        let manifest_path = tmp.path().join("model_weights.json");
        let weights = [0.5f32, 1.0, 1.5, 2.0];
        let mut bytes = Vec::new();
        for value in weights {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: "model.layers.0.post_attention_layernorm.weight".to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: "F32".to_string(),
                shape: vec![weights.len()],
                source_offsets: [0, bytes.len() as u64],
                runtime_offset: 0,
                byte_len: bytes.len() as u64,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();

        let input = [3.0f32, 4.0, -5.0, 12.0];
        let actual = store
            .rms_norm("model.layers.0.post_attention_layernorm.weight", &input)
            .unwrap();
        let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
        let scale = (mean_square + 1e-6).sqrt().recip();
        let expected: Vec<f32> = input
            .iter()
            .zip(weights)
            .map(|(value, weight)| value * scale * weight)
            .collect();

        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 1e-6,
                "rms norm element {idx} diverged: actual={actual}, expected={expected}"
            );
        }
        assert_eq!(
            store
                .norm_weights
                .lock()
                .expect("dense norm weight cache poisoned")
                .len(),
            1
        );
        assert_eq!(
            store
                .decoded_tiles
                .lock()
                .expect("decoded tile cache poisoned")
                .bytes,
            0
        );
    }

    #[test]
    fn dense_bf16_store_projects_synthetic_tensor_like_runtime_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("model_weights.bin");
        let manifest_path = tmp.path().join("model_weights.json");
        let tensor_name = "model.layers.0.self_attn.o_proj.weight";
        let rows = 19;
        let cols = 7;
        let values: Vec<f32> = (0..rows * cols)
            .map(|idx| ((idx as f32) * 0.37).sin() * 0.75 - ((idx % cols) as f32) * 0.03125)
            .collect();
        let bytes = bf16_tensor_bytes(&values);
        fs::write(&dense_path, &bytes).unwrap();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: tensor_name.to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: "BF16".to_string(),
                shape: vec![rows, cols],
                source_offsets: [0, bytes.len() as u64],
                runtime_offset: 0,
                byte_len: bytes.len() as u64,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let input = vec![0.25, -1.0, 0.5, 2.0, -0.75, 1.5, -0.125];
        let decoded = decode_dense_tensor_f32("BF16", &bytes).unwrap();
        let expected = cpu_dense_matvec(&decoded, &input, rows, cols);
        let projected = store
            .project_dense_tensor_with_metal(None, tensor_name, &input, rows)
            .unwrap()
            .unwrap();

        for (row, (actual, expected)) in projected.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 1e-6,
                "row {row}: BF16 projection {actual} diverged from decoded reference {expected}"
            );
        }
        assert_eq!(store.decoded_full_tensor_count(), 1);
    }

    #[test]
    fn dense_q4_store_projects_synthetic_tensor_like_runtime_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("model_weights.bin");
        let manifest_path = tmp.path().join("model_weights.json");
        let tensor_name = "model.layers.0.self_attn.q_proj.weight";
        let values = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 1.5, -1.0, -0.25, 0.75];
        let shape = vec![2, 5];
        let group_size = 3;
        let quantized = quantize_q4(&values, &shape, group_size).unwrap();
        let layout = dense_q4_layout(&shape, group_size).unwrap();
        assert_eq!(layout.rows, 2);
        assert_eq!(layout.cols, 5);
        assert_eq!(layout.row_packed_bytes, 3);
        assert_eq!(layout.groups_per_row, 2);
        assert_eq!(quantized.values.len(), layout.packed_bytes);
        assert_eq!(
            quantized.scales.len() * std::mem::size_of::<f32>(),
            layout.scales_bytes
        );

        let mut bytes = quantized.values.clone();
        for scale in &quantized.scales {
            bytes.extend_from_slice(&scale.to_le_bytes());
        }
        for bias in &quantized.biases {
            bytes.extend_from_slice(&bias.to_le_bytes());
        }
        assert_eq!(bytes.len(), layout.total_bytes);
        fs::write(&dense_path, &bytes).unwrap();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: tensor_name.to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: "F32".to_string(),
                shape: shape.clone(),
                source_offsets: [0, (values.len() * std::mem::size_of::<f32>()) as u64],
                runtime_offset: 0,
                byte_len: layout.total_bytes as u64,
                quantization: TensorQuantization::Q4 {
                    group_size,
                    format: DENSE_Q4_FORMAT.to_string(),
                    scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                },
                q4_sources: None,
            }],
        };
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let entry = store.registry().tensor(tensor_name).unwrap();
        let (packed_row, scales_row, biases_row, timing) =
            store.read_dense_q4_rows(entry, 1, 1, group_size).unwrap();
        assert_eq!(
            packed_row,
            quantized.values[layout.row_packed_bytes..].to_vec()
        );
        assert_eq!(
            scales_row,
            quantized.scales[layout.groups_per_row..].to_vec()
        );
        assert_eq!(
            biases_row,
            quantized.biases[layout.groups_per_row..].to_vec()
        );
        assert_eq!(
            timing.bytes_read,
            (layout.row_packed_bytes + layout.groups_per_row * 2 * std::mem::size_of::<f32>())
                as u64
        );
        let decoded_row = store
            .read_tensor_row_f32(tensor_name, 1, 5)
            .unwrap()
            .unwrap();
        let expected_row = q4_dequantize_rows_with_group_size(
            &quantized.values[layout.row_packed_bytes..],
            &quantized.scales[layout.groups_per_row..],
            &quantized.biases[layout.groups_per_row..],
            1,
            5,
            group_size,
        )
        .unwrap();
        assert_eq!(decoded_row, expected_row);

        let (packed_tile, scales_tile, biases_tile, tile_timing) =
            store.read_dense_q4_rows(entry, 0, 2, group_size).unwrap();
        assert_eq!(packed_tile, quantized.values);
        assert_eq!(scales_tile, quantized.scales);
        assert_eq!(biases_tile, quantized.biases);
        assert_eq!(
            tile_timing.bytes_read,
            (layout.packed_bytes + layout.scales_bytes * 2) as u64
        );

        let input = vec![1.0, -1.0, 0.5, 2.0, -0.25];
        let expected = q4_fma_matvec_with_group_size(
            &quantized.values,
            &input,
            &quantized.scales,
            &quantized.biases,
            2,
            5,
            group_size,
        )
        .unwrap();
        let projected = store
            .project_dense_tensor_with_metal(None, tensor_name, &input, 2)
            .unwrap()
            .unwrap();
        assert_eq!(projected, expected);
        let dense_expected = cpu_dense_matvec(&values, &input, 2, 5);
        for (actual, dense) in projected.iter().zip(dense_expected.iter()) {
            assert!(
                (*actual - *dense).abs() <= 0.12,
                "q4 projection drifted too far from dense matvec: actual={actual}, dense={dense}"
            );
        }
        let decoded = store
            .read_full_tensor_f32_cached(tensor_name)
            .unwrap()
            .unwrap();
        assert_eq!(decoded.len(), values.len());
        for (actual, dense) in decoded.iter().zip(values.iter()) {
            assert!(
                (*actual - *dense).abs() <= 0.12,
                "q4 full decode drifted too far from dense tensor: actual={actual}, dense={dense}"
            );
        }
    }

    #[test]
    #[ignore = "requires a local Metal device"]
    fn arm_macos_dense_q4_mmap_batch_matches_cpu_reference() {
        struct BatchTensor {
            name: String,
            shape: Vec<usize>,
            values: Vec<f32>,
            quantized: QuantizedQ4,
            runtime_offset: u64,
            byte_len: u64,
        }

        fn make_values(rows: usize, cols: usize, seed: f32) -> Vec<f32> {
            (0..rows * cols)
                .map(|idx| {
                    let wave = ((idx as f32 + seed) * 0.17).sin() * 0.625;
                    let slope = ((idx % cols) as f32 - 7.5) * 0.025;
                    wave + slope - seed * 0.03125
                })
                .collect()
        }

        fn append_q4_tensor(
            bytes: &mut Vec<u8>,
            name: &str,
            rows: usize,
            cols: usize,
            group_size: usize,
            values: Vec<f32>,
        ) -> BatchTensor {
            let shape = vec![rows, cols];
            let quantized = quantize_q4(&values, &shape, group_size).unwrap();
            let layout = dense_q4_layout(&shape, group_size).unwrap();
            let runtime_offset = bytes.len() as u64;
            bytes.extend_from_slice(&quantized.values);
            for scale in &quantized.scales {
                bytes.extend_from_slice(&scale.to_le_bytes());
            }
            for bias in &quantized.biases {
                bytes.extend_from_slice(&bias.to_le_bytes());
            }
            let byte_len = bytes.len() as u64 - runtime_offset;
            assert_eq!(byte_len as usize, layout.total_bytes);
            BatchTensor {
                name: name.to_string(),
                shape,
                values,
                quantized,
                runtime_offset,
                byte_len,
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        let cols = 16;
        let group_size = 8;
        let mut bytes = Vec::new();
        let tensors = vec![
            append_q4_tensor(
                &mut bytes,
                "model.layers.0.self_attn.q_proj.weight",
                3,
                cols,
                group_size,
                make_values(3, cols, 1.0),
            ),
            append_q4_tensor(
                &mut bytes,
                "model.layers.0.self_attn.k_proj.weight",
                5,
                cols,
                group_size,
                make_values(5, cols, 2.0),
            ),
            append_q4_tensor(
                &mut bytes,
                "model.layers.0.self_attn.v_proj.weight",
                2,
                cols,
                group_size,
                make_values(2, cols, 3.0),
            ),
        ];
        fs::write(&plan.non_expert_weights, &bytes).unwrap();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: tensors
                .iter()
                .map(|tensor| DenseTensorRef {
                    tensor: tensor.name.clone(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: tensor.shape.clone(),
                    source_offsets: [
                        tensor.runtime_offset,
                        tensor.runtime_offset
                            + (tensor.values.len() * std::mem::size_of::<f32>()) as u64,
                    ],
                    runtime_offset: tensor.runtime_offset,
                    byte_len: tensor.byte_len,
                    quantization: TensorQuantization::Q4 {
                        group_size,
                        format: DENSE_Q4_FORMAT.to_string(),
                        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                    },
                    q4_sources: None,
                })
                .collect(),
        };
        fs::write(
            &plan.tensor_manifest,
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(
            plan.non_expert_weights.clone(),
            plan.tensor_manifest.clone(),
        )
        .unwrap();
        let config = QwenModelConfig {
            model_type: Some("qwen".to_string()),
            architectures: None,
            num_hidden_layers: 1,
            hidden_size: cols,
            num_attention_heads: 1,
            head_dim: None,
            num_key_value_heads: Some(1),
            vocab_size: 32,
            rope_theta: None,
            partial_rotary_factor: None,
            torch_dtype: Some("float32".to_string()),
            num_experts: Some(1),
            num_experts_per_tok: Some(1),
            norm_topk_prob: None,
            moe_intermediate_size: Some(4),
            intermediate_size: None,
            max_position_embeddings: Some(4),
            mrope_section: None,
            tie_word_embeddings: None,
            num_shared_experts: None,
            shared_expert_intermediate_size: None,
            vision_config: None,
        };
        let runtime = DenseTransformerRuntime::new(&config);
        let metal = MetalExecutionFacade::new(&plan, &config, &runtime, &store).unwrap();
        let input: Vec<f32> = (0..cols)
            .map(|idx| ((idx as f32) * 0.11).cos() - 0.1875)
            .collect();
        let projections: Vec<_> = tensors
            .iter()
            .map(|tensor| {
                store
                    .dense_q4_mmap_projection(&tensor.name, tensor.shape[0], cols)
                    .unwrap()
                    .unwrap()
            })
            .collect();
        let projections = projections
            .into_iter()
            .map(ResidentMmapMatvecProjection::Q4)
            .collect::<Vec<_>>();
        let (actual, _timing, dispatches) = metal
            .resident_mmap_matvec_batch(&projections, &input)
            .unwrap();

        assert_eq!(dispatches, 1);
        assert_eq!(actual.len(), tensors.len());
        for (projection_idx, (actual, tensor)) in actual.iter().zip(tensors.iter()).enumerate() {
            let expected = q4_fma_matvec_with_group_size(
                &tensor.quantized.values,
                &input,
                &tensor.quantized.scales,
                &tensor.quantized.biases,
                tensor.shape[0],
                cols,
                group_size,
            )
            .unwrap();
            for (row, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (*actual - *expected).abs() < 1e-4,
                    "projection {projection_idx} row {row}: Metal q4 batch mmap {actual} diverged from CPU reference {expected}"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn arm_macos_resident_dense_mmap_batch_matches_cpu_reference() {
        fn f16_bits(value: f32) -> u16 {
            match value.to_bits() {
                bits if bits == 0.0f32.to_bits() => 0x0000,
                bits if bits == 0.25f32.to_bits() => 0x3400,
                bits if bits == 0.5f32.to_bits() => 0x3800,
                bits if bits == 1.0f32.to_bits() => 0x3c00,
                bits if bits == 2.0f32.to_bits() => 0x4000,
                bits if bits == (-0.5f32).to_bits() => 0xb800,
                bits if bits == (-1.0f32).to_bits() => 0xbc00,
                bits if bits == (-2.0f32).to_bits() => 0xc000,
                _ => panic!("test value {value} is not in the exact F16 fixture"),
            }
        }

        fn append_dense_tensor(
            bytes: &mut Vec<u8>,
            name: &str,
            dtype: &str,
            values: &[f32],
            rows: usize,
            cols: usize,
        ) -> DenseTensorRef {
            while !bytes.len().is_multiple_of(TENSOR_ALIGNMENT as usize) {
                bytes.push(0);
            }
            let runtime_offset = bytes.len() as u64;
            match dtype {
                "BF16" => {
                    for value in values {
                        bytes.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
                    }
                }
                "F16" => {
                    for value in values {
                        bytes.extend_from_slice(&f16_bits(*value).to_le_bytes());
                    }
                }
                "F32" => {
                    for value in values {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                _ => unreachable!(),
            }
            let byte_len = bytes.len() as u64 - runtime_offset;
            DenseTensorRef {
                tensor: name.to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: dtype.to_string(),
                shape: vec![rows, cols],
                source_offsets: [0, byte_len],
                runtime_offset,
                byte_len,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }
        }

        let temp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, temp.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        let rows = 3;
        let cols = 4;
        let values = [
            1.0, -0.5, 0.25, 2.0, -1.0, 0.5, 2.0, -2.0, 0.0, 1.0, -1.0, 0.5,
        ];
        let mut bytes = Vec::new();
        let tensors = [
            append_dense_tensor(&mut bytes, "dense_bf16", "BF16", &values, rows, cols),
            append_dense_tensor(&mut bytes, "dense_f16", "F16", &values, rows, cols),
            append_dense_tensor(&mut bytes, "dense_f32", "F32", &values, rows, cols),
        ];
        let tensor_names = tensors
            .iter()
            .map(|tensor| tensor.tensor.clone())
            .collect::<Vec<_>>();
        fs::write(&plan.non_expert_weights, &bytes).unwrap();
        fs::write(
            &plan.tensor_manifest,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: tensors.into_iter().collect(),
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(
            plan.non_expert_weights.clone(),
            plan.tensor_manifest.clone(),
        )
        .unwrap();
        let config = QwenModelConfig {
            model_type: Some("qwen".to_string()),
            architectures: None,
            num_hidden_layers: 1,
            hidden_size: cols,
            num_attention_heads: 1,
            head_dim: None,
            num_key_value_heads: Some(1),
            vocab_size: 32,
            rope_theta: None,
            partial_rotary_factor: None,
            torch_dtype: Some("bfloat16".to_string()),
            num_experts: Some(1),
            num_experts_per_tok: Some(1),
            norm_topk_prob: None,
            moe_intermediate_size: Some(4),
            intermediate_size: None,
            max_position_embeddings: Some(4),
            mrope_section: None,
            tie_word_embeddings: None,
            num_shared_experts: None,
            shared_expert_intermediate_size: None,
            vision_config: None,
        };
        let runtime = DenseTransformerRuntime::new(&config);
        let metal = MetalExecutionFacade::new(&plan, &config, &runtime, &store).unwrap();
        let input = [0.5, -1.0, 2.0, 0.25];
        let projections = tensor_names
            .iter()
            .map(|tensor_name| {
                store
                    .resident_mmap_projection(tensor_name, rows, cols)
                    .unwrap()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let (actual, _, dispatches) = metal
            .resident_mmap_matvec_batch(&projections, &input)
            .unwrap();

        let expected = values
            .chunks_exact(cols)
            .map(|weights| {
                weights
                    .iter()
                    .zip(input.iter())
                    .map(|(weight, input)| weight * input)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(dispatches, 3);
        for (index, dtype) in ["BF16", "F16", "F32"].iter().enumerate() {
            let output = &actual[index];
            for (row, (actual, expected)) in output.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1e-5,
                    "{dtype} row {row}: Metal {actual} != CPU {expected}"
                );
            }
            let actual_candidates = metal
                .resident_top_candidates(&projections[index], &input, 2, 2)
                .unwrap();
            let expected_candidates = top_k(&expected[..2], 2);
            assert_eq!(actual_candidates.len(), expected_candidates.len());
            for ((actual_id, actual_score), (expected_id, expected_score)) in
                actual_candidates.iter().zip(&expected_candidates)
            {
                assert_eq!(actual_id, expected_id, "{dtype} topK id diverged");
                assert!(
                    (actual_score - expected_score).abs() <= 1e-5,
                    "{dtype} topK score {actual_score} != {expected_score}"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn arm_macos_post_attention_dense_prep_matches_cpu_reference() {
        fn f16_bits(value: f32) -> u16 {
            match value.to_bits() {
                bits if bits == 0.0f32.to_bits() => 0x0000,
                bits if bits == 0.25f32.to_bits() => 0x3400,
                bits if bits == 0.5f32.to_bits() => 0x3800,
                bits if bits == 1.0f32.to_bits() => 0x3c00,
                bits if bits == 2.0f32.to_bits() => 0x4000,
                bits if bits == (-0.5f32).to_bits() => 0xb800,
                bits if bits == (-1.0f32).to_bits() => 0xbc00,
                bits if bits == (-2.0f32).to_bits() => 0xc000,
                _ => panic!("test value {value} is not in the exact F16 fixture"),
            }
        }

        fn append_dense_tensor(
            bytes: &mut Vec<u8>,
            name: &str,
            dtype: &str,
            values: &[f32],
            rows: usize,
            cols: usize,
        ) -> DenseTensorRef {
            while !bytes.len().is_multiple_of(TENSOR_ALIGNMENT as usize) {
                bytes.push(0);
            }
            let runtime_offset = bytes.len() as u64;
            match dtype {
                "BF16" => {
                    for value in values {
                        bytes.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
                    }
                }
                "F16" => {
                    for value in values {
                        bytes.extend_from_slice(&f16_bits(*value).to_le_bytes());
                    }
                }
                "F32" => {
                    for value in values {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                _ => unreachable!(),
            }
            DenseTensorRef {
                tensor: name.to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: dtype.to_string(),
                shape: vec![rows, cols],
                source_offsets: [0, (values.len() * std::mem::size_of::<f32>()) as u64],
                runtime_offset,
                byte_len: bytes.len() as u64 - runtime_offset,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }
        }

        fn matvec(weights: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            assert_eq!(weights.len(), rows * cols);
            weights
                .chunks_exact(cols)
                .map(|row| {
                    row.iter()
                        .zip(input)
                        .map(|(weight, value)| weight * value)
                        .sum()
                })
                .collect()
        }

        let layer = 0;
        let width = 4;
        let attention_width = 4;
        let experts = 3;
        let out_proj_name = attention_tensor_name(layer, "o_proj");
        let router_name = router_tensor_name(layer);
        let out_values = [
            1.0, -0.5, 0.25, 2.0, -1.0, 0.5, 2.0, -2.0, 0.0, 1.0, -1.0, 0.5, 0.25, -2.0, 1.0, 0.5,
        ];
        let router_values = [
            1.0, -0.5, 0.25, 2.0, -1.0, 0.5, 2.0, -2.0, 0.25, -2.0, 1.0, 0.5,
        ];
        let attention_output = [0.5, -1.0, 2.0, 0.25];
        let residual = [0.25, -0.5, 1.0, 2.0];
        let post_norm_weight = [1.0, 0.5, 2.0, 0.25];

        let expected_projected = matvec(&out_values, &attention_output, width, attention_width);
        let mut expected_residual = residual.to_vec();
        add_in_place(&mut expected_residual, &expected_projected);
        let mut expected_normed = expected_residual.clone();
        rms_norm_with_weight_in_place(&mut expected_normed, Some(&post_norm_weight));
        let expected_active = top_k(&matvec(&router_values, &expected_normed, experts, width), 2);

        for dtype in ["BF16", "F16", "F32"] {
            let temp = tempfile::tempdir().unwrap();
            let plan = plan_unchecked(QWEN35_MODEL, temp.path());
            fs::create_dir_all(&plan.runtime_dir).unwrap();
            let mut bytes = Vec::new();
            let out_tensor = append_dense_tensor(
                &mut bytes,
                &out_proj_name,
                dtype,
                &out_values,
                width,
                attention_width,
            );
            let router_tensor = append_dense_tensor(
                &mut bytes,
                &router_name,
                dtype,
                &router_values,
                experts,
                width,
            );
            fs::write(&plan.non_expert_weights, &bytes).unwrap();
            fs::write(
                &plan.tensor_manifest,
                serde_json::to_vec(&FlashMoeManifest {
                    model: QWEN35_MODEL.to_string(),
                    cache_version: CACHE_VERSION.to_string(),
                    dense_shards: vec!["dense.safetensors".to_string()],
                    expert_tensors: Vec::new(),
                    dense_tensors: vec![out_tensor, router_tensor],
                })
                .unwrap(),
            )
            .unwrap();
            let store = DenseStore::open(
                plan.non_expert_weights.clone(),
                plan.tensor_manifest.clone(),
            )
            .unwrap();
            let config = QwenModelConfig {
                model_type: Some("qwen3_moe".to_string()),
                architectures: None,
                num_hidden_layers: 1,
                hidden_size: width,
                num_attention_heads: 1,
                head_dim: None,
                num_key_value_heads: Some(1),
                vocab_size: 32,
                rope_theta: None,
                partial_rotary_factor: None,
                torch_dtype: Some(dtype.to_ascii_lowercase()),
                num_experts: Some(experts),
                num_experts_per_tok: Some(2),
                norm_topk_prob: None,
                moe_intermediate_size: Some(4),
                intermediate_size: None,
                max_position_embeddings: Some(4),
                mrope_section: None,
                tie_word_embeddings: None,
                num_shared_experts: None,
                shared_expert_intermediate_size: None,
                vision_config: None,
            };
            let runtime = DenseTransformerRuntime::new(&config);
            let metal = MetalExecutionFacade::new(&plan, &config, &runtime, &store).unwrap();
            let prep = store
                .post_attention_prep_with_metal(
                    &metal,
                    layer,
                    experts,
                    &out_proj_name,
                    &attention_output,
                    MetalBatchProjectionInput::Cpu(&residual),
                    &post_norm_weight,
                    2,
                )
                .unwrap();

            assert_eq!(prep.active.len(), expected_active.len());
            for ((actual_id, actual_score), (expected_id, expected_score)) in
                prep.active.iter().zip(&expected_active)
            {
                assert_eq!(actual_id, expected_id, "{dtype} route id diverged");
                assert!(
                    (actual_score - expected_score).abs() <= 1e-4,
                    "{dtype} route score {actual_score} != {expected_score}"
                );
            }
            let actual_residual = metal
                .inner
                .read_and_recycle_f32(prep.residual_buffer, width);
            let actual_normed = metal.inner.read_and_recycle_f32(prep.normed_buffer, width);
            for index in 0..width {
                assert!(
                    (actual_residual[index] - expected_residual[index]).abs() <= 1e-4,
                    "{dtype} residual[{index}] {} != {}",
                    actual_residual[index],
                    expected_residual[index]
                );
                assert!(
                    (actual_normed[index] - expected_normed[index]).abs() <= 1e-4,
                    "{dtype} normed[{index}] {} != {}",
                    actual_normed[index],
                    expected_normed[index]
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires a local Metal device"]
    fn arm_macos_post_attention_resident_q4_prep_matches_cpu_reference() {
        fn q4_bytes(
            values: &[f32],
            shape: &[usize],
            group_size: usize,
        ) -> (Vec<u8>, DenseQ4Layout, QuantizedQ4) {
            let quantized = quantize_q4(values, shape, group_size).unwrap();
            let layout = dense_q4_layout(shape, group_size).unwrap();
            let mut bytes = quantized.values.clone();
            for scale in &quantized.scales {
                bytes.extend_from_slice(&scale.to_le_bytes());
            }
            for bias in &quantized.biases {
                bytes.extend_from_slice(&bias.to_le_bytes());
            }
            assert_eq!(bytes.len(), layout.total_bytes);
            (bytes, layout, quantized)
        }

        let tmp = tempfile::tempdir().unwrap();
        let plan = plan_unchecked(QWEN35_MODEL, tmp.path());
        fs::create_dir_all(&plan.runtime_dir).unwrap();
        let layer = 0;
        let width = 8;
        let attention_width = 16;
        let experts = 6;
        let group_size = 4;
        let out_proj_name = linear_attention_tensor_name(layer, "out_proj");
        let router_name = router_tensor_name(layer);
        let out_shape = vec![width, attention_width];
        let router_shape = vec![experts, width];
        let out_values: Vec<f32> = (0..width * attention_width)
            .map(|idx| ((idx as f32) * 0.17).sin() * 0.625 - ((idx % 5) as f32) * 0.03125)
            .collect();
        let router_values: Vec<f32> = (0..experts * width)
            .map(|idx| ((idx as f32) * 0.23).cos() * 0.375 + ((idx % 3) as f32) * 0.0625)
            .collect();
        let (out_bytes, out_layout, out_quantized) = q4_bytes(&out_values, &out_shape, group_size);
        let (router_bytes, router_layout, router_quantized) =
            q4_bytes(&router_values, &router_shape, group_size);
        let router_offset = out_bytes.len();
        let mut dense_bytes = out_bytes;
        dense_bytes.extend_from_slice(&router_bytes);
        fs::write(&plan.non_expert_weights, &dense_bytes).unwrap();

        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![
                DenseTensorRef {
                    tensor: out_proj_name.clone(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: out_shape.clone(),
                    source_offsets: [0, (out_values.len() * std::mem::size_of::<f32>()) as u64],
                    runtime_offset: 0,
                    byte_len: out_layout.total_bytes as u64,
                    quantization: TensorQuantization::Q4 {
                        group_size,
                        format: DENSE_Q4_FORMAT.to_string(),
                        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                    },
                    q4_sources: None,
                },
                DenseTensorRef {
                    tensor: router_name.clone(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: router_shape.clone(),
                    source_offsets: [
                        (out_values.len() * std::mem::size_of::<f32>()) as u64,
                        ((out_values.len() + router_values.len()) * std::mem::size_of::<f32>())
                            as u64,
                    ],
                    runtime_offset: router_offset as u64,
                    byte_len: router_layout.total_bytes as u64,
                    quantization: TensorQuantization::Q4 {
                        group_size,
                        format: DENSE_Q4_FORMAT.to_string(),
                        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                    },
                    q4_sources: None,
                },
            ],
        };
        fs::write(
            &plan.tensor_manifest,
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(
            plan.non_expert_weights.clone(),
            plan.tensor_manifest.clone(),
        )
        .unwrap();
        let config = QwenModelConfig {
            model_type: Some("qwen".to_string()),
            architectures: None,
            num_hidden_layers: 1,
            hidden_size: width,
            num_attention_heads: 1,
            head_dim: None,
            num_key_value_heads: Some(1),
            vocab_size: 32,
            rope_theta: None,
            partial_rotary_factor: None,
            torch_dtype: Some("float32".to_string()),
            num_experts: Some(experts),
            num_experts_per_tok: Some(3),
            norm_topk_prob: None,
            moe_intermediate_size: Some(4),
            intermediate_size: None,
            max_position_embeddings: Some(4),
            mrope_section: None,
            tie_word_embeddings: None,
            num_shared_experts: None,
            shared_expert_intermediate_size: None,
            vision_config: None,
        };
        let runtime = DenseTransformerRuntime::new(&config);
        let metal = MetalExecutionFacade::new(&plan, &config, &runtime, &store).unwrap();

        let attention_output: Vec<f32> = (0..attention_width)
            .map(|idx| ((idx as f32) * 0.11).sin() - 0.375)
            .collect();
        let residual: Vec<f32> = (0..width)
            .map(|idx| ((idx as f32) * 0.29).cos() * 0.5)
            .collect();
        let post_norm_weight: Vec<f32> = (0..width)
            .map(|idx| 0.75 + (idx as f32) * 0.03125)
            .collect();

        let expected_projected = q4_fma_matvec_with_group_size(
            &out_quantized.values,
            &attention_output,
            &out_quantized.scales,
            &out_quantized.biases,
            width,
            attention_width,
            group_size,
        )
        .unwrap();
        let mut expected_residual = residual.clone();
        add_in_place(&mut expected_residual, &expected_projected);
        let mut expected_normed = expected_residual.clone();
        rms_norm_with_weight_in_place(&mut expected_normed, Some(&post_norm_weight));
        let expected_router = q4_fma_matvec_with_group_size(
            &router_quantized.values,
            &expected_normed,
            &router_quantized.scales,
            &router_quantized.biases,
            experts,
            width,
            group_size,
        )
        .unwrap();
        let expected_active = top_k(&expected_router, 3);

        let prep = store
            .post_attention_prep_with_metal(
                &metal,
                layer,
                experts,
                &out_proj_name,
                &attention_output,
                MetalBatchProjectionInput::Cpu(&residual),
                &post_norm_weight,
                3,
            )
            .unwrap();
        assert_eq!(prep.width, width);
        assert_eq!(prep.active.len(), expected_active.len());
        for (slot, ((actual_id, actual_score), (expected_id, expected_score))) in
            prep.active.iter().zip(expected_active.iter()).enumerate()
        {
            assert_eq!(
                actual_id, expected_id,
                "active expert id at slot {slot} diverged"
            );
            assert!(
                (*actual_score - *expected_score).abs() <= 1e-4,
                "active expert score at slot {slot} diverged: actual={actual_score}, expected={expected_score}"
            );
        }
        assert!(prep.routing_command().is_none());

        let actual_residual = metal
            .inner
            .read_and_recycle_f32(prep.residual_buffer, width);
        let actual_normed = metal.inner.read_and_recycle_f32(prep.normed_buffer, width);
        for idx in 0..width {
            assert!(
                (actual_residual[idx] - expected_residual[idx]).abs() <= 1e-4,
                "residual[{idx}] diverged: actual={} expected={}",
                actual_residual[idx],
                expected_residual[idx]
            );
            assert!(
                (actual_normed[idx] - expected_normed[idx]).abs() <= 1e-4,
                "normed[{idx}] diverged: actual={} expected={}",
                actual_normed[idx],
                expected_normed[idx]
            );
        }
    }

    #[test]
    fn dense_manifest_preserves_non_native_dense_weights() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path();
        let q_proj = f32_tensor_bytes(&[0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0]);
        let lm_head = f32_tensor_bytes(&[0.25; 8]);
        let embed = f32_tensor_bytes(&[0.5; 8]);
        let mtp_q_proj = f32_tensor_bytes(&[0.75; 8]);
        let mtp_expert = f32_tensor_bytes(&[1.25; 8]);
        let tensors = vec![
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                "F32".to_string(),
                vec![2, 4],
                q_proj,
            ),
            (
                "lm_head.weight".to_string(),
                "F32".to_string(),
                vec![2, 4],
                lm_head,
            ),
            (
                "model.embed_tokens.weight".to_string(),
                "F32".to_string(),
                vec![2, 4],
                embed,
            ),
            (
                "mtp.layers.0.self_attn.q_proj.weight".to_string(),
                "F32".to_string(),
                vec![2, 4],
                mtp_q_proj,
            ),
            (
                "mtp.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
                "F32".to_string(),
                vec![2, 4],
                mtp_expert,
            ),
        ];
        let fixture_refs = typed_fixture_refs(&tensors);
        fs::write(
            snapshot.join("dense.safetensors"),
            make_typed_safetensors(&fixture_refs),
        )
        .unwrap();
        let mut weight_map = serde_json::Map::new();
        for (name, _, _, _) in &tensors {
            weight_map.insert(
                name.clone(),
                serde_json::Value::String("dense.safetensors".to_string()),
            );
        }
        let index = serde_json::Value::Object(serde_json::Map::from_iter([(
            "weight_map".to_string(),
            serde_json::Value::Object(weight_map),
        )]));
        let index_path = snapshot.join("model.safetensors.index.json");
        fs::write(&index_path, index.to_string()).unwrap();

        let (manifest, visual_refs) = build_manifest(QWEN35_MODEL, snapshot, &index_path).unwrap();
        assert!(visual_refs.is_empty());
        assert!(manifest.expert_tensors.is_empty());
        let registry = TensorRegistry::from_manifest(&manifest);
        let q_proj_entry = registry
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q_proj_entry.quantization, TensorQuantization::None);
        assert_eq!(q_proj_entry.byte_len, 2 * 4 * 4);
        assert_eq!(
            registry.tensor("lm_head.weight").unwrap().quantization,
            TensorQuantization::None
        );
        assert_eq!(
            registry
                .tensor("model.embed_tokens.weight")
                .unwrap()
                .quantization,
            TensorQuantization::None
        );
        assert!(
            registry
                .tensor("mtp.layers.0.self_attn.q_proj.weight")
                .is_none()
        );
        assert!(
            registry
                .tensor("mtp.layers.0.mlp.experts.0.gate_proj.weight")
                .is_none()
        );
    }

    #[test]
    fn dense_manifest_imports_native_mlx_q4_triples() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path();
        let source_tensor_name = "language_model.model.layers.0.self_attn.q_proj.weight";
        let scales_name = "language_model.model.layers.0.self_attn.q_proj.scales";
        let biases_name = "language_model.model.layers.0.self_attn.q_proj.biases";
        let runtime_tensor_name = "model.layers.0.self_attn.q_proj.weight";
        let packed_word = 0x7654_3210u32.to_le_bytes().to_vec();
        let scales = bf16_tensor_bytes(&[0.5]);
        let biases = bf16_tensor_bytes(&[1.0]);
        let tensors = vec![
            (
                source_tensor_name.to_string(),
                "U32".to_string(),
                vec![1, 1],
                packed_word.clone(),
            ),
            (
                scales_name.to_string(),
                EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                vec![1, 1],
                scales.clone(),
            ),
            (
                biases_name.to_string(),
                EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
                vec![1, 1],
                biases.clone(),
            ),
        ];
        let fixture_refs = typed_fixture_refs(&tensors);
        fs::write(
            snapshot.join("dense.safetensors"),
            make_typed_safetensors(&fixture_refs),
        )
        .unwrap();
        let mut weight_map = serde_json::Map::new();
        for (name, _, _, _) in &tensors {
            weight_map.insert(
                name.clone(),
                serde_json::Value::String("dense.safetensors".to_string()),
            );
        }
        let index = serde_json::Value::Object(serde_json::Map::from_iter([(
            "weight_map".to_string(),
            serde_json::Value::Object(weight_map),
        )]));
        let index_path = snapshot.join("model.safetensors.index.json");
        fs::write(&index_path, index.to_string()).unwrap();

        let (manifest, visual_refs) = build_manifest(QWEN35_MODEL, snapshot, &index_path).unwrap();
        assert!(visual_refs.is_empty());
        assert!(manifest.expert_tensors.is_empty());
        assert_eq!(manifest.dense_tensors.len(), 1);
        let dense_ref = &manifest.dense_tensors[0];
        assert_eq!(dense_ref.tensor, runtime_tensor_name);
        assert_eq!(dense_ref.dtype, "U32");
        assert_eq!(dense_ref.shape, vec![1, 8]);
        assert_eq!(
            dense_ref.quantization,
            TensorQuantization::Q4 {
                group_size: GROUP_SIZE,
                format: DENSE_Q4_MLX_FORMAT.to_string(),
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            }
        );
        assert!(dense_ref.q4_sources.is_some());
        let layout = dense_q4_layout_with_scale_bias_dtype(
            &dense_ref.shape,
            GROUP_SIZE,
            EXPERT_SCALE_BIAS_DTYPE_BF16,
        )
        .unwrap();
        assert_eq!(dense_ref.byte_len, layout.total_bytes as u64);
        assert_eq!(layout.packed_bytes, packed_word.len());
        assert_eq!(layout.scales_bytes, scales.len());

        let dense_path = snapshot.join("model_weights.bin");
        write_dense_tensor_store(snapshot, &dense_path, &manifest.dense_tensors).unwrap();
        let mut expected_bytes = packed_word.clone();
        expected_bytes.extend_from_slice(&scales);
        expected_bytes.extend_from_slice(&biases);
        assert_eq!(fs::read(&dense_path).unwrap(), expected_bytes);

        let manifest_path = snapshot.join("model_weights.json");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let entry = store.registry().tensor(runtime_tensor_name).unwrap();
        let (packed, decoded_scales, decoded_biases, timing) =
            store.read_dense_q4_rows(entry, 0, 1, GROUP_SIZE).unwrap();
        assert_eq!(packed, packed_word);
        assert_eq!(decoded_scales, vec![0.5]);
        assert_eq!(decoded_biases, vec![1.0]);
        assert_eq!(
            timing.bytes_read,
            (layout.packed_bytes + layout.scales_bytes * 2) as u64
        );

        let input = vec![1.0; 8];
        let projected = store
            .project_dense_tensor_with_metal(None, runtime_tensor_name, &input, 1)
            .unwrap()
            .unwrap();
        assert_eq!(projected, vec![22.0]);
    }

    #[test]
    fn manifest_classifies_mlx_switch_mlp_tensors_as_aggregate_experts() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path();
        let tensors = vec![
            (
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight".to_string(),
                "U32".to_string(),
                vec![2, 2, 1],
                0x7654_3210u32.to_le_bytes().to_vec(),
            ),
            (
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight".to_string(),
                "U32".to_string(),
                vec![2, 2, 1],
                0x7654_3210u32.to_le_bytes().to_vec(),
            ),
            (
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight".to_string(),
                "U32".to_string(),
                vec![2, 1, 2],
                0x7654_3210u32.to_le_bytes().to_vec(),
            ),
        ];
        let fixture_refs = typed_fixture_refs(&tensors);
        fs::write(
            snapshot.join("experts.safetensors"),
            make_typed_safetensors(&fixture_refs),
        )
        .unwrap();
        let mut weight_map = serde_json::Map::new();
        for (name, _, _, _) in &tensors {
            weight_map.insert(
                name.clone(),
                serde_json::Value::String("experts.safetensors".to_string()),
            );
        }
        let index = serde_json::Value::Object(serde_json::Map::from_iter([(
            "weight_map".to_string(),
            serde_json::Value::Object(weight_map),
        )]));
        let index_path = snapshot.join("model.safetensors.index.json");
        fs::write(&index_path, index.to_string()).unwrap();

        let (manifest, visual_refs) = build_manifest(QWEN35_MODEL, snapshot, &index_path).unwrap();

        assert!(visual_refs.is_empty());
        assert!(manifest.dense_tensors.is_empty());
        assert_eq!(manifest.expert_tensors.len(), 3);
        let names = manifest
            .expert_tensors
            .iter()
            .map(|tensor| tensor.tensor.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"model.layers.0.mlp.switch_mlp.gate_proj.weight"));
        assert!(names.contains(&"model.layers.0.mlp.switch_mlp.up_proj.weight"));
        assert!(names.contains(&"model.layers.0.mlp.switch_mlp.down_proj.weight"));
        assert!(
            manifest
                .expert_tensors
                .iter()
                .all(|tensor| tensor.layer == Some(0) && tensor.expert.is_none())
        );
    }

    #[test]
    fn dense_store_reuses_decoded_tiles_for_repeated_lm_head_sampling() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("model_weights.bin");
        let manifest_path = tmp.path().join("model_weights.json");
        let mut bytes = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        let manifest = FlashMoeManifest {
            model: QWEN35_MODEL.to_string(),
            cache_version: CACHE_VERSION.to_string(),
            dense_shards: vec!["dense.safetensors".to_string()],
            expert_tensors: Vec::new(),
            dense_tensors: vec![DenseTensorRef {
                tensor: "lm_head.weight".to_string(),
                shard: "dense.safetensors".to_string(),
                dtype: "F32".to_string(),
                shape: vec![4, 2],
                source_offsets: [0, bytes.len() as u64],
                runtime_offset: 0,
                byte_len: bytes.len() as u64,
                quantization: TensorQuantization::None,
                q4_sources: None,
            }],
        };
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();

        let first = store
            .read_tensor_rows_f32_cached("lm_head.weight", 0, 2)
            .unwrap();
        assert_eq!(first.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(store.decoded_tensor_tile_count(), 1);

        let second = store
            .read_tensor_rows_f32_cached("lm_head.weight", 0, 2)
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(
            store.decoded_tensor_tile_count(),
            1,
            "cached LM-head tile should not be decoded again for the next token"
        );

        let other = store
            .read_tensor_rows_f32_cached("lm_head.weight", 2, 2)
            .unwrap();
        assert_eq!(other.as_slice(), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(store.decoded_tensor_tile_count(), 2);
    }

    #[test]
    fn dense_store_reports_decoded_tile_cache_hit_and_miss_timing() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("model_weights.bin");
        let manifest_path = tmp.path().join("model_weights.json");
        let mut bytes = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![DenseTensorRef {
                    tensor: "lm_head.weight".to_string(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![2, 2],
                    source_offsets: [0, bytes.len() as u64],
                    runtime_offset: 0,
                    byte_len: bytes.len() as u64,
                    quantization: TensorQuantization::None,
                    q4_sources: None,
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let (_, miss) = store
            .read_tensor_rows_f32_cached_profiled("lm_head.weight", 0, 2)
            .unwrap();
        assert_eq!(miss.cache_misses, 1);
        assert_eq!(miss.cache_hits, 0);
        assert_eq!(miss.cache_inserts, 1);
        assert_eq!(miss.bytes_read, bytes.len() as u64);
        assert_eq!(miss.decoded_bytes, bytes.len() as u64);

        let (_, hit) = store
            .read_tensor_rows_f32_cached_profiled("lm_head.weight", 0, 2)
            .unwrap();
        assert_eq!(hit.cache_hits, 1);
        assert_eq!(hit.cache_misses, 0);
        assert_eq!(hit.bytes_read, 0);
        assert_eq!(hit.decoded_bytes, 0);
    }

    #[test]
    fn dense_q4_mmap_projection_descriptors_are_cached() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("model_weights.bin");
        let manifest_path = tmp.path().join("model_weights.json");
        let tensor_name = "model.layers.0.self_attn.q_proj.weight";
        let shape = vec![2, 4];
        let group_size = 2;
        let values = [0.25, -0.5, 1.0, 0.75, -0.125, 0.375, 0.625, -0.875];
        let quantized = quantize_q4(&values, &shape, group_size).unwrap();
        let layout = dense_q4_layout(&shape, group_size).unwrap();
        let mut bytes = quantized.values.clone();
        for scale in &quantized.scales {
            bytes.extend_from_slice(&scale.to_le_bytes());
        }
        for bias in &quantized.biases {
            bytes.extend_from_slice(&bias.to_le_bytes());
        }
        assert_eq!(bytes.len(), layout.total_bytes);
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![DenseTensorRef {
                    tensor: tensor_name.to_string(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape,
                    source_offsets: [0, bytes.len() as u64],
                    runtime_offset: 0,
                    byte_len: bytes.len() as u64,
                    quantization: TensorQuantization::Q4 {
                        group_size,
                        format: DENSE_Q4_FORMAT.to_string(),
                        scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
                    },
                    q4_sources: None,
                }],
            })
            .unwrap(),
        )
        .unwrap();

        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let first = store
            .dense_q4_mmap_projection(tensor_name, 2, 4)
            .unwrap()
            .unwrap();
        assert_eq!(store.q4_mmap_projection_cache_len(), 1);
        let second = store
            .dense_q4_mmap_projection(tensor_name, 2, 4)
            .unwrap()
            .unwrap();

        assert_eq!(store.q4_mmap_projection_cache_len(), 1);
        assert_eq!(first.tensor_name, second.tensor_name);
        assert_eq!(first.packed_byte_offset, second.packed_byte_offset);
        assert_eq!(first.scales_byte_offset, second.scales_byte_offset);
        assert_eq!(first.biases_byte_offset, second.biases_byte_offset);
        assert_eq!(first.rows, second.rows);
        assert_eq!(first.cols, second.cols);
    }

    #[test]
    fn dense_transformer_runtime_runs_core_blocks() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":300,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":8,"num_experts_per_tok":2}"#,
        )
        .unwrap();
        let runtime = DenseTransformerRuntime::new(&config);
        let mut hidden = vec![1.0; runtime.width];
        rms_norm_in_place(&mut hidden);
        let before = hidden.clone();
        apply_rotary(&mut hidden, 4, runtime.head_dim, config.rope_theta.unwrap());
        let attended = causal_attention(
            &hidden,
            &[(&before, &before)],
            runtime.num_q_heads,
            runtime.kv_heads,
            runtime.head_dim,
        );
        assert_eq!(attended.len(), runtime.width);
    }

    #[test]
    fn gated_delta_recurrence_matches_explicit_reference_update() {
        let layout = LinearAttentionLayout {
            num_key_heads: 2,
            num_value_heads: 4,
            key_dim: 3,
            value_dim: 2,
            total_key_width: 6,
            total_value_width: 8,
            conv_dim: 14,
            conv_kernel_size: 4,
        };
        let matrix_len = layout.value_dim * layout.key_dim;
        let mut state: Vec<f32> = (0..layout.num_value_heads * matrix_len)
            .map(|idx| ((idx as f32) * 0.13).sin() * 0.25)
            .collect();
        let mut expected_state = state.clone();
        let lin_q = [0.2, -0.5, 0.75, -0.1, 0.35, 0.9];
        let lin_k = [0.4, -0.25, 0.6, 0.15, -0.8, 0.5];
        let lin_v = [0.7, -0.2, -0.45, 0.3, 0.1, 0.55, -0.35, 0.85];
        let alpha = [-0.4, 0.2, 0.7, -1.1];
        let beta = [0.6, -0.9, 1.4, -0.25];
        let a_log = [-1.2, -0.7, 0.1, -1.6];
        let dt_bias = [0.05, -0.2, 0.3, -0.1];
        let mut actual = vec![0.0; layout.num_value_heads * layout.value_dim];
        let mut expected = vec![0.0; actual.len()];

        apply_gated_delta_recurrence(
            layout,
            &mut state,
            &lin_q,
            &lin_k,
            &lin_v,
            &alpha,
            &beta,
            &a_log,
            &dt_bias,
            &mut actual,
        );

        let heads_per_key = layout.value_heads_per_key_head();
        for vh in 0..layout.num_value_heads {
            let kh = vh / heads_per_key;
            let decay = (-(a_log[vh].exp()) * (1.0 + (alpha[vh] + dt_bias[vh]).exp()).ln()).exp();
            let beta_gate = 1.0 / (1.0 + (-beta[vh]).exp());
            let state_base = vh * matrix_len;
            let key = &lin_k[kh * layout.key_dim..(kh + 1) * layout.key_dim];
            let query = &lin_q[kh * layout.key_dim..(kh + 1) * layout.key_dim];
            let value = &lin_v[vh * layout.value_dim..(vh + 1) * layout.value_dim];
            for vi in 0..layout.value_dim {
                let row_base = state_base + vi * layout.key_dim;
                for ki in 0..layout.key_dim {
                    expected_state[row_base + ki] *= decay;
                }
                let kv_mem: f32 = (0..layout.key_dim)
                    .map(|ki| expected_state[row_base + ki] * key[ki])
                    .sum();
                let delta = (value[vi] - kv_mem) * beta_gate;
                for ki in 0..layout.key_dim {
                    expected_state[row_base + ki] += delta * key[ki];
                }
                expected[vh * layout.value_dim + vi] = (0..layout.key_dim)
                    .map(|ki| expected_state[row_base + ki] * query[ki])
                    .sum();
            }
        }

        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 1e-6,
                "gated delta output {idx} diverged: actual={actual} expected={expected}"
            );
        }
        for (idx, (actual, expected)) in state.iter().zip(expected_state.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 1e-6,
                "gated delta state {idx} diverged: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn dense_transformer_runtime_uses_full_config_hidden_size() {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":151936,"rope_theta":1000000.0,"torch_dtype":"bfloat16","num_experts":512,"num_experts_per_tok":4}"#,
        )
        .unwrap();
        let runtime = DenseTransformerRuntime::new(&config);
        assert_eq!(runtime.width, 4096);
        assert_eq!(runtime.head_dim, 128);
    }

    #[test]
    fn expert_scheduler_reads_only_active_experts_without_process_cache() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        let packs: Vec<_> = [1usize, 3, 7]
            .into_iter()
            .map(|expert| {
                let tensor = format!("model.layers.0.mlp.experts.{expert}.down_proj.weight");
                let pack = test_expert_pack(&tensor);
                let metadata = test_expert_pack_metadata(0, expert, &tensor, pack.len());
                (expert, pack, metadata)
            })
            .collect();
        write_test_expert_layer(temp.path(), 0, packs, 8).unwrap();

        let mut scheduler =
            ExpertScheduler::new(ExpertSlotStore::open(temp.path().to_path_buf()).unwrap());
        let pending = scheduler.issue(0, &[1, 3]).unwrap();
        assert_eq!(scheduler.worker_count(), 2);
        let experts = scheduler.finish(pending).unwrap();
        assert_eq!(experts.len(), 2);
        assert!(experts.iter().all(|expert| expert.layer() == 0));
        assert_eq!(experts[0].expert(), 1);
        assert_eq!(experts[1].expert(), 3);
        let first = scheduler.snapshot();
        assert_eq!(first.issued_reads, 2);
        assert_eq!(first.positioned_reads, 2);
        assert_eq!(first.read_failures, 0);
        assert_eq!(first.warm_reads, 0);
        assert!(first.bytes_read > 0);
        assert!(first.total_queue_latency >= first.max_queue_latency);
        assert!(first.total_read_latency >= first.max_read_latency);

        let pending = scheduler.issue(0, &[3, 7]).unwrap();
        assert_eq!(scheduler.worker_count(), 2);
        let experts = scheduler.finish(pending).unwrap();
        assert_eq!(experts.len(), 2);
        assert_eq!(experts[0].expert(), 3);
        assert_eq!(experts[1].expert(), 7);
        let second = scheduler.snapshot();
        assert_eq!(second.issued_reads, 4);
        assert_eq!(second.positioned_reads, 4);
        assert_eq!(second.read_failures, 0);
        assert_eq!(second.warm_reads, 1);
        assert!(second.warm_bytes_read > 0);
        assert!(second.total_warm_read_latency >= second.max_warm_read_latency);
        assert!(second.total_read_latency >= second.max_read_latency);

        let pending = scheduler.issue(0, &[3]).unwrap();
        assert_eq!(scheduler.worker_count(), 2);
        let experts = scheduler.finish(pending).unwrap();
        assert_eq!(experts.len(), 1);
        assert_eq!(experts[0].expert(), 3);
        let third = scheduler.snapshot();
        assert_eq!(third.issued_reads, 5);
        assert_eq!(third.positioned_reads, 5);
        assert_eq!(third.read_failures, 0);
        assert_eq!(third.warm_reads, 2);
        assert!(third.warm_bytes_read >= second.warm_bytes_read);
    }

    #[test]
    fn expert_scheduler_guardrails_keep_flashmoe_discarded_experiments_disabled() {
        assert_eq!(
            FLASHMOE_EXPERT_IO_POLICY.expert_read_path,
            ExpertReadPath::PositionedRead
        );
        assert!(!FLASHMOE_EXPERT_IO_POLICY.application_expert_cache);
        assert!(!FLASHMOE_EXPERT_IO_POLICY.lz4_expert_compression);
        assert!(!FLASHMOE_EXPERT_IO_POLICY.speculative_routing);
        assert!(!FLASHMOE_EXPERT_IO_POLICY.broad_ssd_gpu_overlap);
    }

    #[test]
    fn expert_scheduler_preserves_requested_result_order() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        let packs: Vec<_> = [1usize, 3, 7]
            .into_iter()
            .map(|expert| {
                let tensor = format!("model.layers.0.mlp.experts.{expert}.down_proj.weight");
                let pack = test_expert_pack(&tensor);
                let metadata = test_expert_pack_metadata(0, expert, &tensor, pack.len());
                (expert, pack, metadata)
            })
            .collect();
        write_test_expert_layer(temp.path(), 0, packs, 8).unwrap();

        let mut scheduler =
            ExpertScheduler::new(ExpertSlotStore::open(temp.path().to_path_buf()).unwrap());
        let pending = scheduler.issue(0, &[7, 1, 3]).unwrap();
        assert_eq!(scheduler.worker_count(), 3);
        let experts = scheduler.finish(pending).unwrap();
        let order: Vec<_> = experts.iter().map(|expert| expert.expert()).collect();
        assert_eq!(order, vec![7, 1, 3]);
        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.issued_reads, 3);
        assert_eq!(snapshot.positioned_reads, 3);
        assert_eq!(snapshot.read_failures, 0);
    }

    fn test_routing_command(
        layer: usize,
        experts: usize,
        routes: &[(usize, f32)],
    ) -> ScheduledRoutingCommand {
        ScheduledRoutingCommand {
            routing: ScheduledRoutingTopK {
                stage: FlashMoeStageCapability::new(
                    FlashMoeGraphStage::RoutingSoftmaxTopK,
                    FlashMoeStagePlacement::CpuDeclared,
                    FlashMoeStageImplementation::CpuSoftmaxTopK,
                ),
                layer,
                experts,
                active_experts: routes.len(),
                source: ScheduledRoutingCandidateSource::CpuRouterScores,
            },
            layer,
            active_experts: routes.len(),
            source: ScheduledRoutingCandidateSource::CpuRouterScores,
            routes: routes.to_vec(),
        }
    }

    #[test]
    fn expert_scheduler_finishes_routed_set_with_normalized_weights() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        let packs: Vec<_> = [1usize, 3, 7]
            .into_iter()
            .map(|expert| {
                let tensor = format!("model.layers.0.mlp.experts.{expert}.down_proj.weight");
                let pack = test_expert_pack(&tensor);
                let metadata = test_expert_pack_metadata(0, expert, &tensor, pack.len());
                (expert, pack, metadata)
            })
            .collect();
        write_test_expert_layer(temp.path(), 0, packs, 8).unwrap();

        let mut scheduler =
            ExpertScheduler::new(ExpertSlotStore::open(temp.path().to_path_buf()).unwrap());
        let routes = [(7usize, 2.0f32), (1, 1.0), (3, -1.0)];
        let command = test_routing_command(0, 8, &routes);
        let pending = scheduler.issue_routing_command(&command).unwrap();
        assert_eq!(scheduler.worker_count(), 3);
        let scheduled = scheduler.finish_routes(pending).unwrap();

        assert_eq!(scheduled.layer, 0);
        assert_eq!(scheduled.routes.len(), routes.len());
        let order: Vec<_> = scheduled
            .experts
            .iter()
            .map(|expert| expert.expert())
            .collect();
        assert_eq!(order, vec![7, 1, 3]);
        let mut expected_weights: Vec<f32> = routes.iter().map(|(_, score)| *score).collect();
        softmax_in_place(&mut expected_weights);
        for (actual, expected) in scheduled.weights.iter().zip(expected_weights.iter()) {
            assert!((actual - expected).abs() <= 1e-6);
        }
        let weight_sum: f32 = scheduled.weights.iter().sum();
        assert!((weight_sum - 1.0).abs() <= 1e-6);
        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.issued_reads, 3);
        assert_eq!(snapshot.positioned_reads, 3);
        assert_eq!(snapshot.read_failures, 0);
    }

    #[test]
    fn expert_scheduler_applies_configured_routed_expert_scale() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        let packs: Vec<_> = [1usize, 3]
            .into_iter()
            .map(|expert| {
                let tensor = format!("model.layers.0.mlp.experts.{expert}.down_proj.weight");
                let pack = test_expert_pack(&tensor);
                let metadata = test_expert_pack_metadata(0, expert, &tensor, pack.len());
                (expert, pack, metadata)
            })
            .collect();
        write_test_expert_layer(temp.path(), 0, packs, 8).unwrap();

        let store = ExpertSlotStore::open(temp.path().to_path_buf()).unwrap();
        let mut scheduler = ExpertScheduler::new_with_routed_expert_scale(store, 0.9);
        let routes = [(3usize, 2.0f32), (1, 1.0)];
        let command = test_routing_command(0, 4, &routes);
        let pending = scheduler.issue_routing_command(&command).unwrap();
        let scheduled = scheduler.finish_routes(pending).unwrap();
        let mut expected_weights: Vec<f32> = routes.iter().map(|(_, score)| *score).collect();
        softmax_in_place(&mut expected_weights);
        for weight in &mut expected_weights {
            *weight *= 0.9;
        }
        for (actual, expected) in scheduled.weights.iter().zip(expected_weights.iter()) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn expert_scheduler_rejects_invalid_routes_before_issuing_reads() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        let store = ExpertSlotStore::open(temp.path().to_path_buf()).unwrap();
        let mut scheduler = ExpertScheduler::new(store);

        let command = test_routing_command(0, 4, &[(3usize, f32::INFINITY)]);
        let err = scheduler.issue_routing_command(&command).unwrap_err();

        assert!(
            err.to_string()
                .contains("scheduled routing command score for expert 3 is not finite"),
            "{err:#}"
        );
        assert_eq!(scheduler.snapshot().issued_reads, 0);
    }

    #[test]
    fn expert_scheduler_records_worker_read_failures() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        let tensor = "model.layers.0.mlp.experts.2.down_proj.weight";
        let pack = test_expert_pack(tensor);
        let metadata = test_expert_pack_metadata(0, 2, tensor, pack.len());
        write_test_expert_layer(temp.path(), 0, vec![(2, pack, metadata)], 8).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(expert_layer_path(temp.path(), 0))
            .unwrap()
            .set_len(0)
            .unwrap();

        let mut scheduler =
            ExpertScheduler::new(ExpertSlotStore::open(temp.path().to_path_buf()).unwrap());
        let pending = scheduler.issue(0, &[2]).unwrap();
        let err = scheduler.finish(pending).unwrap_err();
        assert!(
            err.to_string().contains("failed to read expert 2"),
            "{err:#}"
        );
        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.issued_reads, 1);
        assert_eq!(snapshot.positioned_reads, 1);
        assert_eq!(snapshot.read_failures, 1);
        assert!(snapshot.total_read_latency >= snapshot.max_read_latency);
    }

    #[test]
    fn expert_store_parses_pbq4expert_records_as_import_data_only() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        let tensor = "model.layers.0.mlp.experts.2.down_proj.weight";
        let pack = test_expert_pack(tensor);
        let metadata = test_expert_pack_metadata(0, 2, tensor, pack.len());
        write_test_expert_layer(temp.path(), 0, vec![(2, pack, metadata)], 8).unwrap();
        let layer_metadata = read_expert_layer_pack_metadata(temp.path(), 0)
            .unwrap()
            .unwrap();
        let expected = ExpectedExpertPack {
            expert: 2,
            packed_bytes: layer_metadata.pack_for(2).unwrap().packed_bytes,
            records: layer_metadata
                .pack_for(2)
                .unwrap()
                .records
                .iter()
                .map(|record| ExpectedExpertPackRecord {
                    tensor: record.tensor.clone(),
                    dtype: record.dtype.clone(),
                    shape: record.shape.clone(),
                    source_offsets: record.source_offsets,
                    source_hash: record.source_hash.clone().unwrap(),
                    packed_bytes: record.packed_bytes,
                    groups: record.groups,
                    group_size: record.group_size,
                    scale_bias_dtype: record.scale_bias_dtype.clone(),
                })
                .collect(),
        };
        assert!(
            expert_layer_slot_is_reusable(
                &expert_layer_path(temp.path(), 0),
                &layer_metadata,
                ExpertLayerStorageFormat::Pbq4Import,
                &expected
            )
            .unwrap()
        );

        let expert = read_one_expert(temp.path(), 0, 2).unwrap();
        assert_eq!(expert.records.len(), 1);
        assert_eq!(expert.records[0].name, tensor);
        assert_eq!(expert.records[0].scales, vec![0.5]);
        assert_eq!(expert.records[0].biases, vec![1.0]);
        let out = expert
            .project_record(&expert.records[0], &[1.0, 2.0, 3.0, 4.0], 1)
            .unwrap()
            .unwrap();
        let expected = (1.0 * 0.5 + 1.0) * 1.0
            + (2.0 * 0.5 + 1.0) * 2.0
            + (3.0 * 0.5 + 1.0) * 3.0
            + (4.0 * 0.5 + 1.0) * 4.0;
        assert!((out[0] - expected).abs() < 1e-6);

        let err = expert.mlp(&[1.0, 2.0, 3.0, 4.0], 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("scheduler-owned fixed-Q4 whole-expert slot is required"),
            "{err:#}"
        );
    }

    #[test]
    fn qwen_tokenizer_loads_special_tokens_and_applies_chat_template() {
        let tokenizer = QwenTokenizer::from_json_bytes(test_tokenizer_json()).unwrap();
        let templated = tokenizer.apply_chat_template("hi");
        assert!(templated.contains("<|im_start|>user"));
        let encoded = tokenizer.encode(&templated).unwrap();
        assert_eq!(encoded, vec![100, 5, 3, 101, 100, 6]);
        assert!(encoded.contains(&100));
        assert!(encoded.contains(&101));
        assert_eq!(tokenizer.decode(&[3, 101]).unwrap(), "hi");
        assert!(tokenizer.candidate_token_ids().contains(&102));
        assert!(tokenizer.candidate_token_ids().len() > 4);
    }

    #[test]
    fn qwen_tokenizer_loads_tokenizer_config_chat_template() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_tokenizer_config_json()),
        )
        .unwrap();
        let templated = tokenizer.apply_chat_template("hi");
        assert_eq!(
            templated,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            tokenizer.encode(&templated).unwrap(),
            vec![100, 5, 3, 101, 100, 6]
        );
    }

    #[test]
    fn qwen_structured_renderer_formats_single_user_prompt() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_tokenizer_config_json()),
        )
        .unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(&[ChatMessage::text(ChatRole::User, "hi")], &[], true)
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen_structured_renderer_falls_back_to_chatml_without_tokenizer_template() {
        let tokenizer = QwenTokenizer::from_json_bytes(test_tokenizer_json()).unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(&[ChatMessage::text(ChatRole::User, "hi")], &[], true)
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen_structured_renderer_formats_system_and_user_messages() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_tokenizer_config_json()),
        )
        .unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[
                    ChatMessage::text(ChatRole::System, "be terse"),
                    ChatMessage::text(ChatRole::User, "hi"),
                ],
                &[],
                true,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen_structured_renderer_formats_multi_turn_chat() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_tokenizer_config_json()),
        )
        .unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[
                    ChatMessage::text(ChatRole::User, "hi"),
                    ChatMessage::text(ChatRole::Assistant, "hello"),
                    ChatMessage::text(ChatRole::User, "again"),
                ],
                &[],
                true,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen_structured_renderer_injects_tool_schema() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_qwen3_tool_tokenizer_config_json()),
        )
        .unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[ChatMessage::text(ChatRole::User, "weather?")],
                &[ChatTool {
                    name: "get_weather".to_string(),
                    description: Some("Get weather.".to_string()),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }),
                }],
                true,
            )
            .unwrap();
        assert!(rendered.starts_with("<|im_start|>system\n<tools>\n"));
        assert!(rendered.contains("<tools>\n"));
        assert!(rendered.contains("\"name\":\"get_weather\""));
        assert!(rendered.contains("\"description\":\"Get weather.\""));
        assert!(rendered.contains("\"parameters\""));
        assert!(rendered.contains("\"city\":{\"type\":\"string\"}"));
        assert!(rendered.contains("</tools>"));
        assert!(rendered.contains("<|im_start|>user\nweather?<|im_end|>\n<|im_start|>assistant\n"));
    }

    #[test]
    fn qwen3_template_renderer_matches_tool_history_output() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_qwen3_tool_tokenizer_config_json()),
        )
        .unwrap();
        let tool = ChatTool {
            name: "get_weather".to_string(),
            description: Some("Get weather.".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
        };
        let mut assistant = ChatMessage::text(ChatRole::Assistant, "checking");
        assistant.tool_calls.push(ChatToolCall {
            id: Some("call_1".to_string()),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({"city": "London"}),
        });
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[
                    ChatMessage::text(ChatRole::System, "be precise"),
                    ChatMessage::text(ChatRole::User, "weather?"),
                    assistant,
                    ChatMessage {
                        role: ChatRole::Tool,
                        content: ChatMessageContent::Text("{\"temp\":12}".to_string()),
                        tool_calls: Vec::new(),
                        tool_call_id: Some("call_1".to_string()),
                        name: Some("get_weather".to_string()),
                    },
                    ChatMessage {
                        role: ChatRole::Tool,
                        content: ChatMessageContent::Text("{\"wind\":\"calm\"}".to_string()),
                        tool_calls: Vec::new(),
                        tool_call_id: Some("call_2".to_string()),
                        name: Some("get_weather".to_string()),
                    },
                ],
                std::slice::from_ref(&tool),
                true,
            )
            .unwrap();
        let tool_json = serde_json::to_string(&qwen_tool_schema_value(&tool)).unwrap();
        assert!(rendered.starts_with("<|im_start|>system\nbe precise\n\n<tools>\n"));
        assert!(rendered.contains(&tool_json));
        assert!(rendered.contains("<|im_start|>user\nweather?<|im_end|>\n"));
        assert!(rendered.contains("<|im_start|>assistant\nchecking\n<tool_call>\n"));
        assert!(rendered.contains("\"name\": \"get_weather\""));
        assert!(rendered.contains("\"arguments\": {\"city\":\"London\"}"));
        assert!(rendered.contains("<tool_response>\n{\"temp\":12}\n</tool_response>"));
        assert!(rendered.contains("<tool_response>\n{\"wind\":\"calm\"}\n</tool_response>"));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn qwen3vl_template_renderer_matches_image_and_tool_output() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_qwen3vl_tokenizer_json(),
            Some(test_qwen3vl_tool_tokenizer_config_json()),
        )
        .unwrap();
        let mut assistant = ChatMessage::text(ChatRole::Assistant, "");
        assistant.tool_calls.push(ChatToolCall {
            id: Some("call_1".to_string()),
            name: "describe_image".to_string(),
            arguments: serde_json::json!({"detail": "short"}),
        });
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[
                    ChatMessage {
                        role: ChatRole::User,
                        content: ChatMessageContent::Parts(vec![
                            ChatContentPart::Text {
                                text: "describe ".to_string(),
                            },
                            ChatContentPart::Image {
                                image: Some("first.png".to_string()),
                                placeholder_tokens: None,
                            },
                            ChatContentPart::Text {
                                text: " now".to_string(),
                            },
                        ]),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        name: None,
                    },
                    assistant,
                    ChatMessage {
                        role: ChatRole::Tool,
                        content: ChatMessageContent::Text("{\"ok\":true}".to_string()),
                        tool_calls: Vec::new(),
                        tool_call_id: Some("call_1".to_string()),
                        name: Some("describe_image".to_string()),
                    },
                ],
                &[],
                true,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\ndescribe <|vision_start|><|image_pad|><|vision_end|> now<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"describe_image\", \"arguments\": {\"detail\":\"short\"}}\n</tool_call>\n<|im_end|>\n<|im_start|>user\n<tool_response>\n{\"ok\":true}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen3_template_renderer_defers_image_parts_to_tokenizer_template() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_qwen3_tool_tokenizer_config_json()),
        )
        .unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[ChatMessage {
                    role: ChatRole::User,
                    content: ChatMessageContent::Parts(vec![ChatContentPart::Image {
                        image: Some("image.png".to_string()),
                        placeholder_tokens: None,
                    }]),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                }],
                &[],
                true,
            )
            .unwrap();
        assert!(rendered.contains("<|im_start|>user\n"));
        assert!(rendered.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn invalid_tokenizer_chat_template_errors_instead_of_falling_back() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(
                br#"{"eos_token":"<|im_end|>","add_bos_token":false,"split_special_tokens":false,"chat_template":"{% if messages %}"}"#,
            ),
        )
        .unwrap();
        let err = tokenizer
            .apply_chat_template_to_messages(&[ChatMessage::text(ChatRole::User, "hi")], &[], true)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("failed to render tokenizer chat_template")
        );
    }

    #[test]
    fn qwen_structured_renderer_formats_assistant_tool_call() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_qwen3_tool_tokenizer_config_json()),
        )
        .unwrap();
        let mut assistant = ChatMessage::text(ChatRole::Assistant, "checking");
        assistant.tool_calls.push(ChatToolCall {
            id: Some("call_1".to_string()),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({"city": "London"}),
        });
        let rendered = tokenizer
            .apply_chat_template_to_messages(&[assistant], &[], false)
            .unwrap();
        assert!(rendered.starts_with("<|im_start|>assistant\nchecking\n<tool_call>\n"));
        assert!(rendered.contains("\"name\": \"get_weather\""));
        assert!(rendered.contains("\"arguments\": {\"city\":\"London\"}"));
        assert!(rendered.ends_with("\n</tool_call>\n<|im_end|>\n"));
    }

    #[test]
    fn qwen_structured_renderer_formats_tool_result_as_user_response() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_qwen3_tool_tokenizer_config_json()),
        )
        .unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[ChatMessage {
                    role: ChatRole::Tool,
                    content: ChatMessageContent::Text("{\"temp\":12}".to_string()),
                    tool_calls: Vec::new(),
                    tool_call_id: Some("call_1".to_string()),
                    name: Some("get_weather".to_string()),
                }],
                &[],
                true,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\n<tool_response>\n{\"temp\":12}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen_structured_renderer_formats_vl_text_with_image_placeholder() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_qwen3vl_tokenizer_json(),
            Some(test_qwen3vl_tool_tokenizer_config_json()),
        )
        .unwrap();
        let rendered = tokenizer
            .apply_chat_template_to_messages(
                &[ChatMessage {
                    role: ChatRole::User,
                    content: ChatMessageContent::Parts(vec![
                        ChatContentPart::Text {
                            text: "describe ".to_string(),
                        },
                        ChatContentPart::Image {
                            image: Some("image.png".to_string()),
                            placeholder_tokens: None,
                        },
                    ]),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                }],
                &[],
                true,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\ndescribe <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen_tool_call_output_parser_extracts_calls_and_content() {
        let (content, calls) = parse_qwen_tool_call_output(
            "checking\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"London\"}}\n</tool_call>\n",
        )
        .unwrap();
        assert_eq!(content, "checking");
        assert_eq!(
            calls,
            vec![ChatToolCall {
                id: None,
                name: "get_weather".to_string(),
                arguments: serde_json::json!({"city": "London"}),
            }]
        );
    }

    #[test]
    fn qwen_tool_call_output_parser_extracts_function_calls() {
        let (content, calls) = parse_qwen_tool_call_output(
            "checking\n<tool_call>\n<function=get_weather>\n<parameter=city>\nLondon\n</parameter>\n<parameter=options>\n{\"unit\":\"c\"}\n</parameter>\n</function>\n</tool_call>\n",
        )
        .unwrap();
        assert_eq!(content, "checking");
        assert_eq!(
            calls,
            vec![ChatToolCall {
                id: None,
                name: "get_weather".to_string(),
                arguments: serde_json::json!({
                    "city": "London",
                    "options": {"unit": "c"}
                }),
            }]
        );
    }

    #[test]
    fn flashmoe_parity_qwen_tool_call_serialization_and_parsing_goldens() {
        let tokenizer = QwenTokenizer::from_json_bytes_with_config(
            test_tokenizer_json(),
            Some(test_qwen3_tool_tokenizer_config_json()),
        )
        .unwrap();
        let mut assistant = ChatMessage::text(ChatRole::Assistant, "");
        assistant.tool_calls = vec![
            ChatToolCall {
                id: Some("call_1".to_string()),
                name: "get_weather".to_string(),
                arguments: serde_json::json!({"city": "London"}),
            },
            ChatToolCall {
                id: Some("call_2".to_string()),
                name: "search".to_string(),
                arguments: serde_json::json!({"query": "forecast"}),
            },
        ];

        let rendered = tokenizer
            .apply_chat_template_to_messages(&[assistant], &[], false)
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"London\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\":\"forecast\"}}\n</tool_call>\n<|im_end|>\n"
        );

        let (content, calls) = parse_qwen_tool_call_output(
            "ready\n<tool_call>\n{\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"London\\\"}\"}}\n</tool_call>\n<tool_call>\n{\"tool_call_id\":\"call_2\",\"name\":\"search\",\"arguments\":{\"query\":\"forecast\"}}\n</tool_call>\n",
        )
        .unwrap();
        assert_eq!(content, "ready");
        assert_eq!(
            calls,
            vec![
                ChatToolCall {
                    id: Some("call_1".to_string()),
                    name: "get_weather".to_string(),
                    arguments: serde_json::json!({"city": "London"}),
                },
                ChatToolCall {
                    id: Some("call_2".to_string()),
                    name: "search".to_string(),
                    arguments: serde_json::json!({"query": "forecast"}),
                },
            ]
        );
    }

    #[test]
    fn qwen_tokenizer_uses_byte_level_bpe_from_tokenizer_json() {
        let tokenizer = QwenTokenizer::from_json_bytes(test_byte_bpe_tokenizer_json()).unwrap();
        assert_eq!(tokenizer.encode("hello world").unwrap(), vec![8, 9, 16]);
        assert_eq!(tokenizer.decode(&[8, 9, 16, 101]).unwrap(), "hello world");
        assert_eq!(
            tokenizer.encode("<|im_start|>hello<|im_end|>").unwrap(),
            vec![100, 8, 101]
        );
    }

    #[test]
    fn lm_head_logits_scores_full_vocab_in_cpu_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("dense.bin");
        let manifest_path = tmp.path().join("manifest.json");

        let tokenizer = QwenTokenizer::from_json_bytes(
            br#"{
  "added_tokens": [
    {"id": 2, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0,
      "a": 2
    },
    "unk_token": "<unk>"
  }
}"#,
        )
        .unwrap();
        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.candidate_token_ids(), &[0, 2]);

        let mut bytes = Vec::new();
        for row in 0..tokenizer.vocab_size() {
            let value = (row as f32) + 1.0;
            bytes.extend_from_slice(&value.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![DenseTensorRef {
                    tensor: "lm_head.weight".to_string(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![tokenizer.vocab_size(), 2],
                    source_offsets: [0, bytes.len() as u64],
                    runtime_offset: 0,
                    byte_len: bytes.len() as u64,
                    quantization: TensorQuantization::None,
                    q4_sources: None,
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let logits = store
            .lm_head_logits("lm_head.weight", &[1.0, 1.0], &tokenizer)
            .unwrap();

        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn lm_head_logits_accepts_padded_vocab_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("dense.bin");
        let manifest_path = tmp.path().join("manifest.json");

        let tokenizer = QwenTokenizer::from_json_bytes(
            br#"{
  "added_tokens": [
    {"id": 2, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0,
      "a": 2
    },
    "unk_token": "<unk>"
  }
}"#,
        )
        .unwrap();

        let mut bytes = Vec::new();
        for row_idx in 0..5usize {
            let value = (row_idx as f32) + 1.0;
            bytes.extend_from_slice(&value.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![DenseTensorRef {
                    tensor: "lm_head.weight".to_string(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![5, 2],
                    source_offsets: [0, bytes.len() as u64],
                    runtime_offset: 0,
                    byte_len: bytes.len() as u64,
                    quantization: TensorQuantization::None,
                    q4_sources: None,
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let logits = store
            .lm_head_logits("lm_head.weight", &[1.0, 1.0], &tokenizer)
            .unwrap();

        assert_eq!(logits, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn expert_q4_payload_borrows_record_buffers() {
        let tensor = PackedExpertTensor {
            name: "model.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
            dtype: "Q4".to_string(),
            shape: vec![2, 4],
            source_offsets: [0, 0],
            source_hash: None,
            group_size: 2,
            scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
            packed: vec![0x10, 0x32, 0x54, 0x76],
            scales: vec![0.5, 0.25, 0.125, 0.0625],
            biases: vec![1.0, 2.0, 3.0, 4.0],
            scale_bytes: vec![
                0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x80, 0x3e, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00,
                0x80, 0x3d,
            ],
            bias_bytes: vec![
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00,
                0x80, 0x40,
            ],
        };
        let payload = tensor.matvec_payload(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();

        assert_eq!(payload.rows, 2);
        assert_eq!(payload.cols, 4);
        assert_eq!(payload.packed.as_ptr(), tensor.packed.as_ptr());
        assert_eq!(payload.scales.as_ptr(), tensor.scales.as_ptr());
        assert_eq!(payload.biases.as_ptr(), tensor.biases.as_ptr());
    }

    #[test]
    fn dense_projection_rejects_input_width_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("dense.bin");
        let manifest_path = tmp.path().join("manifest.json");

        let mut bytes = Vec::new();
        for value in 0..6u32 {
            bytes.extend_from_slice(&(value as f32).to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![DenseTensorRef {
                    tensor: "proj.weight".to_string(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![2, 3],
                    source_offsets: [0, bytes.len() as u64],
                    runtime_offset: 0,
                    byte_len: bytes.len() as u64,
                    quantization: TensorQuantization::None,
                    q4_sources: None,
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let err = store
            .matvec_tensor_prefix("proj.weight", &[1.0, 1.0], 2)
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("proj.weight"), "{err:#}");
        assert!(message.contains("expected shape [2, 2]"), "{err:#}");
        assert!(message.contains("actual shape [2, 3]"), "{err:#}");
        assert!(message.contains("input length 2"), "{err:#}");
    }

    #[test]
    fn dense_projection_rejects_output_width_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("dense.bin");
        let manifest_path = tmp.path().join("manifest.json");

        let mut bytes = Vec::new();
        for value in 0..2u32 {
            bytes.extend_from_slice(&(value as f32).to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![DenseTensorRef {
                    tensor: "proj.weight".to_string(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![1, 2],
                    source_offsets: [0, bytes.len() as u64],
                    runtime_offset: 0,
                    byte_len: bytes.len() as u64,
                    quantization: TensorQuantization::None,
                    q4_sources: None,
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let err = store
            .project_dense_tensor_with_metal(None, "proj.weight", &[1.0, 1.0], 2)
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("proj.weight"), "{err:#}");
        assert!(message.contains("expected shape [2, 2]"), "{err:#}");
        assert!(message.contains("actual shape [1, 2]"), "{err:#}");
    }

    #[test]
    fn lm_head_logits_rejects_missing_vocab_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_path = tmp.path().join("dense.bin");
        let manifest_path = tmp.path().join("manifest.json");

        let tokenizer = QwenTokenizer::from_json_bytes(
            br#"{
  "added_tokens": [
    {"id": 2, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0,
      "a": 2
    },
    "unk_token": "<unk>"
  }
}"#,
        )
        .unwrap();

        let mut bytes = Vec::new();
        for row_idx in 0..2usize {
            let value = (row_idx as f32) + 1.0;
            bytes.extend_from_slice(&value.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&dense_path, &bytes).unwrap();
        fs::write(
            &manifest_path,
            serde_json::to_vec(&FlashMoeManifest {
                model: QWEN35_MODEL.to_string(),
                cache_version: CACHE_VERSION.to_string(),
                dense_shards: vec!["dense.safetensors".to_string()],
                expert_tensors: Vec::new(),
                dense_tensors: vec![DenseTensorRef {
                    tensor: "lm_head.weight".to_string(),
                    shard: "dense.safetensors".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![2, 2],
                    source_offsets: [0, bytes.len() as u64],
                    runtime_offset: 0,
                    byte_len: bytes.len() as u64,
                    quantization: TensorQuantization::None,
                    q4_sources: None,
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let store = DenseStore::open(dense_path, manifest_path).unwrap();
        let err = store
            .lm_head_logits("lm_head.weight", &[1.0, 1.0], &tokenizer)
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("lm_head.weight"), "{err:#}");
        assert!(message.contains("expected at least [3, 2]"), "{err:#}");
        assert!(message.contains("actual shape [2, 2]"), "{err:#}");
    }

    #[test]
    fn dense_q4_group16_reduces_projection_reconstruction_error() {
        let values: Vec<f32> = (0..128)
            .map(|idx| {
                let base = ((idx as f32) * 0.071).sin() * 0.35;
                let trend = ((idx % 17) as f32 - 8.0) * 0.013;
                if idx % 37 == 0 {
                    base + trend + 1.15
                } else {
                    base + trend
                }
            })
            .collect();
        let q64 = quantize_q4(&values, &[1, values.len()], GROUP_SIZE).unwrap();
        let q16 = quantize_q4(&values, &[1, values.len()], DENSE_Q4_GROUP_SIZE).unwrap();
        let reconstruction_error = |quantized: &QuantizedQ4, group_size: usize| -> f32 {
            values
                .iter()
                .enumerate()
                .map(|(col, value)| {
                    let byte = quantized.values[col / 2];
                    let code = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                    let group = col / group_size;
                    let decoded = code.mul_add(quantized.scales[group], quantized.biases[group]);
                    let delta = *value - decoded;
                    delta * delta
                })
                .sum()
        };
        let error64 = reconstruction_error(&q64, GROUP_SIZE);
        let error16 = reconstruction_error(&q16, DENSE_Q4_GROUP_SIZE);
        assert!(
            error16 < error64,
            "group16 reconstruction error {error16} was not below group64 error {error64}"
        );
    }

    #[test]
    fn expert_cache_quantizes_decoded_bf16_values_not_raw_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join(crate::cache_dir_name(QWEN35_MODEL));
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), test_tokenizer_json()).unwrap();
        write_test_config(&snapshot);
        std::fs::write(
            snapshot.join("dense.safetensors"),
            make_safetensors(&[("model.layers.0.self_attn.q_proj.weight", b"dense")]),
        )
        .unwrap();
        let mut gate_bytes = Vec::new();
        for value in 1u32..=128 {
            gate_bytes.extend_from_slice(&(((value as f32).to_bits() >> 16) as u16).to_le_bytes());
        }
        let mut up_bytes = Vec::new();
        for _ in 0..(16 * 8) {
            up_bytes.extend_from_slice(&0x3f80u16.to_le_bytes());
        }
        let mut down_bytes = Vec::new();
        for _ in 0..(8 * 16) {
            down_bytes.extend_from_slice(&0x3f80u16.to_le_bytes());
        }
        std::fs::write(
            snapshot.join("expert.safetensors"),
            make_typed_safetensors(&[
                (
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    "BF16",
                    vec![16, 8],
                    &gate_bytes,
                ),
                (
                    "model.layers.0.mlp.experts.0.up_proj.weight",
                    "BF16",
                    vec![16, 8],
                    &up_bytes,
                ),
                (
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                    "BF16",
                    vec![8, 16],
                    &down_bytes,
                ),
            ]),
        )
        .unwrap();
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            expert_triplet_weight_map(0, 0),
        )
        .unwrap();

        let plan = build_cache_from_hf_snapshot(QWEN35_MODEL, &snapshot).unwrap();
        let expert = read_one_expert(&plan.experts_dir, 0, 0).unwrap();
        let record = expert
            .records
            .iter()
            .find(|record| record.name.ends_with("gate_proj.weight"))
            .unwrap();
        let input = [1.0; 8];
        let payload = record.matvec_payload(&input, 1).unwrap();
        let out =
            q4_fma_matvec(payload.packed, &input, payload.scales, payload.biases, 1, 8).unwrap();
        assert!((out[0] - 36.0).abs() < 1.0, "decoded q4 sum was {}", out[0]);
    }

    #[test]
    fn q4_fma_matvec_dequantizes_nibbles_by_group() {
        let packed = [0x21, 0x43];
        let input = [1.0, 2.0, 3.0, 4.0];
        let scales = [0.5];
        let biases = [1.0];
        let out = q4_fma_matvec(&packed, &input, &scales, &biases, 1, 4).unwrap();
        let expected = (1.0 * 0.5 + 1.0) * 1.0
            + (2.0 * 0.5 + 1.0) * 2.0
            + (3.0 * 0.5 + 1.0) * 3.0
            + (4.0 * 0.5 + 1.0) * 4.0;
        assert!((out[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn q4_fma_matvec_supports_variable_groups_and_odd_shapes() {
        let rows = 3;
        let cols = 5;
        let group_size = 3;
        let packed = [
            0x10, 0x32, 0x04, // row 0: 0, 1, 2, 3, 4
            0x65, 0x87, 0x09, // row 1: 5, 6, 7, 8, 9
            0xba, 0xdc, 0x0e, // row 2: 10, 11, 12, 13, 14
        ];
        let input = [0.25, -1.0, 2.0, 0.5, -0.75];
        let scales = [0.5, -0.25, 0.125, 0.75, -0.5, 0.25];
        let biases = [1.0, 2.0, -1.5, 0.25, 0.0, -0.5];
        let out = q4_fma_matvec_with_group_size(
            &packed, &input, &scales, &biases, rows, cols, group_size,
        )
        .unwrap();

        let mut expected = [0.0f32; 3];
        let groups_per_row = cols.div_ceil(group_size);
        for row in 0..rows {
            for col in 0..cols {
                let byte = packed[row * cols.div_ceil(2) + col / 2];
                let q = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                let group = col / group_size;
                let idx = row * groups_per_row + group;
                expected[row] += q.mul_add(scales[idx] * input[col], biases[idx] * input[col]);
            }
        }

        for (actual, expected) in out.iter().zip(expected) {
            assert!(
                (*actual - expected).abs() < 1e-6,
                "actual {actual} expected {expected}"
            );
        }
    }

    #[test]
    fn q4_fma_matvec_matches_explicit_dequant_reference() {
        let rows = 2;
        let cols = 7;
        let group_size = 4;
        let packed = [
            0xf0, 0x21, 0x43, 0x06, // row 0: 0, 15, 1, 2, 3, 4, 6
            0x75, 0x98, 0xba, 0x0d, // row 1: 5, 7, 8, 9, 10, 11, 13
        ];
        let input = [0.5, -2.0, 1.25, 0.0, -0.75, 3.0, -1.5];
        let scales = [0.03125, -0.125, 0.5, -0.25];
        let biases = [-1.0, 2.0, 0.25, -0.5];

        let actual = q4_fma_matvec_with_group_size(
            &packed, &input, &scales, &biases, rows, cols, group_size,
        )
        .unwrap();

        let packed_stride = cols.div_ceil(2);
        let groups_per_row = cols.div_ceil(group_size);
        let expected: Vec<f32> = (0..rows)
            .map(|row| {
                (0..cols)
                    .map(|col| {
                        let byte = packed[row * packed_stride + col / 2];
                        let code = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                        let group = row * groups_per_row + col / group_size;
                        let decoded = code * scales[group] + biases[group];
                        decoded * input[col]
                    })
                    .sum()
            })
            .collect();

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (*actual - *expected).abs() <= 1e-6,
                "FMA matvec diverged from explicit dequant: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn q4_bf16_expert_pack_matches_flashmoe_uint32_nibble_reference() {
        let tensor = "model.layers.0.mlp.experts.0.gate_proj.weight";
        let input = [1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.75];
        let mut pack = Vec::new();
        pack.extend_from_slice(PBQ4_EXPERT_MAGIC);
        pack.extend_from_slice(&(tensor.len() as u32).to_le_bytes());
        pack.extend_from_slice(tensor.as_bytes());
        pack.extend_from_slice(&4u64.to_le_bytes());
        pack.extend_from_slice(&1u64.to_le_bytes());
        pack.extend_from_slice(&f32_to_bf16_bits(0.5).to_le_bytes());
        pack.extend_from_slice(&f32_to_bf16_bits(1.0).to_le_bytes());
        pack.extend_from_slice(&0x7654_3210u32.to_le_bytes());

        let metadata = ExpertPackMetadata {
            layer: 0,
            expert: 0,
            packed_bytes: pack.len() as u64,
            records: vec![ExpertPackRecord {
                tensor: tensor.to_string(),
                dtype: "Q4".to_string(),
                shape: vec![1, 8],
                source_offsets: [0, 8],
                source_hash: Some("synthetic".to_string()),
                record_offset: PBQ4_EXPERT_MAGIC.len() as u64,
                packed_bytes: 4,
                groups: 1,
                group_size: 8,
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_BF16.to_string(),
            }],
        };
        let records = parse_pbq4_expert_pack(&pack, Some(&metadata)).unwrap();
        let payload = records[0].matvec_payload(&input, 1).unwrap();
        let actual = q4_fma_matvec_with_group_size(
            payload.packed,
            &input,
            payload.scales,
            payload.biases,
            payload.rows,
            payload.cols,
            payload.group_size,
        )
        .unwrap();

        let packed_word = u32::from_le_bytes([
            payload.packed[0],
            payload.packed[1],
            payload.packed[2],
            payload.packed[3],
        ]);
        let expected: f32 = input
            .iter()
            .enumerate()
            .map(|(n, x)| {
                let nibble = ((packed_word >> (n * 4)) & 0x0f) as f32;
                (nibble * 0.5 + 1.0) * x
            })
            .sum();

        assert_eq!(payload.scale_bias_dtype, EXPERT_SCALE_BIAS_DTYPE_BF16);
        assert_eq!(payload.scale_bytes.len(), 2);
        assert_eq!(payload.bias_bytes.len(), 2);
        assert!(
            (actual[0] - expected).abs() <= 1e-6,
            "bf16 q4 matvec diverged from uint32 nibble reference: actual={} expected={expected}",
            actual[0]
        );
    }

    fn test_expert_pack(name: &str) -> Vec<u8> {
        let mut pack = Vec::new();
        pack.extend_from_slice(PBQ4_EXPERT_MAGIC);
        pack.extend_from_slice(&(name.len() as u32).to_le_bytes());
        pack.extend_from_slice(name.as_bytes());
        pack.extend_from_slice(&2u64.to_le_bytes());
        pack.extend_from_slice(&1u64.to_le_bytes());
        pack.extend_from_slice(&0.5f32.to_le_bytes());
        pack.extend_from_slice(&1.0f32.to_le_bytes());
        pack.extend_from_slice(&[0x21, 0x43]);
        pack
    }

    fn test_expert_pack_metadata(
        layer: usize,
        expert: usize,
        tensor: &str,
        packed_bytes: usize,
    ) -> ExpertPackMetadata {
        ExpertPackMetadata {
            layer,
            expert,
            packed_bytes: packed_bytes as u64,
            records: vec![ExpertPackRecord {
                tensor: tensor.to_string(),
                dtype: "F32".to_string(),
                shape: vec![1, 4],
                source_offsets: [0, 4],
                source_hash: Some(format!("hash-{layer}-{expert}")),
                record_offset: PBQ4_EXPERT_MAGIC.len() as u64,
                packed_bytes: 2,
                groups: 1,
                group_size: GROUP_SIZE,
                scale_bias_dtype: EXPERT_SCALE_BIAS_DTYPE_F32.to_string(),
            }],
        }
    }

    fn write_test_expert_layer(
        root: &Path,
        layer: usize,
        packs: Vec<(usize, Vec<u8>, ExpertPackMetadata)>,
        experts: usize,
    ) -> Result<()> {
        let slot_size = packs
            .iter()
            .map(|(_, pack, _)| pack.len() as u64)
            .max()
            .unwrap_or(1)
            .max(1);
        let path = expert_layer_path(root, layer);
        let file = fs::File::create(&path)
            .with_context(|| format!("failed to create test layer {}", path.display()))?;
        file.set_len((experts as u64) * slot_size)?;
        let mut metadata = Vec::new();
        for (expert, pack, mut pack_metadata) in packs {
            pack_metadata.packed_bytes = pack.len() as u64;
            write_all_at_positioned(&file, &pack, expert_slot_offset(expert, slot_size)?)?;
            metadata.push(pack_metadata);
        }
        let layer_metadata = ExpertLayerPackMetadata::new(layer, slot_size, experts, metadata);
        fs::write(
            expert_layer_metadata_path(root, layer),
            serde_json::to_vec(&layer_metadata)?,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod flashmoe_rope_tests {
    use super::*;

    fn assert_close(left: f32, right: f32) {
        let diff = (left - right).abs();
        assert!(
            diff <= 1e-5,
            "values differ: left={left:.9}, right={right:.9}, diff={diff:.9}"
        );
    }

    #[test]
    fn flashmoe_rope_split_half_matches_reference() {
        // Mirrors danveloper/flash-moe:
        //   half = rotary_dim / 2
        //   freq = 1 / pow(theta, 2*i / rotary_dim)
        //   pairs are (x[i], x[i + half]), not adjacent pairs.
        let position = 3usize;
        let head_dim = 8usize;
        let rotary_dim = 4usize;
        let theta = 10_000_000.0f64;

        let mut got = vec![1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0];
        apply_rotary_split_half(&mut got, position, head_dim, rotary_dim, theta);

        let mut expected = vec![1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0];
        let half = rotary_dim / 2;
        for i in 0..half {
            let freq = 1.0f32 / (theta as f32).powf((2 * i) as f32 / rotary_dim as f32);
            let angle = position as f32 * freq;
            let (sin_a, cos_a) = angle.sin_cos();

            let x0 = expected[i];
            let x1 = expected[i + half];
            expected[i] = x0 * cos_a - x1 * sin_a;
            expected[i + half] = x0 * sin_a + x1 * cos_a;
        }

        for (left, right) in got.iter().zip(expected.iter()) {
            assert_close(*left, *right);
        }

        // Non-rotary tail must be untouched.
        assert_eq!(&got[rotary_dim..], &[100.0, 200.0, 300.0, 400.0]);
    }

    #[test]
    fn gated_flashmoe_rope_defaults_to_partial_split_half() {
        let config = QwenModelConfig {
            model_type: Some("qwen".to_string()),
            architectures: None,
            num_hidden_layers: 1,
            hidden_size: 4096,
            num_attention_heads: 32,
            head_dim: Some(256),
            num_key_value_heads: Some(2),
            vocab_size: 248320,
            rope_theta: None,
            partial_rotary_factor: None,
            torch_dtype: Some("bfloat16".to_string()),
            num_experts: Some(512),
            num_experts_per_tok: Some(4),
            norm_topk_prob: None,
            moe_intermediate_size: Some(1024),
            intermediate_size: None,
            max_position_embeddings: None,
            mrope_section: None,
            tie_word_embeddings: None,
            num_shared_experts: None,
            shared_expert_intermediate_size: None,
            vision_config: None,
        };

        let rotary_dim = rotary_dim_for(&config, 256, FullAttentionQLayout::Gated);
        assert_eq!(rotary_dim, 64);
    }

    #[test]
    fn standard_qwen_rope_uses_split_half_pairing() {
        let layout = FullAttentionLayout {
            q_layout: FullAttentionQLayout::Standard,
            q_projection_width: 8,
            q_width: 8,
            kv_width: 8,
            head_dim: 8,
            rotary_dim: 8,
            num_q_heads: 1,
            kv_heads: 1,
            rotary_pairing: RotaryPairing::SplitHalf,
        };

        let mut q = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mut k = q.clone();
        apply_rotary_for_layout(
            &mut q,
            &mut k,
            MropePosition::text(1),
            1_000_000.0,
            layout,
            None,
        );

        // Adjacent pairing would rotate (0,1), (2,3), ...
        // Split-half pairing rotates (0,4), (1,5), ...
        assert_ne!(q[1], 2.0);
        assert_ne!(q[5], 20.0);
    }

    #[test]
    fn qwen3vl_mrope_interleaves_height_and_width_frequency_slots() {
        let position = MropePosition {
            temporal: 2,
            height: 5,
            width: 7,
        };
        let section = [2, 1, 1];
        let head_dim = 8usize;
        let theta = 10_000.0f64;

        let mut got = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        apply_rotary_split_half_mrope(&mut got, position, head_dim, head_dim, theta, section);

        let mut expected = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let half = head_dim / 2;
        for i in 0..half {
            let axis = match i {
                1 => position.height,
                2 => position.width,
                _ => position.temporal,
            };
            let freq = 1.0f32 / (theta as f32).powf((2 * i) as f32 / head_dim as f32);
            let angle = axis as f32 * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = expected[i];
            let x1 = expected[i + half];
            expected[i] = x0 * cos_a - x1 * sin_a;
            expected[i + half] = x0 * sin_a + x1 * cos_a;
        }

        for (left, right) in got.iter().zip(expected.iter()) {
            assert_close(*left, *right);
        }
    }

    #[test]
    fn qwen3vl_image_mrope_positions_match_single_image_get_rope_index_shape() {
        let tokens = [101, 999, 999, 999, 999, 102, 201, 202];
        let (positions, next_position) =
            qwen3vl_single_image_mrope_positions(&tokens, 999, 2, 2).unwrap();

        assert_eq!(positions[0], MropePosition::text(0));
        assert_eq!(
            positions[1],
            MropePosition {
                temporal: 1,
                height: 1,
                width: 1,
            }
        );
        assert_eq!(
            positions[2],
            MropePosition {
                temporal: 1,
                height: 1,
                width: 2,
            }
        );
        assert_eq!(
            positions[3],
            MropePosition {
                temporal: 1,
                height: 2,
                width: 1,
            }
        );
        assert_eq!(
            positions[4],
            MropePosition {
                temporal: 1,
                height: 2,
                width: 2,
            }
        );
        assert_eq!(positions[5], MropePosition::text(3));
        assert_eq!(positions[6], MropePosition::text(4));
        assert_eq!(positions[7], MropePosition::text(5));
        assert_eq!(next_position, 6);
    }
}
