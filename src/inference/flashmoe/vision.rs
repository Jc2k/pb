use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::legacy::{FlashMoePlan, QwenModelConfig};
use super::types::{
    VIT_IMAGE_MEAN, VIT_IMAGE_STD, VIT_MAX_PIXELS, VIT_MERGE_SIZE, VIT_MIN_PIXELS, VIT_PATCH_SIZE,
};
use super::weights::DenseStore;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Qwen3VLVisionConfig {
    pub depth: usize,
    #[serde(alias = "hidden_size")]
    pub embed_dim: usize,
    pub num_heads: usize,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_vit_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "default_vit_patch_size")]
    pub patch_size: usize,
    #[serde(alias = "spatial_merge_size")]
    #[serde(default = "default_vit_merge_size")]
    pub merge_size: usize,
    #[serde(default = "default_vit_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default)]
    pub num_position_embeddings: Option<usize>,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
    #[serde(default)]
    pub out_hidden_size: Option<usize>,
    #[serde(alias = "in_channels")]
    #[serde(default = "default_vit_in_chans")]
    pub in_chans: usize,
}

fn default_vit_mlp_ratio() -> f64 {
    4.0
}

fn default_vit_patch_size() -> usize {
    VIT_PATCH_SIZE
}

fn default_vit_merge_size() -> usize {
    VIT_MERGE_SIZE
}

fn default_vit_temporal_patch_size() -> usize {
    2
}

fn default_vit_in_chans() -> usize {
    3
}

impl Qwen3VLVisionConfig {
    pub fn token_stride(&self) -> usize {
        self.patch_size * self.merge_size
    }

    pub fn patches_per_token(&self) -> usize {
        self.merge_size * self.merge_size
    }

    pub fn patch_flat_dim(&self) -> usize {
        self.in_chans * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    pub fn mlp_hidden_size(&self) -> usize {
        self.intermediate_size
            .unwrap_or_else(|| (self.embed_dim as f64 * self.mlp_ratio).round() as usize)
    }
}

#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub max_pixels: usize,
    pub min_pixels: usize,
}

impl ImagePreprocessor {
    pub fn from_vision_config(config: &Qwen3VLVisionConfig) -> Self {
        Self {
            patch_size: config.patch_size,
            merge_size: config.merge_size,
            temporal_patch_size: config.temporal_patch_size,
            image_mean: VIT_IMAGE_MEAN,
            image_std: VIT_IMAGE_STD,
            max_pixels: VIT_MAX_PIXELS,
            min_pixels: VIT_MIN_PIXELS,
        }
    }

    pub fn default_qwen3_vl() -> Self {
        Self {
            patch_size: VIT_PATCH_SIZE,
            merge_size: VIT_MERGE_SIZE,
            temporal_patch_size: 2,
            image_mean: VIT_IMAGE_MEAN,
            image_std: VIT_IMAGE_STD,
            max_pixels: VIT_MAX_PIXELS,
            min_pixels: VIT_MIN_PIXELS,
        }
    }

    pub fn token_stride(&self) -> usize {
        self.patch_size * self.merge_size
    }

    pub fn patch_flat_dim(&self) -> usize {
        3 * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    pub fn smart_resize(&self, orig_h: u32, orig_w: u32) -> (u32, u32) {
        let stride = self.token_stride() as u32;
        let mut h = round_up_to_stride(orig_h.max(stride), stride);
        let mut w = round_up_to_stride(orig_w.max(stride), stride);
        let pixels = (h as usize) * (w as usize);
        if pixels > self.max_pixels {
            let scale = ((self.max_pixels as f64) / (pixels as f64)).sqrt();
            h = round_to_stride((orig_h as f64) * scale, stride);
            w = round_to_stride((orig_w as f64) * scale, stride);
        } else if pixels < self.min_pixels {
            let scale = ((self.min_pixels as f64) / (pixels as f64)).sqrt();
            h = round_to_stride((orig_h as f64) * scale, stride);
            w = round_to_stride((orig_w as f64) * scale, stride);
        }
        while (h as usize) * (w as usize) > self.max_pixels && (h > stride || w > stride) {
            if h >= w && h > stride {
                h -= stride;
            } else if w > stride {
                w -= stride;
            } else {
                break;
            }
        }
        while (h as usize) * (w as usize) < self.min_pixels {
            if h <= w {
                h = h.saturating_add(stride);
            } else {
                w = w.saturating_add(stride);
            }
        }
        (h.max(stride), w.max(stride))
    }

    pub fn preprocess(&self, path: &Path) -> Result<(usize, usize, Vec<f32>)> {
        let image = image::ImageReader::open(path)
            .with_context(|| format!("vision: failed to open image {}", path.display()))?
            .with_guessed_format()
            .with_context(|| format!("vision: failed to guess format for {}", path.display()))?
            .decode()
            .with_context(|| format!("vision: failed to decode image {}", path.display()))?
            .to_rgb8();
        let (orig_w, orig_h) = image.dimensions();
        let (target_h, target_w) = self.smart_resize(orig_h, orig_w);
        let image = if (orig_h, orig_w) != (target_h, target_w) {
            image::imageops::resize(
                &image,
                target_w,
                target_h,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            image
        };

        let grid_h = target_h as usize / self.patch_size;
        let grid_w = target_w as usize / self.patch_size;
        let patch_pixels = self.patch_size * self.patch_size;
        let patch_flat = self.patch_flat_dim();
        let mut patches = vec![0.0f32; grid_h * grid_w * patch_flat];
        let pixels = image.as_raw();
        let mut patch_index = 0usize;
        for block_y in 0..grid_h / self.merge_size {
            for block_x in 0..grid_w / self.merge_size {
                for dy in 0..self.merge_size {
                    for dx in 0..self.merge_size {
                        let patch_y = block_y * self.merge_size + dy;
                        let patch_x = block_x * self.merge_size + dx;
                        for channel in 0..3usize {
                            for temporal in 0..self.temporal_patch_size {
                                for y in 0..self.patch_size {
                                    for x in 0..self.patch_size {
                                        let source_y = patch_y * self.patch_size + y;
                                        let source_x = patch_x * self.patch_size + x;
                                        let pixel_index =
                                            (source_y * target_w as usize + source_x) * 3 + channel;
                                        let value = pixels[pixel_index] as f32 / 255.0;
                                        let normalized = (value - self.image_mean[channel])
                                            / self.image_std[channel];
                                        let target_index = patch_index * patch_flat
                                            + channel * self.temporal_patch_size * patch_pixels
                                            + temporal * patch_pixels
                                            + y * self.patch_size
                                            + x;
                                        patches[target_index] = normalized;
                                    }
                                }
                            }
                        }
                        patch_index += 1;
                    }
                }
            }
        }
        Ok((grid_h, grid_w, patches))
    }
}

fn round_up_to_stride(value: u32, stride: u32) -> u32 {
    value.div_ceil(stride) * stride
}

fn round_to_stride(value: f64, stride: u32) -> u32 {
    let stride_f = stride as f64;
    ((value / stride_f).max(1.0).round() as u32).max(1) * stride
}

pub(super) fn block_major_patch_coords(
    grid_h: usize,
    grid_w: usize,
    merge_size: usize,
) -> Vec<(usize, usize)> {
    if merge_size == 0 || grid_h == 0 || grid_w == 0 {
        return Vec::new();
    }
    let mut coords = Vec::with_capacity(grid_h.saturating_mul(grid_w));
    for block_y in 0..grid_h / merge_size {
        for block_x in 0..grid_w / merge_size {
            for dy in 0..merge_size {
                for dx in 0..merge_size {
                    coords.push((block_y * merge_size + dy, block_x * merge_size + dx));
                }
            }
        }
    }
    coords
}

fn perfect_square_side(entries: usize) -> Option<usize> {
    let side = (entries as f64).sqrt() as usize;
    if side.saturating_mul(side) == entries {
        Some(side)
    } else if (side + 1).saturating_mul(side + 1) == entries {
        Some(side + 1)
    } else {
        None
    }
}

fn bilinear_position_corners(
    row: usize,
    col: usize,
    grid_h: usize,
    grid_w: usize,
    side: usize,
) -> [(usize, f32); 4] {
    let side = side.max(1);
    let h_pos = if grid_h > 1 {
        row as f32 * (side - 1) as f32 / (grid_h - 1) as f32
    } else {
        0.0
    };
    let w_pos = if grid_w > 1 {
        col as f32 * (side - 1) as f32 / (grid_w - 1) as f32
    } else {
        0.0
    };
    let h_floor = h_pos.floor().clamp(0.0, (side - 1) as f32) as usize;
    let w_floor = w_pos.floor().clamp(0.0, (side - 1) as f32) as usize;
    let h_ceil = (h_floor + 1).min(side - 1);
    let w_ceil = (w_floor + 1).min(side - 1);
    let h_frac = h_pos - h_floor as f32;
    let w_frac = w_pos - w_floor as f32;
    [
        (h_floor * side + w_floor, (1.0 - h_frac) * (1.0 - w_frac)),
        (h_floor * side + w_ceil, (1.0 - h_frac) * w_frac),
        (h_ceil * side + w_floor, h_frac * (1.0 - w_frac)),
        (h_ceil * side + w_ceil, h_frac * w_frac),
    ]
}

pub(super) fn apply_vision_spatial_rotary(
    values: &mut [f32],
    row: usize,
    col: usize,
    head_dim: usize,
    theta: f64,
) {
    let theta = theta.max(1.0) as f32;
    let head_dim = head_dim.max(2);
    let rotary_dim = head_dim - (head_dim % 2);
    let half = rotary_dim / 2;
    let spatial_half = half / 2;
    if spatial_half == 0 {
        return;
    }

    for head in values.chunks_mut(head_dim) {
        if head.len() < rotary_dim {
            continue;
        }
        for i in 0..half {
            let (axis_position, axis_index) = if i < spatial_half {
                (row, i)
            } else {
                (col, i - spatial_half)
            };
            let inv_freq = theta.powf(-((2 * axis_index) as f32) / half as f32);
            let angle = axis_position as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos - x1 * sin;
            head[i + half] = x0 * sin + x1 * cos;
        }
    }
}

#[derive(Debug, Clone)]
pub struct VisionEncoder {
    pub(super) config: Qwen3VLVisionConfig,
    pub(super) text_hidden_size: usize,
    pub(super) dense: DenseStore,
}

impl VisionEncoder {
    pub fn from_plan(plan: &FlashMoePlan, text_config: &QwenModelConfig) -> Result<Option<Self>> {
        let (Some(weights), Some(manifest), Some(config)) = (
            plan.vision_weights.as_ref(),
            plan.vision_manifest.as_ref(),
            text_config.vision_config.as_ref(),
        ) else {
            return Ok(None);
        };
        if !weights.is_file() || !manifest.is_file() {
            tracing::debug!(
                model = %plan.model,
                "vision weights not found on disk; VisionEncoder skipped"
            );
            return Ok(None);
        }
        Ok(Some(Self {
            config: config.clone(),
            text_hidden_size: text_config.hidden_size,
            dense: DenseStore::open(weights.clone(), manifest.clone())?,
        }))
    }

    pub fn encode(
        &self,
        preprocessor: &ImagePreprocessor,
        image_path: &Path,
    ) -> Result<VisionEncoding> {
        let (grid_h, grid_w, flat_patches) = preprocessor.preprocess(image_path)?;
        let num_patches = grid_h * grid_w;
        let patch_flat = self.config.patch_flat_dim();
        let mut hidden = (0..num_patches)
            .map(|index| {
                self.patch_embed(&flat_patches[index * patch_flat..(index + 1) * patch_flat])
            })
            .collect::<Result<Vec<_>>>()?;
        self.add_vision_pos_embeds(&mut hidden, grid_h, grid_w)?;

        let mut deepstack_features = Vec::new();
        for layer in 0..self.config.depth {
            self.vit_block(layer, &mut hidden, grid_h, grid_w)?;
            if let Some(merger_index) = self
                .config
                .deepstack_visual_indexes
                .iter()
                .position(|&index| index == layer)
            {
                deepstack_features.push(self.merge_visual_tokens_with_prefix(
                    &format!("visual.deepstack_merger_list.{merger_index}"),
                    true,
                    &hidden,
                    grid_h,
                    grid_w,
                )?);
            }
        }
        let embeddings = self.merge_visual_tokens(&hidden, grid_h, grid_w)?;
        Ok(VisionEncoding {
            embeddings,
            deepstack_features,
            merged_grid_h: grid_h / self.config.merge_size,
            merged_grid_w: grid_w / self.config.merge_size,
        })
    }

    fn patch_embed(&self, patch: &[f32]) -> Result<Vec<f32>> {
        let name = "visual.patch_embed.proj.weight";
        let embed_dim = self.config.embed_dim;
        let entry = self
            .dense
            .registry()
            .tensor(name)
            .with_context(|| format!("vision: required tensor '{name}' is missing"))?;
        if entry.shape.len() < 2 {
            bail!(
                "vision: {name} has shape {:?}; expected Conv3d weight",
                entry.shape
            );
        }
        let rows = entry.shape.first().copied().unwrap_or(0);
        let cols = entry.shape[1..]
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .context("vision: patch embedding shape overflow")?;
        if rows != embed_dim || cols != patch.len() {
            bail!(
                "vision: {name} shape {:?} is incompatible with embed_dim {embed_dim} and patch len {}",
                entry.shape,
                patch.len()
            );
        }
        let weights = self
            .dense
            .read_full_tensor_f32(name)?
            .with_context(|| format!("vision: required tensor '{name}' is missing"))?;
        let mut projected = vec![0.0f32; embed_dim];
        for row in 0..embed_dim {
            projected[row] = weights[row * cols..(row + 1) * cols]
                .iter()
                .zip(patch.iter())
                .map(|(weight, value)| weight * value)
                .sum();
        }
        self.vit_add_bias("visual.patch_embed.proj.bias", projected)
    }

    fn add_vision_pos_embeds(
        &self,
        hidden: &mut [Vec<f32>],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<()> {
        let pos_embed = self
            .dense
            .read_full_tensor_f32("visual.pos_embed.weight")?
            .context("vision: required tensor 'visual.pos_embed.weight' is missing")?;
        let embed_dim = self.config.embed_dim;
        if embed_dim == 0 || pos_embed.len() % embed_dim != 0 {
            bail!(
                "vision: visual.pos_embed.weight has {} values; expected a multiple of embed_dim {embed_dim}",
                pos_embed.len()
            );
        }
        let entries = pos_embed.len() / embed_dim;
        let side = perfect_square_side(entries).with_context(|| {
            format!("vision: visual.pos_embed.weight has {entries} entries, not a square table")
        })?;
        if let Some(config_entries) = self.config.num_position_embeddings
            && config_entries != entries
        {
            bail!(
                "vision: config num_position_embeddings={config_entries} but visual.pos_embed.weight has {entries} rows"
            );
        }
        let coords = block_major_patch_coords(grid_h, grid_w, self.config.merge_size);
        if coords.len() != hidden.len() {
            bail!(
                "vision: {} patch coordinates for {} hidden patches",
                coords.len(),
                hidden.len()
            );
        }
        for (patch, (row, col)) in hidden.iter_mut().zip(coords) {
            for (index, weight) in bilinear_position_corners(row, col, grid_h, grid_w, side) {
                let start = index * embed_dim;
                for (value, position) in patch
                    .iter_mut()
                    .zip(pos_embed[start..start + embed_dim].iter())
                {
                    *value += weight * *position;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MropePosition {
    pub(super) temporal: usize,
    pub(super) height: usize,
    pub(super) width: usize,
}

impl MropePosition {
    pub(super) fn text(position: usize) -> Self {
        Self {
            temporal: position,
            height: position,
            width: position,
        }
    }

    pub(super) fn axis(self, axis: MropeAxis) -> usize {
        match axis {
            MropeAxis::Temporal => self.temporal,
            MropeAxis::Height => self.height,
            MropeAxis::Width => self.width,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MropeAxis {
    Temporal,
    Height,
    Width,
}

#[derive(Debug, Clone)]
pub struct VisionEncoding {
    pub embeddings: Vec<Vec<f32>>,
    pub deepstack_features: Vec<Vec<Vec<f32>>>,
    pub merged_grid_h: usize,
    pub merged_grid_w: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ImagePlaceholderSpec {
    token_count: usize,
    grid_h: usize,
    grid_w: usize,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VisualTokenSpan {
    start: usize,
    end: usize,
    grid_h: usize,
    grid_w: usize,
}

impl VisualTokenSpan {
    fn image(start: usize, end: usize, grid_h: usize, grid_w: usize) -> Self {
        Self {
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
        self.grid_h.saturating_mul(self.grid_w)
    }

    fn position_advance(self) -> usize {
        self.grid_h.max(self.grid_w)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct QwenVlRuntimeInputs {
    prompt_tokens: Vec<u32>,
    visual_embeddings: Vec<Vec<f32>>,
    deepstack_features: Vec<Vec<Vec<f32>>>,
    image_pad_token: u32,
    mrope_positions: Vec<MropePosition>,
    next_mrope_position: usize,
}

impl QwenVlRuntimeInputs {
    pub(super) fn build(
        prompt_tokens: Vec<u32>,
        vision_start_token: u32,
        vision_end_token: u32,
        image_pad_token: u32,
        visual_encodings: Vec<VisionEncoding>,
    ) -> Result<Self> {
        if visual_encodings.is_empty() {
            bail!("Qwen-VL runtime inputs require at least one visual encoding");
        }

        let mut image_specs = Vec::with_capacity(visual_encodings.len());
        let mut visual_embeddings = Vec::new();
        let mut deepstack_features = None::<Vec<Vec<Vec<f32>>>>;
        for visual in visual_encodings {
            let VisionEncoding {
                embeddings,
                deepstack_features: visual_deepstack,
                merged_grid_h,
                merged_grid_w,
            } = visual;
            let spec = ImagePlaceholderSpec {
                token_count: embeddings.len(),
                grid_h: merged_grid_h,
                grid_w: merged_grid_w,
            };
            spec.validate(image_specs.len())?;
            image_specs.push(spec);
            visual_embeddings.extend(embeddings);
            if let Some(accumulated_deepstack) = deepstack_features.as_mut() {
                if accumulated_deepstack.len() != visual_deepstack.len() {
                    bail!(
                        "vision images produced incompatible DeepStack feature depths: {} vs {}",
                        accumulated_deepstack.len(),
                        visual_deepstack.len()
                    );
                }
                for (dst, mut src) in accumulated_deepstack.iter_mut().zip(visual_deepstack) {
                    dst.append(&mut src);
                }
            } else {
                deepstack_features = Some(visual_deepstack);
            }
        }

        let (prompt_tokens, visual_spans) = expand_image_placeholders(
            prompt_tokens,
            vision_start_token,
            vision_end_token,
            image_pad_token,
            &image_specs,
        )?;
        let (mrope_positions, next_mrope_position) =
            multimodal_mrope_positions(&prompt_tokens, image_pad_token, &visual_spans)?;
        Ok(Self {
            prompt_tokens,
            visual_embeddings,
            deepstack_features: deepstack_features.unwrap_or_default(),
            image_pad_token,
            mrope_positions,
            next_mrope_position,
        })
    }

    pub(super) fn prompt_tokens(&self) -> &[u32] {
        &self.prompt_tokens
    }

    pub(super) fn visual_embeddings(&self) -> &[Vec<f32>] {
        &self.visual_embeddings
    }

    pub(super) fn deepstack_features(&self) -> &[Vec<Vec<f32>>] {
        &self.deepstack_features
    }

    pub(super) fn image_pad_token(&self) -> u32 {
        self.image_pad_token
    }

    pub(super) fn mrope_positions(&self) -> &[MropePosition] {
        &self.mrope_positions
    }

    pub(super) fn next_mrope_position(&self) -> usize {
        self.next_mrope_position
    }
}

fn expand_image_placeholders(
    prompt_tokens: Vec<u32>,
    vision_start_token: u32,
    vision_end_token: u32,
    image_pad_token: u32,
    image_specs: &[ImagePlaceholderSpec],
) -> Result<(Vec<u32>, Vec<VisualTokenSpan>)> {
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
    Ok((expanded, visual_spans))
}

fn multimodal_mrope_positions(
    prompt_tokens: &[u32],
    image_pad_token: u32,
    visual_spans: &[VisualTokenSpan],
) -> Result<(Vec<MropePosition>, usize)> {
    let actual_image_tokens = prompt_tokens
        .iter()
        .filter(|&&token| token == image_pad_token)
        .count();
    let expected_image_tokens = visual_spans.iter().map(|span| span.len()).sum::<usize>();
    if actual_image_tokens != expected_image_tokens {
        bail!(
            "image placeholder count {actual_image_tokens} does not match expected visual token count {expected_image_tokens}"
        );
    }

    let mut positions = Vec::with_capacity(prompt_tokens.len());
    let mut current_pos = 0usize;
    let mut token_index = 0usize;
    let mut span_index = 0usize;
    while token_index < prompt_tokens.len() {
        if span_index < visual_spans.len() && token_index == visual_spans[span_index].start {
            let span = visual_spans[span_index];
            if span.end < span.start || span.end > prompt_tokens.len() {
                bail!(
                    "image span {span_index} has invalid bounds {}..{} for prompt length {}",
                    span.start,
                    span.end,
                    prompt_tokens.len()
                );
            }
            if span.grid_h == 0 || span.grid_w == 0 || span.len() != span.expected_token_count() {
                bail!(
                    "image span {span_index} does not match its declared {}x{} merged grid",
                    span.grid_h,
                    span.grid_w
                );
            }
            let start_position = current_pos;
            let mut image_index = 0usize;
            while token_index < span.end {
                if prompt_tokens[token_index] != image_pad_token {
                    bail!("image span {span_index} contains a non-placeholder token");
                }
                positions.push(MropePosition {
                    temporal: start_position,
                    height: start_position + image_index / span.grid_w,
                    width: start_position + image_index % span.grid_w,
                });
                image_index += 1;
                token_index += 1;
            }
            current_pos += span.position_advance();
            span_index += 1;
        } else if prompt_tokens[token_index] == image_pad_token {
            bail!("image placeholder at token {token_index} is not part of a visual span");
        } else {
            positions.push(MropePosition::text(current_pos));
            current_pos += 1;
            token_index += 1;
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

fn token_run_bounds(tokens: &[u32], needle: u32) -> Vec<(usize, usize, usize)> {
    let mut runs = Vec::new();
    let mut start = None;
    let mut count = 0usize;
    for (index, &token) in tokens.iter().enumerate() {
        if token == needle {
            if start.is_none() {
                start = Some(index);
            }
            count += 1;
        } else if let Some(run_start) = start.take() {
            runs.push((run_start, index, count));
            count = 0;
        }
    }
    if let Some(run_start) = start {
        runs.push((run_start, tokens.len(), count));
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoding(grid_h: usize, grid_w: usize, deepstack_depth: usize) -> VisionEncoding {
        let tokens = grid_h * grid_w;
        VisionEncoding {
            embeddings: (0..tokens).map(|token| vec![token as f32]).collect(),
            deepstack_features: (0..deepstack_depth)
                .map(|depth| {
                    (0..tokens)
                        .map(|token| vec![(depth * 100 + token) as f32])
                        .collect()
                })
                .collect(),
            merged_grid_h: grid_h,
            merged_grid_w: grid_w,
        }
    }

    #[test]
    fn qwen_vl_runtime_inputs_expand_images_and_own_mrope_and_deepstack() {
        let inputs = QwenVlRuntimeInputs::build(
            vec![10, 99, 11, 99, 12],
            97,
            98,
            99,
            vec![encoding(1, 2, 2), encoding(2, 1, 2)],
        )
        .unwrap();

        assert_eq!(
            inputs.prompt_tokens(),
            [10, 97, 99, 99, 98, 11, 97, 99, 99, 98, 12]
        );
        assert_eq!(inputs.visual_embeddings().len(), 4);
        assert_eq!(inputs.deepstack_features().len(), 2);
        assert_eq!(inputs.deepstack_features()[0].len(), 4);
        assert_eq!(inputs.image_pad_token(), 99);
        assert_eq!(inputs.mrope_positions().len(), inputs.prompt_tokens().len());
        assert_eq!(inputs.next_mrope_position(), 11);
        assert_eq!(
            inputs.mrope_positions()[3],
            MropePosition {
                temporal: 2,
                height: 2,
                width: 3,
            }
        );
    }

    #[test]
    fn qwen_vl_runtime_inputs_reject_incompatible_deepstack_depths() {
        let error = QwenVlRuntimeInputs::build(
            vec![99, 99],
            97,
            98,
            99,
            vec![encoding(1, 1, 1), encoding(1, 1, 2)],
        )
        .unwrap_err();

        assert!(error.to_string().contains("incompatible DeepStack"));
    }
}
