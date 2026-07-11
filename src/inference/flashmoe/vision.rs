use anyhow::{Result, bail};

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
