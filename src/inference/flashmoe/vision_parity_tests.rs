//! Owner-local Qwen-VL preprocessing, placeholder, and MRoPE parity tests.

use super::*;
use crate::inference::flashmoe::math::apply_rotary_split_half_mrope;
use crate::inference::flashmoe::test_fixtures::assert_close;
use crate::inference::flashmoe::text::{
    QwenTokenizer, test_qwen3vl_tokenizer_json, test_qwen3vl_tool_tokenizer_config_json,
};
use crate::inference::flashmoe::types::*;

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

    let err =
        qwen3vl_multimodal_mrope_positions(&[9, 9, 9], 9, &[VisualTokenSpan::image(0, 3, 2, 2)])
            .unwrap_err();
    assert!(
        err.to_string()
            .contains("image span 0 does not match its declared 2x2 merged grid"),
        "{err:#}"
    );

    let err = qwen3vl_multimodal_mrope_positions(&[9, 1], 9, &[VisualTokenSpan::image(0, 2, 1, 2)])
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
        qwen3vl_multimodal_mrope_positions(&expanded.tokens, 9, &expanded.visual_spans).unwrap();
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
