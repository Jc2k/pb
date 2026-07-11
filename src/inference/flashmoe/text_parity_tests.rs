//! Owner-local tokenizer, sampling, chat-template, and tool-call parity tests.

use super::*;
use crate::inference::flashmoe::planning::plan_unchecked;
use crate::inference::flashmoe::types::QWEN35_MODEL;

pub(crate) fn test_tokenizer_json() -> &'static [u8] {
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

pub(crate) fn test_tokenizer_config_json() -> &'static [u8] {
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

pub(crate) fn test_qwen3_tool_tokenizer_config_json() -> &'static [u8] {
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

pub(crate) fn test_qwen3vl_tool_tokenizer_config_json() -> &'static [u8] {
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

pub(crate) fn test_qwen3vl_tokenizer_json() -> &'static [u8] {
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
