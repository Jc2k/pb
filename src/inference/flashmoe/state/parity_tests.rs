//! Owner-local session-prefix parity across Qwen tool-call rerenders.

use super::*;
use crate::inference::flashmoe::text::{
    QwenTokenizer, test_qwen3_tool_tokenizer_config_json, test_tokenizer_json,
};
use crate::inference::flashmoe::types::*;

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
    let (first_prompt, next_prompt) = rendered_tool_prompt_pair(assistant_weather_tool_call(""));
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
