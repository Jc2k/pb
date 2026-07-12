use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;
use tokenizers::Tokenizer;

use super::math::{compare_scored_tokens, softmax_in_place};
use super::runtime::GenerationProgress;
use super::types::*;
use crate::inference::chat_template::{ChatTemplateOptions, TokenizerChatTemplate};

pub(super) fn trace_sampling_candidates(
    progress: &GenerationProgress<'_>,
    tokenizer: &QwenTokenizer,
    prompt_len: usize,
    generated: &[u32],
    candidates: &[(usize, f32)],
    vector_stats: Option<(&[f32], &[f32])>,
) {
    let Some((hidden, logits)) = vector_stats else {
        return;
    };
    let rendered = candidates
        .iter()
        .enumerate()
        .map(|(idx, (token, score))| {
            format!(
                "#{rank}:id={token}:score={score:.6}:text={text}",
                rank = idx + 1,
                text = trace_token_text(tokenizer, *token)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    let (hidden_rms, hidden_max, hidden_finite) = vector_rms_max_finite(hidden);
    let (logits_rms, logits_max, logits_finite) = vector_rms_max_finite(logits);
    report_generation_progress(progress, || {
        format!(
            "sampling candidates prompt_tokens={} generated_tokens={} hidden_rms={hidden_rms:.6} hidden_max={hidden_max:.6} hidden_finite={hidden_finite} logits_rms={logits_rms:.6} logits_max={logits_max:.6} logits_finite={logits_finite} {}",
            prompt_len,
            generated.len(),
            rendered
        )
    });
}

fn trace_token_text(tokenizer: &QwenTokenizer, token: usize) -> String {
    let Ok(token) = u32::try_from(token) else {
        return "\"<id-overflow>\"".to_string();
    };
    let decoded = tokenizer
        .decode(&[token])
        .unwrap_or_else(|_| "<decode-error>".to_string());
    let escaped = decoded
        .chars()
        .flat_map(|ch| ch.escape_default())
        .collect::<String>();
    format!("\"{escaped}\"")
}

fn vector_rms_max_finite(values: &[f32]) -> (f32, f32, bool) {
    if values.is_empty() {
        return (0.0, 0.0, true);
    }
    let mut sum_square = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut finite = true;
    for value in values {
        finite &= value.is_finite();
        sum_square += (*value as f64) * (*value as f64);
        max_abs = max_abs.max(value.abs());
    }
    let rms = (sum_square / values.len() as f64).sqrt() as f32;
    (rms, max_abs, finite)
}

#[derive(Debug, Clone)]
pub(super) struct QwenTokenizer {
    tokenizer: Tokenizer,
    config: QwenTokenizerConfig,
    eos_tokens: BTreeSet<u32>,
    primary_eos_token: u32,
    im_start: Option<u32>,
    im_end: Option<u32>,
    vocab_size: usize,
    #[cfg(test)]
    candidate_ids: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
struct QwenTokenizerConfig {
    bos_token: Option<String>,
    eos_tokens: Vec<String>,
    pad_token: Option<String>,
    add_bos_token: bool,
    added_tokens_decoder: BTreeMap<u32, TokenizerConfigAddedToken>,
    additional_special_tokens: Vec<String>,
    split_special_tokens: bool,
    chat_template: Option<TokenizerChatTemplate>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenizerConfigAddedToken {
    content: String,
    special: bool,
}

impl QwenTokenizer {
    pub(super) fn from_files(tokenizer_path: &Path, config_path: &Path) -> Result<Self> {
        if !config_path.is_file() {
            bail!(
                "Flash-MoE tokenizer config is required for chat generation: missing {}",
                config_path.display()
            );
        }
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|err| anyhow::anyhow!("{err}"))
            .with_context(|| format!("failed to load tokenizer {}", tokenizer_path.display()))?;
        let bytes = fs::read(tokenizer_path)
            .with_context(|| format!("failed to read tokenizer {}", tokenizer_path.display()))?;
        let config_bytes = fs::read(config_path).with_context(|| {
            format!("failed to read tokenizer config {}", config_path.display())
        })?;
        Self::from_json_bytes_with_tokenizer(&bytes, tokenizer, &config_bytes).with_context(|| {
            format!(
                "failed to load Flash-MoE tokenizer metadata from {} and {}",
                tokenizer_path.display(),
                config_path.display()
            )
        })
    }

    #[cfg(test)]
    pub(super) fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_json_bytes_with_config(bytes, Some(test_default_tokenizer_config_json()))
    }

    #[cfg(test)]
    pub(super) fn from_json_bytes_with_config(
        bytes: &[u8],
        config_bytes: Option<&[u8]>,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|err| anyhow::anyhow!("{err}"))
            .context("tokenizer JSON is invalid")?;
        let config_bytes = config_bytes.context("test tokenizer config is required")?;
        Self::from_json_bytes_with_tokenizer(bytes, tokenizer, config_bytes)
    }

    fn from_json_bytes_with_tokenizer(
        bytes: &[u8],
        tokenizer: Tokenizer,
        config_bytes: &[u8],
    ) -> Result<Self> {
        let _ = bytes;
        let config = QwenTokenizerConfig::from_bytes(config_bytes)?;
        if config.split_special_tokens {
            bail!(
                "tokenizer_config.json sets split_special_tokens=true, which is unsupported for Flash-MoE because generation stop tokens must remain atomic"
            );
        }
        let vocabulary = tokenizer.get_vocab(true);
        if vocabulary.is_empty() {
            bail!("Qwen tokenizer JSON does not contain model.vocab");
        }
        let mut eos_tokens = BTreeSet::new();
        for token in &config.eos_tokens {
            let id = tokenizer.token_to_id(token).with_context(|| {
                format!(
                    "tokenizer_config.json eos_token {token:?} is not present in tokenizer.json"
                )
            })?;
            eos_tokens.insert(id);
        }
        let primary_eos_token = eos_tokens
            .iter()
            .next()
            .copied()
            .context("tokenizer_config.json must define eos_token for Flash-MoE")?;
        if let Some(token) = &config.bos_token
            && tokenizer.token_to_id(token).is_none()
        {
            bail!("tokenizer_config.json bos_token {token:?} is not present in tokenizer.json");
        }
        if let Some(token) = &config.pad_token
            && tokenizer.token_to_id(token).is_none()
        {
            bail!("tokenizer_config.json pad_token {token:?} is not present in tokenizer.json");
        }
        // Qwen-family tokenizer_config.json files may include modality tokens in
        // added_tokens_decoder/additional_special_tokens that are not exposed by
        // the active tokenizer.json. Keep EOS/BOS/PAD strict, but let optional
        // decoder metadata be advisory so text-only model artifacts still load.
        for token in &config.additional_special_tokens {
            let _ = tokenizer.token_to_id(token);
        }
        for token in config.added_tokens_decoder.values() {
            let _ = tokenizer.token_to_id(&token.content);
        }
        let im_start = tokenizer.token_to_id("<|im_start|>");
        let im_end = tokenizer.token_to_id("<|im_end|>");
        let max_id = vocabulary
            .values()
            .copied()
            .max()
            .unwrap_or(primary_eos_token) as usize;
        let vocab_size = max_id + 1;
        #[cfg(test)]
        let candidate_ids = {
            let mut ids: Vec<u32> = vocabulary
                .values()
                .copied()
                .filter(|id| (*id as usize) < vocab_size)
                .collect();
            ids.sort_unstable();
            ids.dedup();
            if ids.is_empty() {
                bail!("Qwen tokenizer vocabulary is empty");
            }
            ids
        };
        Ok(Self {
            tokenizer,
            config,
            eos_tokens,
            primary_eos_token,
            im_start,
            im_end,
            vocab_size,
            #[cfg(test)]
            candidate_ids,
        })
    }

    #[cfg(test)]
    pub(super) fn apply_chat_template(&self, prompt: &str) -> String {
        if prompt.contains("<|im_start|>") {
            return prompt.to_string();
        }
        if let Some(template) = &self.config.chat_template {
            return render_tokenizer_chat_template(
                template,
                &[ChatMessage::text(ChatRole::User, prompt)],
                &[],
                true,
            )
            .unwrap_or_else(|_| prompt.to_string());
        }
        if self.im_start.is_some() && self.im_end.is_some() && !prompt.contains("<|im_start|>") {
            format!(
                "<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"
            )
        } else {
            prompt.to_string()
        }
    }

    pub(super) fn apply_chat_template_to_messages(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        add_generation_prompt: bool,
    ) -> Result<String> {
        if messages.len() == 1
            && tools.is_empty()
            && let ChatMessageContent::Text(prompt) = &messages[0].content
            && prompt.contains("<|im_start|>")
        {
            return Ok(prompt.clone());
        }
        if let Some(template) = self.config.chat_template.as_ref() {
            return render_tokenizer_chat_template(
                template,
                messages,
                tools,
                add_generation_prompt,
            );
        }
        if self.im_start.is_some() && self.im_end.is_some() {
            return render_qwen_chatml(messages, tools, add_generation_prompt);
        }
        bail!(
            "tokenizer_config.json is missing chat_template and tokenizer.json is missing Qwen chat special tokens; Flash-MoE chat generation requires one of them"
        )
    }

    pub(super) fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| anyhow::anyhow!("{err}"))
            .with_context(|| format!("failed to encode text with tokenizer.json"))?;
        let mut ids = encoding.get_ids().to_vec();
        if self.config.add_bos_token
            && let Some(bos) = self
                .config
                .bos_token
                .as_deref()
                .and_then(|token| self.tokenizer.token_to_id(token))
            && ids.first().copied() != Some(bos)
        {
            ids.insert(0, bos);
        }
        Ok(ids)
    }

    pub(super) fn decode(&self, tokens: &[u32]) -> Result<String> {
        let tokens: Vec<u32> = tokens
            .iter()
            .copied()
            .take_while(|token| !self.is_eos(*token))
            .collect();
        self.tokenizer
            .decode(&tokens, true)
            .map_err(|err| anyhow::anyhow!("{err}"))
            .context("failed to decode tokens with tokenizer.json")
    }

    pub(super) fn is_eos(&self, token: u32) -> bool {
        self.eos_tokens.contains(&token)
    }

    pub(super) fn eos_token_id(&self) -> u32 {
        self.primary_eos_token
    }

    /// Look up a token string and return its ID, or `None` if not present.
    pub(super) fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub(super) fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[cfg(test)]
    pub(super) fn candidate_token_ids(&self) -> &[u32] {
        &self.candidate_ids
    }
}

impl QwenTokenizerConfig {
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let value: Value =
            serde_json::from_slice(bytes).context("tokenizer_config.json is invalid")?;
        let bos_token = config_token_string(value.get("bos_token"), "bos_token")?;
        let eos_tokens = config_token_strings(value.get("eos_token"), "eos_token")?;
        let pad_token = config_token_string(value.get("pad_token"), "pad_token")?;
        let add_bos_token = value
            .get("add_bos_token")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let split_special_tokens = value
            .get("split_special_tokens")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let added_tokens_decoder = parse_added_tokens_decoder(value.get("added_tokens_decoder"))?;
        let additional_special_tokens = parse_config_token_list(
            value.get("additional_special_tokens"),
            "additional_special_tokens",
        )?;
        let chat_template = TokenizerChatTemplate::from_tokenizer_config_value(&value)?;
        if eos_tokens.is_empty() {
            bail!("tokenizer_config.json must define eos_token for Flash-MoE");
        }
        Ok(Self {
            bos_token,
            eos_tokens,
            pad_token,
            add_bos_token,
            added_tokens_decoder,
            additional_special_tokens,
            split_special_tokens,
            chat_template,
        })
    }
}

fn config_token_strings(value: Option<&Value>, field: &str) -> Result<Vec<String>> {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                config_token_string(Some(item), field)?.with_context(|| {
                    format!("tokenizer_config.json {field} entries must not be null")
                })
            })
            .collect(),
        _ => Ok(config_token_string(value, field)?.into_iter().collect()),
    }
}

fn config_token_string(value: Option<&Value>, field: &str) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(token)) => Ok(Some(token.clone())),
        Some(Value::Object(object)) => object
            .get("content")
            .and_then(Value::as_str)
            .map(|content| Some(content.to_string()))
            .with_context(|| {
                format!("tokenizer_config.json {field} object must contain string content")
            }),
        Some(_) => bail!("tokenizer_config.json {field} must be a string, object, array, or null"),
    }
}

fn parse_config_token_list(value: Option<&Value>, field: &str) -> Result<Vec<String>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let Some(items) = value.as_array() else {
        bail!("tokenizer_config.json {field} must be an array");
    };
    let mut tokens = Vec::new();
    for item in items {
        if let Some(token) = config_token_string(Some(item), field)? {
            tokens.push(token);
        }
    }
    Ok(tokens)
}

fn parse_added_tokens_decoder(
    value: Option<&Value>,
) -> Result<BTreeMap<u32, TokenizerConfigAddedToken>> {
    let Some(value) = value else {
        return Ok(BTreeMap::new());
    };
    let Some(object) = value.as_object() else {
        bail!("tokenizer_config.json added_tokens_decoder must be an object");
    };
    let mut out = BTreeMap::new();
    for (id, value) in object {
        let id = id
            .parse::<u32>()
            .with_context(|| format!("invalid added_tokens_decoder id {id:?}"))?;
        let content = config_token_string(Some(value), "added_tokens_decoder")?
            .context("added_tokens_decoder entries must contain token content")?;
        let special = value
            .as_object()
            .and_then(|object| object.get("special"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        out.insert(id, TokenizerConfigAddedToken { content, special });
    }
    Ok(out)
}

fn render_tokenizer_chat_template(
    template: &TokenizerChatTemplate,
    messages: &[ChatMessage],
    tools: &[ChatTool],
    add_generation_prompt: bool,
) -> Result<String> {
    let tools: Vec<Value> = tools.iter().map(qwen_tool_schema_value).collect();
    template.render(
        messages,
        &tools,
        ChatTemplateOptions {
            add_generation_prompt,
            ..ChatTemplateOptions::default()
        },
    )
}

fn render_qwen_chatml(
    messages: &[ChatMessage],
    tools: &[ChatTool],
    add_generation_prompt: bool,
) -> Result<String> {
    render_qwen_xml_tool_template(messages, tools, add_generation_prompt, true)
}

fn render_qwen_xml_tool_template(
    messages: &[ChatMessage],
    tools: &[ChatTool],
    add_generation_prompt: bool,
    supports_vision_parts: bool,
) -> Result<String> {
    let mut out = String::new();

    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        if let Some(message) = messages
            .first()
            .filter(|message| message.role == ChatRole::System)
        {
            let content = render_qwen_template_content(&message.content, supports_vision_parts)?;
            if !content.is_empty() {
                out.push_str(&content);
                out.push_str("\n\n");
            }
        }
        out.push_str(&render_qwen_tool_instructions("", tools)?);
        out.push_str("<|im_end|>\n");
    } else if let Some(message) = messages
        .first()
        .filter(|message| message.role == ChatRole::System)
    {
        out.push_str("<|im_start|>system\n");
        out.push_str(&render_qwen_template_content(
            &message.content,
            supports_vision_parts,
        )?);
        out.push_str("<|im_end|>\n");
    }

    let last_query_index = qwen_last_real_user_index(messages);
    for (index, message) in messages.iter().enumerate() {
        if index == 0 && message.role == ChatRole::System {
            continue;
        }
        match message.role {
            ChatRole::System | ChatRole::User => {
                out.push_str("<|im_start|>");
                out.push_str(message.role.as_qwen_role());
                out.push('\n');
                out.push_str(&render_qwen_template_content(
                    &message.content,
                    supports_vision_parts,
                )?);
                out.push_str("<|im_end|>\n");
            }
            ChatRole::Assistant => {
                out.push_str("<|im_start|>assistant\n");
                let content = render_qwen_assistant_content(
                    &message.content,
                    supports_vision_parts,
                    index,
                    last_query_index,
                )?;
                out.push_str(&content.rendered);
                if !content.is_empty() && !message.tool_calls.is_empty() && !content.ends_with('\n')
                {
                    out.push('\n');
                }
                for tool_call in &message.tool_calls {
                    out.push_str("<tool_call>\n");
                    out.push_str(&render_qwen_tool_call_json(tool_call)?);
                    out.push_str("\n</tool_call>\n");
                }
                out.push_str("<|im_end|>\n");
            }
            ChatRole::Tool => {
                if index == 0
                    || messages
                        .get(index.saturating_sub(1))
                        .map_or(true, |previous| previous.role != ChatRole::Tool)
                {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(&render_qwen_template_content(
                    &message.content,
                    supports_vision_parts,
                )?);
                out.push_str("\n</tool_response>");
                if index + 1 == messages.len()
                    || messages
                        .get(index + 1)
                        .map_or(true, |next| next.role != ChatRole::Tool)
                {
                    out.push_str("<|im_end|>\n");
                }
            }
        }
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RenderedAssistantContent {
    rendered: String,
    logical_content: String,
}

impl RenderedAssistantContent {
    fn is_empty(&self) -> bool {
        self.logical_content.is_empty()
    }

    fn ends_with(&self, ch: char) -> bool {
        self.logical_content.ends_with(ch)
    }
}

fn qwen_last_real_user_index(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, message)| {
            message.role == ChatRole::User
                && match &message.content {
                    ChatMessageContent::Text(content) => {
                        !(content.starts_with("<tool_response>")
                            && content.ends_with("</tool_response>"))
                    }
                    ChatMessageContent::Parts(_) => true,
                }
        })
        .map(|(index, _)| index)
        .unwrap_or_else(|| messages.len().saturating_sub(1))
}

fn render_qwen_assistant_content(
    content: &ChatMessageContent,
    supports_vision_parts: bool,
    index: usize,
    last_query_index: usize,
) -> Result<RenderedAssistantContent> {
    let mut logical_content = render_qwen_template_content(content, supports_vision_parts)?;
    if supports_vision_parts {
        return Ok(RenderedAssistantContent {
            rendered: logical_content.clone(),
            logical_content,
        });
    }

    let mut reasoning_content = String::new();
    if let Some((before, after)) = logical_content.split_once("</think>") {
        reasoning_content = before
            .trim_end_matches('\n')
            .rsplit_once("<think>")
            .map(|(_, reasoning)| reasoning)
            .unwrap_or(before)
            .trim_start_matches('\n')
            .to_string();
        logical_content = after.trim_start_matches('\n').to_string();
    }

    let rendered = if index > last_query_index && !reasoning_content.is_empty() {
        format!(
            "<think>\n{}\n</think>\n\n{}",
            reasoning_content.trim_matches('\n'),
            logical_content.trim_start_matches('\n')
        )
    } else {
        logical_content.clone()
    };
    Ok(RenderedAssistantContent {
        rendered,
        logical_content,
    })
}

fn render_qwen_template_content(
    content: &ChatMessageContent,
    supports_vision_parts: bool,
) -> Result<String> {
    match content {
        ChatMessageContent::Text(text) => Ok(text.clone()),
        ChatMessageContent::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatContentPart::Text { text } => out.push_str(text),
                    ChatContentPart::Image {
                        placeholder_tokens, ..
                    } => {
                        if !supports_vision_parts {
                            bail!(
                                "tokenizer chat_template does not support image content parts for this model family"
                            );
                        }
                        out.push_str("<|vision_start|>");
                        out.push_str(&"<|image_pad|>".repeat((*placeholder_tokens).unwrap_or(1)));
                        out.push_str("<|vision_end|>");
                    }
                }
            }
            Ok(out)
        }
    }
}

fn render_qwen_tool_instructions(system_content: &str, tools: &[ChatTool]) -> Result<String> {
    if tools.is_empty() {
        return Ok(system_content.to_string());
    }

    let mut out = String::new();
    if !system_content.is_empty() {
        out.push_str(system_content);
        if !system_content.ends_with('\n') {
            out.push_str("\n\n");
        }
    }
    out.push_str("# Tools\n\n");
    out.push_str("You may call one or more functions to assist with the user query.\n\n");
    out.push_str("You are provided with function signatures within <tools></tools> XML tags:\n");
    out.push_str("<tools>\n");
    for tool in tools {
        out.push_str(&serde_json::to_string(&qwen_tool_schema_value(tool))?);
        out.push('\n');
    }
    out.push_str("</tools>\n\n");
    out.push_str("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n");
    out.push_str("<tool_call>\n");
    out.push_str("{\"name\": <function-name>, \"arguments\": <args-json-object>}\n");
    out.push_str("</tool_call>");
    Ok(out)
}

pub(super) fn qwen_tool_schema_value(tool: &ChatTool) -> Value {
    let mut function = serde_json::Map::new();
    function.insert("name".to_string(), Value::String(tool.name.clone()));
    if let Some(description) = &tool.description {
        function.insert(
            "description".to_string(),
            Value::String(description.clone()),
        );
    }
    function.insert("parameters".to_string(), tool.input_schema.clone());

    let mut root = serde_json::Map::new();
    root.insert("type".to_string(), Value::String("function".to_string()));
    root.insert("function".to_string(), Value::Object(function));
    Value::Object(root)
}

fn render_qwen_tool_call_json(tool_call: &ChatToolCall) -> Result<String> {
    let name = serde_json::to_string(&tool_call.name)?;
    let arguments = match &tool_call.arguments {
        Value::String(arguments) => arguments.clone(),
        arguments => serde_json::to_string(arguments)?,
    };
    Ok(format!("{{\"name\": {name}, \"arguments\": {arguments}}}"))
}

pub(super) fn parse_qwen_tool_call_output(content: &str) -> Result<(String, Vec<ChatToolCall>)> {
    let mut remaining = content;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    while let Some(start) = remaining.find("<tool_call>") {
        text.push_str(&remaining[..start]);
        let block_start = start + "<tool_call>".len();
        let Some(relative_end) = remaining[block_start..].find("</tool_call>") else {
            text.push_str(&remaining[start..]);
            return Ok((text.trim().to_string(), tool_calls));
        };
        let block_end = block_start + relative_end;
        let block = remaining[block_start..block_end].trim();
        if !block.is_empty() {
            tool_calls.push(parse_qwen_tool_call_block(block)?);
        }
        remaining = &remaining[block_end + "</tool_call>".len()..];
    }
    text.push_str(remaining);
    Ok((text.trim().to_string(), tool_calls))
}

fn parse_qwen_tool_call_block(block: &str) -> Result<ChatToolCall> {
    if block.contains("<function=") {
        return parse_qwen_function_tool_call_block(block);
    }

    let value: Value = serde_json::from_str(block)
        .with_context(|| format!("failed to parse Qwen tool call JSON: {block}"))?;
    let name = value
        .get("name")
        .or_else(|| value.pointer("/function/name"))
        .and_then(Value::as_str)
        .context("Qwen tool call is missing a string name")?
        .to_string();
    let arguments = value
        .get("arguments")
        .or_else(|| value.pointer("/function/arguments"))
        .map(parse_qwen_tool_arguments)
        .transpose()?
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
    let id = value
        .get("id")
        .or_else(|| value.get("tool_call_id"))
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok(ChatToolCall {
        id,
        name,
        arguments,
    })
}

fn parse_qwen_function_tool_call_block(block: &str) -> Result<ChatToolCall> {
    let start = block
        .find("<function=")
        .context("Qwen function tool call is missing <function=...>")?;
    let name_start = start + "<function=".len();
    let name_end = block[name_start..]
        .find('>')
        .map(|end| name_start + end)
        .context("Qwen function tool call has an unterminated function tag")?;
    let name = block[name_start..name_end].trim();
    if name.is_empty() {
        bail!("Qwen function tool call is missing a function name");
    }

    let body_start = name_end + 1;
    let body_end = block[body_start..]
        .rfind("</function>")
        .map(|end| body_start + end)
        .context("Qwen function tool call is missing </function>")?;
    let mut rest = &block[body_start..body_end];
    let mut arguments = serde_json::Map::new();
    while let Some(parameter_start) = rest.find("<parameter=") {
        rest = &rest[parameter_start + "<parameter=".len()..];
        let Some(name_end) = rest.find('>') else {
            bail!("Qwen function tool call has an unterminated parameter tag");
        };
        let parameter_name = rest[..name_end].trim();
        if parameter_name.is_empty() {
            bail!("Qwen function tool call has an empty parameter name");
        }
        rest = &rest[name_end + 1..];
        let Some(value_end) = rest.find("</parameter>") else {
            bail!("Qwen function tool call is missing </parameter>");
        };
        let value = rest[..value_end].trim_matches('\n');
        arguments.insert(
            parameter_name.to_string(),
            parse_qwen_function_parameter_output(value),
        );
        rest = &rest[value_end + "</parameter>".len()..];
    }

    Ok(ChatToolCall {
        id: None,
        name: name.to_string(),
        arguments: Value::Object(arguments),
    })
}

fn parse_qwen_function_parameter_output(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
}

fn parse_qwen_tool_arguments(value: &Value) -> Result<Value> {
    if let Some(text) = value.as_str() {
        return serde_json::from_str(text)
            .with_context(|| format!("failed to parse Qwen tool call arguments JSON: {text}"));
    }
    Ok(value.clone())
}

#[derive(Debug, Clone)]
pub(super) struct TokenSampler {
    temperature: f32,
    pub(super) top_k: usize,
    top_p: f32,
    pub(super) repeat_penalty: f32,
    state: u64,
}

impl TokenSampler {
    pub(super) fn new(temperature: f32, top_k: i32, seed: u32) -> Self {
        let deterministic = temperature <= 0.0 || top_k <= 1;
        Self {
            temperature,
            top_k: usize::try_from(top_k.max(1)).unwrap_or(1),
            top_p: if deterministic { 1.0 } else { 0.95 },
            repeat_penalty: if deterministic { 1.0 } else { 1.05 },
            state: u64::from(seed).max(1),
        }
    }

    #[cfg(test)]
    pub(super) fn sample(
        &mut self,
        logits: &[f32],
        prompt: &[u32],
        generated: &[u32],
    ) -> Result<u32> {
        if logits.is_empty() {
            bail!("cannot sample from empty logits");
        }
        let candidates = self.top_candidates(logits, prompt, generated);
        self.sample_candidates(candidates)
    }

    pub(super) fn sample_candidates(&mut self, mut candidates: Vec<(usize, f32)>) -> Result<u32> {
        if candidates.is_empty() {
            bail!("no logits candidates available");
        }
        if self.temperature <= 0.0 || candidates.len() == 1 {
            return u32::try_from(candidates[0].0).context("sampled token id does not fit u32");
        }
        let inv_temp = 1.0 / self.temperature.max(1e-6);
        let mut probabilities: Vec<f32> = candidates
            .iter()
            .map(|(_, logit)| *logit * inv_temp)
            .collect();
        softmax_in_place(&mut probabilities);
        self.apply_top_p(&mut candidates, &mut probabilities);
        let draw = self.next_f32();
        let mut cumulative = 0.0f32;
        let mut fallback = candidates[0].0;
        for ((token, _), weight) in candidates.into_iter().zip(probabilities) {
            fallback = token;
            cumulative += weight;
            if draw <= cumulative {
                return u32::try_from(token).context("sampled token id does not fit u32");
            }
        }
        u32::try_from(fallback).context("sampled token id does not fit u32")
    }

    pub(super) fn top_candidates(
        &self,
        logits: &[f32],
        prompt: &[u32],
        generated: &[u32],
    ) -> Vec<(usize, f32)> {
        let repeated = self.repeated_tokens(prompt, generated);
        let mut candidates = TopKCandidates::new(self.top_k.min(logits.len()).max(1));
        for (token, logit) in logits.iter().copied().enumerate() {
            candidates.push(token, self.process_logit(token, logit, &repeated));
        }
        candidates.into_sorted_vec()
    }

    fn apply_top_p(&self, candidates: &mut Vec<(usize, f32)>, probabilities: &mut Vec<f32>) {
        if self.top_p >= 1.0 || candidates.len() <= 1 {
            return;
        }
        let mut cumulative = 0.0f32;
        let mut keep = candidates.len();
        for (idx, probability) in probabilities.iter().enumerate() {
            cumulative += *probability;
            if cumulative >= self.top_p {
                keep = idx + 1;
                break;
            }
        }
        keep = keep.max(1);
        candidates.truncate(keep);
        probabilities.truncate(keep);
        let total = probabilities.iter().sum::<f32>();
        if total.is_finite() && total > 0.0 {
            for probability in probabilities {
                *probability /= total;
            }
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 40) as f32) / ((1u64 << 24) as f32)
    }

    pub(super) fn repeated_tokens(&self, prompt: &[u32], generated: &[u32]) -> BTreeSet<usize> {
        if self.repeat_penalty <= 1.0 {
            return BTreeSet::new();
        }
        let window = generated.len().saturating_sub(256);
        prompt
            .iter()
            .chain(generated[window..].iter())
            .map(|token| *token as usize)
            .collect()
    }

    pub(super) fn process_logit(
        &self,
        token: usize,
        logit: f32,
        repeated: &BTreeSet<usize>,
    ) -> f32 {
        process_sample_logit(token, logit, self.repeat_penalty, repeated)
    }
}

pub(super) fn process_sample_logit(
    token: usize,
    logit: f32,
    repeat_penalty: f32,
    repeated: &BTreeSet<usize>,
) -> f32 {
    let mut processed = if logit.is_finite() {
        logit
    } else {
        f32::NEG_INFINITY
    };
    if repeat_penalty > 1.0 && repeated.contains(&token) {
        if processed > 0.0 {
            processed /= repeat_penalty;
        } else {
            processed *= repeat_penalty;
        }
    }
    processed
}

pub(super) fn rerank_resident_lm_head_candidates(
    raw_candidates: &[(usize, f32)],
    top_k: usize,
    repeat_penalty: f32,
    repeated: &BTreeSet<usize>,
) -> Vec<(usize, f32)> {
    let mut candidates = TopKCandidates::new(top_k);
    for (token, value) in raw_candidates.iter().copied() {
        candidates.push(
            token,
            process_sample_logit(token, value, repeat_penalty, repeated),
        );
    }
    candidates.into_sorted_vec()
}

#[derive(Debug, Clone)]
pub(super) struct TopKCandidates {
    limit: usize,
    values: Vec<(usize, f32)>,
}

impl TopKCandidates {
    pub(super) fn new(limit: usize) -> Self {
        let limit = limit.max(1);
        Self {
            limit,
            values: Vec::with_capacity(limit),
        }
    }

    pub(super) fn push(&mut self, token: usize, score: f32) {
        let entry = (token, score);
        let insert_at = self
            .values
            .binary_search_by(|current| compare_scored_tokens(current, &entry))
            .unwrap_or_else(|idx| idx);
        if self.values.len() < self.limit {
            self.values.insert(insert_at.min(self.values.len()), entry);
        } else if insert_at < self.limit {
            self.values.insert(insert_at, entry);
            self.values.pop();
        }
    }

    pub(super) fn into_sorted_vec(self) -> Vec<(usize, f32)> {
        self.values
    }
}

pub(super) fn report_generation_progress<F>(progress: &GenerationProgress<'_>, message: F)
where
    F: FnOnce() -> String,
{
    if let Some(callback) = progress {
        (callback.borrow_mut())(message());
    }
}

#[cfg(test)]
fn test_default_tokenizer_config_json() -> &'static [u8] {
    br#"{
  "eos_token": "<|im_end|>",
  "add_bos_token": false,
  "split_special_tokens": false,
  "chat_template": "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
}"#
}

#[cfg(test)]
#[path = "text_parity_tests.rs"]
mod parity_tests;
#[cfg(test)]
pub(super) use parity_tests::{
    test_qwen3_tool_tokenizer_config_json, test_qwen3vl_tokenizer_json,
    test_qwen3vl_tool_tokenizer_config_json, test_tokenizer_config_json, test_tokenizer_json,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_generation_progress_does_not_format_message() {
        let mut formatted = false;
        report_generation_progress(&None, || {
            formatted = true;
            "unused".to_string()
        });
        assert!(!formatted);
    }

    #[test]
    fn deterministic_sampling_uses_the_highest_candidate() {
        let mut sampler = TokenSampler::new(0.0, 1, 7);
        assert_eq!(
            sampler.sample_candidates(vec![(4, 3.0), (2, 1.0)]).unwrap(),
            4
        );
    }

    #[test]
    fn qwen_tool_output_parser_preserves_text_and_typed_arguments() {
        let (text, calls) = parse_qwen_tool_call_output(
            "before <tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call> after",
        )
        .unwrap();

        assert_eq!(text, "before  after");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather");
        assert_eq!(calls[0].arguments["city"], "Paris");
    }
}
