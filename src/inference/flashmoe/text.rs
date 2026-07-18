use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;
use tokenizers::Tokenizer;

use super::deepseek::DeepSeekV4Tokenizer;
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
    tokenizer: TokenizerBackend,
    config: QwenTokenizerConfig,
    eos_tokens: BTreeSet<u32>,
    primary_eos_token: u32,
    im_start: Option<u32>,
    im_end: Option<u32>,
    vocab_size: usize,
    #[cfg(test)]
    candidate_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
enum TokenizerBackend {
    HuggingFace(Tokenizer),
    DeepSeekV4(DeepSeekV4Tokenizer),
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
    pub(super) fn from_files(
        tokenizer_path: &Path,
        config_path: &Path,
        external_chat_template_path: Option<&Path>,
    ) -> Result<Self> {
        if !config_path.is_file() {
            bail!(
                "Flash-MoE tokenizer config is required for chat generation: missing {}",
                config_path.display()
            );
        }
        let bytes = fs::read(tokenizer_path)
            .with_context(|| format!("failed to read tokenizer {}", tokenizer_path.display()))?;
        if serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|value| {
                value
                    .get("format")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .as_deref()
            == Some("PB_DEEPSEEK_V4_JOYAI_BPE_V1")
        {
            let tokenizer = DeepSeekV4Tokenizer::from_cache_bytes(&bytes)?;
            let config_value: Value =
                serde_json::from_slice(&fs::read(config_path).with_context(|| {
                    format!("failed to read tokenizer config {}", config_path.display())
                })?)
                .with_context(|| {
                    format!("failed to parse tokenizer config {}", config_path.display())
                })?;
            if config_value.get("format").and_then(Value::as_str)
                != Some("PB_DEEPSEEK_V4_JOYAI_BPE_V1")
            {
                bail!("DeepSeek V4 Flash tokenizer config does not match its native cache format");
            }
            let eos = tokenizer.eos_id();
            let vocab_size = tokenizer.vocab_size();
            #[cfg(test)]
            let candidate_ids = (0..u32::try_from(vocab_size)
                .context("DeepSeek tokenizer vocabulary exceeds u32")?)
                .collect();
            return Ok(Self {
                tokenizer: TokenizerBackend::DeepSeekV4(tokenizer),
                config: QwenTokenizerConfig::default(),
                eos_tokens: BTreeSet::from([eos]),
                primary_eos_token: eos,
                im_start: None,
                im_end: None,
                vocab_size,
                #[cfg(test)]
                candidate_ids,
            });
        }
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|err| anyhow::anyhow!("{err}"))
            .with_context(|| format!("failed to load tokenizer {}", tokenizer_path.display()))?;
        let config_bytes = fs::read(config_path).with_context(|| {
            format!("failed to read tokenizer config {}", config_path.display())
        })?;
        let external_chat_template = external_chat_template_path
            .filter(|path| path.is_file())
            .map(fs::read)
            .transpose()
            .with_context(|| {
                format!(
                    "failed to read external tokenizer chat template {}",
                    external_chat_template_path
                        .expect("present when an external template read was attempted")
                        .display()
                )
            })?;
        Self::from_json_bytes_with_tokenizer(
            &bytes,
            tokenizer,
            &config_bytes,
            external_chat_template.as_deref(),
        )
        .with_context(|| {
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
        Self::from_json_bytes_with_tokenizer(bytes, tokenizer, config_bytes, None)
    }

    #[cfg(test)]
    pub(super) fn from_json_bytes_with_config_and_chat_template(
        bytes: &[u8],
        config_bytes: &[u8],
        external_chat_template: &[u8],
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|err| anyhow::anyhow!("{err}"))
            .context("tokenizer JSON is invalid")?;
        Self::from_json_bytes_with_tokenizer(
            bytes,
            tokenizer,
            config_bytes,
            Some(external_chat_template),
        )
    }

    fn from_json_bytes_with_tokenizer(
        bytes: &[u8],
        tokenizer: Tokenizer,
        config_bytes: &[u8],
        external_chat_template: Option<&[u8]>,
    ) -> Result<Self> {
        let _ = bytes;
        let config = QwenTokenizerConfig::from_bytes(config_bytes, external_chat_template)?;
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
            tokenizer: TokenizerBackend::HuggingFace(tokenizer),
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
        if matches!(&self.tokenizer, TokenizerBackend::DeepSeekV4(_)) {
            return render_deepseek_v4_chat(
                &[ChatMessage::text(ChatRole::User, prompt)],
                &[],
                true,
                true,
            )
            .unwrap_or_else(|_| prompt.to_string());
        }
        if prompt.contains("<|im_start|>") {
            return prompt.to_string();
        }
        if let Some(template) = &self.config.chat_template {
            return render_tokenizer_chat_template(
                template,
                &[ChatMessage::text(ChatRole::User, prompt)],
                &[],
                true,
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
        self.apply_chat_template_to_messages_with_thinking(
            messages,
            tools,
            add_generation_prompt,
            true,
        )
    }

    pub(super) fn apply_chat_template_to_messages_with_thinking(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String> {
        if matches!(&self.tokenizer, TokenizerBackend::DeepSeekV4(_)) {
            return render_deepseek_v4_chat(
                messages,
                tools,
                add_generation_prompt,
                enable_thinking,
            );
        }
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
                enable_thinking,
            );
        }
        if self.im_start.is_some() && self.im_end.is_some() {
            let mut rendered = render_qwen_chatml(messages, tools, add_generation_prompt)?;
            if add_generation_prompt && !enable_thinking {
                rendered.push_str("<think>\n\n</think>\n\n");
            }
            return Ok(rendered);
        }
        bail!(
            "tokenizer_config.json and chat_template.jinja are missing a chat template, and tokenizer.json is missing Qwen chat special tokens; Flash-MoE chat generation requires one of them"
        )
    }

    pub(super) fn render_and_encode_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<(String, Vec<u32>)> {
        let prompt = self.apply_chat_template_to_messages_with_thinking(
            messages,
            tools,
            add_generation_prompt,
            enable_thinking,
        )?;
        let tokens = self.encode(&prompt)?;
        Ok((prompt, tokens))
    }

    pub(super) fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let TokenizerBackend::HuggingFace(tokenizer) = &self.tokenizer else {
            let TokenizerBackend::DeepSeekV4(tokenizer) = &self.tokenizer else {
                unreachable!()
            };
            return tokenizer.encode(text);
        };
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|err| anyhow::anyhow!("{err}"))
            .with_context(|| format!("failed to encode text with tokenizer.json"))?;
        let mut ids = encoding.get_ids().to_vec();
        if self.config.add_bos_token
            && let Some(bos) = self
                .config
                .bos_token
                .as_deref()
                .and_then(|token| tokenizer.token_to_id(token))
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
        match &self.tokenizer {
            TokenizerBackend::HuggingFace(tokenizer) => tokenizer
                .decode(&tokens, true)
                .map_err(|err| anyhow::anyhow!("{err}"))
                .context("failed to decode tokens with tokenizer.json"),
            TokenizerBackend::DeepSeekV4(tokenizer) => tokenizer.decode(&tokens),
        }
    }

    pub(super) fn is_eos(&self, token: u32) -> bool {
        self.eos_tokens.contains(&token)
    }

    pub(super) fn eos_token_id(&self) -> u32 {
        self.primary_eos_token
    }

    /// Look up a token string and return its ID, or `None` if not present.
    pub(super) fn token_id(&self, token: &str) -> Option<u32> {
        match &self.tokenizer {
            TokenizerBackend::HuggingFace(tokenizer) => tokenizer.token_to_id(token),
            TokenizerBackend::DeepSeekV4(tokenizer) => tokenizer.token_id(token),
        }
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
    fn from_bytes(bytes: &[u8], external_chat_template: Option<&[u8]>) -> Result<Self> {
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
        let chat_template = match TokenizerChatTemplate::from_tokenizer_config_value(&value)? {
            embedded @ Some(_) => embedded,
            None => external_chat_template
                .map(|template| {
                    TokenizerChatTemplate::from_external_template_bytes(template, &value)
                })
                .transpose()?,
        };
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
    enable_thinking: bool,
) -> Result<String> {
    let tools: Vec<Value> = tools.iter().map(qwen_tool_schema_value).collect();
    template.render(
        messages,
        &tools,
        ChatTemplateOptions {
            add_generation_prompt,
            enable_thinking,
            ..ChatTemplateOptions::default()
        },
    )
}

const DEEPSEEK_BOS: &str = "<｜begin▁of▁sentence｜>";
const DEEPSEEK_EOS: &str = "<｜end▁of▁sentence｜>";
const DEEPSEEK_USER: &str = "<｜User｜>";
const DEEPSEEK_ASSISTANT: &str = "<｜Assistant｜>";
const DEEPSEEK_DSML_CALLS_OPEN: &str = "<｜DSML｜tool_calls>";
const DEEPSEEK_DSML_CALLS_CLOSE: &str = "</｜DSML｜tool_calls>";
const DEEPSEEK_DSML_INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
const DEEPSEEK_DSML_PARAMETER_CLOSE: &str = "</｜DSML｜parameter>";

fn render_deepseek_v4_chat(
    messages: &[ChatMessage],
    tools: &[ChatTool],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<String> {
    let mut system = render_deepseek_tool_instructions(tools)?;
    for message in messages {
        if message.role != ChatRole::System {
            continue;
        }
        let content = render_deepseek_text_content(&message.content)?;
        if !system.is_empty() && !content.is_empty() {
            system.push_str("\n\n");
        }
        system.push_str(&content);
    }

    let tool_context = !tools.is_empty()
        || messages
            .iter()
            .any(|message| message.role == ChatRole::Tool || !message.tool_calls.is_empty());
    let last_user = messages
        .iter()
        .rposition(|message| matches!(message.role, ChatRole::User | ChatRole::Tool));
    let mut out = String::from(DEEPSEEK_BOS);
    out.push_str(&system);
    let mut pending_assistant = false;
    let mut pending_tool_result = false;
    for (index, message) in messages.iter().enumerate() {
        match message.role {
            ChatRole::System => {}
            ChatRole::User => {
                out.push_str(DEEPSEEK_USER);
                out.push_str(&render_deepseek_text_content(&message.content)?);
                pending_assistant = true;
                pending_tool_result = false;
            }
            ChatRole::Tool => {
                if !pending_tool_result {
                    out.push_str(DEEPSEEK_USER);
                }
                out.push_str("<tool_result>");
                out.push_str(&escape_deepseek_tool_result(&render_deepseek_text_content(
                    &message.content,
                )?));
                out.push_str("</tool_result>");
                pending_assistant = true;
                pending_tool_result = true;
            }
            ChatRole::Assistant => {
                if pending_assistant {
                    out.push_str(DEEPSEEK_ASSISTANT);
                    let content = render_deepseek_text_content(&message.content)?;
                    if !content.starts_with("<think>") && !content.starts_with("</think>") {
                        // Historical assistant reasoning is only retained when
                        // the caller supplies the model-native think wrapper.
                        // Otherwise this is an ordinary non-thinking answer.
                        let retain_thinking = enable_thinking
                            && tool_context
                            && last_user.is_some_and(|last| index > last);
                        out.push_str(if retain_thinking {
                            "<think>"
                        } else {
                            "</think>"
                        });
                    }
                    out.push_str(&content);
                } else {
                    out.push_str(&render_deepseek_text_content(&message.content)?);
                }
                render_deepseek_dsml_calls(&mut out, &message.tool_calls)?;
                out.push_str(DEEPSEEK_EOS);
                pending_assistant = false;
                pending_tool_result = false;
            }
        }
    }
    if add_generation_prompt && (pending_assistant || messages.is_empty()) {
        out.push_str(DEEPSEEK_ASSISTANT);
        out.push_str(if enable_thinking {
            "<think>"
        } else {
            "</think>"
        });
    }
    Ok(out)
}

fn render_deepseek_text_content(content: &ChatMessageContent) -> Result<String> {
    match content {
        ChatMessageContent::Text(text) => Ok(text.clone()),
        ChatMessageContent::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatContentPart::Text { text } => out.push_str(text),
                    ChatContentPart::Image { .. } => {
                        bail!("DeepSeek V4 Flash is a text-only execution graph")
                    }
                }
            }
            Ok(out)
        }
    }
}

fn render_deepseek_tool_instructions(tools: &[ChatTool]) -> Result<String> {
    if tools.is_empty() {
        return Ok(String::new());
    }
    let mut out = String::from(
        "## Tools\n\nYou have access to a set of tools to help answer the user question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as raw text and set `string=\"true\"`. Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n",
    );
    for tool in tools {
        let mut schema = serde_json::Map::new();
        schema.insert("name".to_string(), Value::String(tool.name.clone()));
        if let Some(description) = &tool.description {
            schema.insert(
                "description".to_string(),
                Value::String(description.clone()),
            );
        }
        schema.insert("parameters".to_string(), tool.input_schema.clone());
        out.push_str(&serde_json::to_string(&Value::Object(schema))?);
        out.push('\n');
    }
    out.push_str(
        "\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. Use the exact parameter names from the schemas.",
    );
    Ok(out)
}

fn escape_deepseek_tool_result(content: &str) -> String {
    content.replace("</tool_result>", "&lt;/tool_result>")
}

fn render_deepseek_dsml_calls(out: &mut String, calls: &[ChatToolCall]) -> Result<()> {
    if calls.is_empty() {
        return Ok(());
    }
    out.push_str("\n\n");
    out.push_str(DEEPSEEK_DSML_CALLS_OPEN);
    out.push('\n');
    for call in calls {
        out.push_str("<｜DSML｜invoke name=\"");
        out.push_str(&escape_dsml_attribute(&call.name));
        out.push_str("\">\n");
        let arguments = match &call.arguments {
            Value::Object(arguments) => arguments.clone(),
            Value::String(arguments) => serde_json::from_str::<Value>(arguments)
                .ok()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_else(|| {
                    serde_json::Map::from_iter([(
                        "arguments".to_string(),
                        Value::String(arguments.clone()),
                    )])
                }),
            other => serde_json::Map::from_iter([("arguments".to_string(), other.clone())]),
        };
        for (name, value) in arguments {
            out.push_str("<｜DSML｜parameter name=\"");
            out.push_str(&escape_dsml_attribute(&name));
            match value {
                Value::String(value) => {
                    out.push_str("\" string=\"true\">");
                    out.push_str(
                        &value.replace(DEEPSEEK_DSML_PARAMETER_CLOSE, "&lt;/｜DSML｜parameter>"),
                    );
                }
                value => {
                    out.push_str("\" string=\"false\">");
                    out.push_str(&serde_json::to_string(&value)?);
                }
            }
            out.push_str(DEEPSEEK_DSML_PARAMETER_CLOSE);
            out.push('\n');
        }
        out.push_str(DEEPSEEK_DSML_INVOKE_CLOSE);
        out.push('\n');
    }
    out.push_str(DEEPSEEK_DSML_CALLS_CLOSE);
    Ok(())
}

fn escape_dsml_attribute(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
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

#[cfg(test)]
pub(super) fn parse_qwen_tool_call_output(content: &str) -> Result<(String, Vec<ChatToolCall>)> {
    parse_qwen_tool_call_output_with_incomplete(content, false)
}

pub(super) fn parse_qwen_tool_call_output_with_incomplete(
    content: &str,
    allow_incomplete: bool,
) -> Result<(String, Vec<ChatToolCall>)> {
    let mut remaining = content;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    while let Some(start) = remaining.find("<tool_call>") {
        text.push_str(&remaining[..start]);
        let block_start = start + "<tool_call>".len();
        let Some(relative_end) = remaining[block_start..].find("</tool_call>") else {
            if allow_incomplete {
                text.push_str(&remaining[start..]);
                return Ok((text.trim().to_string(), tool_calls));
            }
            bail!("Qwen tool call is missing </tool_call>");
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

pub(super) fn parse_deepseek_tool_call_output_with_incomplete(
    content: &str,
    allow_incomplete: bool,
) -> Result<(String, Vec<ChatToolCall>)> {
    let mut remaining = content;
    let mut text = String::new();
    let mut calls = Vec::new();
    while let Some(start) = remaining.find(DEEPSEEK_DSML_CALLS_OPEN) {
        text.push_str(&remaining[..start]);
        let block_start = start + DEEPSEEK_DSML_CALLS_OPEN.len();
        let Some(relative_end) = remaining[block_start..].find(DEEPSEEK_DSML_CALLS_CLOSE) else {
            if allow_incomplete {
                text.push_str(&remaining[start..]);
                return Ok((text.trim().to_string(), calls));
            }
            bail!("DeepSeek DSML tool call is missing {DEEPSEEK_DSML_CALLS_CLOSE}");
        };
        let block_end = block_start + relative_end;
        calls.extend(parse_deepseek_dsml_block(
            &remaining[block_start..block_end],
        )?);
        remaining = &remaining[block_end + DEEPSEEK_DSML_CALLS_CLOSE.len()..];
    }
    text.push_str(remaining);
    Ok((text.trim().to_string(), calls))
}

fn parse_deepseek_dsml_block(mut block: &str) -> Result<Vec<ChatToolCall>> {
    const INVOKE_OPEN: &str = "<｜DSML｜invoke";
    const PARAMETER_OPEN: &str = "<｜DSML｜parameter";
    let mut calls = Vec::new();
    loop {
        let Some(start) = block.find(INVOKE_OPEN) else {
            if !block.trim().is_empty() {
                bail!("DeepSeek DSML tool_calls contains text outside an invoke block");
            }
            break;
        };
        if !block[..start].trim().is_empty() {
            bail!("DeepSeek DSML tool_calls contains text before an invoke block");
        }
        let header_start = start + INVOKE_OPEN.len();
        let header_end = block[header_start..]
            .find('>')
            .map(|offset| header_start + offset)
            .context("DeepSeek DSML invoke opening tag is incomplete")?;
        let name = parse_dsml_attribute(&block[header_start..header_end], "name")
            .context("DeepSeek DSML invoke is missing name")?;
        let body_start = header_end + 1;
        let body_end = block[body_start..]
            .find(DEEPSEEK_DSML_INVOKE_CLOSE)
            .map(|offset| body_start + offset)
            .context("DeepSeek DSML invoke is missing its closing tag")?;
        let mut body = &block[body_start..body_end];
        let mut arguments = serde_json::Map::new();
        loop {
            let Some(parameter_start) = body.find(PARAMETER_OPEN) else {
                if !body.trim().is_empty() {
                    bail!("DeepSeek DSML invoke contains text outside a parameter block");
                }
                break;
            };
            if !body[..parameter_start].trim().is_empty() {
                bail!("DeepSeek DSML invoke contains text before a parameter block");
            }
            let parameter_header_start = parameter_start + PARAMETER_OPEN.len();
            let parameter_header_end = body[parameter_header_start..]
                .find('>')
                .map(|offset| parameter_header_start + offset)
                .context("DeepSeek DSML parameter opening tag is incomplete")?;
            let header = &body[parameter_header_start..parameter_header_end];
            let parameter_name = parse_dsml_attribute(header, "name")
                .context("DeepSeek DSML parameter is missing name")?;
            let string_value = parse_dsml_attribute(header, "string")
                .context("DeepSeek DSML parameter is missing string=true|false")?;
            let value_start = parameter_header_end + 1;
            let value_end = body[value_start..]
                .find(DEEPSEEK_DSML_PARAMETER_CLOSE)
                .map(|offset| value_start + offset)
                .context("DeepSeek DSML parameter is missing its closing tag")?;
            let raw = &body[value_start..value_end];
            let value = match string_value.as_str() {
                "true" => Value::String(
                    raw.replace("&lt;/｜DSML｜parameter>", DEEPSEEK_DSML_PARAMETER_CLOSE),
                ),
                "false" => serde_json::from_str(raw).with_context(|| {
                    format!(
                        "DeepSeek DSML non-string parameter {parameter_name:?} is not valid JSON"
                    )
                })?,
                other => bail!(
                    "DeepSeek DSML parameter {parameter_name:?} has invalid string attribute {other:?}"
                ),
            };
            if arguments.insert(parameter_name.clone(), value).is_some() {
                bail!("DeepSeek DSML invoke repeats parameter {parameter_name:?}");
            }
            body = &body[value_end + DEEPSEEK_DSML_PARAMETER_CLOSE.len()..];
        }
        calls.push(ChatToolCall {
            id: None,
            name: unescape_dsml_attribute(&name),
            arguments: Value::Object(arguments),
        });
        block = &block[body_end + DEEPSEEK_DSML_INVOKE_CLOSE.len()..];
    }
    Ok(calls)
}

fn parse_dsml_attribute(header: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = header.find(&needle)? + needle.len();
    let end = header[start..].find('"')? + start;
    Some(header[start..end].to_string())
}

fn unescape_dsml_attribute(value: &str) -> String {
    value
        .replace("&quot;", "\"")
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
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

    #[test]
    fn deepseek_renderer_binds_native_markers_and_thinking_policy() {
        let messages = [
            ChatMessage::text(ChatRole::System, "be exact"),
            ChatMessage::text(ChatRole::User, "2+2?"),
        ];
        assert_eq!(
            render_deepseek_v4_chat(&messages, &[], true, true).unwrap(),
            "<｜begin▁of▁sentence｜>be exact<｜User｜>2+2?<｜Assistant｜><think>"
        );
        assert_eq!(
            render_deepseek_v4_chat(&messages, &[], true, false).unwrap(),
            "<｜begin▁of▁sentence｜>be exact<｜User｜>2+2?<｜Assistant｜></think>"
        );
    }

    #[test]
    fn deepseek_dsml_history_and_parser_preserve_typed_parameters() {
        let mut assistant = ChatMessage::text(ChatRole::Assistant, "checking");
        assistant.tool_calls.push(ChatToolCall {
            id: Some("call_1".to_string()),
            name: "weather".to_string(),
            arguments: serde_json::json!({"city": "London", "days": 2}),
        });
        let rendered = render_deepseek_v4_chat(
            &[
                ChatMessage::text(ChatRole::User, "weather?"),
                assistant,
                ChatMessage::text(ChatRole::Tool, "sunny </tool_result> still data"),
            ],
            &[],
            true,
            false,
        )
        .unwrap();
        assert!(rendered.contains("<｜Assistant｜></think>checking\n\n<｜DSML｜tool_calls>"));
        assert!(rendered.contains(
            "<｜DSML｜parameter name=\"city\" string=\"true\">London</｜DSML｜parameter>"
        ));
        assert!(
            rendered.contains(
                "<｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter>"
            )
        );
        assert!(rendered.contains("<tool_result>sunny &lt;/tool_result> still data</tool_result>"));
        assert!(rendered.ends_with("<｜Assistant｜></think>"));

        let (text, calls) = parse_deepseek_tool_call_output_with_incomplete(
            "before <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">London</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> after",
            false,
        )
        .unwrap();
        assert_eq!(text, "before  after");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather");
        assert_eq!(calls[0].arguments["city"], "London");
        assert_eq!(calls[0].arguments["days"], 2);
    }
}
