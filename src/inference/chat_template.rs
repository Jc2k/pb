use anyhow::{Context, Result, bail};
use minijinja::{Environment, Error as MiniJinjaError, ErrorKind as MiniJinjaErrorKind};
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Default)]
pub struct TokenizerChatTemplate {
    default_template: Option<String>,
    tool_template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub pad_token: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ChatTemplateOptions {
    pub add_generation_prompt: bool,
    pub add_vision_id: bool,
    pub enable_thinking: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            add_vision_id: false,
            enable_thinking: true,
        }
    }
}

impl TokenizerChatTemplate {
    pub fn from_tokenizer_config_bytes(config_bytes: Option<&[u8]>) -> Result<Option<Self>> {
        let Some(bytes) = config_bytes else {
            return Ok(None);
        };
        let value: Value =
            serde_json::from_slice(bytes).context("tokenizer_config.json is invalid")?;
        Self::from_tokenizer_config_value(&value)
    }

    pub fn from_tokenizer_config_value(value: &Value) -> Result<Option<Self>> {
        let Some(chat_template) = value.get("chat_template") else {
            return Ok(None);
        };
        let (default_template, tool_template) = parse_chat_template_value(chat_template)?;
        if default_template.is_none() && tool_template.is_none() {
            return Ok(None);
        }
        Ok(Some(Self {
            default_template,
            tool_template,
            bos_token: string_or_null(value.get("bos_token"))?,
            eos_token: string_or_null(value.get("eos_token"))?,
            pad_token: string_or_null(value.get("pad_token"))?,
        }))
    }

    pub fn render<M, T>(
        &self,
        messages: M,
        tools: T,
        options: ChatTemplateOptions,
    ) -> Result<String>
    where
        M: Serialize,
        T: Serialize,
    {
        let messages = serde_json::to_value(messages).context("failed to serialize messages")?;
        let tools = serde_json::to_value(tools).context("failed to serialize tools")?;
        let template = self.template_for_tools(&tools)?;
        let mut env = Environment::new();
        env.add_function("raise_exception", raise_exception);
        env.render_str(
            template,
            json!({
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": options.add_generation_prompt,
                "add_vision_id": options.add_vision_id,
                "enable_thinking": options.enable_thinking,
                "bos_token": self.bos_token.as_deref(),
                "eos_token": self.eos_token.as_deref(),
                "pad_token": self.pad_token.as_deref(),
            }),
        )
        .context("failed to render tokenizer chat_template")
    }

    fn template_for_tools(&self, tools: &Value) -> Result<&str> {
        if value_is_non_empty_array(tools) {
            if let Some(template) = &self.tool_template {
                return Ok(template);
            }
        }
        if let Some(template) = &self.default_template {
            return Ok(template);
        }
        if let Some(template) = &self.tool_template {
            return Ok(template);
        }
        bail!("tokenizer_config.json chat_template did not contain a usable template")
    }
}

fn parse_chat_template_value(value: &Value) -> Result<(Option<String>, Option<String>)> {
    if let Some(template) = value.as_str() {
        return Ok((Some(template.to_string()), None));
    }
    let Some(templates) = value.as_object() else {
        bail!("tokenizer_config.json chat_template must be a string or object");
    };
    let default_template = templates
        .get("default")
        .and_then(Value::as_str)
        .map(str::to_string);
    let tool_template = templates
        .get("tool_use")
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok((default_template, tool_template))
}

fn string_or_null(value: Option<&Value>) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => bail!("tokenizer_config.json special token value must be a string or null"),
    }
}

fn value_is_non_empty_array(value: &Value) -> bool {
    value.as_array().is_some_and(|items| !items.is_empty())
}

fn raise_exception(message: String) -> std::result::Result<String, MiniJinjaError> {
    Err(MiniJinjaError::new(
        MiniJinjaErrorKind::InvalidOperation,
        message,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_simple_message_loop() {
        let template = TokenizerChatTemplate {
            default_template: Some(
                "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}".to_string(),
            ),
            eos_token: Some("<|im_end|>".to_string()),
            ..Default::default()
        };
        let rendered = template
            .render(
                json!([{"role":"user","content":"hi"}]),
                json!([]),
                ChatTemplateOptions::default(),
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn selects_tool_use_template_when_tools_are_present() {
        let config = br#"{
  "chat_template": {
    "default": "plain:{{ messages[0].content }}",
    "tool_use": "tools:{{ tools[0].function.name }}:{{ messages[0].content }}"
  }
}"#;
        let template = TokenizerChatTemplate::from_tokenizer_config_bytes(Some(config))
            .unwrap()
            .unwrap();
        let rendered_without_tools = template
            .render(
                json!([{"role":"user","content":"hi"}]),
                json!([]),
                ChatTemplateOptions::default(),
            )
            .unwrap();
        assert_eq!(rendered_without_tools, "plain:hi");

        let rendered_with_tools = template
            .render(
                json!([{"role":"user","content":"hi"}]),
                json!([{"type":"function","function":{"name":"lookup"}}]),
                ChatTemplateOptions::default(),
            )
            .unwrap();
        assert_eq!(rendered_with_tools, "tools:lookup:hi");
    }

    #[test]
    fn renders_qwen3_vl_macro_template_shape() {
        let config = br##"{
  "chat_template": "{%- set image_count = namespace(value=0) %}\n{%- macro render_content(content, do_vision_count, is_system_content=false) %}\n    {%- if content is string %}\n        {{- content }}\n    {%- elif content is iterable and content is not mapping %}\n        {%- for item in content %}\n            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}\n                {%- if do_vision_count %}{%- set image_count.value = image_count.value + 1 %}{%- endif %}\n                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}\n            {%- elif 'text' in item %}\n                {{- item.text }}\n            {%- endif %}\n        {%- endfor %}\n    {%- endif %}\n{%- endmacro %}\n{%- for message in messages %}\n    {{- '<|im_start|>' + message.role + '\\n' }}\n    {{- render_content(message.content, true) }}\n    {{- '<|im_end|>\\n' }}\n{%- endfor %}\n{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
}"##;
        let template = TokenizerChatTemplate::from_tokenizer_config_bytes(Some(config))
            .unwrap()
            .unwrap();
        let rendered = template
            .render(
                json!([{"role":"user","content":[{"type":"text","text":"describe "},{"type":"image","image":"first.png"}]}]),
                json!([]),
                ChatTemplateOptions::default(),
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\ndescribe <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
