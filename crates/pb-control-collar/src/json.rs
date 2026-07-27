use std::fmt;
use std::sync::Arc;

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::{CollarError, CollarResult, TokenMask};

/// Tokenizer-scoped LLGuidance state shared by requests using one exact vocabulary.
///
/// Grammar matchers remain request-local, so failure or cancellation cannot contaminate a later
/// session.
#[derive(Clone)]
pub struct JsonConstraintFactory {
    parser_factory: Arc<ParserFactory>,
    vocab_size: usize,
    eos_tokens: Arc<[u32]>,
}

impl fmt::Debug for JsonConstraintFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JsonConstraintFactory")
            .field("vocab_size", &self.vocab_size)
            .field("eos_tokens", &self.eos_tokens)
            .finish_non_exhaustive()
    }
}

impl JsonConstraintFactory {
    pub fn from_token_bytes(
        mut token_bytes: Vec<Vec<u8>>,
        eos_tokens: &[u32],
    ) -> CollarResult<Self> {
        if token_bytes.is_empty() {
            return Err(CollarError::InvalidVocabulary(
                "LLGuidance tokenizer vocabulary cannot be empty".to_string(),
            ));
        }
        let vocab_size = token_bytes.len();
        let vocab_size_u32 = u32::try_from(vocab_size).map_err(|_| {
            CollarError::InvalidVocabulary(
                "LLGuidance tokenizer vocabulary exceeds u32".to_string(),
            )
        })?;
        let primary_eos = eos_tokens.first().copied().ok_or_else(|| {
            CollarError::InvalidVocabulary(
                "LLGuidance tokenizer requires at least one EOS token".to_string(),
            )
        })?;
        if eos_tokens.iter().any(|token| *token as usize >= vocab_size) {
            return Err(CollarError::InvalidVocabulary(format!(
                "LLGuidance tokenizer EOS token is outside vocabulary of {vocab_size} entries"
            )));
        }
        for (token, bytes) in token_bytes.iter_mut().enumerate() {
            if bytes.is_empty() {
                let mut marker = vec![TokTrie::SPECIAL_TOKEN_MARKER];
                marker.extend_from_slice(format!("[{token}]").as_bytes());
                *bytes = marker;
            }
        }

        let info = TokRxInfo::new(vocab_size_u32, primary_eos);
        let trie = TokTrie::from(&info, &token_bytes).with_eos_tokens(eos_tokens);
        let token_env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut parser_factory = ParserFactory::new_simple(&token_env).map_err(|error| {
            CollarError::Guidance(format!(
                "failed to initialize LLGuidance tokenizer parser: {error:#}"
            ))
        })?;
        parser_factory.quiet();
        Ok(Self {
            parser_factory: Arc::new(parser_factory),
            vocab_size,
            eos_tokens: Arc::from(eos_tokens),
        })
    }

    pub fn compile(&self, schema: &Value) -> CollarResult<JsonConstraintSession> {
        let schema_bytes = serde_json::to_vec(schema).map_err(|error| {
            CollarError::Guidance(format!(
                "failed to serialize LLGuidance JSON schema: {error}"
            ))
        })?;
        let schema_sha256 = format!("{:x}", Sha256::digest(&schema_bytes));
        let grammar = TopLevelGrammar::from_json_schema(schema.clone());
        let parser = self
            .parser_factory
            .create_parser(grammar)
            .map_err(|error| {
                CollarError::Guidance(format!("LLGuidance rejected the JSON schema: {error:#}"))
            })?;
        Ok(JsonConstraintSession {
            matcher: Matcher::new(Ok(parser)),
            vocab_size: self.vocab_size,
            eos_tokens: Arc::clone(&self.eos_tokens),
            schema_sha256,
            sampled_tokens: 0,
        })
    }
}

pub struct JsonConstraintSession {
    matcher: Matcher,
    vocab_size: usize,
    eos_tokens: Arc<[u32]>,
    schema_sha256: String,
    sampled_tokens: usize,
}

impl fmt::Debug for JsonConstraintSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JsonConstraintSession")
            .field("vocab_size", &self.vocab_size)
            .field("eos_tokens", &self.eos_tokens)
            .field("schema_sha256", &self.schema_sha256)
            .field("sampled_tokens", &self.sampled_tokens)
            .finish_non_exhaustive()
    }
}

impl JsonConstraintSession {
    pub fn allowed_tokens(&mut self) -> CollarResult<TokenMask> {
        let mask = self.matcher.compute_mask_or_eos().map_err(|error| {
            CollarError::Guidance(format!(
                "LLGuidance failed to compute JSON token mask after {} generated tokens: {error:#}",
                self.sampled_tokens
            ))
        })?;
        if mask.len() != self.vocab_size {
            return Err(CollarError::Guidance(format!(
                "LLGuidance produced token mask of length {}, expected {}",
                mask.len(),
                self.vocab_size
            )));
        }
        if mask.num_set() == 0 {
            return Err(CollarError::Guidance(format!(
                "LLGuidance produced an empty JSON token mask after {} generated tokens",
                self.sampled_tokens
            )));
        }
        TokenMask::from_words(mask.len(), mask.into())
    }

    pub fn commit(&mut self, token: u32) -> CollarResult<()> {
        if token as usize >= self.vocab_size {
            return Err(CollarError::InvalidVocabulary(format!(
                "sampled JSON token {token} exceeds LLGuidance vocabulary of {} entries",
                self.vocab_size
            )));
        }
        if self.matcher.is_stopped() && self.eos_tokens.contains(&token) {
            self.sampled_tokens = self.sampled_tokens.saturating_add(1);
            return Ok(());
        }
        self.matcher.consume_token(token).map_err(|error| {
            CollarError::Guidance(format!(
                "LLGuidance rejected sampled JSON token {token} at output position {}: {error:#}",
                self.sampled_tokens
            ))
        })?;
        self.sampled_tokens = self.sampled_tokens.saturating_add(1);
        Ok(())
    }

    pub fn schema_sha256(&self) -> &str {
        &self.schema_sha256
    }

    pub fn has_complete_value(&mut self) -> CollarResult<bool> {
        self.matcher.is_accepting().map_or_else(
            |error| {
                Err(CollarError::Guidance(format!(
                    "LLGuidance could not determine JSON acceptance: {error:#}"
                )))
            },
            |accepting| Ok(self.matcher.is_stopped() || accepting),
        )
    }

    pub fn terminal_state(&mut self) -> CollarResult<&'static str> {
        if self.matcher.is_stopped() {
            Ok("complete_json")
        } else if self.matcher.is_accepting().map_err(|error| {
            CollarError::Guidance(format!(
                "LLGuidance could not determine JSON terminal state: {error:#}"
            ))
        })? {
            Ok("accepting_json")
        } else {
            Ok("incomplete_json")
        }
    }
}

pub fn validate_llguidance_json_schema(schema: &Value) -> CollarResult<()> {
    let environment = ApproximateTokEnv::single_byte_env();
    let trie = environment.tok_trie();
    let token_bytes = (0..trie.vocab_size() as u32)
        .map(|token| trie.token(token).to_vec())
        .collect::<Vec<_>>();
    JsonConstraintFactory::from_token_bytes(token_bytes, trie.eos_tokens())?.compile(schema)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_factory() -> JsonConstraintFactory {
        let environment = ApproximateTokEnv::single_byte_env();
        let trie = environment.tok_trie();
        let token_bytes = (0..trie.vocab_size() as u32)
            .map(|token| trie.token(token).to_vec())
            .collect::<Vec<_>>();
        JsonConstraintFactory::from_token_bytes(token_bytes, trie.eos_tokens()).unwrap()
    }

    fn consume_text(session: &mut JsonConstraintSession, text: &str) {
        for byte in text.bytes() {
            let mask = session.allowed_tokens().unwrap();
            assert!(
                mask.is_allowed(u32::from(byte)),
                "byte {byte:?} was rejected"
            );
            session.commit(u32::from(byte)).unwrap();
        }
    }

    #[test]
    fn schema_mask_preserves_extendable_numeric_prefixes() {
        let factory = byte_factory();
        let mut session = factory
            .compile(&serde_json::json!({"type":"number","minimum":100}))
            .unwrap();

        consume_text(&mut session, "100");

        assert_eq!(session.terminal_state().unwrap(), "accepting_json");
        assert!(session.has_complete_value().unwrap());
    }

    #[test]
    fn schema_mask_handles_nested_unicode_and_escaping() {
        let factory = byte_factory();
        let eos = factory.eos_tokens[0];
        let mut session = factory
            .compile(&serde_json::json!({
                "type":"object",
                "properties": {
                    "items": {"type":"array","items":{"type":"string"}},
                    "enabled": {"type":"boolean"}
                },
                "required":["items","enabled"],
                "additionalProperties":false
            }))
            .unwrap();

        consume_text(
            &mut session,
            r#"{"items":["café","quote: \""] ,"enabled":true}"#,
        );

        assert_eq!(session.terminal_state().unwrap(), "complete_json");
        let mask = session.allowed_tokens().unwrap();
        assert_eq!(mask.allowed_count(), 1);
        assert!(mask.is_allowed(eos));
        session.commit(eos).unwrap();
    }

    #[test]
    fn invalid_schema_fails_before_generation() {
        let factory = byte_factory();
        let error = factory
            .compile(&serde_json::json!({"type":"definitely-not-json-schema"}))
            .unwrap_err();
        assert!(error.to_string().contains("rejected the JSON schema"));
    }

    #[test]
    fn mask_recovers_a_valid_token_below_the_unconstrained_top_128() {
        let factory = byte_factory();
        let mut session = factory
            .compile(&serde_json::json!({"type":"object"}))
            .unwrap();
        let mask = session.allowed_tokens().unwrap();
        let required = b'{' as usize;
        assert!(mask.is_allowed(required as u32));

        let mut logits = vec![-1_000.0; mask.len()];
        logits[required] = -100.0;
        for (token, logit) in logits.iter_mut().enumerate() {
            if !mask.is_allowed(token as u32) {
                *logit = 1_000.0 - token as f32;
            }
        }
        let mut unconstrained = (0..logits.len()).collect::<Vec<_>>();
        unconstrained.sort_unstable_by(|left, right| logits[*right].total_cmp(&logits[*left]));
        assert!(!unconstrained[..128].contains(&required));

        mask.for_each_denied(|token| {
            logits[token] = f32::NEG_INFINITY;
        });
        let selected = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(token, _)| token)
            .unwrap();
        assert_eq!(selected, required);
    }
}
