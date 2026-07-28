use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{CollarError, CollarResult};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ControlToken(pub String);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenSurface {
    Bytes(Vec<u8>),
    Control {
        identity: ControlToken,
        visible_bytes: Vec<u8>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VocabularyEntry {
    pub token_id: u32,
    pub surface: TokenSurface,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vocabulary {
    entries: Vec<VocabularyEntry>,
    eos_tokens: BTreeSet<u32>,
}

pub const MAX_SCHEMA_UTF8_PROBE_CHARS: usize = 4_096;

/// Collect non-ASCII scalar values that carry exact schema/protocol meaning during a split UTF-8
/// token probe. Dialects add their own non-ASCII structural characters through `extras`.
pub fn schema_utf8_probe_chars(schemas: &BTreeMap<String, Value>, extras: &[char]) -> Vec<char> {
    fn collect(value: &Value, chars: &mut BTreeSet<char>) {
        if chars.len() >= MAX_SCHEMA_UTF8_PROBE_CHARS {
            return;
        }
        match value {
            Value::String(value) => {
                for character in value.chars().filter(|ch| !ch.is_ascii()) {
                    if chars.len() >= MAX_SCHEMA_UTF8_PROBE_CHARS {
                        break;
                    }
                    chars.insert(character);
                }
            }
            Value::Array(values) => {
                for value in values {
                    collect(value, chars);
                    if chars.len() >= MAX_SCHEMA_UTF8_PROBE_CHARS {
                        break;
                    }
                }
            }
            Value::Object(values) => {
                for (key, value) in values {
                    for character in key.chars().filter(|ch| !ch.is_ascii()) {
                        if chars.len() >= MAX_SCHEMA_UTF8_PROBE_CHARS {
                            break;
                        }
                        chars.insert(character);
                    }
                    collect(value, chars);
                    if chars.len() >= MAX_SCHEMA_UTF8_PROBE_CHARS {
                        break;
                    }
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }

    let mut chars = extras
        .iter()
        .copied()
        .take(MAX_SCHEMA_UTF8_PROBE_CHARS)
        .collect::<BTreeSet<_>>();
    for (name, schema) in schemas {
        for character in name.chars().filter(|ch| !ch.is_ascii()) {
            if chars.len() >= MAX_SCHEMA_UTF8_PROBE_CHARS {
                break;
            }
            chars.insert(character);
        }
        collect(schema, &mut chars);
        if chars.len() >= MAX_SCHEMA_UTF8_PROBE_CHARS {
            break;
        }
    }
    chars.into_iter().collect()
}

/// Produce a bounded set of complete scalars that can extend an incomplete UTF-8 tail. Exact
/// schema scalars are retained; representative continuation bytes cover unconstrained string and
/// prose positions without enumerating an entire Unicode plane in the vocabulary hot path.
pub fn incomplete_utf8_probe_chars(tail: &[u8], schema_chars: &[char]) -> Vec<char> {
    let expected_len = match tail.first().copied() {
        Some(0xc2..=0xdf) => 2,
        Some(0xe0..=0xef) => 3,
        Some(0xf0..=0xf4) => 4,
        _ => return Vec::new(),
    };
    if tail.len() >= expected_len {
        return Vec::new();
    }

    let mut candidates = BTreeSet::new();
    candidates.extend(schema_chars.iter().copied().filter(|character| {
        let mut bytes = [0u8; 4];
        character
            .encode_utf8(&mut bytes)
            .as_bytes()
            .starts_with(tail)
    }));
    const REPRESENTATIVE_CONTINUATIONS: [u8; 6] = [0x80, 0x8f, 0x9f, 0xa0, 0xb0, 0xbf];
    fn add_representatives(
        bytes: &mut Vec<u8>,
        expected_len: usize,
        candidates: &mut BTreeSet<char>,
    ) {
        if bytes.len() == expected_len {
            if let Ok(value) = std::str::from_utf8(bytes)
                && let Some(character) = value.chars().next()
                && character.len_utf8() == value.len()
            {
                candidates.insert(character);
            }
            return;
        }
        for byte in REPRESENTATIVE_CONTINUATIONS {
            bytes.push(byte);
            add_representatives(bytes, expected_len, candidates);
            bytes.pop();
        }
    }
    let mut bytes = tail.to_vec();
    add_representatives(&mut bytes, expected_len, &mut candidates);
    candidates.into_iter().collect()
}

impl Vocabulary {
    pub fn from_llguidance_token_bytes(
        token_bytes: Vec<Vec<u8>>,
        eos_tokens: &[u32],
        special_marker: u8,
    ) -> CollarResult<Self> {
        let entries = token_bytes
            .into_iter()
            .enumerate()
            .map(|(index, bytes)| {
                let token_id = u32::try_from(index).map_err(|_| {
                    CollarError::InvalidVocabulary(
                        "tokenizer vocabulary exceeds u32 token ids".to_string(),
                    )
                })?;
                let surface = if bytes.first() == Some(&special_marker) {
                    let visible_bytes = bytes[1..].to_vec();
                    let mut identity = String::from_utf8(visible_bytes.clone()).map_err(|_| {
                        CollarError::InvalidVocabulary(format!(
                            "control token {token_id} identity is not UTF-8"
                        ))
                    })?;
                    if identity.is_empty() {
                        // Some pinned tokenizers contain reserved non-rendering special-token slots.
                        // LLGuidance exposes those as the marker alone, so retain a stable identity
                        // without pretending that the token contributes visible transcript bytes.
                        identity = format!("token:{token_id}");
                    }
                    TokenSurface::Control {
                        identity: ControlToken(identity),
                        visible_bytes,
                    }
                } else {
                    TokenSurface::Bytes(bytes)
                };
                Ok(VocabularyEntry { token_id, surface })
            })
            .collect::<CollarResult<Vec<_>>>()?;
        Self::new(entries, eos_tokens)
    }

    pub fn new(entries: Vec<VocabularyEntry>, eos_tokens: &[u32]) -> CollarResult<Self> {
        if entries.is_empty() {
            return Err(CollarError::InvalidVocabulary(
                "vocabulary cannot be empty".to_string(),
            ));
        }
        for (index, entry) in entries.iter().enumerate() {
            if entry.token_id as usize != index {
                return Err(CollarError::InvalidVocabulary(format!(
                    "vocabulary entry {index} declares token id {}",
                    entry.token_id
                )));
            }
            match &entry.surface {
                TokenSurface::Bytes(bytes) if bytes.is_empty() => {
                    return Err(CollarError::InvalidVocabulary(format!(
                        "ordinary token {index} has no bytes"
                    )));
                }
                TokenSurface::Control {
                    identity,
                    visible_bytes: _,
                } if identity.0.is_empty() => {
                    return Err(CollarError::InvalidVocabulary(format!(
                        "control token {index} requires an identity"
                    )));
                }
                _ => {}
            }
        }
        if eos_tokens.is_empty() {
            return Err(CollarError::InvalidVocabulary(
                "vocabulary requires at least one EOS token".to_string(),
            ));
        }
        if let Some(token) = eos_tokens
            .iter()
            .find(|token| **token as usize >= entries.len())
        {
            return Err(CollarError::InvalidVocabulary(format!(
                "EOS token {token} is outside a vocabulary of {} entries",
                entries.len()
            )));
        }
        Ok(Self {
            entries,
            eos_tokens: eos_tokens.iter().copied().collect(),
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[VocabularyEntry] {
        &self.entries
    }

    pub fn surface(&self, token: u32) -> Option<&TokenSurface> {
        self.entries.get(token as usize).map(|entry| &entry.surface)
    }

    pub fn eos_tokens(&self) -> &BTreeSet<u32> {
        &self.eos_tokens
    }

    pub fn llguidance_token_bytes(&self, special_marker: u8) -> Vec<Vec<u8>> {
        self.entries
            .iter()
            .map(|entry| match &entry.surface {
                TokenSurface::Bytes(bytes) => bytes.clone(),
                TokenSurface::Control { visible_bytes, .. } => {
                    let mut bytes = Vec::with_capacity(visible_bytes.len() + 1);
                    bytes.push(special_marker);
                    bytes.extend_from_slice(visible_bytes);
                    bytes
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn control_tokens_keep_identity_and_guidance_marker() {
        let vocabulary = Vocabulary::new(
            vec![
                VocabularyEntry {
                    token_id: 0,
                    surface: TokenSurface::Bytes(b"a".to_vec()),
                },
                VocabularyEntry {
                    token_id: 1,
                    surface: TokenSurface::Control {
                        identity: ControlToken("dsml".to_string()),
                        visible_bytes: "｜DSML｜".as_bytes().to_vec(),
                    },
                },
            ],
            &[1],
        )
        .unwrap();

        assert_eq!(
            vocabulary.llguidance_token_bytes(0xff),
            vec![
                b"a".to_vec(),
                [vec![0xff], "｜DSML｜".as_bytes().to_vec()].concat()
            ]
        );
    }

    #[test]
    fn marker_only_control_tokens_keep_identity_without_visible_bytes() {
        let vocabulary =
            Vocabulary::from_llguidance_token_bytes(vec![b"a".to_vec(), vec![0xff]], &[1], 0xff)
                .unwrap();

        assert_eq!(
            vocabulary.surface(1),
            Some(&TokenSurface::Control {
                identity: ControlToken("token:1".to_string()),
                visible_bytes: Vec::new(),
            })
        );
        assert_eq!(
            vocabulary.llguidance_token_bytes(0xff),
            vec![b"a".to_vec(), vec![0xff]]
        );
    }

    #[test]
    fn utf8_probe_candidates_preserve_exact_schema_scalars_with_bounded_fallbacks() {
        let schemas = BTreeMap::from([(
            "submit_💡".to_string(),
            json!({
                "type":"object",
                "properties":{"café":{"type":"string","const":"💡"}}
            }),
        )]);
        let schema_chars = schema_utf8_probe_chars(&schemas, &['｜']);
        assert!(schema_chars.contains(&'é'));
        assert!(schema_chars.contains(&'💡'));
        assert!(schema_chars.contains(&'｜'));

        let candidates = incomplete_utf8_probe_chars(&[0xf0, 0x9f], &schema_chars);
        assert!(candidates.contains(&'💡'));
        assert!(candidates.len() <= 37);
        assert!(incomplete_utf8_probe_chars(&[0xff], &schema_chars).is_empty());

        let many_scalars = (0x1000..0x1000 + MAX_SCHEMA_UTF8_PROBE_CHARS as u32 + 32)
            .filter_map(char::from_u32)
            .collect::<String>();
        let large = BTreeMap::from([(
            "tool".to_string(),
            json!({"type":"string","const":many_scalars}),
        )]);
        assert_eq!(
            schema_utf8_probe_chars(&large, &[]).len(),
            MAX_SCHEMA_UTF8_PROBE_CHARS
        );
    }
}
