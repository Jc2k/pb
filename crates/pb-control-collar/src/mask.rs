use crate::{CollarError, CollarResult};

const WORD_BITS: usize = u32::BITS as usize;

/// Backend-neutral full-vocabulary hard mask.
///
/// Bit `n` is set exactly when token `n` is allowed. The packed words match the representation used
/// by FlashMoe's vocabulary masking kernel without exposing LLGuidance's bitset type in the collar
/// API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenMask {
    words: Vec<u32>,
    len: usize,
}

impl TokenMask {
    pub fn none(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(WORD_BITS)],
            len,
        }
    }

    pub fn all(len: usize) -> Self {
        let mut mask = Self {
            words: vec![u32::MAX; len.div_ceil(WORD_BITS)],
            len,
        };
        mask.clear_excess_bits();
        mask
    }

    pub fn from_words(len: usize, words: Vec<u32>) -> CollarResult<Self> {
        let expected = len.div_ceil(WORD_BITS);
        if words.len() != expected {
            return Err(CollarError::InvalidVocabulary(format!(
                "token mask has {} words for {len} tokens, expected {expected}",
                words.len()
            )));
        }
        let mut mask = Self { words, len };
        mask.clear_excess_bits();
        Ok(mask)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn allowed_count(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub fn num_set(&self) -> usize {
        self.allowed_count()
    }

    pub fn is_allowed(&self, token: u32) -> bool {
        let token = token as usize;
        if token >= self.len {
            return false;
        }
        self.words[token / WORD_BITS] & (1 << (token % WORD_BITS)) != 0
    }

    pub fn allow(&mut self, token: u32) -> CollarResult<()> {
        let token = self.checked_token(token)?;
        self.words[token / WORD_BITS] |= 1 << (token % WORD_BITS);
        Ok(())
    }

    pub fn deny(&mut self, token: u32) -> CollarResult<()> {
        let token = self.checked_token(token)?;
        self.words[token / WORD_BITS] &= !(1 << (token % WORD_BITS));
        Ok(())
    }

    pub fn intersect(&mut self, other: &Self) -> CollarResult<()> {
        if self.len != other.len {
            return Err(CollarError::InvalidVocabulary(format!(
                "cannot intersect token masks with lengths {} and {}",
                self.len, other.len
            )));
        }
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left &= right;
        }
        Ok(())
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }

    pub fn as_slice(&self) -> &[u32] {
        self.words()
    }

    pub fn for_each_allowed(&self, mut visit: impl FnMut(usize)) {
        self.for_each_entry(true, |_, token| visit(token));
    }

    pub fn for_each_denied(&self, mut visit: impl FnMut(usize)) {
        self.for_each_entry(false, |_, token| visit(token));
    }

    pub fn iter_unset_entries(&self, visit: impl FnMut(usize)) {
        self.for_each_denied(visit);
    }

    fn for_each_entry(&self, wanted: bool, mut visit: impl FnMut(bool, usize)) {
        for token in 0..self.len {
            let allowed = self.is_allowed(token as u32);
            if allowed == wanted {
                visit(allowed, token);
            }
        }
    }

    fn checked_token(&self, token: u32) -> CollarResult<usize> {
        let token = token as usize;
        if token >= self.len {
            return Err(CollarError::InvalidVocabulary(format!(
                "token {token} is outside a vocabulary of {} entries",
                self.len
            )));
        }
        Ok(token)
    }

    fn clear_excess_bits(&mut self) {
        let excess = self.words.len() * WORD_BITS - self.len;
        if excess == 0 {
            return;
        }
        if let Some(last) = self.words.last_mut() {
            *last &= u32::MAX >> excess;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_bound_excess_bits_and_intersect() {
        let mut left = TokenMask::all(35);
        let mut right = TokenMask::none(35);
        right.allow(2).unwrap();
        right.allow(34).unwrap();
        left.intersect(&right).unwrap();

        assert_eq!(left.allowed_count(), 2);
        assert!(left.is_allowed(2));
        assert!(left.is_allowed(34));
        assert!(!left.is_allowed(35));
        assert_eq!(left.words(), &[4, 4]);
    }

    #[test]
    fn mismatched_masks_fail_closed() {
        let mut left = TokenMask::all(8);
        let error = left.intersect(&TokenMask::all(9)).unwrap_err();
        assert!(error.to_string().contains("lengths 8 and 9"));
    }
}
