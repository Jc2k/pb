use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{CollarError, CollarResult, mutation::LogicalPath};

use super::{AnalysisBoundary, SyntaxProfile, Viability};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixRule {
    InvalidUtf8,
    UnmatchedClosingDelimiter,
    InvalidPythonDedent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixReport {
    pub profile: Option<SyntaxProfile>,
    pub viability: Viability,
    pub boundary: Option<AnalysisBoundary>,
    pub rule: Option<PrefixRule>,
}

impl PrefixReport {
    fn unsupported() -> Self {
        Self {
            profile: None,
            viability: Viability::Unknown,
            boundary: None,
            rule: None,
        }
    }

    fn repairable(profile: SyntaxProfile, boundary: Option<AnalysisBoundary>) -> Self {
        Self {
            profile: Some(profile),
            viability: Viability::Repairable,
            boundary,
            rule: None,
        }
    }

    fn impossible(
        profile: SyntaxProfile,
        boundary: Option<AnalysisBoundary>,
        rule: PrefixRule,
    ) -> Self {
        Self {
            profile: Some(profile),
            viability: Viability::Impossible,
            boundary,
            rule: Some(rule),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrefixCheckpoint {
    stream_id: u64,
    source_len: usize,
    state: OracleState,
}

#[derive(Clone, Debug)]
pub struct SourcePrefixOracle {
    profile: Option<SyntaxProfile>,
    source_len: usize,
    max_bytes: usize,
    stream_id: u64,
    state: OracleState,
}

impl SourcePrefixOracle {
    pub fn new(path: LogicalPath, max_bytes: usize) -> CollarResult<Self> {
        if max_bytes == 0 {
            return Err(CollarError::Analysis(
                "source prefix limit must be non-zero".to_string(),
            ));
        }
        let profile = SyntaxProfile::for_path(&path);
        Ok(Self {
            profile,
            source_len: 0,
            max_bytes,
            stream_id: next_stream_id(),
            state: OracleState::new(profile),
        })
    }

    pub fn checkpoint(&self) -> PrefixCheckpoint {
        PrefixCheckpoint {
            stream_id: self.stream_id,
            source_len: self.source_len,
            state: self.state.clone(),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> CollarResult<PrefixReport> {
        let next_len = self
            .source_len
            .checked_add(bytes.len())
            .ok_or_else(|| CollarError::Analysis("source prefix length overflow".to_string()))?;
        if next_len > self.max_bytes {
            return Err(CollarError::Analysis(format!(
                "source prefix exceeds the {}-byte limit",
                self.max_bytes
            )));
        }
        self.source_len = next_len;
        self.state.push(self.profile, bytes);
        Ok(self.state.report(self.profile))
    }

    pub fn rollback(&mut self, checkpoint: PrefixCheckpoint) -> CollarResult<()> {
        if checkpoint.stream_id != self.stream_id {
            return Err(CollarError::Analysis(
                "source prefix checkpoint belongs to another stream".to_string(),
            ));
        }
        if checkpoint.source_len > self.source_len {
            return Err(CollarError::Analysis(
                "source prefix checkpoint is ahead of the current stream".to_string(),
            ));
        }
        self.source_len = checkpoint.source_len;
        self.state = checkpoint.state;
        Ok(())
    }

    pub fn report(&self) -> CollarResult<PrefixReport> {
        Ok(self.state.report(self.profile))
    }

    pub fn len(&self) -> usize {
        self.source_len
    }

    pub fn is_empty(&self) -> bool {
        self.source_len == 0
    }
}

pub fn validate_supported_prefix(path: &LogicalPath, source: &[u8]) -> CollarResult<PrefixReport> {
    let mut oracle = SourcePrefixOracle::new(path.clone(), source.len().max(1))?;
    oracle.push(source)
}

fn next_stream_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Debug)]
struct OracleState {
    utf8_tail: Vec<u8>,
    failure: Option<PrefixRule>,
    boundary: BoundaryState,
    scanner: ScannerState,
}

impl OracleState {
    fn new(profile: Option<SyntaxProfile>) -> Self {
        Self {
            utf8_tail: Vec::with_capacity(4),
            failure: None,
            boundary: BoundaryState::default(),
            scanner: ScannerState::new(profile),
        }
    }

    fn push(&mut self, profile: Option<SyntaxProfile>, bytes: &[u8]) {
        if profile.is_none() {
            return;
        }
        self.boundary.push(bytes);
        if self.failure.is_none() {
            self.validate_utf8(bytes);
        }
        if self.failure.is_none() {
            for byte in bytes {
                if let Some(rule) = self.scanner.push(*byte) {
                    self.failure = Some(rule);
                    break;
                }
            }
        }
    }

    fn validate_utf8(&mut self, bytes: &[u8]) {
        self.utf8_tail.extend_from_slice(bytes);
        match std::str::from_utf8(&self.utf8_tail) {
            Ok(_) => self.utf8_tail.clear(),
            Err(error) if error.error_len().is_none() => {
                let pending = self.utf8_tail.split_off(error.valid_up_to());
                self.utf8_tail = pending;
            }
            Err(_) => self.failure = Some(PrefixRule::InvalidUtf8),
        }
    }

    fn report(&self, profile: Option<SyntaxProfile>) -> PrefixReport {
        let Some(profile) = profile else {
            return PrefixReport::unsupported();
        };
        let boundary = self.boundary.boundary(profile);
        match self.failure {
            Some(rule) => PrefixReport::impossible(profile, boundary, rule),
            None => PrefixReport::repairable(profile, boundary),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BoundaryState {
    last_non_whitespace: Option<u8>,
    ends_with_newline: bool,
}

impl BoundaryState {
    fn push(&mut self, bytes: &[u8]) {
        for byte in bytes {
            if !byte.is_ascii_whitespace() {
                self.last_non_whitespace = Some(*byte);
            }
            self.ends_with_newline = *byte == b'\n';
        }
    }

    fn boundary(self, profile: SyntaxProfile) -> Option<AnalysisBoundary> {
        match self.last_non_whitespace {
            Some(b';') => Some(AnalysisBoundary::Statement),
            Some(b'}')
                if matches!(
                    profile,
                    SyntaxProfile::Rust
                        | SyntaxProfile::TypeScript
                        | SyntaxProfile::Tsx
                        | SyntaxProfile::JavaScript
                        | SyntaxProfile::Css
                ) =>
            {
                Some(AnalysisBoundary::Item)
            }
            Some(_) if self.ends_with_newline => Some(AnalysisBoundary::Statement),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
enum ScannerState {
    Delimiters(DelimiterScanner),
    Python(PythonScanner),
    Deferred,
}

impl ScannerState {
    fn new(profile: Option<SyntaxProfile>) -> Self {
        match profile {
            Some(SyntaxProfile::Rust) => Self::Delimiters(DelimiterScanner::new(ScanFlavor::Rust)),
            Some(SyntaxProfile::Python) => Self::Python(PythonScanner::default()),
            Some(SyntaxProfile::TypeScript | SyntaxProfile::Tsx | SyntaxProfile::JavaScript) => {
                Self::Delimiters(DelimiterScanner::new(ScanFlavor::JavaScript))
            }
            Some(SyntaxProfile::Css) => Self::Delimiters(DelimiterScanner::new(ScanFlavor::Css)),
            Some(SyntaxProfile::Html) | None => Self::Deferred,
        }
    }

    fn push(&mut self, byte: u8) -> Option<PrefixRule> {
        match self {
            Self::Delimiters(scanner) => scanner.push(byte),
            Self::Python(scanner) => scanner.push(byte),
            Self::Deferred => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScanFlavor {
    Rust,
    JavaScript,
    Css,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Quote {
    Single,
    Double,
    Template,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RawCandidate {
    B,
    R { hashes: usize },
    Br { hashes: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockPending {
    None,
    Slash,
    Star,
}

#[derive(Clone, Debug)]
struct PersistentStack<T> {
    head: Option<Arc<StackNode<T>>>,
    len: usize,
}

#[derive(Debug)]
struct StackNode<T> {
    value: T,
    next: Option<Arc<StackNode<T>>>,
}

impl<T> Default for PersistentStack<T> {
    fn default() -> Self {
        Self { head: None, len: 0 }
    }
}

impl<T> Drop for PersistentStack<T> {
    fn drop(&mut self) {
        // `Arc`'s ordinary destruction would recurse through a uniquely owned linked list. Prefix
        // depth is attacker-controlled up to the request byte bound, so unwrap unique nodes in a
        // loop and stop as soon as another checkpoint owns the remaining suffix.
        while let Some(head) = self.head.take() {
            let Ok(mut node) = Arc::try_unwrap(head) else {
                break;
            };
            self.head = node.next.take();
        }
    }
}

impl<T: Copy> PersistentStack<T> {
    fn push(&mut self, value: T) {
        self.head = Some(Arc::new(StackNode {
            value,
            next: self.head.clone(),
        }));
        self.len = self.len.saturating_add(1);
    }

    fn pop(&mut self) -> Option<T> {
        let head = self.head.take()?;
        self.head = head.next.clone();
        self.len = self.len.saturating_sub(1);
        Some(head.value)
    }

    fn last(&self) -> Option<T> {
        self.head.as_ref().map(|node| node.value)
    }

    fn is_empty(&self) -> bool {
        self.head.is_none()
    }
}

#[derive(Clone, Debug)]
enum DelimiterMode {
    Code,
    Quote {
        quote: Quote,
        escaped: bool,
    },
    LineComment,
    BlockComment {
        depth: usize,
        pending: BlockPending,
    },
    RustRaw {
        hashes: usize,
        closing_hashes: Option<usize>,
    },
}

#[derive(Clone, Debug)]
struct DelimiterScanner {
    flavor: ScanFlavor,
    delimiters: PersistentStack<u8>,
    mode: DelimiterMode,
    pending_slash: bool,
    raw_candidate: Option<RawCandidate>,
    ambiguous_javascript_regex: bool,
}

impl DelimiterScanner {
    fn new(flavor: ScanFlavor) -> Self {
        Self {
            flavor,
            delimiters: PersistentStack::default(),
            mode: DelimiterMode::Code,
            pending_slash: false,
            raw_candidate: None,
            ambiguous_javascript_regex: false,
        }
    }

    fn push(&mut self, byte: u8) -> Option<PrefixRule> {
        match &mut self.mode {
            DelimiterMode::Quote { quote, escaped } => {
                if *escaped {
                    *escaped = false;
                } else if byte == b'\\' {
                    *escaped = true;
                } else if matches!(
                    (*quote, byte),
                    (Quote::Single, b'\'') | (Quote::Double, b'"') | (Quote::Template, b'`')
                ) {
                    self.mode = DelimiterMode::Code;
                }
                return None;
            }
            DelimiterMode::LineComment => {
                if byte == b'\n' {
                    self.mode = DelimiterMode::Code;
                }
                return None;
            }
            DelimiterMode::BlockComment { depth, pending } => {
                if *pending == BlockPending::Slash
                    && byte == b'*'
                    && self.flavor == ScanFlavor::Rust
                {
                    *depth = depth.saturating_add(1);
                    *pending = BlockPending::None;
                } else if *pending == BlockPending::Star && byte == b'/' {
                    *depth = depth.saturating_sub(1);
                    if *depth == 0 {
                        self.mode = DelimiterMode::Code;
                    } else {
                        *pending = BlockPending::None;
                    }
                } else {
                    *pending = match byte {
                        b'/' => BlockPending::Slash,
                        b'*' => BlockPending::Star,
                        _ => BlockPending::None,
                    };
                }
                return None;
            }
            DelimiterMode::RustRaw {
                hashes,
                closing_hashes,
            } => {
                if *hashes == 0 && byte == b'"' {
                    self.mode = DelimiterMode::Code;
                    return None;
                }
                match (*closing_hashes, byte) {
                    (None, b'"') => *closing_hashes = Some(0),
                    (Some(count), b'#') if count.saturating_add(1) == *hashes => {
                        self.mode = DelimiterMode::Code;
                    }
                    (Some(count), b'#') => *closing_hashes = Some(count.saturating_add(1)),
                    (Some(_), b'"') => *closing_hashes = Some(0),
                    (Some(_), _) => *closing_hashes = None,
                    (None, _) => {}
                }
                return None;
            }
            DelimiterMode::Code => {}
        }

        if self.flavor == ScanFlavor::Rust
            && let Some(candidate) = self.raw_candidate.take()
        {
            match (candidate, byte) {
                (RawCandidate::B, b'r') => {
                    self.raw_candidate = Some(RawCandidate::Br { hashes: 0 });
                    return None;
                }
                (RawCandidate::R { hashes } | RawCandidate::Br { hashes }, b'#') => {
                    self.raw_candidate = Some(match candidate {
                        RawCandidate::R { .. } => RawCandidate::R {
                            hashes: hashes.saturating_add(1),
                        },
                        RawCandidate::Br { .. } => RawCandidate::Br {
                            hashes: hashes.saturating_add(1),
                        },
                        RawCandidate::B => unreachable!(),
                    });
                    return None;
                }
                (RawCandidate::R { hashes } | RawCandidate::Br { hashes }, b'"') => {
                    self.mode = DelimiterMode::RustRaw {
                        hashes,
                        closing_hashes: None,
                    };
                    return None;
                }
                _ => {}
            }
        }

        if self.pending_slash {
            self.pending_slash = false;
            match byte {
                b'/' if self.flavor != ScanFlavor::Css => {
                    self.mode = DelimiterMode::LineComment;
                    return None;
                }
                b'*' => {
                    self.mode = DelimiterMode::BlockComment {
                        depth: 1,
                        pending: BlockPending::None,
                    };
                    return None;
                }
                _ if self.flavor == ScanFlavor::JavaScript => {
                    // Distinguishing division from a regular-expression literal requires parser
                    // context. Once the slash is known not to start a comment, the conservative
                    // oracle stops making hard delimiter claims for this suffix.
                    self.ambiguous_javascript_regex = true;
                }
                _ => {}
            }
        }

        match byte {
            b'b' if self.flavor == ScanFlavor::Rust => self.raw_candidate = Some(RawCandidate::B),
            b'r' if self.flavor == ScanFlavor::Rust => {
                self.raw_candidate = Some(RawCandidate::R { hashes: 0 })
            }
            b'/' => self.pending_slash = true,
            b'\'' => {
                self.mode = DelimiterMode::Quote {
                    quote: Quote::Single,
                    escaped: false,
                }
            }
            b'"' => {
                self.mode = DelimiterMode::Quote {
                    quote: Quote::Double,
                    escaped: false,
                }
            }
            b'`' if self.flavor == ScanFlavor::JavaScript => {
                self.mode = DelimiterMode::Quote {
                    quote: Quote::Template,
                    escaped: false,
                }
            }
            b'(' | b'[' | b'{' if !self.ambiguous_javascript_regex => self.delimiters.push(byte),
            b')' | b']' | b'}' if !self.ambiguous_javascript_regex => {
                let expected = match byte {
                    b')' => b'(',
                    b']' => b'[',
                    b'}' => b'{',
                    _ => unreachable!(),
                };
                if self.delimiters.pop() != Some(expected) {
                    return Some(PrefixRule::UnmatchedClosingDelimiter);
                }
            }
            _ => {}
        }
        None
    }
}

#[derive(Clone, Debug)]
enum PythonMode {
    Code,
    OpeningQuote {
        delimiter: u8,
        count: u8,
    },
    Quote {
        delimiter: u8,
        triple: bool,
        escaped: bool,
        closing_count: u8,
    },
    Comment,
}

#[derive(Clone, Debug)]
struct PythonScanner {
    delimiters: PersistentStack<u8>,
    indentation: PersistentStack<usize>,
    mode: PythonMode,
    at_line_start: bool,
    leading_spaces: usize,
    indentation_deferred: bool,
    last_line_byte: Option<u8>,
}

impl Default for PythonScanner {
    fn default() -> Self {
        Self {
            delimiters: PersistentStack::default(),
            indentation: {
                let mut levels = PersistentStack::default();
                levels.push(0);
                levels
            },
            mode: PythonMode::Code,
            at_line_start: true,
            leading_spaces: 0,
            indentation_deferred: false,
            last_line_byte: None,
        }
    }
}

impl PythonScanner {
    fn push(&mut self, byte: u8) -> Option<PrefixRule> {
        if byte == b'\t' {
            self.indentation_deferred = true;
        }
        if byte == b'\n' {
            if self.last_line_byte == Some(b'\\') {
                self.indentation_deferred = true;
            }
            self.at_line_start = true;
            self.leading_spaces = 0;
            self.last_line_byte = None;
        } else if byte != b'\r' {
            self.last_line_byte = Some(byte);
        }

        let mode = std::mem::replace(&mut self.mode, PythonMode::Code);
        match mode {
            PythonMode::Comment => {
                self.mode = if byte == b'\n' {
                    PythonMode::Code
                } else {
                    PythonMode::Comment
                };
                return None;
            }
            PythonMode::OpeningQuote { delimiter, count } => {
                if byte == delimiter && count == 1 {
                    self.mode = PythonMode::OpeningQuote {
                        delimiter,
                        count: 2,
                    };
                    return None;
                }
                if byte == delimiter && count == 2 {
                    self.mode = PythonMode::Quote {
                        delimiter,
                        triple: true,
                        escaped: false,
                        closing_count: 0,
                    };
                    return None;
                }
                if count == 1 {
                    self.mode = PythonMode::Quote {
                        delimiter,
                        triple: false,
                        escaped: false,
                        closing_count: 0,
                    };
                    return self.push_quote_byte(byte);
                }
                // Two adjacent quotes are a complete empty string. Reprocess this byte as code.
            }
            PythonMode::Quote {
                delimiter,
                triple,
                escaped,
                closing_count,
            } => {
                self.mode = PythonMode::Quote {
                    delimiter,
                    triple,
                    escaped,
                    closing_count,
                };
                return self.push_quote_byte(byte);
            }
            PythonMode::Code => {}
        }

        if self.at_line_start {
            match byte {
                b' ' => {
                    self.leading_spaces = self.leading_spaces.saturating_add(1);
                    return None;
                }
                b'\n' | b'\r' => return None,
                b'\t' => return None,
                b'#' | b'\'' | b'"' => self.indentation_deferred = true,
                _ => {}
            }
            self.at_line_start = false;
            if !self.indentation_deferred && self.delimiters.is_empty() {
                let current = self.indentation.last().unwrap_or(0);
                if self.leading_spaces > current {
                    self.indentation.push(self.leading_spaces);
                } else if self.leading_spaces < current {
                    while self
                        .indentation
                        .last()
                        .is_some_and(|level| level > self.leading_spaces)
                    {
                        self.indentation.pop();
                    }
                    if self.indentation.last() != Some(self.leading_spaces) {
                        return Some(PrefixRule::InvalidPythonDedent);
                    }
                }
            }
        }

        match byte {
            b'#' => {
                self.indentation_deferred = true;
                self.mode = PythonMode::Comment;
            }
            b'\'' | b'"' => {
                self.indentation_deferred = true;
                self.mode = PythonMode::OpeningQuote {
                    delimiter: byte,
                    count: 1,
                };
            }
            b'(' | b'[' | b'{' => self.delimiters.push(byte),
            b')' | b']' | b'}' => {
                let expected = match byte {
                    b')' => b'(',
                    b']' => b'[',
                    b'}' => b'{',
                    _ => unreachable!(),
                };
                if self.delimiters.pop() != Some(expected) {
                    return Some(PrefixRule::UnmatchedClosingDelimiter);
                }
            }
            _ => {}
        }
        None
    }

    fn push_quote_byte(&mut self, byte: u8) -> Option<PrefixRule> {
        let PythonMode::Quote {
            delimiter,
            triple,
            mut escaped,
            mut closing_count,
        } = self.mode
        else {
            unreachable!();
        };
        if escaped {
            escaped = false;
            closing_count = 0;
        } else if byte == b'\\' {
            escaped = true;
            closing_count = 0;
        } else if !triple && byte == delimiter {
            self.mode = PythonMode::Code;
            return None;
        } else if triple && byte == delimiter {
            closing_count = closing_count.saturating_add(1);
            if closing_count == 3 {
                self.mode = PythonMode::Code;
                return None;
            }
        } else {
            closing_count = 0;
        }
        self.mode = PythonMode::Quote {
            delimiter,
            triple,
            escaped,
            closing_count,
        };
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::validate_supported_syntax;
    use std::time::{Duration, Instant};

    fn report(path: &str, source: &str) -> PrefixReport {
        validate_supported_prefix(&LogicalPath::parse(path).unwrap(), source.as_bytes()).unwrap()
    }

    #[test]
    fn rejects_only_definite_unmatched_closers() {
        assert_eq!(
            report("src/lib.rs", "fn f() { ]").viability,
            Viability::Impossible
        );
        assert_eq!(
            report("src/lib.rs", "fn f() { let value = (").viability,
            Viability::Repairable
        );
        assert_eq!(
            report("src/lib.rs", "fn f() { let value = \" ] \"").viability,
            Viability::Repairable
        );
        assert_eq!(
            report("src/lib.rs", "fn f() { let value = r###\" ] \"###;").viability,
            Viability::Repairable
        );
    }

    #[test]
    fn python_dedent_must_return_to_an_existing_level() {
        assert_eq!(
            report(
                "main.py",
                "if ready:\n    if nested:\n        run()\n  nope()\n"
            )
            .rule,
            Some(PrefixRule::InvalidPythonDedent)
        );
        assert_eq!(
            report(
                "main.py",
                "if ready:\n    if nested:\n        run()\n    again()\n"
            )
            .viability,
            Viability::Repairable
        );
        assert_eq!(
            report("main.py", "text = \"\"\"\n    first\n  second\n\"\"\"\n").viability,
            Viability::Repairable
        );
        assert_eq!(
            report("main.py", "if ready:\n    values = (\n  \"\"\n )\n").viability,
            Viability::Repairable
        );
    }

    #[test]
    fn checkpoints_restore_the_exact_prefix() {
        let mut oracle =
            SourcePrefixOracle::new(LogicalPath::parse("src/lib.rs").unwrap(), 1024).unwrap();
        oracle.push(b"fn f() { ").unwrap();
        let checkpoint = oracle.checkpoint();
        assert_eq!(oracle.push(b"]").unwrap().viability, Viability::Impossible);
        oracle.rollback(checkpoint).unwrap();
        assert_eq!(oracle.push(b"}").unwrap().viability, Viability::Repairable);
    }

    #[test]
    fn javascript_regex_ambiguity_stays_open() {
        assert_eq!(
            report("main.js", "const value = /[]/; ]").viability,
            Viability::Repairable
        );
    }

    #[test]
    fn valid_language_corpus_is_independent_of_byte_and_random_chunk_boundaries() {
        let unicode = "name = \"café ☕\"\n";
        let corpus: &[(&str, &[u8])] = &[
            (
                "src/main.rs",
                b"fn main() { let text = r##\"]\"##; println!(\"{text}\"); }\n",
            ),
            (
                "main.py",
                b"def render(value: int) -> str:\n    text = \"\"\"[ok]\"\"\"\n    return f\"{text}: {value}\"\n",
            ),
            (
                "main.ts",
                b"export const values: Array<number> = [1, 2, 3];\n",
            ),
            (
                "view.tsx",
                b"export const View = () => <main>{['ok'].map(String)}</main>;\n",
            ),
            (
                "main.js",
                b"export const matches = /[\\]})]/u.test(']');\n",
            ),
            (
                "index.html",
                b"<!doctype html><html><body><main>valid</main></body></html>\n",
            ),
            (
                "style.css",
                b":root { --accent: rgb(1 2 3); }\nmain { color: var(--accent); }\n",
            ),
            ("unicode.py", unicode.as_bytes()),
        ];

        for (path, source) in corpus {
            let path = LogicalPath::parse(*path).unwrap();
            assert!(validate_supported_syntax(&path, source).is_ok());

            let mut bytewise = SourcePrefixOracle::new(path.clone(), source.len()).unwrap();
            for end in 1..=source.len() {
                let actual = bytewise.push(&source[end - 1..end]).unwrap();
                let expected = validate_supported_prefix(&path, &source[..end]).unwrap();
                assert_eq!(actual, expected, "byte split {end} for {}", path.as_str());
                assert_ne!(actual.viability, Viability::Impossible);
            }

            for seed in 1usize..=32 {
                let mut chunked = SourcePrefixOracle::new(path.clone(), source.len()).unwrap();
                let mut cursor = 0usize;
                let mut state = seed;
                while cursor < source.len() {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let end = cursor.saturating_add(state % 17 + 1).min(source.len());
                    chunked.push(&source[cursor..end]).unwrap();
                    cursor = end;
                }
                assert_eq!(
                    chunked.report().unwrap(),
                    validate_supported_prefix(&path, source).unwrap(),
                    "random chunk seed {seed} for {}",
                    path.as_str()
                );
            }
        }
    }

    #[test]
    fn utf8_tail_and_hard_rejections_are_monotonic() {
        let path = LogicalPath::parse("main.py").unwrap();
        let mut oracle = SourcePrefixOracle::new(path, 128).unwrap();
        assert_eq!(
            oracle.push(&[0xe2]).unwrap().viability,
            Viability::Repairable
        );
        assert_eq!(
            oracle.push(&[0x98, 0x95]).unwrap().viability,
            Viability::Repairable
        );
        assert_eq!(
            oracle.push(&[0xff]).unwrap().rule,
            Some(PrefixRule::InvalidUtf8)
        );
        assert_eq!(
            oracle.push(b"\n'''future text'''\n").unwrap().rule,
            Some(PrefixRule::InvalidUtf8)
        );

        for (path, invalid, continuation, rule) in [
            (
                "src/lib.rs",
                "fn value() { ]",
                " /* later comment */ }",
                PrefixRule::UnmatchedClosingDelimiter,
            ),
            (
                "main.py",
                "if ready:\n    if nested:\n        run()\n  nope()",
                "\ntext = '''later string'''\n",
                PrefixRule::InvalidPythonDedent,
            ),
            (
                "style.css",
                "main { ]",
                " /* later comment */ }",
                PrefixRule::UnmatchedClosingDelimiter,
            ),
        ] {
            let path = LogicalPath::parse(path).unwrap();
            let mut oracle =
                SourcePrefixOracle::new(path, invalid.len().saturating_add(continuation.len()))
                    .unwrap();
            assert_eq!(oracle.push(invalid.as_bytes()).unwrap().rule, Some(rule));
            assert_eq!(
                oracle.push(continuation.as_bytes()).unwrap().rule,
                Some(rule)
            );
        }
    }

    #[test]
    fn rollback_is_exact_and_checkpoints_are_stream_scoped() {
        let path = LogicalPath::parse("src/lib.rs").unwrap();
        let mut oracle = SourcePrefixOracle::new(path.clone(), 1024).unwrap();
        oracle.push(b"fn value() { let values = [").unwrap();
        let checkpoint = oracle.checkpoint();
        assert_eq!(
            oracle.push(b"}").unwrap().rule,
            Some(PrefixRule::UnmatchedClosingDelimiter)
        );
        oracle.rollback(checkpoint.clone()).unwrap();
        oracle.push(b"1, 2]; }\n").unwrap();
        assert_eq!(
            oracle.report().unwrap(),
            validate_supported_prefix(&path, b"fn value() { let values = [1, 2]; }\n").unwrap()
        );

        let mut other = SourcePrefixOracle::new(path, 1024).unwrap();
        assert!(other.rollback(checkpoint).is_err());
    }

    #[test]
    fn deep_candidate_checkpoints_meet_the_cheap_probe_budget() {
        let path = LogicalPath::parse("src/lib.rs").unwrap();
        let depth = 100_000usize;
        let mut oracle = SourcePrefixOracle::new(path, depth.saturating_add(1)).unwrap();
        oracle.push(&vec![b'('; depth]).unwrap();

        let mut timings = Vec::with_capacity(4_000);
        for _ in 0..4_000 {
            let checkpoint = oracle.checkpoint();
            let started = Instant::now();
            assert_eq!(oracle.push(b")").unwrap().viability, Viability::Repairable);
            oracle.rollback(checkpoint).unwrap();
            timings.push(started.elapsed());
        }
        timings.sort_unstable();
        let p95 = timings[timings.len() * 95 / 100];
        assert!(
            p95 < Duration::from_millis(1),
            "deep prefix candidate p95 was {p95:?}"
        );
    }
}
