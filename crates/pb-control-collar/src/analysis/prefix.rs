use serde::{Deserialize, Serialize};

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

    fn repairable(profile: SyntaxProfile, source: &str) -> Self {
        Self {
            profile: Some(profile),
            viability: Viability::Repairable,
            boundary: boundary_for(profile, source),
            rule: None,
        }
    }

    fn impossible(profile: SyntaxProfile, source: &str, rule: PrefixRule) -> Self {
        Self {
            profile: Some(profile),
            viability: Viability::Impossible,
            boundary: boundary_for(profile, source),
            rule: Some(rule),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixCheckpoint(usize);

#[derive(Clone, Debug)]
pub struct SourcePrefixOracle {
    path: LogicalPath,
    bytes: Vec<u8>,
    max_bytes: usize,
}

impl SourcePrefixOracle {
    pub fn new(path: LogicalPath, max_bytes: usize) -> CollarResult<Self> {
        if max_bytes == 0 {
            return Err(CollarError::Analysis(
                "source prefix limit must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            path,
            bytes: Vec::new(),
            max_bytes,
        })
    }

    pub fn checkpoint(&self) -> PrefixCheckpoint {
        PrefixCheckpoint(self.bytes.len())
    }

    pub fn push(&mut self, bytes: &[u8]) -> CollarResult<PrefixReport> {
        let next_len =
            self.bytes.len().checked_add(bytes.len()).ok_or_else(|| {
                CollarError::Analysis("source prefix length overflow".to_string())
            })?;
        if next_len > self.max_bytes {
            return Err(CollarError::Analysis(format!(
                "source prefix exceeds the {}-byte limit",
                self.max_bytes
            )));
        }
        self.bytes.extend_from_slice(bytes);
        validate_supported_prefix(&self.path, &self.bytes)
    }

    pub fn rollback(&mut self, checkpoint: PrefixCheckpoint) -> CollarResult<()> {
        if checkpoint.0 > self.bytes.len() {
            return Err(CollarError::Analysis(
                "source prefix checkpoint is ahead of the current stream".to_string(),
            ));
        }
        self.bytes.truncate(checkpoint.0);
        Ok(())
    }

    pub fn report(&self) -> CollarResult<PrefixReport> {
        validate_supported_prefix(&self.path, &self.bytes)
    }
}

pub fn validate_supported_prefix(path: &LogicalPath, source: &[u8]) -> CollarResult<PrefixReport> {
    let Some(profile) = SyntaxProfile::for_path(path) else {
        return Ok(PrefixReport::unsupported());
    };
    let source = match std::str::from_utf8(source) {
        Ok(source) => source,
        Err(error) if error.error_len().is_none() => {
            return Ok(PrefixReport::repairable(profile, ""));
        }
        Err(_) => {
            return Ok(PrefixReport::impossible(
                profile,
                "",
                PrefixRule::InvalidUtf8,
            ));
        }
    };

    let failure = match profile {
        SyntaxProfile::Rust => scan_delimiters(source, ScanFlavor::Rust),
        SyntaxProfile::Python => scan_python(source),
        SyntaxProfile::TypeScript | SyntaxProfile::Tsx | SyntaxProfile::JavaScript => {
            scan_delimiters(source, ScanFlavor::JavaScript)
        }
        SyntaxProfile::Css => scan_delimiters(source, ScanFlavor::Css),
        SyntaxProfile::Html => None,
    };
    Ok(match failure {
        Some(rule) => PrefixReport::impossible(profile, source, rule),
        None => PrefixReport::repairable(profile, source),
    })
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

fn scan_delimiters(source: &str, flavor: ScanFlavor) -> Option<PrefixRule> {
    let bytes = source.as_bytes();
    let mut position = 0usize;
    let mut delimiters = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    let mut line_comment = false;
    let mut block_comment_depth = 0usize;
    let mut ambiguous_javascript_regex = false;
    let mut rust_raw_hashes = None;

    while position < bytes.len() {
        let byte = bytes[position];
        if let Some(hashes) = rust_raw_hashes {
            if byte == b'"'
                && bytes
                    .get(position + 1..position + 1 + hashes)
                    .is_some_and(|ending| ending.iter().all(|byte| *byte == b'#'))
            {
                rust_raw_hashes = None;
                position += 1 + hashes;
            } else {
                position += 1;
            }
            continue;
        }
        if line_comment {
            if byte == b'\n' {
                line_comment = false;
            }
            position += 1;
            continue;
        }
        if block_comment_depth > 0 {
            if bytes.get(position..position + 2) == Some(b"/*") && flavor == ScanFlavor::Rust {
                block_comment_depth = block_comment_depth.saturating_add(1);
                position += 2;
            } else if bytes.get(position..position + 2) == Some(b"*/") {
                block_comment_depth -= 1;
                position += 2;
            } else {
                position += 1;
            }
            continue;
        }
        if let Some(active) = quote {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if matches!(
                (active, byte),
                (Quote::Single, b'\'') | (Quote::Double, b'"') | (Quote::Template, b'`')
            ) {
                quote = None;
            }
            position += 1;
            continue;
        }

        if bytes.get(position..position + 2) == Some(b"//") && flavor != ScanFlavor::Css {
            line_comment = true;
            position += 2;
            continue;
        }
        if bytes.get(position..position + 2) == Some(b"/*") {
            block_comment_depth = 1;
            position += 2;
            continue;
        }
        if flavor == ScanFlavor::Rust
            && let Some((hashes, consumed)) = rust_raw_string_open(&bytes[position..])
        {
            rust_raw_hashes = Some(hashes);
            position += consumed;
            continue;
        }
        match byte {
            b'\'' => quote = Some(Quote::Single),
            b'"' => quote = Some(Quote::Double),
            b'`' if flavor == ScanFlavor::JavaScript => quote = Some(Quote::Template),
            b'/' if flavor == ScanFlavor::JavaScript => {
                // JavaScript's division/regular-expression ambiguity requires parser context.
                // Once observed, delimiter characters may belong to a regex character class, so
                // this conservative oracle stops making hard delimiter claims for the suffix.
                ambiguous_javascript_regex = true;
            }
            b'(' | b'[' | b'{' if !ambiguous_javascript_regex => delimiters.push(byte),
            b')' | b']' | b'}' if !ambiguous_javascript_regex => {
                let expected = match byte {
                    b')' => b'(',
                    b']' => b'[',
                    b'}' => b'{',
                    _ => unreachable!(),
                };
                if delimiters.pop() != Some(expected) {
                    return Some(PrefixRule::UnmatchedClosingDelimiter);
                }
            }
            _ => {}
        }
        position += 1;
    }
    None
}

fn rust_raw_string_open(input: &[u8]) -> Option<(usize, usize)> {
    let mut position = match input {
        [b'r', ..] => 1,
        [b'b', b'r', ..] => 2,
        _ => return None,
    };
    let mut hashes = 0usize;
    while input.get(position) == Some(&b'#') {
        hashes = hashes.saturating_add(1);
        position += 1;
    }
    (input.get(position) == Some(&b'"')).then_some((hashes, position + 1))
}

fn scan_python(source: &str) -> Option<PrefixRule> {
    if let Some(rule) = scan_python_delimiters(source) {
        return Some(rule);
    }
    // Strings, comments, explicit continuations, and tabs need the full lexical indentation
    // machine: delimiters inside them can otherwise make the approximation below believe an
    // implicit continuation ended early. The delimiter oracle above remains sound, but this local
    // dedent rule must defer rather than risk rejecting a valid continuation.
    if source.contains(['\'', '"', '#', '\t']) || source.lines().any(|line| line.ends_with('\\')) {
        return None;
    }
    let mut indentation = vec![0usize];
    let mut bracket_depth = 0usize;
    for line in source.split_inclusive('\n') {
        let content = line.trim_end_matches('\n').trim_end_matches('\r');
        let leading = content
            .len()
            .saturating_sub(content.trim_start_matches(' ').len());
        let trimmed = &content[leading..];
        if trimmed.is_empty() || trimmed.starts_with('#') || content[..leading].contains('\t') {
            continue;
        }
        if bracket_depth == 0 {
            let current = *indentation.last().unwrap_or(&0);
            if leading > current {
                indentation.push(leading);
            } else if leading < current {
                while indentation.last().is_some_and(|level| *level > leading) {
                    indentation.pop();
                }
                if indentation.last().copied() != Some(leading) {
                    return Some(PrefixRule::InvalidPythonDedent);
                }
            }
        }
        bracket_depth = approximate_python_bracket_depth(content, bracket_depth);
    }
    None
}

fn scan_python_delimiters(source: &str) -> Option<PrefixRule> {
    let bytes = source.as_bytes();
    let mut position = 0usize;
    let mut delimiters = Vec::new();
    let mut quote = None;
    let mut triple = false;
    let mut escaped = false;
    let mut comment = false;
    while position < bytes.len() {
        let byte = bytes[position];
        if comment {
            if byte == b'\n' {
                comment = false;
            }
            position += 1;
            continue;
        }
        if let Some(active) = quote {
            let delimiter = match active {
                Quote::Single => b'\'',
                Quote::Double => b'"',
                Quote::Template => unreachable!(),
            };
            if escaped {
                escaped = false;
                position += 1;
                continue;
            }
            if byte == b'\\' {
                escaped = true;
                position += 1;
                continue;
            }
            if triple && bytes.get(position..position + 3) == Some(&[delimiter; 3]) {
                quote = None;
                triple = false;
                position += 3;
                continue;
            }
            if !triple && byte == delimiter {
                quote = None;
            }
            position += 1;
            continue;
        }
        match byte {
            b'#' => comment = true,
            b'\'' | b'"' => {
                triple = bytes.get(position..position + 3) == Some(&[byte; 3]);
                quote = Some(if byte == b'\'' {
                    Quote::Single
                } else {
                    Quote::Double
                });
                position += if triple { 3 } else { 1 };
                continue;
            }
            b'(' | b'[' | b'{' => delimiters.push(byte),
            b')' | b']' | b'}' => {
                let expected = match byte {
                    b')' => b'(',
                    b']' => b'[',
                    b'}' => b'{',
                    _ => unreachable!(),
                };
                if delimiters.pop() != Some(expected) {
                    return Some(PrefixRule::UnmatchedClosingDelimiter);
                }
            }
            _ => {}
        }
        position += 1;
    }
    None
}

fn approximate_python_bracket_depth(line: &str, initial: usize) -> usize {
    line.bytes().fold(initial, |depth, byte| match byte {
        b'(' | b'[' | b'{' => depth.saturating_add(1),
        b')' | b']' | b'}' => depth.saturating_sub(1),
        _ => depth,
    })
}

fn boundary_for(profile: SyntaxProfile, source: &str) -> Option<AnalysisBoundary> {
    let last = source
        .as_bytes()
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())?;
    match source.as_bytes()[last] {
        b';' => Some(AnalysisBoundary::Statement),
        b'}' if matches!(
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
        b'\n' => Some(AnalysisBoundary::Statement),
        _ if source.ends_with('\n') => Some(AnalysisBoundary::Statement),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
