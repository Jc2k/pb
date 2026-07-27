use std::str;

use serde::{Deserialize, Serialize};
use tree_sitter::{Language, Node, Parser, Point, Tree};

use crate::{CollarError, CollarResult, mutation::LogicalPath};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntaxProfile {
    Rust,
    Python,
    TypeScript,
    Tsx,
    JavaScript,
    Html,
    Css,
}

impl SyntaxProfile {
    pub fn for_path(path: &LogicalPath) -> Option<Self> {
        let extension = path.as_str().rsplit_once('.')?.1.to_ascii_lowercase();
        match extension.as_str() {
            "rs" => Some(Self::Rust),
            "py" | "pyi" => Some(Self::Python),
            "ts" => Some(Self::TypeScript),
            "tsx" => Some(Self::Tsx),
            "js" | "jsx" | "mjs" | "cjs" => Some(Self::JavaScript),
            "html" | "htm" => Some(Self::Html),
            "css" => Some(Self::Css),
            _ => None,
        }
    }

    fn language(self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::Html => tree_sitter_html::LANGUAGE.into(),
            Self::Css => tree_sitter_css::LANGUAGE.into(),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::TypeScript => "typescript",
            Self::Tsx => "tsx",
            Self::JavaScript => "javascript",
            Self::Html => "html",
            Self::Css => "css",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyntaxReport {
    Valid { profile: SyntaxProfile },
    Unsupported,
}

pub fn validate_supported_syntax(path: &LogicalPath, source: &[u8]) -> CollarResult<SyntaxReport> {
    let Some(profile) = SyntaxProfile::for_path(path) else {
        return Ok(SyntaxReport::Unsupported);
    };
    str::from_utf8(source).map_err(|error| {
        CollarError::Analysis(format!(
            "{} source for {:?} is not valid UTF-8 at byte {}",
            profile.name(),
            path.as_str(),
            error.valid_up_to()
        ))
    })?;
    let tree = parse_complete(profile, source)?;
    reject_tree_errors(profile, &tree)?;
    if profile == SyntaxProfile::Html {
        validate_html_injections(source, &tree)?;
    }
    Ok(SyntaxReport::Valid { profile })
}

fn parse_complete(profile: SyntaxProfile, source: &[u8]) -> CollarResult<Tree> {
    let mut parser = Parser::new();
    parser.set_language(&profile.language()).map_err(|error| {
        CollarError::Analysis(format!(
            "failed to load pinned {} grammar: {error}",
            profile.name()
        ))
    })?;
    parser.parse(source, None).ok_or_else(|| {
        CollarError::Analysis(format!(
            "pinned {} parser returned no syntax tree",
            profile.name()
        ))
    })
}

fn reject_tree_errors(profile: SyntaxProfile, tree: &Tree) -> CollarResult<()> {
    let root = tree.root_node();
    if !root.has_error() && !root.is_error() && !root.is_missing() {
        return Ok(());
    }
    let invalid = first_invalid_node(root).unwrap_or(root);
    let Point { row, column } = invalid.start_position();
    let problem = if invalid.is_missing() {
        "missing syntax"
    } else {
        "syntax error"
    };
    Err(CollarError::Analysis(format!(
        "{} {problem} at {}:{} (byte {})",
        profile.name(),
        row.saturating_add(1),
        column.saturating_add(1),
        invalid.start_byte()
    )))
}

fn first_invalid_node(node: Node<'_>) -> Option<Node<'_>> {
    if node.is_error() || node.is_missing() {
        return Some(node);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.has_error() || child.is_error() || child.is_missing() {
            if let Some(invalid) = first_invalid_node(child) {
                return Some(invalid);
            }
        }
    }
    None
}

fn validate_html_injections(source: &[u8], tree: &Tree) -> CollarResult<()> {
    validate_html_node(source, tree.root_node())
}

fn validate_html_node(source: &[u8], node: Node<'_>) -> CollarResult<()> {
    match node.kind() {
        "element" => validate_html_element(source, node)?,
        "erroneous_end_tag" => {
            let Point { row, column } = node.start_position();
            return Err(CollarError::Analysis(format!(
                "html unexpected end tag at {}:{} (byte {})",
                row.saturating_add(1),
                column.saturating_add(1),
                node.start_byte()
            )));
        }
        "script_element" => validate_script_element(source, node)?,
        "style_element" => validate_style_element(source, node)?,
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        validate_html_node(source, child)?;
    }
    Ok(())
}

fn validate_html_element(source: &[u8], node: Node<'_>) -> CollarResult<()> {
    if named_child_of_kind(node, "self_closing_tag").is_some() {
        return Ok(());
    }
    let Some(start_tag) = named_child_of_kind(node, "start_tag") else {
        return Err(CollarError::Analysis(
            "html element is missing its start tag".to_string(),
        ));
    };
    let start_name = html_tag_name(source, start_tag)?;
    if let Some(end_tag) = named_child_of_kind(node, "end_tag") {
        let end_name = html_tag_name(source, end_tag)?;
        if !start_name.eq_ignore_ascii_case(end_name) {
            return Err(CollarError::Analysis(format!(
                "html element <{start_name}> closes with </{end_name}> at byte {}",
                end_tag.start_byte()
            )));
        }
        return Ok(());
    }
    if html_tag_may_end_implicitly(start_name) {
        return Ok(());
    }
    Err(CollarError::Analysis(format!(
        "html element <{start_name}> has no explicit end tag at byte {}",
        start_tag.start_byte()
    )))
}

fn html_tag_name<'source>(source: &'source [u8], tag: Node<'_>) -> CollarResult<&'source str> {
    let name = named_child_of_kind(tag, "tag_name").ok_or_else(|| {
        CollarError::Analysis(format!("html tag at byte {} has no name", tag.start_byte()))
    })?;
    source_slice(source, name)
}

fn html_tag_may_end_implicitly(name: &str) -> bool {
    [
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr", "html", "head", "body", "li", "dt", "dd", "p", "rt", "rp",
        "optgroup", "option", "colgroup", "thead", "tbody", "tfoot", "tr", "td", "th",
    ]
    .iter()
    .any(|candidate| name.eq_ignore_ascii_case(candidate))
}

fn validate_script_element(source: &[u8], node: Node<'_>) -> CollarResult<()> {
    let script_type = element_type_attribute(source, node)?;
    let profile = match script_type.as_deref() {
        None
        | Some("")
        | Some("module")
        | Some("text/javascript")
        | Some("application/javascript")
        | Some("text/ecmascript")
        | Some("application/ecmascript") => {
            Some(EmbeddedProfile::Syntax(SyntaxProfile::JavaScript))
        }
        Some("text/typescript") | Some("application/typescript") => {
            Some(EmbeddedProfile::Syntax(SyntaxProfile::TypeScript))
        }
        Some("application/json")
        | Some("application/ld+json")
        | Some("importmap")
        | Some("speculationrules") => Some(EmbeddedProfile::Json),
        Some(_) => None,
    };
    validate_embedded_raw_text(source, node, profile)
}

fn validate_style_element(source: &[u8], node: Node<'_>) -> CollarResult<()> {
    let style_type = element_type_attribute(source, node)?;
    let profile = match style_type.as_deref() {
        None | Some("") | Some("text/css") => Some(EmbeddedProfile::Syntax(SyntaxProfile::Css)),
        Some(_) => None,
    };
    validate_embedded_raw_text(source, node, profile)
}

enum EmbeddedProfile {
    Syntax(SyntaxProfile),
    Json,
}

fn validate_embedded_raw_text(
    source: &[u8],
    node: Node<'_>,
    profile: Option<EmbeddedProfile>,
) -> CollarResult<()> {
    let Some(profile) = profile else {
        return Ok(());
    };
    let Some(raw_text) = named_child_of_kind(node, "raw_text") else {
        return Ok(());
    };
    let bytes = source
        .get(raw_text.byte_range())
        .ok_or_else(|| CollarError::Analysis("HTML embedded range is out of bounds".to_string()))?;
    match profile {
        EmbeddedProfile::Syntax(profile) => {
            let tree = parse_complete(profile, bytes)?;
            reject_tree_errors(profile, &tree)
        }
        EmbeddedProfile::Json => serde_json::from_slice::<serde_json::Value>(bytes)
            .map(|_| ())
            .map_err(|error| {
                CollarError::Analysis(format!(
                    "embedded HTML JSON syntax error at line {} column {}",
                    error.line(),
                    error.column()
                ))
            }),
    }
}

fn element_type_attribute(source: &[u8], node: Node<'_>) -> CollarResult<Option<String>> {
    let Some(start_tag) = named_child_of_kind(node, "start_tag") else {
        return Ok(None);
    };
    let mut cursor = start_tag.walk();
    for attribute in start_tag
        .named_children(&mut cursor)
        .filter(|child| child.kind() == "attribute")
    {
        let mut attribute_cursor = attribute.walk();
        let named = attribute
            .named_children(&mut attribute_cursor)
            .collect::<Vec<_>>();
        let Some(name) = named.first() else {
            continue;
        };
        let name = source_slice(source, *name)?;
        if !name.eq_ignore_ascii_case("type") {
            continue;
        }
        let Some(value) = named.get(1) else {
            return Ok(Some(String::new()));
        };
        let value = source_slice(source, *value)?;
        return Ok(Some(
            value
                .trim()
                .trim_matches(['\'', '"'])
                .trim()
                .to_ascii_lowercase(),
        ));
    }
    Ok(None)
}

fn source_slice<'source>(source: &'source [u8], node: Node<'_>) -> CollarResult<&'source str> {
    let bytes = source
        .get(node.byte_range())
        .ok_or_else(|| CollarError::Analysis("syntax node range is out of bounds".to_string()))?;
    str::from_utf8(bytes)
        .map_err(|_| CollarError::Analysis("syntax node is not valid UTF-8".to_string()))
}

fn named_child_of_kind<'tree>(node: Node<'tree>, kind: &str) -> Option<Node<'tree>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .find(|child| child.kind() == kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(path: &str, source: &str) -> CollarResult<SyntaxReport> {
        validate_supported_syntax(&LogicalPath::parse(path).unwrap(), source.as_bytes())
    }

    #[test]
    fn accepts_valid_priority_language_files() {
        for (path, source) in [
            ("src/lib.rs", "pub fn answer() -> i32 { 42 }\n"),
            ("main.py", "def answer() -> int:\n    return 42\n"),
            ("main.ts", "export const answer: number = 42;\n"),
            ("main.tsx", "export const el = <div>{42}</div>;\n"),
            ("main.js", "export const answer = 42;\n"),
            ("index.html", "<!doctype html><p>valid</p>\n"),
            ("style.css", "p { color: red; }\n"),
        ] {
            assert!(
                matches!(validate(path, source), Ok(SyntaxReport::Valid { .. })),
                "expected {path} to be valid"
            );
        }
    }

    #[test]
    fn rejects_invalid_priority_language_files() {
        for (path, source) in [
            ("src/lib.rs", "pub fn answer( {\n"),
            ("main.py", "def answer(:\n    pass\n"),
            ("main.ts", "const answer: = 42;\n"),
            ("main.tsx", "const el = <div>;\n"),
            ("main.js", "const = 42;\n"),
            ("index.html", "<div><span></div>\n"),
            ("style.css", "p { color: red;\n"),
        ] {
            assert!(validate(path, source).is_err(), "accepted invalid {path}");
        }
    }

    #[test]
    fn html_validates_supported_embedded_languages() {
        validate(
            "index.html",
            r#"<script type="module">const value = 1;</script>
<script type="application/json">{"value":1}</script>
<style>body { color: red; }</style>"#,
        )
        .unwrap();

        assert!(validate("index.html", "<script>const = 1;</script>").is_err());
        assert!(
            validate(
                "index.html",
                "<script type=\"application/json\">{invalid}</script>"
            )
            .is_err()
        );
        assert!(validate("index.html", "<style>body {</style>").is_err());
    }

    #[test]
    fn html_leaves_explicit_unknown_raw_text_uninterpreted() {
        validate(
            "index.html",
            "<script type=\"x-shader/x-fragment\">not javascript {{</script>",
        )
        .unwrap();
    }

    #[test]
    fn unsupported_extensions_are_explicit() {
        assert_eq!(
            validate("README.md", "# valid").unwrap(),
            SyntaxReport::Unsupported
        );
    }
}
