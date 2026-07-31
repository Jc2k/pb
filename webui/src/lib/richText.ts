export type RichTextBlock =
  | { type: "paragraph"; lines: string[] }
  | { type: "heading"; level: 1 | 2 | 3; text: string }
  | { type: "unordered_list"; items: string[] }
  | { type: "ordered_list"; items: string[] }
  | { type: "code"; code: string; language?: string };

export type RichTextInline =
  | { type: "text"; text: string }
  | { type: "code"; text: string }
  | { type: "strong"; text: string }
  | { type: "emphasis"; text: string };

const headingPattern = /^(#{1,3})\s+(.+)$/;
const unorderedPattern = /^\s*[-*+]\s+(.+)$/;
const orderedPattern = /^\s*\d+[.)]\s+(.+)$/;
const inlinePattern = /(`[^`\n]+`|~[^~\n]+~|\*\*[^*\n]+\*\*|\*[^*\n]+\*)/g;

export function parseInlineRichText(input: string): RichTextInline[] {
  const parts: RichTextInline[] = [];
  let cursor = 0;

  for (const match of input.matchAll(inlinePattern)) {
    const index = match.index ?? 0;
    if (index > cursor) {
      parts.push({ type: "text", text: input.slice(cursor, index) });
    }

    const token = match[0];
    const content = token.startsWith("**")
      ? token.slice(2, -2)
      : token.slice(1, -1);
    if (token.startsWith("`")) {
      parts.push({ type: "code", text: content });
    } else if (
      token.startsWith("~") &&
      (/[/\\]/.test(content) || /^[\w.-]+\.[\w-]+$/.test(content))
    ) {
      parts.push({ type: "code", text: content });
    } else if (token.startsWith("**")) {
      parts.push({ type: "strong", text: content });
    } else if (token.startsWith("*")) {
      parts.push({ type: "emphasis", text: content });
    } else {
      parts.push({ type: "text", text: token });
    }
    cursor = index + token.length;
  }

  if (cursor < input.length) {
    parts.push({ type: "text", text: input.slice(cursor) });
  }
  return parts.length > 0 ? parts : [{ type: "text", text: input }];
}

function flushParagraph(blocks: RichTextBlock[], paragraph: string[]) {
  if (paragraph.length === 0) return;
  blocks.push({ type: "paragraph", lines: [...paragraph] });
  paragraph.length = 0;
}

function collectList(
  lines: string[],
  start: number,
  pattern: RegExp,
): { items: string[]; nextIndex: number } {
  const items: string[] = [];
  let index = start;
  while (index < lines.length) {
    const match = lines[index].match(pattern);
    if (!match) break;
    items.push(match[1]);
    index += 1;
  }
  return { items, nextIndex: index };
}

export function parseRichText(input: string): RichTextBlock[] {
  const blocks: RichTextBlock[] = [];
  const paragraph: string[] = [];
  const lines = input.replace(/\r\n?/g, "\n").split("\n");

  for (let index = 0; index < lines.length;) {
    const line = lines[index];

    if (line.startsWith("```")) {
      flushParagraph(blocks, paragraph);
      const language = line.slice(3).trim() || undefined;
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      blocks.push({ type: "code", code: codeLines.join("\n"), language });
      continue;
    }

    if (line.trim() === "") {
      flushParagraph(blocks, paragraph);
      index += 1;
      continue;
    }

    const heading = line.match(headingPattern);
    if (heading) {
      flushParagraph(blocks, paragraph);
      blocks.push({
        type: "heading",
        level: heading[1].length as 1 | 2 | 3,
        text: heading[2],
      });
      index += 1;
      continue;
    }

    if (unorderedPattern.test(line)) {
      flushParagraph(blocks, paragraph);
      const list = collectList(lines, index, unorderedPattern);
      blocks.push({ type: "unordered_list", items: list.items });
      index = list.nextIndex;
      continue;
    }

    if (orderedPattern.test(line)) {
      flushParagraph(blocks, paragraph);
      const list = collectList(lines, index, orderedPattern);
      blocks.push({ type: "ordered_list", items: list.items });
      index = list.nextIndex;
      continue;
    }

    paragraph.push(line);
    index += 1;
  }

  flushParagraph(blocks, paragraph);
  return blocks;
}
