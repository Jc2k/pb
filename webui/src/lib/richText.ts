export type RichTextBlock =
  | { type: "paragraph"; lines: string[] }
  | { type: "heading"; level: 1 | 2 | 3; text: string }
  | { type: "unordered_list"; items: string[] }
  | { type: "ordered_list"; items: string[] }
  | { type: "code"; code: string; language?: string };

const headingPattern = /^(#{1,3})\s+(.+)$/;
const unorderedPattern = /^\s*[-*+]\s+(.+)$/;
const orderedPattern = /^\s*\d+[.)]\s+(.+)$/;

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
      blocks.push({ type: "heading", level: heading[1].length as 1 | 2 | 3, text: heading[2] });
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
