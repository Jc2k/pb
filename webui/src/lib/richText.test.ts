/// <reference lib="deno.ns" />
import { deepEqual, equal } from "node:assert/strict";
import { parseInlineRichText, parseRichText } from "./richText.ts";

Deno.test("parseInlineRichText renders Markdown emphasis and model path quoting", () => {
  deepEqual(
    parseInlineRichText(
      "Read `src/lib.rs`, then ~webui/src/App.tsx~ with **care** and *focus*.",
    ),
    [
      { type: "text", text: "Read " },
      { type: "code", text: "src/lib.rs" },
      { type: "text", text: ", then " },
      { type: "code", text: "webui/src/App.tsx" },
      { type: "text", text: " with " },
      { type: "strong", text: "care" },
      { type: "text", text: " and " },
      { type: "emphasis", text: "focus" },
      { type: "text", text: "." },
    ],
  );
});

Deno.test("parseRichText preserves newlines inside paragraphs", () => {
  deepEqual(parseRichText("first line\nsecond line"), [
    { type: "paragraph", lines: ["first line", "second line"] },
  ]);
});

Deno.test("parseRichText recognizes common markdown blocks", () => {
  deepEqual(
    parseRichText(
      "## Title\n\n- one\n- two\n\n1. first\n2. second\n\n```ts\nconst x = 1;\n```",
    ),
    [
      { type: "heading", level: 2, text: "Title" },
      { type: "unordered_list", items: ["one", "two"] },
      { type: "ordered_list", items: ["first", "second"] },
      { type: "code", code: "const x = 1;", language: "ts" },
    ],
  );
});

Deno.test("parseRichText treats HTML as inert text", () => {
  const [block] = parseRichText(
    "<img src=x onerror=alert(1)>\n<script>alert(1)</script>",
  );
  equal(block.type, "paragraph");
  if (block.type === "paragraph") {
    deepEqual(block.lines, [
      "<img src=x onerror=alert(1)>",
      "<script>alert(1)</script>",
    ]);
  }
});
