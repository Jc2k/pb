/// <reference lib="deno.ns" />
import { deepEqual, equal } from "node:assert/strict";
import { parseRichText } from "./richText.ts";

Deno.test("parseRichText preserves newlines inside paragraphs", () => {
  deepEqual(parseRichText("first line\nsecond line"), [
    { type: "paragraph", lines: ["first line", "second line"] },
  ]);
});

Deno.test("parseRichText recognizes common markdown blocks", () => {
  deepEqual(parseRichText("## Title\n\n- one\n- two\n\n1. first\n2. second\n\n```ts\nconst x = 1;\n```"), [
    { type: "heading", level: 2, text: "Title" },
    { type: "unordered_list", items: ["one", "two"] },
    { type: "ordered_list", items: ["first", "second"] },
    { type: "code", code: "const x = 1;", language: "ts" },
  ]);
});

Deno.test("parseRichText treats HTML as inert text", () => {
  const [block] = parseRichText("<img src=x onerror=alert(1)>\n<script>alert(1)</script>");
  equal(block.type, "paragraph");
  if (block.type === "paragraph") {
    deepEqual(block.lines, ["<img src=x onerror=alert(1)>", "<script>alert(1)</script>"]);
  }
});
