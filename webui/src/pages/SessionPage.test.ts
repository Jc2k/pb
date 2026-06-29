/// <reference lib="deno.ns" />
import { ok } from "node:assert/strict";

Deno.test("paused session composer keeps resume action at intrinsic width", async () => {
  const markup = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");

  ok(markup.includes('className="composer paused-composer"'));
  ok(markup.includes('className="btn btn-warning composer-action"'));
  ok(css.includes(".composer .btn.composer-action"));
  ok(css.includes("width: auto;"));
  ok(css.includes("white-space: nowrap;"));
});
