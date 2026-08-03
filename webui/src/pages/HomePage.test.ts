/// <reference lib="deno.ns" />
import { ok } from "node:assert/strict";

Deno.test("home session start surfaces structured server failures safely", async () => {
  const source = await Deno.readTextFile("webui/src/pages/HomePage.tsx");

  ok(source.includes('apiErrorMessage(res, "Could not start the session")'));
  ok(source.includes("LatestRequest"));
  ok(source.includes("startRequest.current.owns(controller)"));
  ok(source.includes('className="alert alert-danger" role="alert"'));
  ok(!source.includes("if (!res.ok) return"));
});
