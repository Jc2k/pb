/// <reference lib="deno.ns" />
import { deepEqual, equal } from "node:assert/strict";
import {
  profilePresentation,
  teamActorPresentation,
  workflowStewardActor,
} from "./team.ts";

Deno.test("profile characters own model actions", () => {
  const kate = teamActorPresentation({ kind: "agent", id: "build" });
  equal(kate.name, "Kate Libby");
  equal(kate.role, "Patch Crafter");
  equal(kate.provenance, "Model");
  equal(kate.avatar, "/static/images/avatar-build.png");
  equal(profilePresentation("review").name, "Eugene Belford");
});

Deno.test("Trinity owns harness workflow stewardship", () => {
  deepEqual(workflowStewardActor(), { kind: "automation", id: "trinity" });
  const trinity = teamActorPresentation(workflowStewardActor());
  equal(trinity.name, "Trinity Walker");
  equal(trinity.role, "Team steward");
  equal(trinity.provenance, "Harness");
});

Deno.test("legacy actorless tool actions remain unattributed", () => {
  const legacy = teamActorPresentation();
  equal(legacy.name, "Agent");
  equal(legacy.provenance, "Legacy");
});
