/// <reference lib="deno.ns" />
import { deepEqual, equal } from "node:assert/strict";
import {
  profileAccentClass,
  profilePresentation,
  teamActorAccentClass,
  teamActorPresentation,
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
  const actor = { kind: "automation", id: "trinity" } as const;
  deepEqual(actor, { kind: "automation", id: "trinity" });
  const trinity = teamActorPresentation(actor);
  equal(trinity.name, "Trinity Walker");
  equal(trinity.role, "Team steward");
  equal(trinity.provenance, "Harness");
});

Deno.test("team message accents follow profile avatar palettes", () => {
  equal(profileAccentClass("review"), "teammate-review");
  equal(profileAccentClass("unknown"), "teammate-neutral");
  equal(
    teamActorAccentClass({ kind: "agent", id: "build" }),
    "teammate-message teammate-build",
  );
  equal(
    teamActorAccentClass({ kind: "automation", id: "trinity" }),
    "teammate-message trinity-message",
  );
});
