import type { TeamActor } from "../types";

export interface TeamPresentation {
  name: string;
  role: string;
  avatar: string;
  provenance: "Model" | "Harness" | "Legacy";
}

const PROFILES: Record<string, Omit<TeamPresentation, "provenance">> = {
  plan: {
    name: "Dade Murphy",
    role: "Ticket Goblin",
    avatar: "/static/images/avatar-plan.png",
  },
  build: {
    name: "Kate Libby",
    role: "Patch Crafter",
    avatar: "/static/images/avatar-build.png",
  },
  review: {
    name: "Eugene Belford",
    role: "Review Gremlin",
    avatar: "/static/images/avatar-review.png",
  },
  scout: {
    name: "Ramon Sanchez",
    role: "Env Scout",
    avatar: "/static/images/avatar-scout.png",
  },
  explore: {
    name: "Paul Cook",
    role: "Repo Mapper",
    avatar: "/static/images/avatar-explore.png",
  },
  research: {
    name: "Emmanuel Goldstein",
    role: "Web Sleuth",
    avatar: "/static/images/avatar-research.png",
  },
  monitor: {
    name: "Trinity Walker",
    role: "Progress Monitor",
    avatar: "/static/images/avatar-monitor.png",
  },
  ask: {
    name: "Joey Pardella",
    role: "Question Wrangler",
    avatar: "/static/images/avatar-ask.png",
  },
};

const UNKNOWN_PROFILE = {
  name: "Jon Appleseed",
  role: "Unknown",
  avatar: "/static/images/avatar.png",
};

const TRINITY_STEWARD: Omit<TeamPresentation, "provenance"> = {
  name: "Trinity Walker",
  role: "Team steward",
  avatar: "/static/images/avatar-monitor.png",
};

export function profilePresentation(profile: string) {
  return PROFILES[profile] || UNKNOWN_PROFILE;
}

export function getAvatarForProfile(profile: string): string {
  return profilePresentation(profile).avatar;
}

export function profileName(profile: string): string {
  return profilePresentation(profile).name;
}

export function profileJobTitle(profile: string): string {
  return profilePresentation(profile).role;
}

export function profileAccentClass(profile: string): string {
  return [
      "plan",
      "build",
      "review",
      "scout",
      "explore",
      "research",
      "ask",
    ].includes(profile)
    ? `teammate-${profile}`
    : "teammate-neutral";
}

export function teamActorPresentation(
  actor?: TeamActor,
): TeamPresentation {
  if (actor?.kind === "agent") {
    return {
      ...profilePresentation(actor.id),
      provenance: "Model",
    };
  }
  if (actor?.kind === "automation") {
    return { ...TRINITY_STEWARD, provenance: "Harness" };
  }
  return {
    name: "Agent",
    role: "Earlier session",
    avatar: "/static/images/avatar.png",
    provenance: "Legacy",
  };
}

export function teamActorKey(actor?: TeamActor): string {
  return actor ? `${actor.kind}:${actor.id}` : "legacy:unknown";
}

export function teamActorAccentClass(actor?: TeamActor): string {
  if (actor?.kind === "automation") {
    return "teammate-message trinity-message";
  }
  if (actor?.kind === "agent") {
    return `teammate-message ${profileAccentClass(actor.id)}`;
  }
  return "";
}

export function workflowStewardActor(): TeamActor {
  return { kind: "automation", id: "trinity" };
}
