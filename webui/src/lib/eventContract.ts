import type {
  EventEnvelope,
  ProjectSessionSnapshot,
  ProjectSessionTerminalTransition,
  SessionDetails,
  SessionStreamSnapshot,
} from "../types";

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function json(text: string, label: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error(`${label} is not valid JSON`);
  }
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
}

function nonNegativeInteger(value: unknown, label: string): number {
  const number = finiteNumber(value, label);
  if (!Number.isSafeInteger(number) || number < 0) {
    throw new Error(`${label} must be a non-negative safe integer`);
  }
  return number;
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || !value) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

const handoffOutcomes = new Set([
  "pending",
  "ready",
  "no_change",
  "checks_failed",
  "executor_unavailable",
  "commit_blocked",
  "repair_exhausted",
  "incomplete",
]);

function terminalTransition(
  value: unknown,
): ProjectSessionTerminalTransition {
  const transition = record(value, "project session terminal transition");
  requiredString(transition.entry_key, "terminal transition entry key");
  nonNegativeInteger(transition.revision, "terminal transition revision");
  requiredString(transition.session_id, "terminal transition session id");
  if (transition.status !== "completed" && transition.status !== "failed") {
    throw new Error("terminal transition status must be completed or failed");
  }
  requiredString(transition.task, "terminal transition task");
  if (transition.title !== null && typeof transition.title !== "string") {
    throw new Error("terminal transition title must be a string or null");
  }
  if (
    transition.handoff_outcome !== null &&
    !handoffOutcomes.has(String(transition.handoff_outcome))
  ) {
    throw new Error("terminal transition handoff outcome is invalid");
  }
  if (transition.project !== null) {
    const project = record(transition.project, "terminal transition project");
    requiredString(project.id, "terminal transition project id");
    requiredString(project.name, "terminal transition project name");
    requiredString(project.path, "terminal transition project path");
    if (typeof project.notify_on_finish !== "boolean") {
      throw new Error(
        "terminal transition notification setting must be boolean",
      );
    }
  }
  return transition as unknown as ProjectSessionTerminalTransition;
}

function eventEnvelope(value: unknown): EventEnvelope {
  const envelope = record(value, "event envelope");
  if (envelope.version !== "v5") {
    throw new Error(
      `unsupported event schema '${String(envelope.version)}'; expected 'v5'`,
    );
  }
  const event = record(envelope.event, "event payload");
  if (typeof event.type !== "string" || !event.type) {
    throw new Error("event payload type must be a non-empty string");
  }
  if (!Array.isArray(envelope.chatter) || !Array.isArray(envelope.evidence)) {
    throw new Error("event projections must be arrays");
  }
  const transcript = record(envelope.transcript, "event transcript");
  if (nonNegativeInteger(transcript.sequence, "event sequence") < 1) {
    throw new Error("event sequence must be positive");
  }
  if (typeof transcript.entry_key !== "string" || !transcript.entry_key) {
    throw new Error("event entry key must be a non-empty string");
  }
  record(transcript.session_effect, "event session effect");
  return envelope as unknown as EventEnvelope;
}

function sessionDetails(value: unknown): SessionDetails {
  const session = record(value, "session snapshot");
  nonNegativeInteger(session.revision, "session revision");
  if (typeof session.session_id !== "string" || !session.session_id) {
    throw new Error("session id must be a non-empty string");
  }
  if (!Array.isArray(session.events)) {
    throw new Error("session events must be an array");
  }
  session.events.forEach(eventEnvelope);
  return session as unknown as SessionDetails;
}

export function parseEventEnvelopeJson(text: string): EventEnvelope {
  return eventEnvelope(json(text, "event envelope"));
}

export function parseSessionDetailsJson(text: string): SessionDetails {
  return sessionDetails(json(text, "session snapshot"));
}

export function parseSessionStreamSnapshotJson(
  text: string,
): SessionStreamSnapshot {
  const snapshot = record(
    json(text, "session stream snapshot"),
    "session stream snapshot",
  );
  if (typeof snapshot.reset_history !== "boolean") {
    throw new Error("session reset_history must be a boolean");
  }
  return {
    session: sessionDetails(snapshot.session),
    reset_history: snapshot.reset_history,
  };
}

export function parseProjectSessionSnapshotJson(
  text: string,
): ProjectSessionSnapshot {
  const snapshot = record(
    json(text, "project session snapshot"),
    "project session snapshot",
  );
  if (typeof snapshot.stream_id !== "string" || !snapshot.stream_id) {
    throw new Error("project session stream id must be a non-empty string");
  }
  const revision = nonNegativeInteger(
    snapshot.revision,
    "project session revision",
  );
  const transitionFloor = nonNegativeInteger(
    snapshot.terminal_transition_floor,
    "project session terminal transition floor",
  );
  if (transitionFloor > revision) {
    throw new Error("terminal transition floor exceeds the snapshot revision");
  }
  if (!Array.isArray(snapshot.terminal_transitions)) {
    throw new Error("project session terminal transitions must be an array");
  }
  const transitions = snapshot.terminal_transitions.map(terminalTransition);
  if (transitions.some((transition) => transition.revision > revision)) {
    throw new Error("terminal transition exceeds the snapshot revision");
  }
  if (
    new Set(transitions.map((transition) => transition.entry_key)).size !==
      transitions.length
  ) {
    throw new Error("project session terminal transitions contain duplicates");
  }
  if (!Array.isArray(snapshot.projects) || !Array.isArray(snapshot.sessions)) {
    throw new Error("project session collections must be arrays");
  }
  record(snapshot.project_usage, "project usage snapshot");
  return snapshot as unknown as ProjectSessionSnapshot;
}
