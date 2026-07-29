/// <reference lib="deno.ns" />
import { deepEqual, equal, ok } from "node:assert/strict";
import {
  appendSpokenText,
  type BrowserSpeechRecognition,
  type SpeechRecognitionErrorEventLike,
  speechRecognitionErrorMessage,
  type SpeechRecognitionResultEventLike,
  type SpeechRecognitionResultLike,
  VoiceInputController,
  type VoiceInputState,
} from "./speechRecognition.ts";

function result(
  transcript: string,
  isFinal = false,
): SpeechRecognitionResultLike {
  const value = [{ transcript }] as unknown as SpeechRecognitionResultLike;
  value.isFinal = isFinal;
  return value;
}

class FakeRecognition implements BrowserSpeechRecognition {
  static instances: FakeRecognition[] = [];
  continuous = false;
  interimResults = false;
  lang = "";
  maxAlternatives = 0;
  processLocally = false;
  onstart: (() => void) | null = null;
  onend: (() => void) | null = null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null = null;
  onresult: ((event: SpeechRecognitionResultEventLike) => void) | null = null;
  starts = 0;
  stops = 0;
  aborts = 0;

  constructor() {
    FakeRecognition.instances.push(this);
  }

  start() {
    this.starts += 1;
    this.onstart?.();
  }

  stop() {
    this.stops += 1;
  }

  abort() {
    this.aborts += 1;
  }

  emitResult(...results: SpeechRecognitionResultLike[]) {
    this.onresult?.({ resultIndex: 0, results });
  }

  emitEnd() {
    this.onend?.();
  }

  emitError(error: string) {
    this.onerror?.({ error });
  }
}

function makeController() {
  const states: VoiceInputState[] = [];
  const values: string[] = [];
  const errors: Array<string | null> = [];
  const controller = new VoiceInputController(FakeRecognition, {
    onStateChange: (state) => states.push(state),
    onValueChange: (value) => values.push(value),
    onError: (error) => errors.push(error),
  });
  return { controller, states, values, errors };
}

Deno.test("spoken text appends without damaging an existing prompt", () => {
  equal(appendSpokenText("", "  Fix the tests  "), "Fix the tests");
  equal(appendSpokenText("Research", "the cache"), "Research the cache");
  equal(appendSpokenText("Research ", "the cache"), "Research the cache");
  equal(appendSpokenText("Research ", "  "), "Research ");
});

Deno.test("voice input exposes interim text, stops cleanly, and uses a fresh recognizer next time", () => {
  FakeRecognition.instances = [];
  const { controller, states, values } = makeController();

  controller.start("Fix", "en-GB");
  const first = FakeRecognition.instances[0];
  ok(first);
  equal(first.continuous, true);
  equal(first.interimResults, true);
  equal(first.lang, "en-GB");
  equal(first.maxAlternatives, 1);
  equal(first.processLocally, true);
  first.emitResult(result("the login", false));
  equal(values.at(-1), "Fix the login");

  const staleResult = first.onresult;
  controller.stop();
  equal(first.stops, 1);
  equal(controller.currentState, "idle");
  staleResult?.({ resultIndex: 0, results: [result("wrong text")] });
  equal(values.at(-1), "Fix the login");

  controller.start("Fix the login", "en-GB");
  equal(first.aborts, 1);
  equal(FakeRecognition.instances.length, 2);
  equal(FakeRecognition.instances[1]?.starts, 1);
  deepEqual(states, ["starting", "listening", "idle", "starting", "listening"]);
  controller.dispose();
});

Deno.test("aborting a hidden or unmounted voice control releases current and retired sessions", () => {
  FakeRecognition.instances = [];
  const { controller } = makeController();
  controller.start("", "en-GB");
  const first = FakeRecognition.instances[0]!;
  controller.abort();
  equal(first.aborts, 1);
  equal(controller.currentState, "idle");

  controller.start("", "en-GB");
  const second = FakeRecognition.instances[1]!;
  second.emitEnd();
  controller.dispose();
  equal(second.aborts, 1);
});

Deno.test("cancelling while browser permission is pending aborts instead of stranding a start", () => {
  FakeRecognition.instances = [];
  class PendingRecognition extends FakeRecognition {
    override start() {
      this.starts += 1;
    }
  }
  const states: VoiceInputState[] = [];
  const controller = new VoiceInputController(PendingRecognition, {
    onStateChange: (state) => states.push(state),
    onValueChange: () => {},
    onError: () => {},
  });
  controller.start("", "en-GB");
  const pending = FakeRecognition.instances[0]!;
  equal(controller.currentState, "starting");
  controller.stop();
  equal(pending.aborts, 1);
  equal(pending.stops, 0);
  deepEqual(states, ["starting", "idle"]);
});

Deno.test("only one prompt control can own the microphone", () => {
  FakeRecognition.instances = [];
  const first = makeController();
  const second = makeController();
  first.controller.start("", "en-GB");
  const firstRecognition = FakeRecognition.instances[0]!;
  second.controller.start("", "en-GB");
  equal(firstRecognition.aborts, 1);
  equal(first.controller.currentState, "idle");
  equal(second.controller.currentState, "listening");
  second.controller.dispose();
});

Deno.test("speech failures are friendly and leave the next recording restartable", () => {
  FakeRecognition.instances = [];
  const { controller, errors } = makeController();
  controller.start("", "en-GB");
  const first = FakeRecognition.instances[0]!;
  first.emitError("not-allowed");
  equal(errors.at(-1), speechRecognitionErrorMessage("not-allowed"));
  equal(controller.currentState, "idle");

  controller.start("", "en-GB");
  equal(first.aborts, 1);
  equal(FakeRecognition.instances.length, 2);
  controller.dispose();
});

Deno.test("every user prompt composer includes the reusable voice control", async () => {
  const home = await Deno.readTextFile("webui/src/pages/HomePage.tsx");
  const project = await Deno.readTextFile("webui/src/pages/ProjectsPage.tsx");
  const session = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const goalStart = await Deno.readTextFile(
    "webui/src/components/GoalStartSheet.tsx",
  );
  const goalAmend = await Deno.readTextFile(
    "webui/src/components/GoalAmendmentSheet.tsx",
  );
  const hook = await Deno.readTextFile("webui/src/lib/useVoiceInput.ts");

  ok(home.includes("<VoiceInputButton"));
  ok(home.includes("disabled={!task.trim() || isSubmitting"));
  ok(project.includes("<VoiceInputButton"));
  ok(project.includes("disabled={!task.trim() || isSubmitting ||"));
  equal(session.match(/<VoiceInputButton/g)?.length, 3);
  ok(session.includes("!runningMessage.trim() || voiceInputActive"));
  ok(session.includes("!answer.trim() || voiceInputActive"));
  ok(session.includes("!followUp.trim() || voiceInputActive"));
  ok(goalStart.includes("<VoiceInputButton"));
  ok(goalAmend.includes("<VoiceInputButton"));
  ok(hook.includes('document.addEventListener("visibilitychange"'));
  ok(hook.includes('window.addEventListener("pagehide"'));
  ok(hook.includes("controller.dispose()"));
});
