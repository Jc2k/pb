export type VoiceInputState = "idle" | "starting" | "listening";

export type SpeechRecognitionResultLike = ArrayLike<{ transcript: string }> & {
  isFinal: boolean;
};

export interface SpeechRecognitionResultEventLike {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike>;
}

export interface SpeechRecognitionErrorEventLike {
  error: string;
}

export interface BrowserSpeechRecognition {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  processLocally?: boolean;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onresult: ((event: SpeechRecognitionResultEventLike) => void) | null;
  abort: () => void;
  start: () => void;
  stop: () => void;
}

export type BrowserSpeechRecognitionConstructor = new () =>
BrowserSpeechRecognition;

type SpeechRecognitionWindow =
  & Window
  & typeof globalThis
  & {
    SpeechRecognition?: BrowserSpeechRecognitionConstructor;
    webkitSpeechRecognition?: BrowserSpeechRecognitionConstructor;
  };

export interface VoiceInputControllerCallbacks {
  onStateChange: (state: VoiceInputState) => void;
  onValueChange: (value: string) => void;
  onError: (message: string | null) => void;
}

let activeController: VoiceInputController | null = null;

export function browserSpeechRecognitionConstructor():
  | BrowserSpeechRecognitionConstructor
  | undefined {
  if (typeof window === "undefined") return undefined;
  const speechWindow = window as SpeechRecognitionWindow;
  return speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
}

export function speechRecognitionErrorMessage(error: string): string {
  switch (error) {
    case "not-allowed":
    case "service-not-allowed":
      return "Microphone access was denied. Allow it in your browser settings and try again.";
    case "audio-capture":
      return "No microphone is available. Check your device settings and try again.";
    case "no-speech":
      return "I did not hear any speech. Try again or type the task instead.";
    case "language-not-supported":
      return "Voice input does not support your browser language. You can still type the task.";
    case "network":
      return "Your browser's speech service is unavailable. Try again or type the task instead.";
    default:
      return "Voice input stopped unexpectedly. Try again or type the task instead.";
  }
}

export function appendSpokenText(
  baseValue: string,
  transcript: string,
): string {
  const spoken = transcript.trim();
  if (!spoken) return baseValue;
  if (!baseValue) return spoken;
  return /\s$/.test(baseValue)
    ? `${baseValue}${spoken}`
    : `${baseValue} ${spoken}`;
}

function detachRecognition(recognition: BrowserSpeechRecognition) {
  recognition.onstart = null;
  recognition.onend = null;
  recognition.onerror = null;
  recognition.onresult = null;
}

/**
 * Owns one short-lived browser recognition session.
 *
 * A fresh browser object is created for every recording. Stopped objects are
 * explicitly aborted before the next start because Safari can otherwise keep
 * the old microphone session alive and reject later recordings.
 */
export class VoiceInputController {
  private current: BrowserSpeechRecognition | null = null;
  private retired: BrowserSpeechRecognition | null = null;
  private state: VoiceInputState = "idle";
  private baseValue = "";

  constructor(
    private readonly Recognition: BrowserSpeechRecognitionConstructor,
    private readonly callbacks: VoiceInputControllerCallbacks,
  ) {}

  get currentState(): VoiceInputState {
    return this.state;
  }

  start(baseValue: string, language: string) {
    this.abort();
    if (activeController && activeController !== this) {
      activeController.abort();
    }
    this.abortRetired();

    const recognition = new this.Recognition();
    this.current = recognition;
    this.baseValue = baseValue;
    activeController = this;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = language;
    recognition.maxAlternatives = 1;
    if ("processLocally" in recognition) recognition.processLocally = true;

    recognition.onstart = () => {
      if (this.current !== recognition) return;
      this.setState("listening");
    };
    recognition.onresult = (event) => {
      if (this.current !== recognition) return;
      const segments: string[] = [];
      for (let index = 0; index < event.results.length; index += 1) {
        const transcript = event.results[index]?.[0]?.transcript.trim();
        if (transcript) segments.push(transcript);
      }
      this.callbacks.onValueChange(
        appendSpokenText(this.baseValue, segments.join(" ")),
      );
    };
    recognition.onerror = (event) => {
      if (this.current !== recognition) return;
      if (event.error !== "aborted") {
        this.callbacks.onError(speechRecognitionErrorMessage(event.error));
      }
      this.finish(recognition);
    };
    recognition.onend = () => {
      if (this.current === recognition) this.finish(recognition);
    };

    this.callbacks.onError(null);
    this.setState("starting");
    try {
      recognition.start();
    } catch {
      this.release(recognition, true);
      this.callbacks.onError(
        "Voice input could not start. Try again or type the task instead.",
      );
    }
  }

  /** Stop and keep the latest visible interim transcript as ordinary text. */
  stop() {
    const recognition = this.current;
    if (!recognition) return;
    this.release(recognition, this.state === "starting");
  }

  /** Abort all browser resources. The controlled input already owns the text. */
  abort() {
    const recognition = this.current;
    if (recognition) this.release(recognition, true);
    this.abortRetired();
  }

  dispose() {
    this.abort();
  }

  private finish(recognition: BrowserSpeechRecognition) {
    if (this.current !== recognition) return;
    detachRecognition(recognition);
    this.current = null;
    this.retired = recognition;
    if (activeController === this) activeController = null;
    this.setState("idle");
  }

  private release(recognition: BrowserSpeechRecognition, abort: boolean) {
    if (this.current === recognition) this.current = null;
    if (this.retired === recognition) this.retired = null;
    detachRecognition(recognition);
    try {
      if (abort) recognition.abort();
      else recognition.stop();
    } catch {
      // The browser may already have ended the recognition object.
    }
    if (!abort) this.retired = recognition;
    if (activeController === this) activeController = null;
    this.setState("idle");
  }

  private abortRetired() {
    const recognition = this.retired;
    if (!recognition) return;
    this.retired = null;
    detachRecognition(recognition);
    try {
      recognition.abort();
    } catch {
      // Ended recognition objects are safe to forget.
    }
  }

  private setState(state: VoiceInputState) {
    if (this.state === state) return;
    this.state = state;
    this.callbacks.onStateChange(state);
  }
}
