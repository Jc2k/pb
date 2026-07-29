import { useCallback, useEffect, useRef, useState } from "react";
import {
  browserSpeechRecognitionConstructor,
  VoiceInputController,
  type VoiceInputState,
} from "./speechRecognition";

interface UseVoiceInputOptions {
  value: string;
  onValueChange: (value: string) => void;
  disabled?: boolean;
  onActiveChange?: (active: boolean) => void;
}

export function useVoiceInput({
  value,
  onValueChange,
  disabled = false,
  onActiveChange,
}: UseVoiceInputOptions) {
  const [Recognition] = useState(() => browserSpeechRecognitionConstructor());
  const [state, setState] = useState<VoiceInputState>("idle");
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<VoiceInputController | null>(null);
  const mountedRef = useRef(true);
  const valueRef = useRef(value);
  const onValueChangeRef = useRef(onValueChange);
  const onActiveChangeRef = useRef(onActiveChange);
  valueRef.current = value;
  onValueChangeRef.current = onValueChange;
  onActiveChangeRef.current = onActiveChange;

  useEffect(() => {
    mountedRef.current = true;
    if (!Recognition) return;
    const controller = new VoiceInputController(Recognition, {
      onStateChange: (nextState) => {
        if (!mountedRef.current) return;
        setState(nextState);
        onActiveChangeRef.current?.(nextState !== "idle");
      },
      onValueChange: (nextValue) => {
        if (mountedRef.current) onValueChangeRef.current(nextValue);
      },
      onError: (message) => {
        if (mountedRef.current) setError(message);
      },
    });
    controllerRef.current = controller;
    return () => {
      mountedRef.current = false;
      controller.dispose();
      controllerRef.current = null;
      onActiveChangeRef.current?.(false);
    };
  }, [Recognition]);

  useEffect(() => {
    const releaseForVisibility = () => {
      if (document.visibilityState === "hidden") controllerRef.current?.abort();
    };
    const releaseForPageHide = () => controllerRef.current?.abort();
    document.addEventListener("visibilitychange", releaseForVisibility);
    window.addEventListener("pagehide", releaseForPageHide);
    return () => {
      document.removeEventListener("visibilitychange", releaseForVisibility);
      window.removeEventListener("pagehide", releaseForPageHide);
    };
  }, []);

  useEffect(() => {
    if (disabled) controllerRef.current?.stop();
  }, [disabled]);

  const toggle = useCallback(() => {
    const controller = controllerRef.current;
    if (!controller || disabled) return;
    if (controller.currentState === "idle") {
      controller.start(valueRef.current, navigator.language || "en-US");
    } else {
      controller.stop();
    }
  }, [disabled]);

  return {
    supported: Boolean(Recognition),
    active: state !== "idle",
    state,
    error,
    toggle,
  };
}
