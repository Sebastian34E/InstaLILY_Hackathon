import { useEffect, useCallback } from "react";

type SendFn = (event: object) => void;

export function useSpeech(
  send: SendFn,
  active: boolean
): { speak: (text: string) => void } {
  useEffect(() => {
    const SpeechRecognitionCtor =
      window.SpeechRecognition ?? window.webkitSpeechRecognition;

    if (!SpeechRecognitionCtor || !active) return;

    const rec = new SpeechRecognitionCtor();
    rec.continuous = true;
    rec.interimResults = false;
    rec.lang = "en-US";

    rec.onresult = (e: SpeechRecognitionEvent) => {
      const result = e.results[e.results.length - 1];
      if (result.isFinal) {
        send({
          type: "STUDENT_SPEECH",
          text: result[0].transcript,
          t: Date.now() / 1000,
        });
      }
    };

    rec.onerror = (e: SpeechRecognitionErrorEvent) =>
      console.error("Speech recognition error:", e.error);

    rec.start();
    return () => rec.stop();
  }, [send, active]);

  const speak = useCallback((text: string) => {
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  }, []);

  return { speak };
}
