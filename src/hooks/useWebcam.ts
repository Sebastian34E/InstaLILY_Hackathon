import { useEffect, useRef, useCallback } from "react";
import type { RefObject } from "react";

type SendFn = (event: object) => void;

export function useWebcam(
  send: SendFn,
  active: boolean
): {
  videoRef: RefObject<HTMLVideoElement | null>;
  hiddenCanvasRef: RefObject<HTMLCanvasElement | null>;
  captureWorksheet: () => void;
} {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sendRef = useRef(send);
  sendRef.current = send;

  // Effect 1: acquire camera once on mount; cleanup stops the stream
  useEffect(() => {
    let cancelled = false;

    async function start() {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: true });
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = s;
        if (videoRef.current) {
          videoRef.current.srcObject = s;
        }
        if (hiddenCanvasRef.current) {
          hiddenCanvasRef.current.width = 320;
          hiddenCanvasRef.current.height = 240;
        }
      } catch {
        console.warn("Camera unavailable — face frames disabled");
      }
    }

    start();

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Effect 2: face-frame interval, only when active
  useEffect(() => {
    if (!active) return;

    const interval = setInterval(() => {
      const video = videoRef.current;
      const canvas = hiddenCanvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
      const image = dataUrl.replace("data:image/jpeg;base64,", "");
      sendRef.current({ type: "FACE_FRAME", image, t: Date.now() / 1000 });
    }, 2000);

    return () => clearInterval(interval);
  }, [active]);

  const captureWorksheet = useCallback(() => {
    const video = videoRef.current;
    const canvas = hiddenCanvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
    const image = dataUrl.replace("data:image/jpeg;base64,", "");
    sendRef.current({ type: "WORKSHEET_PHOTO", image });
  }, []);

  return { videoRef, hiddenCanvasRef, captureWorksheet };
}
