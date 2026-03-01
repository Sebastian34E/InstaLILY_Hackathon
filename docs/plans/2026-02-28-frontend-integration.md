# Frontend Backend Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Wire the existing React/TypeScript frontend to the live FastAPI WebSocket backend — adding canvas drawing, voice I/O, and webcam capture — while preserving the existing hardcoded Q&A as a fallback.

**Architecture:** Four new files (useWebSocket, useSpeech, useWebcam hooks + WhiteboardCanvas component) are added. LessonPage is modified to use them: shows backend-driven canvas+voice UI when connected, falls back to existing text Q&A when disconnected. WebSocket URL comes from `VITE_WS_URL` env var.

**Tech Stack:** React 19, TypeScript (strict), Vite, HTML5 Canvas, Web Speech API, MediaDevices API, WebSocket

---

## Task 1: Environment config

**Files:**
- Create: `src/env.d.ts`
- Create: `.env.local`

**Step 1: Create the env type declaration**

Create `src/env.d.ts`:
```typescript
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_WS_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

**Step 2: Create `.env.local`**

Create `.env.local` in the project root (next to `package.json`):
```
VITE_WS_URL=wss://0e33-34-133-228-240.ngrok-free.app/ws
```

Note: `.env.local` is git-ignored by default in Vite projects. Do not commit it.

**Step 3: Verify**

```bash
npm run build
```
Expected: exits 0, no TypeScript errors.

---

## Task 2: Create `useWebSocket` hook

**Files:**
- Create: `src/hooks/useWebSocket.ts`

**Step 1: Create the hooks directory and file**

Create `src/hooks/useWebSocket.ts`:
```typescript
import { useEffect, useRef, useState, useCallback } from "react";

export type BackendAction = { type: string; [key: string]: unknown };
export type WsStatus = "connecting" | "connected" | "disconnected";

export function useWebSocket(
  url: string,
  onAction: (action: BackendAction) => void
): { send: (event: object) => void; wsStatus: WsStatus } {
  const [wsStatus, setWsStatus] = useState<WsStatus>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const onActionRef = useRef(onAction);
  onActionRef.current = onAction;

  useEffect(() => {
    if (!url) {
      setWsStatus("disconnected");
      return;
    }

    let cancelled = false;

    function connect() {
      if (cancelled) return;
      setWsStatus("connecting");
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) { ws.close(); return; }
        setWsStatus("connected");
        retriesRef.current = 0;
      };

      ws.onmessage = (e) => {
        try {
          onActionRef.current(JSON.parse(e.data) as BackendAction);
        } catch {
          console.error("Bad WS message:", e.data);
        }
      };

      ws.onclose = () => {
        if (cancelled) return;
        if (retriesRef.current < 5) {
          const delay = Math.min(1000 * 2 ** retriesRef.current, 10000);
          retriesRef.current++;
          setTimeout(connect, delay);
        } else {
          setWsStatus("disconnected");
        }
      };

      ws.onerror = () => console.error("WebSocket error");
    }

    connect();

    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, [url]);

  const send = useCallback((event: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(event));
    }
  }, []);

  return { send, wsStatus };
}
```

**Step 2: Type-check**

```bash
npm run build
```
Expected: exits 0.

---

## Task 3: Create `WhiteboardCanvas` component

**Files:**
- Create: `src/WhiteboardCanvas.tsx`

This is the HTML5 canvas that executes drawing commands from the backend. It uses `forwardRef` + `useImperativeHandle` so LessonPage can call `whiteboardRef.current.execute(action)` imperatively.

Coordinate grid (600×480px):
- Divisor at col 140, row 90
- Bracket: vertical line x=145 from y=100 to y=65, horizontal line to x=330
- Dividend digits: col 200, 250, 300 at row 90
- Quotient digits: col 200, 250, 300 at row 40
- Quotient bar: y=60, x=150 to x=330

**Step 1: Create the file**

Create `src/WhiteboardCanvas.tsx`:
```typescript
import { forwardRef, useImperativeHandle, useRef } from "react";
import type { BackendAction } from "./hooks/useWebSocket";

export interface WhiteboardCanvasRef {
  execute(action: BackendAction): void;
}

const POSITIONS: Record<string, { row: number; col: number }> = {
  quotient_0: { row: 40, col: 200 },
  quotient_1: { row: 40, col: 250 },
  quotient_2: { row: 40, col: 300 },
  subtract_row_0: { row: 130, col: 300 },
  subtract_row_1: { row: 220, col: 300 },
  remainder: { row: 310, col: 350 },
};

const LINE_ROWS: Record<string, number> = {
  subtract_line_0: 150,
  subtract_line_1: 240,
};

const WhiteboardCanvas = forwardRef<WhiteboardCanvasRef>((_, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  function getCtx() {
    return canvasRef.current?.getContext("2d") ?? null;
  }

  useImperativeHandle(ref, () => ({
    execute(action: BackendAction) {
      const c = getCtx();
      if (!c) return;

      switch (action.type) {
        case "DRAW_PROBLEM": {
          const dividend = action.dividend as number;
          const divisor = action.divisor as number;
          c.clearRect(0, 0, 600, 480);

          // Divisor
          c.font = "bold 28px monospace";
          c.fillStyle = "#1a1a1a";
          c.textAlign = "right";
          c.textBaseline = "alphabetic";
          c.fillText(String(divisor), 140, 90);

          // Long division bracket
          c.strokeStyle = "#1a1a1a";
          c.lineWidth = 2;
          c.beginPath();
          c.moveTo(145, 100);
          c.lineTo(145, 65);
          c.lineTo(330, 65);
          c.stroke();

          // Dividend digits
          const digits = String(dividend);
          c.textAlign = "center";
          for (let i = 0; i < digits.length; i++) {
            c.fillText(digits[i], 200 + i * 50, 90);
          }

          // Quotient bar
          c.beginPath();
          c.moveTo(150, 60);
          c.lineTo(330, 60);
          c.stroke();
          break;
        }

        case "DRAW_CIRCLE": {
          const target = action.target as string;
          c.strokeStyle = "#e53e3e";
          c.lineWidth = 2;
          c.beginPath();
          if (target === "first_digit") {
            c.ellipse(200, 82, 22, 18, 0, 0, 2 * Math.PI);
          } else if (target === "first_two_digits") {
            c.ellipse(225, 82, 47, 18, 0, 0, 2 * Math.PI);
          } else if (target === "first_three_digits") {
            c.ellipse(250, 82, 72, 18, 0, 0, 2 * Math.PI);
          }
          c.stroke();
          break;
        }

        case "DRAW_NUMBER":
        case "DRAW_MULTIPLY": {
          const pos = POSITIONS[action.position as string];
          if (!pos) break;
          c.font = "bold 28px monospace";
          c.fillStyle = "#1a1a1a";
          c.textAlign = "right";
          c.textBaseline = "alphabetic";
          c.fillText(String(action.value), pos.col, pos.row);
          break;
        }

        case "DRAW_LINE": {
          const row = LINE_ROWS[action.position as string];
          if (!row) break;
          c.strokeStyle = "#1a1a1a";
          c.lineWidth = 2;
          c.beginPath();
          c.moveTo(150, row);
          c.lineTo(330, row);
          c.stroke();
          break;
        }

        case "DRAW_BRING_DOWN": {
          const idx = action.digit_index as number;
          const fromX = 200 + idx * 50;
          c.strokeStyle = "#e53e3e";
          c.lineWidth = 2;
          c.beginPath();
          c.moveTo(fromX, 100);
          c.quadraticCurveTo(fromX + 25, 155, fromX, 185);
          c.stroke();
          // Arrowhead
          c.fillStyle = "#e53e3e";
          c.beginPath();
          c.moveTo(fromX, 185);
          c.lineTo(fromX - 6, 172);
          c.lineTo(fromX + 6, 172);
          c.closePath();
          c.fill();
          break;
        }

        case "DRAW_REMAINDER": {
          c.font = "bold 28px monospace";
          c.fillStyle = "#1a1a1a";
          c.textAlign = "left";
          c.textBaseline = "alphabetic";
          c.fillText(`R ${action.value as number}`, 310, 310);
          break;
        }

        case "DRAW_HINT": {
          if (action.hint_type === "show_multiplication_table") {
            const a = action.a as number;
            c.fillStyle = "rgba(255,255,255,0.95)";
            c.fillRect(340, 40, 210, 265);
            c.strokeStyle = "#1a1a1a";
            c.lineWidth = 1;
            c.strokeRect(340, 40, 210, 265);
            c.font = "16px monospace";
            c.fillStyle = "#1a1a1a";
            c.textAlign = "left";
            c.textBaseline = "alphabetic";
            for (let i = 1; i <= 9; i++) {
              c.fillText(`${a} × ${i} = ${a * i}`, 355, 62 + (i - 1) * 27);
            }
          }
          break;
        }

        default:
          break;
      }
    },
  }));

  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={480}
      style={{
        border: "2px solid #9fd1ff",
        borderRadius: "8px",
        background: "white",
        maxWidth: "100%",
      }}
    />
  );
});

WhiteboardCanvas.displayName = "WhiteboardCanvas";
export default WhiteboardCanvas;
```

**Step 2: Type-check**

```bash
npm run build
```
Expected: exits 0.

---

## Task 4: Create `useSpeech` hook

**Files:**
- Create: `src/hooks/useSpeech.ts`

**Step 1: Create the file**

Create `src/hooks/useSpeech.ts`:
```typescript
import { useEffect, useCallback } from "react";

type SendFn = (event: object) => void;

export function useSpeech(
  send: SendFn,
  active: boolean
): { speak: (text: string) => void } {
  useEffect(() => {
    const SpeechRecognitionCtor =
      window.SpeechRecognition ??
      (
        window as unknown as {
          webkitSpeechRecognition?: typeof SpeechRecognition;
        }
      ).webkitSpeechRecognition;

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
```

**Step 2: Type-check**

```bash
npm run build
```
Expected: exits 0. (SpeechRecognition types are in `lib.dom.d.ts` — already included via tsconfig `"lib": ["ES2020","DOM","DOM.Iterable"]`.)

If you get `Cannot find name 'SpeechRecognition'`, add to `tsconfig.app.json` under `compilerOptions`:
```json
"lib": ["ES2022", "DOM", "DOM.Iterable"]
```

---

## Task 5: Create `useWebcam` hook

**Files:**
- Create: `src/hooks/useWebcam.ts`

**Step 1: Create the file**

Create `src/hooks/useWebcam.ts`:
```typescript
import { useEffect, useRef, useCallback } from "react";

type SendFn = (event: object) => void;

export function useWebcam(
  send: SendFn,
  active: boolean
): {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  hiddenCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  captureWorksheet: () => void;
} {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let interval: ReturnType<typeof setInterval> | null = null;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        if (!active) return;
        interval = setInterval(() => {
          const video = videoRef.current;
          const canvas = hiddenCanvasRef.current;
          if (!video || !canvas || video.readyState < 2) return;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
          const image = dataUrl.replace("data:image/jpeg;base64,", "");
          send({ type: "FACE_FRAME", image, t: Date.now() / 1000 });
        }, 2000);
      } catch {
        console.warn("Camera unavailable — face frames disabled");
      }
    }

    start();

    return () => {
      if (interval) clearInterval(interval);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [send, active]);

  const captureWorksheet = useCallback(() => {
    const video = videoRef.current;
    const canvas = hiddenCanvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
    const image = dataUrl.replace("data:image/jpeg;base64,", "");
    send({ type: "WORKSHEET_PHOTO", image });
  }, [send]);

  return { videoRef, hiddenCanvasRef, captureWorksheet };
}
```

**Step 2: Type-check**

```bash
npm run build
```
Expected: exits 0.

---

## Task 6: Add backend-mode CSS

**Files:**
- Modify: `src/App.css`

**Step 1: Append new styles at end of `src/App.css`**

```css
/* ── Backend mode ── */

.backend-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.phase-badge {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  padding: 4px 10px;
  border-radius: 20px;
  background: #e2e8f0;
  color: #2d3748;
  text-transform: uppercase;
}

.phase-badge.phase-agent-led   { background: #bee3f8; color: #2b6cb0; }
.phase-badge.phase-collaborative { background: #c6f6d5; color: #276749; }
.phase-badge.phase-child-led   { background: #fefcbf; color: #744210; }

.ws-status {
  font-size: 0.8rem;
  color: #718096;
}

.capture-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 24px 0;
}

.webcam-preview-large {
  width: 320px;
  height: 240px;
  border-radius: 8px;
  border: 2px solid #9fd1ff;
  object-fit: cover;
  background: #000;
}

.capture-btn {
  font-size: 1.1rem;
  padding: 12px 32px;
  border-radius: 12px;
  background: linear-gradient(135deg, #3182ce, #2b6cb0);
  color: white;
  border: none;
  cursor: pointer;
  font-weight: 700;
}

.capture-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.mic-status {
  text-align: center;
  font-size: 1rem;
  color: #4a5568;
  padding: 12px;
  animation: monster-bob 1.4s ease-in-out infinite;
}

.session-summary {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 48px 24px;
  text-align: center;
}

.session-summary h2 {
  font-size: 2rem;
  color: #2d3748;
}

.session-summary p {
  font-size: 1.2rem;
  color: #4a5568;
}

.connecting-overlay {
  font-size: 0.9rem;
  color: #718096;
  text-align: center;
  padding: 8px;
}
```

**Step 2: Verify build**

```bash
npm run build
```
Expected: exits 0.

---

## Task 7: Modify `LessonPage.tsx`

**Files:**
- Modify: `src/LessonPage.tsx`

This is the main integration task. Replace the file completely with the version below. It preserves all existing hardcoded-flow state and logic unchanged — the new backend mode is an additional conditional render path.

**Step 1: Replace `src/LessonPage.tsx`**

```typescript
import React, { useMemo, useState, useRef, useCallback } from "react";
import Whiteboard from "./Whiteboard";
import WhiteboardCanvas, { type WhiteboardCanvasRef } from "./WhiteboardCanvas";
import { useWebSocket, type BackendAction } from "./hooks/useWebSocket";
import { useSpeech } from "./hooks/useSpeech";
import { useWebcam } from "./hooks/useWebcam";

type Question = {
  label: string;
  dividend: number;
  divisor: number;
  expectedQuotient: number;
};

type PlacementStep = "none" | "outside" | "inside";
type Phase = "AGENT_LED" | "COLLABORATIVE" | "CHILD_LED";

const questions: Question[] = [
  { label: "144 / 12", dividend: 144, divisor: 12, expectedQuotient: 12 },
  { label: "987 / 3", dividend: 987, divisor: 3, expectedQuotient: 329 },
  { label: "105 / 7", dividend: 105, divisor: 7, expectedQuotient: 15 },
  { label: "936 / 8", dividend: 936, divisor: 8, expectedQuotient: 117 },
];

function extractFirstInt(text: string): number | null {
  const m = text.match(/-?\d+/);
  return m ? Number(m[0]) : null;
}

const LessonPage: React.FC = () => {
  // ── Existing fallback state ──────────────────────────────────────────────
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [chatInput, setChatInput] = useState("");
  const [showWhiteboard, setShowWhiteboard] = useState(false);
  const [setupUnlocked, setSetupUnlocked] = useState(false);
  const [placementStep, setPlacementStep] = useState<PlacementStep>("none");
  const [outsideValue, setOutsideValue] = useState<string | null>(null);
  const [insideValue, setInsideValue] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<string>("");
  const [hasCorrectAnswer, setHasCorrectAnswer] = useState(false);
  const [requiresSetup, setRequiresSetup] = useState(false);

  // ── Backend mode state ───────────────────────────────────────────────────
  const [phase, setPhase] = useState<Phase>("AGENT_LED");
  const [sessionActive, setSessionActive] = useState(false);
  const [sessionDone, setSessionDone] = useState(false);
  const [summary, setSummary] = useState<{
    problems_done: number;
    confidence_end: number;
  } | null>(null);

  const whiteboardRef = useRef<WhiteboardCanvasRef>(null);
  const speakRef = useRef<(text: string) => void>(() => {});

  const handleAction = useCallback((action: BackendAction) => {
    switch (action.type) {
      case "SPEAK":
        speakRef.current(action.text as string);
        break;
      case "SHIFT_CONTROL":
        setPhase(action.to as Phase);
        break;
      case "SHOW_SUMMARY":
        setSessionDone(true);
        setSummary({
          problems_done: action.problems_done as number,
          confidence_end: action.confidence_end as number,
        });
        break;
      default:
        if (action.type.startsWith("DRAW_")) {
          whiteboardRef.current?.execute(action);
        }
    }
  }, []);

  const wsUrl = (import.meta.env.VITE_WS_URL as string | undefined) ?? "";
  const { send, wsStatus } = useWebSocket(wsUrl, handleAction);
  const { speak } = useSpeech(send, sessionActive && !sessionDone);
  speakRef.current = speak;
  const { videoRef, hiddenCanvasRef, captureWorksheet } = useWebcam(
    send,
    sessionActive && !sessionDone
  );

  const handleCapture = () => {
    captureWorksheet();
    setSessionActive(true);
  };

  // ── Fallback (existing) logic ─────────────────────────────────────────────
  const question = questions[currentQuestion];
  const dividendStr = useMemo(() => String(question.dividend), [question.dividend]);
  const divisorStr = useMemo(() => String(question.divisor), [question.divisor]);
  const canGoNext = hasCorrectAnswer && (!requiresSetup || setupUnlocked);
  const isLastQuestion = currentQuestion === questions.length - 1;

  const goToQuestion = (idx: number) => {
    setCurrentQuestion(idx);
    setChatInput("");
    setShowWhiteboard(false);
    setSetupUnlocked(false);
    setPlacementStep("none");
    setOutsideValue(null);
    setInsideValue(null);
    setFeedback("");
    setHasCorrectAnswer(false);
    setRequiresSetup(false);
  };

  const sendChat = () => {
    if (!chatInput.trim()) return;
    const msg = chatInput.trim();
    const typedNumber = extractFirstInt(msg);
    if (typedNumber === null) {
      setFeedback("Not quite. Please enter a number.");
      setChatInput("");
      return;
    }
    if (showWhiteboard && !setupUnlocked && requiresSetup) {
      if (placementStep === "outside") {
        if (typedNumber === question.divisor) {
          setOutsideValue(String(typedNumber));
          setPlacementStep("inside");
          setFeedback("Good. Now type the inside number (dividend) in the chat box.");
        } else {
          setFeedback("Not quite. Enter the divisor for the outside spot.");
        }
        setChatInput("");
        return;
      }
      if (placementStep === "inside") {
        if (typedNumber === question.dividend) {
          setInsideValue(String(typedNumber));
          setSetupUnlocked(true);
          setHasCorrectAnswer(true);
          setFeedback("Great. Setup is correct.");
        } else {
          setFeedback("Not quite. Enter the dividend for the inside spot.");
        }
        setChatInput("");
        return;
      }
    }
    if (typedNumber === question.expectedQuotient) {
      setFeedback("Correct.");
      setHasCorrectAnswer(true);
      setRequiresSetup(false);
      setShowWhiteboard(false);
      setSetupUnlocked(false);
      setPlacementStep("none");
      setOutsideValue(null);
      setInsideValue(null);
      setChatInput("");
      return;
    }
    setFeedback("Not quite. Whiteboard is up. Type the outside number in the chat box.");
    setHasCorrectAnswer(false);
    setRequiresSetup(true);
    setShowWhiteboard(true);
    setSetupUnlocked(false);
    setPlacementStep("outside");
    setOutsideValue(null);
    setInsideValue(null);
    setChatInput("");
  };

  const activeTarget =
    !setupUnlocked
      ? placementStep === "outside" || placementStep === "inside"
        ? placementStep
        : null
      : null;

  // ── Backend mode render ───────────────────────────────────────────────────
  const backendEnabled = wsUrl !== "" && wsStatus !== "disconnected";

  if (backendEnabled) {
    const phaseCss = phase.toLowerCase().replace(/_/g, "-");
    return (
      <div className="lesson-page">
        {/* Hidden elements always in DOM so webcam keeps running */}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={sessionActive ? { display: "none" } : undefined}
          className={!sessionActive ? "webcam-preview-large" : ""}
        />
        <canvas
          ref={hiddenCanvasRef}
          width={320}
          height={240}
          style={{ display: "none" }}
        />

        <div className="lesson-content">
          <div className="backend-header">
            <span className={`phase-badge phase-${phaseCss}`}>
              {phase.replace(/_/g, " ")}
            </span>
            {wsStatus === "connecting" && (
              <span className="ws-status">Connecting…</span>
            )}
          </div>

          {sessionDone && summary ? (
            <div className="session-summary">
              <h2>Session Complete!</h2>
              <p>{summary.problems_done} problem{summary.problems_done !== 1 ? "s" : ""} done</p>
              <p>Confidence: {Math.round(summary.confidence_end * 100)}%</p>
            </div>
          ) : !sessionActive ? (
            <div className="capture-area">
              <button
                className="capture-btn"
                onClick={handleCapture}
                disabled={wsStatus !== "connected"}
              >
                📸 Capture Worksheet
              </button>
              {wsStatus === "connecting" && (
                <p className="connecting-overlay">Waiting for backend…</p>
              )}
            </div>
          ) : (
            <>
              <WhiteboardCanvas ref={whiteboardRef} />
              <div className="mic-status">🎤 Listening for your answer…</div>
            </>
          )}
        </div>
      </div>
    );
  }

  // ── Fallback render (existing, unchanged) ────────────────────────────────
  return (
    <div className="lesson-page">
      <div className="lesson-content">
        <h1>Long Division Practice</h1>

        <div className="problem-box-row">
          <div className="monster-icon" aria-label="monster guide" role="img">
            <div className="monster-eye monster-eye-left" />
            <div className="monster-eye monster-eye-right" />
            <div className="monster-mouth" />
          </div>

          <div className="worksheet">
            <div className="worksheet-question">
              <label>{`${currentQuestion + 1}. Solve: ${question.label}`}</label>
            </div>
            {feedback && <div className="feedback">{feedback}</div>}
          </div>
        </div>

        {showWhiteboard && (
          <Whiteboard
            dividend={dividendStr}
            divisor={divisorStr}
            outsideValue={outsideValue}
            insideValue={insideValue}
            activeTarget={activeTarget}
            setupUnlocked={setupUnlocked}
          />
        )}

        <div className="navigation">
          <button
            onClick={() =>
              goToQuestion(Math.min(currentQuestion + 1, questions.length - 1))
            }
            disabled={!canGoNext || isLastQuestion}
          >
            {isLastQuestion ? "All Problems Completed" : "Next Problem"}
          </button>
        </div>
      </div>

      <div className="chat-area chat-area-fixed">
        <div className="chat-entry-row">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendChat()}
            placeholder=""
          />
          <div
            className={`chat-monster ${chatInput.trim().length > 0 ? "is-talking" : ""}`}
            aria-label="chat helper"
            role="img"
          >
            <div className="chat-monster-eye chat-monster-eye-left" />
            <div className="chat-monster-eye chat-monster-eye-right" />
            <div className="chat-monster-mouth" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default LessonPage;
```

**Step 2: Build**

```bash
npm run build
```
Expected: exits 0, no TypeScript errors.

**Step 3: Manual smoke test (dev server)**

```bash
npm run dev
```

**Without `.env.local` VITE_WS_URL:** Navigate to `/lesson`. Should show existing text Q&A fallback.

**With `.env.local` VITE_WS_URL set:** Navigate to `/lesson`. Should show "AGENT LED" phase badge + "Capture Worksheet" button (disabled while connecting, enabled once WebSocket connects). Click the button — browser requests camera permission — backend receives `WORKSHEET_PHOTO`.

---

## Verification Checklist

- [ ] `npm run build` exits 0 with no TypeScript errors
- [ ] `/lesson` with no env var shows existing fallback Q&A
- [ ] `/lesson` with `VITE_WS_URL` set shows phase badge + capture button
- [ ] Capture button enables when WS status = connected
- [ ] After capture, WhiteboardCanvas renders; backend `DRAW_*` commands draw on canvas
- [ ] Browser speaks TTS when `SPEAK` actions arrive
- [ ] Phase badge updates on `SHIFT_CONTROL`
- [ ] Session summary screen shows on `SHOW_SUMMARY`
- [ ] If backend goes down and retries exhaust, page falls back to Q&A mode
