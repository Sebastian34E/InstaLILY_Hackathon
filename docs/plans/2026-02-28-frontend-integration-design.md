# Frontend Backend Integration — Design

**Date:** 2026-02-28
**Status:** Approved

---

## Goal

Wire the existing React frontend to the live FastAPI/WebSocket backend running on the RTX 6000 VM. Students will speak their answers, see the tutor draw on a canvas whiteboard, and hear tutoring audio — all driven by the backend AI models.

---

## Current State

The frontend has:
- Home page with lesson cards (keep as-is)
- LessonPage with hardcoded 4-question flow + div-based whiteboard
- Styled animated UI (monsters, animations, responsive CSS)
- No WebSocket, no speech, no camera, no canvas drawing

---

## Deliverables

| File | Action | Purpose |
|---|---|---|
| `src/hooks/useWebSocket.ts` | Create | WS connection, auto-reconnect, send helper |
| `src/hooks/useSpeech.ts` | Create | SpeechRecognition (mic) + SpeechSynthesis (TTS) |
| `src/hooks/useWebcam.ts` | Create | Webcam capture + periodic FACE_FRAME sending |
| `src/WhiteboardCanvas.tsx` | Create | HTML5 canvas executing all 8 DRAW_* commands |
| `src/LessonPage.tsx` | Modify | Integrate hooks; backend flow when connected, fallback when not |

---

## Architecture

```
LessonPage
├── useWebSocket(VITE_WS_URL)        → send / wsStatus / onAction dispatch
├── useWebcam(send, sessionActive)   → face frames every 2s, worksheet capture
├── useSpeech(send, sessionActive)   → mic → STUDENT_SPEECH, TTS for SPEAK
├── WhiteboardCanvas(ref)            → draws on canvas via imperative commands
└── Phase badge                      → AGENT_LED / COLLABORATIVE / CHILD_LED
```

---

## Section 1: useWebSocket

- Connects to `import.meta.env.VITE_WS_URL`
- Exponential backoff reconnect: up to 5 retries (1s, 2s, 4s, 8s, 10s max)
- After 5 failed retries → `wsStatus = 'disconnected'` (triggers fallback)
- Exposes: `{ send, wsStatus }`
- Calls `onAction(parsed)` for every incoming JSON message

---

## Section 2: WhiteboardCanvas

HTML5 `<canvas>` element, 600×480px, matching the contract coordinate grid.

Drawing commands handled (imperative, called from LessonPage):
- `DRAW_PROBLEM` — clear canvas, draw long division bracket `divisor ) dividend`
- `DRAW_CIRCLE` — red circle around first_digit / first_two_digits / first_three_digits
- `DRAW_NUMBER` — write value at named position (quotient_0..2, subtract_row_0..1)
- `DRAW_MULTIPLY` — write product right-aligned at subtract_row position
- `DRAW_LINE` — horizontal line at subtract_line_0 or subtract_line_1
- `DRAW_BRING_DOWN` — curved arrow from digit at index to working area
- `DRAW_REMAINDER` — "R {value}" at bottom right
- `DRAW_HINT` — show_multiplication_table: small overlay table beside whiteboard

Font: `bold 28px monospace`, colour `#1a1a1a`. Lines: `2px solid #1a1a1a`. Circles/arrows: `2px solid #e53e3e`.

Canvas is exposed to LessonPage via a ref; drawing commands are called imperatively (not via re-renders).

---

## Section 3: useSpeech

**Mic (SpeechRecognition):**
- Continuous, `interimResults: false`, `lang: 'en-US'`
- On final result → `send({ type: 'STUDENT_SPEECH', text, t: Date.now()/1000 })`
- Starts after worksheet is captured; stops after SHOW_SUMMARY

**TTS (SpeechSynthesis):**
- On SPEAK action: cancel any current utterance, speak new one immediately
- `window.speechSynthesis.speak(new SpeechSynthesisUtterance(text))`
- Stops after SHOW_SUMMARY

---

## Section 4: useWebcam

- `getUserMedia({ video: true })` on mount
- If permission denied: silently disabled (no crash)
- `setInterval(2000)` draws video frame to hidden canvas → base64 JPEG (quality 0.7) → strip data URL prefix → `send({ type: 'FACE_FRAME', image, t })`
- Exposes: `{ videoRef, captureWorksheet }` where `captureWorksheet()` grabs current frame → sends `WORKSHEET_PHOTO` and returns

---

## Section 5: LessonPage Integration

**State additions:**
```typescript
wsStatus: 'connecting' | 'connected' | 'disconnected'
phase: 'AGENT_LED' | 'COLLABORATIVE' | 'CHILD_LED'
sessionActive: boolean   // true after WORKSHEET_PHOTO sent
sessionDone: boolean     // true after SHOW_SUMMARY
summary: { problems_done: number; confidence_end: number } | null
```

**UI layout (backend mode):**
- Top-right: small webcam preview (120×90px) + phase badge
- Center: WhiteboardCanvas (600×480)
- Bottom: "Capture Worksheet" button (disappears after capture) + mic status indicator
- End screen on SHOW_SUMMARY

**Fallback (wsStatus === 'disconnected'):**
- Existing hardcoded 4-question text Q&A flow (unchanged from current LessonPage)

**Action dispatch:**
```
SPEAK          → useSpeech.speak(text)
DRAW_*         → whiteboardRef.current.draw(action)
SHIFT_CONTROL  → setPhase(action.to)
SHOW_SUMMARY   → setSessionDone(true), setSummary(...)
```

---

## Config

`.env.local` (not committed):
```
VITE_WS_URL=wss://0e33-34-133-228-240.ngrok-free.app/ws
```

---

## Out of Scope

- Lesson type selection routing (all cards → same lesson page, unchanged)
- Persistent session storage
- Multi-student support
