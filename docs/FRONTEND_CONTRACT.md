
# Frontend ↔ Backend Contract

**WebSocket endpoint:** `wss://<ngrok-id>.ngrok.io/ws`

Connect once on app load. Reconnect automatically on drop (exponential backoff, max 5 retries).

---

## Messages: Frontend → Backend

### 1. `WORKSHEET_PHOTO`

Send when student captures their worksheet.

```typescript
type WorksheetPhotoEvent = {
  type: "WORKSHEET_PHOTO";
  image: string; // base64-encoded JPEG (no data: prefix)
};
```

**When to send:** On camera capture. Send once per problem session. Backend will respond with drawing commands and `SPEAK` actions.

---

### 2. `STUDENT_SPEECH`

Send each time Web Speech API produces a final transcript.

```typescript
type StudentSpeechEvent = {
  type: "STUDENT_SPEECH";
  text: string;  // final transcript from SpeechRecognition
  t: number;     // Date.now() / 1000 — seconds since epoch
};
```

**When to send:** On `SpeechRecognition` `result` event where `isFinal === true`.

---

### 3. `FACE_FRAME`

Send a webcam snapshot every 2 seconds while a session is active.

```typescript
type FaceFrameEvent = {
  type: "FACE_FRAME";
  image: string; // base64-encoded JPEG from webcam canvas capture
  t: number;     // Date.now() / 1000
};
```

**When to send:** `setInterval` every 2000ms. Draw webcam frame to a hidden canvas, call `canvas.toDataURL("image/jpeg", 0.7)`, strip the `data:image/jpeg;base64,` prefix.

---

## Messages: Backend → Frontend

Handle each `type` in a switch/dispatch. Messages arrive as JSON strings — parse with `JSON.parse(event.data)`.

### Drawing commands

All drawing commands specify **where** to draw using named positions, not raw pixel coordinates. See the **Canvas Coordinate System** section for position → pixel mapping.

#### `DRAW_PROBLEM`
```typescript
type DrawProblemAction = {
  type: "DRAW_PROBLEM";
  dividend: number;  // e.g. 247
  divisor: number;   // e.g. 6
};
```
Clear the canvas. Draw the long division bracket: `6 ) 247` with a horizontal bar above the dividend for the quotient row.

---

#### `DRAW_CIRCLE`
```typescript
type DrawCircleAction = {
  type: "DRAW_CIRCLE";
  target: "first_digit" | "first_two_digits" | "first_three_digits";
};
```
Draw a coloured circle around the specified leading digits of the dividend.

---

#### `DRAW_NUMBER`
```typescript
type DrawNumberAction = {
  type: "DRAW_NUMBER";
  value: number;
  position: string; // see Canvas Coordinate System
};
```
Write `value` at the cell identified by `position`.

---

#### `DRAW_MULTIPLY`
```typescript
type DrawMultiplyAction = {
  type: "DRAW_MULTIPLY";
  value: number;    // the product, e.g. 24
  position: string; // e.g. "subtract_row_0"
};
```
Write `value` at the subtract row position, right-aligned under the working digits.

---

#### `DRAW_LINE`
```typescript
type DrawLineAction = {
  type: "DRAW_LINE";
  position: string; // e.g. "subtract_line_0"
};
```
Draw a horizontal line at the specified subtract-line row.

---

#### `DRAW_BRING_DOWN`
```typescript
type DrawBringDownAction = {
  type: "DRAW_BRING_DOWN";
  digit_index: number; // 0-based index of the dividend digit being brought down
};
```
Draw a curved arrow bringing `dividend[digit_index]` down to the current working area.

---

#### `DRAW_REMAINDER`
```typescript
type DrawRemainderAction = {
  type: "DRAW_REMAINDER";
  value: number; // e.g. 1
};
```
Write `R {value}` at the bottom-right of the working area.

---

#### `DRAW_HINT`
```typescript
type DrawHintAction = {
  type: "DRAW_HINT";
  hint_type: "show_multiplication_table";
  a: number;
  b: number;
};
```
Display a helper overlay. For `show_multiplication_table`, show `a × 1 … a × 9` in a small table beside the whiteboard.

---

### Speech

#### `SPEAK`
```typescript
type SpeakAction = {
  type: "SPEAK";
  text: string; // speak aloud via SpeechSynthesis
};
```
Call `window.speechSynthesis.speak(new SpeechSynthesisUtterance(text))`.

`SPEAK` always arrives **before** drawing commands for the same tutoring step.

---

### Control

#### `SHIFT_CONTROL`
```typescript
type ShiftControlAction = {
  type: "SHIFT_CONTROL";
  to: "AGENT_LED" | "COLLABORATIVE" | "CHILD_LED";
};
```
Update the phase badge/indicator in the UI. No canvas change needed.

Phases:
- `AGENT_LED` — tutor is teaching; student responding
- `COLLABORATIVE` — student says each step aloud; tutor draws
- `CHILD_LED` — student fully directing; tutor validates

---

#### `SHOW_SUMMARY`
```typescript
type ShowSummaryAction = {
  type: "SHOW_SUMMARY";
  problems_done: number;
  confidence_end: number; // 0.0 – 1.0
};
```
Session is complete. Show end-screen: problems reviewed + confidence score. Stop sending `FACE_FRAME` after this.

---

## Canvas Coordinate System

Assume canvas is **600 × 480 px**. Long division layout uses a fixed grid:

```
  col:   50   150  200  250  300  350
         |    |    |    |    |    |

row 40:        [q0] [q1] [q2]         ← quotient digits
row 60:       ──────────────────       ← quotient bar
row 90:  [d]  [  dividend digits  ]   ← divisor + dividend
row 130:       [subtract_row_0   ]    ← first subtract value
row 150:      ──────────────────       ← subtract_line_0
row 180:       [working area 0   ]    ← after first bring-down
row 220:       [subtract_row_1   ]
row 240:      ──────────────────
row 270:       [working area 1   ]
row 310:                    R [rem]   ← remainder
```

### Position string → grid cell

| Position string | Row (px) | Right-aligned to col (px) |
|---|---|---|
| `quotient_0` | 40 | 200 |
| `quotient_1` | 40 | 250 |
| `quotient_2` | 40 | 300 |
| `subtract_row_0` | 130 | 300 |
| `subtract_line_0` | 150 | full width of working area |
| `subtract_row_1` | 220 | 300 |
| `subtract_line_1` | 240 | full width of working area |
| `remainder` | 310 | 350 |

**Circle targets:**
- `first_digit` → dividend digit at col 200, row 90
- `first_two_digits` → cols 200–250, row 90
- `first_three_digits` → cols 200–300, row 90

**Styles:** Font `bold 28px monospace`, colour `#1a1a1a`. Lines `2px solid #1a1a1a`. Circles/arrows `2px solid #e53e3e`.

---

## Message Sequencing

1. `SPEAK` always arrives **before** drawing commands for the same tutoring step.
2. Multiple drawing commands may arrive in a single burst — execute them in order.
3. `SHIFT_CONTROL` is standalone — no drawing commands immediately follow it.
4. `SHOW_SUMMARY` is always the **final** message of a session.
5. Stop sending `FACE_FRAME` events after receiving `SHOW_SUMMARY`.

---

## Minimal `useWebSocket` Hook

```typescript
// src/hooks/useWebSocket.ts
import { useEffect, useRef, useCallback } from "react";

type BackendAction = { type: string; [key: string]: unknown };

export function useWebSocket(
  url: string,
  onAction: (action: BackendAction) => void
) {
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    let retries = 0;

    function connect() {
      ws.current = new WebSocket(url);

      ws.current.onmessage = (e) => {
        try {
          onAction(JSON.parse(e.data) as BackendAction);
        } catch {
          console.error("Bad message", e.data);
        }
      };

      ws.current.onclose = () => {
        if (retries < 5) {
          setTimeout(connect, Math.min(1000 * 2 ** retries, 10000));
          retries++;
        }
      };

      ws.current.onerror = console.error;
    }

    connect();
    return () => ws.current?.close();
  }, [url]); // eslint-disable-line react-hooks/exhaustive-deps

  const send = useCallback((event: object) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(event));
    }
  }, []);

  return { send };
}
```

**Usage:**
```typescript
const { send } = useWebSocket("wss://<ngrok-id>.ngrok.io/ws", (action) => {
  switch (action.type) {
    case "SPEAK":         speak((action as SpeakAction).text);              break;
    case "DRAW_PROBLEM":  drawProblem(action as DrawProblemAction);          break;
    case "DRAW_CIRCLE":   drawCircle(action as DrawCircleAction);            break;
    case "DRAW_NUMBER":   drawNumber(action as DrawNumberAction);            break;
    case "DRAW_MULTIPLY": drawMultiply(action as DrawMultiplyAction);        break;
    case "DRAW_LINE":     drawLine(action as DrawLineAction);                break;
    case "DRAW_BRING_DOWN": drawBringDown(action as DrawBringDownAction);    break;
    case "DRAW_REMAINDER": drawRemainder(action as DrawRemainderAction);     break;
    case "DRAW_HINT":     drawHint(action as DrawHintAction);                break;
    case "SHIFT_CONTROL": setPhase((action as ShiftControlAction).to);      break;
    case "SHOW_SUMMARY":  showSummary(action as ShowSummaryAction);          break;
  }
});

// Send worksheet photo:
send({ type: "WORKSHEET_PHOTO", image: base64Jpeg });

// Send student speech (from SpeechRecognition):
recognition.onresult = (e) => {
  const result = e.results[e.results.length - 1];
  if (result.isFinal) {
    send({ type: "STUDENT_SPEECH", text: result[0].transcript, t: Date.now() / 1000 });
  }
};

// Send face frames every 2 seconds:
const faceInterval = setInterval(() => {
  const ctx = hiddenCanvas.getContext("2d")!;
  ctx.drawImage(videoElement, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
  const dataUrl = hiddenCanvas.toDataURL("image/jpeg", 0.7);
  send({ type: "FACE_FRAME", image: dataUrl.split(",")[1], t: Date.now() / 1000 });
}, 2000);
```
