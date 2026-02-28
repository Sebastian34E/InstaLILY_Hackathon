# Deploy & Frontend Contract — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create three shell scripts that deploy the backend to the RTX 6000 and run fine-tuning, plus a complete frontend contract document for the teammate.

**Architecture:** `deploy.sh` (run locally) copies code + sets up env. `train.sh` (run on VM) runs both LoRA fine-tuning jobs sequentially. `start_server.sh` (run on VM) starts uvicorn. `FRONTEND_CONTRACT.md` is the teammate's complete spec — all WebSocket message shapes, TypeScript types, canvas coordinate system, and a connection hook stub.

**Tech Stack:** bash, scp/ssh, Python venv, uvicorn, ngrok, TypeScript

---

## Task 1: Create `backend/deploy.sh`

**Files:**
- Create: `backend/deploy.sh`

No tests for shell scripts. Verify by doing a dry run with `bash -n` then running against VM.

**Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Run from your local machine: bash backend/deploy.sh
# Deploys backend code to RTX 6000 and sets up the environment.
set -euo pipefail

VM="hackathon@34.133.228.240"
REMOTE_DIR="~/math-tutor"
LOCAL_BACKEND="$(cd "$(dirname "$0")" && pwd)"

echo "=== Step 1: Checking VM state ==="
ssh "$VM" "
  echo '--- HuggingFace cache ---'
  ls ~/.cache/huggingface/hub 2>/dev/null || echo '(empty or missing)'
  echo '--- math-tutor dir ---'
  ls $REMOTE_DIR 2>/dev/null || echo '(does not exist yet)'
"

echo ""
echo "=== Step 2: Copying backend code to VM ==="
ssh "$VM" "mkdir -p $REMOTE_DIR/logs $REMOTE_DIR/adapters"
scp -r "$LOCAL_BACKEND"/ "$VM:$REMOTE_DIR/"
echo "Code copied."

echo ""
echo "=== Step 3: Setting up Python environment ==="
ssh "$VM" "
  cd $REMOTE_DIR
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt
  echo 'Dependencies installed.'
"

echo ""
echo "=== Step 4: Generating training data ==="
ssh "$VM" "
  cd $REMOTE_DIR
  source .venv/bin/activate
  python finetune/generate_data.py
  echo 'Training data generated.'
"

echo ""
echo "=== Done! ==="
echo ""
echo "Next steps — SSH into VM and run:"
echo "  ssh $VM"
echo "  cd ~/math-tutor"
echo "  bash train.sh          # ~60 min, run in tmux or screen"
echo "  bash start_server.sh   # after training completes"
echo ""
echo "To watch training progress:"
echo "  tail -f ~/math-tutor/logs/train_fg.log"
echo "  tail -f ~/math-tutor/logs/train_12b.log"
```

**Step 2: Make it executable**

```bash
chmod +x backend/deploy.sh
```

**Step 3: Dry-run syntax check**

```bash
bash -n backend/deploy.sh
```
Expected: no output, exit code 0.

---

## Task 2: Create `backend/train.sh`

**Files:**
- Create: `backend/train.sh`

This script runs on the VM. It trains FunctionGemma first (~20 min), then Gemma 12B (~40 min). Run it inside `tmux` or `screen` so it survives SSH disconnection.

**Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Run ON THE VM inside tmux/screen:
#   tmux new -s train
#   cd ~/math-tutor && bash train.sh
set -euo pipefail

source .venv/bin/activate

echo "=== [1/2] Training FunctionGemma 270M (~20 min) ==="
python finetune/train_function_gemma.py 2>&1 | tee logs/train_fg.log
echo "FunctionGemma training complete. Adapter saved to adapters/function_gemma/"

echo ""
echo "=== [2/2] Training Gemma 3 12B (~40 min) ==="
python finetune/train_gemma12b.py 2>&1 | tee logs/train_12b.log
echo "Gemma 12B training complete. Adapter saved to adapters/gemma12b/"

echo ""
echo "=== Training complete! Run: bash start_server.sh ==="
```

**Step 2: Make it executable**

```bash
chmod +x backend/train.sh
```

**Step 3: Syntax check**

```bash
bash -n backend/train.sh
```
Expected: no output, exit code 0.

---

## Task 3: Create `backend/start_server.sh`

**Files:**
- Create: `backend/start_server.sh`

Starts uvicorn from the correct directory so relative adapter paths resolve. Reminds user to start ngrok in a second terminal.

**Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Run ON THE VM after training completes:
#   cd ~/math-tutor && bash start_server.sh
set -euo pipefail

source .venv/bin/activate

echo "=== Checking adapters are present ==="
if [ ! -d "adapters/function_gemma" ]; then
  echo "ERROR: adapters/function_gemma/ not found. Run train.sh first."
  exit 1
fi
if [ ! -d "adapters/gemma12b" ]; then
  echo "ERROR: adapters/gemma12b/ not found. Run train.sh first."
  exit 1
fi
echo "Adapters found."

echo ""
echo "=== Starting Adaptive Math Tutor backend ==="
echo ""
echo "IMPORTANT: In a SECOND terminal, run:"
echo "  ngrok http 8000"
echo "Then give your teammate the wss://<ngrok-id>.ngrok.io/ws URL."
echo ""
echo "Starting uvicorn on port 8000..."

uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
```

**Step 2: Make it executable**

```bash
chmod +x backend/start_server.sh
```

**Step 3: Syntax check**

```bash
bash -n backend/start_server.sh
```
Expected: no output, exit code 0.

---

## Task 4: Create `docs/FRONTEND_CONTRACT.md`

**Files:**
- Create: `docs/FRONTEND_CONTRACT.md`

Complete spec for the teammate. No code changes to backend needed — this is documentation only.

**Step 1: Write the document**

````markdown
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

**When to send:** On camera capture. Send once per problem session. Backend will respond
with drawing commands and `SPEAK` actions.

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

**When to send:** `setInterval` every 2000ms. Draw webcam frame to a hidden canvas,
call `canvas.toDataURL("image/jpeg", 0.7)`, strip the `data:image/jpeg;base64,` prefix.

---

## Messages: Backend → Frontend

Handle each `type` in a switch/dispatch. Messages arrive as JSON strings.

### Drawing commands

All drawing commands specify **where** to draw, not pixel coordinates. See the
**Canvas Coordinate System** section below for position → pixel mapping.

#### `DRAW_PROBLEM`
```typescript
type DrawProblemAction = {
  type: "DRAW_PROBLEM";
  dividend: number;  // e.g. 247
  divisor: number;   // e.g. 6
};
```
Clear the canvas. Draw the long division bracket: `6 ) 247` with a horizontal bar
above the dividend for the quotient row.

#### `DRAW_CIRCLE`
```typescript
type DrawCircleAction = {
  type: "DRAW_CIRCLE";
  target: "first_digit" | "first_two_digits" | "first_three_digits";
};
```
Draw a coloured circle around the specified leading digits of the dividend.

#### `DRAW_NUMBER`
```typescript
type DrawNumberAction = {
  type: "DRAW_NUMBER";
  value: number;
  position: string; // see Canvas Coordinate System
};
```
Write `value` at the cell identified by `position`.

#### `DRAW_MULTIPLY`
```typescript
type DrawMultiplyAction = {
  type: "DRAW_MULTIPLY";
  value: number;       // the product, e.g. 24
  position: string;    // e.g. "subtract_row_0"
};
```
Write `value` at the subtract row position (right-aligned under the working digits).

#### `DRAW_LINE`
```typescript
type DrawLineAction = {
  type: "DRAW_LINE";
  position: string; // e.g. "subtract_line_0"
};
```
Draw a horizontal line at the specified subtract-line row.

#### `DRAW_BRING_DOWN`
```typescript
type DrawBringDownAction = {
  type: "DRAW_BRING_DOWN";
  digit_index: number; // 0-based index of the dividend digit being brought down
};
```
Draw a curved arrow bringing `dividend[digit_index]` down to the current working area.

#### `DRAW_REMAINDER`
```typescript
type DrawRemainderAction = {
  type: "DRAW_REMAINDER";
  value: number; // e.g. 1
};
```
Write `R {value}` at the bottom-right of the working area.

#### `DRAW_HINT`
```typescript
type DrawHintAction = {
  type: "DRAW_HINT";
  hint_type: "show_multiplication_table";
  a: number;
  b: number;
};
```
Display a helper overlay: for `show_multiplication_table`, show `a × 1 … a × 9` in a
small table beside the whiteboard.

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

#### `SHOW_SUMMARY`
```typescript
type ShowSummaryAction = {
  type: "SHOW_SUMMARY";
  problems_done: number;
  confidence_end: number; // 0.0 – 1.0
};
```
Session is complete. Show end-screen: problems reviewed + confidence score.

---

## Canvas Coordinate System

Assume canvas is **600 × 480 px**. The long division layout is a fixed grid:

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

| Position string | Row | Right-aligned to col |
|---|---|---|
| `quotient_0` | 40 | 200 |
| `quotient_1` | 40 | 250 |
| `quotient_2` | 40 | 300 |
| `subtract_row_0` | 130 | 300 |
| `subtract_line_0` | 150 | (full width of working area) |
| `subtract_row_1` | 220 | 300 |
| `subtract_line_1` | 240 | (full width of working area) |
| `remainder` | 310 | 350 |

`first_digit` circle target = dividend digit at col 200, row 90.
`first_two_digits` = cols 200–250, row 90.
`first_three_digits` = cols 200–300, row 90.

Font: `bold 28px monospace`. Colour: `#1a1a1a`. Lines: `2px solid #1a1a1a`.
Circles/arrows: `2px solid #e53e3e` (red).

---

## Message Sequencing

1. `SPEAK` always arrives before drawing commands for the same step.
2. Multiple drawing commands may arrive in a single burst — execute them in order.
3. `SHIFT_CONTROL` is a standalone message (no drawing commands follow it immediately).
4. `SHOW_SUMMARY` is always the final message of a session.
5. After `SHOW_SUMMARY`, stop sending `FACE_FRAME` events.

---

## Minimal React Hook

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
    case "SPEAK":        handleSpeak(action as SpeakAction);       break;
    case "DRAW_PROBLEM": handleDrawProblem(action as DrawProblemAction); break;
    case "DRAW_NUMBER":  handleDrawNumber(action as DrawNumberAction);   break;
    case "SHIFT_CONTROL": setPhase((action as ShiftControlAction).to);  break;
    case "SHOW_SUMMARY": showSummary(action as ShowSummaryAction);  break;
    // ... etc
  }
});

// Send worksheet photo:
send({ type: "WORKSHEET_PHOTO", image: base64Jpeg });

// Send speech:
send({ type: "STUDENT_SPEECH", text: transcript, t: Date.now() / 1000 });
```
````

---

## Verification checklist

After all tasks are complete, verify:

- [ ] `bash -n backend/deploy.sh` passes
- [ ] `bash -n backend/train.sh` passes
- [ ] `bash -n backend/start_server.sh` passes
- [ ] `docs/FRONTEND_CONTRACT.md` exists and covers all 3 inbound + 10 outbound message types
- [ ] All TypeScript types in contract match the Pydantic models in `backend/app/models.py`
