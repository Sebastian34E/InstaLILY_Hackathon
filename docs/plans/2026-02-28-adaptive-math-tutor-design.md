# Adaptive Math Tutor — System Design

**Date:** 2026-02-28
**Status:** Approved

---

## Problem Statement

Children in rural and low-income communities get stuck on homework alone. Tutors cost $50–100/hour. Parents aren't always available to help. When kids get stuck they either give up or copy answers from Google — neither builds understanding.

A cloud-based AI tutor is not the answer. A camera watching a child's face, capturing their handwriting, and recording their voice is sensitive personal data about a minor. COPPA (US), GDPR-K (EU), and equivalent laws in most countries make cloud architectures legally and ethically undeployable in schools without complex consent flows and data agreements. A school IT department won't touch it.

**On-device is not a performance optimisation — it's the only architecture that's legally viable.** Zero data leaves the device. Works offline. Deployable in rural schools with no internet.

---

## What We're Building

An on-device adaptive math tutoring agent. The student photos their printed worksheet. The agent reads it, identifies wrong answers, then opens a virtual whiteboard and teaches them through each mistake — drawing the problem step by step, asking Socratic questions, and progressively handing control to the student as they gain confidence.

---

## Hackathon Requirements Mapping

| Requirement | How We Meet It |
|---|---|
| Fine-tuned on-device Gemma model | Gemma 3 12B (tutoring dialogue + drawing commands) + FunctionGemma 270M (response classifier) |
| Agentic behaviour | Autonomous error detection, adaptive control transfer, decides when/how to intervene |
| Visual input | Worksheet photo (OCR), webcam face feed (engagement detection) |
| Genuine on-device reason | COPPA/GDPR-K: child face + voice + handwriting cannot leave device |
| Voice bonus | Web Speech API STT/TTS — student speaks naturally, agent responds aloud |

---

## Stack

| Component | Technology |
|---|---|
| GPU machine | RTX 6000 Ada (48GB VRAM), SSH access |
| Backend | Python 3.11, FastAPI, Uvicorn, WebSocket |
| Vision/dialogue model | Gemma 3 12B + LoRA fine-tune |
| Response classifier | FunctionGemma 270M + LoRA fine-tune |
| Model serving | HuggingFace Transformers + PEFT |
| Tunnel | ngrok |
| Frontend | React + TypeScript + Vite |
| Voice in | Web Speech API (SpeechRecognition) |
| Voice out | Web Speech API (SpeechSynthesis) |
| Whiteboard | HTML5 Canvas — executes drawing commands from backend |

---

## Architecture

```
RTX 6000 Machine (SSH)
├── FastAPI app (port 8000)
│   ├── /ws  ← WebSocket (all comms)
│   ├── /health
│   └── /reset
├── ModelServer
│   ├── Gemma 3 12B + LoRA  [reads worksheet, generates dialogue + drawing commands]
│   └── FunctionGemma 270M + LoRA  [classifies response quality, decides phase shifts]
└── ngrok tunnel

Student's Device (Local)
└── React/TS Vite
    ├── Webcam feed (face engagement frames → backend every 2s)
    ├── Camera input (worksheet photo → backend on submission)
    ├── Web Speech API STT (student voice → text → backend)
    ├── Web Speech API TTS (agent text → spoken aloud)
    └── Canvas whiteboard (renders drawing commands from backend)
```

---

## WebSocket Interface Contract

### Events → backend (frontend sends)

```json
{ "type": "WORKSHEET_PHOTO", "image": "<base64 JPEG>", "session_id": "abc123" }
{ "type": "STUDENT_SPEECH", "text": "bring down the 7", "t": 42.1 }
{ "type": "FACE_FRAME", "image": "<base64 JPEG>", "t": 42.1 }
{ "type": "PHASE_ACK", "phase": "COLLABORATIVE" }
```

### Actions → frontend (backend sends)

```json
{ "type": "DRAW_PROBLEM", "dividend": 247, "divisor": 6 }
{ "type": "DRAW_CIRCLE", "target": "first_two_digits" }
{ "type": "DRAW_NUMBER", "value": 4, "position": "quotient_0" }
{ "type": "DRAW_MULTIPLY", "value": 24, "position": "subtract_row_0" }
{ "type": "DRAW_LINE", "position": "subtract_line_0" }
{ "type": "DRAW_BRING_DOWN", "digit_index": 2 }
{ "type": "DRAW_REMAINDER", "value": 1 }
{ "type": "SPEAK", "text": "Exactly right! So we write 4 above the line." }
{ "type": "SHIFT_CONTROL", "to": "COLLABORATIVE" }
{ "type": "SHOW_SUMMARY", "problems_done": 3, "confidence_end": 0.85 }
```

---

## Agent State Machine

```
READING_WORKSHEET
  └─ WORKSHEET_PHOTO received
     → Gemma 3 12B reads all problems + student answers
     → identifies wrong answer(s)
     → PROBLEM_SELECTED

PROBLEM_SELECTED
  └─ Agent draws DRAW_PROBLEM on whiteboard
     → announces topic via SPEAK
     → AGENT_LED

AGENT_LED  [agent draws each step, asks Socratic question, waits]
  ├─ STUDENT_SPEECH received
  │   ├─ FunctionGemma: "confident_correct" × 2  → SHIFT_CONTROL to COLLABORATIVE
  │   ├─ FunctionGemma: "fundamentally_wrong"    → re-explain step, stay AGENT_LED
  │   └─ FunctionGemma: "hesitant_correct"       → praise, stay AGENT_LED one more
  ├─ FACE_FRAME: frustration detected            → slow down, stay AGENT_LED
  └─ timeout (no speech 15s)                     → gentle prompt, stay AGENT_LED

COLLABORATIVE  [child says the step, agent draws what they say]
  ├─ STUDENT_SPEECH received
  │   ├─ FunctionGemma: "confident_correct" × 3  → SHIFT_CONTROL to CHILD_LED
  │   ├─ FunctionGemma: "wrong"                  → gentle correction, back to AGENT_LED
  │   └─ FunctionGemma: "hesitant_correct"       → stay COLLABORATIVE
  └─ FACE_FRAME: frustration detected            → back to AGENT_LED

CHILD_LED  [child fully directs, agent validates and draws]
  ├─ STUDENT_SPEECH received
  │   ├─ correct → validate, draw, continue
  │   └─ wrong   → back to COLLABORATIVE
  └─ problem complete → UNDERSTANDING_CONFIRMED

UNDERSTANDING_CONFIRMED
  ├─ more wrong problems on worksheet → PROBLEM_SELECTED
  └─ all done → SESSION_COMPLETE → SHOW_SUMMARY
```

**The visible agentic moment:** the agent autonomously decides to say "OK, you tell me what comes next" and stops drawing unprompted. Then pulls back if the student struggles. That decision loop — running continuously, watching responses and face — is the agentic behaviour.

---

## Fine-Tuning Plan

### FunctionGemma 270M — Response quality classifier (~20 min on RTX 6000)

~60 synthetic examples of `(student_response + context) → classification + phase action`:

```json
{"input": {"response": "4 times", "step": "how many times does 6 go into 24", "phase": "AGENT_LED", "correct_streak": 2},
 "output": "classify(confident_correct) | shift_control(COLLABORATIVE)"}

{"input": {"response": "um... 3?", "step": "how many times does 6 go into 24", "phase": "COLLABORATIVE"},
 "output": "classify(hesitant_wrong) | stay(COLLABORATIVE) | draw_hint(show_6x3_vs_6x4)"}

{"input": {"response": "bring down the 7", "step": "next step", "phase": "CHILD_LED", "correct_streak": 3},
 "output": "classify(confident_correct) | validate() | draw(bring_down_7)"}
```

### Gemma 3 12B — Math tutoring + drawing commands (~40 min on RTX 6000)

~150 examples of Socratic long division tutoring. Each includes drawing commands alongside speech:

```json
{
  "context": "child just said '6 goes into 24 four times'",
  "agent_speech": "Exactly right! So we write 4 above the line.",
  "drawing_commands": [
    {"type": "DRAW_NUMBER", "value": 4, "position": "quotient_0"},
    {"type": "DRAW_MULTIPLY", "value": 24, "position": "subtract_row_0"},
    {"type": "DRAW_LINE", "position": "subtract_line_0"}
  ]
}
```

Also fine-tuned on worksheet photo reading:
```json
{
  "image_description": "handwritten long division: 247 ÷ 6 = 40 r7",
  "extracted": {"dividend": 247, "divisor": 6, "student_answer": "40 remainder 7", "correct": false, "error_type": "incomplete_quotient"}
}
```

---

## Frontend UX

```
┌─────────────────────────────────────────────────┐
│  [📷 face cam - small corner]    Phase: AGENT   │
│                                                 │
│  ┌──────────── WHITEBOARD ──────────────────┐   │
│  │                                          │   │
│  │         6 ) 2 4 7                        │   │
│  │           ──────                         │   │
│  │             4                            │   │
│  │            24                            │   │
│  │            ──                            │   │
│  │             0 7  ← drawing now           │   │
│  └──────────────────────────────────────────┘   │
│                                                 │
│  🔊 "Now you tell me — what do we do with       │
│      the remainder 7?"                          │
│                                                 │
│  [🎤 Hold to speak]   [📷 New worksheet]        │
└─────────────────────────────────────────────────┘
```

Student flow:
1. Taps "New worksheet" → photos printed worksheet
2. Agent reads it, selects first wrong problem
3. Whiteboard appears, agent starts drawing and speaking
4. Student holds mic button and responds naturally
5. Phase indicator (AGENT → COLLAB → CHILD) shifts as confidence grows
6. Session summary at end: problems reviewed, concepts mastered

---

## Demo Script

1. Child photos a worksheet with `247 ÷ 6 = 40 r7` (wrong — should be 41 r1)
2. Agent: *"I can see you got 40 remainder 7 for this one. Let's work through it together."*
3. Whiteboard: draws `6 ) 247` bracket
4. Agent: *"First — can 6 go into just the 2?"*
5. Child: *"No, it's too small"*
6. Agent draws circle around 24, *"Right — so we look at 24. How many times does 6 go into 24?"*
7. Child: *"4 times"* → FunctionGemma: confident_correct
8. Agent draws 4, draws 24, draws subtract line → *"Perfect. What's 24 minus 24?"*
9. Child: *"Zero"* → confident_correct streak = 2 → **SHIFT to COLLABORATIVE**
10. Agent: *"Great work. Now you're going to tell me the next step."*
11. Child: *"Bring down the 7"* → Agent draws bring-down
12. Child: *"6 goes into 7 once"* → Agent draws 1 → **SHIFT to CHILD_LED**
13. Child completes: *"remainder 1"* → Agent: *"Exactly right — 41 remainder 1."*
14. Visible shift from agent drawing everything → child directing → child fully independent

---

## What Carries Over From Previous Backend

~70% of existing backend code transfers directly:
- FastAPI app structure + WebSocket endpoint → reuse with new events/actions
- ModelServer class (HuggingFace PEFT loader + fallback) → reuse, new prompts
- Fine-tuning pipeline structure → reuse, new training data
- pyproject.toml, requirements.txt → minor additions only

New code:
- New event/action models (WORKSHEET_PHOTO, STUDENT_SPEECH, DRAW_* actions)
- New SessionState fields (phase, confidence_streak, problems_queue)
- New PolicyEngine state machine (READING_WORKSHEET → AGENT_LED → COLLABORATIVE → CHILD_LED)
- New fine-tuning data (math tutoring dialogues + worksheet reading examples)
- Frontend Canvas whiteboard renderer
