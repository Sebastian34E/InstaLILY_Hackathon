# Forklift Safety AI Trainer — System Design

**Date:** 2026-02-28
**Status:** Approved

---

## Goal

An on-device adaptive training system that watches a learner watch a forklift safety video. The AI sees both the training content and the learner simultaneously, detects distraction or skip attempts, and autonomously pauses the video to ask contextually relevant safety questions — then decides whether to resume, rewind, or repeat based on the answer.

## On-Device Justification

- Forklift certification footage is proprietary compliance material that cannot leave the facility
- Factory floors often have no internet connectivity — training cannot depend on cloud availability
- The pause-quiz-rewind loop requires sub-200ms latency; cloud API round-trips break the UX
- OSHA compliance records are sensitive; learner assessment data stays local

---

## Stack

| Component | Technology |
|---|---|
| GPU machine | RTX 6000 Ada (48GB VRAM), SSH access |
| Backend | Python 3.11, FastAPI, Uvicorn, WebSocket |
| Action model | FunctionGemma 270M + LoRA fine-tune |
| Vision/Q&A model | Gemma 3 12B + LoRA fine-tune |
| Model serving | HuggingFace Transformers + PEFT (load LoRA adapters directly) |
| Tunnel | ngrok (exposes RTX 6000 backend to local frontend) |
| Frontend | React + TypeScript + Vite (runs locally) |
| Fine-tuning | HuggingFace Transformers + PEFT, run on RTX 6000 |
| Training data | Synthetic (generated) + OSHA public guidelines |

---

## Architecture

```
RTX 6000 Machine (SSH)
├── FastAPI app (port 8000)
│   ├── /ws  ← WebSocket endpoint (all comms)
│   ├── /health
│   └── /reset
├── ModelServer
│   ├── FunctionGemma 270M + LoRA adapter  [~1GB VRAM]
│   └── Gemma 3 12B + LoRA adapter         [~24GB VRAM]
└── ngrok tunnel → wss://xxxx.ngrok.io

Local Machine
└── React/TS Vite (port 5173)
    ├── HTML5 <video> player
    ├── Webcam (MediaDevices.getUserMedia)
    ├── Canvas frame capture → base64 JPEG
    └── WebSocket client → ngrok → RTX 6000
```

---

## Visual Input Flow

Every 2 seconds, frontend captures and sends:

```json
{
  "type": "VISUAL_FRAME",
  "video_frame": "<base64 JPEG>",
  "webcam_frame": "<base64 JPEG>",
  "t": 42.1
}
```

Backend processes with Gemma 3 12B (multimodal):

**Prompt template:**
```
You are a forklift safety training assistant.
Left image: training video frame at {t}s.
Right image: learner's webcam feed.

Answer as JSON:
{
  "attentive": true/false,
  "confidence": 0.0-1.0,
  "video_topic": "one phrase describing what safety procedure is shown",
  "question": "one specific safety question about what is shown in the video"
}
```

**Output feeds into FunctionGemma:**
```json
{
  "attentive": false,
  "away_seconds": 7,
  "topic": "pre-operation checklist",
  "wrong_streak": 0,
  "t": 42.1
}
→ pause_video() | show_quiz(prompt="What must be checked before starting the forklift?")
```

---

## Fine-Tuning Plan

### FunctionGemma 270M — Action Selection
- ~60 synthetic training examples
- Format: `(context JSON) → function_call(args)`
- Examples cover: distraction triggers, skip triggers, wrong-answer escalation, correct-answer resume
- LoRA rank 8, ~20 minutes on RTX 6000

**Example training pair:**
```
Input:  {"attentive": false, "away_seconds": 8, "topic": "load capacity check", "wrong_streak": 0}
Output: pause_video() | show_quiz(prompt="What is the maximum load capacity for this forklift?")

Input:  {"seek_skip_seconds": 45, "topic": "pedestrian zone rules", "wrong_streak": 1}
Output: seek_relative(delta=-10) | pause_video() | show_quiz(prompt="What must you do when entering a pedestrian zone?")

Input:  {"attentive": true, "quiz_correct": true, "feedback": "Well done"}
Output: resume_video() | speak(text="Well done. Keep watching.")
```

### Gemma 3 12B — Forklift Q&A + Vision
- ~150 forklift safety Q&A pairs from OSHA guidelines + public training materials
- Format: `(question + context) → (answer + feedback)`
- Makes model OSHA-accurate and procedure-specific
- LoRA rank 16, ~40 minutes on RTX 6000

**Example training pair:**
```
Q: "What is the first check before operating a forklift?"
A: "Inspect using the pre-operation checklist: check tires, forks, fluid levels,
    warning lights, horn, brakes, and ensure the load capacity plate is visible
    and legible. Never operate a forklift that fails any inspection point."
```

---

## Agent State Machine

```
WATCHING
  ├─ VISUAL_FRAME (attentive=false, away≥5s)   → PAUSED_FOR_QUIZ
  ├─ VIDEO_SEEK_ATTEMPT (skip>15s)              → PAUSED_FOR_QUIZ
  └─ VIDEO_TIME_UPDATE                          → stay WATCHING

PAUSED_FOR_QUIZ
  ├─ QUIZ_ANSWER correct                        → WATCHING + RESUME_VIDEO
  ├─ QUIZ_ANSWER wrong (streak < 2)             → PAUSED_FOR_QUIZ + SHOW_QUIZ again
  └─ QUIZ_ANSWER wrong (streak ≥ 2)             → REMEDIATION + SEEK_RELATIVE(-10)

REMEDIATION
  └─ same transitions as PAUSED_FOR_QUIZ

COMPLETED (video end)
  └─ SHOW_SUMMARY { score, interventions, correct }
```

FunctionGemma output maps directly onto these transitions. The state machine enforces hard safety guarantees (can't skip more than 15s, can't ignore distraction) while FunctionGemma provides learned nuance within those rules.

---

## WebSocket Interface Contract

### Events → backend (frontend sends)
```json
{ "type": "VISUAL_FRAME", "video_frame": "base64...", "webcam_frame": "base64...", "t": 42.1 }
{ "type": "VIDEO_TIME_UPDATE", "t": 42.1 }
{ "type": "VIDEO_SEEK_ATTEMPT", "from_t": 40.2, "to_t": 85.0 }
{ "type": "VIDEO_PAUSED", "t": 42.1 }
{ "type": "VIDEO_PLAYED", "t": 42.1 }
{ "type": "QUIZ_ANSWER", "text": "Check the tires and forks first", "t_context": 72.0 }
```

### Actions → frontend (backend sends)
```json
{ "type": "PAUSE_VIDEO" }
{ "type": "RESUME_VIDEO" }
{ "type": "SEEK_RELATIVE", "delta": -10 }
{ "type": "SHOW_QUIZ", "prompt": "What is the first safety check before operating?" }
{ "type": "SHOW_TOAST", "message": "Rewatching critical section" }
{ "type": "SPEAK", "text": "Not quite. Rewatch this section." }
{ "type": "SHOW_SUMMARY", "score": 0.8, "interventions": 3, "correct": 2 }
```

---

## Backend File Structure

```
backend/
├── app/
│   ├── main.py          ← FastAPI app + WebSocket endpoint
│   ├── models.py        ← Pydantic event/action models + parse_event()
│   ├── session.py       ← SessionState + AgentState enum
│   ├── policy.py        ← PolicyEngine (state machine, calls model_server)
│   └── model_server.py  ← Loads FunctionGemma + Gemma 3 12B, runs inference
├── finetune/
│   ├── generate_data.py ← Generates synthetic training examples
│   ├── train_function_gemma.py
│   └── train_gemma12b.py
├── tests/
│   ├── test_models.py
│   ├── test_session.py
│   ├── test_policy.py
│   └── test_model_server.py
├── requirements.txt
└── README.md
```

---

## Build Order (8 hours)

| Hour | Task |
|---|---|
| 0-1 | Scaffold backend, models.py, session.py, policy.py (deterministic fallback) |
| 1-2 | Generate fine-tuning data, run FunctionGemma fine-tune |
| 2-3 | Run Gemma 3 12B LoRA fine-tune |
| 3-4 | Implement model_server.py, wire up to policy.py + main.py |
| 4-6 | Frontend: video player + webcam capture + frame sending + action execution (teammate) |
| 6-7 | Integration: connect frontend to backend over ngrok tunnel |
| 7-8 | Live demo testing with real forklift video |

---

## Demo Script

1. Start forklift safety video — learner watches normally
2. Learner looks away for 6 seconds during pre-operation checklist
3. **Model sees it** — video pauses, quiz appears: *"What must be checked before starting the forklift?"*
4. Learner answers incorrectly → model rewinds 10s, shows toast "Rewatching critical section", asks again
5. Learner answers correctly → video resumes, TTS: *"Correct. Always complete the pre-operation checklist."*
6. Learner tries to skip forward 40 seconds → blocked, model asks about the skipped content
7. Video completes → summary card: score 85%, 3 interventions, 4/5 questions correct
