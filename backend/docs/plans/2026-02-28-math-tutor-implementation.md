# Adaptive Math Tutor — Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the forklift backend with an adaptive math tutoring backend — same FastAPI/WebSocket/dual-model architecture, new state machine that reads worksheet photos, teaches long division via virtual whiteboard drawing commands, and adaptively shifts control from agent-led to child-led based on student confidence.

**Architecture:** FastAPI + WebSocket on RTX 6000. Gemma 3 12B reads worksheet photos and generates Socratic tutoring dialogue + structured drawing commands. FunctionGemma 270M classifies student response quality and decides when to shift the tutoring phase (AGENT_LED → COLLABORATIVE → CHILD_LED). PolicyEngine runs the state machine. ~70% of existing file structure reused, all content replaced.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, HuggingFace Transformers, PEFT, PyTorch, Pillow, Pydantic v2, pytest, pytest-asyncio

---

## What Changes vs What Stays

| File | Action |
|---|---|
| `app/models.py` | Replace entirely — new events + drawing actions |
| `app/session.py` | Replace entirely — TutoringPhase + new fields |
| `app/policy.py` | Replace entirely — new state machine |
| `app/model_server.py` | Replace prompts + methods, keep class structure |
| `app/main.py` | Minor update — new env vars, same WebSocket loop |
| `finetune/generate_data.py` | Replace entirely — math tutoring data |
| `finetune/train_function_gemma.py` | Keep structure, update config |
| `finetune/train_gemma12b.py` | Keep structure, update config |
| `requirements.txt` | No change |
| `pyproject.toml` | No change |
| `tests/` | Replace all test files |

---

## WebSocket Interface Contract (source of truth for teammate)

### Events → backend
```json
{ "type": "WORKSHEET_PHOTO", "image": "<base64 JPEG>" }
{ "type": "STUDENT_SPEECH", "text": "bring down the 7", "t": 42.1 }
{ "type": "FACE_FRAME", "image": "<base64 JPEG>", "t": 42.1 }
```

### Actions → frontend
```json
{ "type": "DRAW_PROBLEM", "dividend": 247, "divisor": 6 }
{ "type": "DRAW_CIRCLE", "target": "first_two_digits" }
{ "type": "DRAW_NUMBER", "value": 4, "position": "quotient_0" }
{ "type": "DRAW_MULTIPLY", "value": 24, "position": "subtract_row_0" }
{ "type": "DRAW_LINE", "position": "subtract_line_0" }
{ "type": "DRAW_BRING_DOWN", "digit_index": 2 }
{ "type": "DRAW_REMAINDER", "value": 1 }
{ "type": "DRAW_HINT", "hint_type": "show_multiplication_table", "a": 6, "b": 4 }
{ "type": "SPEAK", "text": "Exactly right! So we write 4 above the line." }
{ "type": "SHIFT_CONTROL", "to": "COLLABORATIVE" }
{ "type": "SHOW_SUMMARY", "problems_done": 3, "confidence_end": 0.85 }
```

---

## Task 1: Replace models.py

**Files:**
- Modify: `backend/app/models.py` (replace entirely)
- Modify: `backend/tests/test_models.py` (replace entirely)

**Step 1: Write the failing tests**

```python
# backend/tests/test_models.py
import pytest
from app.models import (
    WorksheetPhotoEvent, StudentSpeechEvent, FaceFrameEvent,
    DrawProblemAction, DrawCircleAction, DrawNumberAction,
    DrawMultiplyAction, DrawLineAction, DrawBringDownAction,
    DrawRemainderAction, DrawHintAction, SpeakAction,
    ShiftControlAction, ShowSummaryAction,
    parse_event,
)


def test_parse_worksheet_photo():
    event = parse_event({"type": "WORKSHEET_PHOTO", "image": "base64abc"})
    assert isinstance(event, WorksheetPhotoEvent)
    assert event.image == "base64abc"


def test_parse_student_speech():
    event = parse_event({"type": "STUDENT_SPEECH", "text": "bring down the 7", "t": 42.1})
    assert isinstance(event, StudentSpeechEvent)
    assert event.text == "bring down the 7"
    assert event.t == 42.1


def test_parse_face_frame():
    event = parse_event({"type": "FACE_FRAME", "image": "base64xyz", "t": 10.0})
    assert isinstance(event, FaceFrameEvent)
    assert event.t == 10.0


def test_parse_unknown_raises():
    with pytest.raises(ValueError, match="Unknown event type"):
        parse_event({"type": "UNKNOWN"})


def test_draw_problem_serializes():
    action = DrawProblemAction(dividend=247, divisor=6)
    d = action.model_dump()
    assert d == {"type": "DRAW_PROBLEM", "dividend": 247, "divisor": 6}


def test_draw_number_serializes():
    action = DrawNumberAction(value=4, position="quotient_0")
    d = action.model_dump()
    assert d["type"] == "DRAW_NUMBER"
    assert d["value"] == 4
    assert d["position"] == "quotient_0"


def test_shift_control_serializes():
    action = ShiftControlAction(to="COLLABORATIVE")
    d = action.model_dump()
    assert d == {"type": "SHIFT_CONTROL", "to": "COLLABORATIVE"}


def test_show_summary_serializes():
    action = ShowSummaryAction(problems_done=3, confidence_end=0.85)
    d = action.model_dump()
    assert d == {"type": "SHOW_SUMMARY", "problems_done": 3, "confidence_end": 0.85}
```

**Step 2: Run to verify failure**

```bash
cd backend
python -m pytest tests/test_models.py -v 2>&1 | head -5
```
Expected: `ImportError` or `ModuleNotFoundError`

**Step 3: Implement models.py**

```python
# backend/app/models.py
from __future__ import annotations
from typing import Literal, Union
from pydantic import BaseModel


# ── Events (frontend → backend) ──────────────────────────────────────────────

class WorksheetPhotoEvent(BaseModel):
    type: Literal["WORKSHEET_PHOTO"]
    image: str  # base64 JPEG


class StudentSpeechEvent(BaseModel):
    type: Literal["STUDENT_SPEECH"]
    text: str
    t: float = 0.0


class FaceFrameEvent(BaseModel):
    type: Literal["FACE_FRAME"]
    image: str  # base64 JPEG
    t: float = 0.0


Event = Union[WorksheetPhotoEvent, StudentSpeechEvent, FaceFrameEvent]

_EVENT_MAP: dict[str, type] = {
    "WORKSHEET_PHOTO": WorksheetPhotoEvent,
    "STUDENT_SPEECH": StudentSpeechEvent,
    "FACE_FRAME": FaceFrameEvent,
}


def parse_event(data: dict) -> Event:
    event_type = data.get("type")
    cls = _EVENT_MAP.get(event_type)
    if cls is None:
        raise ValueError(f"Unknown event type: {event_type!r}")
    return cls(**data)


# ── Actions (backend → frontend) ─────────────────────────────────────────────

class DrawProblemAction(BaseModel):
    type: Literal["DRAW_PROBLEM"] = "DRAW_PROBLEM"
    dividend: int
    divisor: int


class DrawCircleAction(BaseModel):
    type: Literal["DRAW_CIRCLE"] = "DRAW_CIRCLE"
    target: str


class DrawNumberAction(BaseModel):
    type: Literal["DRAW_NUMBER"] = "DRAW_NUMBER"
    value: int
    position: str


class DrawMultiplyAction(BaseModel):
    type: Literal["DRAW_MULTIPLY"] = "DRAW_MULTIPLY"
    value: int
    position: str


class DrawLineAction(BaseModel):
    type: Literal["DRAW_LINE"] = "DRAW_LINE"
    position: str


class DrawBringDownAction(BaseModel):
    type: Literal["DRAW_BRING_DOWN"] = "DRAW_BRING_DOWN"
    digit_index: int


class DrawRemainderAction(BaseModel):
    type: Literal["DRAW_REMAINDER"] = "DRAW_REMAINDER"
    value: int


class DrawHintAction(BaseModel):
    type: Literal["DRAW_HINT"] = "DRAW_HINT"
    hint_type: str
    a: int = 0
    b: int = 0


class SpeakAction(BaseModel):
    type: Literal["SPEAK"] = "SPEAK"
    text: str


class ShiftControlAction(BaseModel):
    type: Literal["SHIFT_CONTROL"] = "SHIFT_CONTROL"
    to: str


class ShowSummaryAction(BaseModel):
    type: Literal["SHOW_SUMMARY"] = "SHOW_SUMMARY"
    problems_done: int
    confidence_end: float


Action = Union[
    DrawProblemAction, DrawCircleAction, DrawNumberAction,
    DrawMultiplyAction, DrawLineAction, DrawBringDownAction,
    DrawRemainderAction, DrawHintAction, SpeakAction,
    ShiftControlAction, ShowSummaryAction,
]
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_models.py -v
```
Expected: 8 tests PASS.

**Step 5: Commit**

```bash
git add backend/app/models.py backend/tests/test_models.py
git commit -m "feat(tutor): replace models with math tutor events and drawing actions"
```

---

## Task 2: Replace session.py

**Files:**
- Modify: `backend/app/session.py` (replace entirely)
- Modify: `backend/tests/test_session.py` (replace entirely)

**Step 1: Write failing tests**

```python
# backend/tests/test_session.py
import pytest
from app.session import TutoringSession, TutoringPhase


def test_initial_phase():
    s = TutoringSession()
    assert s.phase == TutoringPhase.READING_WORKSHEET
    assert s.confidence_streak == 0
    assert s.problems_done == 0
    assert s.current_problem is None
    assert s.problems_queue == []


def test_set_current_problem():
    s = TutoringSession()
    problem = {"dividend": 247, "divisor": 6, "student_answer": "40 r7", "error_type": "incomplete"}
    s.set_current_problem(problem)
    assert s.current_problem == problem
    assert s.phase == TutoringPhase.PROBLEM_SELECTED


def test_record_correct_increments_streak():
    s = TutoringSession()
    s.phase = TutoringPhase.AGENT_LED
    s.record_response(correct=True)
    assert s.confidence_streak == 1


def test_record_wrong_resets_streak():
    s = TutoringSession()
    s.phase = TutoringPhase.COLLABORATIVE
    s.confidence_streak = 2
    s.record_response(correct=False)
    assert s.confidence_streak == 0


def test_shift_phase():
    s = TutoringSession()
    s.phase = TutoringPhase.AGENT_LED
    s.shift_phase(TutoringPhase.COLLABORATIVE)
    assert s.phase == TutoringPhase.COLLABORATIVE
    assert TutoringPhase.COLLABORATIVE.value in s.phase_history


def test_complete_problem():
    s = TutoringSession()
    s.phase = TutoringPhase.CHILD_LED
    s.complete_current_problem()
    assert s.problems_done == 1
    assert s.current_problem is None
    assert s.phase == TutoringPhase.UNDERSTANDING_CONFIRMED


def test_competency_ratio_no_problems():
    s = TutoringSession()
    assert s.competency_ratio == 0.0


def test_competency_ratio_with_problems():
    s = TutoringSession()
    s.correct_total = 3
    s.problems_done = 4
    assert s.competency_ratio == 0.75


def test_queue_problems():
    s = TutoringSession()
    problems = [{"dividend": 24, "divisor": 3}, {"dividend": 48, "divisor": 6}]
    s.queue_problems(problems)
    assert len(s.problems_queue) == 2


def test_next_problem_from_queue():
    s = TutoringSession()
    s.problems_queue = [{"dividend": 24, "divisor": 3}]
    problem = s.next_problem()
    assert problem["dividend"] == 24
    assert s.problems_queue == []
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_session.py -v 2>&1 | head -5
```
Expected: `ImportError`

**Step 3: Implement session.py**

```python
# backend/app/session.py
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field


class TutoringPhase(str, Enum):
    READING_WORKSHEET = "READING_WORKSHEET"
    PROBLEM_SELECTED = "PROBLEM_SELECTED"
    AGENT_LED = "AGENT_LED"
    COLLABORATIVE = "COLLABORATIVE"
    CHILD_LED = "CHILD_LED"
    UNDERSTANDING_CONFIRMED = "UNDERSTANDING_CONFIRMED"
    SESSION_COMPLETE = "SESSION_COMPLETE"


@dataclass
class TutoringSession:
    session_id: str = "default"
    phase: TutoringPhase = TutoringPhase.READING_WORKSHEET
    current_problem: dict | None = None
    problems_queue: list = field(default_factory=list)
    confidence_streak: int = 0
    correct_total: int = 0
    problems_done: int = 0
    phase_history: list = field(default_factory=list)
    frustrated: bool = False

    def set_current_problem(self, problem: dict) -> None:
        self.current_problem = problem
        self.confidence_streak = 0
        self.phase = TutoringPhase.PROBLEM_SELECTED

    def queue_problems(self, problems: list[dict]) -> None:
        self.problems_queue = list(problems)

    def next_problem(self) -> dict | None:
        if not self.problems_queue:
            return None
        return self.problems_queue.pop(0)

    def record_response(self, correct: bool) -> None:
        if correct:
            self.confidence_streak += 1
            self.correct_total += 1
        else:
            self.confidence_streak = 0

    def shift_phase(self, new_phase: TutoringPhase) -> None:
        self.phase_history.append(new_phase.value)
        self.phase = new_phase

    def complete_current_problem(self) -> None:
        self.problems_done += 1
        self.current_problem = None
        self.confidence_streak = 0
        self.phase = TutoringPhase.UNDERSTANDING_CONFIRMED

    @property
    def competency_ratio(self) -> float:
        if self.problems_done == 0:
            return 0.0
        return self.correct_total / self.problems_done
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_session.py -v
```
Expected: 10 tests PASS.

**Step 5: Commit**

```bash
git add backend/app/session.py backend/tests/test_session.py
git commit -m "feat(tutor): replace SessionState with TutoringSession and TutoringPhase"
```

---

## Task 3: Replace policy.py

**Files:**
- Modify: `backend/app/policy.py` (replace entirely)
- Modify: `backend/tests/test_policy.py` (replace entirely)

**Step 1: Write failing tests**

```python
# backend/tests/test_policy.py
import pytest
from unittest.mock import AsyncMock
from app.session import TutoringSession, TutoringPhase
from app.models import (
    WorksheetPhotoEvent, StudentSpeechEvent, FaceFrameEvent,
    DrawProblemAction, SpeakAction, ShiftControlAction, ShowSummaryAction,
)
from app.policy import TutoringPolicy


@pytest.fixture
def mock_model_server():
    m = AsyncMock()
    m.read_worksheet.return_value = [
        {"dividend": 247, "divisor": 6, "student_answer": "40 r7", "error_type": "incomplete_quotient"}
    ]
    m.analyze_face.return_value = {"frustrated": False, "engaged": True}
    m.classify_response.return_value = {
        "quality": "confident_correct",
        "next_draw": [{"type": "DRAW_NUMBER", "value": 4, "position": "quotient_0"}],
        "speech": "Exactly right!",
    }
    m.generate_tutoring_step.return_value = {
        "speech": "Can 6 go into just 2?",
        "drawing_commands": [{"type": "DRAW_CIRCLE", "target": "first_digit"}],
    }
    return m


@pytest.fixture
def policy(mock_model_server):
    return TutoringPolicy(
        model_server=mock_model_server,
        agent_led_threshold=2,
        collaborative_threshold=3,
    )


@pytest.fixture
def session():
    return TutoringSession()


@pytest.mark.asyncio
async def test_worksheet_photo_queues_problems(policy, session, mock_model_server):
    event = WorksheetPhotoEvent(type="WORKSHEET_PHOTO", image="base64abc")
    actions = await policy.process(event, session)
    types = [a.type for a in actions]
    assert "DRAW_PROBLEM" in types
    assert "SPEAK" in types
    assert session.phase == TutoringPhase.AGENT_LED


@pytest.mark.asyncio
async def test_worksheet_photo_no_errors_shows_summary(policy, session, mock_model_server):
    mock_model_server.read_worksheet.return_value = []
    event = WorksheetPhotoEvent(type="WORKSHEET_PHOTO", image="base64abc")
    actions = await policy.process(event, session)
    types = [a.type for a in actions]
    assert "SHOW_SUMMARY" in types
    assert session.phase == TutoringPhase.SESSION_COMPLETE


@pytest.mark.asyncio
async def test_correct_response_increments_streak(policy, session):
    session.phase = TutoringPhase.AGENT_LED
    session.set_current_problem({"dividend": 247, "divisor": 6, "student_answer": "40", "error_type": "x"})
    event = StudentSpeechEvent(type="STUDENT_SPEECH", text="4 times", t=10.0)
    actions = await policy.process(event, session)
    assert session.confidence_streak == 1


@pytest.mark.asyncio
async def test_agent_led_shifts_to_collaborative_at_threshold(policy, session, mock_model_server):
    mock_model_server.classify_response.return_value = {
        "quality": "confident_correct",
        "next_draw": [],
        "speech": "Correct!",
    }
    session.phase = TutoringPhase.AGENT_LED
    session.confidence_streak = 1  # one more correct → hits threshold of 2
    session.set_current_problem({"dividend": 247, "divisor": 6, "student_answer": "40", "error_type": "x"})
    session.phase = TutoringPhase.AGENT_LED
    event = StudentSpeechEvent(type="STUDENT_SPEECH", text="4 times", t=10.0)
    actions = await policy.process(event, session)
    types = [a.type for a in actions]
    assert "SHIFT_CONTROL" in types
    shift = next(a for a in actions if a.type == "SHIFT_CONTROL")
    assert shift.to == "COLLABORATIVE"


@pytest.mark.asyncio
async def test_wrong_response_resets_streak_and_stays_agent_led(policy, session, mock_model_server):
    mock_model_server.classify_response.return_value = {
        "quality": "fundamentally_wrong",
        "next_draw": [],
        "speech": "Let me show you again.",
    }
    session.phase = TutoringPhase.COLLABORATIVE
    session.confidence_streak = 2
    session.set_current_problem({"dividend": 247, "divisor": 6, "student_answer": "40", "error_type": "x"})
    session.phase = TutoringPhase.COLLABORATIVE
    event = StudentSpeechEvent(type="STUDENT_SPEECH", text="100 times", t=10.0)
    actions = await policy.process(event, session)
    assert session.confidence_streak == 0
    assert session.phase == TutoringPhase.AGENT_LED


@pytest.mark.asyncio
async def test_face_frustrated_downgrades_to_agent_led(policy, session, mock_model_server):
    mock_model_server.analyze_face.return_value = {"frustrated": True, "engaged": False}
    session.phase = TutoringPhase.CHILD_LED
    session.set_current_problem({"dividend": 247, "divisor": 6, "student_answer": "40", "error_type": "x"})
    session.phase = TutoringPhase.CHILD_LED
    event = FaceFrameEvent(type="FACE_FRAME", image="base64", t=5.0)
    actions = await policy.process(event, session)
    assert session.phase == TutoringPhase.AGENT_LED
    types = [a.type for a in actions]
    assert "SPEAK" in types


@pytest.mark.asyncio
async def test_face_not_frustrated_no_action(policy, session, mock_model_server):
    mock_model_server.analyze_face.return_value = {"frustrated": False, "engaged": True}
    session.phase = TutoringPhase.COLLABORATIVE
    event = FaceFrameEvent(type="FACE_FRAME", image="base64", t=5.0)
    actions = await policy.process(event, session)
    assert actions == []


@pytest.mark.asyncio
async def test_collaborative_shifts_to_child_led_at_threshold(policy, session, mock_model_server):
    mock_model_server.classify_response.return_value = {
        "quality": "confident_correct",
        "next_draw": [],
        "speech": "Well done!",
    }
    session.phase = TutoringPhase.COLLABORATIVE
    session.confidence_streak = 2  # one more → hits threshold of 3
    session.set_current_problem({"dividend": 247, "divisor": 6, "student_answer": "40", "error_type": "x"})
    session.phase = TutoringPhase.COLLABORATIVE
    event = StudentSpeechEvent(type="STUDENT_SPEECH", text="bring down 7", t=10.0)
    actions = await policy.process(event, session)
    types = [a.type for a in actions]
    assert "SHIFT_CONTROL" in types
    shift = next(a for a in actions if a.type == "SHIFT_CONTROL")
    assert shift.to == "CHILD_LED"
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_policy.py -v 2>&1 | head -5
```
Expected: `ImportError`

**Step 3: Implement policy.py**

```python
# backend/app/policy.py
from __future__ import annotations
import logging
from app.session import TutoringSession, TutoringPhase
from app.models import (
    Action, Event,
    WorksheetPhotoEvent, StudentSpeechEvent, FaceFrameEvent,
    DrawProblemAction, DrawCircleAction, DrawNumberAction,
    DrawMultiplyAction, DrawLineAction, DrawBringDownAction,
    DrawRemainderAction, DrawHintAction, SpeakAction,
    ShiftControlAction, ShowSummaryAction,
)

logger = logging.getLogger(__name__)

_PHASE_DOWN = {
    TutoringPhase.CHILD_LED: TutoringPhase.COLLABORATIVE,
    TutoringPhase.COLLABORATIVE: TutoringPhase.AGENT_LED,
    TutoringPhase.AGENT_LED: TutoringPhase.AGENT_LED,
}


class TutoringPolicy:
    def __init__(
        self,
        model_server,
        agent_led_threshold: int = 2,
        collaborative_threshold: int = 3,
    ):
        self.model_server = model_server
        self.agent_led_threshold = agent_led_threshold
        self.collaborative_threshold = collaborative_threshold

    async def process(self, event: Event, session: TutoringSession) -> list[Action]:
        match event:
            case WorksheetPhotoEvent():
                return await self._on_worksheet(event, session)
            case StudentSpeechEvent():
                return await self._on_speech(event, session)
            case FaceFrameEvent():
                return await self._on_face(event, session)
            case _:
                return []

    async def _on_worksheet(self, event: WorksheetPhotoEvent, session: TutoringSession) -> list[Action]:
        problems = await self.model_server.read_worksheet(event.image)
        if not problems:
            session.phase = TutoringPhase.SESSION_COMPLETE
            return [
                SpeakAction(text="I couldn't find any problems to work on, or everything looks correct!"),
                ShowSummaryAction(problems_done=0, confidence_end=1.0),
            ]
        session.queue_problems(problems)
        return await self._start_next_problem(session)

    async def _start_next_problem(self, session: TutoringSession) -> list[Action]:
        problem = session.next_problem()
        if problem is None:
            session.phase = TutoringPhase.SESSION_COMPLETE
            return [ShowSummaryAction(
                problems_done=session.problems_done,
                confidence_end=session.competency_ratio,
            )]
        session.set_current_problem(problem)
        session.phase = TutoringPhase.AGENT_LED
        step = await self.model_server.generate_tutoring_step(problem, TutoringPhase.AGENT_LED, [])
        actions: list[Action] = [DrawProblemAction(
            dividend=problem["dividend"],
            divisor=problem["divisor"],
        )]
        actions += self._parse_drawing_commands(step.get("drawing_commands", []))
        if step.get("speech"):
            actions.append(SpeakAction(text=step["speech"]))
        return actions

    async def _on_speech(self, event: StudentSpeechEvent, session: TutoringSession) -> list[Action]:
        if session.phase not in (TutoringPhase.AGENT_LED, TutoringPhase.COLLABORATIVE, TutoringPhase.CHILD_LED):
            return []
        result = await self.model_server.classify_response(event.text, {
            "phase": session.phase.value,
            "problem": session.current_problem,
            "streak": session.confidence_streak,
        })
        quality = result.get("quality", "hesitant_correct")
        correct = quality in ("confident_correct", "hesitant_correct")
        session.record_response(correct=correct)
        actions: list[Action] = self._parse_drawing_commands(result.get("next_draw", []))
        if result.get("speech"):
            actions.append(SpeakAction(text=result["speech"]))

        if not correct and quality == "fundamentally_wrong":
            session.shift_phase(TutoringPhase.AGENT_LED)
            return actions

        # Check phase upgrade thresholds
        if session.phase == TutoringPhase.AGENT_LED and session.confidence_streak >= self.agent_led_threshold:
            session.shift_phase(TutoringPhase.COLLABORATIVE)
            actions.append(ShiftControlAction(to="COLLABORATIVE"))
            actions.append(SpeakAction(text="Great work — now you tell me the next step."))
        elif session.phase == TutoringPhase.COLLABORATIVE and session.confidence_streak >= self.collaborative_threshold:
            session.shift_phase(TutoringPhase.CHILD_LED)
            actions.append(ShiftControlAction(to="CHILD_LED"))
            actions.append(SpeakAction(text="You've got this — take it from here."))

        # Check problem completion
        if quality == "confident_correct" and session.phase == TutoringPhase.CHILD_LED:
            if self._is_problem_complete(event.text, session.current_problem):
                session.complete_current_problem()
                next_actions = await self._start_next_problem(session)
                return actions + next_actions

        return actions

    async def _on_face(self, event: FaceFrameEvent, session: TutoringSession) -> list[Action]:
        if session.phase not in (TutoringPhase.AGENT_LED, TutoringPhase.COLLABORATIVE, TutoringPhase.CHILD_LED):
            return []
        result = await self.model_server.analyze_face(event.image)
        if not result.get("frustrated", False):
            return []
        # Downgrade phase on frustration
        new_phase = _PHASE_DOWN.get(session.phase, TutoringPhase.AGENT_LED)
        if new_phase != session.phase:
            session.shift_phase(new_phase)
        session.frustrated = True
        return [SpeakAction(text="Let me walk you through this step again.")]

    def _parse_drawing_commands(self, commands: list[dict]) -> list[Action]:
        result = []
        for cmd in commands:
            t = cmd.get("type")
            if t == "DRAW_NUMBER":
                result.append(DrawNumberAction(value=cmd["value"], position=cmd["position"]))
            elif t == "DRAW_MULTIPLY":
                result.append(DrawMultiplyAction(value=cmd["value"], position=cmd["position"]))
            elif t == "DRAW_LINE":
                result.append(DrawLineAction(position=cmd["position"]))
            elif t == "DRAW_BRING_DOWN":
                result.append(DrawBringDownAction(digit_index=cmd["digit_index"]))
            elif t == "DRAW_REMAINDER":
                result.append(DrawRemainderAction(value=cmd["value"]))
            elif t == "DRAW_CIRCLE":
                result.append(DrawCircleAction(target=cmd["target"]))
            elif t == "DRAW_HINT":
                result.append(DrawHintAction(hint_type=cmd["hint_type"], a=cmd.get("a", 0), b=cmd.get("b", 0)))
        return result

    def _is_problem_complete(self, text: str, problem: dict | None) -> bool:
        if problem is None:
            return False
        keywords = {"remainder", "done", "finished", "that's it", "complete", "r"}
        return bool(set(text.lower().split()) & keywords)
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_policy.py -v
```
Expected: 8 tests PASS.

**Step 5: Commit**

```bash
git add backend/app/policy.py backend/tests/test_policy.py
git commit -m "feat(tutor): replace PolicyEngine with TutoringPolicy state machine"
```

---

## Task 4: Replace model_server.py

**Files:**
- Modify: `backend/app/model_server.py` (replace entirely)
- Modify: `backend/tests/test_model_server.py` (replace entirely)

**Step 1: Write failing tests**

```python
# backend/tests/test_model_server.py
import pytest
import base64
from unittest.mock import AsyncMock
from PIL import Image
import io
from app.model_server import MathTutorModelServer


def _tiny_b64() -> str:
    img = Image.new("RGB", (4, 4), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_decode_image_returns_pil():
    srv = MathTutorModelServer.__new__(MathTutorModelServer)
    img = srv._decode_image(_tiny_b64())
    assert isinstance(img, Image.Image)


def test_decode_empty_returns_none():
    srv = MathTutorModelServer.__new__(MathTutorModelServer)
    assert srv._decode_image("") is None


def test_parse_json_valid():
    srv = MathTutorModelServer.__new__(MathTutorModelServer)
    result = srv._parse_json('{"a": 1}', fallback={"a": 0})
    assert result == {"a": 1}


def test_parse_json_fallback_on_bad():
    srv = MathTutorModelServer.__new__(MathTutorModelServer)
    result = srv._parse_json("not json", fallback={"a": 0})
    assert result == {"a": 0}


@pytest.mark.asyncio
async def test_read_worksheet_returns_list():
    srv = MathTutorModelServer.__new__(MathTutorModelServer)
    srv._gemma_generate = AsyncMock(return_value=(
        '[{"dividend": 247, "divisor": 6, "student_answer": "40 r7", "error_type": "incomplete_quotient"}]'
    ))
    result = await srv.read_worksheet(_tiny_b64())
    assert isinstance(result, list)
    assert result[0]["dividend"] == 247


@pytest.mark.asyncio
async def test_read_worksheet_fallback_on_failure():
    srv = MathTutorModelServer.__new__(MathTutorModelServer)
    srv._gemma_generate = AsyncMock(side_effect=RuntimeError("no model"))
    result = await srv.read_worksheet(_tiny_b64())
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_analyze_face_returns_dict():
    srv = MathTutorModelServer.__new__(MathTutorModelServer)
    srv._gemma_generate = AsyncMock(return_value='{"frustrated": false, "engaged": true}')
    result = await srv.analyze_face(_tiny_b64())
    assert result["frustrated"] is False
    assert result["engaged"] is True


@pytest.mark.asyncio
async def test_classify_response_returns_quality():
    srv = MathTutorModelServer.__new__(MathTutorModelServer)
    srv._gemma_generate = AsyncMock(return_value=(
        '{"quality": "confident_correct", "next_draw": [], "speech": "Correct!"}'
    ))
    result = await srv.classify_response("4 times", {"phase": "AGENT_LED"})
    assert result["quality"] == "confident_correct"
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_model_server.py -v 2>&1 | head -5
```
Expected: `ImportError`

**Step 3: Implement model_server.py**

```python
# backend/app/model_server.py
from __future__ import annotations
import asyncio
import base64
import json
import logging
import random
from io import BytesIO
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

_FALLBACK_PROBLEMS = [
    {"dividend": 84, "divisor": 4, "student_answer": "20", "error_type": "incomplete_quotient"},
    {"dividend": 126, "divisor": 3, "student_answer": "40", "error_type": "carry_error"},
]

_FALLBACK_TUTORING_STEPS = [
    {"speech": "Let's look at this problem together. What's the first thing we do in long division?",
     "drawing_commands": []},
    {"speech": "Can the divisor go into just the first digit?",
     "drawing_commands": [{"type": "DRAW_CIRCLE", "target": "first_digit"}]},
]


class MathTutorModelServer:
    def __init__(
        self,
        gemma12b_base: str = "google/gemma-3-12b-it",
        gemma12b_adapter: str = "adapters/gemma12b",
        function_gemma_base: str = "google/functiongemma-270m-it",
        function_gemma_adapter: str = "adapters/function_gemma",
        device: str = "cuda",
        load_in_4bit: bool = True,
    ):
        self.device = device
        self._pipe = None
        self._func_model = None
        self._func_tokenizer = None
        try:
            self._load_models(
                gemma12b_base, gemma12b_adapter,
                function_gemma_base, function_gemma_adapter,
                load_in_4bit,
            )
        except Exception as e:
            logger.warning("Models failed to load, using fallback mode: %s", e)

    def _load_models(self, gemma12b_base, gemma12b_adapter, func_base, func_adapter, load_in_4bit) -> None:
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        logger.info("Loading Gemma 3 12B...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
        ) if load_in_4bit else None

        self._pipe = pipeline(
            "image-text-to-text",
            model=gemma12b_base,
            model_kwargs={"quantization_config": bnb_config} if bnb_config else {},
            device_map="auto",
        )
        try:
            self._pipe.model = PeftModel.from_pretrained(self._pipe.model, gemma12b_adapter)
            logger.info("Gemma 3 12B adapter loaded.")
        except Exception:
            logger.warning("No Gemma 3 12B adapter at %s, using base model.", gemma12b_adapter)

        logger.info("Loading FunctionGemma 270M...")
        self._func_tokenizer = AutoTokenizer.from_pretrained(func_base)
        func_model = AutoModelForCausalLM.from_pretrained(func_base, device_map=self.device)
        try:
            self._func_model = PeftModel.from_pretrained(func_model, func_adapter)
            logger.info("FunctionGemma adapter loaded.")
        except Exception:
            self._func_model = func_model
            logger.warning("No FunctionGemma adapter at %s, using base model.", func_adapter)

        logger.info("All models loaded.")

    # ── Public async API ──────────────────────────────────────────────────────

    async def read_worksheet(self, image_b64: str) -> list[dict]:
        """Read worksheet photo. Returns list of wrong problems found."""
        try:
            img = self._decode_image(image_b64)
            prompt = (
                "You are a math teacher. This image shows a student's completed worksheet.\n"
                "Find all INCORRECT long division problems. For each wrong answer return:\n"
                '{"dividend": int, "divisor": int, "student_answer": "string", "error_type": "string"}\n'
                "Return ONLY a valid JSON array. If all answers are correct return []."
            )
            raw = await self._gemma_generate(prompt=prompt, images=[img] if img else [])
            result = self._parse_json(raw, fallback=[])
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.warning("read_worksheet failed: %s", e)
            return []

    async def analyze_face(self, image_b64: str) -> dict[str, Any]:
        """Detect frustration/engagement from webcam frame."""
        try:
            img = self._decode_image(image_b64)
            prompt = (
                "You see a child's face during a tutoring session.\n"
                'Respond ONLY with valid JSON: {"frustrated": true/false, "engaged": true/false}'
            )
            raw = await self._gemma_generate(prompt=prompt, images=[img] if img else [])
            return self._parse_json(raw, fallback={"frustrated": False, "engaged": True})
        except Exception as e:
            logger.warning("analyze_face failed: %s", e)
            return {"frustrated": False, "engaged": True}

    async def classify_response(self, response: str, context: dict) -> dict[str, Any]:
        """Classify student response quality. Called by FunctionGemma."""
        try:
            prompt = (
                f"Math tutoring context: {json.dumps(context)}\n"
                f"Student said: '{response}'\n\n"
                "Classify the response. Return ONLY valid JSON:\n"
                '{"quality": "confident_correct|hesitant_correct|hesitant_wrong|fundamentally_wrong", '
                '"next_draw": [], "speech": "one sentence response to student"}'
            )
            raw = await self._gemma_generate(prompt=prompt, images=[])
            return self._parse_json(raw, fallback=self._fallback_classify(response))
        except Exception as e:
            logger.warning("classify_response failed: %s", e)
            return self._fallback_classify(response)

    async def generate_tutoring_step(self, problem: dict, phase: Any, history: list) -> dict[str, Any]:
        """Generate next tutoring action for the current problem state."""
        try:
            prompt = (
                f"You are tutoring a student on long division: {problem['dividend']} ÷ {problem['divisor']}.\n"
                f"Their original answer was: {problem.get('student_answer', 'unknown')}.\n"
                f"Current teaching phase: {phase}. History: {history[-3:] if history else []}\n\n"
                "Generate the next tutoring step. Return ONLY valid JSON:\n"
                '{"speech": "what to say to the student", '
                '"drawing_commands": [list of drawing command objects]}'
            )
            raw = await self._gemma_generate(prompt=prompt, images=[])
            return self._parse_json(raw, fallback=random.choice(_FALLBACK_TUTORING_STEPS))
        except Exception as e:
            logger.warning("generate_tutoring_step failed: %s", e)
            return random.choice(_FALLBACK_TUTORING_STEPS)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _gemma_generate(self, prompt: str, images: list) -> str:
        if self._pipe is None:
            raise RuntimeError("Gemma 3 12B not loaded")

        def _run():
            messages = [{"role": "user", "content": []}]
            for img in images:
                messages[0]["content"].append({"type": "image", "image": img})
            messages[0]["content"].append({"type": "text", "text": prompt})
            out = self._pipe(messages, max_new_tokens=512)
            return out[0]["generated_text"][-1]["content"]

        return await asyncio.get_event_loop().run_in_executor(None, _run)

    def _decode_image(self, b64: str) -> Image.Image | None:
        if not b64:
            return None
        try:
            return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            return None

    def _parse_json(self, raw: str, fallback: Any) -> Any:
        try:
            # Handle both array and object responses
            for start_char in ["{", "["]:
                start = raw.find(start_char)
                if start != -1:
                    end_char = "}" if start_char == "{" else "]"
                    end = raw.rfind(end_char) + 1
                    return json.loads(raw[start:end])
            return fallback
        except Exception:
            return fallback

    def _fallback_classify(self, response: str) -> dict:
        positive = {"yes", "right", "times", "bring", "down", "divide", "remainder", "subtract"}
        correct = bool(set(response.lower().split()) & positive)
        return {
            "quality": "confident_correct" if correct else "hesitant_wrong",
            "next_draw": [],
            "speech": "Good thinking!" if correct else "Let me help you with that step.",
        }
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_model_server.py -v
```
Expected: 8 tests PASS.

**Step 5: Commit**

```bash
git add backend/app/model_server.py backend/tests/test_model_server.py
git commit -m "feat(tutor): replace ModelServer with MathTutorModelServer"
```

---

## Task 5: Update main.py

**Files:**
- Modify: `backend/app/main.py`

**Step 1: Replace main.py**

```python
# backend/app/main.py
from __future__ import annotations
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.models import parse_event
from app.session import TutoringSession
from app.policy import TutoringPolicy
from app.model_server import MathTutorModelServer

logger = logging.getLogger(__name__)

model_server: MathTutorModelServer | None = None
policy: TutoringPolicy | None = None
session: TutoringSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_server, policy, session
    logger.info("Loading models...")
    model_server = MathTutorModelServer(
        gemma12b_base=os.getenv("GEMMA12B_BASE", "google/gemma-3-12b-it"),
        gemma12b_adapter=os.getenv("GEMMA12B_ADAPTER", "adapters/gemma12b"),
        function_gemma_base=os.getenv("FUNC_GEMMA_BASE", "google/functiongemma-270m-it"),
        function_gemma_adapter=os.getenv("FUNC_GEMMA_ADAPTER", "adapters/function_gemma"),
        load_in_4bit=os.getenv("LOAD_IN_4BIT", "true").lower() == "true",
    )
    policy = TutoringPolicy(
        model_server=model_server,
        agent_led_threshold=int(os.getenv("AGENT_LED_THRESHOLD", "2")),
        collaborative_threshold=int(os.getenv("COLLAB_THRESHOLD", "3")),
    )
    session = TutoringSession()
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Adaptive Math Tutor Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "phase": session.phase if session else None}


@app.post("/reset")
async def reset_session():
    global session
    session = TutoringSession()
    return {"status": "reset"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            try:
                event = parse_event(data)
            except ValueError as e:
                logger.warning("Unknown event: %s", e)
                continue
            actions = await policy.process(event, session)
            for action in actions:
                await websocket.send_text(action.model_dump_json())
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
```

**Step 2: Verify**

```bash
python -c "from app.main import app; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add backend/app/main.py
git commit -m "feat(tutor): update main.py for math tutor models and session"
```

---

## Task 6: Fine-tuning data generation

**Files:**
- Modify: `backend/finetune/generate_data.py` (replace entirely)

**Step 1: Replace generate_data.py**

```python
# backend/finetune/generate_data.py
"""
Generates synthetic training data for:
  1. FunctionGemma 270M (response quality classifier)
  2. Gemma 3 12B (math tutoring dialogue + drawing commands)

Run: python finetune/generate_data.py
Output: finetune/data/function_gemma_train.jsonl
        finetune/data/gemma12b_train.jsonl
"""
import json, os

os.makedirs("finetune/data", exist_ok=True)

# ── FunctionGemma: response quality + action ──────────────────────────────────

FUNCTION_GEMMA_EXAMPLES = [
    # confident_correct → shift toward child-led
    {"input": {"response": "4 times", "step": "how many times does 6 go into 24", "phase": "AGENT_LED", "streak": 2},
     "output": "classify(confident_correct) | shift_control(COLLABORATIVE) | speak('Great work — now you tell me the next step.')"},
    {"input": {"response": "bring down the 7", "step": "next step after subtraction", "phase": "COLLABORATIVE", "streak": 3},
     "output": "classify(confident_correct) | shift_control(CHILD_LED) | draw(bring_down_7)"},
    {"input": {"response": "6 goes into 7 once", "step": "divide into remainder", "phase": "CHILD_LED", "streak": 1},
     "output": "classify(confident_correct) | draw(quotient_1) | speak('Exactly right!')"},
    {"input": {"response": "zero remainder 1", "step": "final remainder", "phase": "CHILD_LED", "streak": 2},
     "output": "classify(confident_correct) | draw(remainder_1) | speak('Perfect — 41 remainder 1.')"},
    # hesitant_correct → stay in current phase
    {"input": {"response": "um... 4 times?", "step": "how many times does 6 go into 24", "phase": "AGENT_LED", "streak": 0},
     "output": "classify(hesitant_correct) | speak('Yes, that is right! 4 times. Good thinking.')"},
    {"input": {"response": "I think bring down the 7?", "step": "next step", "phase": "COLLABORATIVE", "streak": 1},
     "output": "classify(hesitant_correct) | draw(bring_down_7) | speak('That is correct. Well done.')"},
    # hesitant_wrong → gentle nudge, stay
    {"input": {"response": "um... 3 times?", "step": "how many times does 6 go into 24", "phase": "AGENT_LED", "streak": 0},
     "output": "classify(hesitant_wrong) | draw_hint(show_6x3=18_6x4=24) | speak('Not quite — what is 6 times 4?')"},
    {"input": {"response": "write down 2?", "step": "what to write in quotient", "phase": "COLLABORATIVE", "streak": 1},
     "output": "classify(hesitant_wrong) | speak('Almost — we found 6 goes in 4 times. So what number do we write?')"},
    # fundamentally_wrong → back to agent-led
    {"input": {"response": "100 times", "step": "how many times does 6 go into 24", "phase": "COLLABORATIVE", "streak": 2},
     "output": "classify(fundamentally_wrong) | shift_control(AGENT_LED) | speak('Let me walk you through this again.')"},
    {"input": {"response": "just write the 7", "step": "what comes after subtraction", "phase": "CHILD_LED", "streak": 1},
     "output": "classify(fundamentally_wrong) | shift_control(AGENT_LED) | speak('Let me show you the bring-down step again.')"},
    # frustration response
    {"input": {"response": "I don't get it", "step": "any", "phase": "AGENT_LED", "streak": 0},
     "output": "classify(frustrated) | speak(\"That's okay. Let's go even slower. I'll do the first part.\")"},
    {"input": {"response": "this is confusing", "step": "any", "phase": "COLLABORATIVE", "streak": 0},
     "output": "classify(frustrated) | shift_control(AGENT_LED) | speak(\"No problem. Let me take over for a bit.\")"},
]

FUNCTION_GEMMA_EXAMPLES = (FUNCTION_GEMMA_EXAMPLES * 5)[:60]

# ── Gemma 3 12B: tutoring steps with drawing commands ────────────────────────

GEMMA12B_EXAMPLES = [
    # Worksheet reading
    {"type": "worksheet_read",
     "image_description": "247 ÷ 6 written with answer 40 r7",
     "output": '[{"dividend": 247, "divisor": 6, "student_answer": "40 r7", "error_type": "incomplete_quotient"}]'},
    {"type": "worksheet_read",
     "image_description": "84 ÷ 4 written with answer 20",
     "output": '[{"dividend": 84, "divisor": 4, "student_answer": "20", "error_type": "missing_digit"}]'},
    {"type": "worksheet_read",
     "image_description": "All problems answered correctly",
     "output": "[]"},
    # Tutoring steps — opening
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "AGENT_LED", "step": "start"},
     "output": {"speech": "Let's work through 247 divided by 6. First — can 6 go into just the 2?",
                "drawing_commands": [{"type": "DRAW_CIRCLE", "target": "first_digit"}]}},
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "AGENT_LED", "step": "first_two_digits"},
     "output": {"speech": "Right — 2 is too small. So we look at the first two digits: 24. How many times does 6 go into 24?",
                "drawing_commands": [{"type": "DRAW_CIRCLE", "target": "first_two_digits"}]}},
    # Tutoring steps — middle
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "COLLABORATIVE", "step": "after_first_digit"},
     "output": {"speech": "Good — we write 4 above the line. Now what do we multiply?",
                "drawing_commands": [
                    {"type": "DRAW_NUMBER", "value": 4, "position": "quotient_0"},
                    {"type": "DRAW_MULTIPLY", "value": 24, "position": "subtract_row_0"},
                    {"type": "DRAW_LINE", "position": "subtract_line_0"}
                ]}},
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "CHILD_LED", "step": "bring_down"},
     "output": {"speech": "You tell me — what comes next after we subtract?",
                "drawing_commands": []}},
    # Hint steps
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "AGENT_LED", "step": "hint_multiplication"},
     "output": {"speech": "Let me show you: 6 times 3 is 18, and 6 times 4 is 24. Which one fits?",
                "drawing_commands": [{"type": "DRAW_HINT", "hint_type": "multiplication_table", "a": 6, "b": 4}]}},
    # Completion
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "CHILD_LED", "step": "final"},
     "output": {"speech": "247 divided by 6 is 41 remainder 1. You worked that out yourself — well done!",
                "drawing_commands": [{"type": "DRAW_REMAINDER", "value": 1}]}},
    # Face analysis
    {"type": "face_analysis",
     "image_description": "child looking at tablet, frowning, chin in hand",
     "output": '{"frustrated": true, "engaged": true}'},
    {"type": "face_analysis",
     "image_description": "child looking at screen, sitting upright, appears focused",
     "output": '{"frustrated": false, "engaged": true}'},
    {"type": "face_analysis",
     "image_description": "child looking away from screen, playing with pencil",
     "output": '{"frustrated": false, "engaged": false}'},
]

GEMMA12B_EXAMPLES = (GEMMA12B_EXAMPLES * 13)[:150]


def write_jsonl(path: str, records: list) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records to {path}")


if __name__ == "__main__":
    write_jsonl("finetune/data/function_gemma_train.jsonl", FUNCTION_GEMMA_EXAMPLES)
    write_jsonl("finetune/data/gemma12b_train.jsonl", GEMMA12B_EXAMPLES)
    print("Data generation complete.")
```

**Step 2: Run**

```bash
cd backend && python finetune/generate_data.py
```
Expected:
```
Wrote 60 records to finetune/data/function_gemma_train.jsonl
Wrote 150 records to finetune/data/gemma12b_train.jsonl
Data generation complete.
```

**Step 3: Commit**

```bash
git add backend/finetune/generate_data.py backend/finetune/data/
git commit -m "feat(tutor): replace fine-tuning data with math tutoring examples"
```

---

## Task 7: Update fine-tuning scripts

**Files:**
- Modify: `backend/finetune/train_function_gemma.py`
- Modify: `backend/finetune/train_gemma12b.py`

Both scripts keep the same structure as before. Only the data formatting and prompt templates change.

**Step 1: Update train_function_gemma.py**

```python
# backend/finetune/train_function_gemma.py
"""
LoRA fine-tune FunctionGemma 270M on response quality classification.
Run on RTX 6000: python finetune/train_function_gemma.py
Output: adapters/function_gemma/
"""
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

BASE_MODEL = "google/functiongemma-270m-it"
DATA_PATH = "finetune/data/function_gemma_train.jsonl"
OUTPUT_DIR = "adapters/function_gemma"


def load_data(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            text = (
                f"<context>{json.dumps(r['input'])}</context>\n"
                f"<action>{r['output']}</action>"
            )
            records.append({"text": text})
    return Dataset.from_list(records)


def tokenize(batch, tokenizer, max_length=256):
    return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_data(DATA_PATH)
    tokenized = dataset.map(lambda b: tokenize(b, tokenizer), batched=True)
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        warmup_steps=10,
        save_strategy="no",
        logging_steps=10,
        fp16=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"FunctionGemma adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

**Step 2: Update train_gemma12b.py**

```python
# backend/finetune/train_gemma12b.py
"""
LoRA fine-tune Gemma 3 12B on math tutoring dialogues + worksheet reading.
Run on RTX 6000: python finetune/train_gemma12b.py
Output: adapters/gemma12b/
"""
import json, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

BASE_MODEL = "google/gemma-3-12b-it"
DATA_PATH = "finetune/data/gemma12b_train.jsonl"
OUTPUT_DIR = "adapters/gemma12b"


def load_data(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "worksheet_read":
                text = (
                    f"<start_of_turn>user\n"
                    f"Read this worksheet image and identify wrong answers.\n"
                    f"Image shows: {r['image_description']}<end_of_turn>\n"
                    f"<start_of_turn>model\n{r['output']}<end_of_turn>"
                )
            elif r.get("type") == "face_analysis":
                text = (
                    f"<start_of_turn>user\n"
                    f"Analyze this student's face: {r['image_description']}<end_of_turn>\n"
                    f"<start_of_turn>model\n{r['output']}<end_of_turn>"
                )
            else:
                text = (
                    f"<start_of_turn>user\n"
                    f"Math tutoring context: {json.dumps(r.get('context', {}))}<end_of_turn>\n"
                    f"<start_of_turn>model\n{json.dumps(r.get('output', {}))}<end_of_turn>"
                )
            records.append({"text": text})
    return Dataset.from_list(records)


def tokenize(batch, tokenizer, max_length=512):
    return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_data(DATA_PATH)
    tokenized = dataset.map(lambda b: tokenize(b, tokenizer), batched=True)
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=20,
        save_strategy="no",
        logging_steps=10,
        fp16=True,
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Gemma 3 12B adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add backend/finetune/train_function_gemma.py backend/finetune/train_gemma12b.py
git commit -m "feat(tutor): update fine-tuning scripts for math tutoring domain"
```

---

## Task 8: Full test suite

**Step 1: Run all tests**

```bash
cd backend
python -m pytest tests/ -v
```

Expected (34 tests total):
```
tests/test_models.py          8 passed
tests/test_session.py        10 passed
tests/test_policy.py          8 passed
tests/test_model_server.py    8 passed
34 passed
```

**Step 2: Fix any failures before proceeding.**

**Step 3: Commit**

```bash
git add -A
git commit -m "chore(tutor): all 34 tests passing, backend pivot complete"
```

---

## Frontend Contract for Teammate

WebSocket URL: `wss://<ngrok-id>.ngrok.io/ws`

**Send worksheet photo:**
```ts
const canvas = document.createElement('canvas');
canvas.width = 1280; canvas.height = 960;
canvas.getContext('2d')!.drawImage(videoEl, 0, 0);  // or from <img>
const b64 = canvas.toDataURL('image/jpeg', 0.85).split(',')[1];
ws.send(JSON.stringify({ type: 'WORKSHEET_PHOTO', image: b64 }));
```

**Send student speech (after Web Speech API transcription):**
```ts
const recognition = new webkitSpeechRecognition();
recognition.onresult = (e) => {
  const text = e.results[0][0].transcript;
  ws.send(JSON.stringify({ type: 'STUDENT_SPEECH', text, t: Date.now() / 1000 }));
};
```

**Send face frame every 3 seconds:**
```ts
setInterval(() => {
  const canvas = document.createElement('canvas');
  canvas.width = 320; canvas.height = 240;
  canvas.getContext('2d')!.drawImage(webcamEl, 0, 0, 320, 240);
  const b64 = canvas.toDataURL('image/jpeg', 0.6).split(',')[1];
  ws.send(JSON.stringify({ type: 'FACE_FRAME', image: b64, t: Date.now() / 1000 }));
}, 3000);
```

**Handle actions (Canvas whiteboard + TTS):**
```ts
ws.onmessage = (e) => {
  const action = JSON.parse(e.data);
  switch (action.type) {
    case 'DRAW_PROBLEM':     drawProblem(action.dividend, action.divisor); break;
    case 'DRAW_NUMBER':      drawNumber(action.value, action.position); break;
    case 'DRAW_CIRCLE':      drawCircle(action.target); break;
    case 'DRAW_MULTIPLY':    drawMultiply(action.value, action.position); break;
    case 'DRAW_LINE':        drawLine(action.position); break;
    case 'DRAW_BRING_DOWN':  drawBringDown(action.digit_index); break;
    case 'DRAW_REMAINDER':   drawRemainder(action.value); break;
    case 'DRAW_HINT':        drawHint(action.hint_type, action.a, action.b); break;
    case 'SPEAK':            speechSynthesis.speak(new SpeechSynthesisUtterance(action.text)); break;
    case 'SHIFT_CONTROL':    setPhase(action.to); break;
    case 'SHOW_SUMMARY':     showSummary(action); break;
  }
};
```

---

## VM Setup (RTX 6000)

```bash
ssh hackathon@34.133.228.240
cd /path/to/InstaLILY_Hackathon/backend
source .venv/bin/activate
python finetune/generate_data.py
python finetune/train_function_gemma.py   # ~20 min
python finetune/train_gemma12b.py         # ~40 min
uvicorn app.main:app --host 0.0.0.0 --port 8000
# separate terminal:
ngrok http 8000
```
