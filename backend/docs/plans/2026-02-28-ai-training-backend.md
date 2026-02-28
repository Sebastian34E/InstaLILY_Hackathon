# AI Training Video Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a FastAPI + WebSocket Python backend that acts as the AI "brain" for an adaptive training video system — receiving events from the frontend, running deterministic policy rules + LLM question generation, and emitting actions back to control the video player.

**Architecture:** A single FastAPI app with a `/ws` WebSocket endpoint handles all real-time communication. An in-memory `SessionState` object tracks learner progress. A deterministic `PolicyEngine` processes state transitions and emits `Action` objects. An `LLMBrain` module wraps a local Ollama instance (fallback to canned questions if unavailable).

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, WebSockets (built-in to FastAPI), Pydantic v2, httpx (for Ollama HTTP calls), pytest + pytest-asyncio

---

## Project Layout (what we're building)

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           ← FastAPI app + WebSocket endpoint
│   ├── models.py         ← Pydantic event/action/state models
│   ├── session.py        ← SessionState class (in-memory store)
│   ├── policy.py         ← PolicyEngine (deterministic rules)
│   └── llm.py            ← LLMBrain (Ollama wrapper + fallback)
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_session.py
│   ├── test_policy.py
│   └── test_llm.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Interface Contract (source of truth)

### Events → backend (from frontend via WebSocket)
```json
{ "type": "VIDEO_TIME_UPDATE", "t": 42.1 }
{ "type": "VIDEO_SEEK_ATTEMPT", "from_t": 40.2, "to_t": 85.0 }
{ "type": "VIDEO_PAUSED", "t": 42.1 }
{ "type": "VIDEO_PLAYED", "t": 42.1 }
{ "type": "ENGAGEMENT", "state": "AWAY", "away_seconds": 6 }
{ "type": "QUIZ_ANSWER", "text": "You check the brakes first", "t_context": 72.0 }
```

### Actions → frontend (from backend via WebSocket)
```json
{ "type": "PAUSE_VIDEO" }
{ "type": "RESUME_VIDEO" }
{ "type": "SEEK_RELATIVE", "delta": -10 }
{ "type": "SHOW_QUIZ", "prompt": "What is the first safety check?" }
{ "type": "SHOW_TOAST", "message": "Rewatching critical section" }
{ "type": "SPEAK", "text": "Not quite. Rewatch this section." }
{ "type": "SHOW_SUMMARY", "score": 0.8, "interventions": 3, "correct": 2 }
```

---

## Task 1: Project scaffold + dependencies

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/tests/__init__.py`

**Step 1: Create requirements.txt**

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0
httpx>=0.27.0
pytest>=8.2.0
pytest-asyncio>=0.23.0
websockets>=12.0
```

**Step 2: Create pyproject.toml**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 3: Create empty __init__ files**

Both `app/__init__.py` and `tests/__init__.py` are empty files.

**Step 4: Install dependencies**

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate   # Windows
# OR: source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

Expected: All packages install without error.

**Step 5: Commit**

```bash
git add backend/requirements.txt backend/pyproject.toml backend/app/__init__.py backend/tests/__init__.py
git commit -m "feat(backend): scaffold project + dependencies"
```

---

## Task 2: Pydantic models for events and actions

**Files:**
- Create: `backend/app/models.py`
- Create: `backend/tests/test_models.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_models.py
import pytest
from app.models import (
    VideoTimeUpdateEvent, VideoSeekAttemptEvent, EngagementEvent,
    QuizAnswerEvent, parse_event,
    PauseVideoAction, ResumeVideoAction, SeekRelativeAction,
    ShowQuizAction, ShowToastAction, SpeakAction, ShowSummaryAction,
)

def test_parse_video_time_update():
    event = parse_event({"type": "VIDEO_TIME_UPDATE", "t": 42.1})
    assert isinstance(event, VideoTimeUpdateEvent)
    assert event.t == 42.1

def test_parse_seek_attempt():
    event = parse_event({"type": "VIDEO_SEEK_ATTEMPT", "from_t": 10.0, "to_t": 80.0})
    assert isinstance(event, VideoSeekAttemptEvent)
    assert event.from_t == 10.0
    assert event.to_t == 80.0

def test_parse_engagement_away():
    event = parse_event({"type": "ENGAGEMENT", "state": "AWAY", "away_seconds": 8})
    assert isinstance(event, EngagementEvent)
    assert event.state == "AWAY"
    assert event.away_seconds == 8

def test_parse_quiz_answer():
    event = parse_event({"type": "QUIZ_ANSWER", "text": "check brakes", "t_context": 55.0})
    assert isinstance(event, QuizAnswerEvent)
    assert event.text == "check brakes"

def test_parse_unknown_raises():
    with pytest.raises(ValueError, match="Unknown event type"):
        parse_event({"type": "UNKNOWN_EVENT"})

def test_seek_relative_action_serializes():
    action = SeekRelativeAction(delta=-10)
    d = action.model_dump()
    assert d["type"] == "SEEK_RELATIVE"
    assert d["delta"] == -10

def test_show_quiz_action_serializes():
    action = ShowQuizAction(prompt="What is step 1?")
    d = action.model_dump()
    assert d["type"] == "SHOW_QUIZ"
    assert d["prompt"] == "What is step 1?"

def test_show_summary_action_serializes():
    action = ShowSummaryAction(score=0.8, interventions=3, correct=2)
    d = action.model_dump()
    assert d == {"type": "SHOW_SUMMARY", "score": 0.8, "interventions": 3, "correct": 2}
```

**Step 2: Run to verify failure**

```bash
cd backend
pytest tests/test_models.py -v
```
Expected: `ImportError` — models.py doesn't exist yet.

**Step 3: Implement models.py**

```python
# backend/app/models.py
from __future__ import annotations
from typing import Literal, Union
from pydantic import BaseModel


# ── Events (frontend → backend) ──────────────────────────────────────────────

class VideoTimeUpdateEvent(BaseModel):
    type: Literal["VIDEO_TIME_UPDATE"]
    t: float


class VideoSeekAttemptEvent(BaseModel):
    type: Literal["VIDEO_SEEK_ATTEMPT"]
    from_t: float
    to_t: float


class VideoPausedEvent(BaseModel):
    type: Literal["VIDEO_PAUSED"]
    t: float


class VideoPlayedEvent(BaseModel):
    type: Literal["VIDEO_PLAYED"]
    t: float


class EngagementEvent(BaseModel):
    type: Literal["ENGAGEMENT"]
    state: Literal["ENGAGED", "AWAY"]
    away_seconds: float = 0.0


class QuizAnswerEvent(BaseModel):
    type: Literal["QUIZ_ANSWER"]
    text: str
    t_context: float


Event = Union[
    VideoTimeUpdateEvent,
    VideoSeekAttemptEvent,
    VideoPausedEvent,
    VideoPlayedEvent,
    EngagementEvent,
    QuizAnswerEvent,
]

_EVENT_MAP = {
    "VIDEO_TIME_UPDATE": VideoTimeUpdateEvent,
    "VIDEO_SEEK_ATTEMPT": VideoSeekAttemptEvent,
    "VIDEO_PAUSED": VideoPausedEvent,
    "VIDEO_PLAYED": VideoPlayedEvent,
    "ENGAGEMENT": EngagementEvent,
    "QUIZ_ANSWER": QuizAnswerEvent,
}


def parse_event(data: dict) -> Event:
    event_type = data.get("type")
    cls = _EVENT_MAP.get(event_type)
    if cls is None:
        raise ValueError(f"Unknown event type: {event_type!r}")
    return cls(**data)


# ── Actions (backend → frontend) ─────────────────────────────────────────────

class PauseVideoAction(BaseModel):
    type: Literal["PAUSE_VIDEO"] = "PAUSE_VIDEO"


class ResumeVideoAction(BaseModel):
    type: Literal["RESUME_VIDEO"] = "RESUME_VIDEO"


class SeekRelativeAction(BaseModel):
    type: Literal["SEEK_RELATIVE"] = "SEEK_RELATIVE"
    delta: float


class ShowQuizAction(BaseModel):
    type: Literal["SHOW_QUIZ"] = "SHOW_QUIZ"
    prompt: str


class ShowToastAction(BaseModel):
    type: Literal["SHOW_TOAST"] = "SHOW_TOAST"
    message: str


class SpeakAction(BaseModel):
    type: Literal["SPEAK"] = "SPEAK"
    text: str


class ShowSummaryAction(BaseModel):
    type: Literal["SHOW_SUMMARY"] = "SHOW_SUMMARY"
    score: float
    interventions: int
    correct: int


Action = Union[
    PauseVideoAction,
    ResumeVideoAction,
    SeekRelativeAction,
    ShowQuizAction,
    ShowToastAction,
    SpeakAction,
    ShowSummaryAction,
]
```

**Step 4: Run tests to verify pass**

```bash
pytest tests/test_models.py -v
```
Expected: 8 tests PASS.

**Step 5: Commit**

```bash
git add backend/app/models.py backend/tests/test_models.py
git commit -m "feat(backend): add Pydantic event/action models with parser"
```

---

## Task 3: SessionState — in-memory learner state

**Files:**
- Create: `backend/app/session.py`
- Create: `backend/tests/test_session.py`

**Step 1: Write failing tests**

```python
# backend/tests/test_session.py
import pytest
from app.session import SessionState, AgentState


def test_initial_state():
    s = SessionState()
    assert s.current_t == 0.0
    assert s.last_allowed_t == 0.0
    assert s.agent_state == AgentState.WATCHING
    assert s.competency_score == 0
    assert s.away_count == 0
    assert s.total_interventions == 0
    assert s.quiz_attempts == 0
    assert s.quiz_correct == 0
    assert s.current_question is None


def test_update_time_advances_last_allowed():
    s = SessionState()
    s.update_time(10.0)
    assert s.current_t == 10.0
    assert s.last_allowed_t == 10.0


def test_update_time_does_not_advance_last_allowed_when_paused():
    s = SessionState()
    s.update_time(10.0)
    s.agent_state = AgentState.PAUSED_FOR_QUIZ
    s.update_time(30.0)  # seek forward while locked
    assert s.current_t == 30.0
    assert s.last_allowed_t == 10.0  # should NOT advance


def test_record_away():
    s = SessionState()
    s.record_away()
    assert s.away_count == 1


def test_record_correct_quiz():
    s = SessionState()
    s.record_quiz_result(correct=True)
    assert s.quiz_correct == 1
    assert s.quiz_attempts == 1
    assert s.competency_score == 1


def test_record_wrong_quiz():
    s = SessionState()
    s.record_quiz_result(correct=False)
    assert s.quiz_correct == 0
    assert s.quiz_attempts == 1
    assert s.competency_score == 0


def test_record_intervention():
    s = SessionState()
    s.record_intervention()
    assert s.total_interventions == 1


def test_competency_ratio():
    s = SessionState()
    assert s.competency_ratio == 0.0
    s.record_quiz_result(correct=True)
    s.record_quiz_result(correct=False)
    assert s.competency_ratio == 0.5
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_session.py -v
```
Expected: `ImportError`.

**Step 3: Implement session.py**

```python
# backend/app/session.py
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field


class AgentState(str, Enum):
    WATCHING = "WATCHING"
    PAUSED_FOR_QUIZ = "PAUSED_FOR_QUIZ"
    REMEDIATION = "REMEDIATION"
    COMPLETED = "COMPLETED"


@dataclass
class SessionState:
    session_id: str = "default"
    current_t: float = 0.0
    last_allowed_t: float = 0.0
    agent_state: AgentState = AgentState.WATCHING
    away_count: int = 0
    total_interventions: int = 0
    quiz_attempts: int = 0
    quiz_correct: int = 0
    competency_score: int = 0
    current_question: str | None = None
    wrong_streak: int = 0

    def update_time(self, t: float) -> None:
        self.current_t = t
        if self.agent_state == AgentState.WATCHING:
            self.last_allowed_t = t

    def record_away(self) -> None:
        self.away_count += 1

    def record_intervention(self) -> None:
        self.total_interventions += 1

    def record_quiz_result(self, correct: bool) -> None:
        self.quiz_attempts += 1
        if correct:
            self.quiz_correct += 1
            self.competency_score += 1
            self.wrong_streak = 0
        else:
            self.wrong_streak += 1

    @property
    def competency_ratio(self) -> float:
        if self.quiz_attempts == 0:
            return 0.0
        return self.quiz_correct / self.quiz_attempts
```

**Step 4: Run tests**

```bash
pytest tests/test_session.py -v
```
Expected: 8 tests PASS.

**Step 5: Commit**

```bash
git add backend/app/session.py backend/tests/test_session.py
git commit -m "feat(backend): add SessionState with AgentState machine"
```

---

## Task 4: PolicyEngine — deterministic rules → actions

**Files:**
- Create: `backend/app/policy.py`
- Create: `backend/tests/test_policy.py`

**Step 1: Write failing tests**

```python
# backend/tests/test_policy.py
import pytest
from app.session import SessionState, AgentState
from app.models import (
    PauseVideoAction, ResumeVideoAction, SeekRelativeAction,
    ShowQuizAction, ShowToastAction, ShowSummaryAction,
    VideoTimeUpdateEvent, VideoSeekAttemptEvent, EngagementEvent, QuizAnswerEvent,
)
from app.policy import PolicyEngine


@pytest.fixture
def engine():
    return PolicyEngine(
        away_threshold_seconds=5,
        seek_forward_threshold=15,
        video_duration=300.0,
    )


@pytest.fixture
def state():
    return SessionState()


def test_time_update_watching_no_actions(engine, state):
    event = VideoTimeUpdateEvent(type="VIDEO_TIME_UPDATE", t=10.0)
    actions = engine.process(event, state, question="")
    assert actions == []
    assert state.current_t == 10.0


def test_seek_too_far_triggers_intervention(engine, state):
    state.update_time(40.0)
    event = VideoSeekAttemptEvent(type="VIDEO_SEEK_ATTEMPT", from_t=40.0, to_t=80.0)
    actions = engine.process(event, state, question="What is step 1?")
    types = [a.type for a in actions]
    assert "SEEK_RELATIVE" in types
    assert "PAUSE_VIDEO" in types
    assert "SHOW_QUIZ" in types
    assert state.agent_state == AgentState.PAUSED_FOR_QUIZ


def test_small_seek_allowed(engine, state):
    state.update_time(40.0)
    event = VideoSeekAttemptEvent(type="VIDEO_SEEK_ATTEMPT", from_t=40.0, to_t=50.0)
    actions = engine.process(event, state, question="")
    assert actions == []


def test_engagement_away_over_threshold_triggers_intervention(engine, state):
    state.update_time(30.0)
    event = EngagementEvent(type="ENGAGEMENT", state="AWAY", away_seconds=8)
    actions = engine.process(event, state, question="What did you learn?")
    types = [a.type for a in actions]
    assert "PAUSE_VIDEO" in types
    assert "SHOW_QUIZ" in types
    assert state.agent_state == AgentState.PAUSED_FOR_QUIZ


def test_engagement_away_under_threshold_no_action(engine, state):
    event = EngagementEvent(type="ENGAGEMENT", state="AWAY", away_seconds=3)
    actions = engine.process(event, state, question="")
    assert actions == []


def test_engagement_engaged_no_action(engine, state):
    event = EngagementEvent(type="ENGAGEMENT", state="ENGAGED", away_seconds=0)
    actions = engine.process(event, state, question="")
    assert actions == []


def test_correct_quiz_resumes(engine, state):
    state.agent_state = AgentState.PAUSED_FOR_QUIZ
    event = QuizAnswerEvent(type="QUIZ_ANSWER", text="correct answer", t_context=30.0)
    actions = engine.process(event, state, question="", answer_correct=True, feedback="Well done!")
    types = [a.type for a in actions]
    assert "RESUME_VIDEO" in types
    assert state.agent_state == AgentState.WATCHING


def test_wrong_quiz_first_time_shows_quiz_again(engine, state):
    state.agent_state = AgentState.PAUSED_FOR_QUIZ
    event = QuizAnswerEvent(type="QUIZ_ANSWER", text="wrong", t_context=30.0)
    actions = engine.process(event, state, question="Try again!", answer_correct=False, feedback="Not quite.")
    types = [a.type for a in actions]
    assert "SHOW_QUIZ" in types
    assert "RESUME_VIDEO" not in types


def test_wrong_quiz_twice_rewinds_and_shows_quiz(engine, state):
    state.agent_state = AgentState.PAUSED_FOR_QUIZ
    state.wrong_streak = 1
    event = QuizAnswerEvent(type="QUIZ_ANSWER", text="wrong again", t_context=30.0)
    actions = engine.process(event, state, question="Retry!", answer_correct=False, feedback="Rewatch.")
    types = [a.type for a in actions]
    assert "SEEK_RELATIVE" in types
    assert "SHOW_TOAST" in types
    assert "SHOW_QUIZ" in types
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_policy.py -v
```
Expected: `ImportError`.

**Step 3: Implement policy.py**

```python
# backend/app/policy.py
from __future__ import annotations
from app.session import SessionState, AgentState
from app.models import (
    Action, Event,
    PauseVideoAction, ResumeVideoAction, SeekRelativeAction,
    ShowQuizAction, ShowToastAction, SpeakAction, ShowSummaryAction,
    VideoTimeUpdateEvent, VideoSeekAttemptEvent,
    VideoPausedEvent, VideoPlayedEvent,
    EngagementEvent, QuizAnswerEvent,
)


class PolicyEngine:
    def __init__(
        self,
        away_threshold_seconds: float = 5.0,
        seek_forward_threshold: float = 15.0,
        video_duration: float = 600.0,
    ):
        self.away_threshold = away_threshold_seconds
        self.seek_threshold = seek_forward_threshold
        self.video_duration = video_duration

    def process(
        self,
        event: Event,
        state: SessionState,
        question: str,
        answer_correct: bool = False,
        feedback: str = "",
    ) -> list[Action]:
        match event:
            case VideoTimeUpdateEvent():
                return self._on_time_update(event, state)
            case VideoSeekAttemptEvent():
                return self._on_seek_attempt(event, state, question)
            case EngagementEvent():
                return self._on_engagement(event, state, question)
            case QuizAnswerEvent():
                return self._on_quiz_answer(event, state, question, answer_correct, feedback)
            case _:
                return []

    # ── handlers ────────────────────────────────────────────────────────────

    def _on_time_update(self, event: VideoTimeUpdateEvent, state: SessionState) -> list[Action]:
        state.update_time(event.t)
        # Check if video reached end
        if event.t >= self.video_duration - 1:
            state.agent_state = AgentState.COMPLETED
            return [ShowSummaryAction(
                score=state.competency_ratio,
                interventions=state.total_interventions,
                correct=state.quiz_correct,
            )]
        return []

    def _on_seek_attempt(
        self, event: VideoSeekAttemptEvent, state: SessionState, question: str
    ) -> list[Action]:
        skip_distance = event.to_t - event.from_t
        if skip_distance > self.seek_threshold:
            state.agent_state = AgentState.PAUSED_FOR_QUIZ
            state.record_intervention()
            state.current_question = question
            return [
                SeekRelativeAction(delta=-10),
                PauseVideoAction(),
                ShowQuizAction(prompt=question),
            ]
        return []

    def _on_engagement(
        self, event: EngagementEvent, state: SessionState, question: str
    ) -> list[Action]:
        if event.state == "AWAY" and event.away_seconds >= self.away_threshold:
            state.record_away()
            state.agent_state = AgentState.PAUSED_FOR_QUIZ
            state.record_intervention()
            state.current_question = question
            return [
                PauseVideoAction(),
                ShowQuizAction(prompt=question),
            ]
        return []

    def _on_quiz_answer(
        self,
        event: QuizAnswerEvent,
        state: SessionState,
        question: str,
        answer_correct: bool,
        feedback: str,
    ) -> list[Action]:
        state.record_quiz_result(correct=answer_correct)

        if answer_correct:
            state.agent_state = AgentState.WATCHING
            state.current_question = None
            actions: list[Action] = [ResumeVideoAction()]
            if feedback:
                actions.append(SpeakAction(text=feedback))
            return actions

        # Wrong answer
        if state.wrong_streak >= 2:
            # Rewind + show quiz again
            state.agent_state = AgentState.REMEDIATION
            state.current_question = question
            return [
                SeekRelativeAction(delta=-10),
                ShowToastAction(message="Rewatching critical section"),
                ShowQuizAction(prompt=question),
            ]
        else:
            # First wrong — just show quiz again
            state.current_question = question
            actions = [ShowQuizAction(prompt=question)]
            if feedback:
                actions.append(SpeakAction(text=feedback))
            return actions
```

**Step 4: Run tests**

```bash
pytest tests/test_policy.py -v
```
Expected: 9 tests PASS.

**Step 5: Commit**

```bash
git add backend/app/policy.py backend/tests/test_policy.py
git commit -m "feat(backend): add PolicyEngine with deterministic state-machine rules"
```

---

## Task 5: LLMBrain — Ollama wrapper with fallback canned questions

**Files:**
- Create: `backend/app/llm.py`
- Create: `backend/tests/test_llm.py`

**Step 1: Write failing tests**

```python
# backend/tests/test_llm.py
import pytest
from unittest.mock import AsyncMock, patch
from app.llm import LLMBrain, EvalResult


@pytest.fixture
def brain():
    return LLMBrain(ollama_url="http://localhost:11434", model="llama3.2")


@pytest.mark.asyncio
async def test_generate_question_returns_string(brain):
    q = await brain.generate_question(topic="forklift safety", t=42.0)
    assert isinstance(q, str)
    assert len(q) > 5


@pytest.mark.asyncio
async def test_evaluate_answer_correct_path(brain):
    with patch.object(brain, "_ollama_complete", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = '{"correct": true, "score": 0.9, "feedback": "Great!"}'
        result = await brain.evaluate_answer(
            question="What is step 1?",
            answer="Check brakes",
            topic="forklift safety",
            t=42.0,
        )
    assert isinstance(result, EvalResult)
    assert result.correct is True
    assert result.score == 0.9
    assert result.feedback == "Great!"


@pytest.mark.asyncio
async def test_evaluate_answer_fallback_on_llm_failure(brain):
    with patch.object(brain, "_ollama_complete", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = Exception("Ollama not running")
        result = await brain.evaluate_answer(
            question="What is step 1?",
            answer="I don't know",
            topic="safety",
            t=10.0,
        )
    assert isinstance(result, EvalResult)
    assert result.correct in (True, False)  # fallback gives a result
    assert result.feedback != ""


@pytest.mark.asyncio
async def test_generate_question_falls_back_on_llm_failure(brain):
    with patch.object(brain, "_ollama_complete", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = Exception("Ollama not running")
        q = await brain.generate_question(topic="safety", t=10.0)
    assert isinstance(q, str)
    assert len(q) > 5
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_llm.py -v
```
Expected: `ImportError`.

**Step 3: Implement llm.py**

```python
# backend/app/llm.py
from __future__ import annotations
import json
import random
from dataclasses import dataclass
import httpx


# ── Canned fallback questions ─────────────────────────────────────────────────

CANNED_QUESTIONS = [
    "What is the most important safety check before starting this procedure?",
    "What should you do if you notice an unexpected hazard?",
    "Which step in this process requires the most attention and why?",
    "What personal protective equipment (PPE) is required here?",
    "What is the correct sequence for completing this task safely?",
]

CANNED_FEEDBACK_CORRECT = [
    "Well done! That is correct.",
    "Great answer! You understood that section.",
    "Correct! Keep going.",
]

CANNED_FEEDBACK_WRONG = [
    "Not quite. Rewatch this section and look for the key safety step.",
    "That is not correct. Pay attention to the process shown in the video.",
    "Incorrect. The video covers this — try again after rewatching.",
]


@dataclass
class EvalResult:
    correct: bool
    score: float
    feedback: str


class LLMBrain:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.ollama_url = ollama_url
        self.model = model

    async def generate_question(self, topic: str, t: float) -> str:
        prompt = (
            f"You are a training assessment AI. "
            f"The learner is watching a training video about '{topic}' at time {t:.0f}s. "
            f"Generate ONE short, specific comprehension question (max 20 words) to check their understanding. "
            f"Return only the question, no explanation."
        )
        try:
            text = await self._ollama_complete(prompt)
            return text.strip().strip('"')
        except Exception:
            return random.choice(CANNED_QUESTIONS)

    async def evaluate_answer(
        self, question: str, answer: str, topic: str, t: float
    ) -> EvalResult:
        prompt = (
            f"Training topic: '{topic}'. Video time: {t:.0f}s.\n"
            f"Question asked: {question}\n"
            f"Learner answered: {answer}\n\n"
            f"Evaluate this answer. Respond with ONLY valid JSON: "
            f'{{\"correct\": true/false, \"score\": 0.0-1.0, \"feedback\": \"one sentence\"}}'
        )
        try:
            raw = await self._ollama_complete(prompt)
            # Extract JSON from response (model may wrap it)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
            return EvalResult(
                correct=bool(data["correct"]),
                score=float(data.get("score", 0.5)),
                feedback=str(data.get("feedback", "")),
            )
        except Exception:
            # Keyword-based fallback for demo
            positive_keywords = {"yes", "correct", "right", "check", "first", "safety", "stop"}
            words = set(answer.lower().split())
            guessed_correct = bool(words & positive_keywords)
            feedback = (
                random.choice(CANNED_FEEDBACK_CORRECT)
                if guessed_correct
                else random.choice(CANNED_FEEDBACK_WRONG)
            )
            return EvalResult(
                correct=guessed_correct,
                score=0.8 if guessed_correct else 0.2,
                feedback=feedback,
            )

    async def _ollama_complete(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json()["response"]
```

**Step 4: Run tests**

```bash
pytest tests/test_llm.py -v
```
Expected: 4 tests PASS.

**Step 5: Commit**

```bash
git add backend/app/llm.py backend/tests/test_llm.py
git commit -m "feat(backend): add LLMBrain with Ollama wrapper and fallback canned questions"
```

---

## Task 6: FastAPI main app + WebSocket endpoint

**Files:**
- Create: `backend/app/main.py`

**Step 1: No separate unit test — this is integration glue. We'll smoke test manually.**

**Step 2: Implement main.py**

```python
# backend/app/main.py
from __future__ import annotations
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.models import parse_event, QuizAnswerEvent, Action
from app.session import SessionState
from app.policy import PolicyEngine
from app.llm import LLMBrain

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Training Video Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (one session for hackathon demo) ───────────────────────────────

TRAINING_TOPIC = "workplace safety and equipment operation"
VIDEO_DURATION = 600.0  # seconds — adjust to your video

session = SessionState()
policy = PolicyEngine(
    away_threshold_seconds=5,
    seek_forward_threshold=15,
    video_duration=VIDEO_DURATION,
)
llm = LLMBrain()


# ── HTTP health check ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_session():
    """Reset learner session — useful for demo resets."""
    global session
    session = SessionState()
    return {"status": "reset"}


# ── WebSocket endpoint ────────────────────────────────────────────────────────

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

            # LLM inputs — precompute before calling policy
            question = ""
            answer_correct = False
            feedback = ""

            if isinstance(event, QuizAnswerEvent):
                eval_result = await llm.evaluate_answer(
                    question=session.current_question or "What did you observe?",
                    answer=event.text,
                    topic=TRAINING_TOPIC,
                    t=event.t_context,
                )
                answer_correct = eval_result.correct
                feedback = eval_result.feedback
                # Pregenerate next question for potential retry
                question = await llm.generate_question(topic=TRAINING_TOPIC, t=event.t_context)
            else:
                # Generate question in background for non-quiz events that might trigger one
                question = await llm.generate_question(topic=TRAINING_TOPIC, t=getattr(event, "t", session.current_t))

            actions = policy.process(
                event=event,
                state=session,
                question=question,
                answer_correct=answer_correct,
                feedback=feedback,
            )

            for action in actions:
                await websocket.send_text(action.model_dump_json())

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
```

**Step 3: Create README.md**

```markdown
# AI Training Video — Backend

FastAPI + WebSocket backend. Acts as the policy engine and LLM brain.

## Setup

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

## Run (dev)

```bash
uvicorn app.main:app --reload --port 8000
```

WebSocket endpoint: `ws://localhost:8000/ws`
Health check: `http://localhost:8000/health`
Reset session: `POST http://localhost:8000/reset`

## Ollama setup (optional — fallback works without it)

```bash
ollama pull llama3.2
ollama serve
```

## Run tests

```bash
pytest -v
```
```

**Step 4: Smoke test — start server**

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Expected: `Uvicorn running on http://127.0.0.1:8000`

In another terminal:
```bash
curl http://localhost:8000/health
```
Expected: `{"status":"ok"}`

**Step 5: Commit**

```bash
git add backend/app/main.py backend/README.md
git commit -m "feat(backend): add FastAPI WebSocket endpoint tying policy + LLM together"
```

---

## Task 7: Full test suite + run all tests

**Step 1: Run entire test suite**

```bash
cd backend
pytest -v
```

Expected: All tests PASS (≥21 tests across 4 test files).

**Step 2: If any fail, fix them before continuing.**

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore(backend): all tests passing, backend complete for hackathon"
```

---

## Quick-Start for Teammate (Frontend Contract)

Your teammate connects to `ws://localhost:8000/ws` and:

1. Sends JSON events (see interface contract above)
2. Listens for JSON actions and executes them

Example browser-side:
```ts
const ws = new WebSocket("ws://localhost:8000/ws");
ws.onmessage = (e) => {
  const action = JSON.parse(e.data);
  handleAction(action); // switch on action.type
};
// Send event:
ws.send(JSON.stringify({ type: "VIDEO_TIME_UPDATE", t: videoRef.current.currentTime }));
```

---

## Policy Rules Summary (for judges demo)

| Trigger | Actions Emitted |
|---|---|
| AWAY ≥ 5s | PAUSE_VIDEO + SHOW_QUIZ |
| Skip > 15s forward | SEEK_RELATIVE(-10) + PAUSE_VIDEO + SHOW_QUIZ |
| Quiz wrong (1st time) | SHOW_QUIZ again |
| Quiz wrong (2nd+ time) | SEEK_RELATIVE(-10) + SHOW_TOAST + SHOW_QUIZ |
| Quiz correct | RESUME_VIDEO + SPEAK(feedback) |
| Video end | SHOW_SUMMARY |
