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
    total_responses: int = 0
    problems_done: int = 0
    phase_history: list = field(default_factory=list)
    conversation_history: list = field(default_factory=list)
    frustrated: bool = False

    def add_history(self, role: str, text: str) -> None:
        self.conversation_history.append({"role": role, "text": text})
        # Keep last 10 turns to avoid prompt bloat
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def set_current_problem(self, problem: dict) -> None:
        self.current_problem = problem
        self.phase = TutoringPhase.PROBLEM_SELECTED

    def queue_problems(self, problems: list[dict]) -> None:
        self.problems_queue = list(problems)

    def next_problem(self) -> dict | None:
        if not self.problems_queue:
            return None
        return self.problems_queue.pop(0)

    def record_response(self, correct: bool) -> None:
        self.total_responses += 1
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
        self.conversation_history = []
        self.phase = TutoringPhase.UNDERSTANDING_CONFIRMED

    @property
    def competency_ratio(self) -> float:
        if self.total_responses == 0:
            return 0.0
        return self.correct_total / self.total_responses
