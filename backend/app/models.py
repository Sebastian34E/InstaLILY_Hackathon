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


class StartProblemEvent(BaseModel):
    type: Literal["START_PROBLEM"]
    dividend: int
    divisor: int


Event = Union[WorksheetPhotoEvent, StudentSpeechEvent, FaceFrameEvent, StartProblemEvent]

_EVENT_MAP: dict[str, type] = {
    "WORKSHEET_PHOTO": WorksheetPhotoEvent,
    "STUDENT_SPEECH": StudentSpeechEvent,
    "FACE_FRAME": FaceFrameEvent,
    "START_PROBLEM": StartProblemEvent,
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
    digit: int | None = None           # the digit value being brought down, e.g. 4
    working_number: int | None = None  # combined number at the working row, e.g. 24


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
    to: Literal["AGENT_LED", "COLLABORATIVE", "CHILD_LED"]


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
