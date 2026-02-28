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
