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
    session.phase = TutoringPhase.AGENT_LED
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
