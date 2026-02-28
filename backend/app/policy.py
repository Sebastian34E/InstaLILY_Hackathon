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
    TutoringPhase.CHILD_LED: TutoringPhase.AGENT_LED,
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
                ShowSummaryAction(problems_done=0, confidence_end=0.0),
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

        if not correct:
            if quality == "fundamentally_wrong":
                session.shift_phase(TutoringPhase.AGENT_LED)
            return actions

        # Check phase upgrade thresholds
        phase_shifted = False
        if session.phase == TutoringPhase.AGENT_LED and session.confidence_streak >= self.agent_led_threshold:
            session.shift_phase(TutoringPhase.COLLABORATIVE)
            actions.append(ShiftControlAction(to="COLLABORATIVE"))
            actions.append(SpeakAction(text="Great work — now you tell me the next step."))
            phase_shifted = True
        elif session.phase == TutoringPhase.COLLABORATIVE and session.confidence_streak >= self.collaborative_threshold:
            session.shift_phase(TutoringPhase.CHILD_LED)
            actions.append(ShiftControlAction(to="CHILD_LED"))
            actions.append(SpeakAction(text="You've got this — take it from here."))
            phase_shifted = True

        # Check problem completion — only if no phase shift happened this call
        if not phase_shifted and quality == "confident_correct" and session.phase == TutoringPhase.CHILD_LED:
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
            try:
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
            except (KeyError, TypeError) as e:
                logger.warning("Skipping malformed drawing command %r: %s", cmd, e)
        return result

    def _is_problem_complete(self, text: str, problem: dict | None) -> bool:
        if problem is None:
            return False
        keywords = {"remainder", "done", "finished", "that's it", "complete", "r"}
        return bool(set(text.lower().split()) & keywords)
