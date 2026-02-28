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


def test_competency_ratio_no_responses():
    s = TutoringSession()
    assert s.competency_ratio == 0.0


def test_competency_ratio_with_responses():
    s = TutoringSession()
    s.correct_total = 3
    s.total_responses = 4
    assert s.competency_ratio == 0.75


def test_record_response_increments_total():
    s = TutoringSession()
    s.record_response(correct=True)
    s.record_response(correct=False)
    assert s.total_responses == 2
    assert s.correct_total == 1


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
