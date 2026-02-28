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
