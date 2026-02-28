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

        return await asyncio.get_running_loop().run_in_executor(None, _run)

    def _decode_image(self, b64: str) -> Image.Image | None:
        if not b64:
            return None
        try:
            return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            return None

    def _parse_json(self, raw: str, fallback: Any) -> Any:
        try:
            obj_start = raw.find("{")
            arr_start = raw.find("[")
            # Pick whichever opening bracket appears first
            if obj_start == -1 and arr_start == -1:
                return fallback
            if obj_start == -1:
                start_char = "["
            elif arr_start == -1:
                start_char = "{"
            else:
                start_char = "[" if arr_start < obj_start else "{"
            start = raw.find(start_char)
            end_char = "]" if start_char == "[" else "}"
            end = raw.rfind(end_char) + 1
            return json.loads(raw[start:end])
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
