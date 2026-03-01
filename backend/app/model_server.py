# backend/app/model_server.py
from __future__ import annotations
import asyncio
import base64
import json
import logging
import re
from io import BytesIO
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

_FALLBACK_TUTORING_STEPS = [
    {"speech": "Let's start. Look at the first digit of the dividend. How many times does the divisor go into it?", "drawing_commands": []},
    {"speech": "Think about how many times the divisor fits. Multiply it and subtract.", "drawing_commands": []},
]


def _compute_division_steps(dividend: int, divisor: int) -> str:
    """Return a human-readable walkthrough of the long division steps."""
    digits = [int(d) for d in str(dividend)]
    steps = []
    current = 0
    quotient_digits = []
    for i, digit in enumerate(digits):
        current = current * 10 + digit
        q = current // divisor
        product = q * divisor
        remainder = current - product
        quotient_digits.append(str(q))
        steps.append(
            f"  Step {i+1}: Bring down {digit} → working number is {current}. "
            f"{current} ÷ {divisor} = {q}. Write {q}. "
            f"{q} × {divisor} = {product}. {current} − {product} = {remainder}."
        )
        current = remainder
    answer = "".join(quotient_digits).lstrip("0") or "0"
    steps.append(f"  Final answer: {answer}, remainder {current}.")
    return "\n".join(steps)


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
        """Read worksheet photo and return the problems visible on it."""
        try:
            img = self._decode_image(image_b64)
            prompt = (
                "This image shows a math worksheet with long division problems.\n"
                "List every problem you can see. For each one return:\n"
                '{"dividend": int, "divisor": int, "student_answer": "string or unknown", "error_type": "needs_guidance"}\n'
                "Return ONLY a valid JSON array."
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
        """
        Classify the student's response for the current problem step.
        The full worked solution is injected so the model can check correctness.
        """
        try:
            problem = context.get("problem") or {}
            dividend = problem.get("dividend", 0)
            divisor = problem.get("divisor", 1)
            phase = context.get("phase", "AGENT_LED")
            history = context.get("history", [])

            solution = _compute_division_steps(dividend, divisor)
            correct_answer = dividend // divisor
            remainder = dividend % divisor

            history_text = ""
            if history:
                lines = [f"  {h['role'].upper()}: {h['text']}" for h in history[-6:]]
                history_text = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

            prompt = (
                f"You are evaluating a student's response during a long division tutoring session.\n\n"
                f"Problem: {dividend} ÷ {divisor}\n"
                f"Correct solution:\n{solution}\n\n"
                f"{history_text}"
                f"Student just said: \"{response}\"\n\n"
                f"Teaching phase: {phase}\n\n"
                "IMPORTANT: You only respond to math-related answers. If the student says something unrelated to math, "
                "reply with quality=hesitant_wrong and redirect them to the problem.\n\n"
                "Decide:\n"
                "- Is the student's response correct or on the right track for the CURRENT step?\n"
                "- Give a short, encouraging 1-sentence reply that confirms, corrects, or redirects.\n"
                "- Do NOT give away the full answer if they're wrong.\n\n"
                "Return ONLY valid JSON (no markdown):\n"
                '{"quality": "confident_correct|hesitant_correct|hesitant_wrong|fundamentally_wrong", '
                '"speech": "your 1-sentence response to the student", "next_draw": []}'
            )
            raw = await self._gemma_generate(prompt=prompt, images=[])
            result = self._parse_json(raw, fallback=None)
            if result is None:
                return self._fallback_classify(response, correct_answer, remainder)
            return result
        except Exception as e:
            logger.warning("classify_response failed: %s", e)
            problem = context.get("problem") or {}
            return self._fallback_classify(
                response,
                problem.get("dividend", 0) // max(problem.get("divisor", 1), 1),
                problem.get("dividend", 0) % max(problem.get("divisor", 1), 1),
            )

    async def generate_tutoring_step(self, problem: dict, phase: Any, history: list, step_index: int = 0) -> dict[str, Any]:
        """
        Generate the next tutoring step.
        step_index = number of correct answers so far for this problem.
        Injects the full worked solution so Gemma narrates concretely.
        """
        try:
            dividend = problem['dividend']
            divisor = problem['divisor']

            # Pre-compute all steps in Python so Gemma never has to do arithmetic
            steps = self.division_steps_list(dividend, divisor)
            total_steps = len(steps)
            current_step = steps[min(step_index, total_steps - 1)]
            next_step = steps[min(step_index + 1, total_steps - 1)] if step_index + 1 < total_steps else None

            history_text = ""
            if history:
                lines = [f"  {h['role'].upper()}: {h['text']}" for h in history[-4:]]
                history_text = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

            phase_str = str(phase).split(".")[-1]

            final_answer = dividend // divisor
            remainder = dividend % divisor
            all_steps_done = step_index >= total_steps

            if all_steps_done:
                # All digits worked through — ask for the assembled final answer
                remainder_text = f" with a remainder of {remainder}" if remainder > 0 else ""
                task = (
                    f"All the steps are done! Ask the student to read off the final answer. "
                    f"The correct answer is {final_answer}{remainder_text}."
                )
            elif step_index == 0:
                task = (
                    f"Introduce the problem {dividend} ÷ {divisor} and ask: "
                    f"\"{current_step['question']}\""
                )
            elif phase_str == "AGENT_LED":
                task = (
                    f"The student just got step {step_index} right. "
                    f"Confirm briefly, then state: \"{current_step['action']}\" "
                    f"and ask: \"{current_step['question']}\""
                )
            elif phase_str == "COLLABORATIVE":
                task = f"Ask the student what comes next. Hint: {current_step['question']}"
            else:
                task = f"Nudge: {current_step['question']}"

            prompt = (
                f"You are a friendly math tutor helping a student (age 10-13) solve {dividend} ÷ {divisor}.\n"
                f"Problem step {step_index + 1} of {total_steps}.\n\n"
                f"Your task: {task}\n\n"
                f"{history_text}"
                "Rules:\n"
                "- Use the EXACT numbers from the task above. Do not invent numbers.\n"
                "- Do NOT repeat anything already said.\n"
                "- 1-2 sentences max. Simple language for a 10-year-old.\n\n"
                "Return ONLY valid JSON (no markdown):\n"
                '{"speech": "your sentence here", "drawing_commands": []}'
            )
            raw = await self._gemma_generate(prompt=prompt, images=[])
            result = self._parse_json(raw, fallback=None)
            if result and result.get("speech"):
                return result
            # Fallback: use the pre-computed step directly
            return {"speech": f"{current_step['action']} {current_step['question']}", "drawing_commands": []}
        except Exception as e:
            logger.warning("generate_tutoring_step failed: %s", e)
            return {"speech": f"Let's work through {dividend} ÷ {divisor} together. What do you think the answer is?", "drawing_commands": []}

    def division_steps_list(self, dividend: int, divisor: int) -> list[dict]:
        """Return each long division step with q digit and canvas position."""
        digits = [int(d) for d in str(dividend)]
        steps = []
        current = 0
        pos_idx = 0  # tracks which quotient_ position to write to
        for i, digit in enumerate(digits):
            current = current * 10 + digit
            q = current // divisor
            product = q * divisor
            remainder = current - product
            if i == 0 and q == 0:
                steps.append({
                    "action": f"{divisor} doesn't go into {current}, bring down the next digit.",
                    "question": f"What do we get when we bring down the next digit?",
                    "q": 0,
                    "position": None,  # no quotient digit to write for leading zero
                })
                current = remainder
                continue
            steps.append({
                "action": f"{divisor} goes into {current} {q} time{'s' if q != 1 else ''}. Write {q} above. Then {q} × {divisor} = {product}. Subtract: {current} − {product} = {remainder}.",
                "question": f"How many times does {divisor} go into {current}?",
                "q": q,
                "position": f"quotient_{pos_idx}",
            })
            pos_idx += 1
            current = remainder
        if not steps:
            q = dividend // divisor
            steps.append({
                "action": f"{divisor} goes into {dividend} exactly {q} times.",
                "question": f"How many times does {divisor} go into {dividend}?",
                "q": q,
                "position": "quotient_0",
            })
        return steps

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _gemma_generate(self, prompt: str, images: list) -> str:
        if self._pipe is None:
            raise RuntimeError("Gemma 3 12B not loaded")

        def _run():
            messages = [{"role": "user", "content": []}]
            for img in images:
                messages[0]["content"].append({"type": "image", "image": img})
            messages[0]["content"].append({"type": "text", "text": prompt})
            out = self._pipe(messages, max_new_tokens=256)
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
            # Strip markdown code fences if present
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            obj_start = raw.find("{")
            arr_start = raw.find("[")
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

    def _fallback_classify(self, response: str, correct_answer: int, remainder: int) -> dict:
        """Numerically check if the student's response matches the correct answer."""
        nums = [int(m) for m in re.findall(r"\d+", response)]
        if correct_answer in nums or (remainder > 0 and remainder in nums):
            return {
                "quality": "confident_correct",
                "next_draw": [],
                "speech": "That's right! Good work.",
            }
        return {
            "quality": "hesitant_wrong",
            "next_draw": [],
            "speech": "Not quite — let's think through this step together.",
        }
