# backend/finetune/generate_data.py
"""
Generates synthetic training data for:
  1. FunctionGemma 270M (response quality classifier)
  2. Gemma 3 12B (math tutoring dialogue + drawing commands)

Run: python finetune/generate_data.py
Output: finetune/data/function_gemma_train.jsonl
        finetune/data/gemma12b_train.jsonl
"""
import json, os

os.makedirs("finetune/data", exist_ok=True)

# ── FunctionGemma: response quality + action ──────────────────────────────────

FUNCTION_GEMMA_EXAMPLES = [
    # confident_correct → shift toward child-led
    {"input": {"response": "4 times", "step": "how many times does 6 go into 24", "phase": "AGENT_LED", "streak": 2},
     "output": "classify(confident_correct) | shift_control(COLLABORATIVE) | speak('Great work — now you tell me the next step.')"},
    {"input": {"response": "bring down the 7", "step": "next step after subtraction", "phase": "COLLABORATIVE", "streak": 3},
     "output": "classify(confident_correct) | shift_control(CHILD_LED) | draw(bring_down_7)"},
    {"input": {"response": "6 goes into 7 once", "step": "divide into remainder", "phase": "CHILD_LED", "streak": 1},
     "output": "classify(confident_correct) | draw(quotient_1) | speak('Exactly right!')"},
    {"input": {"response": "zero remainder 1", "step": "final remainder", "phase": "CHILD_LED", "streak": 2},
     "output": "classify(confident_correct) | draw(remainder_1) | speak('Perfect — 41 remainder 1.')"},
    # hesitant_correct → stay in current phase
    {"input": {"response": "um... 4 times?", "step": "how many times does 6 go into 24", "phase": "AGENT_LED", "streak": 0},
     "output": "classify(hesitant_correct) | speak('Yes, that is right! 4 times. Good thinking.')"},
    {"input": {"response": "I think bring down the 7?", "step": "next step", "phase": "COLLABORATIVE", "streak": 1},
     "output": "classify(hesitant_correct) | draw(bring_down_7) | speak('That is correct. Well done.')"},
    # hesitant_wrong → gentle nudge, stay
    {"input": {"response": "um... 3 times?", "step": "how many times does 6 go into 24", "phase": "AGENT_LED", "streak": 0},
     "output": "classify(hesitant_wrong) | draw_hint(show_6x3=18_6x4=24) | speak('Not quite — what is 6 times 4?')"},
    {"input": {"response": "write down 2?", "step": "what to write in quotient", "phase": "COLLABORATIVE", "streak": 1},
     "output": "classify(hesitant_wrong) | speak('Almost — we found 6 goes in 4 times. So what number do we write?')"},
    # fundamentally_wrong → back to agent-led
    {"input": {"response": "100 times", "step": "how many times does 6 go into 24", "phase": "COLLABORATIVE", "streak": 2},
     "output": "classify(fundamentally_wrong) | shift_control(AGENT_LED) | speak('Let me walk you through this again.')"},
    {"input": {"response": "just write the 7", "step": "what comes after subtraction", "phase": "CHILD_LED", "streak": 1},
     "output": "classify(fundamentally_wrong) | shift_control(AGENT_LED) | speak('Let me show you the bring-down step again.')"},
    # frustration response
    {"input": {"response": "I don't get it", "step": "any", "phase": "AGENT_LED", "streak": 0},
     "output": "classify(frustrated) | speak(\"That's okay. Let's go even slower. I'll do the first part.\")"},
    {"input": {"response": "this is confusing", "step": "any", "phase": "COLLABORATIVE", "streak": 0},
     "output": "classify(frustrated) | shift_control(AGENT_LED) | speak(\"No problem. Let me take over for a bit.\")"},
]

FUNCTION_GEMMA_EXAMPLES = (FUNCTION_GEMMA_EXAMPLES * 5)[:60]

# ── Gemma 3 12B: tutoring steps with drawing commands ────────────────────────

GEMMA12B_EXAMPLES = [
    # Worksheet reading
    {"type": "worksheet_read",
     "image_description": "247 ÷ 6 written with answer 40 r7",
     "output": '[{"dividend": 247, "divisor": 6, "student_answer": "40 r7", "error_type": "incomplete_quotient"}]'},
    {"type": "worksheet_read",
     "image_description": "84 ÷ 4 written with answer 20",
     "output": '[{"dividend": 84, "divisor": 4, "student_answer": "20", "error_type": "missing_digit"}]'},
    {"type": "worksheet_read",
     "image_description": "All problems answered correctly",
     "output": "[]"},
    # Tutoring steps — opening
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "AGENT_LED", "step": "start"},
     "output": {"speech": "Let's work through 247 divided by 6. First — can 6 go into just the 2?",
                "drawing_commands": [{"type": "DRAW_CIRCLE", "target": "first_digit"}]}},
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "AGENT_LED", "step": "first_two_digits"},
     "output": {"speech": "Right — 2 is too small. So we look at the first two digits: 24. How many times does 6 go into 24?",
                "drawing_commands": [{"type": "DRAW_CIRCLE", "target": "first_two_digits"}]}},
    # Tutoring steps — middle
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "COLLABORATIVE", "step": "after_first_digit"},
     "output": {"speech": "Good — we write 4 above the line. Now what do we multiply?",
                "drawing_commands": [
                    {"type": "DRAW_NUMBER", "value": 4, "position": "quotient_0"},
                    {"type": "DRAW_MULTIPLY", "value": 24, "position": "subtract_row_0"},
                    {"type": "DRAW_LINE", "position": "subtract_line_0"}
                ]}},
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "CHILD_LED", "step": "bring_down"},
     "output": {"speech": "You tell me — what comes next after we subtract?",
                "drawing_commands": []}},
    # Hint steps
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "AGENT_LED", "step": "hint_multiplication"},
     "output": {"speech": "Let me show you: 6 times 3 is 18, and 6 times 4 is 24. Which one fits?",
                "drawing_commands": [{"type": "DRAW_HINT", "hint_type": "multiplication_table", "a": 6, "b": 4}]}},
    # Completion
    {"type": "tutoring_step",
     "context": {"problem": "247 ÷ 6", "phase": "CHILD_LED", "step": "final"},
     "output": {"speech": "247 divided by 6 is 41 remainder 1. You worked that out yourself — well done!",
                "drawing_commands": [{"type": "DRAW_REMAINDER", "value": 1}]}},
    # Face analysis
    {"type": "face_analysis",
     "image_description": "child looking at tablet, frowning, chin in hand",
     "output": '{"frustrated": true, "engaged": true}'},
    {"type": "face_analysis",
     "image_description": "child looking at screen, sitting upright, appears focused",
     "output": '{"frustrated": false, "engaged": true}'},
    {"type": "face_analysis",
     "image_description": "child looking away from screen, playing with pencil",
     "output": '{"frustrated": false, "engaged": false}'},
]

GEMMA12B_EXAMPLES = (GEMMA12B_EXAMPLES * 13)[:150]


def write_jsonl(path: str, records: list) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records to {path}")


if __name__ == "__main__":
    write_jsonl("finetune/data/function_gemma_train.jsonl", FUNCTION_GEMMA_EXAMPLES)
    write_jsonl("finetune/data/gemma12b_train.jsonl", GEMMA12B_EXAMPLES)
    print("Data generation complete.")
