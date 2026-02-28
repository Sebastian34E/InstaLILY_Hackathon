# Hackathon Requirements

## The Challenge

Build something that combines:
- A **fine-tuned on-device model** adapted to your task
- **Agentic behavior** — your system decides and acts autonomously
- **Visual input** — camera, video, or screen feeding into the system
- **A genuine reason this runs on-device**

Optional bonus: voice input and/or output.

## Models Available

Gemma family via HuggingFace:
- **Gemma 3** (4B or 12B) — most capable; reasons about images + text; native function calling from 1B up
- **Gemma 3n E4B** — optimized for on-device; handles text, image, audio, video in a single model; 160ms audio chunks
- **Gemma 3 270M** — tiny, designed for fine-tuning on narrow tasks
- **FunctionGemma 270M** — tiny, specifically for function calling; fine-tune on examples of actions
- **PaliGemma 2** — dedicated vision model; fine-tune for detection, segmentation, OCR

## On-Device Justification

Ask: "What becomes possible when the model lives on the device?"

Best projects are things that **only work on-device**:
- Latency physically impossible with cloud round-trip
- Visual data that cannot leave the device
- Offline operation
- Economics that don't work with per-call API pricing

## Fine-Tuning

Pick a Gemma model. Fine-tune to make it yours.
Tools: Hugging Face, Unsloth, Keras, NeMo
Deployment: LM Studio, Ollama, llama.cpp, LiteRT-LM, MLX

## Agentic Behavior

System should take autonomous actions toward goals.
- **FunctionGemma**: define API surface, fine-tune on action examples → outputs right function call
- Pair with larger Gemma 3 for deeper reasoning
- Gemma 3 native function calling (1B+) also works

## Visual Input

Must use visual input in a way that connects to what the agent does.
- Gemma 3 (4B, 12B): reasons about images + text
- Gemma 3n E4B: images + video natively with MobileNet-V5 encoder
- PaliGemma 2: fine-tune for detection, segmentation, OCR

## Voice (Optional Bonus)

- Gemma 3n: native audio processing, 160ms chunks on-device
- Cloud voice providers also fine (flexible on this one)

## Judging Criteria

- Real reason it runs on-device
- Components work together as a system (not just side-by-side)
- Fine-tuning makes it noticeably better at its specific task
- Live demo that actually runs
- Voice adds value when included

## Time Budget (8 hours)

- LoRA fine-tuning on smaller Gemma models: under 1 hour
- Middle of day: wire up agent loop + vision pipeline
- Last 2 hours: integration + testing with real inputs

## Example Projects

1. **Surgical instrument tracker** — camera watches procedure tray, identifies instruments, tracks usage, flags missing/out-of-sequence. On-device: patient video can't leave room, real-time feedback needed.
2. **Industrial anomaly detector** — monitors robotic arm/assembly, spots drift/defects, autonomously flags or adjusts. On-device: sub-50ms control loop, no connectivity.
3. **Workspace inventory tracker** — camera watches physical space, tracks items taken/misplaced, maintains running inventory. On-device: continuous visual feed, persistent state across sessions.

## Resources

- Gemma models & docs: ai.google.dev/gemma
- All models on HuggingFace: huggingface.co/google
- Gemma Cookbook: github.com/google-gemini/gemma-cookbook
- Google AI Edge / LiteRT-LM: github.com/google-ai-edge/LiteRT-LM
- Fine-tuning guide: ai.google.dev/gemma/docs/tune
- FunctionGemma: huggingface.co/google/functiongemma-270m-it
