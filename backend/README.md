# Forklift Safety Trainer — Backend

FastAPI + WebSocket backend. Uses FunctionGemma 270M and Gemma 3 12B with LoRA adapters to adaptively control a training video based on learner engagement.

## Setup (RTX 6000 via SSH)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Fine-tune (run once before starting server)

```bash
python finetune/generate_data.py
python finetune/train_function_gemma.py   # ~20 min
python finetune/train_gemma12b.py         # ~40 min
```

## Run server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Expose via ngrok

```bash
ngrok http 8000
```

Frontend connects to: `wss://<ngrok-id>.ngrok.io/ws`

## Environment variables

| Variable | Default | Description |
|---|---|---|
| GEMMA12B_BASE | google/gemma-3-12b-it | HuggingFace model ID |
| GEMMA12B_ADAPTER | adapters/gemma12b | Path to LoRA adapter |
| FUNC_GEMMA_BASE | google/functiongemma-270m-it | HuggingFace model ID |
| FUNC_GEMMA_ADAPTER | adapters/function_gemma | Path to LoRA adapter |
| LOAD_IN_4BIT | true | 4-bit quantization |
| SEEK_THRESHOLD | 15 | Seconds before skip triggers quiz |
| VIDEO_DURATION | 600 | Total video length in seconds |

## Run tests

```bash
pytest -v
```

## Frontend integration (WebSocket at ws://localhost:8000/ws)

Send events every 2 seconds:
```ts
ws.send(JSON.stringify({
  type: 'VISUAL_FRAME',
  video_frame: videoFrameBase64,   // 320x180 JPEG base64
  webcam_frame: webcamFrameBase64, // 320x180 JPEG base64
  t: videoEl.currentTime,
}));
```

Handle actions:
```ts
ws.onmessage = (e) => {
  const action = JSON.parse(e.data);
  switch (action.type) {
    case 'PAUSE_VIDEO':    videoEl.pause(); break;
    case 'RESUME_VIDEO':   videoEl.play(); break;
    case 'SEEK_RELATIVE':  videoEl.currentTime += action.delta; break;
    case 'SHOW_QUIZ':      setQuizPrompt(action.prompt); setShowQuiz(true); break;
    case 'SHOW_TOAST':     showToast(action.message); break;
    case 'SPEAK':          speechSynthesis.speak(new SpeechSynthesisUtterance(action.text)); break;
    case 'SHOW_SUMMARY':   setShowSummary(action); break;
  }
};
```
