# Deployment & Frontend Contract — Design

**Date:** 2026-02-28
**Status:** Approved

---

## Goal

Get the backend running on the RTX 6000 VM with fine-tuned adapters, and produce a
complete handoff document for the frontend teammate.

---

## Deliverables

| Artifact | Location | Purpose |
|---|---|---|
| `deploy.sh` | `backend/deploy.sh` | One-shot deploy + train on VM |
| `start_server.sh` | `backend/start_server.sh` | Restart server after training |
| `FRONTEND_CONTRACT.md` | `docs/FRONTEND_CONTRACT.md` | Teammate spec |

---

## Section 1: deploy.sh

Single bash script run from local machine. Steps in order:

1. **Check VM state** — SSH in, print HuggingFace cache contents, check if `~/math-tutor/` exists
2. **Copy code** — `scp -r backend/ hackathon@34.133.228.240:~/math-tutor/`
3. **Setup environment** — create venv at `~/math-tutor/.venv`, `pip install -r requirements.txt`
4. **Write .env** — create `~/math-tutor/.env` with correct `ADAPTER_PATH`, `BASE_MODEL_PATH`, `FUNCTION_GEMMA_PATH` pointing to VM directories
5. **Launch training** — in order:
   - `python finetune/generate_data.py` (fast, blocking)
   - `nohup python finetune/train_function_gemma.py > logs/train_fg.log 2>&1 &` (~20 min)
   - Wait for FunctionGemma to finish, then:
   - `nohup python finetune/train_gemma12b.py > logs/train_12b.log 2>&1 &` (~40 min)
6. **Print next steps** — remind user to run `start_server.sh` once training completes

Script is idempotent: checks before overwriting, does not kill a running training job.

---

## Section 2: Server Startup

`start_server.sh` does:
1. `source .venv/bin/activate`
2. `uvicorn app.main:app --host 0.0.0.0 --port 8000`
3. In a second terminal: `ngrok http 8000`

The `.env` written by `deploy.sh` ensures `main.py` finds adapters at correct paths on VM.

`app/main.py` must load env vars for:
- `GEMMA_MODEL_PATH` — base Gemma 3 12B weights
- `FUNCTION_GEMMA_PATH` — base FunctionGemma 270M weights
- `GEMMA_ADAPTER_PATH` — LoRA adapter output from `train_gemma12b.py`
- `FUNCTION_GEMMA_ADAPTER_PATH` — LoRA adapter output from `train_function_gemma.py`

---

## Section 3: Frontend Contract Document

`docs/FRONTEND_CONTRACT.md` contains:

1. WebSocket URL and connection instructions
2. TypeScript types for all 3 inbound events (frontend → backend)
3. TypeScript types for all 10 outbound actions (backend → frontend)
4. Canvas coordinate system spec for all drawing position strings
5. Message sequencing rules
6. Minimal `useWebSocket` React hook stub

---

## Out of Scope

- Frontend UI design (teammate handles)
- Canvas rendering implementation (teammate handles)
- Web Speech API integration (teammate handles)
