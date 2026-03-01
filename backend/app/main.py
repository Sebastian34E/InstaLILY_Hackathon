# backend/app/main.py
from __future__ import annotations
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.models import parse_event
from app.session import TutoringSession
from app.policy import TutoringPolicy
from app.model_server import MathTutorModelServer

logger = logging.getLogger(__name__)

model_server: MathTutorModelServer | None = None
policy: TutoringPolicy | None = None
session: TutoringSession | None = None
worksheet_problems: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_server, policy, session, worksheet_problems
    logger.info("Loading models...")
    model_server = MathTutorModelServer(
        gemma12b_base=os.getenv("GEMMA12B_BASE", "google/gemma-3-12b-it"),
        gemma12b_adapter=os.getenv("GEMMA12B_ADAPTER", "adapters/gemma12b"),
        function_gemma_base=os.getenv("FUNC_GEMMA_BASE", "google/functiongemma-270m-it"),
        function_gemma_adapter=os.getenv("FUNC_GEMMA_ADAPTER", "adapters/function_gemma"),
        load_in_4bit=os.getenv("LOAD_IN_4BIT", "true").lower() == "true",
    )
    policy = TutoringPolicy(
        model_server=model_server,
        agent_led_threshold=int(os.getenv("AGENT_LED_THRESHOLD", "2")),
        collaborative_threshold=int(os.getenv("COLLAB_THRESHOLD", "3")),
    )
    session = TutoringSession()
    # Load worksheet problems from division.json if present (frontend is the source of truth)
    json_path = Path(os.getenv("WORKSHEET_JSON", "worksheets/division.json"))
    try:
        with open(json_path) as f:
            raw = json.load(f)
        worksheet_problems = [
            {"dividend": p["dividend"], "divisor": p["divisor"]}
            for p in raw["worksheet"]["problems"]
        ]
        logger.info("Loaded %d problems from %s", len(worksheet_problems), json_path)
    except FileNotFoundError:
        logger.info("division.json not found — /problems endpoint will return empty list (frontend manages problems)")
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Adaptive Math Tutor Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/problems")
async def get_problems():
    """Return the ordered list of worksheet problems."""
    return worksheet_problems


@app.get("/health")
async def health():
    return {"status": "ok", "phase": session.phase if session else None}


@app.post("/reset")
async def reset_session():
    global session
    session = TutoringSession()
    return {"status": "reset"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            try:
                event = parse_event(data)
            except ValueError as e:
                logger.warning("Unknown event: %s", e)
                continue
            actions = await policy.process(event, session)
            for action in actions:
                await websocket.send_text(action.model_dump_json())
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
