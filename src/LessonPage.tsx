import React, { useMemo, useState, useRef, useCallback, useEffect } from "react";
import { useLocation } from "react-router-dom";
import Whiteboard from "./Whiteboard";
import WhiteboardCanvas, { type WhiteboardCanvasRef } from "./WhiteboardCanvas";
import { useWebSocket, type BackendAction } from "./hooks/useWebSocket";
import { useSpeech } from "./hooks/useSpeech";
import { useWebcam } from "./hooks/useWebcam";

type Question = {
  label: string;
  dividend: number;
  divisor: number;
  expectedQuotient: number;
};

type PlacementStep = "none" | "outside" | "inside";
type Phase = "AGENT_LED" | "COLLABORATIVE" | "CHILD_LED";

const questions: Question[] = [
  { label: "144 / 12", dividend: 144, divisor: 12, expectedQuotient: 12 },
  { label: "987 / 3", dividend: 987, divisor: 3, expectedQuotient: 329 },
  { label: "105 / 7", dividend: 105, divisor: 7, expectedQuotient: 15 },
  { label: "936 / 8", dividend: 936, divisor: 8, expectedQuotient: 117 },
];

function extractFirstInt(text: string): number | null {
  const m = text.match(/-?\d+/);
  return m ? Number(m[0]) : null;
}

function generateWorksheetImage(dividend: number, divisor: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = 800;
  canvas.height = 600;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, 800, 600);
  ctx.fillStyle = "#333";
  ctx.font = "24px Arial";
  ctx.fillText("Math Worksheet — Long Division", 50, 60);
  ctx.fillStyle = "black";
  ctx.font = "bold 72px monospace";
  ctx.fillText(`${dividend} ÷ ${divisor} = ?`, 100, 320);
  return canvas.toDataURL("image/jpeg", 0.9).replace("data:image/jpeg;base64,", "");
}

const LessonPage: React.FC = () => {
  // ── Problem from router state (set by clicking a card on the home page) ──
  const location = useLocation();
  const routeState = location.state as { dividend?: number; divisor?: number } | null;
  const activeProblem = {
    dividend: routeState?.dividend ?? 247,
    divisor: routeState?.divisor ?? 6,
  };

  // ── Existing fallback state ──────────────────────────────────────────────
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [chatInput, setChatInput] = useState("");
  const [showWhiteboard, setShowWhiteboard] = useState(false);
  const [setupUnlocked, setSetupUnlocked] = useState(false);
  const [placementStep, setPlacementStep] = useState<PlacementStep>("none");
  const [outsideValue, setOutsideValue] = useState<string | null>(null);
  const [insideValue, setInsideValue] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<string>("");
  const [hasCorrectAnswer, setHasCorrectAnswer] = useState(false);
  const [requiresSetup, setRequiresSetup] = useState(false);

  // ── Backend mode state ───────────────────────────────────────────────────
  const [phase, setPhase] = useState<Phase>("AGENT_LED");
  const [sessionActive, setSessionActive] = useState(false);
  const [sessionDone, setSessionDone] = useState(false);
  const [summary, setSummary] = useState<{
    problems_done: number;
    confidence_end: number;
  } | null>(null);

  const whiteboardRef = useRef<WhiteboardCanvasRef>(null);
  const speakRef = useRef<(text: string) => void>(() => {});

  const handleAction = useCallback((action: BackendAction) => {
    switch (action.type) {
      case "SPEAK":
        speakRef.current(action.text as string);
        break;
      case "SHIFT_CONTROL":
        setPhase(action.to as Phase);
        break;
      case "SHOW_SUMMARY":
        setSessionDone(true);
        setSummary({
          problems_done: action.problems_done as number,
          confidence_end: action.confidence_end as number,
        });
        break;
      default:
        if (action.type.startsWith("DRAW_")) {
          whiteboardRef.current?.execute(action);
        }
    }
  }, []);

  const wsUrl = (import.meta.env.VITE_WS_URL as string | undefined) ?? "";
  const { send, wsStatus } = useWebSocket(wsUrl, handleAction);
  const { speak } = useSpeech(send, sessionActive && !sessionDone);
  speakRef.current = speak;
  const { videoRef, hiddenCanvasRef } = useWebcam(
    send,
    sessionActive && !sessionDone
  );

  // Auto-start: send the hardcoded problem image as soon as WS connects
  useEffect(() => {
    if (wsStatus === "connected" && !sessionActive) {
      const image = generateWorksheetImage(activeProblem.dividend, activeProblem.divisor);
      send({ type: "WORKSHEET_PHOTO", image });
      setSessionActive(true);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wsStatus]);

  // ── Fallback (existing) logic ─────────────────────────────────────────────
  const question = questions[currentQuestion];
  const dividendStr = useMemo(() => String(question.dividend), [question.dividend]);
  const divisorStr = useMemo(() => String(question.divisor), [question.divisor]);
  const canGoNext = hasCorrectAnswer && (!requiresSetup || setupUnlocked);
  const isLastQuestion = currentQuestion === questions.length - 1;

  const goToQuestion = (idx: number) => {
    setCurrentQuestion(idx);
    setChatInput("");
    setShowWhiteboard(false);
    setSetupUnlocked(false);
    setPlacementStep("none");
    setOutsideValue(null);
    setInsideValue(null);
    setFeedback("");
    setHasCorrectAnswer(false);
    setRequiresSetup(false);
  };

  const sendChat = () => {
    if (!chatInput.trim()) return;
    const msg = chatInput.trim();
    const typedNumber = extractFirstInt(msg);
    if (typedNumber === null) {
      setFeedback("Not quite. Please enter a number.");
      setChatInput("");
      return;
    }
    if (showWhiteboard && !setupUnlocked && requiresSetup) {
      if (placementStep === "outside") {
        if (typedNumber === question.divisor) {
          setOutsideValue(String(typedNumber));
          setPlacementStep("inside");
          setFeedback("Good. Now type the inside number (dividend) in the chat box.");
        } else {
          setFeedback("Not quite. Enter the divisor for the outside spot.");
        }
        setChatInput("");
        return;
      }
      if (placementStep === "inside") {
        if (typedNumber === question.dividend) {
          setInsideValue(String(typedNumber));
          setSetupUnlocked(true);
          setHasCorrectAnswer(true);
          setFeedback("Great. Setup is correct.");
        } else {
          setFeedback("Not quite. Enter the dividend for the inside spot.");
        }
        setChatInput("");
        return;
      }
    }
    if (typedNumber === question.expectedQuotient) {
      setFeedback("Correct.");
      setHasCorrectAnswer(true);
      setRequiresSetup(false);
      setShowWhiteboard(false);
      setSetupUnlocked(false);
      setPlacementStep("none");
      setOutsideValue(null);
      setInsideValue(null);
      setChatInput("");
      return;
    }
    setFeedback("Not quite. Whiteboard is up. Type the outside number in the chat box.");
    setHasCorrectAnswer(false);
    setRequiresSetup(true);
    setShowWhiteboard(true);
    setSetupUnlocked(false);
    setPlacementStep("outside");
    setOutsideValue(null);
    setInsideValue(null);
    setChatInput("");
  };

  const activeTarget =
    !setupUnlocked
      ? placementStep === "outside" || placementStep === "inside"
        ? placementStep
        : null
      : null;

  // ── Backend mode render ───────────────────────────────────────────────────
  const backendConfigured = wsUrl !== "";

  // If backend is configured but exhausted all retries, show explicit error
  if (backendConfigured && wsStatus === "disconnected") {
    return (
      <div className="lesson-page">
        <div className="lesson-content">
          <div className="session-summary">
            <h2>Backend Unreachable</h2>
            <p>Could not connect to the tutoring server.</p>
            <p>Check that the backend is running and reload the page.</p>
          </div>
        </div>
      </div>
    );
  }

  if (backendConfigured) {
    const phaseCss = phase.toLowerCase().replace(/_/g, "-");
    return (
      <div className="lesson-page">
        {/* Video always in DOM while backend mode active — needed for face frames */}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={sessionActive ? { display: "none" } : undefined}
          className={!sessionActive ? "webcam-preview-large" : ""}
        />
        <canvas
          ref={hiddenCanvasRef}
          width={320}
          height={240}
          style={{ display: "none" }}
        />

        <div className="lesson-content">
          <div className="backend-header">
            <span className={`phase-badge phase-${phaseCss}`}>
              {phase.replace(/_/g, " ")}
            </span>
            {wsStatus === "connecting" && (
              <span className="ws-status">Connecting…</span>
            )}
          </div>

          {sessionDone && summary ? (
            <div className="session-summary">
              <h2>Session Complete!</h2>
              <p>{summary.problems_done} problem{summary.problems_done !== 1 ? "s" : ""} done</p>
              <p>Confidence: {Math.round(summary.confidence_end * 100)}%</p>
            </div>
          ) : !sessionActive ? (
            <div className="capture-area">
              <p className="connecting-overlay">Connecting to tutor…</p>
            </div>
          ) : (
            <>
              <WhiteboardCanvas ref={whiteboardRef} />
              <div className="mic-status">🎤 Listening for your answer…</div>
            </>
          )}
        </div>
      </div>
    );
  }

  // ── Fallback render (existing, unchanged) ────────────────────────────────
  return (
    <div className="lesson-page">
      <div className="lesson-content">
        <h1>Long Division Practice</h1>

        <div className="problem-box-row">
          <div className="monster-icon" aria-label="monster guide" role="img">
            <div className="monster-eye monster-eye-left" />
            <div className="monster-eye monster-eye-right" />
            <div className="monster-mouth" />
          </div>

          <div className="worksheet">
            <div className="worksheet-question">
              <label>{`${currentQuestion + 1}. Solve: ${question.label}`}</label>
            </div>
            {feedback && <div className="feedback">{feedback}</div>}
          </div>
        </div>

        {showWhiteboard && (
          <Whiteboard
            dividend={dividendStr}
            divisor={divisorStr}
            outsideValue={outsideValue}
            insideValue={insideValue}
            activeTarget={activeTarget}
            setupUnlocked={setupUnlocked}
          />
        )}

        <div className="navigation">
          <button
            onClick={() =>
              goToQuestion(Math.min(currentQuestion + 1, questions.length - 1))
            }
            disabled={!canGoNext || isLastQuestion}
          >
            {isLastQuestion ? "All Problems Completed" : "Next Problem"}
          </button>
        </div>
      </div>

      <div className="chat-area chat-area-fixed">
        <div className="chat-entry-row">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendChat()}
            placeholder=""
          />
          <div
            className={`chat-monster ${chatInput.trim().length > 0 ? "is-talking" : ""}`}
            aria-label="chat helper"
            role="img"
          >
            <div className="chat-monster-eye chat-monster-eye-left" />
            <div className="chat-monster-eye chat-monster-eye-right" />
            <div className="chat-monster-mouth" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default LessonPage;
