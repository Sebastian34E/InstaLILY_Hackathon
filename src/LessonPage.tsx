import React, { useMemo, useState } from "react";
import Whiteboard from "./Whiteboard";

type Question = {
  label: string;
  dividend: number;
  divisor: number;
  expectedQuotient: number;
};

type PlacementStep = "none" | "outside" | "inside";

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

const LessonPage: React.FC = () => {
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

  const activeTarget = !setupUnlocked ? (placementStep === "outside" || placementStep === "inside" ? placementStep : null) : null;

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
            onClick={() => goToQuestion(Math.min(currentQuestion + 1, questions.length - 1))}
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
