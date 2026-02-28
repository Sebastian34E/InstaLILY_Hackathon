import React from "react";

interface WhiteboardProps {
  dividend: string;
  divisor: string;
  outsideValue: string | null;
  insideValue: string | null;
  activeTarget: "outside" | "inside" | null;
  setupUnlocked?: boolean;
}

const Whiteboard: React.FC<WhiteboardProps> = ({
  dividend,
  divisor,
  outsideValue,
  insideValue,
  activeTarget,
  setupUnlocked = false,
}) => {
  return (
    <div className="whiteboard">
      <h2>Long Division Setup</h2>

      <div className="division-drawing">
        {!setupUnlocked && <p className="setup-prompt">Use the chat box to fill the highlighted spot.</p>}

        <div className="division-row">
          <div className={`divisor-box ${activeTarget === "outside" ? "is-active" : ""}`}>
            {outsideValue ?? (setupUnlocked ? divisor : "?")}
          </div>

          <div className="bracket-box">
            <div className="quotient-line">
              <span className="quotient-placeholder"> </span>
            </div>

            <div className="top-bar" />

            <div className={`dividend-box ${activeTarget === "inside" ? "is-active" : ""}`}>
              {insideValue ?? (setupUnlocked ? dividend : "?")}
            </div>
          </div>
        </div>

        {setupUnlocked && (
          <p className="hint-text">
            Write the quotient on the top line, then show multiply/subtract steps in the work area.
          </p>
        )}
      </div>

      <div className="whiteboard-wheels" aria-hidden="true">
        <span className="whiteboard-wheel" />
        <span className="whiteboard-wheel" />
      </div>
    </div>
  );
};

export default Whiteboard;
