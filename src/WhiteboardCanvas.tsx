import { forwardRef, useImperativeHandle, useRef } from "react";
import type { BackendAction } from "./hooks/useWebSocket";

export interface WhiteboardCanvasRef {
  execute(action: BackendAction): void;
}

const POSITIONS: Record<string, { row: number; col: number }> = {
  quotient_0: { row: 40, col: 200 },
  quotient_1: { row: 40, col: 250 },
  quotient_2: { row: 40, col: 300 },
  subtract_row_0: { row: 130, col: 300 },
  subtract_row_1: { row: 220, col: 300 },
  remainder: { row: 310, col: 350 },
};

const LINE_ROWS: Record<string, number> = {
  subtract_line_0: 150,
  subtract_line_1: 240,
};

const WhiteboardCanvas = forwardRef<WhiteboardCanvasRef>((_, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  function getCtx() {
    return canvasRef.current?.getContext("2d") ?? null;
  }

  useImperativeHandle(ref, () => ({
    execute(action: BackendAction) {
      const c = getCtx();
      if (!c) return;

      switch (action.type) {
        case "DRAW_PROBLEM": {
          const dividend = action.dividend as number;
          const divisor = action.divisor as number;
          c.clearRect(0, 0, 600, 480);

          // Divisor
          c.font = "bold 28px monospace";
          c.fillStyle = "#1a1a1a";
          c.textAlign = "right";
          c.textBaseline = "alphabetic";
          c.fillText(String(divisor), 140, 90);

          // Long division bracket
          c.strokeStyle = "#1a1a1a";
          c.lineWidth = 2;
          c.beginPath();
          c.moveTo(145, 100);
          c.lineTo(145, 65);
          c.lineTo(330, 65);
          c.stroke();

          // Dividend digits
          const digits = String(dividend);
          c.textAlign = "center";
          for (let i = 0; i < digits.length; i++) {
            c.fillText(digits[i], 200 + i * 50, 90);
          }

          // Quotient bar
          c.beginPath();
          c.moveTo(150, 60);
          c.lineTo(330, 60);
          c.stroke();
          break;
        }

        case "DRAW_CIRCLE": {
          const target = action.target as string;
          c.strokeStyle = "#e53e3e";
          c.lineWidth = 2;
          c.beginPath();
          if (target === "first_digit") {
            c.ellipse(200, 82, 22, 18, 0, 0, 2 * Math.PI);
          } else if (target === "first_two_digits") {
            c.ellipse(225, 82, 47, 18, 0, 0, 2 * Math.PI);
          } else if (target === "first_three_digits") {
            c.ellipse(250, 82, 72, 18, 0, 0, 2 * Math.PI);
          }
          c.stroke();
          break;
        }

        case "DRAW_NUMBER":
        case "DRAW_MULTIPLY": {
          const pos = POSITIONS[action.position as string];
          if (!pos) break;
          c.font = "bold 28px monospace";
          c.fillStyle = "#1a1a1a";
          c.textAlign = "right";
          c.textBaseline = "alphabetic";
          c.fillText(String(action.value), pos.col, pos.row);
          break;
        }

        case "DRAW_LINE": {
          const row = LINE_ROWS[action.position as string];
          if (row === undefined) break;
          c.strokeStyle = "#1a1a1a";
          c.lineWidth = 2;
          c.beginPath();
          c.moveTo(150, row);
          c.lineTo(330, row);
          c.stroke();
          break;
        }

        case "DRAW_BRING_DOWN": {
          const idx = action.digit_index as number;
          const fromX = 200 + idx * 50;
          c.strokeStyle = "#e53e3e";
          c.lineWidth = 2;
          c.beginPath();
          c.moveTo(fromX, 100);
          c.quadraticCurveTo(fromX + 25, 155, fromX, 185);
          c.stroke();
          // Arrowhead
          c.fillStyle = "#e53e3e";
          c.beginPath();
          c.moveTo(fromX, 185);
          c.lineTo(fromX - 6, 172);
          c.lineTo(fromX + 6, 172);
          c.closePath();
          c.fill();
          c.fillStyle = "#1a1a1a";
          break;
        }

        case "DRAW_REMAINDER": {
          c.font = "bold 28px monospace";
          c.fillStyle = "#1a1a1a";
          c.textAlign = "left";
          c.textBaseline = "alphabetic";
          c.fillText(`R ${action.value as number}`, 310, 310);
          break;
        }

        case "DRAW_HINT": {
          if (action.hint_type === "show_multiplication_table") {
            const a = action.a as number;
            c.fillStyle = "rgba(255,255,255,0.95)";
            c.fillRect(340, 40, 210, 265);
            c.strokeStyle = "#1a1a1a";
            c.lineWidth = 1;
            c.strokeRect(340, 40, 210, 265);
            c.font = "16px monospace";
            c.fillStyle = "#1a1a1a";
            c.textAlign = "left";
            c.textBaseline = "alphabetic";
            for (let i = 1; i <= 9; i++) {
              c.fillText(`${a} × ${i} = ${a * i}`, 355, 62 + (i - 1) * 27);
            }
          }
          break;
        }

        default:
          break;
      }
    },
  }));

  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={480}
      style={{
        border: "2px solid #9fd1ff",
        borderRadius: "8px",
        background: "white",
        maxWidth: "100%",
      }}
    />
  );
});

WhiteboardCanvas.displayName = "WhiteboardCanvas";
export default WhiteboardCanvas;
