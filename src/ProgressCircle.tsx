import React from 'react';

interface ProgressCircleProps {
  label: string;
  percent: number;
}

const ProgressCircle: React.FC<ProgressCircleProps> = ({ label, percent }) => {
  const radius = 50;
  const stroke = 8;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (percent / 100) * circumference;

  return (
    <div className="progress-circle">
      <svg height={radius * 2} width={radius * 2}>
        <circle
          stroke="#eee"
          fill="transparent"
          strokeWidth={stroke}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
        <circle
          className="progress-circle__value"
          stroke="#3498db"
          fill="transparent"
          strokeWidth={stroke}
          strokeDasharray={`${circumference} ${circumference}`}
          style={{ strokeDashoffset }}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
      </svg>
      <div className="progress-circle__label">
        <span>{label}</span>
        <span>{percent}%</span>
      </div>
    </div>
  );
};

export default ProgressCircle;
