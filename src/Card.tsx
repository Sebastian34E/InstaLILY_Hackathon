import React from 'react';

interface CardProps {
  title: string;
  description?: string;
  variant?: 'division' | 'times' | 'addition';
}

const Card: React.FC<CardProps> = ({ title, description, variant }) => {
  const variantClass = variant ? `card--${variant}` : '';
  return (
    <div className={`card ${variantClass}`.trim()}>
      <h2>{title}</h2>
      {description && <p>{description}</p>}
    </div>
  );
};

export default Card;
