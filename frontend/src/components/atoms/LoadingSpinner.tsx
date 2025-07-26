/**
 * Loading Spinner Component
 * 
 * A simple, accessible loading spinner with customizable size and color.
 */

import React from 'react';
import './LoadingSpinner.css';

interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  color?: string;
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'medium',
  color = '#2563eb',
  className = '',
}) => {
  return (
    <div 
      className={`loading-spinner loading-spinner--${size} ${className}`}
      role="status"
      aria-label="Loading"
    >
      <div 
        className="loading-spinner__circle"
        style={{ borderTopColor: color }}
      />
      <span className="sr-only">Loading...</span>
    </div>
  );
};