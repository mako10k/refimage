/**
 * Image Grid Component
 * 
 * Responsive grid layout for displaying multiple image cards.
 * Includes loading states and empty states.
 */

import React from 'react';
import { SearchResult } from '../../types/api';
import { ImageCard } from '../molecules/ImageCard';
import { LoadingSpinner } from '../atoms/LoadingSpinner';
import './ImageGrid.css';

interface ImageGridProps {
  results: SearchResult[];
  loading?: boolean;
  error?: string | null;
  onImageClick?: (result: SearchResult) => void;
  className?: string;
}

export const ImageGrid: React.FC<ImageGridProps> = ({
  results,
  loading = false,
  error = null,
  onImageClick,
  className = '',
}) => {
  // Loading state
  if (loading) {
    return (
      <div className={`image-grid image-grid--loading ${className}`}>
        <div className="image-grid__loading">
          <LoadingSpinner size="large" />
          <p className="image-grid__loading-text">Searching images...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className={`image-grid image-grid--error ${className}`}>
        <div className="image-grid__error">
          <div className="image-grid__error-icon">
            <svg
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeJoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <h3 className="image-grid__error-title">Search Error</h3>
          <p className="image-grid__error-message">{error}</p>
        </div>
      </div>
    );
  }

  // Empty state
  if (results.length === 0) {
    return (
      <div className={`image-grid image-grid--empty ${className}`}>
        <div className="image-grid__empty">
          <div className="image-grid__empty-icon">
            <svg
              width="64"
              height="64"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1"
              strokeLinecap="round"
              strokeJoin="round"
            >
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21,15 16,10 5,21" />
            </svg>
          </div>
          <h3 className="image-grid__empty-title">No Images Found</h3>
          <p className="image-grid__empty-message">
            Try searching with different keywords or upload some images to get started.
          </p>
        </div>
      </div>
    );
  }

  // Grid with results
  return (
    <div className={`image-grid ${className}`}>
      <div className="image-grid__container">
        {results.map((result) => (
          <ImageCard
            key={result.image_id}
            result={result}
            onClick={onImageClick}
            className="image-grid__item"
          />
        ))}
      </div>
    </div>
  );
};