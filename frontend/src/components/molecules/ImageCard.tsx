/**
 * Image Card Component
 * 
 * Displays an individual image with metadata and semantic information.
 * Supports accessibility and responsive design.
 */

import React from 'react';
import { SearchResult } from '../../types/api';
import './ImageCard.css';

interface ImageCardProps {
  result: SearchResult;
  onClick?: (result: SearchResult) => void;
  className?: string;
}

export const ImageCard: React.FC<ImageCardProps> = ({
  result,
  onClick,
  className = '',
}) => {
  const { metadata, similarity_score } = result;

  if (!metadata) {
    return (
      <div className={`image-card image-card--no-metadata ${className}`}>
        <div className="image-card__placeholder">
          <span>No metadata available</span>
        </div>
      </div>
    );
  }

  const handleClick = () => {
    onClick?.(result);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleClick();
    }
  };

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Format similarity score as percentage
  const similarityPercentage = Math.round(similarity_score * 100);

  return (
    <div
      className={`image-card ${onClick ? 'image-card--clickable' : ''} ${className}`}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      tabIndex={onClick ? 0 : -1}
      role={onClick ? 'button' : 'article'}
      aria-label={metadata.description || metadata.filename}
    >
      {/* Image placeholder - in real implementation, this would show the actual image */}
      <div className="image-card__image">
        <div className="image-card__placeholder">
          <div className="image-card__placeholder-icon">
            <svg
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21,15 16,10 5,21" />
            </svg>
          </div>
          <div className="image-card__dimensions">
            {metadata.width} × {metadata.height}
          </div>
        </div>
        
        {/* Similarity score badge */}
        <div className="image-card__similarity">
          {similarityPercentage}%
        </div>
      </div>

      {/* Metadata */}
      <div className="image-card__content">
        <h3 className="image-card__title" title={metadata.filename}>
          {metadata.filename}
        </h3>
        
        {metadata.description && (
          <p className="image-card__description" title={metadata.description}>
            {metadata.description}
          </p>
        )}

        {/* Tags */}
        {metadata.tags.length > 0 && (
          <div className="image-card__tags">
            {metadata.tags.slice(0, 3).map((tag, index) => (
              <span key={index} className="image-card__tag">
                {tag}
              </span>
            ))}
            {metadata.tags.length > 3 && (
              <span className="image-card__tag image-card__tag--more">
                +{metadata.tags.length - 3}
              </span>
            )}
          </div>
        )}

        {/* File info */}
        <div className="image-card__info">
          <span className="image-card__file-size">
            {formatFileSize(metadata.file_size)}
          </span>
          <span className="image-card__format">
            {metadata.image_format}
          </span>
        </div>
      </div>
    </div>
  );
};