/**
 * Main App Component
 * 
 * Root component that integrates the RefImage gallery with API functionality.
 * Includes search interface, health status, and image gallery display.
 */

import React, { useState } from 'react';
import { useGallery } from './hooks/useGallery';
import { ImageGrid } from './components/organisms/ImageGrid';
import { LoadingSpinner } from './components/atoms/LoadingSpinner';
import { SearchResult } from './types/api';
import './App.css';

const App: React.FC = () => {
  const {
    images,
    loading,
    error,
    apiAvailable,
    healthStatus,
    searchQuery,
    searchTime,
    searchImages,
    retryConnection,
  } = useGallery();

  const [currentQuery, setCurrentQuery] = useState('');
  const [searchLimit, setSearchLimit] = useState(10);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    await searchImages(currentQuery.trim(), searchLimit);
  };

  const handleImageClick = (result: SearchResult) => {
    console.log('Image clicked:', result);
    // In a real app, this might open a modal or navigate to a detail page
  };

  const handleRetry = async () => {
    await retryConnection();
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app__header">
        <div className="app__header-content">
          <h1 className="app__title">RefImage Gallery</h1>
          <p className="app__subtitle">AI-Powered Image Search with CLIP Embeddings</p>
          
          {/* API Status */}
          <div className={`app__status ${apiAvailable ? 'app__status--healthy' : 'app__status--error'}`}>
            <div className="app__status-indicator" />
            <span className="app__status-text">
              {apiAvailable ? 'API Connected' : 'API Unavailable - Using Mock Data'}
            </span>
            {!apiAvailable && (
              <button 
                onClick={handleRetry}
                className="app__retry-button"
                title="Retry API connection"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="23 4 23 10 17 10" />
                  <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
                </svg>
              </button>
            )}
          </div>

          {/* Health Status Details */}
          {healthStatus && (
            <div className="app__health-details">
              <span className="app__health-model">
                Model: {healthStatus.components.clip_model.model_name}
              </span>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="app__main">
        <div className="app__container">
          {/* Search Form */}
          <section className="app__search">
            <form onSubmit={handleSearch} className="app__search-form">
              <div className="app__search-input-group">
                <input
                  type="text"
                  value={currentQuery}
                  onChange={(e) => setCurrentQuery(e.target.value)}
                  placeholder="Search images with natural language (e.g., 'red car', 'sunset beach')..."
                  className="app__search-input"
                  disabled={loading}
                />
                <select
                  value={searchLimit}
                  onChange={(e) => setSearchLimit(Number(e.target.value))}
                  className="app__search-limit"
                  disabled={loading}
                >
                  <option value={5}>5 results</option>
                  <option value={10}>10 results</option>
                  <option value={20}>20 results</option>
                  <option value={50}>50 results</option>
                </select>
              </div>
              <button
                type="submit"
                disabled={loading}
                className="app__search-button"
              >
                {loading ? (
                  <>
                    <LoadingSpinner size="small" color="white" />
                    <span>Searching...</span>
                  </>
                ) : (
                  <>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="11" cy="11" r="8" />
                      <path d="m21 21-4.35-4.35" />
                    </svg>
                    <span>Search</span>
                  </>
                )}
              </button>
            </form>
          </section>

          {/* Search Results Info */}
          {searchQuery && (
            <section className="app__results-info">
              <div className="app__results-summary">
                <span className="app__results-count">
                  {images.length} results {searchQuery && `for "${searchQuery}"`}
                </span>
                <span className="app__search-time">
                  ({searchTime}ms)
                </span>
              </div>
            </section>
          )}

          {/* Image Gallery */}
          <section className="app__gallery">
            <ImageGrid
              results={images}
              loading={loading}
              error={error}
              onImageClick={handleImageClick}
            />
          </section>
        </div>
      </main>
    </div>
  );
};

export default App;