/**
 * Gallery hook with API integration and fallback functionality
 * 
 * Provides state management for image gallery with real API calls
 * and graceful fallback to mock data when API is unavailable.
 */

import { useState, useEffect, useCallback } from 'react';
import { refImageAPI } from '../api/RefImageAPI';
import { SearchResult, HealthResponse } from '../types/api';

// Mock data for fallback when API is unavailable
const mockSearchResults: SearchResult[] = [
  {
    image_id: 'mock-1',
    similarity_score: 0.95,
    metadata: {
      id: 'mock-1',
      filename: 'sample-cat.jpg',
      description: 'A beautiful cat sitting on a windowsill',
      tags: ['cat', 'pet', 'animal', 'indoor'],
      file_size: 2048576,
      image_format: 'JPEG',
      width: 1024,
      height: 768,
      created_at: '2024-01-15T10:30:00Z',
      updated_at: '2024-01-15T10:30:00Z',
    },
  },
  {
    image_id: 'mock-2',
    similarity_score: 0.87,
    metadata: {
      id: 'mock-2',
      filename: 'sunset-beach.jpg',
      description: 'Beautiful sunset over ocean waves',
      tags: ['sunset', 'beach', 'ocean', 'landscape'],
      file_size: 3145728,
      image_format: 'JPEG',
      width: 1920,
      height: 1080,
      created_at: '2024-01-14T18:45:00Z',
      updated_at: '2024-01-14T18:45:00Z',
    },
  },
  {
    image_id: 'mock-3',
    similarity_score: 0.82,
    metadata: {
      id: 'mock-3',
      filename: 'mountain-view.jpg',
      description: 'Snow-capped mountains with clear blue sky',
      tags: ['mountain', 'snow', 'landscape', 'nature'],
      file_size: 4194304,
      image_format: 'JPEG',
      width: 2048,
      height: 1365,
      created_at: '2024-01-13T14:20:00Z',
      updated_at: '2024-01-13T14:20:00Z',
    },
  },
];

interface GalleryState {
  images: SearchResult[];
  loading: boolean;
  error: string | null;
  apiAvailable: boolean;
  healthStatus: HealthResponse | null;
  searchQuery: string;
  searchTime: number;
}

export const useGallery = () => {
  const [state, setState] = useState<GalleryState>({
    images: [],
    loading: false,
    error: null,
    apiAvailable: false,
    healthStatus: null,
    searchQuery: '',
    searchTime: 0,
  });

  /**
   * Check API health and availability
   */
  const checkHealth = useCallback(async () => {
    try {
      const response = await refImageAPI.healthCheck();
      
      if (response.data) {
        setState(prev => ({
          ...prev,
          apiAvailable: true,
          healthStatus: response.data!,
          error: null,
        }));
        return true;
      } else {
        setState(prev => ({
          ...prev,
          apiAvailable: false,
          healthStatus: null,
          error: response.error?.message || 'API health check failed',
        }));
        return false;
      }
    } catch (error) {
      setState(prev => ({
        ...prev,
        apiAvailable: false,
        healthStatus: null,
        error: 'Failed to connect to API',
      }));
      return false;
    }
  }, []);

  /**
   * Search for images with API or fallback to mock data
   */
  const searchImages = useCallback(async (query: string, limit: number = 10) => {
    setState(prev => ({ ...prev, loading: true, error: null, searchQuery: query }));
    
    const startTime = Date.now();
    
    try {
      // Try API first
      const response = await refImageAPI.searchImages({
        query,
        limit,
        include_metadata: true,
      });

      const searchTime = Date.now() - startTime;

      if (response.data) {
        // API call successful
        setState(prev => ({
          ...prev,
          images: response.data!.results,
          loading: false,
          apiAvailable: true,
          searchTime,
        }));
      } else {
        // API call failed, use mock data
        console.warn('API search failed, using mock data:', response.error);
        
        // Filter mock data based on query
        const filteredMockData = query.trim() 
          ? mockSearchResults.filter(result => 
              result.metadata?.description?.toLowerCase().includes(query.toLowerCase()) ||
              result.metadata?.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase()))
            )
          : mockSearchResults;

        setState(prev => ({
          ...prev,
          images: filteredMockData.slice(0, limit),
          loading: false,
          apiAvailable: false,
          error: `API unavailable - showing mock data (${response.error?.message || 'unknown error'})`,
          searchTime,
        }));
      }
    } catch (error) {
      // Fallback to mock data on any error
      console.error('Search failed, using mock data:', error);
      
      const searchTime = Date.now() - startTime;
      const filteredMockData = query.trim() 
        ? mockSearchResults.filter(result => 
            result.metadata?.description?.toLowerCase().includes(query.toLowerCase()) ||
            result.metadata?.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase()))
          )
        : mockSearchResults;

      setState(prev => ({
        ...prev,
        images: filteredMockData.slice(0, limit),
        loading: false,
        apiAvailable: false,
        error: 'API unavailable - showing mock data',
        searchTime,
      }));
    }
  }, []);

  /**
   * Load initial data (empty search to get all images)
   */
  const loadInitialData = useCallback(async () => {
    await searchImages('', 20);
  }, [searchImages]);

  /**
   * Retry API connection
   */
  const retryConnection = useCallback(async () => {
    const isAvailable = await checkHealth();
    if (isAvailable) {
      // Reload current search if API becomes available
      if (state.searchQuery) {
        await searchImages(state.searchQuery);
      } else {
        await loadInitialData();
      }
    }
  }, [checkHealth, searchImages, loadInitialData, state.searchQuery]);

  // Initialize on mount
  useEffect(() => {
    const initialize = async () => {
      await checkHealth();
      await loadInitialData();
    };
    
    initialize();
  }, [checkHealth, loadInitialData]);

  return {
    ...state,
    searchImages,
    checkHealth,
    retryConnection,
    loadInitialData,
  };
};