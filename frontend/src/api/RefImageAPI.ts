/**
 * RefImage API Client
 * 
 * HTTP client for communicating with the RefImage backend API.
 * Includes error handling and fallback mechanisms.
 */

import {
  ApiResponse,
  HealthResponse,
  SearchRequest,
  SearchResponse,
  ErrorResponse
} from '../types/api';

class RefImageAPIError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public response?: ErrorResponse
  ) {
    super(message);
    this.name = 'RefImageAPIError';
  }
}

export class RefImageAPI {
  private baseURL: string;
  private timeout: number;

  constructor(baseURL: string = 'http://localhost:8000', timeout: number = 10000) {
    this.baseURL = baseURL;
    this.timeout = timeout;
  }

  /**
   * Make HTTP request with error handling
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({})) as ErrorResponse;
        throw new RefImageAPIError(
          `HTTP ${response.status}: ${errorData.message || response.statusText}`,
          response.status,
          errorData
        );
      }

      const data = await response.json();
      return { data };

    } catch (error) {
      if (error instanceof RefImageAPIError) {
        return { error: error.response || { error: 'APIError', message: error.message } };
      }

      // Network or timeout errors
      return {
        error: {
          error: 'NetworkError',
          message: error instanceof Error ? error.message : 'Unknown network error'
        }
      };
    }
  }

  /**
   * Check API health status
   */
  async healthCheck(): Promise<ApiResponse<HealthResponse>> {
    return this.request<HealthResponse>('/health');
  }

  /**
   * Search for images
   */
  async searchImages(request: SearchRequest): Promise<ApiResponse<SearchResponse>> {
    return this.request<SearchResponse>('/images/search', {
      method: 'POST',
      body: JSON.stringify({
        query: request.query,
        limit: request.limit || 10,
        threshold: request.threshold || 0.0,
        include_metadata: request.include_metadata !== false,
        tags_filter: request.tags_filter,
      }),
    });
  }

  /**
   * Upload an image (placeholder for future implementation)
   */
  async uploadImage(file: File, description?: string, tags?: string[]): Promise<ApiResponse<any>> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (description) {
      formData.append('description', description);
    }
    
    if (tags && tags.length > 0) {
      formData.append('tags', tags.join(','));
    }

    return this.request<any>('/images/upload', {
      method: 'POST',
      body: formData,
      headers: {}, // Remove Content-Type to let browser set it for FormData
    });
  }

  /**
   * Check if API is available
   */
  async isAvailable(): Promise<boolean> {
    try {
      const response = await this.healthCheck();
      return response.data?.status === 'healthy';
    } catch {
      return false;
    }
  }
}

// Export singleton instance
export const refImageAPI = new RefImageAPI();