/**
 * Type definitions for RefImage API.
 * 
 * These types correspond to the Pydantic models in the backend API.
 */

export interface ImageMetadata {
  id: string;
  filename: string;
  description?: string;
  tags: string[];
  file_size: number;
  image_format: string;
  width: number;
  height: number;
  created_at: string;
  updated_at: string;
}

export interface SearchResult {
  image_id: string;
  similarity_score: number;
  metadata?: ImageMetadata;
}

export interface SearchRequest {
  query: string;
  limit?: number;
  threshold?: number;
  include_metadata?: boolean;
  tags_filter?: string[];
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total_results: number;
  search_time_ms: number;
}

export interface HealthResponse {
  status: string;
  components: {
    clip_model: {
      model_name: string;
      status: string;
      device?: string;
      parameters?: number;
    };
    storage: any;
    search_engine: any;
  };
}

export interface ErrorResponse {
  error: string;
  message: string;
}

export interface ApiResponse<T> {
  data?: T;
  error?: ErrorResponse;
}