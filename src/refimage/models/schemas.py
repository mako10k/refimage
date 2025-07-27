"""
RefImage API v2 Schemas - Pipeline-oriented search architecture.

Complete redesign with new request/response models for:
- Conversions: text-to-dsl, dsl-to-vector tools
- Search: vector, dsl, text unified
- Metadata: Full CRUD
- Images: CRD only
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# ========================================
# CONVERSION SCHEMAS
# ========================================


class TextToVectorRequest(BaseModel):
    """Request for text to vector conversion."""

    text: str = Field(..., description="Text to convert to vector")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate text input."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class TextToVectorResponse(BaseModel):
    """Response for text to vector conversion."""

    text: str = Field(..., description="Original text")
    vector: List[float] = Field(..., description="Generated vector")
    dimension: int = Field(..., description="Vector dimension")
    model: str = Field(..., description="Model used for encoding")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class VectorComponent(BaseModel):
    """Component of a DSL vector."""

    text: str = Field(..., description="Text component")
    vector: List[float] = Field(..., description="Component vector")
    weight: float = Field(..., description="Component weight")
    operation: str = Field(..., description="Operation type (INCLUDE/EXCLUDE)")


class DSLToVectorRequest(BaseModel):
    """Request for DSL to vector conversion."""

    dsl_query: str = Field(..., description="DSL query to convert")
    include_weights: bool = Field(True, description="Include component weights")
    normalize: bool = Field(True, description="Normalize final vector")


class DSLToVectorResponse(BaseModel):
    """Response for DSL to vector conversion."""

    dsl_query: str = Field(..., description="Original DSL query")
    components: List[VectorComponent] = Field(..., description="Vector components")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class DSLExample(BaseModel):
    """DSL syntax example."""

    query: str = Field(..., description="Example DSL query")
    description: str = Field(..., description="Human-readable description")
    explanation: str = Field(..., description="Technical explanation")


class DSLSyntaxResponse(BaseModel):
    """Response for DSL syntax reference."""

    syntax_version: str = Field(..., description="DSL syntax version")
    description: str = Field(..., description="DSL description")
    operators: Dict[str, str] = Field(..., description="Available operators")
    examples: List[DSLExample] = Field(..., description="Usage examples")


# ========================================
# LLM SCHEMAS
# ========================================


class TextToDSLRequest(BaseModel):
    """Request for text to DSL conversion using LLM."""

    text: str = Field(..., description="Natural language query")
    provider: Optional[str] = Field(
        None, description="LLM provider (openai/claude/local)"
    )
    temperature: float = Field(0.3, description="Temperature for LLM generation")
    include_examples: bool = Field(True, description="Include examples in prompt")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate text input."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class TextToDSLResponse(BaseModel):
    """Response for text to DSL conversion."""

    original_text: str = Field(..., description="Original natural language query")
    dsl_query: str = Field(..., description="Generated DSL query")
    explanation: str = Field(..., description="Explanation of conversion")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Model name")
    processing_time_ms: int = Field(..., description="Processing time")


class LLMProviderInfo(BaseModel):
    """LLM provider information."""

    name: str = Field(..., description="Provider name")
    available: bool = Field(..., description="Provider availability")
    model: Optional[str] = Field(None, description="Current model")
    description: str = Field(..., description="Provider description")


class LLMProvidersResponse(BaseModel):
    """Response for LLM providers list."""

    current_provider: str = Field(..., description="Current active provider")
    providers: List[LLMProviderInfo] = Field(..., description="Available providers")


class LLMSwitchRequest(BaseModel):
    """Request to switch LLM provider."""

    provider: str = Field(..., description="Provider to switch to")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider name."""
        valid_providers = ["openai", "claude", "local"]
        if v.lower() not in valid_providers:
            raise ValueError(f"Provider must be one of: {', '.join(valid_providers)}")
        return v.lower()


class LLMSwitchResponse(BaseModel):
    """Response for LLM provider switch."""

    success: bool = Field(..., description="Switch success status")
    previous_provider: str = Field(..., description="Previous provider")
    current_provider: str = Field(..., description="New current provider")
    message: str = Field(..., description="Status message")


# ========================================
# SEARCH SCHEMAS
# ========================================


class VectorSearchRequest(BaseModel):
    """Request for vector search."""

    vector: List[float] = Field(..., description="Query vector")
    limit: int = Field(50, description="Maximum number of results")
    threshold: float = Field(0.0, description="Minimum similarity threshold")

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v: List[float]) -> List[float]:
        """Validate vector."""
        if not v:
            raise ValueError("Vector cannot be empty")
        if len(v) != 512:  # CLIP dimension
            raise ValueError("Vector must have 512 dimensions")
        return v


class DSLSearchRequest(BaseModel):
    """Request for DSL search."""

    query: str = Field(..., description="DSL query")
    limit: int = Field(50, description="Maximum number of results")
    threshold: float = Field(0.3, description="Minimum similarity threshold")


class TextSearchRequest(BaseModel):
    """Request for text search."""

    text: str = Field(..., description="Natural language query")
    limit: int = Field(50, description="Maximum number of results")
    threshold: float = Field(0.3, description="Minimum similarity threshold")
    include_pipeline_debug: bool = Field(
        False, description="Include pipeline debug info"
    )


class SearchResult(BaseModel):
    """Individual search result."""

    image_id: UUID = Field(..., description="Image ID")
    metadata: "ImageMetadata" = Field(..., description="Image metadata")
    score: float = Field(..., description="Similarity score")


class VectorSearchResponse(BaseModel):
    """Response for vector search."""

    vector: List[float] = Field(..., description="Query vector")
    results: List[SearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results")
    search_params: Dict[str, Any] = Field(..., description="Search parameters")
    processing_time_ms: int = Field(..., description="Processing time")


class DSLSearchResponse(BaseModel):
    """Response for DSL search."""

    query: str = Field(..., description="DSL query")
    results: List["ImageMetadata"] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results")
    query_info: Dict[str, Any] = Field(..., description="Query information")
    processing_time_ms: int = Field(..., description="Processing time")


class PipelineDebugInfo(BaseModel):
    """Debug information for search pipeline."""

    text_to_vector: Optional[Dict[str, Any]] = Field(
        None, description="Text to vector stage"
    )
    vector_search: Optional[Dict[str, Any]] = Field(
        None, description="Vector search stage"
    )


class TextSearchResponse(BaseModel):
    """Response for text search."""

    text: str = Field(..., description="Query text")
    results: List[SearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results")
    search_params: Dict[str, Any] = Field(..., description="Search parameters")
    pipeline_debug: Optional[PipelineDebugInfo] = Field(
        None, description="Pipeline debug info"
    )
    processing_time_ms: int = Field(..., description="Processing time")


# ========================================
# METADATA SCHEMAS
# ========================================


class ImageDimensions(BaseModel):
    """Image dimensions."""

    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class ImageMetadata(BaseModel):
    """Image metadata model."""

    id: UUID = Field(..., description="Unique image identifier")
    filename: str = Field(..., description="Original filename")
    description: Optional[str] = Field(None, description="Image description")
    tags: List[str] = Field(default_factory=list, description="Image tags")
    file_size: int = Field(..., description="File size in bytes")
    file_path: Path = Field(..., description="File storage path")
    mime_type: str = Field(..., description="MIME type")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    class Config:
        """Pydantic configuration."""

        json_encoders = {UUID: str, datetime: lambda v: v.isoformat(), Path: str}


class MetadataCreateRequest(BaseModel):
    """Request for creating metadata."""

    filename: str = Field(..., description="Image filename")
    description: Optional[str] = Field(None, description="Image description")
    tags: Optional[List[str]] = Field(None, description="Image tags")
    file_size: int = Field(..., description="File size in bytes")
    dimensions: Optional[ImageDimensions] = Field(None, description="Image dimensions")


class MetadataCreateResponse(BaseModel):
    """Response for metadata creation."""

    metadata: ImageMetadata = Field(..., description="Created metadata")
    created: bool = Field(..., description="Creation success status")
    message: str = Field(..., description="Status message")


class MetadataListResponse(BaseModel):
    """Response for metadata listing."""

    metadata: List[ImageMetadata] = Field(..., description="Metadata list")
    total_count: int = Field(..., description="Total number of records")
    limit: int = Field(..., description="Query limit")
    offset: int = Field(..., description="Query offset")
    sort_by: str = Field(..., description="Sort field")
    sort_order: str = Field(..., description="Sort order")


class MetadataUpdateRequest(BaseModel):
    """Request for updating metadata."""

    description: Optional[str] = Field(None, description="New description")
    tags: Optional[List[str]] = Field(None, description="New tags")

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Clean and validate tags."""
        if v is None:
            return None
        cleaned_tags = [tag.strip() for tag in v if tag.strip()]
        return list(dict.fromkeys(cleaned_tags))  # Remove duplicates


class MetadataUpdateResponse(BaseModel):
    """Response for metadata update."""

    metadata_id: UUID = Field(..., description="Updated metadata ID")
    updated: bool = Field(..., description="Update success status")
    metadata: ImageMetadata = Field(..., description="Updated metadata")
    message: str = Field(..., description="Status message")


class MetadataDeleteResponse(BaseModel):
    """Response for metadata deletion."""

    metadata_id: UUID = Field(..., description="Deleted metadata ID")
    deleted: bool = Field(..., description="Deletion success status")
    message: str = Field(..., description="Status message")


# ========================================
# IMAGE SCHEMAS
# ========================================


class ImageEmbedding(BaseModel):
    """CLIP embedding for an image."""

    image_id: UUID = Field(..., description="Associated image ID")
    embedding: List[float] = Field(..., description="CLIP embedding vector")
    model_name: str = Field(..., description="CLIP model used for embedding")
    created_at: datetime = Field(..., description="Embedding timestamp")

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        """Validate embedding vector."""
        if not v:
            raise ValueError("Embedding cannot be empty")
        if len(v) != 512:  # CLIP dimension
            raise ValueError("Embedding must have 512 dimensions")
        return v

    class Config:
        """Pydantic configuration."""

        json_encoders = {UUID: str, datetime: lambda v: v.isoformat()}


class ImageUploadResponse(BaseModel):
    """Response for image upload."""

    image_id: UUID = Field(..., description="Uploaded image ID")
    metadata: ImageMetadata = Field(..., description="Image metadata")
    upload_success: bool = Field(..., description="Upload success status")
    embedding_generated: bool = Field(..., description="Embedding generation status")
    message: str = Field(..., description="Status message")


class ImageListResponse(BaseModel):
    """Response for image listing."""

    images: List[ImageMetadata] = Field(..., description="Image list")
    total_count: int = Field(..., description="Total number of images")
    limit: int = Field(..., description="Query limit")
    offset: int = Field(..., description="Query offset")
    sort_by: str = Field(..., description="Sort field")
    sort_order: str = Field(..., description="Sort order")


class ImageDeleteResponse(BaseModel):
    """Response for image deletion."""

    image_id: UUID = Field(..., description="Deleted image ID")
    deleted: bool = Field(..., description="Deletion success status")
    message: str = Field(..., description="Status message")


# ========================================
# SYSTEM SCHEMAS
# ========================================


class HealthResponse(BaseModel):
    """System health check response."""

    status: str = Field(..., description="Overall system status")
    components: Dict[str, Any] = Field(..., description="Component status details")


# Forward reference resolution
SearchResult.model_rebuild()
DSLSearchResponse.model_rebuild()
