"""
FastAPI application for RefImage image store and search engine.

This module provides REST API endpoints for image upload,
storage, and semantic search functionality.
"""

import logging
import time
from typing import Callable, Optional
from uuid import UUID

from fastapi import (
    Depends, FastAPI, File, HTTPException,
    Query, UploadFile, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .dsl import DSLError, DSLExecutor
from .models.clip_model import CLIPModel, CLIPModelError
from .models.schemas import (
    DSLQuery,
    DSLResponse,
    ErrorResponse,
    ImageDeleteResponse,
    ImageDetailResponse,
    ImageListRequest,
    ImageListResponse,
    ImageUpdateRequest,
    ImageUpdateResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    UploadResponse,
)
from .search import VectorSearchEngine, VectorSearchError
from .storage import StorageError, StorageManager

logger = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None, config: Optional[Settings] = None
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        settings: Application settings (preferred parameter name)
        config: Application configuration (backward compatibility)

    Returns:
        Configured FastAPI application

    Raises:
        RuntimeError: If initialization fails
    """
    # Support both parameter names for backward compatibility
    app_settings = settings or config
    if app_settings is None:
        app_settings = Settings()

    assert app_settings is not None, "Settings object is required"

    try:
        # Initialize components with app_settings
        clip_model = CLIPModel(app_settings)
        storage_manager = StorageManager(app_settings)
        search_engine = VectorSearchEngine(app_settings)

        # Load existing embeddings into search engine
        embeddings = storage_manager.get_all_embeddings()
        if embeddings:
            search_engine.add_embeddings_batch(embeddings)
            logger.info(
                f"Loaded {len(embeddings)} embeddings into search engine"
            )

        # Create FastAPI app
        app = FastAPI(
            title="RefImage API",
            description="""
🔍 **RefImage**: AI-powered image store and search engine
using CLIP embeddings

## Features
- **Semantic Image Search**: Natural language queries like
  "red car" or "sunset landscape"
- **CLIP Embeddings**: OpenAI's CLIP model for image-text
  understanding
- **Dynamic Search Language (DSL)**: Complex queries with
  AND/OR/NOT operators
- **FAISS Vector Search**: High-performance similarity search
- **Metadata Management**: Tags, descriptions, and file information

## Quick Start
1. Upload images via `/images/upload`
2. Search with text via `/images/search`
3. Use advanced DSL queries via `/search/dsl`

## Example Queries
- Simple: `{"query": "cat sitting on couch"}`
- DSL: `{"query": "AND(TEXT(beach), NOT(TEXT(people)))"}`
            """,
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_tags=[
                {
                    "name": "images",
                    "description": "Image upload and management operations"
                },
                {
                    "name": "search",
                    "description": "Image search operations using text queries"
                },
                {
                    "name": "dsl",
                    "description": "Advanced search using Dynamic Search "
                                   "Language"
                },
                {
                    "name": "health",
                    "description": "Health check and system status"
                }
            ]
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Dependency injection functions
        def get_clip_model() -> CLIPModel:
            """Get CLIP model instance."""
            return clip_model

        def get_storage_manager() -> StorageManager:
            """Get storage manager instance."""
            return storage_manager

        def get_search_engine() -> VectorSearchEngine:
            """Get search engine instance."""
            return search_engine

        def get_settings() -> Settings:
            """Get application settings."""
            return app_settings

        # Include error handlers
        @app.exception_handler(HTTPException)
        async def http_exception_handler(
            request: Request, exc: HTTPException
        ) -> JSONResponse:
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error="HTTPError",
                    message=str(exc.detail),
                    details=None
                ).model_dump(),
            )

        @app.exception_handler(Exception)
        async def general_exception_handler(
            request: Request, exc: Exception
        ) -> JSONResponse:
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="InternalError",
                    message="Internal server error",
                    details=None
                ).model_dump(),
            )

        # Register API endpoints
        _register_endpoints(
            app, get_clip_model, get_storage_manager, get_search_engine
        )

        return app

    except Exception as e:
        error_msg = f"Failed to create application: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def _register_endpoints(
    app: FastAPI,
    get_clip_model: Callable[[], CLIPModel],
    get_storage_manager: Callable[[], StorageManager],
    get_search_engine: Callable[[], VectorSearchEngine],
):
    """Register all API endpoints with the FastAPI app."""

    @app.post(
        "/images/upload",
        response_model=UploadResponse,
        tags=["images"],
        summary="Upload an image",
        description="""
        Upload an image file to the RefImage system. The image will be:
        1. Validated for proper format (JPEG, PNG, etc.)
        2. Stored with metadata (filename, size, dimensions)
        3. Processed through CLIP model for semantic embeddings
        4. Indexed for future semantic search

        **Supported formats**: JPEG, PNG, GIF, BMP, WEBP
        **Max file size**: Configurable (default: 10MB)
        **Tags**: Comma-separated list for easy filtering
        """,
        responses={
            200: {
                "description": "Image uploaded successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "image_id": "550e8400-e29b-41d4-a716-446655440000",
                            "metadata": {
                                "filename": "sunset.jpg",
                                "file_size": 1024000,
                                "width": 1920,
                                "height": 1080,
                                "tags": ["nature", "sunset"]
                            },
                            "processing_time_ms": 250.5
                        }
                    }
                }
            },
            400: {"description": "Invalid file format or empty file"},
            413: {"description": "File too large"},
            500: {"description": "Internal processing error"}
        }
    )
    async def upload_image(
        file: UploadFile = File(
            ...,
            description="Image file to upload (JPEG, PNG, GIF, BMP, WEBP)"
        ),
        description: Optional[str] = Query(
            None,
            description="Optional description for the image",
            max_length=500
        ),
        tags: Optional[str] = Query(
            None,
            description=(
                "Comma-separated tags (e.g., 'nature,sunset,landscape')"
            ),
            max_length=200
        ),
        storage: StorageManager = Depends(get_storage_manager),
        clip: CLIPModel = Depends(get_clip_model),
        search: VectorSearchEngine = Depends(get_search_engine),
    ):
        """
        Upload and process an image.

        Args:
            file: Image file to upload
            description: Optional image description
            tags: Optional comma-separated tags

        Returns:
            Upload response with metadata and processing time

        Raises:
            HTTPException: If upload fails
        """
        start_time = time.time()

        try:
            # Validate file
            if (not file.content_type or
                    not file.content_type.startswith("image/")):
                raise HTTPException(
                    status_code=400, detail="File must be an image"
                )

            # Read file content
            file_content = await file.read()
            if len(file_content) == 0:
                raise HTTPException(status_code=400, detail="Empty file")

            # Parse tags
            tags_list = []
            if tags:
                tags_list = [
                    tag.strip() for tag in tags.split(",") if tag.strip()
                ]

            # Store image and metadata
            metadata = storage.store_image(
                image_data=file_content,
                filename=file.filename or "uploaded_image",
                description=description,
                tags=tags_list,
            )

            # Generate and store embedding
            embedding_vector = clip.encode_image(file_content)

            from .models.schemas import ImageEmbedding

            embedding = ImageEmbedding(
                image_id=metadata.id,
                embedding=embedding_vector.tolist(),
                model_name=clip.settings.clip_model_name,
            )

            storage.store_embedding(embedding)
            search.add_embedding(embedding)

            processing_time = (time.time() - start_time) * 1000

            return UploadResponse(
                image_id=metadata.id,
                metadata=metadata,
                processing_time_ms=processing_time,
            )

        except (StorageError, CLIPModelError, VectorSearchError) as e:
            logger.error(f"Upload error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected upload error: {e}")
            raise HTTPException(status_code=500, detail="Upload failed")

    @app.post(
        "/images/search",
        response_model=SearchResponse,
        tags=["search"],
        summary="Search images with natural language",
        description="""
        Search for images using natural language queries. The system uses
        CLIP embeddings to understand semantic relationships between text
        and images.

        **How it works:**
        1. Your text query is encoded using CLIP
        2. Vector similarity search finds matching images
        3. Results are ranked by semantic similarity score
        4. Optional metadata and tag filtering applied

        **Example queries:**
        - "a red car in the parking lot"
        - "sunset over mountains"
        - "people playing on the beach"
        - "cat sitting on a couch"

        **Filtering:** Use tags_filter to narrow results by specific tags
        **Threshold:** Higher values (0.8+) = more precise,
        lower (0.3+) = broader
        """,
        responses={
            200: {
                "description": "Search completed successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "query": "red car",
                            "results": [
                                {
                                    "image_id": "550e8400-e29b-41d4",
                                    "similarity_score": 0.85,
                                    "metadata": {
                                        "filename": "red_sports_car.jpg",
                                        "tags": ["car", "red", "vehicle"]
                                    }
                                }
                            ],
                            "total_results": 1,
                            "search_time_ms": 45.2
                        }
                    }
                }
            },
            400: {"description": "Invalid search parameters"},
            500: {"description": "Search processing error"}
        }
    )
    async def search_images(
        request: SearchRequest,
        clip: CLIPModel = Depends(get_clip_model),
        storage: StorageManager = Depends(get_storage_manager),
        search: VectorSearchEngine = Depends(get_search_engine),
    ):
        """
        Search for similar images.

        Args:
            request: Search request with query and parameters

        Returns:
            Search response with results and timing

        Raises:
            HTTPException: If search fails
        """
        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = clip.encode_text(request.query)

            # Perform vector search
            search_results = search.search(
                query_embedding=query_embedding,
                k=request.limit,
                threshold=request.threshold,
            )

            # Build response
            results = []
            for image_id_str, similarity_score in search_results:
                image_id = UUID(image_id_str)

                # Get metadata if requested
                metadata = None
                if request.include_metadata:
                    metadata = storage.get_metadata(image_id)

                result = SearchResult(
                    image_id=image_id,
                    similarity_score=similarity_score,
                    metadata=metadata
                )

                # Apply tag filtering if specified
                if request.tags_filter and result.metadata:
                    if not any(
                        tag in result.metadata.tags
                        for tag in request.tags_filter
                    ):
                        continue

                results.append(result)

            # Apply final limit after filtering
            results = results[: request.limit]

            search_time = (time.time() - start_time) * 1000

            return SearchResponse(
                query=request.query,
                results=results,
                total_results=len(results),
                search_time_ms=search_time,
            )

        except (CLIPModelError, VectorSearchError, StorageError) as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
            raise HTTPException(status_code=500, detail="Search failed")

    @app.get(
        "/health",
        tags=["health"],
        summary="System health check",
        description="""
        Check the health and status of all RefImage system components.

        **Checks performed:**
        - CLIP model availability and configuration
        - Storage system connectivity and statistics
        - Vector search engine status and performance
        - Overall system readiness

        Returns detailed information about each component's status
        and performance metrics.
        """,
        responses={
            200: {
                "description": "System is healthy",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "healthy",
                            "timestamp": "2025-07-26T08:30:00Z",
                            "services": {
                                "clip_model": {"status": "ready"},
                                "storage": {"status": "ready", "images": 150},
                                "search_engine": {"status": "ready"}
                            }
                        }
                    }
                }
            },
            503: {"description": "System is unhealthy"}
        }
    )
    async def health_check(
        clip: CLIPModel = Depends(get_clip_model),
        storage: StorageManager = Depends(get_storage_manager),
        search: VectorSearchEngine = Depends(get_search_engine),
    ):
        """Health check endpoint."""
        try:
            return {
                "status": "healthy",
                "components": {
                    "clip_model": clip.get_model_info(),
                    "storage": storage.get_storage_stats(),
                    "search_engine": search.get_stats(),
                },
            }

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(e)}
            )

    # Image CRUD endpoints
    @app.get(
        "/images",
        response_model=ImageListResponse,
        tags=["images"],
        summary="List all images",
        description="""
        Retrieve a paginated list of all images in the system.

        **Features:**
        - Pagination with configurable limit and offset
        - Tag filtering to narrow results
        - Sorting by creation date, filename, or file size
        - Ascending or descending sort order

        **Use cases:**
        - Browse all uploaded images
        - Filter images by specific tags
        - Paginate through large image collections
        - Sort images by various criteria
        """,
        responses={
            200: {
                "description": "Images retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "images": [
                                {
                                    "id": "550e8400-e29b-41d4",
                                    "filename": "sunset.jpg",
                                    "tags": ["nature", "sunset"]
                                }
                            ],
                            "total_count": 25,
                            "limit": 20,
                            "offset": 0,
                            "has_more": True
                        }
                    }
                }
            }
        }
    )
    async def list_images(
        request: ImageListRequest = Depends(),
        storage: StorageManager = Depends(get_storage_manager),
    ):
        """List all images with pagination and filtering."""
        try:
            # Get images from storage with filtering and pagination
            images, total_count = storage.list_images(
                limit=request.limit,
                offset=request.offset,
                tags_filter=request.tags_filter,
                sort_by=request.sort_by,
                sort_order=request.sort_order,
            )

            has_more = (request.offset + len(images)) < total_count

            return ImageListResponse(
                images=images,
                total_count=total_count,
                limit=request.limit,
                offset=request.offset,
                has_more=has_more,
            )

        except Exception as e:
            logger.error(f"Failed to list images: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to list images"
            )

    @app.get(
        "/images/{image_id}",
        response_model=ImageDetailResponse,
        tags=["images"],
        summary="Get image details",
        description="""
        Retrieve detailed information about a specific image.

        **Information included:**
        - Complete metadata (filename, size, dimensions, tags, etc.)
        - Embedding availability status
        - File existence verification

        **Use cases:**
        - View detailed image information
        - Verify image integrity
        - Check embedding status
        """,
        responses={
            200: {"description": "Image details retrieved successfully"},
            404: {"description": "Image not found"}
        }
    )
    async def get_image_details(
        image_id: UUID,
        storage: StorageManager = Depends(get_storage_manager),
    ):
        """Get detailed information about a specific image."""
        try:
            metadata = storage.get_metadata(image_id)
            if not metadata:
                raise HTTPException(status_code=404, detail="Image not found")

            # Check if embedding exists
            has_embedding = storage.has_embedding(image_id)

            # Check if file exists on disk
            file_exists = metadata.file_path.exists()

            return ImageDetailResponse(
                metadata=metadata,
                has_embedding=has_embedding,
                file_exists=file_exists,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get image details: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to get image details"
            )

    @app.delete(
        "/images/{image_id}",
        response_model=ImageDeleteResponse,
        tags=["images"],
        summary="Delete an image",
        description="""
        Permanently delete an image and all associated data.

        **What gets deleted:**
        - Image file from storage
        - Metadata from database
        - CLIP embedding vectors
        - Search index entries

        **Warning:** This operation cannot be undone!

        **Use cases:**
        - Remove unwanted images
        - Clean up storage space
        - Comply with data deletion requests
        """,
        responses={
            200: {"description": "Image deleted successfully"},
            404: {"description": "Image not found"},
            500: {"description": "Deletion failed"}
        }
    )
    async def delete_image(
        image_id: UUID,
        storage: StorageManager = Depends(get_storage_manager),
        search: VectorSearchEngine = Depends(get_search_engine),
    ):
        """Delete an image and all associated data."""
        try:
            # Check if image exists
            metadata = storage.get_metadata(image_id)
            if not metadata:
                raise HTTPException(status_code=404, detail="Image not found")

            # Delete from search index
            try:
                search.remove_embedding(str(image_id))
            except Exception as e:
                logger.warning(f"Failed to remove from search index: {e}")

            # Delete from storage (metadata, embedding, and file)
            success = storage.delete_image(image_id)

            if success:
                return ImageDeleteResponse(
                    image_id=image_id,
                    deleted=True,
                    message="Image deleted successfully"
                )
            else:
                raise HTTPException(
                    status_code=500, detail="Failed to delete image"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete image: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to delete image"
            )

    @app.put(
        "/images/{image_id}",
        response_model=ImageUpdateResponse,
        tags=["Images"],
        summary="Update image metadata",
        description="""
        Update metadata for an existing image.

        **Supported updates:**
        - Description: Human-readable description of the image
        - Tags: List of keyword tags for categorization

        **Field behavior:**
        - Only specified fields are updated
        - Omitted fields remain unchanged
        - Empty string clears the field
        - Tags are automatically cleaned (trimmed, deduplicated)

        **Use cases:**
        - Add descriptive text to uploaded images
        - Update categorization tags
        - Correct or enhance existing metadata
        """,
        responses={
            200: {"description": "Image metadata updated successfully"},
            404: {"description": "Image not found"},
            400: {"description": "Invalid request data"},
            500: {"description": "Internal server error"},
        }
    )
    async def update_image_metadata(
        image_id: UUID,
        request: ImageUpdateRequest,
        storage: StorageManager = Depends(get_storage_manager),
    ) -> ImageUpdateResponse:
        """Update image metadata."""
        try:
            # Attempt to update metadata
            updated_metadata = storage.update_metadata(
                image_id=image_id,
                description=request.description,
                tags=request.tags
            )

            if updated_metadata is None:
                raise HTTPException(
                    status_code=404, detail="Image not found"
                )

            return ImageUpdateResponse(
                image_id=image_id,
                updated=True,
                metadata=updated_metadata,
                message="Metadata updated successfully"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update image metadata: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to update metadata"
            )

    @app.post(
        "/search/dsl",
        response_model=DSLResponse,
        tags=["dsl"],
        summary="Advanced search with Dynamic Search Language",
        description="""
        Execute complex search queries using RefImage's Dynamic Search
        Language (DSL). DSL allows combining multiple search criteria
        with logical operators.

        **DSL Operators:**
        - `TEXT(query)`: Semantic text search
        - `AND(expr1, expr2)`: Both conditions must match
        - `OR(expr1, expr2)`: Either condition must match
        - `NOT(expr)`: Exclude matching results

        **Example queries:**
        - `TEXT(cat)`: Simple text search for cats
        - `AND(TEXT(beach), TEXT(sunset))`: Beach sunset images
        - `OR(TEXT(dog), TEXT(puppy))`: Dogs or puppies
        - `AND(TEXT(car), NOT(TEXT(red)))`: Cars that are not red
        - `AND(OR(TEXT(ocean), TEXT(sea)), NOT(TEXT(people)))`:
          Ocean/sea without people

        **Use cases:**
        - Content filtering and exclusion
        - Multi-criteria search
        - Complex semantic queries
        - Precise result refinement
        """,
        responses={
            200: {
                "description": "DSL query executed successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "query": "AND(TEXT(beach), NOT(TEXT(people)))",
                            "results": [
                                {
                                    "filename": "empty_beach.jpg",
                                    "tags": ["beach", "nature", "peaceful"]
                                }
                            ],
                            "total_count": 1,
                            "query_info": {
                                "type": "dsl",
                                "threshold": 0.3
                            }
                        }
                    }
                }
            },
            400: {"description": "Invalid DSL syntax"},
            500: {"description": "Query execution error"}
        }
    )
    async def search_dsl(
        request: DSLQuery,
        clip: CLIPModel = Depends(get_clip_model),
        search: VectorSearchEngine = Depends(get_search_engine),
        storage: StorageManager = Depends(get_storage_manager),
    ):
        """
        Execute DSL (Dynamic Search Language) query.

        Args:
            request: DSL query request

        Returns:
            Search results with metadata

        Raises:
            HTTPException: If search fails
        """
        try:
            # Create DSL executor
            dsl_executor = DSLExecutor(clip, search, storage)

            # Execute DSL query
            image_ids = dsl_executor.execute_query(
                query_string=request.query,
                limit=request.limit,
                threshold=request.threshold,
            )

            # Get metadata for results
            results = []
            for image_id in image_ids:
                try:
                    metadata = storage.get_metadata(UUID(image_id))
                    if metadata:
                        results.append(metadata)
                except (ValueError, StorageError):
                    # Skip invalid or missing images
                    continue

            return DSLResponse(
                query=request.query,
                results=results,
                total_count=len(results),
                query_info={
                    "query": request.query,
                    "type": "dsl",
                    "threshold": request.threshold,
                },
            )

        except DSLError as e:
            logger.error(f"DSL search error: {e}")
            raise HTTPException(status_code=400, detail=f"DSL error: {e}")
        except Exception as e:
            logger.error(f"DSL search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Legacy support: create default app instance
# This will be removed after migration is complete
def _create_legacy_app():
    """Create legacy app instance for backward compatibility."""
    try:
        return create_app()
    except Exception as e:
        logger.warning(f"Failed to create legacy app: {e}")
        # Return minimal app for development
        return FastAPI(title="RefImage API (Development)")


app = _create_legacy_app()
