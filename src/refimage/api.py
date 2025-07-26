"""
FastAPI application for RefImage image store and search engine.

This module provides REST API endpoints for image upload,
storage, and semantic search functionality.
"""

import logging
import time
from typing import Callable, Optional
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .dsl import DSLError, DSLExecutor
from .models.clip_model import CLIPModel, CLIPModelError
from .models.schemas import (
    DSLQuery,
    DSLResponse,
    ErrorResponse,
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
            logger.info(f"Loaded {len(embeddings)} embeddings into search engine")

        # Create FastAPI app
        app = FastAPI(
            title="RefImage API",
            description="Image store and search engine with CLIP embeddings",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
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
        async def http_exception_handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error="HTTPError", message=str(exc.detail)
                ).model_dump(),
            )

        @app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="InternalError", message="Internal server error"
                ).model_dump(),
            )

        # Register API endpoints
        _register_endpoints(app, get_clip_model, get_storage_manager, get_search_engine)

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

    @app.post("/images/upload", response_model=UploadResponse)
    async def upload_image(
        file: UploadFile = File(...),
        description: Optional[str] = Query(None),
        tags: Optional[str] = Query(None),
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
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")

            # Read file content
            file_content = await file.read()
            if len(file_content) == 0:
                raise HTTPException(status_code=400, detail="Empty file")

            # Parse tags
            tags_list = []
            if tags:
                tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

            # Store image and metadata
            metadata = storage.store_image(
                image_data=file_content,
                filename=file.filename,
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

    @app.post("/images/search", response_model=SearchResponse)
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

                result = SearchResult(
                    image_id=image_id, similarity_score=similarity_score
                )

                # Include metadata if requested
                if request.include_metadata:
                    metadata = storage.get_metadata(image_id)
                    result.metadata = metadata

                # Apply tag filtering if specified
                if request.tags_filter and result.metadata:
                    if not any(
                        tag in result.metadata.tags for tag in request.tags_filter
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

    @app.get("/health")
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
                status_code=503, content={"status": "unhealthy", "error": str(e)}
            )

    @app.post("/search/dsl", response_model=DSLResponse)
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
                    metadata = storage.get_image_metadata(UUID(image_id))
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
