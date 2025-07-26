"""
RefImage API - Pipeline-oriented search architecture.

Complete redesign with:
- Metadata: Full CRUD
- Image data: CRD only (no UPDATE)
- Conversions: text-to-dsl, dsl-to-vector tools
- Search: vector, dsl, text unified
- LLM integration: OpenAI/Local provider switching
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .config import Settings
from .dsl import DSLError, DSLExecutor
from .models.clip_model import CLIPModel, CLIPModelError
from .models.schemas import (  # Conversion schemas; Search schemas; Metadata schemas; Image schemas; System schemas
    DSLExample,
    DSLSearchRequest,
    DSLSearchResponse,
    DSLSyntaxResponse,
    DSLToVectorRequest,
    DSLToVectorResponse,
    HealthResponse,
    ImageDeleteResponse,
    ImageListResponse,
    ImageMetadata,
    ImageUploadResponse,
    MetadataCreateRequest,
    MetadataCreateResponse,
    MetadataDeleteResponse,
    MetadataListResponse,
    MetadataUpdateRequest,
    MetadataUpdateResponse,
    PipelineDebugInfo,
    SearchResult,
    TextSearchRequest,
    TextSearchResponse,
    TextToVectorRequest,
    TextToVectorResponse,
    VectorComponent,
    VectorSearchRequest,
    VectorSearchResponse,
)
from .search import VectorSearchEngine, VectorSearchError
from .storage import StorageError, StorageManager

logger = logging.getLogger(__name__)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create RefImage FastAPI application."""
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="RefImage API",
        description="Pipeline-oriented image search with CLIP embeddings",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Initialize components
    storage_manager = StorageManager(settings)

    clip_model = CLIPModel(settings)

    search_engine = VectorSearchEngine(settings)

    dsl_executor = DSLExecutor(clip_model, search_engine, storage_manager)

    # Dependency providers
    def get_storage_manager() -> StorageManager:
        return storage_manager

    def get_clip_model() -> CLIPModel:
        return clip_model

    def get_search_engine() -> VectorSearchEngine:
        return search_engine

    def get_dsl_executor() -> DSLExecutor:
        return dsl_executor

    # ========================================
    # CONVERSION TOOLS
    # ========================================

    @app.post(
        "/conversions/text-to-vector",
        response_model=TextToVectorResponse,
        tags=["Conversions"],
        summary="Convert text to CLIP vector",
        description="Generate CLIP embedding vector from text input.",
    )
    async def text_to_vector(
        request: TextToVectorRequest,
        clip_model: CLIPModel = Depends(get_clip_model),
    ) -> TextToVectorResponse:
        """Convert text to CLIP embedding vector."""
        try:
            vector = clip_model.encode_text(request.text)

            return TextToVectorResponse(
                text=request.text,
                vector=vector.tolist(),
                dimension=len(vector),
                model=clip_model.model_name,
                processing_time_ms=0,  # TODO: measure actual time
            )
        except CLIPModelError as e:
            logger.error(f"CLIP model error: {e}")
            raise HTTPException(status_code=500, detail=f"CLIP model error: {e}")
        except Exception as e:
            logger.error(f"Text to vector conversion failed: {e}")
            raise HTTPException(status_code=500, detail="Conversion failed")

    @app.post(
        "/conversions/dsl-to-vector",
        response_model=DSLToVectorResponse,
        tags=["Conversions"],
        summary="Convert DSL to vectors",
        description="Parse DSL query and generate component vectors.",
    )
    async def dsl_to_vector(
        request: DSLToVectorRequest,
        clip_model: CLIPModel = Depends(get_clip_model),
        dsl_executor: DSLExecutor = Depends(get_dsl_executor),
    ) -> DSLToVectorResponse:
        """Convert DSL query to component vectors."""
        try:
            # Parse DSL query
            # TODO: Implement proper DSL component extraction

            # For now, treat as simple text query
            vector = clip_model.encode_text(request.dsl_query)

            return DSLToVectorResponse(
                dsl_query=request.dsl_query,
                components=[
                    VectorComponent(
                        text=request.dsl_query,
                        vector=vector.tolist(),
                        weight=1.0,
                        operation="INCLUDE",
                    )
                ],
                processing_time_ms=0,  # TODO: measure actual time
            )
        except DSLError as e:
            logger.error(f"DSL parsing error: {e}")
            raise HTTPException(status_code=400, detail=f"DSL error: {e}")
        except CLIPModelError as e:
            logger.error(f"CLIP model error: {e}")
            raise HTTPException(status_code=500, detail=f"CLIP model error: {e}")
        except Exception as e:
            logger.error(f"DSL to vector conversion failed: {e}")
            raise HTTPException(status_code=500, detail="Conversion failed")

    @app.get(
        "/conversions/dsl-syntax",
        response_model=DSLSyntaxResponse,
        tags=["Conversions"],
        summary="Get DSL syntax reference",
        description="Retrieve DSL syntax documentation and examples.",
    )
    async def get_dsl_syntax() -> DSLSyntaxResponse:
        """Get DSL syntax reference."""
        return DSLSyntaxResponse(
            syntax_version="1.0",
            description="RefImage DSL for complex search queries",
            operators={
                "AND": "Logical AND - all conditions must match",
                "OR": "Logical OR - any condition can match",
                "NOT": "Logical NOT - exclude matching results",
                "^": "Weight operator - adjust relevance (0.0-1.0)",
                "#": "Tag filter - match specific tags",
            },
            examples=[
                DSLExample(
                    query="cat #pet",
                    description="Find cats with pet tag",
                    explanation="Text 'cat' AND tag filter 'pet'",
                ),
                DSLExample(
                    query="beach sunset NOT person",
                    description="Beach sunset without people",
                    explanation="Text 'beach sunset' excluding 'person'",
                ),
                DSLExample(
                    query="red car^0.8 OR blue car^0.6",
                    description="Weighted color preferences",
                    explanation="Red cars (weight 0.8) OR blue cars (weight 0.6)",
                ),
            ],
        )

    # ========================================
    # SEARCH EXECUTION
    # ========================================

    @app.post(
        "/search/vector",
        response_model=VectorSearchResponse,
        tags=["Search"],
        summary="Vector similarity search",
        description="Search images using pre-computed vectors.",
    )
    async def search_vector(
        request: VectorSearchRequest,
        search_engine: VectorSearchEngine = Depends(get_search_engine),
        storage: StorageManager = Depends(get_storage_manager),
    ) -> VectorSearchResponse:
        """Execute vector similarity search."""
        try:
            # Perform vector search
            import numpy as np

            vector_array = np.array(request.vector)
            results = search_engine.search(
                query_embedding=vector_array,
                k=request.limit,
                threshold=request.threshold,
            )

            # Get metadata for results
            search_results = []
            for image_id, score in results:
                try:
                    metadata = storage.get_metadata(UUID(image_id))
                    if metadata:
                        search_results.append(
                            SearchResult(
                                image_id=UUID(image_id), metadata=metadata, score=score
                            )
                        )
                except (ValueError, StorageError):
                    continue

            return VectorSearchResponse(
                vector=request.vector,
                results=search_results,
                total_count=len(search_results),
                search_params={
                    "limit": request.limit,
                    "threshold": request.threshold,
                },
                processing_time_ms=0,  # TODO: measure actual time
            )

        except VectorSearchError as e:
            logger.error(f"Vector search error: {e}")
            raise HTTPException(status_code=500, detail=f"Search error: {e}")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise HTTPException(status_code=500, detail="Search failed")

    @app.post(
        "/search/dsl",
        response_model=DSLSearchResponse,
        tags=["Search"],
        summary="DSL query search",
        description="Execute complex search using DSL syntax.",
    )
    async def search_dsl(
        request: DSLSearchRequest,
        dsl_executor: DSLExecutor = Depends(get_dsl_executor),
        clip_model: CLIPModel = Depends(get_clip_model),
        search_engine: VectorSearchEngine = Depends(get_search_engine),
        storage: StorageManager = Depends(get_storage_manager),
    ) -> DSLSearchResponse:
        """Execute DSL query search."""
        try:
            # Execute DSL query
            image_ids = dsl_executor.execute(
                query_string=request.query,
                clip_model=clip_model,
                search_engine=search_engine,
                storage_manager=storage,
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
                    continue

            return DSLSearchResponse(
                query=request.query,
                results=results,
                total_count=len(results),
                query_info={
                    "query": request.query,
                    "type": "dsl",
                    "threshold": request.threshold,
                },
                processing_time_ms=0,  # TODO: measure actual time
            )

        except DSLError as e:
            logger.error(f"DSL search error: {e}")
            raise HTTPException(status_code=400, detail=f"DSL error: {e}")
        except Exception as e:
            logger.error(f"DSL search failed: {e}")
            raise HTTPException(status_code=500, detail="Search failed")

    @app.post(
        "/search/text",
        response_model=TextSearchResponse,
        tags=["Search"],
        summary="Natural language search",
        description="Search images using natural language with full pipeline.",
    )
    async def search_text(
        request: TextSearchRequest,
        clip_model: CLIPModel = Depends(get_clip_model),
        search_engine: VectorSearchEngine = Depends(get_search_engine),
        storage: StorageManager = Depends(get_storage_manager),
    ) -> TextSearchResponse:
        """Execute natural language search."""
        try:
            # Convert text to vector
            vector = clip_model.encode_text(request.text)

            # Perform vector search
            results = search_engine.search(
                query_embedding=vector,
                k=request.limit,
                threshold=request.threshold,
            )

            # Get metadata for results
            search_results = []
            for image_id, score in results:
                try:
                    metadata = storage.get_metadata(UUID(image_id))
                    if metadata:
                        search_results.append(
                            SearchResult(
                                image_id=UUID(image_id), metadata=metadata, score=score
                            )
                        )
                except (ValueError, StorageError):
                    continue

            pipeline_debug = None
            if request.include_pipeline_debug:
                pipeline_debug = PipelineDebugInfo(
                    text_to_vector={
                        "dimension": len(vector),
                        "model": clip_model.model_name,
                        "time_ms": 0,  # TODO: measure
                    },
                    vector_search={
                        "searched_vectors": len(results),
                        "time_ms": 0,  # TODO: measure
                    },
                )

            return TextSearchResponse(
                text=request.text,
                results=search_results,
                total_count=len(search_results),
                search_params={
                    "limit": request.limit,
                    "threshold": request.threshold,
                },
                pipeline_debug=pipeline_debug,
                processing_time_ms=0,  # TODO: measure actual time
            )

        except CLIPModelError as e:
            logger.error(f"CLIP model error: {e}")
            raise HTTPException(status_code=500, detail=f"CLIP model error: {e}")
        except VectorSearchError as e:
            logger.error(f"Vector search error: {e}")
            raise HTTPException(status_code=500, detail=f"Search error: {e}")
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            raise HTTPException(status_code=500, detail="Search failed")

    # ========================================
    # METADATA CRUD
    # ========================================

    @app.post(
        "/metadata",
        response_model=MetadataCreateResponse,
        tags=["Metadata"],
        summary="Create metadata",
        description="Create new image metadata record.",
    )
    async def create_metadata(
        request: MetadataCreateRequest,
        storage: StorageManager = Depends(get_storage_manager),
    ) -> MetadataCreateResponse:
        """Create new metadata record."""
        try:
            # Create metadata in storage
            metadata = storage.create_metadata(
                filename=request.filename,
                description=request.description,
                tags=request.tags or [],
                file_size=request.file_size,
                dimensions=request.dimensions,
            )

            return MetadataCreateResponse(
                metadata=metadata, created=True, message="Metadata created successfully"
            )

        except StorageError as e:
            logger.error(f"Storage error: {e}")
            raise HTTPException(status_code=500, detail=f"Storage error: {e}")
        except Exception as e:
            logger.error(f"Metadata creation failed: {e}")
            raise HTTPException(status_code=500, detail="Creation failed")

    @app.get(
        "/metadata",
        response_model=MetadataListResponse,
        tags=["Metadata"],
        summary="List metadata",
        description="Get paginated list of metadata records.",
    )
    async def list_metadata(
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        storage: StorageManager = Depends(get_storage_manager),
    ) -> MetadataListResponse:
        """List metadata records with pagination."""
        try:
            # Get paginated metadata
            metadata_list, total_count = storage.list_images(
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order,
            )

            return MetadataListResponse(
                metadata=metadata_list,
                total_count=total_count,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order,
            )

        except StorageError as e:
            logger.error(f"Storage error: {e}")
            raise HTTPException(status_code=500, detail=f"Storage error: {e}")
        except Exception as e:
            logger.error(f"Metadata listing failed: {e}")
            raise HTTPException(status_code=500, detail="Listing failed")

    @app.get(
        "/metadata/{metadata_id}",
        response_model=ImageMetadata,
        tags=["Metadata"],
        summary="Get metadata",
        description="Get specific metadata record by ID.",
    )
    async def get_metadata(
        metadata_id: UUID,
        storage: StorageManager = Depends(get_storage_manager),
    ) -> ImageMetadata:
        """Get specific metadata record."""
        try:
            metadata = storage.get_metadata(metadata_id)
            if metadata is None:
                raise HTTPException(status_code=404, detail="Metadata not found")

            return metadata

        except StorageError as e:
            logger.error(f"Storage error: {e}")
            raise HTTPException(status_code=500, detail=f"Storage error: {e}")
        except Exception as e:
            logger.error(f"Metadata retrieval failed: {e}")
            raise HTTPException(status_code=500, detail="Retrieval failed")

    @app.put(
        "/metadata/{metadata_id}",
        response_model=MetadataUpdateResponse,
        tags=["Metadata"],
        summary="Update metadata",
        description="Update existing metadata record.",
    )
    async def update_metadata(
        metadata_id: UUID,
        request: MetadataUpdateRequest,
        storage: StorageManager = Depends(get_storage_manager),
    ) -> MetadataUpdateResponse:
        """Update existing metadata record."""
        try:
            updated_metadata = storage.update_metadata(
                image_id=metadata_id,
                description=request.description,
                tags=request.tags,
            )

            if updated_metadata is None:
                raise HTTPException(status_code=404, detail="Metadata not found")

            return MetadataUpdateResponse(
                metadata_id=metadata_id,
                updated=True,
                metadata=updated_metadata,
                message="Metadata updated successfully",
            )

        except StorageError as e:
            logger.error(f"Storage error: {e}")
            raise HTTPException(status_code=500, detail=f"Storage error: {e}")
        except Exception as e:
            logger.error(f"Metadata update failed: {e}")
            raise HTTPException(status_code=500, detail="Update failed")

    @app.delete(
        "/metadata/{metadata_id}",
        response_model=MetadataDeleteResponse,
        tags=["Metadata"],
        summary="Delete metadata",
        description="Delete metadata record (keeps image file).",
    )
    async def delete_metadata(
        metadata_id: UUID,
        storage: StorageManager = Depends(get_storage_manager),
    ) -> MetadataDeleteResponse:
        """Delete metadata record only."""
        try:
            # Delete only metadata, not the image file
            success = storage.delete_metadata_only(metadata_id)

            if not success:
                raise HTTPException(status_code=404, detail="Metadata not found")

            return MetadataDeleteResponse(
                metadata_id=metadata_id,
                deleted=True,
                message="Metadata deleted successfully",
            )

        except StorageError as e:
            logger.error(f"Storage error: {e}")
            raise HTTPException(status_code=500, detail=f"Storage error: {e}")
        except Exception as e:
            logger.error(f"Metadata deletion failed: {e}")
            raise HTTPException(status_code=500, detail="Deletion failed")

    # ========================================
    # IMAGE DATA (CRD)
    # ========================================

    @app.post(
        "/images",
        response_model=ImageUploadResponse,
        tags=["Images"],
        summary="Upload image",
        description="Upload image file and generate embeddings.",
    )
    async def upload_image(
        file: UploadFile = File(...),
        description: Optional[str] = None,
        tags: Optional[str] = None,
        storage: StorageManager = Depends(get_storage_manager),
        clip_model: CLIPModel = Depends(get_clip_model),
        search_engine: VectorSearchEngine = Depends(get_search_engine),
    ) -> ImageUploadResponse:
        """Upload image and generate embeddings."""
        try:
            # Parse tags
            tag_list = []
            if tags:
                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

            # Store image and create metadata
            metadata = storage.store_image(
                file=file.file,
                filename=file.filename or "unknown",
                description=description,
                tags=tag_list,
            )

            # Generate embedding
            embedding = clip_model.encode_image_file(
                storage.get_image_path(metadata.id)
            )

            # Store embedding
            storage.store_embedding(metadata.id, embedding)

            # Add to search index
            search_engine.add_image(str(metadata.id), embedding)

            return ImageUploadResponse(
                image_id=metadata.id,
                metadata=metadata,
                upload_success=True,
                embedding_generated=True,
                message="Image uploaded and processed successfully",
            )

        except CLIPModelError as e:
            logger.error(f"CLIP model error: {e}")
            raise HTTPException(status_code=500, detail=f"CLIP model error: {e}")
        except StorageError as e:
            logger.error(f"Storage error: {e}")
            raise HTTPException(status_code=500, detail=f"Storage error: {e}")
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            raise HTTPException(status_code=500, detail="Upload failed")

    @app.get(
        "/images",
        response_model=ImageListResponse,
        tags=["Images"],
        summary="List images",
        description="Get paginated list of images.",
    )
    async def list_images(
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        storage: StorageManager = Depends(get_storage_manager),
    ) -> ImageListResponse:
        """List images with pagination."""
        try:
            images, total_count = storage.list_images(
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order,
            )

            return ImageListResponse(
                images=images,
                total_count=total_count,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order,
            )

        except StorageError as e:
            logger.error(f"Storage error: {e}")
            raise HTTPException(status_code=500, detail=f"Storage error: {e}")
        except Exception as e:
            logger.error(f"Image listing failed: {e}")
            raise HTTPException(status_code=500, detail="Listing failed")

    @app.get(
        "/images/{image_id}",
        tags=["Images"],
        summary="Get image file",
        description="Download image file by ID.",
    )
    async def get_image_file(
        image_id: UUID,
        storage: StorageManager = Depends(get_storage_manager),
    ) -> StreamingResponse:
        """Get image file."""
        try:
            # Check if image exists
            metadata = storage.get_metadata(image_id)
            if metadata is None:
                raise HTTPException(status_code=404, detail="Image not found")

            # Get image path
            image_path = storage.get_image_path(image_id)
            if not image_path.exists():
                raise HTTPException(status_code=404, detail="Image file not found")

            # Return streaming response
            def iterfile():
                with open(image_path, "rb") as file:
                    yield from file

            return StreamingResponse(
                iterfile(),
                media_type="image/jpeg",  # TODO: detect actual MIME type
                headers={
                    "Content-Disposition": f"inline; filename={metadata.filename}"
                },
            )

        except StorageError as e:
            logger.error(f"Storage error: {e}")
            raise HTTPException(status_code=500, detail=f"Storage error: {e}")
        except Exception as e:
            logger.error(f"Image retrieval failed: {e}")
            raise HTTPException(status_code=500, detail="Retrieval failed")

    @app.delete(
        "/images/{image_id}",
        response_model=ImageDeleteResponse,
        tags=["Images"],
        summary="Delete image",
        description="Delete image file and all associated data.",
    )
    async def delete_image(
        image_id: UUID,
        storage: StorageManager = Depends(get_storage_manager),
        search_engine: VectorSearchEngine = Depends(get_search_engine),
    ) -> ImageDeleteResponse:
        """Delete image and all associated data."""
        try:
            # Remove from search index
            try:
                search_engine.remove_image(str(image_id))
            except VectorSearchError:
                logger.warning(f"Could not remove {image_id} from search index")

            # Delete from storage (metadata, embedding, and file)
            success = storage.delete_image(image_id)

            if success:
                return ImageDeleteResponse(
                    image_id=image_id,
                    deleted=True,
                    message="Image deleted successfully",
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to delete image")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete image: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete image")

    # ========================================
    # SYSTEM
    # ========================================

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check",
        description="Get system health status.",
    )
    async def health_check(
        storage: StorageManager = Depends(get_storage_manager),
        clip_model: CLIPModel = Depends(get_clip_model),
        search_engine: VectorSearchEngine = Depends(get_search_engine),
    ) -> HealthResponse:
        """Get system health status."""
        try:
            # Get component status
            clip_status = {
                "model_name": clip_model.model_name,
                "device": str(clip_model.device),
                "embedding_dim": clip_model.embedding_dim,
                "is_loaded": True,
            }

            storage_status = {
                "total_images": len(storage.list_images(limit=1000000)[0]),
                "total_embeddings": storage.count_embeddings(),
                "total_storage_bytes": storage.get_storage_size(),
                "storage_path": str(storage.images_dir),
                "database_path": str(storage.db_path),
            }

            search_status = {
                "total_embeddings": search_engine.count(),
                "embedding_dimension": search_engine.dimension,
                "index_type": search_engine.index_type,
                "is_trained": search_engine.is_trained,
            }

            return HealthResponse(
                status="healthy",
                components={
                    "clip_model": clip_status,
                    "storage": storage_status,
                    "search_engine": search_status,
                },
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail="Health check failed")

    return app


# Create default app instance
app = create_app()
