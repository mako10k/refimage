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
import time
from typing import Optional
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .config import Settings
from .dsl import DSLError, DSLExecutor
from .llm import (
    TEXT_TO_DSL_EXAMPLES,
    TEXT_TO_DSL_SYSTEM_PROMPT,
    LLMError,
    LLMManager,
    LLMMessage,
)
from .models.clip_model import CLIPModel, CLIPModelError
from .models.schemas import (  # Conversion, Search, Metadata, etc.
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
    LLMProviderInfo,
    LLMProvidersResponse,
    LLMSwitchRequest,
    LLMSwitchResponse,
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
    TextToDSLRequest,
    TextToDSLResponse,
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
    # Contract Programming: Validate settings
    if settings is None:
        settings = Settings()
    assert settings is not None, "Settings must be provided or created"

    app = FastAPI(
        title="RefImage API",
        description="Pipeline-oriented image search with CLIP embeddings",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Initialize core components first (non-faiss dependent)
    storage_manager = StorageManager(settings)
    assert storage_manager is not None, "Storage manager initialization failed"

    # LLM manager - independent of faiss
    llm_manager = LLMManager(settings)
    assert llm_manager is not None, "LLM manager initialization failed"

    # Lazy initialization containers for faiss-dependent components
    _clip_model = None
    _search_engine = None
    _dsl_executor = None

    # Dependency providers
    def get_storage_manager() -> StorageManager:
        """Get storage manager (always available)."""
        assert storage_manager is not None, "Storage manager not initialized"
        return storage_manager

    def get_clip_model() -> CLIPModel:
        """Get CLIP model with lazy initialization."""
        nonlocal _clip_model
        if _clip_model is None:
            try:
                _clip_model = CLIPModel(settings)
                assert _clip_model is not None, "CLIP model creation failed"
            except Exception as e:
                logger.error(f"CLIP model initialization failed: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="CLIP model not available - dependency error",
                )
        return _clip_model

    def get_search_engine() -> VectorSearchEngine:
        """Get search engine with lazy initialization."""
        nonlocal _search_engine
        if _search_engine is None:
            try:
                _search_engine = VectorSearchEngine(settings)
                assert _search_engine is not None, "Search engine creation failed"
            except Exception as e:
                logger.error(f"Search engine initialization failed: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Search engine not available - faiss dependency error",
                )
        return _search_engine

    def get_dsl_executor() -> DSLExecutor:
        """Get DSL executor with lazy initialization."""
        nonlocal _dsl_executor
        if _dsl_executor is None:
            try:
                clip_model = get_clip_model()
                search_engine = get_search_engine()
                _dsl_executor = DSLExecutor(clip_model, search_engine, storage_manager)
                assert _dsl_executor is not None, "DSL executor creation failed"
            except Exception as e:
                logger.error(f"DSL executor initialization failed: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="DSL executor not available - dependency error",
                )
        return _dsl_executor

    def get_llm_manager() -> LLMManager:
        """Get LLM manager (always available)."""
        assert llm_manager is not None, "LLM manager not initialized"
        return llm_manager

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
    # LLM INTEGRATION
    # ========================================

    @app.post(
        "/conversions/text-to-dsl",
        response_model=TextToDSLResponse,
        tags=["Conversions"],
        summary="Convert natural language to DSL using LLM",
        description=("Use LLM to convert natural language queries to DSL format."),
    )
    async def text_to_dsl(
        request: TextToDSLRequest,
        llm_manager: LLMManager = Depends(get_llm_manager),
    ):
        """Convert natural language to DSL using LLM."""
        try:
            start_time = time.time()

            # Build prompt with examples if requested
            prompt_parts = [TEXT_TO_DSL_SYSTEM_PROMPT]

            if request.include_examples:
                examples_text = "\n\nExamples:\n"
                for example in TEXT_TO_DSL_EXAMPLES:
                    examples_text += f"Input: {example['input']}\n"
                    examples_text += f"Output: {example['output']}\n"
                    explanation = f"Explanation: {example['explanation']}\n\n"
                    examples_text += explanation
                prompt_parts.append(examples_text)

            prompt_parts.append(f"\nQuery: {request.text}")

            system_prompt = "\n".join(prompt_parts)

            # Create messages
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=request.text),
            ]

            provider_enum = None
            if request.provider:
                from .llm import LLMProvider

                provider_enum = LLMProvider(request.provider)

            # Generate DSL using LLM
            response = await llm_manager.generate(
                messages,
                provider=provider_enum,
                temperature=request.temperature,
                max_tokens=200,
            )

            # Parse response to extract DSL and explanation
            dsl_query = response.content.strip()
            explanation = f"Converted using {response.provider} ({response.model})"

            # Simple confidence estimation based on response quality
            confidence = 0.8 if len(dsl_query) > 10 else 0.5

            processing_time = int((time.time() - start_time) * 1000)

            return TextToDSLResponse(
                original_text=request.text,
                dsl_query=dsl_query,
                explanation=explanation,
                confidence=confidence,
                provider=response.provider,
                model=response.model,
                processing_time_ms=processing_time,
            )

        except LLMError as e:
            logger.error(f"LLM error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")
        except Exception as e:
            logger.error(f"Text to DSL conversion failed: {e}")
            raise HTTPException(status_code=500, detail="Text to DSL conversion failed")

    @app.get(
        "/llm/providers",
        response_model=LLMProvidersResponse,
        tags=["LLM"],
        summary="List available LLM providers",
        description="Get information about available LLM providers.",
    )
    async def list_llm_providers(
        llm_manager: LLMManager = Depends(get_llm_manager),
    ):
        """List available LLM providers."""
        try:
            current_provider = llm_manager.get_current_provider()
            available_providers = llm_manager.get_available_providers()

            providers = []
            from .llm import LLMProvider

            for provider_name in ["openai", "claude", "local"]:
                try:
                    provider_enum = LLMProvider(provider_name)
                    is_available = provider_enum in available_providers
                    model_name = (
                        llm_manager.providers[provider_enum].get_model_name()
                        if is_available
                        else None
                    )
                    providers.append(
                        LLMProviderInfo(
                            name=provider_name,
                            available=is_available,
                            model=model_name,
                            description=f"{provider_name.title()} LLM provider",
                        )
                    )
                except ValueError:
                    # Invalid provider name
                    continue

            return LLMProvidersResponse(
                current_provider=current_provider,
                providers=providers,
            )

        except Exception as e:
            logger.error(f"Failed to list LLM providers: {e}")
            raise HTTPException(status_code=500, detail="Failed to list LLM providers")

    @app.post(
        "/llm/switch",
        response_model=LLMSwitchResponse,
        tags=["LLM"],
        summary="Switch LLM provider",
        description="Switch to a different LLM provider.",
    )
    async def switch_llm_provider(
        request: LLMSwitchRequest,
        llm_manager: LLMManager = Depends(get_llm_manager),
    ):
        """Switch LLM provider."""
        try:
            previous_provider = llm_manager.get_current_provider()

            from .llm import LLMProvider

            provider_enum = LLMProvider(request.provider)
            llm_manager.switch_provider(provider_enum)

            return LLMSwitchResponse(
                success=True,
                previous_provider=previous_provider,
                current_provider=request.provider,
                message=f"Successfully switched to {request.provider} provider",
            )

        except LLMError as e:
            logger.error(f"LLM provider switch failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to switch LLM provider: {e}")
            raise HTTPException(status_code=500, detail="Failed to switch provider")

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
                                image_id=UUID(image_id),
                                metadata=metadata,
                                score=score,
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
                                image_id=UUID(image_id),
                                metadata=metadata,
                                score=score,
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
                metadata=metadata,
                created=True,
                message="Metadata created successfully",
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
        description: Optional[str] = Form(None),
        tags: Optional[str] = Form(None),
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
            image_data = file.file.read()
            metadata = storage.store_image(
                image_data=image_data,
                filename=file.filename or "unknown",
                description=description,
                tags=tag_list,
            )

            # Generate embedding
            embedding_vector = clip_model.encode_image(metadata.file_path)

            # Store embedding
            from datetime import datetime

            from .models.schemas import ImageEmbedding

            embedding = ImageEmbedding(
                image_id=metadata.id,
                embedding=embedding_vector.tolist(),
                model_name=clip_model.model_name,
                created_at=datetime.utcnow(),
            )
            storage.store_embedding(embedding)

            # Add to search index (temporarily commented for testing)
            # search_engine.add_image(str(metadata.id), embedding_vector)

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
            # Check if this is a duplicate image error
            if "Duplicate image detected" in str(e):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Duplicate image detected. "
                        "This image has already been uploaded."
                    ),
                )
            else:
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
                "storage_path": str(storage.image_storage_path),
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
