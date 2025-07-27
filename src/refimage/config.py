"""
Configuration management for RefImage application.

This module handles loading and validation of configuration settings
from environment variables and provides type-safe configuration objects.
"""

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings."""

    # Model settings
    clip_model_name: str = Field(default="ViT-B/32", description="CLIP model name")
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto", description="Device for model inference"
    )

    # Storage settings
    image_storage_path: Path = Field(
        default=Path("./data/images"), description="Path for image storage"
    )
    index_storage_path: Path = Field(
        default=Path("./data/indexes"),
        description="Path for FAISS index storage",
    )
    metadata_storage_path: Path = Field(
        default=Path("./data/metadata"),
        description="Path for metadata storage",
    )
    database_path: Path = Field(
        default=Path("./data/refimage.db"), description="SQLite database path"
    )

    # API settings
    max_image_size: int = Field(
        default=10485760, description="Maximum image size in bytes (10MB)"
    )
    allowed_image_types: List[str] = Field(
        default=["jpg", "jpeg", "png", "webp"],
        description="Allowed image file types",
    )

    # FAISS settings
    index_type: Literal["flat", "ivf", "hnsw"] = Field(
        default="flat", description="FAISS index type"
    )
    search_k: int = Field(
        default=100, description="Number of candidates for IVF search"
    )

    # Server settings
    server_host: str = Field(default="0.0.0.0", description="Server host")
    server_port: int = Field(default=8000, description="Server port")
    log_level: str = Field(default="INFO", description="Log level")
    debug: bool = Field(default=False, description="Debug mode")

    # LLM settings
    llm_provider: str = Field(
        default="openai",
        description="Default LLM provider (openai/claude/local)",
    )

    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="OpenAI model name")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI base URL"
    )

    # Claude settings
    claude_api_key: Optional[str] = Field(default=None, description="Claude API key")
    claude_model: str = Field(
        default="claude-3-sonnet-20240229", description="Claude model name"
    )
    claude_base_url: str = Field(
        default="https://api.anthropic.com/v1", description="Claude base URL"
    )

    # Local LLM settings
    local_llm_enabled: bool = Field(default=False, description="Enable local LLM")
    local_llm_model: str = Field(default="llama2", description="Local LLM model name")
    local_llm_base_url: str = Field(
        default="http://localhost:11434", description="Local LLM base URL"
    )

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # 追加のフィールドを無視


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()
