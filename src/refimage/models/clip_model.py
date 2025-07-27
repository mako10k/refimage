"""
CLIP model wrapper for image and text embedding generation.

This module provides a unified interface for CLIP model operations,
including image encoding, text encoding, and similarity calculations.
"""

import io
import logging
from pathlib import Path
from typing import List, Union

import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import Settings

logger = logging.getLogger(__name__)


class CLIPModelError(Exception):
    """CLIP model related errors."""


class CLIPModel:
    """
    CLIP model wrapper for embedding generation.

    Provides methods for encoding images and text into embeddings
    using OpenAI's CLIP model.
    """

    def __init__(self, settings: Settings):
        """
        Initialize CLIP model.

        Args:
            settings: Application settings containing model configuration

        Raises:
            CLIPModelError: If model initialization fails
        """
        assert settings is not None, "Settings object is required"

        self.settings = settings
        self.device = self._determine_device()
        self.model = None
        self.preprocess = None
        self.model_name = settings.clip_model_name  # Add model_name attribute
        self._load_model()

    def _determine_device(self) -> torch.device:
        """Determine the appropriate device for model inference."""
        if self.settings.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using CUDA device for CLIP model")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device for CLIP model")
        else:
            device = torch.device(self.settings.device)
            logger.info(f"Using {self.settings.device} device for CLIP model")

        return device

    def _load_model(self) -> None:
        """Load CLIP model and preprocessing functions."""
        try:
            logger.info(f"Loading CLIP model: {self.settings.clip_model_name}")
            self.model, self.preprocess = clip.load(
                self.settings.clip_model_name, device=self.device
            )
            self.model.eval()  # Set to evaluation mode
            logger.info("CLIP model loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load CLIP model: {e}"
            logger.error(error_msg)
            raise CLIPModelError(error_msg) from e

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            return 512  # Default CLIP embedding dimension
        # Get actual dimension from loaded model
        with torch.no_grad():
            dummy_text = clip.tokenize(["test"]).to(self.device)
            text_features = self.model.encode_text(dummy_text)
            return text_features.shape[-1]

    def encode_image(self, image_input: Union[Image.Image, bytes, Path]) -> np.ndarray:
        """
        Encode image to CLIP embedding.

        Args:
            image_input: Image as PIL Image, bytes, or file path

        Returns:
            Normalized embedding vector as numpy array

        Raises:
            CLIPModelError: If image encoding fails
        """
        assert image_input is not None, "Image input is required"
        assert self.model is not None, "Model not loaded"
        assert self.preprocess is not None, "Preprocessing function not loaded"

        try:
            # Convert input to PIL Image
            if isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, Path):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                error_msg = f"Unsupported image input type: {type(image_input)}"
                raise CLIPModelError(error_msg)

            # Preprocess and encode
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)

            # Convert to numpy and return
            embedding = image_features.cpu().numpy().flatten()

            logger.debug(f"Generated image embedding with shape: {embedding.shape}")
            return embedding

        except Exception as e:
            error_msg = f"Failed to encode image: {e}"
            logger.error(error_msg)
            raise CLIPModelError(error_msg) from e

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to CLIP embedding.

        Args:
            text: Text string to encode

        Returns:
            Normalized embedding vector as numpy array

        Raises:
            CLIPModelError: If text encoding fails
        """
        assert text is not None, "Text input is required"
        assert isinstance(text, str), f"Text must be string, got {type(text)}"
        assert len(text.strip()) > 0, "Text cannot be empty"
        assert self.model is not None, "Model not loaded"

        try:
            # Tokenize text
            text_tokens = clip.tokenize([text]).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                # Normalize features
                text_features = F.normalize(text_features, dim=-1)

            # Convert to numpy and return
            embedding = text_features.cpu().numpy().flatten()

            logger.debug(f"Generated text embedding with shape: {embedding.shape}")
            return embedding

        except Exception as e:
            error_msg = f"Failed to encode text: {e}"
            logger.error(error_msg)
            raise CLIPModelError(error_msg) from e

    def encode_batch_images(
        self, images: List[Union[Image.Image, bytes, Path]]
    ) -> np.ndarray:
        """
        Encode multiple images in batch for efficiency.

        Args:
            images: List of images to encode

        Returns:
            Array of embeddings with shape (batch_size, embedding_dim)

        Raises:
            CLIPModelError: If batch encoding fails
        """
        assert images is not None, "Images list is required"
        assert len(images) > 0, "Images list cannot be empty"
        assert self.model is not None, "Model not loaded"
        assert self.preprocess is not None, "Preprocessing function not loaded"

        try:
            # Convert all images to PIL format and preprocess
            processed_images = []
            for img_input in images:
                if isinstance(img_input, bytes):
                    image = Image.open(io.BytesIO(img_input)).convert("RGB")
                elif isinstance(img_input, Path):
                    image = Image.open(img_input).convert("RGB")
                elif isinstance(img_input, Image.Image):
                    image = img_input.convert("RGB")
                else:
                    error_msg = f"Unsupported image input type: {type(img_input)}"
                    raise CLIPModelError(error_msg)

                processed_images.append(self.preprocess(image))

            # Stack images into batch tensor
            batch_tensor = torch.stack(processed_images).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(batch_tensor)
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)

            # Convert to numpy
            embeddings = image_features.cpu().numpy()

            logger.debug(f"Generated batch embeddings with shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            error_msg = f"Failed to encode image batch: {e}"
            logger.error(error_msg)
            raise CLIPModelError(error_msg) from e

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1

        Raises:
            CLIPModelError: If similarity calculation fails
        """
        assert embedding1 is not None, "First embedding is required"
        assert embedding2 is not None, "Second embedding is required"
        assert (
            embedding1.shape == embedding2.shape
        ), f"Embedding shapes must match: {embedding1.shape} != {embedding2.shape}"

        try:
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2

            # Calculate cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)

            # Ensure result is in [0, 1] range (convert from [-1, 1])
            similarity = (similarity + 1) / 2

            return float(similarity)

        except Exception as e:
            error_msg = f"Failed to calculate similarity: {e}"
            logger.error(error_msg)
            raise CLIPModelError(error_msg) from e

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.settings.clip_model_name,
            "device": str(self.device),
            "embedding_dim": self.model.visual.output_dim if self.model else None,
            "is_loaded": self.model is not None,
        }
