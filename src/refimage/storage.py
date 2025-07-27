"""
Storage layer for image files and metadata management.

This module handles file storage, metadata persistence,
and database operations for the RefImage application.
"""

import hashlib
import io
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from PIL import Image

from .config import Settings
from .models.schemas import ImageEmbedding, ImageMetadata

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Storage related errors."""


class StorageManager:
    """
    Manages file storage and metadata persistence.

    Handles image file storage, metadata database operations,
    and embedding persistence with SQLite backend.
    """

    def __init__(self, settings: Settings):
        """
        Initialize storage manager.

        Args:
            settings: Application settings

        Raises:
            StorageError: If initialization fails
        """
        assert settings is not None, "Settings object is required"

        self.settings = settings
        self.image_storage_path = Path(settings.image_storage_path)
        self.db_path = Path(settings.database_path)

        self._setup_storage()
        self._setup_database()

    def _setup_storage(self) -> None:
        """Setup storage directories."""
        try:
            self.image_storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Image storage path: {self.image_storage_path}")

        except Exception as e:
            error_msg = f"Failed to setup storage directories: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def _setup_database(self) -> None:
        """Setup SQLite database and tables."""
        try:
            # Create database directory
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Initialize database with tables
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS images (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        mime_type TEXT NOT NULL,
                        width INTEGER NOT NULL,
                        height INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        description TEXT,
                        tags TEXT,
                        file_hash TEXT UNIQUE
                    )
                """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        image_id TEXT PRIMARY KEY,
                        embedding BLOB NOT NULL,
                        model_name TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                """
                )

                # Create indexes for better performance
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_images_created_at
                    ON images (created_at)
                """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_images_file_hash
                    ON images (file_hash)
                """
                )

                conn.commit()

            logger.info(f"Database initialized: {self.db_path}")

        except Exception as e:
            error_msg = f"Failed to setup database: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        assert file_path.exists(), f"File not found: {file_path}"

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _create_image_metadata_from_row(
        self, row: sqlite3.Row
    ) -> ImageMetadata:
        """
        Create ImageMetadata instance from database row.

        Args:
            row: Database row containing image metadata

        Returns:
            ImageMetadata instance
        """
        return ImageMetadata(
            id=UUID(row["id"]),
            filename=row["filename"],
            file_path=Path(row["file_path"]),
            file_size=row["file_size"],
            mime_type=row["mime_type"],
            width=row["width"],
            height=row["height"],
            created_at=datetime.fromisoformat(row["created_at"]),
            description=row["description"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
        )

    def store_image(
        self,
        image_data: bytes,
        filename: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ImageMetadata:
        """
        Store image file and metadata.

        Args:
            image_data: Image file data
            filename: Original filename
            description: Optional description
            tags: Optional tags list

        Returns:
            Image metadata object

        Raises:
            StorageError: If storage fails
        """
        assert image_data is not None, "Image data is required"
        assert filename is not None, "Filename is required"
        assert len(filename.strip()) > 0, "Filename cannot be empty"

        try:
            # Open image to get dimensions and validate
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size

            # Determine MIME type
            format_map = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "GIF": "image/gif",
                "BMP": "image/bmp",
                "WEBP": "image/webp",
            }
            mime_type = format_map.get(
                image.format, "application/octet-stream"
            )

            # Generate unique file path
            file_extension = Path(filename).suffix
            if not file_extension:
                file_extension = f".{image.format.lower()}"

            # Create metadata object
            from uuid import uuid4
            from datetime import datetime
            
            metadata = ImageMetadata(
                id=uuid4(),
                filename=filename,
                file_path=Path("placeholder"),  # Will be updated
                file_size=len(image_data),
                mime_type=mime_type,
                width=width,
                height=height,
                description=description,
                tags=tags or [],
                created_at=datetime.utcnow(),
                updated_at=None,
            )

            # Create storage path
            storage_file_path = (
                self.image_storage_path / f"{metadata.id}{file_extension}"
            )
            metadata.file_path = storage_file_path

            # Write file to storage
            with open(storage_file_path, "wb") as f:
                f.write(image_data)

            # Calculate file hash
            file_hash = self._calculate_file_hash(storage_file_path)

            # Store metadata in database
            self._store_metadata(metadata, file_hash)

            logger.info(f"Stored image: {metadata.id}")
            return metadata

        except Exception as e:
            error_msg = f"Failed to store image: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def _store_metadata(self, metadata: ImageMetadata, file_hash: str) -> None:
        """Store metadata in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO images (
                        id, filename, file_path, file_size, mime_type,
                        width, height, created_at, description, tags, file_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(metadata.id),
                        metadata.filename,
                        str(metadata.file_path),
                        metadata.file_size,
                        metadata.mime_type,
                        metadata.width,
                        metadata.height,
                        metadata.created_at.isoformat(),
                        metadata.description,
                        json.dumps(metadata.tags),
                        file_hash,
                    ),
                )
                conn.commit()

        except sqlite3.IntegrityError as e:
            if "file_hash" in str(e):
                raise StorageError("Duplicate image detected") from e
            raise StorageError(f"Database integrity error: {e}") from e

    def get_metadata(self, image_id: UUID) -> Optional[ImageMetadata]:
        """
        Get image metadata by ID.

        Args:
            image_id: Image identifier

        Returns:
            Image metadata or None if not found

        Raises:
            StorageError: If retrieval fails
        """
        assert image_id is not None, "Image ID is required"

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM images WHERE id = ?
                """,
                    (str(image_id),),
                )

                row = cursor.fetchone()
                if row is None:
                    return None

                return self._create_image_metadata_from_row(row)

        except Exception as e:
            error_msg = f"Failed to get metadata: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def list_images(
        self,
        limit: int = 100,
        offset: int = 0,
        tags_filter: Optional[List[str]] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> Tuple[List[ImageMetadata], int]:
        """
        List stored images with pagination and sorting.

        Args:
            limit: Maximum number of results
            offset: Results offset
            tags_filter: Filter by tags
            sort_by: Field to sort by (created_at, filename, file_size)
            sort_order: Sort order (asc, desc)

        Returns:
            Tuple of (list of image metadata, total count)

        Raises:
            StorageError: If listing fails
        """
        assert limit > 0, f"Invalid limit: {limit}"
        assert offset >= 0, f"Invalid offset: {offset}"
        assert sort_by in [
            "created_at",
            "filename",
            "file_size",
        ], f"Invalid sort_by: {sort_by}"
        assert sort_order in [
            "asc",
            "desc",
        ], f"Invalid sort_order: {sort_order}"

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Build base query
                base_query = "FROM images"
                where_conditions = []
                params = []

                if tags_filter:
                    # Simple tag filtering
                    tag_conditions = []
                    for tag in tags_filter:
                        tag_conditions.append("tags LIKE ?")
                        params.append(f'%"{tag}"%')

                    where_conditions.extend(tag_conditions)

                where_clause = ""
                if where_conditions:
                    where_clause = " WHERE " + " AND ".join(where_conditions)

                # Count total results
                count_query = f"SELECT COUNT(*) {base_query}{where_clause}"
                cursor = conn.execute(count_query, params)
                total_count = cursor.fetchone()[0]

                # Get paginated results with sorting
                data_query = (
                    f"SELECT * {base_query}{where_clause} "
                    f"ORDER BY {sort_by} {sort_order.upper()} "
                    f"LIMIT ? OFFSET ?"
                )
                data_params = params + [limit, offset]

                cursor = conn.execute(data_query, data_params)
                rows = cursor.fetchall()

                images = []
                for row in rows:
                    metadata = self._create_image_metadata_from_row(row)
                    images.append(metadata)

                return images, total_count

        except Exception as e:
            error_msg = f"Failed to list images: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def update_metadata(
        self,
        image_id: UUID,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[ImageMetadata]:
        """
        Update image metadata.

        Args:
            image_id: Image identifier
            description: New description (None to keep existing)
            tags: New tags list (None to keep existing)

        Returns:
            Updated metadata if successful, None if image not found

        Raises:
            StorageError: If update fails
        """
        assert image_id is not None, "Image ID is required"

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Check if image exists
                cursor = conn.execute(
                    "SELECT * FROM images WHERE id = ?", (str(image_id),)
                )
                row = cursor.fetchone()
                if not row:
                    return None

                # Prepare update fields
                update_fields = []
                params = []

                if description is not None:
                    update_fields.append("description = ?")
                    params.append(description)

                if tags is not None:
                    update_fields.append("tags = ?")
                    params.append(json.dumps(tags))

                if not update_fields:
                    # No updates requested, return current metadata
                    return self._create_image_metadata_from_row(row)

                # Perform update
                update_query = (
                    f"UPDATE images SET {', '.join(update_fields)} "
                    f"WHERE id = ?"
                )
                params.append(str(image_id))

                conn.execute(update_query, params)
                conn.commit()

                # Fetch and return updated metadata
                cursor = conn.execute(
                    "SELECT * FROM images WHERE id = ?", (str(image_id),)
                )
                updated_row = cursor.fetchone()
                return self._create_image_metadata_from_row(updated_row)

        except Exception as e:
            error_msg = f"Failed to update metadata for {image_id}: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def delete_image(self, image_id: UUID) -> bool:
        """
        Delete image and metadata.

        Args:
            image_id: Image identifier

        Returns:
            True if deleted, False if not found

        Raises:
            StorageError: If deletion fails
        """
        assert image_id is not None, "Image ID is required"

        try:
            # Get metadata first
            metadata = self.get_metadata(image_id)
            if metadata is None:
                return False

            # Delete from database first
            with sqlite3.connect(self.db_path) as conn:
                # Delete embedding if exists
                conn.execute(
                    """
                    DELETE FROM embeddings WHERE image_id = ?
                """,
                    (str(image_id),),
                )

                # Delete metadata
                cursor = conn.execute(
                    """
                    DELETE FROM images WHERE id = ?
                """,
                    (str(image_id),),
                )

                if cursor.rowcount == 0:
                    return False

                conn.commit()

            # Delete file if exists
            if metadata.file_path.exists():
                metadata.file_path.unlink()

            logger.info(f"Deleted image: {image_id}")
            return True

        except Exception as e:
            error_msg = f"Failed to delete image: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def store_embedding(self, embedding: ImageEmbedding) -> None:
        """
        Store image embedding.

        Args:
            embedding: Image embedding to store

        Raises:
            StorageError: If storage fails
        """
        assert embedding is not None, "Embedding object is required"
        assert embedding.image_id is not None, "Image ID is required"
        assert embedding.embedding is not None, "Embedding vector is required"

        try:
            # Serialize embedding vector
            embedding_blob = json.dumps(embedding.embedding).encode("utf-8")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings (
                        image_id, embedding, model_name, created_at
                    ) VALUES (?, ?, ?, ?)
                """,
                    (
                        str(embedding.image_id),
                        embedding_blob,
                        embedding.model_name,
                        embedding.created_at.isoformat(),
                    ),
                )
                conn.commit()

            logger.debug(f"Stored embedding for image {embedding.image_id}")

        except Exception as e:
            error_msg = f"Failed to store embedding: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def get_embedding(self, image_id: UUID) -> Optional[ImageEmbedding]:
        """
        Get image embedding by ID.

        Args:
            image_id: Image identifier

        Returns:
            Image embedding or None if not found

        Raises:
            StorageError: If retrieval fails
        """
        assert image_id is not None, "Image ID is required"

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM embeddings WHERE image_id = ?
                """,
                    (str(image_id),),
                )

                row = cursor.fetchone()
                if row is None:
                    return None

                # Deserialize embedding vector
                embedding_vector = json.loads(row["embedding"].decode("utf-8"))

                return ImageEmbedding(
                    image_id=UUID(row["image_id"]),
                    embedding=embedding_vector,
                    model_name=row["model_name"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )

        except Exception as e:
            error_msg = f"Failed to get embedding: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def get_all_embeddings(self) -> List[ImageEmbedding]:
        """
        Get all stored embeddings.

        Returns:
            List of all embeddings

        Raises:
            StorageError: If retrieval fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM embeddings ORDER BY created_at
                """
                )

                embeddings = []
                for row in cursor.fetchall():
                    embedding_vector = json.loads(
                        row["embedding"].decode("utf-8")
                    )

                    embedding = ImageEmbedding(
                        image_id=UUID(row["image_id"]),
                        embedding=embedding_vector,
                        model_name=row["model_name"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                    embeddings.append(embedding)

                return embeddings

        except Exception as e:
            error_msg = f"Failed to get all embeddings: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def has_embedding(self, image_id: UUID) -> bool:
        """
        Check if an image has an embedding.

        Args:
            image_id: Image identifier

        Returns:
            True if embedding exists, False otherwise
        """
        assert image_id is not None, "Image ID is required"

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE image_id = ?",
                    (str(image_id),),
                )
                count = cursor.fetchone()[0]
                return count > 0

        except Exception as e:
            logger.error(f"Failed to check embedding existence: {e}")
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary containing storage statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count images
                cursor = conn.execute("SELECT COUNT(*) FROM images")
                image_count = cursor.fetchone()[0]

                # Count embeddings
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                embedding_count = cursor.fetchone()[0]

                # Calculate total file size
                cursor = conn.execute("SELECT SUM(file_size) FROM images")
                total_size = cursor.fetchone()[0] or 0

            return {
                "total_images": image_count,
                "total_embeddings": embedding_count,
                "total_storage_bytes": total_size,
                "storage_path": str(self.image_storage_path),
                "database_path": str(self.db_path),
            }

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}

    def create_metadata(
        self,
        filename: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        file_size: int = 0,
        dimensions: Optional[Tuple[int, int]] = None,
    ) -> ImageMetadata:
        """
        Create metadata record without image file.

        Args:
            filename: Image filename
            description: Image description
            tags: Image tags
            file_size: File size in bytes
            dimensions: Image dimensions (width, height)

        Returns:
            Created metadata

        Raises:
            StorageError: If creation fails
        """
        assert filename is not None, "Filename is required"
        assert file_size >= 0, "File size must be non-negative"

        try:
            from uuid import uuid4

            image_id = uuid4()

            # Prepare dimensions
            dimensions_dict = None
            if dimensions:
                dimensions_dict = {
                    "width": dimensions[0],
                    "height": dimensions[1],
                }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO images
                    (id, filename, description, tags, file_size, dimensions,
                     created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(image_id),
                        filename,
                        description,
                        json.dumps(tags or []),
                        file_size,
                        (
                            json.dumps(dimensions_dict)
                            if dimensions_dict
                            else None
                        ),
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()

            # Return created metadata
            metadata = self.get_metadata(image_id)
            if metadata is None:
                raise StorageError("Failed to retrieve created metadata")
            return metadata

        except Exception as e:
            error_msg = f"Failed to create metadata: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def delete_metadata_only(self, image_id: UUID) -> bool:
        """
        Delete metadata record without deleting image file.

        Args:
            image_id: Image identifier

        Returns:
            True if successful, False if not found
        """
        assert image_id is not None, "Image ID is required"

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM images WHERE id = ?", (str(image_id),)
                )
                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            error_msg = f"Failed to delete metadata: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    def count_embeddings(self) -> int:
        """
        Count total number of embeddings.

        Returns:
            Number of embeddings
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                return cursor.fetchone()[0]

        except Exception as e:
            logger.error(f"Failed to count embeddings: {e}")
            return 0

    def get_storage_size(self) -> int:
        """
        Get total storage size in bytes.

        Returns:
            Total size in bytes
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT SUM(file_size) FROM images")
                result = cursor.fetchone()[0]
                return result or 0

        except Exception as e:
            logger.error(f"Failed to get storage size: {e}")
            return 0
