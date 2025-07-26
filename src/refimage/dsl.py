"""
Dynamic Search Language (DSL) parser for complex image queries.

This module provides parsing and execution of advanced search queries
that combine multiple conditions, operators, and filters.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models.clip_model import CLIPModel
from .search import VectorSearchEngine
from .storage import StorageManager

logger = logging.getLogger(__name__)


class DSLError(Exception):
    """DSL parsing and execution errors."""


class QueryNode(ABC):
    """Abstract base class for query AST nodes."""

    @abstractmethod
    def execute(
        self,
        clip_model: CLIPModel,
        search_engine: VectorSearchEngine,
        storage_manager: StorageManager,
        context: Dict[str, Any],
    ) -> List[str]:
        """
        Execute query node and return list of matching image IDs.

        Args:
            clip_model: CLIP model for embedding generation
            search_engine: Vector search engine
            storage_manager: Storage manager for metadata
            context: Execution context

        Returns:
            List of matching image IDs
        """


class TextQuery(QueryNode):
    """Text-based similarity query."""

    def __init__(self, text: str, weight: float = 1.0):
        """
        Initialize text query.

        Args:
            text: Query text
            weight: Query weight (0.0 - 1.0)
        """
        assert text is not None, "Query text is required"
        assert len(text.strip()) > 0, "Query text cannot be empty"
        assert 0.0 <= weight <= 1.0, f"Invalid weight: {weight}"

        self.text = text.strip()
        self.weight = weight

    def execute(
        self,
        clip_model: CLIPModel,
        search_engine: VectorSearchEngine,
        storage_manager: StorageManager,
        context: Dict[str, Any],
    ) -> List[str]:
        """Execute text similarity search."""
        try:
            # Generate text embedding
            embedding = clip_model.encode_text(self.text)

            # Search for similar images
            results = search_engine.search(
                query_embedding=embedding,
                k=context.get("limit", 50),
                threshold=context.get("threshold", 0.3),
            )

            # Apply weight to scores and filter
            weighted_results = []
            for image_id, score in results:
                weighted_score = score * self.weight
                if weighted_score >= context.get("min_score", 0.0):
                    weighted_results.append((image_id, weighted_score))

            # Sort by weighted score
            weighted_results.sort(key=lambda x: x[1], reverse=True)

            return [image_id for image_id, _ in weighted_results]

        except Exception as e:
            raise DSLError(f"Text query execution failed: {e}") from e


class TagFilter(QueryNode):
    """Tag-based filtering query."""

    def __init__(self, tags: List[str], mode: str = "any"):
        """
        Initialize tag filter.

        Args:
            tags: List of tags to filter by
            mode: Filter mode ('any' or 'all')
        """
        assert tags is not None, "Tags list is required"
        assert len(tags) > 0, "Tags list cannot be empty"
        assert mode in ["any", "all"], f"Invalid mode: {mode}"

        self.tags = [tag.strip().lower() for tag in tags]
        self.mode = mode

    def execute(
        self,
        clip_model: CLIPModel,
        search_engine: VectorSearchEngine,
        storage_manager: StorageManager,
        context: Dict[str, Any],
    ) -> List[str]:
        """Execute tag filtering."""
        try:
            # Get all images with pagination
            all_images = []
            offset = 0
            batch_size = 100

            while True:
                batch = storage_manager.list_images(
                    limit=batch_size, offset=offset
                )
                if not batch:
                    break
                all_images.extend(batch)
                offset += batch_size

            # Filter by tags
            matching_ids = []
            for image in all_images:
                image_tags = [tag.lower() for tag in image.tags]

                if self.mode == "any":
                    if any(tag in image_tags for tag in self.tags):
                        matching_ids.append(str(image.id))
                elif self.mode == "all":
                    if all(tag in image_tags for tag in self.tags):
                        matching_ids.append(str(image.id))

            return matching_ids

        except Exception as e:
            raise DSLError(f"Tag filter execution failed: {e}") from e


class AndQuery(QueryNode):
    """Logical AND operation on multiple queries."""

    def __init__(self, operands: List[QueryNode]):
        """
        Initialize AND query.

        Args:
            operands: List of query nodes to combine with AND
        """
        assert operands is not None, "Operands list is required"
        assert len(operands) >= 2, "AND requires at least 2 operands"

        self.operands = operands

    def execute(
        self,
        clip_model: CLIPModel,
        search_engine: VectorSearchEngine,
        storage_manager: StorageManager,
        context: Dict[str, Any],
    ) -> List[str]:
        """Execute AND operation."""
        try:
            # Execute first operand
            result_sets = []
            for operand in self.operands:
                results = operand.execute(
                    clip_model, search_engine, storage_manager, context
                )
                result_sets.append(set(results))

            # Find intersection of all result sets
            intersection = result_sets[0]
            for result_set in result_sets[1:]:
                intersection = intersection.intersection(result_set)

            return list(intersection)

        except Exception as e:
            raise DSLError(f"AND query execution failed: {e}") from e


class OrQuery(QueryNode):
    """Logical OR operation on multiple queries."""

    def __init__(self, operands: List[QueryNode]):
        """
        Initialize OR query.

        Args:
            operands: List of query nodes to combine with OR
        """
        assert operands is not None, "Operands list is required"
        assert len(operands) >= 2, "OR requires at least 2 operands"

        self.operands = operands

    def execute(
        self,
        clip_model: CLIPModel,
        search_engine: VectorSearchEngine,
        storage_manager: StorageManager,
        context: Dict[str, Any],
    ) -> List[str]:
        """Execute OR operation."""
        try:
            # Execute all operands and collect results
            all_results = set()
            for operand in self.operands:
                results = operand.execute(
                    clip_model, search_engine, storage_manager, context
                )
                all_results.update(results)

            return list(all_results)

        except Exception as e:
            raise DSLError(f"OR query execution failed: {e}") from e


class NotQuery(QueryNode):
    """Logical NOT operation (exclusion)."""

    def __init__(self, base_query: QueryNode, exclude_query: QueryNode):
        """
        Initialize NOT query.

        Args:
            base_query: Base query to start with
            exclude_query: Query results to exclude
        """
        assert base_query is not None, "Base query is required"
        assert exclude_query is not None, "Exclude query is required"

        self.base_query = base_query
        self.exclude_query = exclude_query

    def execute(
        self,
        clip_model: CLIPModel,
        search_engine: VectorSearchEngine,
        storage_manager: StorageManager,
        context: Dict[str, Any],
    ) -> List[str]:
        """Execute NOT operation."""
        try:
            # Execute base query
            base_results = set(
                self.base_query.execute(
                    clip_model, search_engine, storage_manager, context
                )
            )

            # Execute exclude query
            exclude_results = set(
                self.exclude_query.execute(
                    clip_model, search_engine, storage_manager, context
                )
            )

            # Return difference
            result = base_results - exclude_results
            return list(result)

        except Exception as e:
            raise DSLError(f"NOT query execution failed: {e}") from e


class DSLParser:
    """
    Parser for Dynamic Search Language queries.

    Supports natural language queries with operators and filters.
    """

    def __init__(self):
        """Initialize DSL parser."""
        # Compile regex patterns for parsing
        self.tag_pattern = re.compile(r"#(\w+)", re.IGNORECASE)
        self.weight_pattern = re.compile(r"\^(\d*\.?\d+)", re.IGNORECASE)
        self.operator_pattern = re.compile(r"\b(AND|OR|NOT)\b", re.IGNORECASE)

    def parse(self, query_string: str) -> QueryNode:
        """
        Parse DSL query string into execution tree.

        Args:
            query_string: DSL query string

        Returns:
            Parsed query node tree

        Raises:
            DSLError: If parsing fails
        """
        assert query_string is not None, "Query string is required"
        assert len(query_string.strip()) > 0, "Query string cannot be empty"

        try:
            # Clean and normalize query
            query = query_string.strip()

            # Handle simple cases first
            if not self._has_operators(query):
                return self._parse_simple_query(query)

            # Parse complex query with operators
            return self._parse_complex_query(query)

        except Exception as e:
            raise DSLError(f"Query parsing failed: {e}") from e

    def _has_operators(self, query: str) -> bool:
        """Check if query contains logical operators."""
        return bool(self.operator_pattern.search(query))

    def _parse_simple_query(self, query: str) -> QueryNode:
        """Parse simple query without operators."""
        # Extract tags
        tags = self.tag_pattern.findall(query)

        # Extract weight
        weight_match = self.weight_pattern.search(query)
        weight = float(weight_match.group(1)) if weight_match else 1.0

        # Remove tags and weight from text
        text_query = self.tag_pattern.sub("", query)
        text_query = self.weight_pattern.sub("", text_query).strip()

        # Create query nodes
        nodes = []

        # Add text query if present
        if text_query:
            nodes.append(TextQuery(text_query, weight))

        # Add tag filter if present
        if tags:
            nodes.append(TagFilter(tags, mode="any"))

        # Combine with AND if multiple nodes
        if len(nodes) == 1:
            return nodes[0]
        elif len(nodes) > 1:
            return AndQuery(nodes)
        else:
            raise DSLError("No valid query components found")

    def _parse_complex_query(self, query: str) -> QueryNode:
        """Parse complex query with operators."""
        # This is a simplified parser - could be expanded with proper
        # tokenization and precedence handling

        # Handle OR operations (lowest precedence)
        or_parts = re.split(r"\bOR\b", query, flags=re.IGNORECASE)
        if len(or_parts) > 1:
            operands = [
                self._parse_and_expression(part.strip()) for part in or_parts
            ]
            return OrQuery(operands)

        # Handle AND operations
        return self._parse_and_expression(query)

    def _parse_and_expression(self, query: str) -> QueryNode:
        """Parse AND expression."""
        and_parts = re.split(r"\bAND\b", query, flags=re.IGNORECASE)
        if len(and_parts) > 1:
            operands = [
                self._parse_not_expression(part.strip()) for part in and_parts
            ]
            return AndQuery(operands)

        return self._parse_not_expression(query)

    def _parse_not_expression(self, query: str) -> QueryNode:
        """Parse NOT expression."""
        not_match = re.search(r"(.+?)\bNOT\b(.+)", query, re.IGNORECASE)
        if not_match:
            base_query = self._parse_simple_query(not_match.group(1).strip())
            exclude_query = self._parse_simple_query(
                not_match.group(2).strip()
            )
            return NotQuery(base_query, exclude_query)

        return self._parse_simple_query(query)


class DSLExecutor:
    """
    Executor for DSL queries.

    Coordinates execution of parsed query trees with system components.
    """

    def __init__(
        self,
        clip_model: CLIPModel,
        search_engine: VectorSearchEngine,
        storage_manager: StorageManager,
    ):
        """
        Initialize DSL executor.

        Args:
            clip_model: CLIP model for embeddings
            search_engine: Vector search engine
            storage_manager: Storage manager
        """
        assert clip_model is not None, "CLIP model is required"
        assert search_engine is not None, "Search engine is required"
        assert storage_manager is not None, "Storage manager is required"

        self.clip_model = clip_model
        self.search_engine = search_engine
        self.storage_manager = storage_manager
        self.parser = DSLParser()

    def execute_query(
        self,
        query_string: str,
        limit: int = 10,
        threshold: float = 0.3,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Execute DSL query and return matching image IDs.

        Args:
            query_string: DSL query string
            limit: Maximum number of results
            threshold: Similarity threshold
            context: Additional execution context

        Returns:
            List of matching image IDs

        Raises:
            DSLError: If execution fails
        """
        assert query_string is not None, "Query string is required"
        assert limit > 0, f"Invalid limit: {limit}"
        assert 0.0 <= threshold <= 1.0, f"Invalid threshold: {threshold}"

        try:
            # Parse query
            query_tree = self.parser.parse(query_string)

            # Prepare execution context
            exec_context = {
                "limit": limit,
                "threshold": threshold,
                "min_score": threshold,
            }
            if context:
                exec_context.update(context)

            # Execute query tree
            results = query_tree.execute(
                self.clip_model,
                self.search_engine,
                self.storage_manager,
                exec_context,
            )

            # Apply final limit
            return results[:limit]

        except Exception as e:
            raise DSLError(f"Query execution failed: {e}") from e
