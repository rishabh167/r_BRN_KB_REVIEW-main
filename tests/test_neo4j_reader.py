"""Unit tests for Neo4jReader query construction.

Verifies that all queries properly filter by is_active to exclude
soft-deleted (disabled) documents and entities.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def reader():
    """Create a Neo4jReader with a mocked Neo4j driver."""
    with patch("app.graph_db.neo4j_reader.GraphDatabase") as mock_gdb, \
         patch("app.graph_db.neo4j_reader.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://fake:7687"
        mock_settings.NEO4J_USERNAME = "neo4j"
        mock_settings.NEO4J_PASSWORD = "test"
        mock_settings.NEO4J_DATABASE = "testdb"

        from app.graph_db.neo4j_reader import Neo4jReader
        r = Neo4jReader()

        # Set up session mock that captures the query
        mock_session = MagicMock()
        mock_session.run.return_value = []
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        r._driver.session.return_value = mock_session

        r._mock_session = mock_session
        yield r


class TestDocumentChunksFiltering:
    def test_filters_by_is_active_true(self, reader):
        reader.get_document_chunks("tenant-1")

        query = reader._mock_session.run.call_args[0][0]
        assert "is_active: true" in query


class TestEntityChunkMappingsFiltering:
    def test_filters_documents_by_is_active_true(self, reader):
        reader.get_entity_chunk_mappings("tenant-1")

        query = reader._mock_session.run.call_args[0][0]
        assert "is_active: true" in query


class TestEntityRelationshipsFiltering:
    def test_filters_entities_by_is_active(self, reader):
        """Entities disabled by the doc_disable API must be excluded."""
        reader.get_entity_relationships("tenant-1")

        query = reader._mock_session.run.call_args[0][0]
        assert "coalesce(a.is_active, true) = true" in query
        assert "coalesce(b.is_active, true) = true" in query

    def test_filters_relationships_by_is_active(self, reader):
        """Relationships disabled by the doc_disable API must be excluded."""
        reader.get_entity_relationships("tenant-1")

        query = reader._mock_session.run.call_args[0][0]
        assert "coalesce(r.is_active, true) = true" in query

    def test_still_excludes_mentions_and_belongs_to(self, reader):
        """The existing type filters must remain intact."""
        reader.get_entity_relationships("tenant-1")

        query = reader._mock_session.run.call_args[0][0]
        assert "MENTIONS" in query
        assert "BELONGS_TO_TENANT" in query
