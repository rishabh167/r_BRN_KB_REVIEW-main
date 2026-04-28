import logging
from neo4j import GraphDatabase
from app.core import settings

logger = logging.getLogger("kb_review")


class Neo4jReader:
    """Read-only access to Neo4j for loading Document chunks and Entity data."""

    def __init__(self):
        self._driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        )
        self._database = settings.NEO4J_DATABASE

    def close(self):
        self._driver.close()

    # ------------------------------------------------------------------
    # Document chunks
    # ------------------------------------------------------------------

    def get_document_chunks(self, tenant_id: str) -> list[dict]:
        """Return all active Document nodes for a tenant.

        Each dict contains: text, source, page, split_id, document_id,
        document_content_id, embedding (may be None), is_active.
        """
        query = """
        MATCH (d:Document {tenant_id: $tenant_id, is_active: true})
        RETURN d.text          AS text,
               d.source        AS source,
               d.page          AS page,
               d.split_id      AS split_id,
               d.document_id   AS document_id,
               d.document_content_id AS document_content_id,
               d.embedding      AS embedding,
               d.total_pages    AS total_pages
        ORDER BY d.source, d.page, d.split_id
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query, tenant_id=tenant_id)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Entities linked to documents
    # ------------------------------------------------------------------

    def get_entity_chunk_mappings(self, tenant_id: str) -> list[dict]:
        """Return (entity_id, entity_type, document_id, page, split_id) tuples.

        This tells us which entities appear in which document chunks.
        """
        query = """
        MATCH (d:Document {tenant_id: $tenant_id, is_active: true})
              -[:MENTIONS]->(e:__Entity__ {tenant_id: $tenant_id})
        RETURN e.id        AS entity_id,
               labels(e)   AS entity_labels,
               d.document_id AS document_id,
               d.source      AS source,
               d.page        AS page,
               d.split_id    AS split_id
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query, tenant_id=tenant_id)
            return [dict(record) for record in result]

    def get_entity_relationships(self, tenant_id: str) -> list[dict]:
        """Return entity-to-entity relationships for the tenant.

        Only returns relationships where both entities and the relationship
        itself are active (is_active is true or unset).  Documents that were
        soft-deleted via the disable API have is_active=false on their
        orphaned entities and relationships — this filter excludes them.

        Each dict: source_entity, target_entity, relationship_type.
        """
        query = """
        MATCH (a:__Entity__ {tenant_id: $tenant_id})
              -[r]->(b:__Entity__ {tenant_id: $tenant_id})
        WHERE type(r) <> 'MENTIONS' AND type(r) <> 'BELONGS_TO_TENANT'
          AND coalesce(a.is_active, true) = true
          AND coalesce(b.is_active, true) = true
          AND coalesce(r.is_active, true) = true
        RETURN a.id  AS source_entity,
               b.id  AS target_entity,
               type(r) AS relationship_type
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query, tenant_id=tenant_id)
            return [dict(record) for record in result]


neo4j_reader = Neo4jReader()
