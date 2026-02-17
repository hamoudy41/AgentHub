"""add_document_chunks_for_rag

Revision ID: d7ffa8f9c684
Revises: f907662aa0d6
Create Date: 2026-02-16 12:06:43.593424

"""

import logging
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

logger = logging.getLogger("alembic.runtime.migration")


# revision identifiers, used by Alembic.
revision: str = "d7ffa8f9c684"
down_revision: Union[str, Sequence[str], None] = "f907662aa0d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TRACKING_TABLE = "_alembic_d7ffa8f9c684_created"


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()
    created: list[str] = []

    if "ai_call_audit" not in tables:
        op.create_table(
            "ai_call_audit",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("flow_name", sa.String(length=128), nullable=False),
            sa.Column("request_payload", sa.JSON(), nullable=False),
            sa.Column("response_payload", sa.JSON(), nullable=True),
            sa.Column("success", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("tenant_id", sa.String(length=64), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_ai_call_audit_flow_name"), "ai_call_audit", ["flow_name"], unique=False
        )
        op.create_index(
            op.f("ix_ai_call_audit_tenant_id"), "ai_call_audit", ["tenant_id"], unique=False
        )
        created.append("ai_call_audit")

    if "document_chunks" not in tables:
        op.create_table(
            "document_chunks",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("document_id", sa.String(length=64), nullable=False),
            sa.Column("chunk_index", sa.Integer(), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("embedding", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("tenant_id", sa.String(length=64), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_document_chunks_document_id"), "document_chunks", ["document_id"], unique=False
        )
        op.create_index(
            op.f("ix_document_chunks_tenant_id"), "document_chunks", ["tenant_id"], unique=False
        )
        created.append("document_chunks")

    if "documents" not in tables:
        op.create_table(
            "documents",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("tenant_id", sa.String(length=64), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_documents_tenant_id"), "documents", ["tenant_id"], unique=False)
        created.append("documents")

    if created:
        if _TRACKING_TABLE not in tables:
            op.create_table(
                _TRACKING_TABLE,
                sa.Column("table_name", sa.String(length=64), nullable=False),
                sa.PrimaryKeyConstraint("table_name"),
            )
        dialect = conn.dialect.name
        for t in created:
            if dialect == "sqlite":
                op.execute(
                    sa.text(
                        f"INSERT OR IGNORE INTO {_TRACKING_TABLE} (table_name) VALUES (:t)"
                    ).bindparams(t=t)
                )
            else:
                op.execute(
                    sa.text(
                        f"INSERT INTO {_TRACKING_TABLE} (table_name) VALUES (:t) "
                        "ON CONFLICT (table_name) DO NOTHING"
                    ).bindparams(t=t)
                )


def downgrade() -> None:
    """Downgrade schema. Only drops tables that this migration created."""
    logger.info("Starting migration downgrade for d7ffa8f9c684")
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    created: list[str] = []
    if _TRACKING_TABLE in tables:
        result = conn.execute(sa.text(f"SELECT table_name FROM {_TRACKING_TABLE}"))
        created = [row[0] for row in result]
        logger.info(f"Found tracking table with {len(created)} created table(s): {created}")
    else:
        logger.warning(f"Tracking table '{_TRACKING_TABLE}' not found - no tables will be dropped")

    for table_name in ["documents", "document_chunks", "ai_call_audit"]:
        if table_name in created and table_name in tables:
            try:
                logger.info(f"Dropping table '{table_name}' and its indexes")
                if table_name == "documents":
                    op.drop_index(op.f("ix_documents_tenant_id"), table_name="documents")
                elif table_name == "document_chunks":
                    op.drop_index(op.f("ix_document_chunks_tenant_id"), table_name="document_chunks")
                    op.drop_index(op.f("ix_document_chunks_document_id"), table_name="document_chunks")
                elif table_name == "ai_call_audit":
                    op.drop_index(op.f("ix_ai_call_audit_tenant_id"), table_name="ai_call_audit")
                    op.drop_index(op.f("ix_ai_call_audit_flow_name"), table_name="ai_call_audit")
                op.drop_table(table_name)
                logger.info(f"Successfully dropped table '{table_name}'")
            except Exception as e:
                logger.error(f"Failed to drop table '{table_name}': {e}")
                raise
        elif table_name not in created:
            logger.warning(f"Skipping table '{table_name}' - not created by this migration")
        elif table_name not in tables:
            logger.warning(f"Skipping table '{table_name}' - table does not exist in database")

    if _TRACKING_TABLE in tables:
        try:
            logger.info(f"Dropping tracking table '{_TRACKING_TABLE}'")
            op.drop_table(_TRACKING_TABLE)
            logger.info(f"Successfully dropped tracking table '{_TRACKING_TABLE}'")
        except Exception as e:
            logger.error(f"Failed to drop tracking table '{_TRACKING_TABLE}': {e}")
            raise
    
    logger.info("Migration downgrade for d7ffa8f9c684 completed successfully")
