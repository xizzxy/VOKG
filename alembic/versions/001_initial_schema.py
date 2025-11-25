"""Initial database schema

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # This will be auto-generated, but here's the structure
    # Users table created by models
    # Videos table created by models
    # ProcessingJobs table created by models
    # Frames table created by models
    # DetectedObjects table created by models
    # Interactions table created by models
    # GraphSnapshots table created by models
    pass


def downgrade() -> None:
    pass
