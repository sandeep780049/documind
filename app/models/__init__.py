"""ORM model registry.

Importing this package registers every table on ``Base.metadata`` so that
Alembic autogenerate and ``create_all`` see the complete schema.
"""

from app.models.chunk import DocumentChunk
from app.models.conversation import Conversation
from app.models.document import Document, DocumentStatus
from app.models.message import Message, MessageRole
from app.models.user import User

__all__ = [
    "Conversation",
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    "Message",
    "MessageRole",
    "User",
]
