from typing import NamedTuple, Optional, List
from datetime import datetime
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, PendingWrite
from langchain_core.runnables import RunnableConfig


class CheckpointTuple(NamedTuple):
    """A tuple containing a checkpoint and its associated data."""

    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: Optional[RunnableConfig] = None
    pending_writes: Optional[List[PendingWrite]] = None
    created_at: Optional[datetime] = None
