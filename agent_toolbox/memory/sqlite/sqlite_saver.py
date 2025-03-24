import sqlite3
import queue
import threading
from contextlib import closing
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple


from langgraph.checkpoint.sqlite import SqliteSaver as BaseSQL
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    SerializerProtocol,
    get_checkpoint_id,
)
from langgraph.checkpoint.sqlite.utils import search_where

from typing import Optional
from ..common import CheckpointTuple


class CustomSqliteSaver(BaseSQL):
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        serde: Optional[SerializerProtocol] = None,
        checkpoint_table: str = "checkpoints",
        intermediate_table: str = "writes",
    ) -> None:
        # Initialize the base class (BaseCheckpointSaver)
        super().__init__(conn=conn, serde=serde)

        # Set custom table names (checkpoint_table and writes_table)
        self.checkpoint_table = checkpoint_table
        self.intermediate_table = intermediate_table

    def setup(self) -> None:
        """Set up the checkpoint database.

        This method creates the necessary tables in the SQLite database if they don't
        already exist. It is called automatically when needed and should not be called
        directly by the user.
        """
        if self.is_setup:
            return

        # Use dynamic table names for checkpoints and writes tables
        self.conn.executescript(
            f"""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS {self.checkpoint_table} (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata BLOB,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                );
                CREATE TABLE IF NOT EXISTS {self.intermediate_table} (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    value BLOB,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                );
                """
        )

        self.is_setup = True

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the SQLite database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        with self.cursor(transaction=False) as cur:
            # find the latest checkpoint for the thread_id
            if checkpoint_id := get_checkpoint_id(config):
                cur.execute(
                    f"SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at FROM {self.checkpoint_table} WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        checkpoint_id,
                    ),
                )
            else:
                cur.execute(
                    f"SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at FROM {self.checkpoint_table} WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1",
                    (str(config["configurable"]["thread_id"]), checkpoint_ns),
                )

            # if a checkpoint is found, return it
            if value := cur.fetchone():

                (
                    thread_id,
                    checkpoint_id,
                    parent_checkpoint_id,
                    type,
                    checkpoint,
                    metadata,
                    created_at,  # Added created_at field here
                ) = value

                if not get_checkpoint_id(config):
                    config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    }

                # find any pending writes using dynamic table name
                cur.execute(
                    f"SELECT task_id, channel, type, value FROM {self.intermediate_table} WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx",
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        str(config["configurable"]["checkpoint_id"]),
                    ),
                )

                # deserialize the checkpoint and metadata
                return CheckpointTuple(
                    config,
                    self.serde.loads_typed((type, checkpoint)),
                    self.jsonplus_serde.loads(metadata) if metadata is not None else {},
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        }
                        if parent_checkpoint_id
                        else None
                    ),
                    [
                        (task_id, channel, self.serde.loads_typed((type, value)))
                        for task_id, channel, type, value in cur
                    ],
                    created_at=created_at,  # Return the created_at field in CheckpointTuple
                )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the SQLite database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata. Defaults to None.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        where, param_values = search_where(config, filter, before)

        # Add user_id to the query if it's in the config
        user_id = config["configurable"].get("user_id") if config else None
        if user_id:
            where += " AND user_id = ?"
            param_values.append(user_id)

        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at
        FROM {self.checkpoint_table}
        {where}
        ORDER BY checkpoint_id DESC"""

        if limit:
            query += f" LIMIT {limit}"

        with self.cursor(transaction=False) as cur, closing(self.conn.cursor()) as wcur:
            cur.execute(query, param_values)
            for (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                type,
                checkpoint,
                metadata,
                created_at,
            ) in cur:
                wcur.execute(
                    f"SELECT task_id, channel, type, value FROM {self.intermediate_table} WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    self.serde.loads_typed((type, checkpoint)),
                    self.jsonplus_serde.loads(metadata) if metadata is not None else {},
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        }
                        if parent_checkpoint_id
                        else None
                    ),
                    [
                        (task_id, channel, self.serde.loads_typed((type, value)))
                        for task_id, channel, type, value in wcur
                    ],
                    created_at=created_at,
                )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the SQLite database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Examples:

            >>> from langgraph.checkpoint.sqlite import SqliteSaver
            >>> with SqliteSaver.from_conn_string(":memory:") as memory:
            >>>     config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            >>>     checkpoint = {"ts": "2024-05-04T06:32:42.235444+00:00", "id": "1ef4f797-8335-6428-8001-8a1503f9b875", "data": {"key": "value"}}
            >>>     saved_config = memory.put(config, checkpoint, {"source": "input", "step": 1, "writes": {"key": "value"}}, {})
            >>> print(saved_config)
            {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef4f797-8335-6428-8001-8a1503f9b875'}}
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.jsonplus_serde.dumps(metadata)
        with self.cursor() as cur:
            cur.execute(
                f"INSERT OR REPLACE INTO {self.checkpoint_table} (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(config["configurable"]["thread_id"]),
                    checkpoint_ns,
                    checkpoint["id"],
                    config["configurable"].get("checkpoint_id"),
                    type_,
                    serialized_checkpoint,
                    serialized_metadata,
                ),
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the SQLite database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        with self.cursor() as cur:
            cur.executemany(
                f"INSERT OR IGNORE INTO {self.intermediate_table} (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        str(config["configurable"]["thread_id"]),
                        str(config["configurable"]["checkpoint_ns"]),
                        str(config["configurable"]["checkpoint_id"]),
                        task_id,
                        WRITES_IDX_MAP.get(channel, idx),
                        channel,
                        *self.serde.dumps_typed(value),
                    )
                    for idx, (channel, value) in enumerate(writes)
                ],
            )

    def get_ids(self, user_id: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """Get all unique thread IDs where parent_checkpoint_id is NULL, and optionally filter by user_id.

        Extract the user message from metadata and return the created_at timestamp.

        Args:
            user_id (Optional[str]): The ID of the user to filter by. If None, no filtering is applied.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary where the keys are thread IDs and the values are dictionaries
                                    containing the user messages and the created_at timestamp.
        """
        thread_message_map = {}

        # Base query to get thread_id, metadata, and created_at where parent_checkpoint_id is NULL
        query = f"""
            SELECT thread_id, metadata, created_at
            FROM {self.checkpoint_table}
            WHERE parent_checkpoint_id IS NULL
        """

        # Add user_id filtering to the query if user_id is provided
        if user_id:
            query += " AND user_id = ?"
        with self.cursor(transaction=False) as cur:
            if user_id:
                cur.execute(query, (user_id,))
            else:
                cur.execute(query)
            results = cur.fetchall()
            for thread_id, metadata, created_at in results:
                # Load metadata as JSON
                metadata_dict = self.jsonplus_serde.loads(metadata)
                # Extract the message from the "writes" key in metadata
                user_message = metadata_dict["writes"]["__start__"]["messages"][1]
                # Add both user_message and created_at to the map

                thread_message_map[thread_id] = {
                    "user_message": user_message,
                    "created_at": created_at,
                }
        return thread_message_map
