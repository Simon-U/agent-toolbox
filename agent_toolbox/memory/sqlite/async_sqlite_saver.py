import aiosqlite
import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as BaseSQL
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Optional,
    Sequence,
    Tuple,
)

import aiosqlite
from langchain_core.runnables import RunnableConfig
from contextlib import asynccontextmanager
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    maybe_add_typed_methods,
)
from langgraph.checkpoint.sqlite.utils import search_where
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import asyncio
import random
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import aiosqlite
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import ChannelProtocol
from langgraph.checkpoint.sqlite.utils import search_where
from ..common import CheckpointTuple

T = TypeVar("T", bound=Callable)

__all__ = ["CustomAsyncSqliteSaver"]


class CustomAsyncSqliteSaver(BaseCheckpointSaver[str]):
    lock: asyncio.Lock
    is_setup: bool

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        serde: Optional[SerializerProtocol] = None,
        checkpoint_table: str = "checkpoints",
        intermediate_table: str = "writes",
    ):
        super().__init__(serde=serde)
        self.checkpoint_table = checkpoint_table
        self.intermediate_table = intermediate_table
        self.jsonplus_serde = JsonPlusSerializer()
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.is_setup = False

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, conn_string: str
    ) -> AsyncIterator["CustomAsyncSqliteSaver"]:
        """Create a new AsyncSqliteSaver instance from a connection string.

        Args:
            conn_string (str): The SQLite connection string.

        Yields:
            AsyncSqliteSaver: A new AsyncSqliteSaver instance.
        """
        async with aiosqlite.connect(conn_string) as conn:
            yield CustomAsyncSqliteSaver(conn)

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
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncSqliteSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the SQLite database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
        """
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

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
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str
    ) -> None:
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id), self.loop
        ).result()

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the SQLite database if they don't
        already exist. It is called automatically when needed and should not be called
        directly by the user.
        """
        if self.is_setup:
            return

        async with self.lock:
            if self.is_setup:  # Double-check pattern
                return
            if not self.conn.is_alive():
                await self.conn
            async with self.conn.executescript(
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
            ):
                await self.conn.commit()

            self.is_setup = True

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the SQLite database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        await self.setup()
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        async with self.lock, self.conn.cursor() as cur:
            # find the latest checkpoint for the thread_id
            if checkpoint_id := get_checkpoint_id(config):
                await cur.execute(
                    "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        checkpoint_id,
                    ),
                )
            else:
                await cur.execute(
                    "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1",
                    (str(config["configurable"]["thread_id"]), checkpoint_ns),
                )
            # if a checkpoint is found, return it
            if value := await cur.fetchone():
                (
                    thread_id,
                    checkpoint_id,
                    parent_checkpoint_id,
                    type,
                    checkpoint,
                    metadata,
                    created_at,  # Add created_at field here
                ) = value

                if not get_checkpoint_id(config):
                    config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    }
                # find any pending writes
                await cur.execute(
                    "SELECT task_id, channel, type, value FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx",
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
                        async for task_id, channel, type, value in cur
                    ],
                    created_at=created_at,  # Add created_at in the CheckpointTuple return
                )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the SQLite database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        await self.setup()
        where, params = search_where(config, filter, before)
        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at
        FROM {self.checkpoint_table}
        {where}
        ORDER BY checkpoint_id DESC"""
        if limit:
            query += f" LIMIT {limit}"
        async with self.lock, self.conn.execute(
            query, params
        ) as cur, self.conn.cursor() as wcur:
            async for (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                type,
                checkpoint,
                metadata,
                created_at,
            ) in cur:
                await wcur.execute(
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
                        async for task_id, channel, type, value in wcur
                    ],
                    created_at=created_at,
                )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        query = (
            f"INSERT OR REPLACE INTO {self.intermediate_table} (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else f"INSERT OR IGNORE INTO {self.intermediate_table} (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        await self.setup()

        # Get the user_id from the config, default to None if not present
        user_id = str(config["configurable"].get("user_id", None))

        async with self.lock, self.conn.cursor() as cur:
            await cur.executemany(
                query,
                [
                    (
                        str(config["configurable"]["thread_id"]),
                        str(config["configurable"]["checkpoint_ns"]),
                        str(config["configurable"]["checkpoint_id"]),
                        task_id,
                        WRITES_IDX_MAP.get(channel, idx),
                        channel,
                        *self.serde.dumps_typed(value),
                        user_id,  # Add user_id as the last value
                    )
                    for idx, (channel, value) in enumerate(writes)
                ],
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the SQLite database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.jsonplus_serde.dumps(metadata)

        # Get the user_id from the config, default to None if not present
        user_id = str(config["configurable"].get("user_id", None))

        async with self.lock, self.conn.execute(
            f"INSERT OR REPLACE INTO {self.checkpoint_table} (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(config["configurable"]["thread_id"]),
                checkpoint_ns,
                checkpoint["id"],
                config["configurable"].get("checkpoint_id"),
                type_,
                serialized_checkpoint,
                serialized_metadata,
                user_id,  # Add user_id here
            ),
        ):
            await self.conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the SQLite database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        await self.setup()
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        user_id = config["configurable"].get(
            "user_id"
        )  # Get user_id from config if present

        async with self.lock, self.conn.cursor() as cur:
            # find the latest checkpoint for the thread_id
            if checkpoint_id := get_checkpoint_id(config):
                if user_id:  # If user_id is present, filter by it
                    await cur.execute(
                        f"SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM {self.checkpoint_table} WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? AND user_id = ?",
                        (
                            str(config["configurable"]["thread_id"]),
                            checkpoint_ns,
                            checkpoint_id,
                            user_id,  # Include user_id in query parameters
                        ),
                    )
                else:
                    await cur.execute(
                        f"SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM {self.checkpoint_table} WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                        (
                            str(config["configurable"]["thread_id"]),
                            checkpoint_ns,
                            checkpoint_id,
                        ),
                    )
            else:
                if user_id:  # If user_id is present, filter by it
                    await cur.execute(
                        f"SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM {self.checkpoint_table} WHERE thread_id = ? AND checkpoint_ns = ? AND user_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                        (
                            str(config["configurable"]["thread_id"]),
                            checkpoint_ns,
                            user_id,  # Include user_id in query parameters
                        ),
                    )
                else:
                    await cur.execute(
                        f"SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM {self.checkpoint_table} WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1",
                        (
                            str(config["configurable"]["thread_id"]),
                            checkpoint_ns,
                        ),
                    )
            # if a checkpoint is found, return it
            if value := await cur.fetchone():
                (
                    thread_id,
                    checkpoint_id,
                    parent_checkpoint_id,
                    type,
                    checkpoint,
                    metadata,
                ) = value
                if not get_checkpoint_id(config):
                    config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    }
                # find any pending writes
                await cur.execute(
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
                        async for task_id, channel, type, value in cur
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

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        """Generate the next version ID for a channel.

        This method creates a new version identifier for a channel based on its current version.

        Args:
            current (Optional[str]): The current version identifier of the channel.
            channel (BaseChannel): The channel being versioned.

        Returns:
            str: The next version identifier, which is guaranteed to be monotonically increasing.
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
