import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, Union, Dict

from langchain_core.runnables import RunnableConfig
from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline
from psycopg.errors import UndefinedTable
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    get_checkpoint_id,
)
from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol

Conn = Union[AsyncConnection[DictRow], AsyncConnectionPool[AsyncConnection[DictRow]]]
from langgraph.checkpoint.serde.types import TASKS


from .PostgresBaseSaver import BasePostgresSaver
from .PostgresBaseSaver import (
    MIGRATIONS,
    SELECT_SQL,
    UPSERT_CHECKPOINT_BLOBS_SQL,
    UPSERT_CHECKPOINTS_SQL,
    UPSERT_CHECKPOINT_WRITES_SQL,
    INSERT_CHECKPOINT_WRITES_SQL,
)
from ..common import CheckpointTuple

__all__ = ["AsyncPostgresSaver"]


@asynccontextmanager
async def _get_connection(
    conn: Conn,
) -> AsyncIterator[AsyncConnection[DictRow]]:
    if isinstance(conn, AsyncConnection):
        yield conn
    elif isinstance(conn, AsyncConnectionPool):
        async with conn.connection() as conn:
            yield conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")


class AsyncPostgresSaver(BasePostgresSaver):
    lock: asyncio.Lock

    def __init__(
        self,
        conn: Union[AsyncConnection, AsyncConnectionPool],
        pipe: Optional[AsyncPipeline] = None,
        serde: Optional[SerializerProtocol] = None,
        checkpoint_table: str = "",
        intermediate_table: str = "",
        migration_table: str = "",
        blob_table: str = "",
    ) -> None:
        # Initialize the base class (BaseCheckpointSaver)
        super().__init__(serde=serde)

        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

        self.checkpoint_table = checkpoint_table
        self.writes_table = intermediate_table
        self.blobs_table = blob_table
        self.migration_table = migration_table

        # Format the SQL queries with the correct table names
        self.MIGRATIONS = [
            query.format(
                checkpoint_migrations=self.migration_table,
                checkpoint_table=self.checkpoint_table,
                blobs_table=self.blobs_table,
                writes_table=self.writes_table,
            )
            for query in MIGRATIONS
        ]

        self.SELECT_SQL = SELECT_SQL.format(
            checkpoint_table=self.checkpoint_table,
            blobs_table=self.blobs_table,
            writes_table=self.writes_table,
            TASKS=TASKS,  # Assuming TASKS is already imported and available
        )

        self.UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL.format(
            blobs_table=self.blobs_table
        )

        self.UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL.format(
            checkpoint_table=self.checkpoint_table
        )

        self.UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL.format(
            writes_table=self.writes_table
        )

        self.INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL.format(
            writes_table=self.writes_table
        )

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        serde: Optional[SerializerProtocol] = None,
        table_config: Optional[
            "AsyncPostgresSaver"
        ] = None,  # This param allows table inheritance
    ) -> AsyncIterator["AsyncPostgresSaver"]:
        """Create a new PostgresSaver instance from a connection string.

        Args:
            conn_string (str): The Postgres connection info string.
            pipeline (bool): whether to use AsyncPipeline
            table_config: Optional configuration from an existing AsyncPostgresSaver to inherit table names.

        Returns:
            AsyncPostgresSaver: A new AsyncPostgresSaver instance.
        """
        async with await AsyncConnection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            if pipeline:
                async with conn.pipeline() as pipe:
                    saver = AsyncPostgresSaver(conn=conn, pipe=pipe, serde=serde)
            else:
                saver = AsyncPostgresSaver(conn=conn, serde=serde)

            # If a table configuration is provided, inherit table names
            if table_config:
                saver.checkpoint_table = table_config.checkpoint_table
                saver.writes_table = table_config.writes_table
                saver.migration_table = table_config.migration_table
                saver.blobs_table = table_config.blobs_table

                # Reinitialize with the new table names
                saver.__init__(
                    conn=saver.conn,
                    pipe=saver.pipe,
                    serde=saver.serde,
                    checkpoint_table=table_config.checkpoint_table,
                    intermediate_table=table_config.writes_table,
                    migration_table=table_config.migration_table,
                    blob_table=table_config.blobs_table,
                )

            yield saver

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        async with self._cursor() as cur:
            try:
                results = await cur.execute(
                    f"SELECT v FROM {self.migration_table} ORDER BY v DESC LIMIT 1"
                )
                version = (await results.fetchone())["v"]
            except UndefinedTable:
                version = -1
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
            ):
                await cur.execute(migration)
                await cur.execute(
                    f"INSERT INTO {self.migration_table} (v) VALUES ({v})"
                )
        if self.pipe:
            await self.pipe.sync()

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
            query += " AND user_id = %s"
        with self._cursor() as cur:
            if user_id:
                cur.execute(query, (user_id,))
            else:
                cur.execute(query)
            results = cur.fetchall()
            for result in results:
                # Load metadata as JSON
                thread_id = result["thread_id"]
                metadata_dict = result["metadata"]
                created_at = result["created_at"]
                if not isinstance(metadata_dict, dict):
                    metadata_dict = self.jsonplus_serde.loads(metadata_dict)
                # Extract the message from the "writes" key in metadata
                user_message = metadata_dict["writes"]["__start__"]["messages"][1]
                # Add both user_message and created_at to the map

                thread_message_map[thread_id] = {
                    "user_message": user_message,
                    "created_at": created_at,
                }
        return thread_message_map

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the Postgres database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        # if we change this to use .stream() we need to make sure to close the cursor
        async with self._cursor() as cur:
            await cur.execute(query, args, binary=True)
            async for value in cur:
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": value["checkpoint_ns"],
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint,
                        value["checkpoint"],
                        value["channel_values"],
                        value["pending_sends"],
                    ),
                    self._load_metadata(value["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": value["thread_id"],
                                "checkpoint_ns": value["checkpoint_ns"],
                                "checkpoint_id": value["parent_checkpoint_id"],
                            }
                        }
                        if value["parent_checkpoint_id"]
                        else None
                    ),
                    await asyncio.to_thread(self._load_writes, value["pending_writes"]),
                    value["created_at"],
                )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the Postgres database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        async with self._cursor() as cur:
            await cur.execute(
                self.SELECT_SQL + where,
                args,
                binary=True,
            )

            async for value in cur:
                return CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint,
                        value["checkpoint"],
                        value["channel_values"],
                        value["pending_sends"],
                    ),
                    self._load_metadata(value["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": value["parent_checkpoint_id"],
                            }
                        }
                        if value["parent_checkpoint_id"]
                        else None
                    ),
                    await asyncio.to_thread(self._load_writes, value["pending_writes"]),
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the Postgres database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )
        user_id = configurable.pop("user_id")
        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        async with self._cursor(pipeline=True) as cur:
            await cur.executemany(
                self.UPSERT_CHECKPOINT_BLOBS_SQL,
                await asyncio.to_thread(
                    self._dump_blobs,
                    thread_id,
                    checkpoint_ns,
                    copy.pop("channel_values"),  # type: ignore[misc]
                    new_versions,
                ),
            )
            await cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_id,
                    Jsonb(self._dump_checkpoint(copy)),
                    self._dump_metadata(metadata),
                    user_id,
                ),
            )
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
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
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        params = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            writes,
            config["configurable"]["user_id"],
        )
        async with self._cursor(pipeline=True) as cur:
            await cur.executemany(query, params)

    @asynccontextmanager
    async def _cursor(
        self, *, pipeline: bool = False
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        async with _get_connection(self.conn) as conn:
            if self.pipe:
                # a connection in pipeline mode can be used concurrently
                # in multiple threads/coroutines, but only one cursor can be
                # used at a time
                try:
                    async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                finally:
                    if pipeline:
                        await self.pipe.sync()
            elif pipeline:
                # a connection not in pipeline mode can only be used by one
                # thread/coroutine at a time, so we acquire a lock
                async with self.lock, conn.pipeline(), conn.cursor(
                    binary=True, row_factory=dict_row
                ) as cur:
                    yield cur
            else:
                async with self.lock, conn.cursor(
                    binary=True, row_factory=dict_row
                ) as cur:
                    yield cur

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the Postgres database based
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

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the Postgres database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
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
                    "Synchronous calls to AsyncPostgresSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the Postgres database. The checkpoint is associated
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
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id), self.loop
        ).result()
