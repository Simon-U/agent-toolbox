import random
from typing import Any, List, Optional, Sequence, Tuple, cast, NamedTuple

from langchain_core.runnables import RunnableConfig
from psycopg.types.json import Jsonb

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import TASKS, ChannelProtocol

MetadataInput = Optional[dict[str, Any]]

"""
To add a new migration, add a new string to the MIGRATIONS list.
The position of the migration in the list is the version number.
"""
MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS {checkpoint_migrations} (
    v INTEGER PRIMARY KEY
);""",
    """CREATE TABLE IF NOT EXISTS {checkpoint_table} (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{{}}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);""",
    """CREATE TABLE IF NOT EXISTS {blobs_table} (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);""",
    """CREATE TABLE IF NOT EXISTS {writes_table} (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);""",
    "ALTER TABLE {blobs_table} ALTER COLUMN blob DROP not null;",
]

SELECT_SQL = """
SELECT
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    created_at,
    user_id,
    (
        SELECT array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob])
        FROM jsonb_each_text(checkpoint -> 'channel_versions')
        INNER JOIN {blobs_table} bl
            ON bl.thread_id = {checkpoint_table}.thread_id
            AND bl.checkpoint_ns = {checkpoint_table}.checkpoint_ns
            AND bl.channel = jsonb_each_text.key
            AND bl.version = jsonb_each_text.value
    ) AS channel_values,
    (
        SELECT
        array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] ORDER BY cw.task_id, cw.idx)
        FROM {writes_table} cw
        WHERE cw.thread_id = {checkpoint_table}.thread_id
            AND cw.checkpoint_ns = {checkpoint_table}.checkpoint_ns
            AND cw.checkpoint_id = {checkpoint_table}.checkpoint_id
    ) AS pending_writes,
    (
        SELECT array_agg(array[cw.type::bytea, cw.blob] ORDER BY cw.idx)
        FROM {writes_table} cw
        WHERE cw.thread_id = {checkpoint_table}.thread_id
            AND cw.checkpoint_ns = {checkpoint_table}.checkpoint_ns
            AND cw.checkpoint_id = {checkpoint_table}.parent_checkpoint_id
            AND cw.channel = '{TASKS}'
    ) AS pending_sends
FROM {checkpoint_table}
"""

UPSERT_CHECKPOINT_BLOBS_SQL = """
    INSERT INTO {blobs_table} (thread_id, checkpoint_ns, channel, version, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING
"""

UPSERT_CHECKPOINTS_SQL = """
    INSERT INTO {checkpoint_table} (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata, user_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
    DO UPDATE SET
        checkpoint = EXCLUDED.checkpoint,
        metadata = EXCLUDED.metadata;
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO {writes_table} (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob, user_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
        channel = EXCLUDED.channel,
        type = EXCLUDED.type,
        blob = EXCLUDED.blob;
"""

INSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO {writes_table} (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob, user_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
"""


class BasePostgresSaver(BaseCheckpointSaver[str]):
    SELECT_SQL = SELECT_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    jsonplus_serde = JsonPlusSerializer()

    def _load_checkpoint(
        self,
        checkpoint: dict[str, Any],
        channel_values: list[tuple[bytes, bytes, bytes]],
        pending_sends: list[tuple[bytes, bytes]],
    ) -> Checkpoint:
        return {
            **checkpoint,
            "pending_sends": [
                self.serde.loads_typed((c.decode(), b)) for c, b in pending_sends or []
            ],
            "channel_values": self._load_blobs(channel_values),
        }

    def _dump_checkpoint(self, checkpoint: Checkpoint) -> dict[str, Any]:
        return {**checkpoint, "pending_sends": []}

    def _load_blobs(
        self, blob_values: list[tuple[bytes, bytes, bytes]]
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k.decode(): self.serde.loads_typed((t.decode(), v))
            for k, t, v in blob_values
            if t.decode() != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, Optional[bytes]]]:
        if not versions:
            return []

        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                cast(str, ver),
                *(
                    self.serde.dumps_typed(values[k])
                    if k in values
                    else ("empty", None)
                ),
            )
            for k, ver in versions.items()
        ]

    def _load_writes(
        self, writes: list[tuple[bytes, bytes, bytes, bytes]]
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid.decode(),
                    channel.decode(),
                    self.serde.loads_typed((t.decode(), v)),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        writes: Sequence[tuple[str, Any]],
        user_id: str,  # Add user_id parameter
    ) -> list[
        tuple[str, str, str, str, int, str, str, bytes, str]
    ]:  # Update return type hint
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
                user_id,  # Add user_id to the tuple
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def _load_metadata(self, metadata: dict[str, Any]) -> CheckpointMetadata:
        return self.jsonplus_serde.loads(self.jsonplus_serde.dumps(metadata))

    def _dump_metadata(self, metadata: CheckpointMetadata) -> str:
        serialized_metadata = self.jsonplus_serde.dumps(metadata)
        # NOTE: we're using JSON serializer (not msgpack), so we need to remove null characters before writing
        return serialized_metadata.decode().replace("\\u0000", "")

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _search_where(
        self,
        config: Optional[RunnableConfig],
        filter: MetadataInput,
        before: Optional[RunnableConfig] = None,
    ) -> Tuple[str, List[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before.

        This method returns a tuple of a string and a tuple of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = $1 AND column2 IS $2". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            user_id = int(config["configurable"].get("user_id"))
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = %s")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = %s ")
                param_values.append(int(checkpoint_id))

            if user_id is not None:
                wheres.append("user_id = %s ")
                param_values.append(user_id)

        # construct predicate for metadata filter
        if filter:
            wheres.append("metadata @> %s ")
            param_values.append(Jsonb(filter))

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < %s ")
            param_values.append(get_checkpoint_id(before))

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )


PendingWrite = Tuple[str, str, Any]


class CheckpointTuple(NamedTuple):
    """A tuple containing a checkpoint and its associated data."""

    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: Optional[RunnableConfig] = None
    pending_writes: Optional[List[PendingWrite]] = None
