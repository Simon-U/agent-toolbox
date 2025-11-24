"""
Memory/checkpoint storage factory for LangGraph applications.

This module provides the Memory() factory function for creating checkpoint savers
that persist agent state and conversation history across sessions.

Supported backends:
- memory: In-memory storage (for testing/development)
- sqlite: Local SQLite database (for single-machine persistence)
- asyncsqlite: Async SQLite database (for non-blocking single-machine persistence)
- postgres: PostgreSQL database (for distributed production systems)
"""

from os import environ

from .memory_config import (
    MemoryType,
    PostgresType,
    MemoryConfig,
    SqliteConfig,
    PostgresConfig,
)
from .memory_handlers import MemoryFactory

__all__ = ["Memory"]


def Memory(
    type_of=environ.get("DB", "memory"),
    sqlite_path=None,
    postgres_type="sync_single",
    postgres_connection=None,
    max_pool=8,
    checkpoint_table: str = "checkpoints",
    intermediate_table: str = "writes",
    migration_table: str = "checkpoint_migrations",
    blob_table: str = "blobs",
):
    """
    Memory checkpoint for graph applications. This class saves the previous input history
    and allows continuing a conversion. It supports multiple storage backends such as in-memory,
    SQLite, and PostgreSQL, with both synchronous and asynchronous PostgreSQL connection types.

    Supported Types:
    - 'memory': Default type, saves in local memory.
    - 'sqlite': Saves in a local SQLite database. Requires a valid `sqlite_path`.
    - 'postgres': Saves in a PostgreSQL database. Supports the following connection types:
        - 'sync_single': Single synchronous PostgreSQL connection.
        - 'sync_pool': Connection pool for synchronous PostgreSQL connections.
        - 'async_single': Single asynchronous PostgreSQL connection.
        - 'async_pool': Connection pool for asynchronous PostgreSQL connections.

    Parameters:
    ----------
    type : str
        The type of memory to use. It must be one of 'memory', 'sqlite', or 'postgres'.
    sqlite_path : str, optional
        The file path to the SQLite database. Required if `type` is 'sqlite'.
    postgres_type : str, optional
        The type of PostgreSQL connection to use if `type` is 'postgres'. Must be one of
        'sync_single', 'sync_pool', 'async_single', or 'async_pool'.
    postgres_connection : str, optional
        The connection string to the PostgreSQL database. Required if `type` is 'postgres'.

    Returns:
    --------
    A saver object with appropriate backend handling based on the provided type:
    - `MemorySaver` for in-memory storage
    - `SqliteSaver` for SQLite storage
    - `PostgresSaver` for synchronous PostgreSQL storage
    - `AsyncPostgresSaver` for asynchronous PostgreSQL storage
    - `max_pool` defines the max connection size for pooling connections. Only relevant to Postgres pools
    - `checkpoint_table` refers to the table name for the checkpoint table. Relevant for SQLite / Postgres
    - `intermediate_table` refers to the table name for the intermediate steps table. Relevant for SQLite / Postgres
    - `migration_table` refers to the table name for the migrations table. Relevant for Postgres
    - `blob_table` refers to the table name for the state value blob table. Relevant for Postgres

    ### Example Usage:
    --------------
    #### In-memory storage
        >>> checkpointer = Memory(type="memory")
        >>> graph = your_agent.create_agent(model=model, checkpointer=checkpointer)
        >>> config = {"configurable": {"thread_id": "1"}}
        >>> res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)
        >>> checkpoint = graph.get_state(config)

    #### SQLite storage
        >>> checkpointer = Memory(type="sqlite", sqlite_path="/path/to/sqlite.db")
        >>> graph = your_agent.create_agent(model=model, checkpointer=checkpointer)
        >>> config = {"configurable": {"thread_id": "1"}}
        >>> res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)
        >>> checkpoint = graph.get_state(config)

    #### PostgreSQL storage (synchronous, single connection)
        >>> checkpointer = Memory(type="postgres", postgres_type="sync_single", postgres_connection="connection_string")
        >>> with checkpointer.conn as pool:
            >>> checkpointer.setup()
            >>> graph = your_agent.create_agent(model=model, checkpointer=checkpointer)
            >>> config = {"configurable": {"thread_id": "1"}}
            >>> res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)
            >>> checkpoint = checkpointer.get(config)

    #### PostgreSQL storage (asynchronous, connection pool)
        >>> checkpointer = Memory(type="postgres", postgres_type="async_pool", postgres_connection="connection_string")
        >>> async with checkpointer.conn as pool:
            >>> await checkpointer.setup()
            >>> graph = your_agent.create_agent(model=model, checkpointer=checkpointer)
            >>> config = {"configurable": {"thread_id": "1"}}
            >>> res = await graph.ainvoke({"messages": [("human", "what's the weather in sf")]}, config)
            >>> checkpoint = await checkpointer.aget(config)
    """
    # Convert string type to MemoryType enum
    try:
        memory_type = MemoryType(type_of)
    except ValueError:
        supported = ", ".join([t.value for t in MemoryType])
        raise ValueError(
            f"Unsupported memory type '{type_of}'. "
            f"Supported types: {supported}"
        )

    # Handle in-memory (no config needed)
    if memory_type == MemoryType.MEMORY:
        handler = MemoryFactory.get_handler(memory_type)
        return handler.get_saver(MemoryConfig())

    # Handle SQLite and AsyncSQLite
    if memory_type in [MemoryType.SQLITE, MemoryType.ASYNCSQLITE]:
        config = SqliteConfig(
            sqlite_path=sqlite_path,
            checkpoint_table=checkpoint_table,
            intermediate_table=intermediate_table,
        )
        handler = MemoryFactory.get_handler(memory_type)
        return handler.get_saver(config)

    # Handle PostgreSQL
    if memory_type == MemoryType.POSTGRES:
        # Convert postgres_type string to enum
        try:
            pg_type = PostgresType(postgres_type)
        except ValueError:
            supported = ", ".join([t.value for t in PostgresType])
            raise ValueError(
                f"postgres_type must be one of {supported}. "
                f"Got: {postgres_type}"
            )

        config = PostgresConfig(
            postgres_type=pg_type,
            postgres_connection=postgres_connection,
            max_pool=max_pool,
            checkpoint_table=checkpoint_table,
            intermediate_table=intermediate_table,
            migration_table=migration_table,
            blob_table=blob_table,
        )
        handler = MemoryFactory.get_handler(memory_type)
        return handler.get_saver(config)
