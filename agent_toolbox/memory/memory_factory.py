import sqlite3
from os import environ

from psycopg import Connection, AsyncConnection
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.memory import MemorySaver
from ..connectors.database.database_connector import DatabaseConnector
from .sqlite.sqlite_saver import CustomSqliteSaver
from .sqlite.async_sqlite_saver import CustomAsyncSqliteSaver
from .postgres.sync_saver import PostgresSaver
from .postgres.async_saver import AsyncPostgresSaver

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
    if type_of == "memory":
        return MemorySaver()

    elif type_of == "sqlite":
        if not sqlite_path:
            raise ValueError("Please specify a sqlite_path for the database.")
        conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        return CustomSqliteSaver(
            conn,
            checkpoint_table=checkpoint_table,
            intermediate_table=intermediate_table,
        )
    elif type_of == "asyncsqlite":
        if not sqlite_path:
            raise ValueError("Please specify a sqlite_path for the database.")
        conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        return CustomAsyncSqliteSaver(
            conn,
            checkpoint_table=checkpoint_table,
            intermediate_table=intermediate_table,
        )

    elif type_of == "postgres":
        if postgres_type not in [
            "sync_single",
            "sync_pool",
            "async_single",
            "async_pool",
        ]:
            raise ValueError(
                "postgres_type must be one of 'sync_single', 'sync_pool', 'async_single', or 'async_pool'."
            )

        if not postgres_connection:
            postgres_connection = DatabaseConnector().uri

        if postgres_type == "sync_single":
            saver = PostgresSaver(
                postgres_connection,
                checkpoint_table=checkpoint_table,
                intermediate_table=intermediate_table,
                migration_table=migration_table,
                blob_table=blob_table,
            )
            return saver

        elif postgres_type == "async_single":
            saver = AsyncPostgresSaver(
                postgres_connection,
                checkpoint_table=checkpoint_table,
                intermediate_table=intermediate_table,
                migration_table=migration_table,
                blob_table=blob_table,
            )
            return saver

    else:
        raise ValueError(
            f"Unsupported memory type '{type}'. Supported types: 'memory', 'sqlite', 'postgres'."
        )
