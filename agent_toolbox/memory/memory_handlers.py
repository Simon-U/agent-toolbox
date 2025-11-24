"""
Memory/checkpoint handlers for different storage backends.

This module implements a handler pattern for creating checkpoint savers from
different storage backends. Each handler encapsulates backend-specific logic.
"""

import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Union

from langgraph.checkpoint.memory import MemorySaver
from psycopg import Connection, AsyncConnection
from psycopg_pool import AsyncConnectionPool

from .sqlite.sqlite_saver import CustomSqliteSaver
from .sqlite.async_sqlite_saver import CustomAsyncSqliteSaver
from .postgres.sync_saver import PostgresSaver
from .postgres.async_saver import AsyncPostgresSaver
from .memory_config import (
    MemoryType,
    PostgresType,
    MemoryConfig,
    SqliteConfig,
    PostgresConfig,
)
from ..connectors.database.database_connector import DatabaseConnector


class MemoryHandler(ABC):
    """
    Abstract base class for memory/checkpoint storage handlers.

    Each concrete handler implements backend-specific logic for instantiating
    and configuring checkpoint savers.
    """

    @abstractmethod
    def get_saver(self, config: MemoryConfig) -> Any:
        """
        Get a configured checkpoint saver instance.

        Args:
            config: Memory configuration specific to the backend.

        Returns:
            Configured saver instance (MemorySaver, SqliteSaver, PostgresSaver, etc.).
        """
        pass


class InMemoryHandler(MemoryHandler):
    """
    Handler for in-memory checkpoint storage.

    Stores checkpoints and state in memory only. Best for single-session testing
    and development. State is lost when the process ends.
    """

    def get_saver(self, config: MemoryConfig) -> MemorySaver:
        """
        Get an in-memory checkpoint saver.

        Args:
            config: Memory configuration (unused for in-memory).

        Returns:
            MemorySaver instance for in-memory storage.
        """
        return MemorySaver()


class SqliteHandler(MemoryHandler):
    """
    Handler for synchronous SQLite checkpoint storage.

    Stores checkpoints persistently in a local SQLite database with
    synchronous (blocking) access.
    """

    def get_saver(self, config: SqliteConfig) -> CustomSqliteSaver:
        """
        Get a synchronous SQLite checkpoint saver.

        Args:
            config: SQLite configuration with path and table names.

        Returns:
            CustomSqliteSaver instance for synchronous SQLite operations.

        Raises:
            ValueError: If sqlite_path is not provided in config.
        """
        if not config.sqlite_path:
            raise ValueError("Please specify a sqlite_path for the database.")

        # Use provided connection or create new one
        if config.connection:
            conn = config.connection
        else:
            conn = sqlite3.connect(config.sqlite_path, check_same_thread=False)

        return CustomSqliteSaver(
            conn,
            checkpoint_table=config.checkpoint_table,
            intermediate_table=config.intermediate_table,
        )


class AsyncSqliteHandler(MemoryHandler):
    """
    Handler for asynchronous SQLite checkpoint storage.

    Stores checkpoints persistently in a local SQLite database with
    asynchronous (non-blocking) access.
    """

    def get_saver(self, config: SqliteConfig) -> CustomAsyncSqliteSaver:
        """
        Get an asynchronous SQLite checkpoint saver.

        Args:
            config: SQLite configuration with path and table names.

        Returns:
            CustomAsyncSqliteSaver instance for asynchronous SQLite operations.

        Raises:
            ValueError: If sqlite_path is not provided in config.
        """
        if not config.sqlite_path:
            raise ValueError("Please specify a sqlite_path for the database.")

        # Use provided connection or create new one
        if config.connection:
            conn = config.connection
        else:
            conn = sqlite3.connect(config.sqlite_path, check_same_thread=False)

        return CustomAsyncSqliteSaver(
            conn,
            checkpoint_table=config.checkpoint_table,
            intermediate_table=config.intermediate_table,
        )


class PostgresHandler(MemoryHandler):
    """
    Handler for PostgreSQL checkpoint storage.

    Stores checkpoints persistently in a PostgreSQL database with support
    for both synchronous and asynchronous connections, with optional pooling.
    """

    def get_saver(
        self, config: PostgresConfig
    ) -> Union[PostgresSaver, AsyncPostgresSaver]:
        """
        Get a PostgreSQL checkpoint saver (sync or async based on config).

        Args:
            config: PostgreSQL configuration with connection string and type.

        Returns:
            PostgresSaver for sync_single/sync_pool or AsyncPostgresSaver
            for async_single/async_pool.

        Raises:
            ValueError: If postgres_type is not supported.
            ValueError: If postgres_connection is not provided.
        """
        # Validate postgres_type
        if config.postgres_type not in [
            PostgresType.SYNC_SINGLE,
            PostgresType.SYNC_POOL,
            PostgresType.ASYNC_SINGLE,
            PostgresType.ASYNC_POOL,
        ]:
            raise ValueError(
                f"postgres_type must be one of {[t.value for t in PostgresType]}. "
                f"Got: {config.postgres_type}"
            )

        # Get connection string from config or DatabaseConnector
        if not config.postgres_connection:
            config.postgres_connection = DatabaseConnector().uri

        # Synchronous handlers
        if config.postgres_type in [PostgresType.SYNC_SINGLE, PostgresType.SYNC_POOL]:
            return PostgresSaver(
                config.postgres_connection,
                checkpoint_table=config.checkpoint_table,
                intermediate_table=config.intermediate_table,
                migration_table=config.migration_table,
                blob_table=config.blob_table,
            )

        # Asynchronous handlers
        if config.postgres_type in [
            PostgresType.ASYNC_SINGLE,
            PostgresType.ASYNC_POOL,
        ]:
            return AsyncPostgresSaver(
                config.postgres_connection,
                checkpoint_table=config.checkpoint_table,
                intermediate_table=config.intermediate_table,
                migration_table=config.migration_table,
                blob_table=config.blob_table,
            )


class MemoryFactory:
    """
    Factory for creating memory/checkpoint handlers.

    This factory manages the creation and caching of memory handlers,
    enabling easy switching between storage backends and adding new ones.

    Example:
        >>> handler = MemoryFactory.get_handler(MemoryType.SQLITE)
        >>> config = SqliteConfig(sqlite_path="/path/to/db.sqlite")
        >>> saver = handler.get_saver(config)
    """

    _handlers = {
        MemoryType.MEMORY: InMemoryHandler(),
        MemoryType.SQLITE: SqliteHandler(),
        MemoryType.ASYNCSQLITE: AsyncSqliteHandler(),
        MemoryType.POSTGRES: PostgresHandler(),
    }

    @classmethod
    def get_handler(cls, memory_type: MemoryType) -> MemoryHandler:
        """
        Get the handler for a specific memory type.

        Args:
            memory_type: The memory type (from MemoryType enum).

        Returns:
            The corresponding MemoryHandler instance.

        Raises:
            ValueError: If the memory type is not supported.
        """
        handler = cls._handlers.get(memory_type)
        if not handler:
            supported = ", ".join([t.value for t in MemoryType])
            raise ValueError(
                f"Unsupported memory type: {memory_type.value}. "
                f"Supported types are: {supported}"
            )
        return handler

    @classmethod
    def register_handler(cls, memory_type: MemoryType, handler: MemoryHandler) -> None:
        """
        Register a custom memory handler.

        This allows extending the system with new storage backends without
        modifying the existing code.

        Args:
            memory_type: The memory type identifier.
            handler: The handler instance to register.

        Example:
            >>> class DynamoDBHandler(MemoryHandler):
            ...     def get_saver(self, config):
            ...         # Custom implementation
            ...         pass
            >>> MemoryFactory.register_handler(
            ...     MemoryType.DYNAMODB,
            ...     DynamoDBHandler()
            ... )
        """
        cls._handlers[memory_type] = handler
