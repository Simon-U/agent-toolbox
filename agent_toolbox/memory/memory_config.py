"""
Memory configuration and models for checkpoint storage backends.

This module defines the supported memory/checkpoint storage backends and their
configurations, allowing for flexible, extensible memory management.
"""

import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MemoryType(str, Enum):
    """
    Supported memory/checkpoint storage backends.

    Each backend stores agent state and conversation history:
    - MEMORY: In-memory storage (default, good for single-session testing)
    - SQLITE: Local SQLite database (single machine, persistent)
    - ASYNCSQLITE: Async SQLite database (single machine, non-blocking)
    - POSTGRES: PostgreSQL database (distributed, production-ready)
    """
    MEMORY = 'memory'
    SQLITE = 'sqlite'
    ASYNCSQLITE = 'asyncsqlite'
    POSTGRES = 'postgres'


class PostgresType(str, Enum):
    """
    PostgreSQL connection types for checkpoint storage.

    Controls how the PostgreSQL connection is managed:
    - SYNC_SINGLE: Single synchronous connection (simple, blocking)
    - SYNC_POOL: Connection pool for sync connections (concurrent, blocking)
    - ASYNC_SINGLE: Single asynchronous connection (simple, non-blocking)
    - ASYNC_POOL: Connection pool for async connections (concurrent, non-blocking)
    """
    SYNC_SINGLE = 'sync_single'
    SYNC_POOL = 'sync_pool'
    ASYNC_SINGLE = 'async_single'
    ASYNC_POOL = 'async_pool'


@dataclass
class MemoryConfig:
    """
    Base configuration for memory/checkpoint storage.

    Attributes:
        checkpoint_table: Name of the table storing checkpoints. Defaults to "checkpoints".
        intermediate_table: Name of the table storing intermediate steps. Defaults to "writes".
    """
    checkpoint_table: str = "checkpoints"
    intermediate_table: str = "writes"


@dataclass
class SqliteConfig(MemoryConfig):
    """
    SQLite-specific configuration for local database storage.

    Attributes:
        sqlite_path: File path to the SQLite database. Required for operation.
        connection: Optional pre-configured SQLite connection. If None, will create one.
    """
    sqlite_path: Optional[str] = None
    connection: Optional[sqlite3.Connection] = None


@dataclass
class PostgresConfig(MemoryConfig):
    """
    PostgreSQL-specific configuration for distributed database storage.

    Attributes:
        postgres_type: Type of PostgreSQL connection to use.
        postgres_connection: Connection string to PostgreSQL. Required if not using DatabaseConnector.
        max_pool: Maximum pool size for pooled connections. Defaults to 8.
        migration_table: Name of the migrations tracking table. Defaults to "checkpoint_migrations".
        blob_table: Name of the blob/state value storage table. Defaults to "blobs".
    """
    postgres_type: PostgresType = PostgresType.SYNC_SINGLE
    postgres_connection: Optional[str] = None
    max_pool: int = 8
    migration_table: str = "checkpoint_migrations"
    blob_table: str = "blobs"


# Default configurations for each memory type
DEFAULT_CONFIGS = {
    MemoryType.MEMORY: MemoryConfig(),
    MemoryType.SQLITE: SqliteConfig(),
    MemoryType.ASYNCSQLITE: SqliteConfig(),
    MemoryType.POSTGRES: PostgresConfig(),
}
