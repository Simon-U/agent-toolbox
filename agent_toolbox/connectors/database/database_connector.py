import sqlite3
from sqlalchemy import text
from sqlalchemy import create_engine
from os import environ

__all__ = ["DatabaseConnector"]


class DatabaseConnector:
    SUPPORTED_DB_TYPES = ["sqlite", "postgres"]

    def __init__(
        self,
        db_type=environ.get("DB"),
        uri="/home/simon/Documents/Pure_Inference/Malvius/database/sqlite.db",
    ):
        """
        Initialize DatabaseConnector.

        Args:
            db_type (str): Type of database (default from env)
            uri (str): Database URI

        Raises:
            ConfigurationError: If database configuration is invalid
        """
        try:
            self.db_type = db_type
            self.uri = uri

            if db_type == "postgres":
                required_env_vars = [
                    "DB_USER",
                    "DB_PASS",
                    "DB_PORT",
                    "DB_HOST",
                    "DB_NAME",
                ]
                missing_vars = [
                    var for var in required_env_vars if not environ.get(var)
                ]


                self.uri = (
                    "postgresql://{username}:{password}@{host}:{port}/{dbname}".format(
                        username=environ.get("DB_USER"),
                        password=environ.get("DB_PASS"),
                        port=environ.get("DB_PORT"),
                        host=environ.get("DB_HOST"),
                        dbname=environ.get("DB_NAME"),
                    )
                )

        except Exception as e:
            error_msg = f"Failed to initialize database connector: {str(e)}"

    def connect(self):
        """
        Establish database connection.

        Returns:
            SQLAlchemy Engine

        Raises:
            DatabaseConnectionError: If connection fails
        """
        try:

            engine = create_engine(self.uri)
            # Test the connection
            with engine.connect():

                return engine
        except Exception as e:
            error_msg = f"Failed to connect to database: {str(e)}"


    def execute_query(self, query, params=None):
        """
        Execute a database query.

        Args:
            query (str): SQL query
            params (dict, optional): Query parameters

        Returns:
            Query results for SELECT queries

        Raises:
            QueryError: If query execution fails
        """
        try:

            with self.connect().connect() as connection:
                if params:

                    result = connection.execute(text(query), params)
                else:
                    result = connection.execute(text(query))

                # Fetch results if SELECT query
                if query.strip().lower().startswith("select"):
                    result = result.fetchall()

                else:
                    pass
                return result

        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"


    def execute_write(self, query, params=None):
        """
        Execute a write operation.

        Args:
            query (str): SQL query
            params (dict, optional): Query parameters

        Returns:
            bool: Success status

        Raises:
            QueryError: If write operation fails
        """
        try:

            with self.connect().begin() as connection:
                if params:

                    connection.execute(text(query), params)
                else:
                    connection.execute(text(query))

                return True

        except Exception as e:
            error_msg = f"Write operation failed: {str(e)}"
