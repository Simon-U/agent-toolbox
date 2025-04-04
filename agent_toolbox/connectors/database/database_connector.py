import sqlite3
from sqlalchemy import text
from sqlalchemy import create_engine
from os import environ

__all__ = ["DatabaseConnector"]


class DatabaseConnector:
    SUPPORTED_DB_TYPES = ["sqlite", "postgres"]
    def __init__(
        self,
        db_type=None,
        uri="/home/simon/Documents/Pure_Inference/Malvius/database/sqlite.db",
        settings=None,
    ):
        """
        Initialize DatabaseConnector.
        Args:
            db_type (str): Type of database (default from env or settings)
            uri (str): Database URI
            settings (object, optional): Settings object with configuration attributes.
                                         If None, environment variables are used.
        Raises:
            ConfigurationError: If database configuration is invalid
        """
        try:
            # Use environ if settings is None
            env = settings if settings is not None else environ
            
            # Get db_type from parameter, or from settings/environ if not specified
            if db_type is not None:
                self.db_type = db_type
            else:
                if hasattr(env, 'get'):  # If env has get method (like environ)
                    self.db_type = env.get("DB")
                else:  # If env is a settings object with attributes
                    self.db_type = getattr(env, "DB", None)
            self.uri = uri
            
            if self.db_type == "postgres":
                required_vars = [
                    "DB_USER",
                    "DB_PASS",
                    "DB_PORT",
                    "DB_HOST",
                    "DB_NAME",
                ]
                
                # Check for missing variables
                missing_vars = []
                for var in required_vars:
                    if hasattr(env, 'get'):  # If env has get method (like environ)
                        if not env.get(var):
                            missing_vars.append(var)
                    else:  # If env is a settings object with attributes
                        if not hasattr(env, var) or getattr(env, var) is None:
                            missing_vars.append(var)
                
                # Format the PostgreSQL URI
                if hasattr(env, 'get'):  # If env has get method (like environ)
                    self.uri = (
                        "postgresql://{username}:{password}@{host}:{port}/{dbname}".format(
                            username=env.get("DB_USER"),
                            password=env.get("DB_PASS"),
                            port=env.get("DB_PORT"),
                            host=env.get("DB_HOST"),
                            dbname=env.get("DB_NAME"),
                        )
                    )
                else:  # If env is a settings object with attributes
                    self.uri = (
                        "postgresql://{username}:{password}@{host}:{port}/{dbname}".format(
                            username=env.DB_USER,
                            password=env.DB_PASS,
                            port=env.DB_PORT,
                            host=env.DB_HOST,
                            dbname=env.DB_NAME,
                        )
                    )
                
                # Handle missing variables if needed
                if missing_vars:
                    # You might want to raise an exception here or handle it another way
                    pass
                    
        except Exception as e:
            error_msg = f"Failed to initialize database connector: {str(e)}"
            # You might want to raise a ConfigurationError here
            raise Exception(error_msg)  # Replace with your custom exception

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
