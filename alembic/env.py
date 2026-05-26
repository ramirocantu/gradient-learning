import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Resolve DB URL with this precedence: ALEMBIC_DATABASE_URL env var
# (explicit override, used by tests/CI), then settings.DATABASE_URL
# (which pydantic-settings reads from .env so worktrees on non-default
# ports work without an inline override), then the empty sqlalchemy.url
# in alembic.ini (which raises a clear error if neither is set).
_url_override = os.environ.get("ALEMBIC_DATABASE_URL")
if not _url_override:
    from app.config import settings  # noqa: E402 — pydantic-settings reads .env

    _url_override = settings.DATABASE_URL
if _url_override:
    config.set_main_option("sqlalchemy.url", _url_override)

from app.database import Base  # noqa: E402 — must come after fileConfig
import app.models  # noqa: E402, F401 — registers all models on Base.metadata

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
