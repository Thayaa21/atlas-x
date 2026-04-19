from pathlib import Path
import asyncpg


def sql_path(repo_root: Path) -> Path:
    return repo_root / "src/monitoring/sql/schema.sql"


async def init_postgres(dsn: str, *, repo_root: Path) -> None:
    """
    Initialize monitoring tables used by streaming consumer.
    """
    schema_sql = sql_path(repo_root).read_text()
    conn = await asyncpg.connect(dsn)
    try:
        # Execute whole schema in a single connection.
        await conn.execute(schema_sql)
    finally:
        await conn.close()

