# utils/db.py
import aiosqlite
import polars as pl
from pathlib import Path
import asyncio

DB_PATH = "./data/market_research.db"

async def init_db():
    """Initialize the database and create necessary tables if they don't exist."""
    db_path = Path(DB_PATH)
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True)

    tables = [
        "census_data",
        "priority_zips",
        "potential_sites",
        "potential_sites_filtered",
        "site_attributes",
        "site_attributes_final",
        "competitor_camps"
    ]

    async def create_table(table):
        try:
            db = await aiosqlite.connect(DB_PATH)
            await db.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY AUTOINCREMENT)")
            await db.commit()
            await db.close()
        except Exception as e:
            print(f"Error creating table {table}: {e}")

    for table in tables:
        await create_table(table)

async def load_table(table_name: str) -> pl.DataFrame:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(f"SELECT * FROM {table_name}")
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pl.DataFrame(rows, schema=columns, orient="row")

async def save_df(df: pl.DataFrame, table_name: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"DROP TABLE IF EXISTS {table_name}")
        await db.commit()
        await db.execute(
            f"CREATE TABLE {table_name} ({', '.join(f'{col} TEXT' for col in df.columns)})"
        )
        await db.executemany(
            f"INSERT INTO {table_name} VALUES ({', '.join(['?' for _ in df.columns])})",
            [tuple(row) for row in df.iter_rows()],
        )
        await db.commit()
