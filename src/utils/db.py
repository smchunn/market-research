# utils/db.py
import aiosqlite
import polars as pl

DB_PATH = "./data/market_research.db"


async def load_table(table_name: str) -> pl.DataFrame:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(f"SELECT * FROM {table_name}")
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pl.DataFrame(rows, schema=columns)


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
