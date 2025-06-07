import streamlit as st
import polars as pl
import pandas as pd


def safe_polars_editor(
    df: pl.DataFrame, bool_cols: list[str] = [], key: str = "editor", **editor_kwargs
) -> pl.DataFrame:
    """
    Renders a Polars DataFrame in st.data_editor and returns a clean Polars DataFrame.
    Ensures correct handling of boolean values and avoids dtype casting errors.

    Args:
        df (pl.DataFrame): The Polars DataFrame to edit.
        bool_cols (list[str]): Columns to normalize to Boolean.
        key (str): Unique key for Streamlit widget.
        editor_kwargs: Extra arguments passed to st.data_editor.

    Returns:
        pl.DataFrame: Cleaned, type-safe Polars DataFrame.
    """

    # Step 1: Convert to pandas for Streamlit editing
    pd_df = df.to_pandas()

    # Step 2: Configure checkbox columns
    column_config = {
        col: st.column_config.CheckboxColumn(col, default=True)
        for col in bool_cols
        if col in pd_df.columns
    }

    # Step 3: Render Streamlit editor
    edited_pd = st.data_editor(
        pd_df,
        use_container_width=True,
        column_config=column_config,
        num_rows="dynamic",
        key=key,
        **editor_kwargs
    )

    # Step 4: Convert back to Polars (strict=False prevents type inference failure)
    # 1. Normalize booleans in Pandas before Polars conversion
    for col in bool_cols:
        if col in edited_pd.columns:
            edited_pd[col] = edited_pd[col].apply(
                lambda x: str(x).lower() in ["true", "1", "yes"]
            )

    # 2. Now convert safely
    edited_pl = pl.DataFrame(edited_pd, strict=False)

    # Step 5: Normalize boolean columns
    for col in bool_cols:
        if col in edited_pl.columns:
            edited_pl = edited_pl.with_columns(
                pl.when(pl.col(col).is_in([True, "True", 1, "1"]))
                .then(True)
                .otherwise(False)
                .alias(col)
            )

    return edited_pl
