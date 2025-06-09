import streamlit as st
import polars as pl
import pandas as pd
from typing import List, Dict, Any, Optional


def safe_polars_editor(
    df: pl.DataFrame,
    bool_cols: List[str] = None,
    select_cols: Dict[str, List[str]] = None,
    key: str = "editor",
    **editor_kwargs
) -> pl.DataFrame:
    """
    A robust Streamlit data editor for Polars DataFrames with proper type handling.
    
    Args:
        df (pl.DataFrame): Input Polars DataFrame
        bool_cols (List[str]): List of column names to be treated as boolean
        select_cols (Dict[str, List[str]]): Dict mapping column names to their allowed values
        key (str): Unique key for the Streamlit widget
        **editor_kwargs: Additional arguments for st.data_editor
        
    Returns:
        pl.DataFrame: Edited DataFrame with proper types
    """
    if bool_cols is None:
        bool_cols = []
    if select_cols is None:
        select_cols = {}
        
    # Convert to pandas for editing
    pd_df = df.to_pandas()
    
    # Configure column types
    column_config = {}
    
    # Configure boolean columns
    for col in bool_cols:
        if col in pd_df.columns:
            column_config[col] = st.column_config.CheckboxColumn(
                col,
                default=False,
                required=True
            )
            # Ensure boolean type
            pd_df[col] = pd_df[col].fillna(False).astype(bool)
    
    # Configure select columns
    for col, options in select_cols.items():
        if col in pd_df.columns:
            column_config[col] = st.column_config.SelectboxColumn(
                col,
                options=options,
                required=True,
                default=options[0]
            )
            # Ensure valid values
            pd_df[col] = pd_df[col].fillna(options[0])
            invalid_mask = ~pd_df[col].isin(options)
            if invalid_mask.any():
                pd_df.loc[invalid_mask, col] = options[0]
    
    # Configure text columns
    text_cols = [col for col in pd_df.columns if col not in bool_cols and col not in select_cols]
    for col in text_cols:
        column_config[col] = st.column_config.TextColumn(
            col,
            required=True
        )
        # Fill NaN with empty string for text columns
        pd_df[col] = pd_df[col].fillna("")
    
    try:
        # Render the editor
        edited_df = st.data_editor(
            pd_df,
            use_container_width=True,
            column_config=column_config,
            num_rows="dynamic",
            key=key,
            **editor_kwargs
        )
        
        # Convert back to Polars with proper type handling
        result_df = pl.DataFrame(edited_df)
        
        # Ensure boolean columns are properly typed
        for col in bool_cols:
            if col in result_df.columns:
                result_df = result_df.with_columns(
                    pl.col(col).cast(pl.Boolean)
                )
        
        # Ensure select columns contain valid values
        for col, options in select_cols.items():
            if col in result_df.columns:
                result_df = result_df.with_columns(
                    pl.when(~pl.col(col).is_in(options))
                    .then(options[0])
                    .otherwise(pl.col(col))
                    .alias(col)
                )
        
        return result_df
        
    except Exception as e:
        st.error(f"Error in data editor: {str(e)}")
        return df  # Return original DataFrame on error

def validate_dataframe(df: pl.DataFrame, required_cols: List[str] = None) -> bool:
    """
    Validate a DataFrame has required columns and non-null values.
    
    Args:
        df (pl.DataFrame): DataFrame to validate
        required_cols (List[str]): List of required column names
        
    Returns:
        bool: True if valid, False otherwise
    """
    if required_cols is None:
        required_cols = []
        
    try:
        # Check required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return False
            
        # Check for null values in required columns
        null_counts = df.select(pl.col(required_cols).null_count())
        for col in required_cols:
            if null_counts[col][0] > 0:
                st.warning(f"Column {col} contains {null_counts[col][0]} null values")
                return False
                
        return True
        
    except Exception as e:
        st.error(f"Error validating DataFrame: {str(e)}")
        return False
