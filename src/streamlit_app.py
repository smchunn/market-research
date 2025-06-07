# Inside streamlit_app.py
import streamlit as st
import polars as pl
import time
import asyncio
from utils.db import load_table, save_df
from utils.utils import (
    get_zip_codes,
    get_census_data,
    filter_zip_codes,
    find_potential_sites,
)  # <-- your existing code
from pathlib import Path

DATA_DB = Path("./data/market_research.db")

st.sidebar.header("🔧 Parameters")

# --- API KEYS ---
GOOGLE_API_KEY = st.sidebar.text_input("Google Maps API Key", type="password")
CENSUS_API_KEY = st.sidebar.text_input("Census API Key", type="password")

# --- STAGE 1 PARAMS ---
region_state = st.sidebar.text_input("Region State Abbrev", value="DC")
start_zip = st.sidebar.text_input("Start Zip Code", value="20001")

with st.sidebar:
    radius = st.slider("Radius (miles)", 1, 20, 5)
    pop_density_threshold = st.number_input("Min Population Density", 100, 20000, 1000)
    income_threshold = st.number_input("Min Median Income", 10000, 250000, 75000)
    children_threshold = st.slider("Min Proportion Children", 0.0, 1.0, 0.15)
    min_rating = st.slider("Min Site Rating", 1.0, 5.0, 3.5)
    max_day_camps = st.number_input("Max Competitor Camps per Zip", 0, 10, 3)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📍 Stage 1: Zip Codes",
        "🏫 Stage 2: Potential Sites",
        "🏊 Stage 3: Facility Attributes",
        "🧭 Stage 4: Competitive Map",
        "✅ Stage 5: Top Sites",
    ]
)
with tab1:
    st.header("📍 Stage 1: Zip Code Filter")

    if st.button("🔁 Refresh Zip Code Data"):
        with st.spinner("Fetching and filtering zip code data..."):
            zip_codes = get_zip_codes(region_state, start_zip, radius)
            census_df = get_census_data(zip_codes, CENSUS_API_KEY)  # Pass API key
            if census_df.is_empty():
                st.error("No data found. Check ZIP code, radius, or API key.")
            else:
                filtered_df = filter_zip_codes(
                    census_df,
                    pop_density_threshold,
                    income_threshold,
                    children_threshold,
                )
                asyncio.run(save_df(census_df, "census_data"))
                asyncio.run(save_df(filtered_df, "priority_zips"))
                st.success(
                    f"Saved {filtered_df.height} priority zip codes to database."
                )

    if DATA_DB.exists():
        st.subheader("📄 Priority Zip Codes from Database")
        try:
            df = asyncio.run(load_table("priority_zips"))
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error loading from database: {e}")
    else:
        st.warning("No data yet. Click refresh to generate.")
with tab2:
    st.header("🏫 Stage 2: Discover Potential Day Camp Sites")

    if st.button("🔍 Search for Sites in Priority Zips"):
        try:
            with st.spinner("Loading priority zip codes and querying Google Places..."):
                priority_zips = asyncio.run(load_table("priority_zips"))
                if priority_zips.is_empty():
                    st.error(
                        "No priority zip codes found in database. Run Stage 1 first."
                    )
                else:
                    site_df = find_potential_sites(priority_zips, GOOGLE_API_KEY)
                    if site_df.is_empty():
                        st.warning("No potential sites found.")
                    else:
                        # Add checkbox column defaulted to True
                        site_df = site_df.with_columns(pl.lit(True).alias("Include"))
                        asyncio.run(save_df(site_df, "potential_sites"))
                        asyncio.run(
                            save_df(
                                site_df.filter(pl.col("Include") == True),
                                "potential_sites_filtered",
                            )
                        )
                        st.success(f"{site_df.height} sites saved to database.")
        except Exception as e:
            st.error(f"Error during site search: {e}")

    # Display saved sites with checkboxes
    try:
        sites_df = asyncio.run(load_table("potential_sites"))

        if not sites_df.is_empty():
            st.subheader("📋 Approve Sites for Stage 3 Input")

            from utils.ui_helpers import safe_polars_editor

            editable_df = sites_df.select(
                ["zip_code", "name", "rating", "place_type", "Include"]
            )

            edited_df = safe_polars_editor(editable_df, bool_cols=["Include"], key="site_editor")

            if st.button("💾 Save Filtered Sites for Stage 3"):
                st.write("🔍 Edited Data Preview:")
                st.dataframe(edited_df)

                if "Include" not in edited_df.columns:
                    st.error("❌ 'Include' column missing from edited data.")
                else:
                    # Fix pandas→Polars dtype mismatch
                    filtered = pl.DataFrame(edited_df, strict=False)

                    # Normalize Include values: "True", 1, etc → True
                    filtered = filtered.with_columns(
                        pl.when(pl.col("Include").is_in([True, "True", 1, "1"]))
                        .then(True)
                        .otherwise(False)
                        .alias("Include")
                    )

                    included = filtered.filter(pl.col("Include") == True)
                    st.write(f"✅ {included.height} sites marked for inclusion.")
                    st.dataframe(included)

                    if included.height == 0:
                        st.warning(
                            "⚠️ No rows marked as 'Include = True'. Nothing will show in Stage 3."
                        )
                    else:
                        asyncio.run(save_df(filtered, "potential_sites"))
                        asyncio.run(save_df(included, "potential_sites_filtered"))
                        st.success("✅ Saved updated selections.")
    except Exception as e:
        st.warning(f"No saved site data available yet: {e}")
with tab3:
    st.header("🏊 Stage 3: Facility Attributes Lookup via Gemini")

    GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")

    # Step 1: Choose amenities
    amenities = st.multiselect(
        "Select Amenities to Check",
        ["Pool", "Field", "Gym", "Auditorium", "Kitchen", "Playground"],
        default=["Pool", "Field", "Gym"],
    )

    # Step 2: Load filtered sites
    try:
        filtered_sites_df = asyncio.run(load_table("potential_sites_filtered"))

        if filtered_sites_df.is_empty():
            st.warning("No filtered sites found. Approve some in Stage 2.")
        else:
            st.dataframe(
                filtered_sites_df.select(["name", "address", "zip_code", "rating"])
            )

            if st.button("🔍 Lookup Amenities with Gemini"):
                from utils.utils import get_site_amenities_with_gemini

                enriched_sites = []
                with st.spinner("Querying Gemini for each site..."):
                    for row in filtered_sites_df.iter_rows(named=True):
                        result = get_site_amenities_with_gemini(
                            site_name=row["name"],
                            address=row.get("address", f"{row['zip_code']}"),
                            amenities_list=amenities,
                            gemini_api_key=GEMINI_API_KEY,
                        )
                        site_entry = row.copy()
                        for amenity in amenities:
                            col = amenity.lower()
                            site_entry[col] = result.get(col, False)
                        enriched_sites.append(site_entry)
                        time.sleep(1.2)  # prevent rate limiting

                # Save to DB
                enriched_df = pl.DataFrame(enriched_sites)
                asyncio.run(save_df(enriched_df, "site_attributes"))
                st.success("Amenities saved to database.")

    except Exception as e:
        st.error(f"Error loading filtered sites or querying Gemini: {e}")

    # Optional: Show current attributes table
    try:
        attrs_df = asyncio.run(load_table("site_attributes"))
        if not attrs_df.is_empty():
            st.subheader("✅ Saved Site Attributes")
            st.dataframe(attrs_df)
    except Exception:
        pass
