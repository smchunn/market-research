# Inside streamlit_app.py
import streamlit as st
import polars as pl
import time
import asyncio
from utils.db import load_table, save_df, init_db
from utils.utils import (
    get_zip_codes,
    get_census_data,
    filter_zip_codes,
    find_potential_sites,
    validate_api_key,
    find_competitor_camps,
    ZIPCODE_DATA_URL,
    setup_logging,
    get_site_amenities_with_gemini,
)  # <-- your existing code
from pathlib import Path
import os
import requests
import pandas as pd
from dotenv import load_dotenv
import sys

# Check for debug flag in environment variable
debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
setup_logging(debug=debug_mode)

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'census': os.getenv('CENSUS_API_KEY', ''),
        'google': os.getenv('GOOGLE_API_KEY', ''),
        'gemini': os.getenv('GEMINI_API_KEY', '')
    }

# Initialize database
DATA_DB = Path("./data/market_research.db")
if not DATA_DB.parent.exists():
    DATA_DB.parent.mkdir(parents=True)
import asyncio
asyncio.run(init_db())

st.set_page_config(layout="wide")

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

st.sidebar.header("🔑 API Keys")

# API Key inputs in sidebar
with st.sidebar:
    CENSUS_API_KEY = st.text_input("Census API Key", type="password", value=st.session_state.api_keys['census'])
    if CENSUS_API_KEY != st.session_state.api_keys['census']:
        st.session_state.api_keys['census'] = CENSUS_API_KEY
    
    GOOGLE_API_KEY = st.text_input("Google Maps API Key", type="password", value=st.session_state.api_keys['google'])
    if GOOGLE_API_KEY != st.session_state.api_keys['google']:
        st.session_state.api_keys['google'] = GOOGLE_API_KEY
    
    GEMINI_API_KEY = st.text_input("Gemini API Key", type="password", value=st.session_state.api_keys['gemini'])
    if GEMINI_API_KEY != st.session_state.api_keys['gemini']:
        st.session_state.api_keys['gemini'] = GEMINI_API_KEY

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
    
    # Create two columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        start_zip = st.text_input("Start Zip Code", value="20001")
        # Infer state from zip code
        try:
            zip_df = pl.read_csv(ZIPCODE_DATA_URL)
            zip_info = zip_df.filter(pl.col("code").cast(str).str.zfill(5) == start_zip)
            if not zip_info.is_empty():
                region_state = zip_info["state"][0]
                st.info(f"Detected State: {region_state}")
            else:
                st.error("Invalid zip code. Please enter a valid US zip code.")
                region_state = None
        except Exception as e:
            st.error(f"Error looking up zip code: {str(e)}")
            region_state = None
            
        radius = st.number_input("Radius (miles)", min_value=1, max_value=20, value=5, step=1)
    
    with col2:
        pop_density_threshold = st.number_input("Min Population Density", 100, 20000, 1000)
        income_threshold = st.number_input("Min Median Income", 10000, 250000, 75000)
        children_threshold = st.slider("Min Proportion Children", 0.0, 1.0, 0.15)
    
    if st.button("🔁 Refresh Zip Code Data"):
        if not region_state:
            st.error("Please enter a valid zip code first.")
        elif not validate_api_key('census', st.session_state.api_keys['census']):
            st.error("Invalid Census API Key. Please check and try again.")
        else:
            with st.spinner("Fetching and filtering zip code data..."):
                try:
                    zip_codes = get_zip_codes(region_state, start_zip, radius)
                    if not zip_codes:
                        st.error("No zip codes found in the specified radius.")
                    else:
                        census_df = get_census_data(zip_codes, st.session_state.api_keys['census'])
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
                            st.success(f"Saved {filtered_df.height} priority zip codes to database.")
                except Exception as e:
                    st.error(f"Error processing zip codes: {str(e)}")

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
    
    # Site Search Parameters
    st.subheader("🔍 Search Parameters")
    col1, col2 = st.columns(2)
    with col1:
        search_radius_miles = st.number_input("Search Radius (miles)", min_value=1, max_value=10, value=2, step=1)
        search_radius = int(search_radius_miles * 1609.34)  # Convert miles to meters
        min_rating = st.number_input("Minimum Rating", min_value=0.0, max_value=5.0, value=3.5, step=0.1)
    with col2:
        place_types = st.multiselect(
            "Place Types",
            ["school", "park", "community_center", "recreation_center"],
            default=["school", "park"]
        )
        max_results_per_zip = st.number_input("Max Results per Zip Code", min_value=1, max_value=10, value=3)

    if st.button("🔍 Search for Potential Sites"):
        if not validate_api_key('google', st.session_state.api_keys['google']):
            st.error("Invalid Google Maps API Key. Please check and try again.")
        else:
            try:
                # Load priority zip codes
                with st.spinner("Loading priority zip codes..."):
                    priority_zips = asyncio.run(load_table("priority_zips"))
                
                if priority_zips.is_empty():
                    st.error("No priority zip codes found. Please run Stage 1 first.")
                else:
                    # Initialize search
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    all_sites = []
                    total_zips = len(priority_zips)
                    
                    # Process each zip code
                    for idx, zip_row in enumerate(priority_zips.iter_rows(named=True)):
                        zip_code = zip_row['zip_code']
                        status_text.text(f"Searching zip code {zip_code} ({idx + 1}/{total_zips})")
                        
                        try:
                            # Get coordinates
                            zip_df = pl.read_csv(ZIPCODE_DATA_URL)
                            zip_info = zip_df.filter(
                                pl.col("code").cast(str).str.zfill(5) == zip_code
                            )
                            
                            if not zip_info.is_empty():
                                lat = zip_info["lat"][0]
                                lng = zip_info["lon"][0]
                                
                                # Search each place type
                                zip_sites = []
                                for place_type in place_types:
                                    try:
                                        # Search parameters
                                        params = {
                                            "location": f"{lat},{lng}",
                                            "radius": search_radius,
                                            "key": st.session_state.api_keys['google']
                                        }
                                        
                                        if place_type == "recreation_center":
                                            params["keyword"] = "recreation center"
                                        else:
                                            params["type"] = place_type
                                        
                                        # API request
                                        response = requests.get(
                                            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                                            params=params,
                                            headers={"Referer": "http://localhost"}
                                        )
                                        response.raise_for_status()
                                        data = response.json()
                                        
                                        if data["status"] == "OK":
                                            # Process results
                                            for place in data.get("results", []):
                                                if place.get("rating", 0) >= min_rating:
                                                    site = {
                                                        "zip_code": zip_code,
                                                        "name": place.get("name", ""),
                                                        "address": place.get("vicinity", ""),
                                                        "place_id": place.get("place_id", ""),
                                                        "rating": place.get("rating", 0.0),
                                                        "latitude": place.get("geometry", {}).get("location", {}).get("lat", 0.0),
                                                        "longitude": place.get("geometry", {}).get("location", {}).get("lng", 0.0),
                                                        "place_type": place_type,
                                                        "Include": True,
                                                        "facility_type": "Other"
                                                    }
                                                    zip_sites.append(site)
                                            
                                            # Handle pagination
                                            while "next_page_token" in data:
                                                time.sleep(2)
                                                params = {
                                                    "pagetoken": data["next_page_token"],
                                                    "key": st.session_state.api_keys['google']
                                                }
                                                response = requests.get(
                                                    "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                                                    params=params,
                                                    headers={"Referer": "http://localhost"}
                                                )
                                                response.raise_for_status()
                                                data = response.json()
                                                
                                                if data["status"] == "OK":
                                                    for place in data.get("results", []):
                                                        if place.get("rating", 0) >= min_rating:
                                                            site = {
                                                                "zip_code": zip_code,
                                                                "name": place.get("name", ""),
                                                                "address": place.get("vicinity", ""),
                                                                "place_id": place.get("place_id", ""),
                                                                "rating": place.get("rating", 0.0),
                                                                "latitude": place.get("geometry", {}).get("location", {}).get("lat", 0.0),
                                                                "longitude": place.get("geometry", {}).get("location", {}).get("lng", 0.0),
                                                                "place_type": place_type,
                                                                "Include": True,
                                                                "facility_type": "Other"
                                                            }
                                                            zip_sites.append(site)
                                        else:
                                            st.warning(f"API error for {place_type} in {zip_code}: {data.get('error_message', data['status'])}")
                                            
                                    except Exception as e:
                                        st.warning(f"Error searching for {place_type} in {zip_code}: {str(e)}")
                                
                                # Sort and limit results
                                zip_sites.sort(key=lambda x: x["rating"], reverse=True)
                                zip_sites = zip_sites[:max_results_per_zip]
                                all_sites.extend(zip_sites)
                            
                        except Exception as e:
                            st.warning(f"Error processing zip code {zip_code}: {str(e)}")
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / total_zips)
                    
                    # Save and display results
                    if all_sites:
                        sites_df = pl.DataFrame(all_sites)
                        
                        # Load zip code data for validation
                        zip_df = pl.read_csv(ZIPCODE_DATA_URL)
                        zip_df = zip_df.with_columns(
                            pl.col("code").cast(str).str.zfill(5).alias("code")
                        )
                        
                        # Function to validate zip code against address
                        def validate_zip_code(row):
                            # Try to find the zip code in our zip code database
                            zip_info = zip_df.filter(pl.col("code") == row["zip_code"])
                            if not zip_info.is_empty():
                                # If we found the zip code, check if the city matches
                                city = zip_info["city"][0].lower()
                                address = row["address"].lower()
                                if city in address:
                                    return row["zip_code"]
                            
                            # If no match found, return the original zip code
                            return row["zip_code"]
                        
                        # Apply validation to each row
                        sites_df = sites_df.with_columns(
                            pl.struct(["zip_code", "address"])
                            .map_elements(validate_zip_code)
                            .alias("zip_code")
                        )
                        
                        # Deduplicate sites based on name and address
                        sites_df = sites_df.unique(subset=["name", "address"], keep="first")
                        
                        # Ensure lat/long columns are preserved
                        required_cols = ["name", "address", "zip_code", "rating", "place_type", "facility_type", "Include", "latitude", "longitude"]
                        for col in required_cols:
                            if col not in sites_df.columns:
                                if col == "Include":
                                    sites_df = sites_df.with_columns(pl.lit(True).alias(col))
                                elif col == "facility_type":
                                    sites_df = sites_df.with_columns(pl.lit("Other").alias(col))
                                else:
                                    sites_df = sites_df.with_columns(pl.lit("").alias(col))
                        
                        asyncio.run(save_df(sites_df, "potential_sites"))
                        asyncio.run(save_df(sites_df.filter(pl.col("Include") == True), "potential_sites_filtered"))
                        st.success(f"Found and saved {len(sites_df)} unique potential sites.")
                        
                        st.subheader("📋 Search Results")
                        st.dataframe(sites_df.select([
                            "name", "address", "zip_code", "rating", "place_type",
                            "facility_type"
                        ]))
                    else:
                        st.warning("No potential sites found matching the criteria.")
                        
            except Exception as e:
                st.error(f"Error during site search: {str(e)}")

    # Display and edit saved sites
    try:
        sites_df = asyncio.run(load_table("potential_sites"))
        if not sites_df.is_empty():
            st.subheader("✏️ Edit Site Details")
            
            # Ensure all required columns exist with default values
            required_cols = {
                "Include": True,
                "facility_type": "Other",
                "latitude": 0.0,
                "longitude": 0.0
            }
            
            for col, default_value in required_cols.items():
                if col not in sites_df.columns:
                    sites_df = sites_df.with_columns(pl.lit(default_value).alias(col))
            
            # Convert to pandas for editing
            pd_df = sites_df.select([
                "name", "address", "zip_code", "rating", "place_type",
                "facility_type", "Include"
            ]).to_pandas()
            
            # Configure column types
            column_config = {
                "name": st.column_config.TextColumn("Name", required=True),
                "address": st.column_config.TextColumn("Address", required=True),
                "zip_code": st.column_config.TextColumn("Zip Code", required=True),
                "rating": st.column_config.NumberColumn(
                    "Rating",
                    required=True,
                    min_value=0.0,
                    max_value=5.0,
                    format="%.1f"
                ),
                "place_type": st.column_config.TextColumn("Place Type", required=True),
                "facility_type": st.column_config.SelectboxColumn(
                    "Facility Type",
                    options=["School", "Park", "Community Center", "Recreation Center", "Other"],
                    required=True,
                    default="Other"
                ),
                "Include": st.column_config.CheckboxColumn(
                    "Include",
                    default=True,
                    required=True
                )
            }
            
            # Pre-process data
            pd_df["Include"] = pd_df["Include"].fillna(True).astype(bool)
            pd_df["facility_type"] = pd_df["facility_type"].fillna("Other")
            text_cols = ["name", "address", "zip_code", "place_type"]
            for col in text_cols:
                pd_df[col] = pd_df[col].fillna("")
            
            try:
                # Render the editor
                edited_df = st.data_editor(
                    pd_df,
                    use_container_width=True,
                    column_config=column_config,
                    num_rows="dynamic",
                    key="site_editor"
                )
                
                if st.button("💾 Save Changes"):
                    try:
                        # Convert back to Polars
                        edited_pl_df = pl.DataFrame(edited_df)
                        
                        # Get lat/long from original data
                        lat_long_df = sites_df.select(["name", "latitude", "longitude"])
                        
                        # Ensure name column is string type in both DataFrames
                        edited_pl_df = edited_pl_df.with_columns(pl.col("name").cast(pl.Utf8))
                        lat_long_df = lat_long_df.with_columns(pl.col("name").cast(pl.Utf8))
                        
                        # Join the DataFrames
                        final_df = edited_pl_df.join(
                            lat_long_df,
                            on="name",
                            how="left"
                        )
                        
                        # Filter for included sites
                        included = final_df.filter(
                            (pl.col("Include") == True)
                        )
                        
                        if included.height == 0:
                            st.warning("No sites marked for inclusion. Please select at least one site to proceed.")
                        else:
                            asyncio.run(save_df(final_df, "potential_sites"))
                            asyncio.run(save_df(included, "potential_sites_filtered"))
                            st.success(f"Saved {included.height} sites for further analysis.")
                            
                            # Show included sites
                            st.subheader("✅ Selected Sites")
                            st.dataframe(included.select([
                                "name", "address", "zip_code", "rating", "place_type",
                                "facility_type"
                            ]))
                    except Exception as e:
                        st.error(f"Error saving data: {str(e)}")
            except Exception as e:
                st.error(f"Error in data editor: {str(e)}")
                st.dataframe(pd_df)  # Fallback to simple display
    except Exception as e:
        st.warning(f"No saved site data available: {str(e)}")
with tab3:
    st.header("🏊 Stage 3: Facility Attributes Lookup via Gemini")
    
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
                if not validate_api_key('gemini', st.session_state.api_keys['gemini']):
                    st.error("Invalid Gemini API Key. Please check and try again.")
                else:
                    enriched_sites = []
                    with st.spinner("Querying Gemini for each site..."):
                        # Create progress bar and status text
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Get total number of sites
                        total_sites = len(filtered_sites_df)
                        
                        for idx, row in enumerate(filtered_sites_df.iter_rows(named=True)):
                            # Update progress
                            progress = (idx + 1) / total_sites
                            progress_bar.progress(progress)
                            status_text.text(f"Processing site {idx + 1} of {total_sites}: {row['name']}")
                            
                            result = get_site_amenities_with_gemini(
                                site_name=row["name"],
                                address=row.get("address", f"{row['zip_code']}"),
                                amenities_list=amenities,
                                gemini_api_key=st.session_state.api_keys['gemini'],
                            )
                            site_entry = row.copy()
                            # Add all fields from the Gemini response
                            for amenity in amenities:
                                col = amenity.lower()
                                site_entry[col] = result.get(col, False)
                            site_entry["estimated_capacity"] = result.get("estimated_capacity", "")
                            site_entry["local_reputation"] = result.get("local_reputation", "")
                            site_entry["program_currently_hosted"] = result.get("program_currently_hosted", "Unknown")
                            site_entry["program_description"] = result.get("program_description", "")
                            enriched_sites.append(site_entry)
                            time.sleep(1.2)  # prevent rate limiting
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()

                    # Save to DB
                    enriched_df = pl.DataFrame(enriched_sites)
                    # Ensure we have lat/long columns
                    if "latitude" not in enriched_df.columns or "longitude" not in enriched_df.columns:
                        # Get lat/long from original data
                        lat_long_df = filtered_sites_df.select(["name", "latitude", "longitude"])
                        # Join with enriched data
                        enriched_df = enriched_df.join(
                            lat_long_df,
                            on="name",
                            how="left"
                        )
                    asyncio.run(save_df(enriched_df, "site_attributes"))
                    st.success("Amenities saved to database.")
                    
                    # Force a rerun to show the data editor
                    st.rerun()

    except Exception as e:
        st.error(f"Error loading filtered sites or querying Gemini: {e}")

    # Editable table for manual correction and extra fields
    try:
        attrs_df = asyncio.run(load_table("site_attributes"))
        if not attrs_df.is_empty():
            st.subheader("✅ Edit & Finalize Site Attributes")
            # Ensure all amenity columns, estimated_capacity, and local_reputation exist
            for col in [a.lower() for a in amenities] + ["estimated_capacity", "local_reputation", "program_currently_hosted", "proceed_score"]:
                if col not in attrs_df.columns:
                    attrs_df = attrs_df.with_columns(pl.lit("").alias(col))
            
            # Keep lat/long in the data but exclude from display
            editable_cols = ["name", "address", "zip_code", "rating"] + [a.lower() for a in amenities] + ["estimated_capacity", "local_reputation", "program_currently_hosted", "proceed_score", "program_description"]
            
            # Convert to pandas for editing, excluding lat/long from display
            pd_df = attrs_df.select(editable_cols).to_pandas()
            
            try:
                # Render the editor
                edited_df = st.data_editor(
                    pd_df,
                    use_container_width=True,
                    column_config=column_config,
                    num_rows="dynamic",
                    key="site_attrs_editor"
                )
                
                if st.button("💾 Save Final Site Attributes"):
                    try:
                        # Convert back to Polars
                        edited_pl_df = pl.DataFrame(edited_df)
                        
                        # Get lat/long from original data
                        lat_long_df = attrs_df.select(["name", "latitude", "longitude"])
                        
                        # Ensure name column is string type in both DataFrames
                        edited_pl_df = edited_pl_df.with_columns(pl.col("name").cast(pl.Utf8))
                        lat_long_df = lat_long_df.with_columns(pl.col("name").cast(pl.Utf8))
                        
                        # Join the DataFrames
                        final_df = edited_pl_df.join(
                            lat_long_df,
                            on="name",
                            how="left"
                        )
                        
                        # Save to database
                        asyncio.run(save_df(final_df, "site_attributes_final"))
                        st.success("Final site attributes saved.")
                    except Exception as e:
                        st.error(f"Error saving data: {str(e)}")
            except Exception as e:
                st.error(f"Error in data editor: {str(e)}")
                st.dataframe(pd_df)  # Fallback to simple display

            # Plot on map if lat/lon available
            if "latitude" in attrs_df.columns and "longitude" in attrs_df.columns:
                st.subheader("🗺️ Map of Sites")
                try:
                    # Convert to pandas and ensure numeric types
                    map_df = attrs_df.select(["name", "address", "latitude", "longitude", "rating", "place_type"]).to_pandas()
                    map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors='coerce')
                    map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors='coerce')
                    
                    # Drop any rows with invalid coordinates
                    map_df = map_df.dropna(subset=["latitude", "longitude"])
                    
                    if not map_df.empty:
                        import folium
                        from streamlit_folium import folium_static
                        
                        # Create a map centered on the mean of all points
                        center_lat = map_df["latitude"].mean()
                        center_lon = map_df["longitude"].mean()
                        
                        # Create the map with custom styling
                        m = folium.Map(
                            location=[center_lat, center_lon],
                            zoom_start=12,
                            tiles='OpenStreetMap'
                        )
                        
                        # Add a layer control
                        folium.LayerControl().add_to(m)
                        
                        # Define color mapping for place types
                        place_type_colors = {
                            'school': 'blue',
                            'park': 'green',
                            'community_center': 'purple',
                            'recreation_center': 'orange'
                        }
                        
                        # Add markers for each site with custom styling
                        for idx, row in map_df.iterrows():
                            # Create popup content with HTML formatting
                            popup_content = f"""
                            <div style='font-family: Arial, sans-serif;'>
                                <h4 style='margin: 0 0 5px 0;'>{row['name']}</h4>
                                <p style='margin: 0 0 5px 0;'><b>Address:</b> {row['address']}</p>
                                <p style='margin: 0 0 5px 0;'><b>Type:</b> {row['place_type'].replace('_', ' ').title()}</p>
                                <p style='margin: 0;'><b>Rating:</b> {float(row['rating']):.1f} {'⭐' * round(float(row['rating']))}</p>
                            </div>
                            """
                            
                            # Get color based on place type
                            color = place_type_colors.get(row['place_type'], 'red')
                            
                            # Create marker with custom icon
                            folium.Marker(
                                location=[row["latitude"], row["longitude"]],
                                popup=folium.Popup(popup_content, max_width=300),
                                tooltip=row["name"],
                                icon=folium.Icon(
                                    color=color,
                                    icon='info-sign',
                                    prefix='fa'
                                )
                            ).add_to(m)
                        
                        # Add a legend
                        legend_html = """
                        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                        padding: 10px; border: 2px solid grey; border-radius: 5px;">
                        <p style="margin: 0 0 5px 0;"><b>Place Types:</b></p>
                        """
                        for place_type, color in place_type_colors.items():
                            legend_html += f"""
                            <p style="margin: 0 0 5px 0;">
                                <span style="color: {color};">●</span> {place_type.replace('_', ' ').title()}
                            </p>
                            """
                        legend_html += "</div>"
                        m.get_root().html.add_child(folium.Element(legend_html))
                        
                        # Display the map
                        folium_static(m, width=800, height=600)
                    else:
                        st.warning("No valid coordinates found for mapping.")
                except Exception as e:
                    st.error(f"Error creating map: {str(e)}")
            else:
                st.info("No latitude/longitude columns found for mapping.")
    except Exception:
        pass

with tab4:
    st.header("🧭 Stage 4: Competitive Research")
    st.write("Automatically discover and analyze existing day camps in the area.")

    # Search parameters for competitor search
    st.subheader("🔍 Competitor Search Parameters")
    col1, col2 = st.columns(2)
    with col1:
        comp_search_radius_miles = st.number_input("Search Radius (miles)", min_value=1, max_value=10, value=3, step=1)
        comp_search_radius = int(comp_search_radius_miles * 1609.34)  # Convert miles to meters
        comp_min_rating = st.number_input("Minimum Rating", min_value=0.0, max_value=5.0, value=3.5, step=0.5)
    with col2:
        comp_types = st.multiselect(
            "Competitor Types",
            ["day camp", "summer camp", "child care", "campground"],
            default=["day camp", "summer camp"]
        )
        comp_max_results = st.number_input("Max Results per Zip Code", min_value=1, max_value=20, value=10)

    # Load priority zip codes
    try:
        priority_zips_df = asyncio.run(load_table("priority_zips"))
        if priority_zips_df.is_empty():
            st.warning("No priority zip codes found. Please run Stage 1 first.")
        else:
            zip_codes = priority_zips_df["zip_code"].to_list()
            if st.button("🔍 Find Competitor Camps Automatically"):
                if not validate_api_key('google', st.session_state.api_keys['google']):
                    st.error("Invalid Google Maps API Key. Please check and try again.")
                else:
                    from utils.utils import find_competitor_camps
                    with st.spinner("Searching for competitor camps using Google Maps API..."):
                        competitor_df = find_competitor_camps(
                            zip_codes,
                            st.session_state.api_keys['google'],
                            radius=comp_search_radius,
                            min_rating=comp_min_rating,
                            competitor_types=comp_types,
                            max_results_per_zip=comp_max_results
                        )
                        if competitor_df.is_empty():
                            st.warning("No competitor camps found in the selected area.")
                        else:
                            asyncio.run(save_df(competitor_df, "competitor_camps"))
                            st.success(f"Found and saved {competitor_df.height} competitor camps.")

    except Exception as e:
        st.error(f"Error loading priority zip codes: {e}")

    # Display and edit competitor camps
    try:
        comp_df = asyncio.run(load_table("competitor_camps"))
        if not comp_df.is_empty():
            st.subheader("📋 Competitor Camps (Auto-Discovered)")
            st.dataframe(comp_df.select(["name", "address", "zip_code", "rating", "place_type"]))

            # Plot competitor camps on map
            if "latitude" in comp_df.columns and "longitude" in comp_df.columns:
                st.subheader("🗺️ Map of Competitor Camps")
                import folium
                from streamlit_folium import folium_static
                import pandas as pd
                map_df = comp_df.select(["name", "address", "latitude", "longitude", "place_type"]).to_pandas()
                map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors='coerce')
                map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors='coerce')
                map_df = map_df.dropna(subset=["latitude", "longitude"])
                if not map_df.empty:
                    center_lat = map_df["latitude"].mean()
                    center_lon = map_df["longitude"].mean()
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')
                    place_type_colors = {
                        'day camp': 'blue',
                        'summer camp': 'green',
                        'child care': 'purple',
                        'campground': 'orange'
                    }
                    for idx, row in map_df.iterrows():
                        color = place_type_colors.get(row['place_type'], 'red')
                        folium.Marker(
                            location=[row["latitude"], row["longitude"]],
                            popup=f"{row['name']}<br>{row['address']}<br>{row['place_type'].title()}",
                            tooltip=row["name"],
                            icon=folium.Icon(color=color, icon='info-sign', prefix='fa')
                        ).add_to(m)
                    folium_static(m, width=800, height=600)
                else:
                    st.warning("No valid coordinates found for mapping.")
            else:
                st.info("No latitude/longitude columns found for mapping.")
        else:
            st.info("No competitor camps data available. Click the button above to search.")
    except Exception as e:
        st.error(f"Error displaying competitor camps: {e}")

    # Optional: Calculate distance to nearest competitor for each site
    try:
        site_df = asyncio.run(load_table("site_attributes_final"))
        comp_df = asyncio.run(load_table("competitor_camps"))
        if ("latitude" in site_df.columns and "longitude" in site_df.columns and
            "latitude" in comp_df.columns and "longitude" in comp_df.columns):
            from geopy.distance import geodesic
            site_rows = site_df.to_dicts()
            comp_rows = comp_df.to_dicts()
            nearest_distances = []
            for site in site_rows:
                site_loc = (float(site["latitude"]), float(site["longitude"]))
                min_dist = None
                for comp in comp_rows:
                    try:
                        comp_loc = (float(comp["latitude"]), float(comp["longitude"]))
                        dist = geodesic(site_loc, comp_loc).miles
                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                    except Exception:
                        continue
                nearest_distances.append(min_dist if min_dist is not None else "N/A")
            site_df = site_df.with_columns(pl.Series("nearest_competitor_miles", nearest_distances))
            st.write("Distance to nearest competitor (miles):")
            st.dataframe(site_df.select(["name", "nearest_competitor_miles"]))
    except Exception:
        pass

with tab5:
    st.header("✅ Stage 5: Top Sites")
    st.write("Filter for Top Sites based on Rating and Competitor Density.")
    
    # Add parameters for filtering
    col1, col2 = st.columns(2)
    with col1:
        min_rating = st.number_input(
            "Minimum Rating",
            min_value=0.0,
            max_value=5.0,
            value=3.5,
            step=0.1,
            help="Minimum rating required for a site to be considered"
        )
        max_competitors = st.number_input(
            "Maximum Competitors per Zip Code",
            min_value=0,
            max_value=20,
            value=3,
            step=1,
            help="Maximum number of competitor camps allowed in the same zip code"
        )
        min_capacity = st.number_input(
            "Minimum Estimated Capacity",
            min_value=0,
            max_value=1000,
            value=50,
            step=10,
            help="Minimum estimated capacity required for a site"
        )
    
    with col2:
        required_amenities = st.multiselect(
            "Required Amenities",
            ["Pool", "Field", "Gym", "Auditorium", "Kitchen", "Playground"],
            default=[],
            help="Sites must have at least one of these amenities"
        )
        max_sites = st.number_input(
            "Maximum Number of Sites to Show",
            min_value=1,
            max_value=50,
            value=18,
            step=1,
            help="Maximum number of top sites to display"
        )
        sort_by = st.selectbox(
            "Sort By",
            ["Rating", "Capacity", "Competitor Distance"],
            help="Primary sorting criteria for the sites"
        )
    
    try:
        site_df = asyncio.run(load_table("site_attributes_final"))
        comp_df = asyncio.run(load_table("competitor_camps"))
        
        if site_df.is_empty():
            st.warning("No site data available. Please complete previous stages first.")
        else:
            # Apply filters
            filtered = site_df.with_columns(
                pl.col("rating").cast(pl.Float64).alias("rating")
            ).filter(pl.col("rating") >= min_rating)
            
            # Filter by required amenities if any are selected
            if required_amenities:
                amenity_cols = [a.lower() for a in required_amenities]
                filtered = filtered.filter(
                    pl.any_horizontal(pl.col(amenity_cols))
                )
            
            # Filter by minimum capacity if available
            if "estimated_capacity" in filtered.columns:
                filtered = filtered.with_columns(
                    pl.col("estimated_capacity").cast(pl.Int64).alias("estimated_capacity")
                ).filter(pl.col("estimated_capacity") >= min_capacity)
            
            # Count competitor camps per zip_code
            if not comp_df.is_empty() and "zip_code" in comp_df.columns:
                comp_counts = comp_df.group_by("zip_code").agg(
                    pl.col("name").count().alias("comp_count")
                )
                filtered = filtered.join(comp_counts, on="zip_code", how="left")
                filtered = filtered.with_columns(
                    pl.col("comp_count").fill_null(0)
                )
                filtered = filtered.filter(pl.col("comp_count") <= max_competitors)
            
            # Sort based on selected criteria
            if sort_by == "Rating":
                filtered = filtered.sort("rating", descending=True)
            elif sort_by == "Capacity":
                filtered = filtered.sort("estimated_capacity", descending=True)
            elif sort_by == "Competitor Distance":
                if "nearest_competitor_miles" in filtered.columns:
                    filtered = filtered.sort("nearest_competitor_miles", descending=True)
                else:
                    st.warning("Competitor distance data not available. Sorting by rating instead.")
                    filtered = filtered.sort("rating", descending=True)
            
            # Limit to max sites
            top_sites = filtered.head(max_sites)
            
            # Display results
            st.subheader(f"Top {len(top_sites)} Sites")
            
            # Define columns to display, checking if they exist
            display_cols = ["name", "address", "zip_code", "rating"]
            optional_cols = ["place_type", "estimated_capacity", "local_reputation", "program_currently_hosted"]
            
            # Add optional columns if they exist
            for col in optional_cols:
                if col in top_sites.columns:
                    display_cols.append(col)
            
            # Display the dataframe with available columns
            st.dataframe(top_sites.select(display_cols))
            
            # Show statistics
            st.subheader("📊 Statistics")
            col1, col2, col3 = st.columns(3)
            
            # Rating statistics
            with col1:
                if "rating" in top_sites.columns:
                    try:
                        rating_mean = top_sites['rating'].mean()
                        if rating_mean is not None:
                            st.metric("Average Rating", f"{rating_mean:.1f}")
                        else:
                            st.metric("Average Rating", "N/A")
                    except:
                        st.metric("Average Rating", "N/A")
                else:
                    st.metric("Average Rating", "N/A")
            
            # Capacity statistics
            with col2:
                if "estimated_capacity" in top_sites.columns:
                    try:
                        # Convert to numeric, handling any non-numeric values
                        capacity_series = top_sites['estimated_capacity'].cast(pl.Float64)
                        capacity_mean = capacity_series.mean()
                        if capacity_mean is not None:
                            st.metric("Average Capacity", f"{int(capacity_mean)}")
                        else:
                            st.metric("Average Capacity", "N/A")
                    except:
                        st.metric("Average Capacity", "N/A")
                else:
                    st.metric("Average Capacity", "N/A")
            
            # Competitor statistics
            with col3:
                if "comp_count" in top_sites.columns:
                    try:
                        comp_mean = top_sites['comp_count'].mean()
                        if comp_mean is not None:
                            st.metric("Average Competitors", f"{comp_mean:.1f}")
                        else:
                            st.metric("Average Competitors", "N/A")
                    except:
                        st.metric("Average Competitors", "N/A")
                else:
                    st.metric("Average Competitors", "N/A")
            
            # Plot on map if lat/lon available
            if "latitude" in top_sites.columns and "longitude" in top_sites.columns:
                st.subheader("🗺️ Map of Top Sites")
                try:
                    top_map_df = top_sites.select(["name", "address", "latitude", "longitude"]).to_pandas()
                    top_map_df["latitude"] = pd.to_numeric(top_map_df["latitude"], errors='coerce')
                    top_map_df["longitude"] = pd.to_numeric(top_map_df["longitude"], errors='coerce')
                    top_map_df = top_map_df.dropna(subset=["latitude", "longitude"])
                    if not top_map_df.empty:
                        st.map(top_map_df.rename(columns={"latitude": "lat", "longitude": "lon"}))
                    else:
                        st.warning("No valid coordinates found for mapping.")
                except Exception as e:
                    st.error(f"Error creating map: {str(e)}")
            else:
                st.info("No latitude/longitude columns found for mapping.")
    except Exception as e:
        st.error(f"Error in Stage 5: {e}")
