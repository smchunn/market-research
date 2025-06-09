import polars as pl
import requests
from census import Census
import geopandas as gpd
import folium
from bs4 import BeautifulSoup
import time
import us
from us.states import State
import os
from math import radians, sin, cos, sqrt, atan2
import google.generativeai as genai
import logging
from datetime import datetime
import sys
import re

# Configure logging based on debug flag
def setup_logging(debug=False):
    """Configure logging with appropriate level and handlers."""
    level = logging.DEBUG if debug else logging.INFO
    handlers = [logging.FileHandler('api_requests.log')]
    
    # Add console handler only in debug mode
    if debug:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        handlers.append(console_handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

# Configuration
REGION_STATE = "DC"
# REGION_CITY = "New York"
START_ZIP_CODE = "20001"  # Starting zip code for radius search
RADIUS_MILES = 5  # Radius in miles to collect zip codes
POP_DENSITY_THRESHOLD = 1000  # people per sq mile
INCOME_THRESHOLD = 75000  # median household income
CHILDREN_THRESHOLD = 0.15  # proportion under 18
MAX_DAY_CAMPS_PER_ZIP = 3
MIN_PROCEED_SCORE = 3.5
TOP_SITES_COUNT = 18
ZIPCODE_DATA_URL = "https://raw.githubusercontent.com/midwire/free_zipcode_data/master/all_us_zipcodes.csv"
ZCTA_SHAPEFILE_URL = (
    "/Users/smchunn/Downloads/tl_2023_us_zcta520/tl_2023_us_zcta520.shp"
    # f"https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/tl_2023_us_zcta520.zip"
)


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in miles using Haversine formula."""
    R = 3958.8  # Earth's radius in miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# Stage 1: Zip Code List + Neighborhoods
def get_zcta_areas(zip_codes):
    """Fetch ZCTA shapefile and return a dictionary of ZCTA to area (in sq miles)."""
    try:
        print(f"Fetching shapefile from {ZCTA_SHAPEFILE_URL}")
        gdf = gpd.read_file(ZCTA_SHAPEFILE_URL)
        print(f"Shapefile columns: {list(gdf.columns)}")
        # Use ZCTA5CE23 for 2023 shapefile
        col = "ZCTA5CE20"
        if col not in gdf.columns:
            raise KeyError(f"Column '{col}' not found in shapefile")
        # Filter for zip codes being processed
        gdf = gdf[gdf[col].isin(zip_codes)]
        print(f"Original CRS: {gdf.crs}")
        # Reproject to UTM Zone 18N (EPSG:32618) for accurate area calculation
        gdf = gdf.to_crs(epsg=32618)
        if not isinstance(gdf, gpd.GeoDataFrame):
            return {}
        print(f"Reprojected CRS: {gdf.crs}")
        # Calculate area in square meters and convert to square miles (1 sq meter = 3.861e-7 sq miles)
        gdf["area_sq_miles"] = gdf.geometry.area * 3.861e-7
        areas = dict(zip(gdf[col], gdf["area_sq_miles"]))
        print(f"Loaded areas for {len(areas)} ZCTAs: {list(areas.items())[:5]}")
        return areas
    except Exception as e:
        print(f"Error fetching or processing ZCTA shapefile: {e}")
        return {}


def old_get_zip_codes(state, city):
    """Fetch valid zip codes from free_zipcode_data GitHub repository."""
    try:
        df = pl.read_csv(ZIPCODE_DATA_URL)
        nyc_boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        df = df.filter(
            (pl.col("state") == state)
            & (pl.col("city").is_in(nyc_boroughs) | (pl.col("city") == city))
        )
        zip_codes = df["code"].cast(str).str.zfill(5).unique().to_list()
        # Exclude known non-ZCTA ranges (e.g., 101XX, 102XX, and other invalid codes)
        zip_codes = [
            z
            for z in zip_codes
            if not z.startswith("101")
            and not z.startswith("102")
            and not z.startswith("1007")  # Exclude 10072, etc.
        ]
        print(f"Found {len(zip_codes)} zip codes: {zip_codes[:10]}...")
        return zip_codes
    except Exception as e:
        print(f"Error fetching zip code data: {e}")
        return []


def get_zip_codes(state, start_zip, radius_miles):
    """Fetch zip codes within radius_miles from start_zip in the given state."""
    try:
        df = pl.read_csv(ZIPCODE_DATA_URL)
        df = df.with_columns(
            pl.col("code").cast(str).str.zfill(5).alias("code")
        ).filter(pl.col("state") == state)

        # Get coordinates of starting zip code
        start_zip_info = df.filter(pl.col("code") == start_zip)
        if start_zip_info.is_empty():
            print(f"Error: Starting zip code {start_zip} not found in {state}.")
            return []
        start_lat = start_zip_info["lat"][0]
        start_lon = start_zip_info["lon"][0]
        if not start_lat or not start_lon:
            print(
                f"Error: Invalid coordinates for {start_zip} (lat={start_lat}, lon={start_lon})."
            )
            return []

        # Calculate distance to all zip codes
        df = df.with_columns(
            pl.struct(["lat", "lon"])
            .map_elements(
                lambda x: haversine(start_lat, start_lon, x["lat"], x["lon"]),
                return_dtype=pl.Float64,
            )
            .alias("distance_miles")
        )

        # Filter zip codes within radius
        zip_codes = (
            df.filter(pl.col("distance_miles") <= radius_miles)["code"]
            .unique()
            .to_list()
        )
        # Exclude non-ZCTA ranges (e.g., 101XX, 102XX)
        zip_codes = [
            z
            for z in zip_codes
            if not z.startswith("101")
            and not z.startswith("102")
            and not z.startswith("1007")
        ]
        print(
            f"Found {len(zip_codes)} zip codes within {radius_miles} miles of {start_zip}: {zip_codes[:10]}..."
        )
        return zip_codes
    except Exception as e:
        print(f"Error fetching zip code data: {e}")
        return []


def log_api_request(api_type: str, endpoint: str, params: dict = None, status: str = None, error: str = None):
    """Log API request details."""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'api_type': api_type,
        'endpoint': endpoint,
        'params': {k: v for k, v in (params or {}).items() if k != 'key'},  # Exclude API keys
        'status': status,
        'error': error
    }
    logging.info(f"API Request: {log_data}")


def get_census_data(zip_codes, census_api_key=None):
    """Fetch population, children, and income data from Census API, supplement with free_zipcode_data."""
    census = Census(census_api_key)
    zip_df = pl.read_csv(ZIPCODE_DATA_URL)
    zip_df = zip_df.with_columns(
        pl.col("code").cast(str).str.zfill(5).alias("code")
    ).filter(pl.col("code").is_in(zip_codes))

    data = []
    zcta_areas = get_zcta_areas(zip_codes)
    default_area = 1.0
    # state = us.states.lookup(REGION_STATE)
    # state_fips = state.fips if isinstance(state, State) else None
    # if not state_fips:
    #     print(f"Invalid state code: {REGION_STATE}")
    #     return pl.DataFrame()
    for zip_code in zip_codes:
        print(zip_code)
        try:
            log_api_request('census', 'acs5.zipcode', {'zip_code': zip_code})
            result = census.acs5.zipcode(
                fields=(
                    "B01001_001E",  # Total population
                    "B01001_003E",
                    "B01001_004E",
                    "B01001_005E",
                    "B01001_006E",  # Males under 18
                    "B01001_027E",
                    "B01001_028E",
                    "B01001_029E",
                    "B01001_030E",  # Females under 18
                    "B06011_001E",  # Median household income
                ),
                state_fips="*",
                county_fips="*",
                tract="*",
                zcta=zip_code,
                year=2023,
            )
            log_api_request('census', 'acs5.zipcode', {'zip_code': zip_code}, 'success')
            if result:
                pop = result[0]["B01001_001E"]
                if pop <= 0:
                    print(f"Skipping {zip_code}: No population (non-residential).")
                    continue
                children_fields = [
                    "B01001_003E",
                    "B01001_004E",
                    "B01001_005E",
                    "B01001_006E",
                    "B01001_027E",
                    "B01001_028E",
                    "B01001_029E",
                    "B01001_030E",
                ]
                children = 0
                for field in children_fields:
                    try:
                        children += (
                            result[0][field] if result[0][field] is not None else 0
                        )
                    except KeyError as e:
                        print(f"Field {field} missing for {zip_code}: {e}")
                        continue
                income = (
                    result[0]["B06011_001E"]
                    if result[0]["B06011_001E"] is not None
                    and result[0]["B06011_001E"] > 0
                    else 0
                )
                area = zcta_areas.get(zip_code, default_area)
                density = pop / area if area > 0 else 0
                children_prop = children / pop if pop > 0 else 0
                data_point = {
                    "zip_code": zip_code,
                    "population": pop,
                    "children_proportion": children_prop,
                    "median_income": income,
                    "pop_density": density,
                }
                print(data_point)
                data.append(data_point)
        except Exception as e:
            print(f"Error fetching census data for {zip_code}: {e}")
            continue
    df = pl.DataFrame(data)
    if df.is_empty():
        print("No census data retrieved. Check zip codes or API key.")
    return df


def filter_zip_codes(df, pop_density_threshold, income_threshold, children_threshold):
    """Filter zip codes by population density, income, and children."""
    if df.is_empty():
        print("Cannot filter empty DataFrame. Exiting.")
        return df
    return df.filter(
        (pl.col("pop_density") > POP_DENSITY_THRESHOLD)
        & (pl.col("median_income") > INCOME_THRESHOLD)
        & (pl.col("children_proportion") > CHILDREN_THRESHOLD)
    ).sort(["pop_density", "median_income"], descending=True)


# Stage 2: Identify Potential Day Camp Sites
def find_potential_sites(priority_zips, google_api_key):
    """Search for potential day camp sites in priority zip codes using Google Maps Places API (HTTP)."""
    zip_df = pl.read_csv(ZIPCODE_DATA_URL)
    zip_df = zip_df.with_columns(
        pl.col("code").cast(str).str.zfill(5).alias("code")
    ).filter(pl.col("code").is_in(priority_zips["zip_code"].to_list()))

    # Verify column names
    print(f"Zip code CSV columns: {zip_df.columns}")
    if "lat" not in zip_df.columns or "lon" not in zip_df.columns:
        print("Error: Required columns 'lat' and/or 'lon' not found in zip code CSV.")
        return pl.DataFrame()

    place_types = ["community_center", "school", "park", "recreation_center"]
    sites = []
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    headers = {"Referer": "http://localhost"}  # Add referer for API keys requiring it

    for zip_code in priority_zips["zip_code"]:
        zip_info = zip_df.filter(pl.col("code") == zip_code)
        if zip_info.is_empty():
            print(f"Skipping {zip_code}: No data found in zipcode CSV.")
            continue
        try:
            lat = zip_info["lat"][0]
            lng = zip_info["lon"][0]
            if not lat or not lng or None in (lat, lng):
                print(f"Skipping {zip_code}: Invalid or missing coordinates.")
                continue
        except Exception as e:
            print(f"Error accessing coordinates for {zip_code}: {e}")
            continue

        print(f"Searching for sites in zip {zip_code} (lat={lat}, lng={lng})")
        for place_type in place_types:
            params = {"location": f"{lat},{lng}", "radius": 5000, "key": google_api_key}
            # Use keyword for recreation_center, as it's not a supported type
            if place_type == "recreation_center":
                params["keyword"] = "recreation center"
            else:
                params["type"] = place_type
            try:
                log_api_request('google', 'place/nearbysearch', params)
                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                log_api_request('google', 'place/nearbysearch', params, data.get('status'))
                if data["status"] != "OK":
                    print(
                        f"API error for {place_type} in {zip_code}: {data.get('error_message', data['status'])}"
                    )
                    print(f"Full API response: {data}")
                    break
                for place in data.get("results", []):
                    if "rating" not in place or place["rating"] < MIN_PROCEED_SCORE:
                        continue
                    site = {
                        "zip_code": zip_code,
                        "name": place.get("name", ""),
                        "address": place.get("vicinity", ""),
                        "place_id": place.get("place_id", ""),
                        "rating": place.get("rating", 0.0),
                        "latitude": place.get("geometry", {})
                        .get("location", {})
                        .get("lat", 0.0),
                        "longitude": place.get("geometry", {})
                        .get("location", {})
                        .get("lng", 0.0),
                        "place_type": place_type,
                    }
                    sites.append(site)
                next_page_token = data.get("next_page_token")
                if not next_page_token:
                    break
                time.sleep(2)
                params = {"pagetoken": next_page_token, "key": google_api_key}
            except Exception as e:
                print(f"Error searching for {place_type} in {zip_code}: {e}")
                print(f"Request URL: {response.url}")
                continue

        # Limit to MAX_DAY_CAMPS_PER_ZIP
        zip_sites = [s for s in sites if s["zip_code"] == zip_code]
        if len(zip_sites) > MAX_DAY_CAMPS_PER_ZIP:
            zip_sites = sorted(zip_sites, key=lambda x: x["rating"], reverse=True)[
                :MAX_DAY_CAMPS_PER_ZIP
            ]
            sites = [s for s in sites if s["zip_code"] != zip_code] + zip_sites

    df = pl.DataFrame(sites)
    if df.is_empty():
        print("No potential sites found.")
    else:
        print(
            f"Found {len(df)} potential sites across {len(df['zip_code'].unique())} zip codes."
        )
    return df


def get_site_amenities_with_gemini(site_name, address, amenities_list, gemini_api_key):
    genai.configure(api_key=gemini_api_key)
    genai.list_models()
    model = genai.GenerativeModel("gemini-1.5-pro-002")

    prompt = f"""Analyze this place and provide information about its amenities, capacity, reputation, and current programs.
For each amenity, respond with EXACTLY "Yes" or "No" (case sensitive).
For capacity, provide a SINGLE NUMBER estimate of how many people the facility can accommodate.
Respond with one line per item in this format:
Amenity: Yes or No
Capacity: [single number estimate of people capacity]
Reputation: [1-3 keywords describing local reputation, comma separated]
Current Programs: [Yes/No/Unknown] - [brief description if Yes]

Place:
Name: {site_name}
Address: {address}

Amenities to check: {', '.join(amenities_list)}
"""
    try:
        print(f"\n=== Processing Gemini request for {site_name} ===")
        print(f"Address: {address}")
        print(f"Amenities to check: {amenities_list}")
        
        log_api_request('gemini', 'generate_content', {'site_name': site_name})
        response = model.generate_content(prompt)
        log_api_request('gemini', 'generate_content', {'site_name': site_name}, 'success')
        
        print("\n=== Raw Gemini Response ===")
        print(response.text)
        
        result_lines = response.text.strip().split("\n")
        parsed = {}
        
        # Initialize all amenities as False
        for amenity in amenities_list:
            parsed[amenity.lower()] = False
        
        print("\n=== Parsing Response ===")
        for line in result_lines:
            if ":" in line:
                k, v = line.split(":", 1)
                key = k.strip().lower()
                value = v.strip()
                print(f"Processing line - Key: '{key}', Value: '{value}'")
                
                # Handle amenities - only set to True if explicitly "Yes"
                if key in [a.lower() for a in amenities_list]:
                    # Convert value to lowercase and strip whitespace for comparison
                    value = value.lower().strip()
                    # Only set to True if the value is exactly "yes"
                    parsed[key] = value == "yes"
                    print(f"  Amenity '{key}': {parsed[key]} (value was '{value}')")
                # Handle capacity - extract single number
                elif key == "capacity":
                    try:
                        # Extract first number found in the string
                        numbers = re.findall(r'\d+', value)
                        if numbers:
                            capacity = int(numbers[0])
                            parsed["estimated_capacity"] = capacity
                            print(f"  Capacity: {capacity}")
                        else:
                            parsed["estimated_capacity"] = 0
                            print(f"  No capacity number found in: {value}")
                    except:
                        parsed["estimated_capacity"] = 0
                        print(f"  Error parsing capacity from: {value}")
                # Handle reputation
                elif key == "reputation":
                    keywords = [k.strip() for k in value.split(",")]
                    keywords = keywords[:3]  # Take only first 3 keywords
                    parsed["local_reputation"] = ", ".join(keywords)
                    print(f"  Reputation: {parsed['local_reputation']}")
                # Handle current programs
                elif key == "current programs":
                    parts = value.split("-", 1)
                    status = parts[0].strip().lower()
                    description = parts[1].strip() if len(parts) > 1 else ""
                    
                    if status == "yes":
                        parsed["program_currently_hosted"] = "Yes"
                        parsed["program_description"] = description
                    elif status == "no":
                        parsed["program_currently_hosted"] = "No"
                    else:
                        parsed["program_currently_hosted"] = "Unknown"
                    print(f"  Program Status: {parsed['program_currently_hosted']}")
                    if description:
                        print(f"  Program Description: {description}")
        
        print("\n=== Final Parsed Results ===")
        for key, value in parsed.items():
            print(f"{key}: {value}")
        
        return parsed
    except Exception as e:
        print(f"\n=== Error Processing {site_name} ===")
        print(f"Error: {str(e)}")
        # Return default values on error
        default_parsed = {a.lower(): False for a in amenities_list}
        default_parsed.update({
            "estimated_capacity": 0,
            "local_reputation": "",
            "program_currently_hosted": "Unknown"
        })
        print("Returning default values due to error")
        return default_parsed


def validate_api_key(api_type: str, api_key: str) -> bool:
    """Validate API keys before use.
    
    Args:
        api_type: Type of API ('census', 'google', or 'gemini')
        api_key: The API key to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not api_key:
        return False
        
    try:
        if api_type == 'census':
            # Test Census API with a simple query
            census = Census(api_key)
            census.acs5.zipcode(
                fields=("B01001_001E",),
                state_fips="*",
                county_fips="*",
                tract="*",
                zcta="20001",
                year=2023
            )
            return True
            
        elif api_type == 'google':
            # Test Google Maps API with a simple Places query
            base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                "location": "38.8977,-77.0365",  # DC coordinates
                "radius": 1000,
                "key": api_key
            }
            response = requests.get(base_url, params=params)
            return response.status_code == 200
            
        elif api_type == 'gemini':
            # Test Gemini API by initializing the model
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-pro-002")
            return True
            
        return False
    except Exception as e:
        print(f"Error validating {api_type} API key: {e}")
        return False


def find_competitor_camps(
    zip_codes,
    google_api_key,
    radius=5000,
    min_rating=0.0,
    competitor_types=None,
    max_results_per_zip=10
):
    """
    Search for competitor day camps in the given zip codes using Google Maps Places API.
    Returns a Polars DataFrame of competitor camps.
    """
    if competitor_types is None or len(competitor_types) == 0:
        competitor_types = ["day camp", "summer camp", "child care", "campground"]
    all_competitors = []
    zip_df = pl.read_csv(ZIPCODE_DATA_URL)
    zip_df = zip_df.with_columns(
        pl.col("code").cast(str).str.zfill(5).alias("code")
    ).filter(pl.col("code").is_in(zip_codes))
    for zip_code in zip_codes:
        zip_info = zip_df.filter(pl.col("code") == zip_code)
        if zip_info.is_empty():
            continue
        lat, lng = zip_info["lat"][0], zip_info["lon"][0]
        zip_sites = []
        for keyword in competitor_types:
            params = {
                "location": f"{lat},{lng}",
                "radius": radius,
                "keyword": keyword,
                "key": google_api_key
            }
            try:
                log_api_request('google', 'place/nearbysearch', params)
                response = requests.get(
                    "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                    params=params
                )
                data = response.json()
                log_api_request('google', 'place/nearbysearch', params, data.get('status'))
                if data.get("status") == "OK":
                    for place in data.get("results", []):
                        rating = place.get("rating", 0.0)
                        if rating == "":
                            rating = 0.0
                        try:
                            rating = float(rating)
                        except Exception:
                            rating = 0.0
                        if rating >= min_rating:
                            zip_sites.append({
                                "name": place.get("name", ""),
                                "address": place.get("vicinity", ""),
                                "zip_code": zip_code,
                                "latitude": place.get("geometry", {}).get("location", {}).get("lat", 0.0),
                                "longitude": place.get("geometry", {}).get("location", {}).get("lng", 0.0),
                                "place_type": keyword,
                                "rating": rating,
                                "website": place.get("website", ""),
                                "reputation": "",
                                "estimated_size": "",
                                "price_point": ""
                            })
                # Handle pagination if needed
                while "next_page_token" in data:
                    time.sleep(2)
                    params = {
                        "pagetoken": data["next_page_token"],
                        "key": google_api_key
                    }
                    response = requests.get(
                        "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                        params=params
                    )
                    data = response.json()
                    if data.get("status") == "OK":
                        for place in data.get("results", []):
                            rating = place.get("rating", 0.0)
                            if rating == "":
                                rating = 0.0
                            try:
                                rating = float(rating)
                            except Exception:
                                rating = 0.0
                            if rating >= min_rating:
                                zip_sites.append({
                                    "name": place.get("name", ""),
                                    "address": place.get("vicinity", ""),
                                    "zip_code": zip_code,
                                    "latitude": place.get("geometry", {}).get("location", {}).get("lat", 0.0),
                                    "longitude": place.get("geometry", {}).get("location", {}).get("lng", 0.0),
                                    "place_type": keyword,
                                    "rating": rating,
                                    "website": place.get("website", ""),
                                    "reputation": "",
                                    "estimated_size": "",
                                    "price_point": ""
                                })
            except Exception as e:
                print(f"Error searching for competitors in {zip_code} with keyword '{keyword}': {e}")
                continue
        # Sort and limit results per zip code
        zip_sites = sorted(zip_sites, key=lambda x: x["rating"], reverse=True)[:max_results_per_zip]
        all_competitors.extend(zip_sites)
    return pl.DataFrame(all_competitors)
