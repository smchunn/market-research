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

# Configuration
GOOGLE_API_KEY = "INSERT-KEY-HERE"
CENSUS_API_KEY = "INSERT-KEY-HERE"
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
DATA_DIR = "./data/"
# Initialize APIs
census = Census(CENSUS_API_KEY)


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


def get_census_data(zip_codes):
    """Fetch population, children, and income data from Census API, supplement with free_zipcode_data."""
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


def filter_zip_codes(df):
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
def find_potential_sites(priority_zips):
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
            params = {"location": f"{lat},{lng}", "radius": 5000, "key": GOOGLE_API_KEY}
            # Use keyword for recreation_center, as it's not a supported type
            if place_type == "recreation_center":
                params["keyword"] = "recreation center"
            else:
                params["type"] = place_type
            try:
                while True:
                    response = requests.get(base_url, params=params, headers=headers)
                    response.raise_for_status()
                    data = response.json()
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
                    params = {"pagetoken": next_page_token, "key": GOOGLE_API_KEY}
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


def main():
    # Stage 1: Filter Zip Codes
    census_data_path = os.path.join(DATA_DIR, "census_data.xlsx")
    priority_zips_path = os.path.join(DATA_DIR, "priority_zips.xlsx")

    if os.path.exists(census_data_path) and os.path.exists(priority_zips_path):
        try:
            print(f"Loading existing files: {census_data_path}, {priority_zips_path}")
            census_data = pl.read_excel(census_data_path)
            priority_zips = pl.read_excel(priority_zips_path)
            # Validate schemas
            expected_columns = [
                "zip_code",
                "population",
                "children_proportion",
                "median_income",
                "pop_density",
            ]
            if not all(col in census_data.columns for col in expected_columns):
                raise ValueError(
                    f"{census_data_path} missing required columns: {expected_columns}"
                )
            expected_priority_columns = [
                "zip_code",
                "pop_density",
                "median_income",
                "children_proportion",
            ]
            if not all(
                col in priority_zips.columns for col in expected_priority_columns
            ):
                raise ValueError(
                    f"{priority_zips_path} missing required columns: {expected_priority_columns}"
                )
            print("Loaded census_data and priority_zips from Excel files.")
        except Exception as e:
            print(f"Error loading Excel files: {e}. Recalculating Stage 1.")
            census_data, priority_zips = None, None
    else:
        print("Excel files not found. Running Stage 1 calculations.")
        census_data, priority_zips = None, None

    if census_data is None or priority_zips is None:
        zip_codes = get_zip_codes(REGION_STATE, START_ZIP_CODE, RADIUS_MILES)
        if not zip_codes:
            print("No zip codes found. Exiting.")
            return
        census_data = get_census_data(zip_codes)
        if census_data.is_empty():
            print("No valid census data. Exiting.")
            return
        print(census_data.head())
        census_data.write_excel(census_data_path)
        priority_zips = filter_zip_codes(census_data)
        print(
            "Priority Zip Codes:",
            priority_zips.select(
                ["zip_code", "pop_density", "median_income", "children_proportion"]
            ),
        )
        priority_zips.write_excel(priority_zips_path)
    else:
        print(
            "Using loaded Priority Zip Codes:",
            priority_zips.select(
                ["zip_code", "pop_density", "median_income", "children_proportion"]
            ),
        )
    # Stage 2: Identify Potential Sites
    potential_sites = find_potential_sites(priority_zips)
    if not potential_sites.is_empty():
        print(
            "Potential Sites:",
            potential_sites.select(
                ["zip_code", "name", "address", "rating", "place_type"]
            ),
        )
        potential_sites.write_excel("./data/potential_sites.xlsx")


if __name__ == "__main__":
    main()
