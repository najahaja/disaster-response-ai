import osmnx as ox
import geopandas as gpd
import sys

def test_map_loading(location_name="Lahore, Pakistan"):
    print(f"Testing map loading for: {location_name}")
    # Define what features we want
    tags = {
        'building': True,
        'highway': True,
        'leisure': 'park',
        'amenity': 'hospital'
    }

    try:
        # 1. Download geospatial data from OpenStreetMap
        print(f"Downloading data for {location_name}...")

        # Get the bounding box for the location
        gdf = ox.geocode_to_gdf(location_name)
        print("Geocoding successful.")
        bbox = gdf.total_bounds
        print(f"Bounding box: {bbox}")

        # Download the features
        print("Downloading features...")
        features_gdf = ox.features_from_bbox(bbox=bbox, tags=tags)
        print(f"Features downloaded. Count: {len(features_gdf)}")
        
        print("✅ Map loading test successful!")

    except Exception as e:
        print(f"❌ Map loading failed with initial method: {e}")
        try:
            print("🔄 Attempting fallback: Geocoding to point and creating bbox...")
            # Fallback: Get point and create bbox
            lat, lon = ox.geocode(location_name)
            print(f"Geocoded to point: {lat}, {lon}")
            
            # Create a bbox (e.g., 1km radius)
            # roughly 0.01 degrees is ~1km
            north, south = lat + 0.01, lat - 0.01
            east, west = lon + 0.01, lon - 0.01
            bbox = (west, south, east, north) # minx, miny, maxx, maxy
            
            print(f"Created bbox: {bbox}")
            
            # Download features using this bbox
            print("Downloading features from bbox...")
            features_gdf = ox.features_from_bbox(bbox=bbox, tags=tags)
            print(f"Features downloaded. Count: {len(features_gdf)}")
            print("✅ Map loading test successful with fallback!")
            
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_map_loading()
