#!/usr/bin/env python3
"""
Utility script to download and cache OpenStreetMap data
"""

import os
import sys
import argparse
import osmnx as ox
import pickle
import time
from typing import List, Optional, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.utils.map_loader import MapLoader

def download_maps(locations: List[str], cache_dir: str = "./data/maps/osm_cache"):
    """
    Download and cache maps for specified locations with improved error handling
    """
    map_loader = MapLoader(cache_dir)
    
    print("🗺️  Starting map download process...")
    print("=" * 50)
    
    successful_downloads = 0
    
    for location in locations:
        print(f"📥 Downloading map for: {location}")
        
        try:
            # Add delay to avoid rate limiting
            time.sleep(1)
            
            # Try multiple approaches for geocoding
            graph = None
            geocoding_methods = [
                location,  # Try as-is first
                f"{location}, Pakistan",  # Add country if not specified
                location.replace(", Pakistan", ""),  # Try without country
            ]
            
            for method in geocoding_methods:
                try:
                    print(f"  🔍 Trying geocoding: '{method}'")
                    graph = map_loader.load_cached_map(method)
                    if graph:
                        break
                except Exception as e:
                    print(f"    ⚠️  Geocoding failed for '{method}': {e}")
                    continue
            
            if graph:
                print(f"✅ Success: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
                
                # Convert to grid for verification
                try:
                    grid, node_positions, edge_data = map_loader.graph_to_grid(graph)
                    if grid is not None:
                        print(f"📊 Grid created: {grid.shape}")
                        successful_downloads += 1
                    else:
                        print("❌ Failed to create grid from graph")
                except Exception as e:
                    print(f"❌ Error converting graph to grid: {e}")
            else:
                print("❌ All geocoding attempts failed")
                
        except Exception as e:
            print(f"❌ Error downloading {location}: {e}")
        
        print("-" * 30)
    
    # List available cached maps
    available_maps = map_loader.get_available_locations()
    print(f"\n📁 Available cached maps: {len(available_maps)}")
    for map_name in available_maps:
        print(f"  📍 {map_name}")
    
    return successful_downloads

def download_sample_maps(cache_dir: str = "./data/maps/osm_cache"):
    """
    Download sample maps that are known to work well
    """
    sample_locations = [
        # Smaller, well-defined areas that work reliably
        "Centaurus Mall, Islamabad, Pakistan",
        "Lahore Fort, Lahore, Pakistan", 
        "Clifton, Karachi, Pakistan",
        "Faisal Mosque, Islamabad, Pakistan",
        "Bahria Town, Lahore, Pakistan"
    ]
    
    print("🔄 Trying sample locations that typically work...")
    return download_maps(sample_locations, cache_dir)

def check_osmnx_setup():
    """Check if OSMnx is properly configured"""
    try:
        # Test basic functionality
        test_location = "Centaurus Mall, Islamabad"
        graph = ox.graph_from_place(test_location, network_type='drive', simplify=True)
        print("✅ OSMnx is working correctly")
        return True
    except Exception as e:
        print(f"❌ OSMnx configuration issue: {e}")
        return False

def main():
    """Main function for map download script"""
    parser = argparse.ArgumentParser(description="Download OpenStreetMap data for disaster response simulation")
    parser.add_argument(
        "--locations", 
        nargs="+",
        default=["Lahore, Pakistan", "Karachi, Pakistan", "Islamabad, Pakistan"],
        help="List of locations to download maps for"
    )
    parser.add_argument(
        "--cache-dir",
        default="./data/maps/osm_cache",
        help="Directory to cache downloaded maps"
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Use sample locations that are known to work"
    )
    parser.add_argument(
        "--test-osmnx",
        action="store_true", 
        help="Test OSMnx configuration first"
    )
    
    args = parser.parse_args()
    
    print("🚀 Disaster Response AI - Map Download Utility")
    print("==============================================")
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Test OSMnx if requested
    if args.test_osmnx:
        if not check_osmnx_setup():
            print("\n💡 Troubleshooting tips:")
            print("1. Check internet connection")
            print("2. Try: pip install osmnx --upgrade")
            print("3. Some locations may not be available in OSM")
            return
    
    # Download maps
    if args.use_samples:
        successful_downloads = download_sample_maps(args.cache_dir)
    else:
        successful_downloads = download_maps(args.locations, args.cache_dir)
    
    if successful_downloads > 0:
        print(f"\n🎉 Successfully downloaded {successful_downloads} maps!")
        print("📁 Cached maps are ready for use in the simulation")
    else:
        print(f"\n⚠️  No maps were successfully downloaded")
        print("💡 Try these solutions:")
        print("   - Use --use-samples flag for reliable locations")
        print("   - Use --test-osmnx to check configuration")
        print("   - Try smaller, more specific location names")
        print("   - Check your internet connection")
        print("   - Some locations may not be available in OpenStreetMap")

if __name__ == "__main__":
    main()