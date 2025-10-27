#!/usr/bin/env python3
"""
Utility script to download and cache OpenStreetMap data
"""

import os
import sys
import argparse
import osmnx as ox
import pickle
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.utils.map_loader import MapLoader

def download_maps(locations: List[str], cache_dir: str = "./data/maps/osm_cache"):
    """
    Download and cache maps for specified locations
    """
    map_loader = MapLoader(cache_dir)
    
    print("🗺️  Starting map download process...")
    print("=" * 50)
    
    for location in locations:
        print(f"📥 Downloading map for: {location}")
        
        try:
            # This will download and cache the map
            graph = map_loader.load_cached_map(location)
            
            if graph:
                print(f"✅ Success: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
                
                # Convert to grid for verification
                grid, node_positions, edge_data = map_loader.graph_to_grid(graph)
                if grid is not None:
                    print(f"📊 Grid created: {grid.shape}")
                else:
                    print("❌ Failed to create grid")
            else:
                print("❌ Failed to download map")
                
        except Exception as e:
            print(f"❌ Error downloading {location}: {e}")
        
        print("-" * 30)
    
    # List available cached maps
    available_maps = map_loader.get_available_locations()
    print(f"\n📁 Available cached maps: {len(available_maps)}")
    for map_name in available_maps:
        print(f"  📍 {map_name}")

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
    
    args = parser.parse_args()
    
    print("🚀 Disaster Response AI - Map Download Utility")
    print("==============================================")
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Download maps
    download_maps(args.locations, args.cache_dir)
    
    print("\n🎉 Map download process completed!")
    print("📁 Cached maps are ready for use in the simulation")

if __name__ == "__main__":
    main()