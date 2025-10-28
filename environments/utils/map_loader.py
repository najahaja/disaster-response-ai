import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MapLoader:
    """
    Simplified map loader that works without external dependencies
    """
    
    def __init__(self, cache_dir="./data/maps/osm_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_map_with_fallback(self, location_name: str) -> Any:
        """
        Load map with fallback to synthetic generation
        """
        logger.info(f"Loading map for: {location_name}")
        
        # Try to load from cache first
        cached_map = self._load_cached_map(location_name)
        if cached_map:
            return cached_map
        
        # Generate synthetic map
        return self._generate_synthetic_map(location_name)
    
    def _load_cached_map(self, location_name: str) -> Optional[Any]:
        """Try to load cached map"""
        cache_file = os.path.join(self.cache_dir, f"{self._sanitize_filename(location_name)}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    logger.info(f"Loaded cached map: {location_name}")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached map: {e}")
        
        return None
    
    def _generate_synthetic_map(self, location_name: str) -> Any:
        """Generate a synthetic city map"""
        logger.info(f"Generating synthetic map for: {location_name}")
        
        class SyntheticMap:
            def __init__(self, grid):
                self.grid = grid
                self.location = location_name
        
        # Create a realistic-looking city grid
        grid_size = 30
        grid = np.full((grid_size, grid_size), 1)  # Default to buildings
        
        # Create main roads (arterial roads)
        for i in range(0, grid_size, 5):
            grid[i, :] = 0  # Horizontal roads
            grid[:, i] = 0  # Vertical roads
        
        # Create smaller streets
        for i in range(2, grid_size, 5):
            if i + 1 < grid_size:
                grid[i, :] = 0
                grid[:, i] = 0
        
        # Add parks (larger green areas)
        park_locations = [(5, 5), (20, 5), (5, 20), (20, 20)]
        for y, x in park_locations:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid_size and 0 <= nx < grid_size:
                        grid[ny, nx] = 3  # PARK
        
        # Add hospitals (larger areas)
        hospital_locations = [(2, 2), (2, grid_size-3), (grid_size-3, 2), (grid_size-3, grid_size-3)]
        for y, x in hospital_locations:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid_size and 0 <= nx < grid_size:
                        grid[ny, nx] = 2  # HOSPITAL
        
        # Add some residential areas (mix of buildings and small roads)
        for y in range(8, 15):
            for x in range(8, 15):
                if (x + y) % 3 == 0:
                    grid[y, x] = 0  # Small road
                else:
                    grid[y, x] = 1  # Building
        
        # Cache the synthetic map
        synthetic_map = SyntheticMap(grid)
        self._cache_map(location_name, synthetic_map)
        
        return synthetic_map
    
    def _cache_map(self, location_name: str, map_data: Any):
        """Cache the generated map"""
        cache_file = os.path.join(self.cache_dir, f"{self._sanitize_filename(location_name)}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(map_data, f)
            logger.info(f"Cached synthetic map: {location_name}")
        except Exception as e:
            logger.warning(f"Failed to cache map: {e}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Convert string to safe filename"""
        return "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
    
    def get_available_locations(self) -> List[str]:
        """Get list of available cached locations"""
        locations = []
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                location_name = filename.replace('.pkl', '').replace('_', ' ')
                locations.append(location_name)
        return locations

def test_map_loader():
    """Test the map loader"""
    loader = MapLoader()
    
    # Test with a location
    location = "Lahore, Pakistan"
    map_data = loader.load_map_with_fallback(location)
    
    if map_data and hasattr(map_data, 'grid'):
        print(f"✅ Successfully loaded/generated map: {map_data.grid.shape}")
        print(f"   Location: {map_data.location}")
        
        # Show grid composition
        unique, counts = np.unique(map_data.grid, return_counts=True)
        cell_types = {0: 'Roads', 1: 'Buildings', 2: 'Hospitals', 3: 'Parks'}
        for val, count in zip(unique, counts):
            type_name = cell_types.get(val, f'Unknown({val})')
            print(f"   {type_name}: {count} cells")
    else:
        print("❌ Failed to load/generate map")

if __name__ == "__main__":
    test_map_loader()