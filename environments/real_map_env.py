import osmnx as ox
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import yaml
import sys
import os
import traceback
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.simple_grid_env import SimpleGridEnv
from environments.utils.map_loader import MapLoader
from environments.utils.visualization import VisualizationUtils
from environments.utils.disaster_generator import DisasterGenerator

class RealMapEnv(SimpleGridEnv):
    """
    Real map environment that uses OpenStreetMap data or falls back to generated maps
    """
    
    def __init__(self, location_name=None, config_path="config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Gymnasium required attributes
        self.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
        
        # Real map attributes
        self.location_name = location_name
        self.real_map_loaded = False
        self.map_data = None
        
        # Environment parameters
        self.grid_size = self.config['environment']['grid_size']
        self.cell_size = self.config['environment']['cell_size']
        self.cell_types = self.config['environment']['cell_types']
        self.colors = self.config['visualization']['colors']
        self.max_steps = self.config['environment']['max_steps']
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.grid_size, self.grid_size, 3),
            dtype=np.uint8
        )
        
        # Environment state
        self.grid = None
        self.agents = {}
        self.civilians = []
        self.collapsed_buildings = []
        self.blocked_roads = []
        self.disaster_triggered = False
        self.step_count = 0
        
        # PyGame setup
        self.screen = None
        self.font = None
        self.display_available = False
        
        self.initialize_pygame()
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment with a real map or fallback to generated grid"""
        print(f"🗺️  Loading map for {self.location_name or 'default location'}...")

        if self.location_name:
            try:
                # 1. Download geospatial data from OpenStreetMap
                print(f"Downloading data for {self.location_name}...")

                # Get the bounding box for the location
                try:
                    gdf = ox.geocode_to_gdf(self.location_name)
                    bbox = gdf.total_bounds
                except (TypeError, ValueError):
                    print(f"⚠️ Could not get polygon for {self.location_name}, falling back to point-based bbox...")
                    lat, lon = ox.geocode(self.location_name)
                    # Create a ~2km x 2km bbox (0.01 degrees is roughly 1.1km)
                    north, south = lat + 0.01, lat - 0.01
                    east, west = lon + 0.01, lon - 0.01
                    bbox = (west, south, east, north)

                # Define what features we want
                tags = {
                    'building': True,
                    'highway': True,
                    'leisure': 'park',
                    'amenity': 'hospital'
                }

                # Download the features
                features_gdf = ox.features_from_bbox(bbox=bbox, tags=tags)

                # Separate the features
                buildings = features_gdf[features_gdf['building'].notnull()]
                roads = features_gdf[features_gdf['highway'].notnull()]
                hospitals = features_gdf[features_gdf['amenity'] == 'hospital']
                parks = features_gdf[features_gdf['leisure'] == 'park']

                print("✅ Data downloaded. Rasterizing map...")

                # 2. Create an empty grid
                # We fill with 'OPEN_SPACE' first, then add buildings, etc.
                self.grid = np.full((self.grid_size, self.grid_size), 
                                self.cell_types['OPEN_SPACE'], dtype=np.uint8)

                # 3. Create a "transform" to map geometries to grid pixels
                # This defines the boundaries of our grid
                map_transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], 
                                            self.grid_size, self.grid_size)

                # 4. "Burn" the features onto the grid
                # We create (geometry, value) pairs using mapping() to ensure rasterio compatibility

                # Burn buildings (as default 'BUILDING')
                building_shapes = [(mapping(geom), int(self.cell_types['BUILDING'])) for geom in buildings.geometry]
                features.rasterize(shapes=building_shapes, fill=0, out=self.grid, 
                                transform=map_transform, all_touched=True)

                # Burn parks (overwrites buildings)
                park_shapes = [(mapping(geom), int(self.cell_types['OPEN_SPACE'])) for geom in parks.geometry]
                features.rasterize(shapes=park_shapes, fill=0, out=self.grid, 
                                transform=map_transform, all_touched=True)

                # Burn roads (overwrites buildings and parks)
                road_shapes = [(mapping(geom), int(self.cell_types['ROAD'])) for geom in roads.geometry]
                features.rasterize(shapes=road_shapes, fill=0, out=self.grid, 
                                transform=map_transform, all_touched=True)

                # Burn hospitals (overwrites everything else)
                hospital_shapes = [(mapping(geom), int(self.cell_types['HOSPITAL'])) for geom in hospitals.geometry]
                features.rasterize(shapes=hospital_shapes, fill=0, out=self.grid, 
                                transform=map_transform, all_touched=True)

                self.real_map_loaded = True
                print(f"✅ Real map loaded: {self.grid_size}x{self.grid_size} grid for {self.location_name}")
                return

            except Exception as e:
                print(f"⚠️  Failed to load real map: {e}. Using fallback.")
                traceback.print_exc()
                # self.handle_error(f"MapLoad Error: {e}") # Log to dashboard

        # Fallback to simple grid
        print("🔄 Using enhanced generated grid as fallback")
        self._initialize_enhanced_grid()
        self.real_map_loaded = False
    
    def _initialize_enhanced_grid(self):
        """Initialize an enhanced grid with more realistic features"""
        self.grid = np.full((self.grid_size, self.grid_size), 
                           self.cell_types['BUILDING'])
        
        # Create realistic road network
        self._create_road_network()
        
        # Add realistic features
        self._add_realistic_features()
    
    def _create_road_network(self):
        """Create a more realistic road network"""
        # Main arterial roads
        for i in range(0, self.grid_size, 4):
            if i < self.grid_size:
                self.grid[i, :] = self.cell_types['ROAD']  # Horizontal roads
                self.grid[:, i] = self.cell_types['ROAD']  # Vertical roads
        
        # Secondary roads
        for i in range(2, self.grid_size, 6):
            if i < self.grid_size:
                # Connect neighborhoods with smaller roads
                start = max(0, i-1)
                end = min(self.grid_size, i+2)
                self.grid[i, start:end] = self.cell_types['ROAD']
                self.grid[start:end, i] = self.cell_types['ROAD']
    
    def _add_realistic_features(self):
        """Add realistic city features"""
        # Add hospitals at strategic locations
        hospital_positions = [
            [2, 2], [2, self.grid_size-3], 
            [self.grid_size-3, 2], [self.grid_size-3, self.grid_size-3],
            [self.grid_size//2, self.grid_size//2]  # Central hospital
        ]
        for pos in hospital_positions:
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                # Create hospital complex (3x3 area)
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        y, x = pos[0] + dy, pos[1] + dx
                        if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                            self.grid[y, x] = self.cell_types['HOSPITAL']
        
        # Add parks and open spaces
        park_locations = [
            [5, 5], [5, self.grid_size-6], [self.grid_size-6, 5]
        ]
        for center in park_locations:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    y, x = center[0] + dy, center[1] + dx
        
        # Draw grid
        VisualizationUtils.draw_grid(self.screen, self.grid, self.colors, self.cell_size)
        
        # Draw civilians
        VisualizationUtils.draw_civilians(self.screen, self.civilians, self.cell_size)
        
        # Draw agents
        VisualizationUtils.draw_agents(self.screen, self.agents, self.cell_size, self.font)
        
        # Draw info panel
        self._draw_real_map_info()
        # DO NOT call pygame.display.flip()
        
        # RETURN the surface for Streamlit to use
        return self.screen
    
    def _draw_real_map_info(self):
        """Draw real map information panel"""
        panel_x = self.grid_size * self.cell_size
        panel_width = 220
        
        # Draw panel background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                         (panel_x, 0, panel_width, self.screen.get_height()))
        
        # Draw title
        title_text = "REAL MAP INFO" if self.real_map_loaded else "GENERATED MAP"
        title_surface = self.font.render(title_text, True, (255, 255, 255))
        self.screen.blit(title_surface, (panel_x + 10, 20))
        
        # Draw location info
        location_text = f"Location: {self.location_name or 'Not specified'}"
        location_surface = self.small_font.render(location_text, True, (200, 200, 200))
        self.screen.blit(location_surface, (panel_x + 10, 60))
        
        # Draw grid info
        grid_text = f"Grid Size: {self.grid_size}x{self.grid_size}"
        grid_surface = self.small_font.render(grid_text, True, (200, 200, 200))
        self.screen.blit(grid_surface, (panel_x + 10, 85))
        
        # Draw map type
        map_type = "Real OpenStreetMap" if self.real_map_loaded else "Generated City"
        type_surface = self.small_font.render(f"Map Type: {map_type}", True, (200, 200, 200))
        self.screen.blit(type_surface, (panel_x + 10, 110))
        
        # Draw cell statistics
        if self.grid is not None:
            unique, counts = np.unique(self.grid, return_counts=True)
            y_pos = 150
            for val, count in zip(unique, counts):
                cell_name = self._get_cell_type_name(val)
                percentage = (count / (self.grid_size * self.grid_size)) * 100
                stat_text = f"{cell_name}: {count} ({percentage:.1f}%)"
                stat_surface = self.small_font.render(stat_text, True, (200, 200, 200))
                self.screen.blit(stat_surface, (panel_x + 10, y_pos))
                y_pos += 25
        
        # Draw disaster info
        if self.disaster_triggered:
            disaster_text = "🚨 DISASTER ACTIVE 🚨"
            disaster_surface = self.font.render(disaster_text, True, (255, 50, 50))
            self.screen.blit(disaster_surface, (panel_x + 10, self.grid_size * self.cell_size - 100))
    
    def _get_cell_type_name(self, cell_value):
        """Get human-readable name for cell type"""
        for name, value in self.cell_types.items():
            if value == cell_value:
                return name.replace('_', ' ').title()
        return f"Unknown ({cell_value})"
    
    def trigger_disaster(self):
        """Trigger disaster scenario on the real map"""
        if self.disaster_triggered:
            return
            
        self.disaster_triggered = True
        
        if self.real_map_loaded and self.map_data:
            # Use real map data for disaster generation
            self._trigger_real_map_disaster()
        else:
            # Use standard disaster generation
            print("🚨 Triggering disaster on generated grid...")
            self._trigger_real_map_disaster()
    
    def _trigger_real_map_disaster(self):
        """Trigger disaster on real map data"""
        print("🚨 Triggering disaster on real map...")
        
        # 1. FIND all valid locations FIRST
        building_locations = np.argwhere(self.grid == self.cell_types['BUILDING'])
        road_locations = np.argwhere(self.grid == self.cell_types['ROAD'])

        # 2. GENERATE civilians and place them in buildings
        #    (Do this *before* you collapse them)
        print(f"Generating civilians... Found {len(building_locations)} building locations.")
        num_civilians = self.config['environment'].get('num_civilians', 0)
        
        if len(building_locations) > 0:
            for _ in range(num_civilians):
                # Pick a random valid building
                idx = np.random.randint(len(building_locations))
                y, x = building_locations[idx]
                
                # Add the civilian to the environment's list
                self.civilians.append({
                    'position': np.array([x, y]),
                    'rescued': False,
                    'injury_level': np.random.choice(['minor', 'major', 'critical'], p=[0.6, 0.3, 0.1])
                })
            print(f"✅ Added {num_civilians} civilians to the map.")
        else:
            print("⚠️ No building locations found to place civilians!")

        # 3. NOW, simulate building collapses
        if len(building_locations) > 0:
            num_collapses = min(10, len(building_locations) // 10)
            collapse_indices = np.random.choice(len(building_locations), num_collapses, replace=False)
            
            for idx in collapse_indices:
                y, x = building_locations[idx]
                self.collapsed_buildings.append((x, y))
                if 'COLLAPSED' in self.cell_types:
                    self.grid[y, x] = self.cell_types['COLLAPSED']
        
        # 4. NOW, block some roads
        if len(road_locations) > 0:
            num_blocked = min(5, len(road_locations) // 20)
            block_indices = np.random.choice(len(road_locations), num_blocked, replace=False)
            
            for idx in block_indices:
                y, x = road_locations[idx]
                self.blocked_roads.append((x, y))
                if 'BLOCKED' in self.cell_types:
                    self.grid[y, x] = self.cell_types['BLOCKED']
            
    def get_map_info(self):
        """Get information about the current map"""
        return {
            'real_map_loaded': self.real_map_loaded,
            'location': self.location_name,
            'grid_size': self.grid_size,
            'total_cells': self.grid_size * self.grid_size,
            'agents_count': len(self.agents),
            'civilians_count': len(self.civilians),
            'step_count': self.step_count
        }
    
    def __str__(self):
        map_type = "Real Map" if self.real_map_loaded else "Generated Map"
        return f"RealMapEnv({map_type}, {self.location_name}, agents={len(self.agents)}, steps={self.step_count})"

# Test function
def test_real_map_env():
    """Test the RealMapEnv class"""
    print("🧪 Testing RealMapEnv...")
    
    try:
        # Test with real map
        env = RealMapEnv("Lahore, Pakistan")
        obs, info = env.reset()
        print(f"✅ RealMapEnv created: {info}")
        
        # Test disaster triggering
        env.trigger_disaster()
        print("✅ Disaster triggered")
        
        # Test rendering
        env.render()
        print("✅ Rendering successful")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ RealMapEnv test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_real_map_env()