import osmnx as ox
import networkx as nx
import numpy as np
from typing import Dict, Tuple, List
import pickle
import os

class MapLoader:
    """
    Utility class for loading and processing real maps from OpenStreetMap
    """
    
    def __init__(self, cache_dir="./data/maps/osm_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def graph_to_grid(self, graph, grid_size=20):
        """
        Convert networkx graph to grid representation
        """
        if not graph or len(graph.nodes) == 0:
            return None, {}, {}
        
        # Get node positions and normalize to grid
        node_positions = {}
        all_x = [data['x'] for node, data in graph.nodes(data=True)]
        all_y = [data['y'] for node, data in graph.nodes(data=True)]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Normalize positions to grid coordinates
        for node, data in graph.nodes(data=True):
            x_norm = int((data['x'] - min_x) / (max_x - min_x) * (grid_size - 1))
            y_norm = int((data['y'] - min_y) / (max_y - min_y) * (grid_size - 1))
            node_positions[node] = (x_norm, y_norm)
        
        # Create grid
        grid = np.full((grid_size, grid_size), 1)  # Default to buildings
        
        # Mark roads
        edge_data = {}
        for u, v, data in graph.edges(data=True):
            if u in node_positions and v in node_positions:
                u_pos = node_positions[u]
                v_pos = node_positions[v]
                
                # Mark both nodes as roads
                grid[u_pos[1], u_pos[0]] = 0  # ROAD
                grid[v_pos[1], v_pos[0]] = 0  # ROAD
                
                # Store edge information
                edge_data[(u, v)] = {
                    'positions': [u_pos, v_pos],
                    'length': data.get('length', 1),
                    'type': data.get('highway', 'road')
                }
        
        # Add hospitals at central nodes
        central_nodes = sorted(node_positions.items(), 
                             key=lambda x: abs(x[1][0] - grid_size//2) + abs(x[1][1] - grid_size//2))
        
        for i, (node_id, pos) in enumerate(central_nodes[:4]):  # 4 hospitals
            grid[pos[1], pos[0]] = 2  # HOSPITAL
        
        return grid, node_positions, edge_data
    
    def load_cached_map(self, location_name):
        """Load map from cache or download"""
        cache_file = os.path.join(self.cache_dir, f"{location_name.replace(' ', '_')}.pkl")
        
        if os.path.exists(cache_file):
            print(f"📁 Loading cached map: {location_name}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"🌐 Downloading map: {location_name}")
            try:
                graph = ox.graph_from_place(location_name, network_type='drive', simplify=True)
                # Cache the graph
                with open(cache_file, 'wb') as f:
                    pickle.dump(graph, f)
                return graph
            except Exception as e:
                print(f"❌ Failed to download map: {e}")
                return None
    
    def get_available_locations(self):
        """Get list of available cached locations"""
        locations = []
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                location_name = filename.replace('.pkl', '').replace('_', ' ')
                locations.append(location_name)
        return locations