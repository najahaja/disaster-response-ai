import numpy as np
import random

class DisasterGenerator:
    """
    Utility class for generating disaster scenarios
    """
    
    @staticmethod
    def collapse_buildings(grid, num_collapses, building_type, collapsed_type):
        """
        Randomly collapse buildings in the grid
        """
        building_positions = np.argwhere(grid == building_type)
        
        if len(building_positions) == 0:
            return grid, []
            
        num_collapses = min(num_collapses, len(building_positions))
        collapsed_positions = random.sample(list(building_positions), num_collapses)
        
        for pos in collapsed_positions:
            grid[tuple(pos)] = collapsed_type
            
        return grid, collapsed_positions
    
    @staticmethod
    def block_roads(grid, num_blocks, road_type, blocked_type):
        """
        Randomly block roads in the grid
        """
        road_positions = np.argwhere(grid == road_type)
        
        if len(road_positions) == 0:
            return grid, []
            
        num_blocks = min(num_blocks, len(road_positions))
        blocked_positions = random.sample(list(road_positions), num_blocks)
        
        for pos in blocked_positions:
            grid[tuple(pos)] = blocked_type
            
        return grid, blocked_positions
    
    @staticmethod
    def spawn_civilians(collapsed_positions, spawn_chance):
        """
        Spawn civilians in collapsed buildings
        """
        civilians = []
        for pos in collapsed_positions:
            if random.random() < spawn_chance:
                civilians.append({
                    'position': pos.tolist(),
                    'rescued': False
                })
        return civilians
    
    @staticmethod
    def generate_disaster(grid, config):
        """
        Generate a complete disaster scenario
        """
        building_type = config['environment']['cell_types']['BUILDING']
        collapsed_type = config['environment']['cell_types']['COLLAPSED']
        road_type = config['environment']['cell_types']['ROAD']
        blocked_type = config['environment']['cell_types']['BLOCKED']
        
        # Collapse buildings
        grid, collapsed_positions = DisasterGenerator.collapse_buildings(
            grid, 
            config['disaster']['collapsed_buildings'],
            building_type,
            collapsed_type
        )
        
        # Block roads
        grid, blocked_positions = DisasterGenerator.block_roads(
            grid,
            config['disaster']['blocked_roads'],
            road_type,
            blocked_type
        )
        
        # Spawn civilians
        civilians = DisasterGenerator.spawn_civilians(
            collapsed_positions,
            config['disaster']['civilian_spawn_chance']
        )
        
        return grid, civilians, collapsed_positions, blocked_positions