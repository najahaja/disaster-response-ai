import streamlit as st
import pygame
import numpy as np
from typing import Optional
import io
from PIL import Image

class SimulationViewer:
    """
    Component for displaying the simulation visualization
    """
    
    def __init__(self):
        self.current_frame = None
    
    def update_frame(self, pygame_surface):
        """Update the current frame from PyGame surface"""
        try:
            # Convert PyGame surface to image
            frame_string = pygame.image.tostring(pygame_surface, 'RGB')
            frame_image = Image.frombytes('RGB', pygame_surface.get_size(), frame_string)
            self.current_frame = frame_image
        except Exception as e:
            st.error(f"Error updating frame: {e}")
            self.current_frame = None
    
    def render(self):
        """Render the simulation viewer"""
        if self.current_frame:
            # Display the current frame
            st.image(self.current_frame, width='stretch', 
                    caption="Live Simulation View")
        else:
            # Placeholder when no frame available
            st.info("🎮 Simulation visualization will appear here when running")
            
            # Show sample grid
            self.render_sample_grid()
    
    def render_sample_grid(self):
        """Render a sample grid for demonstration"""
        # Create a sample grid visualization
        grid_size = 15
        cell_size = 20
        
        # Create sample data
        sample_grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        # Add roads
        sample_grid[7, :] = [200, 200, 200]  # Horizontal road
        sample_grid[:, 7] = [200, 200, 200]  # Vertical road
        
        # Add buildings
        for i in range(grid_size):
            for j in range(grid_size):
                if sample_grid[i, j, 0] == 0:  # Not a road
                    sample_grid[i, j] = [139, 69, 19]  # Brown buildings
        
        # Add hospitals
        sample_grid[0, 0] = [255, 255, 255]  # White hospitals
        sample_grid[0, -1] = [255, 255, 255]
        sample_grid[-1, 0] = [255, 255, 255]
        sample_grid[-1, -1] = [255, 255, 255]
        
        # Add some collapsed buildings
        sample_grid[3, 3] = [105, 105, 105]
        sample_grid[10, 5] = [105, 105, 105]
        sample_grid[7, 12] = [105, 105, 105]
        
        # Create image
        img = Image.fromarray(sample_grid, 'RGB')
        img = img.resize((grid_size * cell_size, grid_size * cell_size), Image.Resampling.NEAREST)
        
        st.image(img,width='stretch', caption="Sample Environment Layout")