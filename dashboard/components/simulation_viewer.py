import streamlit as st
import pygame
import numpy as np
from typing import Optional
import io
from PIL import Image
import plotly.graph_objects as go  # ← ADD THIS IMPORT

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
    
    def render(self,image_placeholder):
        """Render the simulation viewer"""
        if self.current_frame:
            # Display the current frame
            image_placeholder.image(self.current_frame, width='stretch', 
                    caption="Live Simulation View")
        else:
            # Placeholder when no frame available
            image_placeholder.info("🎮 Simulation visualization will appear here when running")
            
            # Show sample grid
            self.render_sample_grid(image_placeholder)
    
    def render_sample_grid(self,image_placeholder):
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
        
        image_placeholder.image(img, width='stretch', caption="Sample Environment Layout")

    def create_simulation_plot(self, environment):
        """Create a Plotly visualization of the simulation - FIXED COLORSCALE"""
        try:
            # Get grid data from environment
            if hasattr(environment, 'grid'):
                grid_data = environment.grid
            else:
                # Create a simple grid if no environment grid exists
                grid_size = getattr(environment, 'grid_size', 15)
                grid_data = np.zeros((grid_size, grid_size))
                
                # Mark agent positions
                for agent_id, agent in environment.agents.items():
                    pos = getattr(agent, 'position', None)
                    if pos is not None and len(pos) == 2:
                        x, y = int(pos[0]), int(pos[1])
                        if 0 <= x < grid_size and 0 <= y < grid_size:
                            if 'drone' in agent_id:
                                grid_data[x, y] = 1  # Drones
                            elif 'ambulance' in agent_id:
                                grid_data[x, y] = 2  # Ambulances
                            elif 'rescue' in agent_id:
                                grid_data[x, y] = 3  # Rescue teams
            
            # Create heatmap with CORRECT colorscale format
            fig = go.Figure(data=go.Heatmap(
                z=grid_data,
                colorscale=[
                    [0.0, 'lightgray'],    # Empty - 0.0 instead of 0
                    [0.25, 'blue'],        # Drones - 0.25 instead of 1
                    [0.5, 'green'],        # Ambulances - 0.5 instead of 2  
                    [0.75, 'orange'],      # Rescue teams - 0.75 instead of 3
                    [1.0, 'red']           # Other - 1.0 instead of 4
                ],
                showscale=True,
                hoverinfo='z'
            ))
            
            fig.update_layout(
                title="Simulation Grid View",
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                width=600,
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating simulation plot: {e}")
            return None