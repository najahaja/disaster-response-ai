import pygame
import numpy as np

class VisualizationUtils:
    """
    Utility class for visualization functions
    """
    
    @staticmethod
    def draw_grid(screen, grid, colors, cell_size):
        """
        Draw the grid environment
        """
        grid_size = grid.shape[0]
        for y in range(grid_size):
            for x in range(grid_size):
                cell_type = grid[y, x]
                color = colors.get(cell_type, (0, 0, 0))
                rect = pygame.Rect(x * cell_size, y * cell_size, 
                                 cell_size, cell_size)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (50, 50, 50), rect, 1)
    
    @staticmethod
    def draw_agents(screen, agents, cell_size, font):
        """
        Draw all agents on the grid with improved visibility
        """
        for agent in agents.values():
            x, y = agent.position
            center_x = x * cell_size + cell_size // 2
            center_y = y * cell_size + cell_size // 2
            
            # Draw agent circle (larger for better visibility)
            circle_radius = int(cell_size * 0.4)  # Increased from 1/3 to 2/5
            pygame.draw.circle(screen, agent.color, 
                             (center_x, center_y), circle_radius)
            
            # Draw agent type abbreviation with bold font
            agent_type_abbr = agent.agent_type[0].upper()
            # Make text bold and larger
            text = font.render(agent_type_abbr, True, (0, 0, 0))
            text_rect = text.get_rect(center=(center_x, center_y))
            screen.blit(text, text_rect)
    
    @staticmethod
    def draw_civilians(screen, civilians, cell_size):
        """
        Draw civilians on the grid
        """
        for civilian in civilians:
            if not civilian['rescued']:
                x, y = civilian['position']
                center_x = x * cell_size + cell_size // 2
                center_y = y * cell_size + cell_size // 2
                pygame.draw.circle(screen, (255, 255, 0), 
                                 (center_x, center_y), cell_size // 4)
    
    @staticmethod
    def draw_info_panel(screen, step_count, agents, civilians, font, width=250):
        """
        Draw clean information panel matching second image style
        """
        panel_x = screen.get_width() - width
        
        # Draw panel background (darker for better contrast)
        pygame.draw.rect(screen, (30, 30, 30), 
                        (panel_x, 0, width, screen.get_height()))
        
        y_pos = 20
        
        # Step count at the top
        step_text = f"Step: {step_count}"
        step_surface = font.render(step_text, True, (255, 255, 255))
        screen.blit(step_surface, (panel_x + 10, y_pos))
        y_pos += 40
        
        # Agents section header
        agents_header = "AGENTS:"
        agents_surface = font.render(agents_header, True, (255, 255, 255))
        screen.blit(agents_surface, (panel_x + 10, y_pos))
        y_pos += 30
        
        # List each agent with their rescue count
        for agent_id, agent in agents.items():
            agent_info = f"{agent_id}: {getattr(agent, 'civilians_rescued', 0)}"
            agent_surface = font.render(agent_info, True, (180, 180, 180))
            screen.blit(agent_surface, (panel_x + 10, y_pos))
            y_pos += 22
        
        y_pos += 15
        
        # Civilians section
        civs_rescued = sum(1 for c in civilians if c.get('rescued', False))
        civs_total = len(civilians)
        civs_text = f"CIVILIANS: {civs_rescued}/{civs_total} rescued"
        civs_surface = font.render(civs_text, True, (255, 255, 255))
        screen.blit(civs_surface, (panel_x + 10, y_pos))