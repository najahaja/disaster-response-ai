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
        Draw all agents on the grid
        """
        for agent in agents.values():
            x, y = agent.position
            center_x = x * cell_size + cell_size // 2
            center_y = y * cell_size + cell_size // 2
            
            # Draw agent circle
            pygame.draw.circle(screen, agent.color, 
                             (center_x, center_y), cell_size // 3)
            
            # Draw agent type abbreviation
            agent_type_abbr = agent.agent_type[0].upper()
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
    def draw_info_panel(screen, step_count, agents, civilians, font, width=200):
        """
        Draw information panel on the right side
        """
        panel_x = screen.get_width() - width
        pygame.draw.rect(screen, (40, 40, 40), 
                        (panel_x, 0, width, screen.get_height()))
        
        y_offset = 10
        # Step count
        step_text = font.render(f"Step: {step_count}", True, (255, 255, 255))
        screen.blit(step_text, (panel_x + 10, y_offset))
        y_offset += 30
        
        # Agent info
        agents_text = font.render("AGENTS:", True, (255, 255, 255))
        screen.blit(agents_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        for agent_id, agent in agents.items():
            agent_info = f"{agent_id}: {agent.civilians_rescued} rescued"
            text = font.render(agent_info, True, (200, 200, 200))
            screen.blit(text, (panel_x + 20, y_offset))
            y_offset += 20
        
        y_offset += 10
        # Civilian info
        civs_rescued = sum(1 for c in civilians if c['rescued'])
        civs_total = len(civilians)
        civ_text = font.render(f"CIVILIANS: {civs_rescued}/{civs_total} rescued", 
                             True, (255, 255, 255))
        screen.blit(civ_text, (panel_x + 10, y_offset))