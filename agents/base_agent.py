import numpy as np

class BaseAgent:
    """
    Base class for all agents in the disaster response simulation
    """
    
    def __init__(self, agent_id, position, agent_type, config):
        self.agent_id = agent_id
        self.position = np.array(position, dtype=int)
        self.agent_type = agent_type
        self.config = config
        
        # Load agent-specific properties from config
        agent_config = config['agents'][agent_type]
        self.speed = agent_config['speed']
        self.capacity = agent_config['capacity']
        self.color = tuple(agent_config['color'])
        
        # State tracking
        self.civilians_rescued = 0
        self.steps_taken = 0
        self.total_reward = 0
        
        # Movement directions
        self.directions = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0),
            'STAY': (0, 0),
            'REST': (0, 0)
        }
    
    def move(self, direction, grid):
        """
        Move the agent in the specified direction.
        Can accept an integer (0-5) or a string ("UP", "DOWN", etc.).
        Returns: Boolean indicating if move was successful
        """
        
        # --- START OF FIX ---
        # Decode the action if it's an integer
        if isinstance(direction, int):
            # This list must match the one in simple_grid_env.py
            actions_list = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'REST']
            direction_str = actions_list[direction] if 0 <= direction < len(actions_list) else 'STAY'
        else:
            direction_str = direction
        # --- END OF FIX ---

        # Now, use the guaranteed string 'direction_str' for all checks
        if direction_str not in self.directions:
            # This is the line that's printing your error
            print(f"⚠️  Invalid direction: {direction_str}")
            return False
            
        new_position = self._calculate_new_position(direction_str)
        
        if self._is_valid_move(new_position, grid):
            self.position = new_position
            self.steps_taken += 1
            return True
        return False
    
    def _calculate_new_position(self, direction):
        """Calculate new position based on direction and speed"""
        direction_vector = np.array(self.directions[direction])
        return self.position + direction_vector * self.speed
    
    def _is_valid_move(self, position, grid):
        """
        Check if the move is valid
        - Within grid bounds
        - Not moving into blocked cells
        """
        x, y = position
        grid_size = grid.shape[0]
        
        # Check bounds
        if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
            return False
            
        # Check if cell is blocked
        blocked_type = self.config['environment']['cell_types']['BLOCKED']
        if grid[y, x] == blocked_type:
            return False
            
        return True
    
    def rescue_civilian(self):
        """
        Rescue a civilian if capacity allows
        Returns: Boolean indicating if rescue was successful
        """
        if self.civilians_rescued < self.capacity:
            self.civilians_rescued += 1
            return True
        return False
    
    def get_state(self):
        """Return current state of the agent"""
        return {
            'agent_id': self.agent_id,
            'type': self.agent_type,
            'position': self.position.tolist(),
            'civilians_rescued': self.civilians_rescued,
            'steps_taken': self.steps_taken,
            'capacity': f"{self.civilians_rescued}/{self.capacity}",
            'color': self.color
        }
    
    def update_reward(self, reward):
        """Update the agent's total reward"""
        self.total_reward += reward
    
    def __str__(self):
        return f"{self.agent_id}({self.agent_type}) at {self.position.tolist()}"
    
    def __repr__(self):
        return self.__str__()