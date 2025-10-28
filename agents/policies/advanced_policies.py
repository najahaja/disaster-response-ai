import numpy as np
from typing import Dict, List, Any
import heapq
from collections import deque

class AStarPolicy:
    """
    A* pathfinding policy for intelligent navigation
    """
    
    def __init__(self):
        self.open_set = set()
        self.closed_set = set()
        self.g_score = {}
        self.f_score = {}
        self.came_from = {}
    
    def find_path(self, start, goal, grid):
        """Find path from start to goal using A* algorithm"""
        if not self.is_valid_position(start, grid) or not self.is_valid_position(goal, grid):
            return []
            
        self.open_set = {tuple(start)}
        self.closed_set = set()
        self.g_score = {tuple(start): 0}
        self.f_score = {tuple(start): self.heuristic(start, goal)}
        self.came_from = {}
        
        while self.open_set:
            current = min(self.open_set, key=lambda x: self.f_score.get(x, float('inf')))
            
            if current == tuple(goal):
                return self.reconstruct_path(current)
            
            self.open_set.remove(current)
            self.closed_set.add(current)
            
            for neighbor in self.get_neighbors(current, grid):
                if neighbor in self.closed_set:
                    continue
                
                tentative_g_score = self.g_score[current] + self.get_move_cost(current, neighbor, grid)
                
                if neighbor not in self.open_set:
                    self.open_set.add(neighbor)
                elif tentative_g_score >= self.g_score.get(neighbor, float('inf')):
                    continue
                
                self.came_from[neighbor] = current
                self.g_score[neighbor] = tentative_g_score
                self.f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
        
        return []  # No path found
    
    def heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, position, grid):
        """Get valid neighboring positions"""
        x, y = position
        neighbors = []
        grid_size = grid.shape[0]
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-directional movement
            nx, ny = x + dx, y + dy
            if self.is_valid_position((nx, ny), grid):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def is_valid_position(self, position, grid):
        """Check if position is valid and not blocked"""
        x, y = position
        grid_size = grid.shape[0]
        
        if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
            return False
        
        # Check if cell is blocked (cell type 4 = BLOCKED)
        if grid[y, x] == 4:
            return False
            
        return True
    
    def get_move_cost(self, from_pos, to_pos, grid):
        """Get movement cost between positions"""
        # Different costs for different terrain types
        x, y = to_pos
        cell_type = grid[y, x]
        
        if cell_type == 0:  # ROAD
            return 0.8  # Faster on roads
        elif cell_type == 3:  # COLLAPSED
            return 2.0  # Slower in collapsed areas
        else:  # BUILDING, HOSPITAL, etc.
            return 1.0  # Normal speed
    
    def reconstruct_path(self, current):
        """Reconstruct path from goal to start"""
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start->goal

class CommunicationSystem:
    """
    Manages communication between agents
    """
    
    def __init__(self):
        self.messages = {}
        self.message_log = []
        self.communication_range = 5  # Cells
    
    def send_message(self, from_agent, to_agent_id, message_type, data):
        """Send message from one agent to another"""
        message = {
            'from': from_agent.agent_id,
            'to': to_agent_id,
            'type': message_type,
            'data': data,
            'step': getattr(from_agent, 'steps_taken', 0)
        }
        
        if to_agent_id not in self.messages:
            self.messages[to_agent_id] = []
        
        self.messages[to_agent_id].append(message)
        self.message_log.append(message)
        
        print(f"📡 {from_agent.agent_id} -> {to_agent_id}: {message_type}")
    
    def broadcast_message(self, from_agent, message_type, data, environment):
        """Broadcast message to all agents within range"""
        recipients = 0
        for agent_id, agent in environment.agents.items():
            if agent_id != from_agent.agent_id:
                distance = np.linalg.norm(np.array(agent.position) - np.array(from_agent.position))
                if distance <= self.communication_range:
                    self.send_message(from_agent, agent_id, message_type, data)
                    recipients += 1
        
        print(f"📢 {from_agent.agent_id} broadcast to {recipients} agents: {message_type}")
        return recipients
    
    def get_messages(self, agent_id):
        """Get messages for an agent"""
        messages = self.messages.get(agent_id, [])
        # Clear messages after reading
        if agent_id in self.messages:
            self.messages[agent_id] = []
        return messages
    
    def get_communication_stats(self):
        """Get communication statistics"""
        return {
            'total_messages': len(self.message_log),
            'active_messages': sum(len(msgs) for msgs in self.messages.values()),
            'recent_messages': self.message_log[-10:] if self.message_log else []
        }

class CommunicativeAgentMixin:
    """
    Mixin to add communication capabilities to agents
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.communication_system = None
        self.received_messages = []
        self.known_civilians = []  # Shared knowledge
        self.known_blockages = []  # Shared knowledge
        self.agent_ref = None  # Reference to the actual agent object
    
    def set_communication_system(self, comm_system):
        """Set the communication system"""
        self.communication_system = comm_system
    
    def set_agent_reference(self, agent):
        """Set reference to the actual agent object"""
        self.agent_ref = agent
    
    def process_messages(self):
        """Process received messages"""
        if self.communication_system:
            messages = self.communication_system.get_messages(self.agent_id)
            self.received_messages.extend(messages)
            
            for message in messages:
                self._handle_message(message)
    
    def _handle_message(self, message):
        """Handle incoming message"""
        message_type = message['type']
        data = message['data']
        
        if message_type == 'CIVILIAN_LOCATION':
            position = data['position']
            if position not in self.known_civilians:
                self.known_civilians.append(position)
                print(f"🆘 {self.agent_id} learned about civilian at {position}")
        
        elif message_type == 'ROAD_BLOCKED':
            position = data['position']
            if position not in self.known_blockages:
                self.known_blockages.append(position)
                print(f"🚧 {self.agent_id} learned about road blockage at {position}")
        
        elif message_type == 'RESOURCE_REQUEST':
            print(f"🔄 {self.agent_id} received resource request from {message['from']}")
            # Could implement resource sharing logic here
        
        elif message_type == 'TARGET_ASSIGNMENT':
            print(f"🎯 {self.agent_id} assigned target: {data['target']}")
            # Could set navigation target here
    
    def send_civilian_location(self, position, environment):
        """Broadcast civilian location"""
        if self.communication_system and self.agent_ref:
            self.communication_system.broadcast_message(
                self.agent_ref, 'CIVILIAN_LOCATION', {'position': position}, environment
            )
    
    def send_road_blockage(self, position, environment):
        """Broadcast road blockage"""
        if self.communication_system and self.agent_ref:
            self.communication_system.broadcast_message(
                self.agent_ref, 'ROAD_BLOCKED', {'position': position}, environment
            )
    
    def send_resource_request(self, resource_type, environment):
        """Broadcast resource request"""
        if self.communication_system and self.agent_ref:
            self.communication_system.broadcast_message(
                self.agent_ref, 'RESOURCE_REQUEST', {'resource_type': resource_type}, environment
            )

class CooperativePolicy(CommunicativeAgentMixin):
    """
    Policy that enables cooperative behavior between agents
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.astar = AStarPolicy()
        self.assigned_targets = {}
        self.team_knowledge = {
            'civilian_locations': [],
            'blocked_roads': [],
            'hospital_locations': [],
            'explored_areas': set()
        }
        self.last_action = 'STAY'
        self.agent_id = None  # Will be set when used with an agent
    
    def set_agent_id(self, agent_id):
        """Set the agent ID for this policy"""
        self.agent_id = agent_id
    
    def get_action(self, agent, environment):
        """Get cooperative action for agent"""
        # Set agent ID and reference if not set
        if self.agent_id is None:
            self.agent_id = agent.agent_id
        if self.agent_ref is None:
            self.set_agent_reference(agent)
            
        self.process_messages()  # Process any received messages
        
        # Update team knowledge
        self.update_team_knowledge(agent, environment)
        
        # Choose action based on agent type and situation
        if agent.agent_type == 'drone':
            action = self.drone_policy(agent, environment)
        elif agent.agent_type == 'ambulance':
            action = self.ambulance_policy(agent, environment)
        elif agent.agent_type == 'rescue_team':
            action = self.rescue_team_policy(agent, environment)
        else:
            action = 'STAY'
        
        self.last_action = action
        return action
    
    def drone_policy(self, agent, environment):
        """Drone policy - scout and communicate"""
        # If civilian found nearby, communicate location
        civilians_nearby = self.get_civilians_nearby(agent, environment, radius=1)
        if civilians_nearby:
            civilian_pos = civilians_nearby[0]['position']
            if civilian_pos not in self.team_knowledge['civilian_locations']:
                self.team_knowledge['civilian_locations'].append(civilian_pos)
                self.send_civilian_location(civilian_pos, environment)
        
        # If know civilian locations, help guide others
        if self.team_knowledge['civilian_locations']:
            nearest_civilian = self.find_nearest_target(agent, self.team_knowledge['civilian_locations'])
            if nearest_civilian and np.linalg.norm(agent.position - np.array(nearest_civilian)) > 3:
                return self.move_toward(agent, nearest_civilian, environment.grid)
        
        # Otherwise, explore unexplored areas
        return self.explore_policy(agent, environment)
    
    def ambulance_policy(self, agent, environment):
        """Ambulance policy - transport civilians"""
        # If carrying civilians, go to hospital
        if hasattr(agent, 'civilians_on_board') and agent.civilians_on_board > 0:
            hospital_pos = self.find_nearest_hospital(agent, environment)
            if hospital_pos:
                return self.move_toward(agent, hospital_pos, environment.grid)
        
        # If know civilian locations, go to nearest one
        if self.team_knowledge['civilian_locations']:
            nearest_civilian = self.find_nearest_target(agent, self.team_knowledge['civilian_locations'])
            if nearest_civilian:
                return self.move_toward(agent, nearest_civilian, environment.grid)
        
        # Otherwise, explore or assist
        return self.explore_policy(agent, environment)
    
    def rescue_team_policy(self, agent, environment):
        """Rescue team policy - complex rescue operations"""
        # Similar to ambulance but with different priorities
        return self.ambulance_policy(agent, environment)
    
    def explore_policy(self, agent, environment):
        """Intelligent exploration policy"""
        # Mark current position as explored
        self.team_knowledge['explored_areas'].add(tuple(agent.position))
        
        # Prefer moving toward unexplored areas
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        unexplored_actions = []
        
        for action in possible_actions:
            new_pos = agent._calculate_new_position(action)
            if (agent._is_valid_move(new_pos, environment.grid) and
                tuple(new_pos) not in self.team_knowledge['explored_areas']):
                unexplored_actions.append(action)
        
        if unexplored_actions:
            return np.random.choice(unexplored_actions)
        
        # If all explored, move randomly but avoid recent actions
        valid_actions = [a for a in possible_actions if a == 'STAY' or 
                        agent._is_valid_move(agent._calculate_new_position(a), environment.grid)]
        
        # Avoid repeating the same action
        if len(valid_actions) > 1 and self.last_action in valid_actions:
            valid_actions.remove(self.last_action)
        
        return np.random.choice(valid_actions) if valid_actions else 'STAY'
    
    def move_toward(self, agent, target, grid):
        """Move toward target position using pathfinding"""
        path = self.astar.find_path(agent.position.tolist(), target, grid)
        
        if len(path) > 1:
            next_pos = path[1]
            current_pos = agent.position
            
            # Determine direction
            if next_pos[0] > current_pos[0]:
                return 'RIGHT'
            elif next_pos[0] < current_pos[0]:
                return 'LEFT'
            elif next_pos[1] > current_pos[1]:
                return 'DOWN'
            elif next_pos[1] < current_pos[1]:
                return 'UP'
        
        # Fallback to simple movement if pathfinding fails
        return self.simple_move_toward(agent, target, grid)
    
    def simple_move_toward(self, agent, target, grid):
        """Simple movement toward target (no pathfinding)"""
        current_pos = agent.position
        target_pos = np.array(target)
        
        # Calculate direction
        diff = target_pos - current_pos
        
        if abs(diff[0]) > abs(diff[1]):
            # Move horizontally
            if diff[0] > 0 and agent._is_valid_move(current_pos + [1, 0], grid):
                return 'RIGHT'
            elif diff[0] < 0 and agent._is_valid_move(current_pos + [-1, 0], grid):
                return 'LEFT'
        else:
            # Move vertically
            if diff[1] > 0 and agent._is_valid_move(current_pos + [0, 1], grid):
                return 'DOWN'
            elif diff[1] < 0 and agent._is_valid_move(current_pos + [0, -1], grid):
                return 'UP'
        
        # If direct movement not possible, try alternative directions
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        valid_actions = [a for a in possible_actions if 
                        agent._is_valid_move(agent._calculate_new_position(a), grid)]
        
        return np.random.choice(valid_actions) if valid_actions else 'STAY'
    
    def get_civilians_nearby(self, agent, environment, radius=2):
        """Get civilians within radius of agent"""
        nearby = []
        for civilian in environment.civilians:
            if not civilian['rescued']:
                dist = np.linalg.norm(np.array(agent.position) - np.array(civilian['position']))
                if dist <= radius:
                    nearby.append(civilian)
        return nearby
    
    def find_nearest_hospital(self, agent, environment):
        """Find nearest hospital"""
        hospital_positions = np.argwhere(environment.grid == environment.cell_types['HOSPITAL'])
        if len(hospital_positions) > 0:
            distances = [np.linalg.norm(agent.position - pos) for pos in hospital_positions]
            nearest_idx = np.argmin(distances)
            return hospital_positions[nearest_idx].tolist()
        return None
    
    def find_nearest_target(self, agent, targets):
        """Find nearest target from list"""
        if not targets:
            return None
        distances = [np.linalg.norm(agent.position - np.array(target)) for target in targets]
        nearest_idx = np.argmin(distances)
        return targets[nearest_idx]
    
    def update_team_knowledge(self, agent, environment):
        """Update shared team knowledge"""
        # Update hospital locations (once)
        if not self.team_knowledge['hospital_locations']:
            hospital_positions = np.argwhere(environment.grid == environment.cell_types['HOSPITAL'])
            self.team_knowledge['hospital_locations'] = [pos.tolist() for pos in hospital_positions]
        
        # Update known civilians from environment
        for civilian in environment.civilians:
            if not civilian['rescued']:
                civ_pos = civilian['position']
                if civ_pos not in self.team_knowledge['civilian_locations']:
                    self.team_knowledge['civilian_locations'].append(civ_pos)
        
        # Update blocked roads - but only send message if we have the agent reference
        blocked_positions = np.argwhere(environment.grid == environment.cell_types['BLOCKED'])
        for pos in blocked_positions:
            pos_list = pos.tolist()
            if pos_list not in self.team_knowledge['blocked_roads']:
                self.team_knowledge['blocked_roads'].append(pos_list)
                # Only send blockage message if we have communication system and agent reference
                if self.communication_system and self.agent_ref:
                    self.send_road_blockage(pos_list, environment)
    
    def get_policy_stats(self):
        """Get statistics about the policy's performance"""
        return {
            'agent_id': self.agent_id,
            'known_civilians': len(self.team_knowledge['civilian_locations']),
            'known_blockages': len(self.team_knowledge['blocked_roads']),
            'explored_areas': len(self.team_knowledge['explored_areas']),
            'messages_processed': len(self.received_messages),
            'assigned_targets': len(self.assigned_targets),
            'has_agent_ref': self.agent_ref is not None,
            'has_comm_system': self.communication_system is not None
        }

# Simple alternative policies for testing
class RandomPolicy:
    """Simple random movement policy for testing"""
    
    def get_action(self, agent, environment):
        """Get random action"""
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
        valid_actions = [a for a in possible_actions if a == 'STAY' or 
                        agent._is_valid_move(agent._calculate_new_position(a), environment.grid)]
        return np.random.choice(valid_actions) if valid_actions else 'STAY'

class ExplorePolicy:
    """Simple exploration policy"""
    
    def __init__(self):
        self.visited_positions = set()
    
    def get_action(self, agent, environment):
        """Get exploration action"""
        self.visited_positions.add(tuple(agent.position))
        
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        unexplored_actions = []
        
        for action in possible_actions:
            new_pos = agent._calculate_new_position(action)
            if (agent._is_valid_move(new_pos, environment.grid) and
                tuple(new_pos) not in self.visited_positions):
                unexplored_actions.append(action)
        
        if unexplored_actions:
            return np.random.choice(unexplored_actions)
        
        # If all explored, move randomly
        valid_actions = [a for a in possible_actions if 
                        agent._is_valid_move(agent._calculate_new_position(a), environment.grid)]
        return np.random.choice(valid_actions) if valid_actions else 'STAY'