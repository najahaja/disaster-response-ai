import random
from typing import Dict, List, Any
import numpy as np

class DynamicEventManager:
    """
    Manages dynamic events during simulation
    """
    
    def __init__(self, config):
        self.config = config
        self.event_log = []
        self.event_probabilities = {
            'aftershock': 0.02,      # 2% chance per step after step 50
            'resource_depletion': 0.01,  # 1% chance per step
            'road_closure': 0.015,   # 1.5% chance per step
            'civilian_discovery': 0.03  # 3% chance per step
        }
    
    def check_events(self, step_count, environment):
        """
        Check for and generate dynamic events
        """
        events = []
        
        # Aftershocks (only after initial disaster)
        if step_count > 50 and random.random() < self.event_probabilities['aftershock']:
            events.append({
                'type': 'aftershock',
                'intensity': random.uniform(0.5, 1.5),
                'step': step_count,
                'description': 'Aftershock causes additional collapses'
            })
        
        # Resource depletion
        if random.random() < self.event_probabilities['resource_depletion']:
            agent_types = ['drone', 'ambulance', 'rescue_team']
            events.append({
                'type': 'resource_depletion',
                'agent_type': random.choice(agent_types),
                'step': step_count,
                'description': f'Resource depletion affects {random.choice(agent_types)}'
            })
        
        # Road closures
        if random.random() < self.event_probabilities['road_closure']:
            # Find some road positions to close
            road_positions = np.argwhere(environment.grid == environment.cell_types['ROAD'])
            if len(road_positions) > 0:
                num_closures = min(3, len(road_positions))
                closure_locations = random.sample(list(road_positions), num_closures)
                events.append({
                    'type': 'road_closure',
                    'locations': [pos.tolist() for pos in closure_locations],
                    'step': step_count,
                    'description': f'Road closures block {num_closures} routes'
                })
        
        # Civilian discoveries
        if (random.random() < self.event_probabilities['civilian_discovery'] and 
            len(environment.civilians) < 20):  # Limit total civilians
            # Find collapsed buildings without civilians
            collapsed_positions = np.argwhere(environment.grid == environment.cell_types['COLLAPSED'])
            empty_collapsed = [pos for pos in collapsed_positions 
                             if not any(tuple(c['position']) == tuple(pos) for c in environment.civilians)]
            
            if empty_collapsed:
                discovery_pos = random.choice(empty_collapsed)
                environment.add_civilian(discovery_pos.tolist())
                events.append({
                    'type': 'civilian_discovery',
                    'position': discovery_pos.tolist(),
                    'step': step_count,
                    'description': 'New civilian discovered in rubble'
                })
        
        # Log events
        self.event_log.extend(events)
        
        return events
    
    def get_event_summary(self):
        """Get summary of all events"""
        event_counts = {}
        for event in self.event_log:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            'total_events': len(self.event_log),
            'event_counts': event_counts,
            'recent_events': self.event_log[-5:] if self.event_log else []
        }