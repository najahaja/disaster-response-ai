from typing import Dict, List, Any
import numpy as np

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
                distance = np.linalg.norm(agent.position - from_agent.position)
                if distance <= self.communication_range:
                    self.send_message(from_agent, agent_id, message_type, data)
                    recipients += 1
        
        print(f"📢 {from_agent.agent_id} broadcast to {recipients} agents: {message_type}")
    
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
    
    def set_communication_system(self, comm_system):
        """Set the communication system"""
        self.communication_system = comm_system
    
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
            print(f"🆘 {self.agent_id} received civilian location at {data['position']}")
            # Could set navigation target here
        
        elif message_type == 'ROAD_BLOCKED':
            print(f"🚧 {self.agent_id} received road blockage at {data['position']}")
            # Could update internal map here
        
        elif message_type == 'RESOURCE_REQUEST':
            print(f"🔄 {self.agent_id} received resource request from {message['from']}")
            # Could coordinate resource sharing here
    
    def send_civilian_location(self, position, environment):
        """Broadcast civilian location"""
        if self.communication_system:
            self.communication_system.broadcast_message(
                self, 'CIVILIAN_LOCATION', {'position': position}, environment
            )
    
    def send_road_blockage(self, position, environment):
        """Broadcast road blockage"""
        if self.communication_system:
            self.communication_system.broadcast_message(
                self, 'ROAD_BLOCKED', {'position': position}, environment
            )