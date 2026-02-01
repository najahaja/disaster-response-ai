"""
Communication System for Multi-Agent Coordination
Enables agents to share information about civilian locations, rescue requests, etc.
"""

class CommunicationHub:
    """
    Central communication hub for agent coordination
    Allows agents to share information and coordinate rescue operations
    """
    
    def __init__(self):
        # Shared knowledge base
        self.discovered_civilians = {}  # {civilian_id: {position, discovered_by, status}}
        self.rescue_requests = []  # List of rescue requests
        self.agent_messages = []  # Message queue for agents
        
    def report_civilian_location(self, civilian_id, position, discovered_by_agent):
        """
        Agent reports a discovered civilian location
        
        Args:
            civilian_id: Unique identifier for the civilian
            position: [x, y] position of the civilian
            discovered_by_agent: ID of the agent who discovered the civilian
        """
        if civilian_id not in self.discovered_civilians:
            self.discovered_civilians[civilian_id] = {
                'position': position,
                'discovered_by': discovered_by_agent,
                'status': 'discovered',
                'assigned_to': None,
                'rescue_in_progress': False
            }
            
            # Create a rescue request
            self.rescue_requests.append({
                'civilian_id': civilian_id,
                'position': position,
                'priority': 'high',
                'requested_by': discovered_by_agent,
                'assigned_to': None
            })
            
            return True
        return False
    
    def get_nearest_unassigned_civilian(self, agent_position, agent_type=None):
        """
        Find the nearest civilian that hasn't been assigned to an agent
        
        Args:
            agent_position: [x, y] position of the requesting agent
            agent_type: Type of agent (drone, ambulance, rescue_team)
        
        Returns:
            Dictionary with civilian info or None
        """
        import numpy as np
        
        nearest_civilian = None
        min_distance = float('inf')
        
        for civ_id, civ_info in self.discovered_civilians.items():
            if civ_info['assigned_to'] is None and civ_info['status'] == 'discovered':
                distance = np.linalg.norm(
                    np.array(agent_position) - np.array(civ_info['position'])
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_civilian = {
                        'civilian_id': civ_id,
                        'position': civ_info['position'],
                        'distance': distance
                    }
        
        return nearest_civilian
    
    def assign_civilian_to_agent(self, civilian_id, agent_id):
        """
        Assign a civilian rescue to a specific agent
        
        Args:
            civilian_id: ID of the civilian
            agent_id: ID of the agent taking the assignment
        """
        if civilian_id in self.discovered_civilians:
            self.discovered_civilians[civilian_id]['assigned_to'] = agent_id
            self.discovered_civilians[civilian_id]['rescue_in_progress'] = True
            
            # Update rescue request
            for request in self.rescue_requests:
                if request['civilian_id'] == civilian_id:
                    request['assigned_to'] = agent_id
                    break
            
            return True
        return False
    
    def mark_civilian_rescued(self, civilian_id, rescued_by_agent):
        """
        Mark a civilian as successfully rescued
        
        Args:
            civilian_id: ID of the civilian
            rescued_by_agent: ID of the agent who rescued them
        """
        if civilian_id in self.discovered_civilians:
            self.discovered_civilians[civilian_id]['status'] = 'rescued'
            self.discovered_civilians[civilian_id]['rescued_by'] = rescued_by_agent
            
            # Remove from rescue requests
            self.rescue_requests = [
                req for req in self.rescue_requests 
                if req['civilian_id'] != civilian_id
            ]
            
            return True
        return False
    
    def get_all_discovered_civilians(self):
        """Get all discovered civilian locations"""
        return self.discovered_civilians
    
    def get_pending_rescue_requests(self):
        """Get all pending rescue requests"""
        return [req for req in self.rescue_requests if req['assigned_to'] is None]
    
    def broadcast_message(self, sender_id, message_type, content):
        """
        Broadcast a message to all agents
        
        Args:
            sender_id: ID of the sending agent
            message_type: Type of message (e.g., 'civilian_found', 'need_assistance')
            content: Message content (dictionary)
        """
        self.agent_messages.append({
            'sender': sender_id,
            'type': message_type,
            'content': content,
            'timestamp': len(self.agent_messages)
        })
    
    def get_messages_for_agent(self, agent_id, message_type=None):
        """
        Get messages relevant to a specific agent
        
        Args:
            agent_id: ID of the agent
            message_type: Optional filter by message type
        
        Returns:
            List of messages
        """
        messages = [
            msg for msg in self.agent_messages 
            if msg['sender'] != agent_id  # Don't return own messages
        ]
        
        if message_type:
            messages = [msg for msg in messages if msg['type'] == message_type]
        
        return messages
    
    def clear_old_messages(self, keep_last_n=100):
        """Clear old messages to prevent memory buildup"""
        if len(self.agent_messages) > keep_last_n:
            self.agent_messages = self.agent_messages[-keep_last_n:]
    
    def get_coordination_stats(self):
        """Get statistics about coordination"""
        total_discovered = len(self.discovered_civilians)
        rescued = sum(1 for civ in self.discovered_civilians.values() if civ['status'] == 'rescued')
        in_progress = sum(1 for civ in self.discovered_civilians.values() if civ['rescue_in_progress'])
        unassigned = sum(1 for civ in self.discovered_civilians.values() if civ['assigned_to'] is None)
        
        return {
            'total_discovered': total_discovered,
            'rescued': rescued,
            'in_progress': in_progress,
            'unassigned': unassigned,
            'pending_requests': len(self.rescue_requests)
        }
