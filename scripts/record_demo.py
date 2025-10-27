#!/usr/bin/env python3
"""
Utility script to record simulation demonstrations as video
"""

import pygame
import numpy as np
import sys
import os
from datetime import datetime
import imageio
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.real_map_env import RealMapEnv
from agents.drone_agent import DroneAgent
from agents.ambulance_agent import AmbulanceAgent
from agents.policies.advanced_policies import CooperativePolicy

class DemoRecorder:
    """
    Records simulation sessions as video files
    """
    
    def __init__(self, output_dir="./data/videos"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.frames = []
    
    def capture_frame(self, pygame_surface):
        """Capture a frame from PyGame surface"""
        try:
            # Convert PyGame surface to numpy array
            frame_string = pygame.image.tostring(pygame_surface, 'RGB')
            frame_array = np.frombuffer(frame_string, dtype=np.uint8)
            frame_array = frame_array.reshape((pygame_surface.get_height(), pygame_surface.get_width(), 3))
            self.frames.append(frame_array)
        except Exception as e:
            print(f"⚠️  Error capturing frame: {e}")
    
    def save_video(self, filename=None, fps=10):
        """Save captured frames as video"""
        if not self.frames:
            print("❌ No frames captured")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"disaster_response_demo_{timestamp}.mp4"
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            with imageio.get_writer(output_path, fps=fps) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
            
            print(f"✅ Video saved: {output_path}")
            print(f"📹 Frames: {len(self.frames)}, FPS: {fps}, Duration: {len(self.frames)/fps:.1f}s")
            
        except Exception as e:
            print(f"❌ Error saving video: {e}")
    
    def clear_frames(self):
        """Clear captured frames"""
        self.frames = []

def record_cooperative_demo():
    """Record a demonstration of cooperative agent behavior"""
    print("🎬 Recording Cooperative Agent Demo")
    print("===================================")
    
    # Initialize environment and recorder
    env = RealMapEnv("Lahore, Pakistan")
    recorder = DemoRecorder()
    
    # Add agents with cooperative policy
    agents = [
        DroneAgent("demo_drone_1", [5, 5], env.config),
        DroneAgent("demo_drone_2", [12, 12], env.config),
        AmbulanceAgent("demo_ambulance", [7, 7], env.config)
    ]
    
    # Set up cooperative policies
    comm_system = None
    for agent in agents:
        env.add_agent(agent)
    
    # Trigger disaster
    env.trigger_disaster()
    
    print(f"🤖 Agents: {len(env.agents)}")
    print(f"👥 Civilians: {len(env.civilians)}")
    print("🎥 Starting recording...")
    
    # Record simulation
    max_steps = 100
    for step in range(max_steps):
        # Capture frame
        env.render()
        recorder.capture_frame(env.screen)
        
        # Simple cooperative behavior
        actions = {}
        for agent_id, agent in env.agents.items():
            # Simple logic: move toward center or civilians
            if agent.agent_type == 'drone':
                # Drones explore
                action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])
            else:
                # Ambulances move toward civilians
                action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])
            
            actions[agent_id] = action
        
        # Step environment
        observation, rewards, done, info = env.step(actions)
        
        # Print progress
        if step % 20 == 0:
            rescued = sum(1 for c in env.civilians if c['rescued'])
            print(f"📊 Step {step}: {rescued}/{len(env.civilians)} civilians rescued")
        
        if done:
            break
    
    # Save video
    env.close()
    recorder.save_video("cooperative_demo.mp4")
    
    print("✅ Demo recording completed!")

def record_real_map_demo():
    """Record a demonstration with real map data"""
    print("🎬 Recording Real Map Demo")
    print("==========================")
    
    env = RealMapEnv("Lahore, Pakistan")
    recorder = DemoRecorder()
    
    # Add agents
    agents = [
        DroneAgent("map_drone_1", [5, 5], env.config),
        AmbulanceAgent("map_ambulance", [10, 10], env.config)
    ]
    
    for agent in agents:
        env.add_agent(agent)
    
    # Trigger disaster
    env.trigger_disaster()
    
    # Get map info
    map_info = env.get_map_info()
    print(f"🗺️  Real Map: {map_info['location']}")
    print(f"📊 Grid: {map_info['grid_size']}x{map_info['grid_size']}")
    
    # Record simulation
    for step in range(80):
        env.render()
        recorder.capture_frame(env.screen)
        
        # Random actions for demo
        actions = {}
        for agent_id in env.agents:
            action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])
            actions[agent_id] = action
        
        env.step(actions)
        
        if step % 15 == 0:
            print(f"🎥 Recorded frame {step}")
    
    env.close()
    recorder.save_video("real_map_demo.mp4")
    print("✅ Real map demo recorded!")

def main():
    """Main function for recording demos"""
    print("🚀 Disaster Response AI - Demo Recorder")
    print("=======================================")
    
    print("Available demos:")
    print("1. Cooperative Agent Demo")
    print("2. Real Map Demo")
    print("3. Custom Recording")
    
    choice = input("🎬 Choose demo to record (1-3): ").strip()
    
    if choice == "1":
        record_cooperative_demo()
    elif choice == "2":
        record_real_map_demo()
    elif choice == "3":
        print("🔧 Custom recording not implemented yet")
    else:
        print("❌ Invalid choice")
    
    print("\n🎉 Demo recording session completed!")

if __name__ == "__main__":
    main()