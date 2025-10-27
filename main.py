#!/usr/bin/env python3
"""
Main entry point for Disaster Response AI Simulation - Week 1
"""

import pygame
import random
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.simple_grid_env import SimpleGridEnv
from agents.drone_agent import DroneAgent
from agents.ambulance_agent import AmbulanceAgent
from agents.rescue_team_agent import RescueTeamAgent
from agents.policies.random_policy import RandomPolicy

def main():
    """Main function to run the Week 1 demonstration"""
    print("🚀 Starting Disaster Response AI Simulation - Week 1")
    print("=" * 60)
    
    # Create environment
    env = SimpleGridEnv("config.yaml")
    
    # Add agents
    drone1 = DroneAgent("drone_1", [2, 2], env.config)
    drone2 = DroneAgent("drone_2", [12, 12], env.config)
    ambulance1 = AmbulanceAgent("ambulance_1", [7, 7], env.config)
    rescue_team1 = RescueTeamAgent("rescue_team_1", [4, 10], env.config)
    
    env.add_agent(drone1)
    env.add_agent(drone2)
    env.add_agent(ambulance1)
    env.add_agent(rescue_team1)
    
    # Trigger disaster
    print("🔥 Triggering disaster scenario...")
    env.trigger_disaster()
    
    # Create random policy for demo
    policy = RandomPolicy()
    
    print("✅ Simulation initialized!")
    print(f"📊 Grid size: {env.grid_size}x{env.grid_size}")
    print(f"🤖 Agents: {len(env.agents)}")
    print(f"👥 Civilians to rescue: {len(env.civilians)}")
    print(f"🏥 Hospitals: 4")
    print(f"🔥 Collapsed buildings: {len(env.collapsed_buildings)}")
    print(f"🚧 Blocked roads: {len(env.blocked_roads)}")
    print("\n🎮 Starting simulation with random policy...")
    print("⏹️  Close the window to exit")
    print("=" * 60)
    
    # Main simulation loop
    running = True
    clock = pygame.time.Clock()
    demo_steps = 200  # Limit for demo
    
    while running and env.step_count < demo_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get actions from random policy
        actions = policy.get_actions(list(env.agents.keys()), env.get_observation())
        
        # Step the environment
        observation, rewards, done, info = env.step(actions)
        
        # Render
        env.render()
        clock.tick(5)  # 5 FPS for visibility
        
        # Print progress every 20 steps
        if env.step_count % 20 == 0:
            civs_rescued = info['civilians_rescued']
            total_civs = info['total_civilians']
            print(f"📈 Step {env.step_count}: {civs_rescued}/{total_civs} civilians rescued")
        
        if done:
            print("🏁 Simulation completed!")
            running = False
    
    # Final statistics
    print("\n" + "=" * 60)
    print("📊 FINAL STATISTICS:")
    print("=" * 60)
    
    civs_rescued = sum(1 for c in env.civilians if c['rescued'])
    total_civs = len(env.civilians)
    
    print(f"🎯 Steps taken: {env.step_count}")
    print(f"👥 Civilians rescued: {civs_rescued}/{total_civs}")
    print(f"📈 Rescue rate: {civs_rescued/max(total_civs,1)*100:.1f}%")
    
    print("\n🤖 AGENT PERFORMANCE:")
    for agent_id, agent in env.agents.items():
        state = agent.get_state()
        print(f"  {agent_id}:")
        print(f"    🎯 Civilians rescued: {state['civilians_rescued']}")
        print(f"    👣 Steps taken: {state['steps_taken']}")
        print(f"    📍 Final position: {state['position']}")
    
    env.close()
    print("\n🎉 Week 1 demonstration completed successfully!")
    print("🚀 Ready for Week 2: MARL Integration!")

if __name__ == "__main__":
    main()