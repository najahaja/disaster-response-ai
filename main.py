#!/usr/bin/env python3
"""
Main entry point for Disaster Response AI Simulation - Week 1
(Fixed for Gymnasium API)
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
    
    try:
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
            
            # --- THIS IS THE FIX ---
            # Step the environment
            # Old/broken line: observation, reward, done, info, extra_value = env.step(actions)
            # This was assigning the 'truncated' boolean to 'info'
            
            # Correct Gymnasium unpacking:
            observation, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated  # Combine termination flags
            # --- END OF FIX ---
            
            # Render
            env.render()
            clock.tick(5)  # 5 FPS for visibility
            
            # Print progress every 20 steps
            if env.step_count % 20 == 0:
                # Use .get() for safe dictionary access, now that 'info' is correct
                civs_rescued = info.get('civilians_rescued', 0)
                total_civs = info.get('total_civilians', len(env.civilians))
                print(f"📈 Step {env.step_count}: {civs_rescued}/{total_civs} civilians rescued")
            
            if done:
                print("🏁 Simulation completed!")
                running = False
        
        # Final statistics
        print("\n" + "=" * 60)
        print("📊 FINAL STATISTICS:")
        print("=" * 60)
        
        # Use info from final step if available, otherwise count manually
        final_info = info if 'info' in locals() else {}
        civs_rescued = final_info.get('civilians_rescued', sum(1 for c in env.civilians if c['rescued']))
        total_civs = final_info.get('total_civilians', len(env.civilians))
        
        print(f"🎯 Steps taken: {env.step_count}")
        print(f"👥 Civilians rescued: {civs_rescued}/{total_civs}")
        print(f"📈 Rescue rate: {civs_rescued/max(total_civs,1)*100:.1f}%")
        
        print("\n🤖 AGENT PERFORMANCE:")
        for agent_id, agent in env.agents.items():
            state = agent.get_state()
            print(f"   {agent_id}:")
            # Use .get() for safe dictionary access
            print(f"     🎯 Civilians rescued: {state.get('civilians_rescued', 0)}")
            print(f"     👣 Steps taken: {state.get('steps_taken', 0)}")
            print(f"     📍 Final position: {state.get('position', 'N/A')}")
        
        env.close()
        print("\n🎉 Week 1 demonstration completed successfully!")
        print("🚀 Ready for Week 2: MARL Integration!")

    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        print("Please ensure all dependencies are installed and paths are correct.")
        print("Try running: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ An error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
