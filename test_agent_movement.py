"""
Quick test to verify agents can move properly
"""

import sys
sys.path.append('.')

from environments.simple_grid_env import SimpleGridEnv
import numpy as np

def test_agent_movement():
    print("=" * 70)
    print("AGENT MOVEMENT TEST")
    print("=" * 70)
    
    # Create environment
    env = SimpleGridEnv(
        grid_size=20,
        n_drones=1,
        n_ambulances=1,
        n_rescue_teams=1,
        spawn_civilians=False
    )
    
    print(f"\n✅ Environment created with {len(env.agents)} agents")
    
    # Show agent starting positions
    print(f"\n📍 Starting Positions:")
    for agent_id, agent in env.agents.items():
        pos = agent.position
        cell_type = env.grid[int(pos[1]), int(pos[0])]
        print(f"   - {agent_id} ({agent.agent_type}): {pos} on cell type {cell_type}")
    
    print(f"\n🔄 Testing movement for 10 steps...")
    print("=" * 70)
    
    # Test movement for 10 steps
    for step in range(10):
        print(f"\n--- Step {step + 1} ---")
        
        # Try to move each agent
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Show positions after move
        if step < 5:  # Only show first 5 steps to avoid spam
            for agent_id, agent in env.agents.items():
                print(f"   {agent_id}: {agent.position} (steps taken: {agent.steps_taken})")
    
    print(f"\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    for agent_id, agent in env.agents.items():
        print(f"{agent_id}:")
        print(f"  - Type: {agent.agent_type}")
        print(f"  - Final position: {agent.position}")
        print(f"  - Total steps taken: {agent.steps_taken}")
        print(f"  - Movement success: {'✅ YES' if agent.steps_taken > 0 else '❌ NO'}")
    
    env.close()
    
    # Check if any agent moved
    any_moved = any(agent.steps_taken > 0 for agent in env.agents.values())
    
    if any_moved:
        print(f"\n✅ SUCCESS: Agents are moving!")
    else:
        print(f"\n❌ FAILURE: No agents moved!")
    
    return any_moved

if __name__ == "__main__":
    success = test_agent_movement()
    sys.exit(0 if success else 1)
