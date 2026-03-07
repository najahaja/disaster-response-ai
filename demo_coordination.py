"""
Demonstration of Multi-Agent Coordination System
Shows how drones scout and share civilian locations with ground teams
"""

import sys
sys.path.append('.')

from environments.simple_grid_env import SimpleGridEnv
import time
from blockchain.blockchain_logger import BlockchainLogger

def demonstrate_coordination():
    print("=" * 70)
    print("MULTI-AGENT COORDINATION DEMONSTRATION")
    print("=" * 70)
    print("\nScenario: Civilians trapped in buildings need rescue")
    print("- Drones scout and find civilians")
    print("- Drones share coordinates with ground teams")
    print("- Ground vehicles navigate to rescue locations")
    print("=" * 70)
    
    # Create environment with multiple agents and civilians
    env = SimpleGridEnv(
        grid_size=20,
        n_drones=2,
        n_ambulances=1,
        n_rescue_teams=1,
        spawn_civilians=True,
        n_civilians=5
    )
    
    print(f"\n✅ Environment created:")
    print(f"   - Grid size: {env.grid_size}x{env.grid_size}")
    print(f"   - Agents: {len(env.agents)}")
    print(f"   - Civilians: {len(env.civilians)}")
    
    print(f"\n🔗 Connecting to Blockchain...")
    try:
        logger = BlockchainLogger()
        print("   ✅ Blockchain Logger Initialized!")
    except Exception as e:
        print(f"   ❌ Failed to initialize Blockchain Logger: {e}")
        logger = None
        
    logged_discoveries = set()
    logged_rescues = set()
    
    # Show initial agent positions
    print(f"\n📍 Initial Agent Positions:")
    for agent_id, agent in env.agents.items():
        print(f"   - {agent_id} ({agent.agent_type}): {agent.position}")
    
    # Show civilian positions
    print(f"\n👥 Civilian Locations:")
    for i, civ in enumerate(env.civilians):
        pos = civ['position']
        cell_type = env.grid[pos[1], pos[0]]
        in_building = (cell_type == env.cell_types['BUILDING'] or 
                      cell_type == env.cell_types.get('COLLAPSED', -1))
        location_type = "in building" if in_building else "in open area"
        print(f"   - Civilian {i}: {pos} ({location_type})")
    
    print(f"\n" + "=" * 70)
    print("STARTING COORDINATION SIMULATION")
    print("=" * 70)
    
    # Run simulation for several steps
    for step in range(15):
        print(f"\n--- Step {step + 1} ---")
        
        # Take random actions (in real scenario, AI would control this)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get coordination status
        coord_info = env.get_coordination_info()
        stats = coord_info['stats']
        
        print(f"Coordination Status:")
        print(f"  - Civilians discovered: {stats['total_discovered']}")
        print(f"  - Rescue in progress: {stats['in_progress']}")
        print(f"  - Successfully rescued: {stats['rescued']}")
        print(f"  - Unassigned: {stats['unassigned']}")
        
        # Show discovered civilians
        if coord_info['discovered_civilians']:
            print(f"\n  Discovered Civilians:")
            for civ_id, civ_data in coord_info['discovered_civilians'].items():
                status = civ_data['status']
                discovered_by = civ_data['discovered_by']
                assigned_to = civ_data.get('assigned_to', 'None')
                print(f"    - {civ_id}: {status} (found by {discovered_by}, assigned to {assigned_to})")
                
                pos = civ_data.get('position', (0,0))
                loc_str = f"x:{pos[0]}, y:{pos[1]}"
                
                if logger and civ_id not in logged_discoveries:
                    logger.log_event(str(discovered_by), "SURVIVOR_DISCOVERED", loc_str)
                    logged_discoveries.add(civ_id)
                
                if logger and status == 'rescued' and civ_id not in logged_rescues:
                    logger.log_event(str(assigned_to), "SURVIVOR_RESCUED", loc_str)
                    logged_rescues.add(civ_id)
        
        # Show pending rescue requests
        if coord_info['pending_requests']:
            print(f"\n  Pending Rescue Requests: {len(coord_info['pending_requests'])}")
        
        # Check if all rescued
        if terminated:
            print(f"\n🎉 ALL CIVILIANS RESCUED!")
            break
        
        if truncated:
            print(f"\n⏱️ Time limit reached")
            break
        
        time.sleep(0.1)  # Small delay for readability
    
    print(f"\n" + "=" * 70)
    print("FINAL COORDINATION SUMMARY")
    print("=" * 70)
    
    final_stats = env.get_coordination_info()['stats']
    print(f"Total civilians discovered: {final_stats['total_discovered']}")
    print(f"Successfully rescued: {final_stats['rescued']}")
    print(f"Rescue success rate: {final_stats['rescued'] / max(final_stats['total_discovered'], 1) * 100:.1f}%")
    
    print(f"\n" + "=" * 70)
    print("KEY FEATURES DEMONSTRATED:")
    print("=" * 70)
    print("✅ Drones automatically scout for civilians")
    print("✅ Civilian locations shared via communication hub")
    print("✅ Ground vehicles receive coordinates")
    print("✅ Rescue assignments tracked and coordinated")
    print("✅ Real-time status updates for all agents")
    print("=" * 70)
    
    env.close()

if __name__ == "__main__":
    demonstrate_coordination()
