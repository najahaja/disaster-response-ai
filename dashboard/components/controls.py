"""
Control Panel Component for Dashboard
Handles user inputs and simulation controls
"""

import streamlit as st
from typing import Dict, Any, List, Optional

class ControlPanel:
    """
    Component for simulation controls and configuration
    """
    
    def __init__(self):
        self.available_locations = [
            "Lahore, Pakistan",
            "Karachi, Pakistan",
            "Islamabad, Pakistan", 
            "Faisalabad, Pakistan",
            "Rawalpindi, Pakistan",
            "Generated City"
        ]
        
        self.agent_types = {
            "drone": {
                "name": "🛸 Drone",
                "description": "Aerial reconnaissance and scouting",
                "max_count": 5,
                "default_count": 2
            },
            "ambulance": {
                "name": "🚑 Ambulance", 
                "description": "Medical evacuation and transport",
                "max_count": 4,
                "default_count": 2
            },
            "rescue_team": {
                "name": "👷 Rescue Team",
                "description": "Search and rescue operations",
                "max_count": 3,
                "default_count": 1
            }
        }
    
    def render_environment_setup(self) -> Dict[str, Any]:
        """Render environment setup controls"""
        st.subheader("🗺️ Environment Configuration")
        
        # Location selection
        selected_location = st.selectbox(
            "Select Location",
            options=self.available_locations,
            index=0,
            help="Choose a real location or generated city"
        )
        
        # Map type information
        if selected_location == "Generated City":
            st.info("🏙️ Using procedurally generated city layout")
        else:
            st.info("🗺️ Using OpenStreetMap data for realistic layout")
        
        # Environment parameters
        col1, col2 = st.columns(2)
        
        with col1:
            grid_size = st.slider(
                "Grid Size",
                min_value=15,
                max_value=50,
                value=25,
                help="Size of the simulation grid"
            )
        
        with col2:
            disaster_intensity = st.slider(
                "Disaster Intensity",
                min_value=1,
                max_value=10,
                value=5,
                help="Severity of the disaster scenario"
            )
        
        return {
            'location': selected_location,
            'grid_size': grid_size,
            'disaster_intensity': disaster_intensity
        }
    
    def render_agent_configuration(self) -> Dict[str, int]:
        """Render agent configuration controls"""
        st.subheader("🤖 Agent Configuration")
        
        agent_counts = {}
        
        for agent_key, agent_info in self.agent_types.items():
            col1, col2 = st.columns([2, 3])
            
            with col1:
                count = st.slider(
                    agent_info["name"],
                    min_value=0,
                    max_value=agent_info["max_count"],
                    value=agent_info["default_count"],
                    key=f"agent_{agent_key}"
                )
                agent_counts[agent_key] = count
            
            with col2:
                st.caption(agent_info["description"])
        
        # Total agents warning
        total_agents = sum(agent_counts.values())
        if total_agents == 0:
            st.error("❌ Please add at least one agent")
        elif total_agents > 8:
            st.warning("⚠️ High agent count may affect performance")
        else:
            st.success(f"✅ Total agents: {total_agents}")
        
        return agent_counts
    
    def render_simulation_controls(self) -> Dict[str, bool]:
        """Render simulation control buttons"""
        st.subheader("🎮 Simulation Controls")
        
        control_states = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Simulation", width='stretch'):
                control_states['start'] = True
            else:
                control_states['start'] = False
        
        with col2:
            if st.button("⏸️ Pause Simulation", width='stretch'):
                control_states['pause'] = True
            else:
                control_states['pause'] = False
        
        with col3:
            if st.button("⏹️ Stop Simulation", width='stretch'):
                control_states['stop'] = True
            else:
                control_states['stop'] = False
        
        # Additional controls
        col4, col5 = st.columns(2)
        
        with col4:
            if st.button("🚨 Trigger Disaster", width='stretch'):
                control_states['disaster'] = True
            else:
                control_states['disaster'] = False
        
        with col5:
            if st.button("🔄 Reset Simulation", width='stretch'):
                control_states['reset'] = True
            else:
                control_states['reset'] = False
        
        return control_states
    
    def render_simulation_settings(self) -> Dict[str, Any]:
        """Render simulation settings"""
        st.subheader("⚙️ Simulation Settings")
        
        settings = {}
        
        # Simulation speed
        settings['speed'] = st.slider(
            "Simulation Speed",
            min_value=1,
            max_value=10,
            value=3,
            help="Faster speed may reduce visualization quality"
        )
        
        # Auto-step
        settings['auto_step'] = st.checkbox(
            "Auto-advance steps",
            value=True,
            help="Automatically advance simulation steps"
        )
        
        # Step delay
        if settings['auto_step']:
            settings['step_delay'] = st.slider(
                "Step Delay (ms)",
                min_value=100,
                max_value=2000,
                value=500,
                help="Delay between auto-steps in milliseconds"
            )
        
        # Visualization options
        col1, col2 = st.columns(2)
        
        with col1:
            settings['show_grid'] = st.checkbox("Show Grid", value=True)
            settings['show_agents'] = st.checkbox("Show Agents", value=True)
        
        with col2:
            settings['show_civilians'] = st.checkbox("Show Civilians", value=True)
            settings['show_paths'] = st.checkbox("Show Paths", value=False)
        
        return settings
    
    def render_advanced_settings(self) -> Dict[str, Any]:
        """Render advanced simulation settings"""
        with st.expander("🔧 Advanced Settings"):
            advanced = {}
            
            # MARL settings
            st.markdown("**Multi-Agent RL Settings**")
            advanced['learning_enabled'] = st.checkbox(
                "Enable Learning",
                value=False,
                help="Enable reinforcement learning for agents"
            )
            
            if advanced['learning_enabled']:
                advanced['learning_rate'] = st.slider(
                    "Learning Rate",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    format="%.3f"
                )
                
                advanced['exploration_rate'] = st.slider(
                    "Exploration Rate",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.1,
                    format="%.2f"
                )
            
            # Communication settings
            st.markdown("**Communication Settings**")
            advanced['communication_enabled'] = st.checkbox(
                "Enable Agent Communication",
                value=True,
                help="Allow agents to share information"
            )
            
            if advanced['communication_enabled']:
                advanced['comm_range'] = st.slider(
                    "Communication Range",
                    min_value=1,
                    max_value=10,
                    value=5
                )
            
            # Performance settings
            st.markdown("**Performance Settings**")
            advanced['max_steps'] = st.number_input(
                "Maximum Steps",
                min_value=100,
                max_value=10000,
                value=1000,
                help="Maximum simulation steps before auto-reset"
            )
            
            advanced['parallel_processing'] = st.checkbox(
                "Parallel Processing",
                value=False,
                help="Use parallel processing for agent decisions (experimental)"
            )
            
            return advanced
    
    def render_quick_actions(self):
        """Render quick action buttons"""
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Save Snapshot", width='stretch'):
                st.session_state.save_snapshot = True
        
        with col2:
            if st.button("📹 Record Video", width='stretch'):
                st.session_state.record_video = True
        
        with col3:
            if st.button("📋 Export Data", width='stretch'):
                st.session_state.export_data = True
        
        with col4:
            if st.button("🖼️ Screenshot", width='stretch'):
                st.session_state.take_screenshot = True
    
    def render_preset_scenarios(self) -> Optional[str]:
        """Render preset scenario selection"""
        st.subheader("🎯 Preset Scenarios")
        
        scenarios = {
            "earthquake": {
                "name": "🌋 Earthquake Response",
                "description": "Urban earthquake with building collapses",
                "agents": {"drone": 3, "ambulance": 3, "rescue_team": 2},
                "intensity": 8
            },
            "flood": {
                "name": "🌊 Flood Response", 
                "description": "Flooding scenario with water obstacles",
                "agents": {"drone": 2, "ambulance": 2, "rescue_team": 1},
                "intensity": 6
            },
            "fire": {
                "name": "🔥 Fire Response",
                "description": "Urban fire with spreading hazards",
                "agents": {"drone": 4, "ambulance": 2, "rescue_team": 1},
                "intensity": 7
            },
            "custom": {
                "name": "⚙️ Custom Scenario",
                "description": "Configure your own scenario",
                "agents": {},
                "intensity": 5
            }
        }
        
        scenario_options = {key: f"{scenario['name']} - {scenario['description']}" 
                          for key, scenario in scenarios.items()}
        
        selected_scenario = st.selectbox(
            "Choose Scenario",
            options=list(scenario_options.keys()),
            format_func=lambda x: scenario_options[x],
            index=3  # Default to custom
        )
        
        # Show scenario details
        if selected_scenario != "custom":
            scenario = scenarios[selected_scenario]
            st.info(f"**{scenario['name']}**\n\n{scenario['description']}")
            
            # Auto-configure if requested
            if st.button(f"Apply {scenario['name']} Configuration", width='stretch'):
                st.session_state.preset_scenario = selected_scenario
                st.session_state.agent_config = scenario['agents']
                st.session_state.disaster_intensity = scenario['intensity']
                st.success(f"✅ {scenario['name']} configuration applied!")
        
        return selected_scenario
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """Get the current configuration from all controls"""
        config = {}
        
        # Environment config
        config['environment'] = self.render_environment_setup()
        
        # Agent config  
        config['agents'] = self.render_agent_configuration()
        
        # Simulation settings
        config['settings'] = self.render_simulation_settings()
        
        # Advanced settings
        config['advanced'] = self.render_advanced_settings()
        
        # Preset scenario
        config['scenario'] = self.render_preset_scenarios()
        
        # Quick actions
        self.render_quick_actions()
        
        return config
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate the configuration and return any errors"""
        errors = []
        
        # Check agent counts
        total_agents = sum(config['agents'].values())
        if total_agents == 0:
            errors.append("At least one agent must be configured")
        
        if total_agents > 10:
            errors.append("Maximum 10 agents allowed for performance reasons")
        
        # Check environment settings
        if config['environment']['grid_size'] < 10:
            errors.append("Grid size must be at least 10x10")
        
        # Check advanced settings
        if config['advanced'].get('max_steps', 1000) < 100:
            errors.append("Maximum steps must be at least 100")
        
        return errors

# Test function
def test_control_panel():
    """Test the control panel component"""
    st.title("Control Panel Test")
    
    panel = ControlPanel()
    
    # Test all components
    config = panel.get_current_configuration()
    
    st.subheader("Current Configuration")
    st.json(config)
    
    # Validate configuration
    errors = panel.validate_configuration(config)
    if errors:
        st.error("Configuration errors:")
        for error in errors:
            st.write(f"❌ {error}")
    else:
        st.success("✅ Configuration is valid!")

if __name__ == "__main__":
    test_control_panel()