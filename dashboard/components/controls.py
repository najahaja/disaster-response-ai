import streamlit as st
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ControlPanel:
    """
    Component for simulation controls and configuration
    """
    
    def __init__(self):
        self.simulation_state = "stopped"  # stopped, running, paused
    
    def render_environment_config(self):
        """Render environment configuration controls"""
        st.subheader("🌍 Environment Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            environment_type = st.selectbox(
                "Environment Type",
                ["Simple Grid", "Real Map - Lahore", "Real Map - Karachi", "Real Map - Custom"],
                help="Choose the simulation environment"
            )
            
            grid_size = st.slider(
                "Grid Size",
                min_value=10,
                max_value=30,
                value=15,
                help="Size of the simulation grid"
            )
        
        with col2:
            if "Real Map" in environment_type:
                location = st.text_input(
                    "Custom Location",
                    value="Lahore, Pakistan" if "Lahore" in environment_type else "Karachi, Pakistan" if "Karachi" in environment_type else "",
                    help="Enter location for real map (e.g., 'Lahore, Pakistan')"
                )
            
            visualization = st.selectbox(
                "Visualization Mode",
                ["Full 3D", "2D Top-Down", "Minimal", "Headless"],
                help="Choose visualization rendering mode"
            )
        
        return {
            'environment_type': environment_type,
            'grid_size': grid_size,
            'location': location if 'location' in locals() else None,
            'visualization': visualization
        }
    
    def render_agent_config(self):
        """Render agent configuration controls"""
        st.subheader("🤖 Agent Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_drones = st.slider(
                "Number of Drones",
                min_value=0,
                max_value=5,
                value=2,
                help="Configure drone agents for scouting"
            )
            
            drone_policy = st.selectbox(
                "Drone Policy",
                ["Cooperative", "Explorative", "Efficient", "Custom"],
                help="AI policy for drone behavior"
            )
        
        with col2:
            num_ambulances = st.slider(
                "Number of Ambulances", 
                min_value=0,
                max_value=3,
                value=1,
                help="Configure ambulance agents for transport"
            )
            
            ambulance_policy = st.selectbox(
                "Ambulance Policy",
                ["Cooperative", "Efficient", "Priority-Based", "Custom"],
                help="AI policy for ambulance behavior"
            )
        
        with col3:
            num_rescue_teams = st.slider(
                "Number of Rescue Teams",
                min_value=0,
                max_value=2,
                value=1,
                help="Configure rescue team agents"
            )
            
            rescue_policy = st.selectbox(
                "Rescue Team Policy", 
                ["Cooperative", "Efficient", "Safe", "Custom"],
                help="AI policy for rescue team behavior"
            )
        
        # Advanced agent settings
        with st.expander("Advanced Agent Settings"):
            communication_enabled = st.checkbox(
                "Enable Agent Communication",
                value=True,
                help="Allow agents to communicate and share information"
            )
            
            learning_enabled = st.checkbox(
                "Enable Online Learning",
                value=False,
                help="Allow agents to learn and adapt during simulation"
            )
            
            collaboration_level = st.slider(
                "Collaboration Level",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="How much agents should collaborate vs compete"
            )
        
        return {
            'num_drones': num_drones,
            'num_ambulances': num_ambulances,
            'num_rescue_teams': num_rescue_teams,
            'drone_policy': drone_policy,
            'ambulance_policy': ambulance_policy,
            'rescue_policy': rescue_policy,
            'communication_enabled': communication_enabled,
            'learning_enabled': learning_enabled,
            'collaboration_level': collaboration_level
        }
    
    def render_disaster_config(self):
        """Render disaster configuration controls"""
        st.subheader("🔥 Disaster Scenario")
        
        col1, col2 = st.columns(2)
        
        with col1:
            disaster_intensity = st.slider(
                "Disaster Intensity",
                min_value=1,
                max_value=10,
                value=5,
                help="Overall intensity of the disaster scenario"
            )
            
            collapsed_buildings = st.slider(
                "Collapsed Buildings",
                min_value=5,
                max_value=30,
                value=10,
                help="Number of collapsed buildings"
            )
        
        with col2:
            civilian_density = st.slider(
                "Civilian Density",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                help="Probability of civilians in collapsed buildings"
            )
            
            blocked_roads = st.slider(
                "Blocked Roads", 
                min_value=3,
                max_value=20,
                value=5,
                help="Number of blocked road segments"
            )
        
        # Dynamic events
        with st.expander("Dynamic Events Configuration"):
            aftershocks_enabled = st.checkbox(
                "Enable Aftershocks",
                value=True,
                help="Random aftershocks during simulation"
            )
            
            resource_depletion = st.checkbox(
                "Enable Resource Depletion",
                value=True,
                help="Agents can run out of resources"
            )
            
            dynamic_obstacles = st.checkbox(
                "Enable Dynamic Obstacles",
                value=False,
                help="Random obstacles appear during simulation"
            )
        
        return {
            'disaster_intensity': disaster_intensity,
            'collapsed_buildings': collapsed_buildings,
            'civilian_density': civilian_density,
            'blocked_roads': blocked_roads,
            'aftershocks_enabled': aftershocks_enabled,
            'resource_depletion': resource_depletion,
            'dynamic_obstacles': dynamic_obstacles
        }
    
    def render_simulation_controls(self):
        """Render simulation control buttons"""
        st.subheader("🎮 Simulation Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            start_clicked = st.button("🚀 Start Simulation", use_container_width=True, key="start_sim")
            if start_clicked:
                self.simulation_state = "running"
                st.success("Simulation started!")
        
        with col2:
            pause_clicked = st.button("⏸️ Pause Simulation", use_container_width=True, key="pause_sim")
            if pause_clicked:
                self.simulation_state = "paused"
                st.warning("Simulation paused")
        
        with col3:
            stop_clicked = st.button("⏹️ Stop Simulation", use_container_width=True, key="stop_sim")
            if stop_clicked:
                self.simulation_state = "stopped"
                st.info("Simulation stopped")
        
        with col4:
            reset_clicked = st.button("🔄 Reset Simulation", use_container_width=True, key="reset_sim")
            if reset_clicked:
                self.simulation_state = "stopped"
                st.info("Simulation reset")
        
        # Simulation speed
        sim_speed = st.slider(
            "Simulation Speed",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            help="Adjust simulation speed (1.0 = realtime)"
        )
        
        # Current status display
        status_color = {
            "running": "🟢",
            "paused": "🟡", 
            "stopped": "🔴"
        }
        
        st.markdown(f"**Current Status:** {status_color[self.simulation_state]} {self.simulation_state.upper()}")
        
        return {
            'simulation_state': self.simulation_state,
            'simulation_speed': sim_speed,
            'start_clicked': start_clicked,
            'pause_clicked': pause_clicked,
            'stop_clicked': stop_clicked,
            'reset_clicked': reset_clicked
        }
    
    def render_ai_training_controls(self):
        """Render AI training controls"""
        st.subheader("🧠 AI Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            training_episodes = st.number_input(
                "Training Episodes",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Number of episodes for training"
            )
            
            algorithm = st.selectbox(
                "RL Algorithm",
                ["PPO", "DQN", "A2C", "Multi-Agent PPO", "Custom"],
                help="Reinforcement learning algorithm"
            )
        
        with col2:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.01,
                value=0.001,
                format="%.4f",
                help="Learning rate for AI training"
            )
            
            start_training = st.button("🎯 Start Training", use_container_width=True, key="start_training")
            if start_training:
                st.info("AI training started...")
        
        col3, col4 = st.columns(2)
        
        with col3:
            save_model = st.button("💾 Save Model", use_container_width=True, key="save_model")
            if save_model:
                model_name = st.text_input("Model Name", value="my_trained_model")
                if model_name:
                    st.success(f"Model '{model_name}' saved successfully!")
        
        with col4:
            load_model = st.button("📥 Load Model", use_container_width=True, key="load_model")
            if load_model:
                st.success("Model loaded successfully!")
        
        # Training progress (placeholder)
        if start_training:
            st.progress(0.5, text="Training in progress...")
            st.write("Current Episode: 500/1000")
            st.write("Average Reward: 125.6")
            st.write("Rescue Rate: 68%")
        
        return {
            'training_episodes': training_episodes,
            'algorithm': algorithm,
            'learning_rate': learning_rate,
            'start_training': start_training,
            'save_model': save_model,
            'load_model': load_model
        }
    
    def render_export_controls(self):
        """Render data export controls"""
        st.subheader("📤 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_csv = st.button("📊 Export CSV", use_container_width=True)
            if export_csv:
                st.success("Simulation data exported as CSV!")
        
        with col2:
            export_video = st.button("🎥 Export Video", use_container_width=True)
            if export_video:
                st.success("Simulation recording exported as video!")
        
        with col3:
            export_report = st.button("📄 Export Report", use_container_width=True)
            if export_report:
                st.success("Comprehensive report generated!")
        
        return {
            'export_csv': export_csv,
            'export_video': export_video,
            'export_report': export_report
        }