#!/usr/bin/env python3
"""
Main Dashboard Application for Disaster Response AI
Streamlit-based dashboard for monitoring and controlling simulations
"""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project components
from environments.real_map_env import RealMapEnv
from environments.simple_grid_env import SimpleGridEnv
from agents.drone_agent import DroneAgent
from agents.ambulance_agent import AmbulanceAgent
from agents.rescue_team_agent import RescueTeamAgent
from dashboard.components.controls import ControlPanel
from dashboard.components.metrics_display import MetricsDisplay
from dashboard.components.simulation_viewer import SimulationViewer
import pygame
pygame.init()
class DisasterResponseDashboard:
    """
    Main dashboard class for Disaster Response AI simulation
    """
    
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()
        self.control_panel = ControlPanel()
        self.metrics_display = MetricsDisplay()
        self.simulation_viewer = SimulationViewer()
        
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Disaster Response AI Dashboard",
            page_icon="🚨",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #ff4b4b;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #ff4b4b;
        }
        .success-metric {
            border-left: 5px solid #00cc96;
        }
        .warning-metric {
            border-left: 5px solid #ffa726;
        }
        .danger-metric {
            border-left: 5px solid #ff4b4b;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
        if 'environment' not in st.session_state:
            st.session_state.environment = None
        if 'simulation_data' not in st.session_state:
            st.session_state.simulation_data = {
                'steps': [],
                'metrics': [],
                'agent_positions': {},
                'civilian_rescues': 0,
                'start_time': None
            }
        if 'selected_location' not in st.session_state:
            st.session_state.selected_location = "Lahore, Pakistan"
        if 'disaster_triggered' not in st.session_state:
            st.session_state.disaster_triggered = False
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🚨 Disaster Response AI Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Status indicator
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "🟢 Running" if st.session_state.simulation_running else "🔴 Stopped"
            st.metric("Simulation Status", status)
        
        with col2:
            if st.session_state.environment:
                env_type = "Real Map" if hasattr(st.session_state.environment, 'real_map_loaded') and st.session_state.environment.real_map_loaded else "Generated"
                st.metric("Environment", env_type)
            else:
                st.metric("Environment", "Not Loaded")
        
        with col3:
            civilian_count = st.session_state.simulation_data.get('civilian_rescues', 0)
            st.metric("Civilians Rescued", civilian_count)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("🎮 Simulation Controls")
            
            # Environment selection
            st.subheader("Environment Setup")
            location_options = [
                "Lahore, Pakistan",
                "Karachi, Pakistan", 
                "Islamabad, Pakistan",
                "Generated City"
            ]
            selected_location = st.selectbox(
                "Select Location",
                options=location_options,
                index=0
            )
            
            # Agent configuration
            st.subheader("Agent Configuration")
            col1, col2 = st.columns(2)
            with col1:
                num_drones = st.slider("Drones", 1, 5, 2)
                num_ambulances = st.slider("Ambulances", 1, 5, 2)
            with col2:
                num_rescue_teams = st.slider("Rescue Teams", 1, 3, 1)
            
            # Simulation controls
            st.subheader("Simulation Control")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Start Simulation", width='stretch'):
                    self.start_simulation(selected_location, num_drones, num_ambulances, num_rescue_teams)
            
            with col2:
                if st.button("⏹️ Stop Simulation", width='stretch'):
                    self.stop_simulation()
            
            if st.button("🚨 Trigger Disaster", width='stretch'):
                self.trigger_disaster()
            
            # Simulation speed
            sim_speed = st.slider("Simulation Speed", 1, 10, 3)
            st.session_state.simulation_speed = sim_speed
            
            # Real-time metrics
            st.subheader("📊 Live Metrics")
            if st.session_state.simulation_running and st.session_state.environment:
                self.render_live_metrics()
    
    def render_live_metrics(self):
        """Render live metrics in sidebar"""
        env = st.session_state.environment
        metrics = {
            "Step": env.step_count,
            "Active Agents": len(env.agents),
            "Civilians": len(env.civilians),
            "Rescued": sum(1 for c in env.civilians if c.get('rescued', False)),
            "Collapsed Buildings": len(env.collapsed_buildings),
            "Blocked Roads": len(env.blocked_roads)
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
    
    def render_main_content(self):
        """Render main dashboard content"""
        if not st.session_state.simulation_running:
            self.render_welcome_screen()
        else:
            self.render_simulation_dashboard()
    
    def render_welcome_screen(self):
        """Render welcome screen when no simulation is running"""
        st.markdown("""
        ## 🌟 Welcome to Disaster Response AI Dashboard
        
        This dashboard allows you to monitor and control AI-powered disaster response simulations.
        
        ### 🎯 Features:
        - **Real Map Integration**: Use OpenStreetMap data for realistic scenarios
        - **Multi-Agent Coordination**: Drones, ambulances, and rescue teams working together
        - **Live Monitoring**: Real-time metrics and visualization
        - **Performance Analytics**: Detailed analysis of response effectiveness
        
        ### 🚀 Getting Started:
        1. Select a location from the sidebar
        2. Configure your agents (drones, ambulances, rescue teams)
        3. Click 'Start Simulation' to begin
        4. Trigger disasters and monitor the response
        
        ### 📊 What you'll see:
        - Real-time agent positions and movements
        - Civilian rescue progress
        - Environmental damage assessment
        - Agent coordination metrics
        """)
        
        # Add some sample visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample performance chart
            st.subheader("Sample Performance Metrics")
            sample_data = pd.DataFrame({
                'Time': range(10),
                'Rescues': [0, 2, 5, 8, 12, 15, 18, 20, 22, 25],
                'Efficiency': [0.1, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87]
            })
            fig = px.line(sample_data, x='Time', y=['Rescues', 'Efficiency'], 
                         title="Expected Performance Trend")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sample agent distribution
            st.subheader("Sample Agent Distribution")
            agent_data = pd.DataFrame({
                'Agent Type': ['Drones', 'Ambulances', 'Rescue Teams'],
                'Count': [3, 2, 1],
                'Color': ['#1f77b4', '#ff7f0e', '#2ca02c']
            })
            fig = px.pie(agent_data, values='Count', names='Agent Type', 
                        title="Agent Team Composition")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_simulation_dashboard(self):
        """Render simulation dashboard when running"""
        # Main simulation view
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Live Simulation View")
            self.render_simulation_view()
        
        with col2:
            st.subheader("📈 Performance Metrics")
            self.render_performance_metrics()
        
        # Detailed analytics
        st.subheader("📊 Detailed Analytics")
        self.render_detailed_analytics()
    def update_simulation_data(self):
        """Log current metrics to the session state for historical charts"""
        if not st.session_state.environment:
            return
            
        env = st.session_state.environment
        
        # --- Make sure env.civilians is a list ---
        # (This is a safety check based on your screenshot)
        if not isinstance(env.civilians, list):
             env.civilians = [] # Or handle as needed

        rescued = sum(1 for c in env.civilians if c.get('rescued', False))
        efficiency = min(env.step_count / max(1, rescued) if rescued > 0 else env.step_count, 100)
        
        current_data = {
            'rescues': rescued,
            'efficiency': efficiency,
            'active_agents': len(env.agents),
            'collapsed_buildings': len(env.collapsed_buildings)
        }
        
        st.session_state.simulation_data['steps'].append(current_data)
    def render_simulation_view(self):
        """Render the simulation visualization AND advance the step"""
        if st.session_state.environment:
            env = st.session_state.environment
            
            try:
                # --- THIS IS THE FIX ---

                # 1. ADVANCE THE SIMULATION ONE STEP
                # We create a random action for each agent
                actions = {}
                for agent_id in env.agents.keys():
                    actions[agent_id] = env.action_space.sample() 
                
                # Call the step function to move the simulation forward
                obs, reward, terminated, truncated, info = env.step(actions)
                
                # 2. RENDER the new state
                # This calls the render() function in your environment
                pygame_surface = env.render()
                
                if pygame_surface is not None:
                    # 3. Update the viewer's frame with the new image
                    self.simulation_viewer.update_frame(pygame_surface)
                
                # 4. Tell the viewer to display the image (using st.image)
                self.simulation_viewer.render()

                # 5. Check if simulation is over
                if terminated or truncated:
                    st.session_state.simulation_running = False
                    st.balloons()
                    st.success("Simulation Complete!")
                
                # --- END OF FIX ---

            except Exception as e:
                st.error(f"Failed to render environment: {e}")
                st.session_state.simulation_running = False
            
            # Auto-refresh
            if st.session_state.simulation_running:
                # Log the data for the charts
                self.update_simulation_data() 
                time.sleep(1.0 / st.session_state.simulation_speed)
                st.rerun()
        else:
            # If no env, just tell the viewer to render its default placeholder
            self.simulation_viewer.render()
    
    def render_performance_metrics(self):
        """Render performance metrics"""
        if not st.session_state.environment:
            return
        
        env = st.session_state.environment
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rescued = sum(1 for c in env.civilians if c.get('rescued', False))
            total = len(env.civilians)
            rescue_rate = (rescued / total * 100) if total > 0 else 0
            st.metric("Rescue Rate", f"{rescue_rate:.1f}%")
        
        with col2:
            efficiency = min(env.step_count / max(1, rescued) if rescued > 0 else env.step_count, 100)
            st.metric("Efficiency", f"{efficiency:.1f}")
        
        with col3:
            collaboration = len(env.agents) * 10  # Simplified metric
            st.metric("Collaboration", f"{collaboration}%")
        
        # Progress bars
        st.progress(rescue_rate / 100, text="Rescue Progress")
        
        # Agent status
        st.subheader("🤖 Agent Status")
        for agent_id, agent in env.agents.items():
            status = "🟢 Active" if hasattr(agent, 'active') and agent.active else "🟡 Idle"
            st.write(f"{agent_id}: {status}")
    
    def render_detailed_analytics(self):
        """Render detailed analytics"""
        if not st.session_state.environment:
            return
        
        env = st.session_state.environment
        
        # Analytics tabs
        tab1, tab2, tab3 = st.tabs(["📈 Performance", "🗺️ Environment", "🤖 Agents"])
        
        with tab1:
            self.render_performance_analytics()
        
        with tab2:
            self.render_environment_analytics()
        
        with tab3:
            self.render_agent_analytics()
    
    def render_performance_analytics(self):
        """Render performance analytics"""
        # Time series data
        time_data = pd.DataFrame({
            'Step': range(len(st.session_state.simulation_data['steps'])),
            'Rescues': [data.get('rescues', 0) for data in st.session_state.simulation_data['steps']],
            'Efficiency': [data.get('efficiency', 0) for data in st.session_state.simulation_data['steps']]
        })
        
        fig = px.line(time_data, x='Step', y=['Rescues', 'Efficiency'],
                     title="Performance Over Time")
        st.plotly_chart(fig,use_container_width=True)
    
    def render_environment_analytics(self):
        """Render environment analytics"""
        env = st.session_state.environment
        
        # Environment statistics
        if hasattr(env, 'grid'):
            unique, counts = np.unique(env.grid, return_counts=True)
            cell_data = []
            for val, count in zip(unique, counts):
                cell_name = "Unknown"
                for name, cell_val in env.cell_types.items():
                    if cell_val == val:
                        cell_name = name.replace('_', ' ').title()
                        break
                cell_data.append({'Type': cell_name, 'Count': count})
            
            df = pd.DataFrame(cell_data)
            fig = px.bar(df, x='Type', y='Count', title="Environment Composition")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_agent_analytics(self):
        """Render agent analytics"""
        env = st.session_state.environment
        
        # Agent statistics
        agent_types = {}
        for agent_id, agent in env.agents.items():
            agent_type = type(agent).__name__.replace('Agent', '')
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        if agent_types:
            df = pd.DataFrame({
                'Agent Type': list(agent_types.keys()),
                'Count': list(agent_types.values())
            })
            fig = px.pie(df, values='Count', names='Agent Type', title="Agent Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    def start_simulation(self, location, num_drones, num_ambulances, num_rescue_teams):
        """Start a new simulation"""
        try:
            # Create environment
            if location == "Generated City":
                env = RealMapEnv()  # Will use generated map
            else:
                env = RealMapEnv(location)
            
            # Reset environment
            obs, info = env.reset()
            
            # Add agents
            for i in range(num_drones):
                drone = DroneAgent(f"drone_{i}", np.array([5 + i, 5]), env.config)
                env.add_agent(drone)
            
            for i in range(num_ambulances):
                ambulance = AmbulanceAgent(f"ambulance_{i}", np.array([10, 10 + i]), env.config)
                env.add_agent(ambulance)
            
            for i in range(num_rescue_teams):
                rescue_team = RescueTeamAgent(f"rescue_team_{i}", np.array([15, 5 + i]), env.config)
                env.add_agent(rescue_team)
            
            # Update session state
            st.session_state.environment = env
            st.session_state.simulation_running = True
            st.session_state.selected_location = location
            st.session_state.simulation_data['start_time'] = datetime.now()
            
            st.success(f"🚀 Simulation started for {location}!")
            
        except Exception as e:
            st.error(f"❌ Failed to start simulation: {e}")
    
    def stop_simulation(self):
        """Stop the current simulation"""
        if st.session_state.environment:
            st.session_state.environment.close()
        
        st.session_state.simulation_running = False
        st.session_state.environment = None
        st.success("⏹️ Simulation stopped.")
    
    def trigger_disaster(self):
        """Trigger disaster in the simulation"""
        if st.session_state.environment and not st.session_state.disaster_triggered:
            st.session_state.environment.trigger_disaster()
            st.session_state.disaster_triggered = True
            st.success("🚨 Disaster triggered!")
        elif st.session_state.disaster_triggered:
            st.warning("⚠️ Disaster already triggered!")
        else:
            st.error("❌ No active simulation!")
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()

# Main application
def main():
    """Main dashboard application"""
    dashboard = DisasterResponseDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()