import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.components.simulation_viewer import SimulationViewer
from dashboard.components.metrics_display import MetricsDisplay
from dashboard.components.controls import ControlPanel

class DisasterResponseDashboard:
    """
    Main Streamlit dashboard for the Disaster Response AI system
    """
    
    def __init__(self):
        self.setup_page()
        self.simulation_viewer = SimulationViewer()
        self.metrics_display = MetricsDisplay()
        self.control_panel = ControlPanel()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="AI Disaster Response System",
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
        .agent-drone { color: #1f77b4; }
        .agent-ambulance { color: #d62728; }
        .agent-rescue { color: #2ca02c; }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.title("🚨 Control Panel")
        
        # Environment selection
        env_type = st.sidebar.selectbox(
            "Environment Type",
            ["Simple Grid", "Real Map - Lahore", "Real Map - Custom"]
        )
        
        # Agent configuration
        st.sidebar.subheader("🤖 Agent Configuration")
        num_drones = st.sidebar.slider("Number of Drones", 1, 5, 2)
        num_ambulances = st.sidebar.slider("Number of Ambulances", 1, 3, 1)
        num_rescue_teams = st.sidebar.slider("Number of Rescue Teams", 1, 2, 1)
        
        # Disaster parameters
        st.sidebar.subheader("🔥 Disaster Parameters")
        collapsed_buildings = st.sidebar.slider("Collapsed Buildings", 5, 20, 10)
        blocked_roads = st.sidebar.slider("Blocked Roads", 3, 15, 5)
        
        # AI Policy
        st.sidebar.subheader("🧠 AI Policy")
        policy_type = st.sidebar.selectbox(
            "Policy Type",
            ["Cooperative", "Competitive", "Random", "Rule-Based"]
        )
        
        # Control buttons
        st.sidebar.subheader("🎮 Simulation Controls")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_sim = st.button("🚀 Start Simulation", use_container_width=True)
        with col2:
            stop_sim = st.button("⏹️ Stop Simulation", use_container_width=True)
        
        st.sidebar.button("📊 Reset Metrics", use_container_width=True)
        
        return {
            'env_type': env_type,
            'num_drones': num_drones,
            'num_ambulances': num_ambulances,
            'num_rescue_teams': num_rescue_teams,
            'collapsed_buildings': collapsed_buildings,
            'blocked_roads': blocked_roads,
            'policy_type': policy_type,
            'start_sim': start_sim,
            'stop_sim': stop_sim
        }
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🚨 AI Disaster Response System</h1>', 
                   unsafe_allow_html=True)
        
        # Status bar
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Simulation Status", "Ready", delta="Stopped")
        with col2:
            st.metric("Active Agents", "0")
        with col3:
            st.metric("Civilians Rescued", "0/0")
        with col4:
            st.metric("Time Elapsed", "0s")
    
    def render_simulation_view(self, controls):
        """Render the main simulation view"""
        st.subheader("🎮 Live Simulation")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🌍 Map View", "📊 Analytics", "🤖 Agent View"])
        
        with tab1:
            # Placeholder for PyGame visualization
            st.info("🎮 Simulation visualization will appear here")
            st.image("https://via.placeholder.com/800x400?text=Simulation+View", 
                    use_column_width=True)
            
            # Simulation controls
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.button("▶️ Play")
            with col2:
                st.button("⏸️ Pause")
            with col3:
                st.button("⏹️ Stop")
            with col4:
                st.button("🔄 Reset")
        
        with tab2:
            self.render_analytics_tab()
        
        with tab3:
            self.render_agent_tab()
    
    def render_analytics_tab(self):
        """Render analytics tab"""
        st.subheader("📈 Performance Analytics")
        
        # Sample data - replace with real data
        col1, col2 = st.columns(2)
        
        with col1:
            # Rescue progress chart
            st.plotly_chart(self.create_rescue_chart(), use_container_width=True)
        
        with col2:
            # Agent performance
            st.plotly_chart(self.create_agent_performance_chart(), use_container_width=True)
        
        # Training metrics
        st.subheader("🧠 AI Training Metrics")
        col3, col4 = st.columns(2)
        
        with col3:
            st.plotly_chart(self.create_training_rewards_chart(), use_container_width=True)
        
        with col4:
            st.plotly_chart(self.create_efficiency_chart(), use_container_width=True)
    
    def render_agent_tab(self):
        """Render agent monitoring tab"""
        st.subheader("🤖 Agent Monitoring")
        
        # Agent status table
        agent_data = {
            'Agent ID': ['drone_1', 'drone_2', 'ambulance_1', 'rescue_1'],
            'Type': ['Drone', 'Drone', 'Ambulance', 'Rescue Team'],
            'Status': ['Scouting', 'Idle', 'Transporting', 'Rescuing'],
            'Position': ['(5, 7)', '(12, 3)', '(8, 8)', '(3, 10)'],
            'Civilians Rescued': [2, 1, 5, 3],
            'Battery/Resources': ['85%', '92%', '78%', '65%']
        }
        
        df = pd.DataFrame(agent_data)
        st.dataframe(df, use_container_width=True)
        
        # Agent communication
        st.subheader("📡 Agent Communication")
        comm_data = {
            'Time': ['00:01:23', '00:01:45', '00:02:10', '00:02:35'],
            'From': ['drone_1', 'ambulance_1', 'drone_2', 'rescue_1'],
            'To': ['All', 'drone_1', 'ambulance_1', 'All'],
            'Message': ['Civilian found at (7, 4)', 'Need assistance', 'Route blocked', 'Rescue complete']
        }
        
        comm_df = pd.DataFrame(comm_data)
        st.dataframe(comm_df, use_container_width=True)
    
    def create_rescue_chart(self):
        """Create rescue progress chart"""
        # Sample data
        time_points = list(range(0, 100, 10))
        civilians_rescued = [0, 2, 5, 8, 12, 15, 15, 15, 15, 15]
        civilians_total = [15] * len(time_points)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_points, y=civilians_rescued, 
                               mode='lines+markers', name='Rescued',
                               line=dict(color='green', width=3)))
        fig.add_trace(go.Scatter(x=time_points, y=civilians_total,
                               mode='lines', name='Total',
                               line=dict(color='red', width=2, dash='dash')))
        
        fig.update_layout(
            title='Civilian Rescue Progress',
            xaxis_title='Time (seconds)',
            yaxis_title='Civilians',
            showlegend=True
        )
        
        return fig
    
    def create_agent_performance_chart(self):
        """Create agent performance chart"""
        agents = ['Drone 1', 'Drone 2', 'Ambulance', 'Rescue Team']
        rescues = [3, 2, 8, 4]
        efficiency = [85, 78, 92, 88]
        
        fig = go.Figure(data=[
            go.Bar(name='Civilians Rescued', x=agents, y=rescues, marker_color='lightblue'),
            go.Bar(name='Efficiency %', x=agents, y=efficiency, marker_color='lightcoral')
        ])
        
        fig.update_layout(
            title='Agent Performance',
            barmode='group',
            xaxis_title='Agents',
            yaxis_title='Metrics'
        )
        
        return fig
    
    def create_training_rewards_chart(self):
        """Create training rewards chart"""
        episodes = list(range(1, 101))
        rewards = np.cumsum(np.random.normal(10, 5, 100))
        
        fig = px.line(x=episodes, y=rewards, 
                     title='Training Rewards Over Time',
                     labels={'x': 'Episode', 'y': 'Total Reward'})
        
        fig.update_traces(line_color='blue', line_width=2)
        
        return fig
    
    def create_efficiency_chart(self):
        """Create efficiency metrics chart"""
        metrics = ['Response Time', 'Resource Usage', 'Collaboration', 'Success Rate']
        scores = [85, 78, 92, 88]
        
        fig = go.Figure(data=[
            go.Scatterpolar(r=scores, theta=metrics, fill='toself', 
                          name='Efficiency Metrics')
        ])
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title='System Efficiency Metrics'
        )
        
        return fig
    
    def run(self):
        """Run the dashboard"""
        # Render components
        controls = self.render_sidebar()
        self.render_header()
        self.render_simulation_view(controls)

# Run the dashboard
if __name__ == "__main__":
    dashboard = DisasterResponseDashboard()
    dashboard.run()