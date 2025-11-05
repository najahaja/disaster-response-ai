"""
Metrics Display Component for Dashboard
Handles real-time metrics and performance visualization
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

class MetricsDisplay:
    """
    Component for displaying real-time metrics and analytics
    """
    
    def __init__(self):
        self.metrics_history = {
            'time': [],
            'rescues': [],
            'efficiency': [],
            'agent_activity': [],
            'collaboration': [],
            'environment_damage': []
        }
    
    def update_metrics(self, environment, step_data: Dict[str, Any]):
        """Update metrics with new simulation data"""
        current_time = datetime.now()
        
        # Basic metrics
        rescued = sum(1 for c in environment.civilians if c.get('rescued', False))
        total_civilians = len(environment.civilians)
        rescue_rate = (rescued / total_civilians * 100) if total_civilians > 0 else 0
        
        # Efficiency metric (simplified)
        efficiency = min(environment.step_count / max(1, rescued) if rescued > 0 else 0, 100)
        
        # Agent activity
        agent_activity = sum(
            getattr(agent, 'steps_taken', 0) for agent in environment.agents.values()
        ) / max(1, len(environment.agents))
        
        # Collaboration metric (simplified)
        collaboration = min(len(environment.agents) * 15, 100)
        
        # Environment damage
        damage = (len(environment.collapsed_buildings) + len(environment.blocked_roads)) / 10
        
        # Update history
        self.metrics_history['time'].append(current_time)
        self.metrics_history['rescues'].append(rescue_rate)
        self.metrics_history['efficiency'].append(efficiency)
        self.metrics_history['agent_activity'].append(agent_activity)
        self.metrics_history['collaboration'].append(collaboration)
        self.metrics_history['environment_damage'].append(damage)
        
        # Keep only last 100 data points
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > 100:
                self.metrics_history[key] = self.metrics_history[key][-100:]
    
    def render_key_metrics(self, environment):
        """Render key performance metrics"""
        st.subheader("📊 Key Performance Indicators")
        
        if not environment:
            self._render_empty_metrics()
            return
        
        # Calculate current metrics
        rescued = sum(1 for c in environment.civilians if c.get('rescued', False))
        total_civilians = len(environment.civilians)
        rescue_rate = (rescued / total_civilians * 100) if total_civilians > 0 else 0
        
        # Create metric columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_metric_card(
                "Civilians Rescued",
                f"{rescued}/{total_civilians}",
                rescue_rate,
                "success" if rescue_rate > 75 else "warning" if rescue_rate > 50 else "danger"
            )
        
        with col2:
            efficiency = min(environment.step_count / max(1, rescued) if rescued > 0 else environment.step_count, 100)
            self._render_metric_card(
                "Operational Efficiency",
                f"{efficiency:.1f}",
                efficiency / 100,
                "success" if efficiency < 20 else "warning" if efficiency < 50 else "danger"
            )
        
        with col3:
            collaboration = len(environment.agents) * 10
            self._render_metric_card(
                "Team Collaboration",
                f"{collaboration}%",
                collaboration / 100,
                "success" if collaboration > 70 else "warning" if collaboration > 40 else "danger"
            )
        
        with col4:
            response_time = environment.step_count
            self._render_metric_card(
                "Response Time",
                f"{response_time} steps",
                min(response_time / 100, 1),
                "success" if response_time < 50 else "warning" if response_time < 100 else "danger"
            )
    
    def _render_metric_card(self, title: str, value: str, progress: float, style: str):
        """Render a single metric card"""
        # Map style to color
        color_map = {
            "success": "#00cc96",
            "warning": "#ffa726", 
            "danger": "#ff4b4b"
        }
        color = color_map.get(style, "#1f77b4")
        
        # Create custom HTML for metric card
        st.markdown(f"""
        <div class="metric-card {style}-metric" style="text-align: center;">
            <h3 style="margin: 0; font-size: 1.2rem; color: #333;">{title}</h3>
            <h2 style="margin: 10px 0; font-size: 2rem; color: {color};">{value}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(progress)
    
    def _render_empty_metrics(self):
        """Render placeholder when no metrics available"""
        col1, col2, col3, col4 = st.columns(4)
        
        for col in [col1, col2, col3, col4]:
            with col:
                st.metric("--", "--")
                st.progress(0)
    
    def render_performance_trends(self):
        """Render performance trend charts"""
        st.subheader("📈 Performance Trends")
        
        if len(self.metrics_history['time']) < 2:
            st.info("📊 Performance data will appear once simulation starts")
            return
        
        # Create tabs for different trend views
        tab1, tab2, tab3 = st.tabs(["Rescue Progress", "Agent Performance", "Environment Impact"])
        
        with tab1:
            self._render_rescue_trends()
        
        with tab2:
            self._render_agent_trends()
        
        with tab3:
            self._render_environment_trends()
    
    def _render_rescue_trends(self):
        """Render rescue-related trend charts"""
        # Rescue rate over time
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=self.metrics_history['time'],
            y=self.metrics_history['rescues'],
            mode='lines+markers',
            name='Rescue Rate',
            line=dict(color='green', width=3)
        ))
        fig1.update_layout(
            title="Rescue Rate Over Time",
            xaxis_title="Time",
            yaxis_title="Rescue Rate (%)",
            height=300
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Efficiency trend
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=self.metrics_history['time'],
            y=self.metrics_history['efficiency'],
            mode='lines',
            name='Operational Efficiency',
            line=dict(color='blue', width=2)
        ))
        fig2.update_layout(
            title="Operational Efficiency",
            xaxis_title="Time",
            yaxis_title="Efficiency Score",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    def _render_agent_trends(self):
        """Render agent-related trend charts"""
        # Agent activity
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=self.metrics_history['time'],
            y=self.metrics_history['agent_activity'],
            mode='lines+markers',
            name='Agent Activity',
            line=dict(color='orange', width=2)
        ))
        fig1.update_layout(
            title="Agent Activity Level",
            xaxis_title="Time",
            yaxis_title="Activity Score",
            height=300
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Collaboration
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=self.metrics_history['time'],
            y=self.metrics_history['collaboration'],
            mode='lines',
            name='Team Collaboration',
            line=dict(color='purple', width=2)
        ))
        fig2.update_layout(
            title="Team Collaboration",
            xaxis_title="Time",
            yaxis_title="Collaboration Score",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    def _render_environment_trends(self):
        """Render environment-related trend charts"""
        # Environmental damage
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.metrics_history['time'],
            y=self.metrics_history['environment_damage'],
            mode='lines+markers',
            name='Environmental Damage',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title="Environmental Damage Assessment",
            xaxis_title="Time",
            yaxis_title="Damage Index",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_agent_analytics(self, environment):
        """Render detailed agent analytics"""
        st.subheader("🤖 Agent Analytics")
        
        if not environment or not hasattr(environment, 'agents'):
            st.info("No agent data available")
            return
        
        # Agent performance summary
        agent_data = []
        for agent_id, agent in environment.agents.items():
            agent_type = type(agent).__name__.replace('Agent', '')
            performance_data = {
                'Agent ID': agent_id,
                'Type': agent_type,
                'Steps Taken': getattr(agent, 'steps_taken', 0),
                'Civilians Rescued': getattr(agent, 'civilians_rescued', 0),
                'Current Reward': getattr(agent, 'total_reward', 0),
                'Status': 'Active' if getattr(agent, 'active', True) else 'Inactive'
            }
            agent_data.append(performance_data)
        
        if agent_data:
            df = pd.DataFrame(agent_data)
            
            # Display agent table
            st.dataframe(
                df,
                width='stretch',
                hide_index=True
            )
            
            # Agent type distribution
            type_counts = df['Type'].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Agent Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agents in simulation")
    
    def render_environment_analytics(self, environment):
        """Render detailed environment analytics"""
        st.subheader("🗺️ Environment Analytics")
        
        if not environment:
            st.info("No environment data available")
            return
        
        # Environment composition
        if hasattr(environment, 'grid'):
            unique, counts = np.unique(environment.grid, return_counts=True)
            
            cell_data = []
            for val, count in zip(unique, counts):
                cell_name = "Unknown"
                if hasattr(environment, 'cell_types'):
                    for name, cell_val in environment.cell_types.items():
                        if cell_val == val:
                            cell_name = name.replace('_', ' ').title()
                            break
                
                percentage = (count / (environment.grid_size * environment.grid_size)) * 100
                cell_data.append({
                    'Cell Type': cell_name,
                    'Count': count,
                    'Percentage': percentage
                })
            
            df = pd.DataFrame(cell_data)
            
            # Display composition chart
            fig = px.bar(
                df,
                x='Cell Type',
                y='Count',
                title="Environment Composition",
                color='Cell Type'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display composition table
            st.dataframe(
                df.sort_values('Count', ascending=False),
                width='stretch',
                hide_index=True
            )
    
    def render_performance_summary(self, environment):
        """Render overall performance summary"""
        st.subheader("🏆 Performance Summary")
        
        if not environment:
            st.info("No performance data available")
            return
        
        # Calculate summary metrics
        total_steps = environment.step_count
        rescued = sum(1 for c in environment.civilians if c.get('rescued', False))
        total_civilians = len(environment.civilians)
        rescue_rate = (rescued / total_civilians * 100) if total_civilians > 0 else 0
        
        # Efficiency metrics
        avg_efficiency = np.mean(self.metrics_history['efficiency']) if self.metrics_history['efficiency'] else 0
        max_efficiency = np.max(self.metrics_history['efficiency']) if self.metrics_history['efficiency'] else 0
        
        # Create summary cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Simulation Time", f"{total_steps} steps")
        
        with col2:
            st.metric("Overall Rescue Rate", f"{rescue_rate:.1f}%")
        
        with col3:
            st.metric("Average Efficiency", f"{avg_efficiency:.1f}")
        
        # Performance rating
        performance_score = (rescue_rate * 0.6 + (100 - avg_efficiency) * 0.4) / 100
        
        if performance_score > 0.8:
            rating = "⭐️⭐️⭐️⭐️⭐️ Excellent"
            color = "green"
        elif performance_score > 0.6:
            rating = "⭐️⭐️⭐️⭐️ Good"
            color = "blue"
        elif performance_score > 0.4:
            rating = "⭐️⭐️⭐️ Average"
            color = "orange"
        else:
            rating = "⭐️⭐️ Needs Improvement"
            color = "red"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 10px;">
            <h3 style="color: {color}; margin: 0;">Performance Rating</h3>
            <h2 style="color: {color}; margin: 10px 0;">{rating}</h2>
            <p style="margin: 0;">Score: {performance_score:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics data for download"""
        return {
            'metrics_history': self.metrics_history,
            'summary': {
                'total_data_points': len(self.metrics_history['time']),
                'time_range': {
                    'start': self.metrics_history['time'][0] if self.metrics_history['time'] else None,
                    'end': self.metrics_history['time'][-1] if self.metrics_history['time'] else None
                }
            }
        }
    
    def clear_metrics(self):
        """Clear all metrics history"""
        for key in self.metrics_history:
            self.metrics_history[key] = []

# Test function
def test_metrics_display():
    """Test the metrics display component"""
    st.title("Metrics Display Test")
    
    metrics = MetricsDisplay()
    
    # Test with sample data
    class MockAgent:
        def __init__(self, steps, rescues):
            self.steps_taken = steps
            self.civilians_rescued = rescues
            self.total_reward = rescues * 10
            self.active = True
    
    class MockEnvironment:
        def __init__(self):
            self.step_count = 50
            self.civilians = [
                {'rescued': True}, {'rescued': True}, {'rescued': False}, {'rescued': False}
            ]
            self.agents = {
                'drone_1': MockAgent(30, 2),
                'ambulance_1': MockAgent(20, 1),
                'rescue_1': MockAgent(25, 0)
            }
            self.collapsed_buildings = [(1, 1), (2, 2)]
            self.blocked_roads = [(3, 3)]
            self.grid_size = 20
            self.cell_types = {
                'ROAD': 0,
                'BUILDING': 1,
                'HOSPITAL': 2
            }
            self.grid = np.random.choice([0, 1, 2], (20, 20))
    
    mock_env = MockEnvironment()
    
    # Update metrics
    metrics.update_metrics(mock_env, {})
    
    # Render all components
    metrics.render_key_metrics(mock_env)
    metrics.render_performance_trends()
    metrics.render_agent_analytics(mock_env)
    metrics.render_environment_analytics(mock_env)
    metrics.render_performance_summary(mock_env)
    
    st.success("✅ Metrics display components working!")

if __name__ == "__main__":
    test_metrics_display()