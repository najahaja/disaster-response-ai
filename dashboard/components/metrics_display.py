import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

class MetricsDisplay:
    """
    Component for displaying simulation metrics and analytics
    """
    
    def __init__(self):
        self.metrics_history = {
            'time': [],
            'civilians_rescued': [],
            'agents_active': [],
            'efficiency': []
        }
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics with new data"""
        current_time = len(self.metrics_history['time'])
        self.metrics_history['time'].append(current_time)
        
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def render_live_metrics(self):
        """Render live metrics cards"""
        st.subheader("📊 Live Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Civilians Rescued", 
                f"{self.metrics_history['civilians_rescued'][-1] if self.metrics_history['civilians_rescued'] else 0}",
                delta="+2" if len(self.metrics_history['civilians_rescued']) > 1 else None
            )
        
        with col2:
            st.metric(
                "Active Agents",
                f"{self.metrics_history['agents_active'][-1] if self.metrics_history['agents_active'] else 0}"
            )
        
        with col3:
            st.metric(
                "Efficiency Score",
                f"{self.metrics_history['efficiency'][-1] if self.metrics_history['efficiency'] else 0}%"
            )
        
        with col4:
            st.metric(
                "Time Elapsed",
                f"{len(self.metrics_history['time'])}s"
            )
    
    def render_analytics(self):
        """Render analytics charts"""
        st.subheader("📈 Performance Analytics")
        
        if len(self.metrics_history['time']) > 1:
            # Create analytics tabs
            tab1, tab2, tab3 = st.tabs(["Rescue Progress", "Agent Performance", "System Efficiency"])
            
            with tab1:
                self.render_rescue_analytics()
            
            with tab2:
                self.render_agent_analytics()
            
            with tab3:
                self.render_efficiency_analytics()
        else:
            st.info("📊 Analytics will appear here once simulation data is available")
    
    def render_rescue_analytics(self):
        """Render rescue-related analytics"""
        if len(self.metrics_history['time']) > 1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.metrics_history['time'],
                y=self.metrics_history['civilians_rescued'],
                mode='lines+markers',
                name='Civilians Rescued',
                line=dict(color='green', width=3)
            ))
            
            fig.update_layout(
                title='Rescue Progress Over Time',
                xaxis_title='Time (seconds)',
                yaxis_title='Civilians Rescued',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_agent_analytics(self):
        """Render agent performance analytics"""
        # Sample agent data - replace with real data
        agent_data = {
            'Agent': ['Drone 1', 'Drone 2', 'Ambulance', 'Rescue Team'],
            'Missions': [8, 6, 12, 9],
            'Success Rate': [87, 92, 95, 88],
            'Efficiency': [85, 78, 92, 86]
        }
        
        df = pd.DataFrame(agent_data)
        
        fig = px.bar(df, x='Agent', y=['Missions', 'Success Rate', 'Efficiency'],
                    title='Agent Performance Metrics', barmode='group')
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_efficiency_analytics(self):
        """Render system efficiency analytics"""
        if len(self.metrics_history['efficiency']) > 1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.metrics_history['time'],
                y=self.metrics_history['efficiency'],
                mode='lines',
                name='System Efficiency',
                line=dict(color='blue', width=3)
            ))
            
            fig.update_layout(
                title='System Efficiency Over Time',
                xaxis_title='Time (seconds)',
                yaxis_title='Efficiency (%)',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_communication_log(self, communication_data: List[Dict]):
        """Render communication log"""
        st.subheader("📡 Agent Communication")
        
        if communication_data:
            # Convert to DataFrame for nice display
            comm_df = pd.DataFrame(communication_data)
            st.dataframe(comm_df, use_container_width=True)
        else:
            st.info("No communication data available yet")
    
    def render_event_log(self, event_data: List[Dict]):
        """Render event log"""
        st.subheader("📋 Event Log")
        
        if event_data:
            for event in event_data[-10:]:  # Show last 10 events
                event_type = event.get('type', 'Unknown')
                description = event.get('description', 'No description')
                step = event.get('step', 0)
                
                if event_type == 'aftershock':
                    icon = "💥"
                    color = "red"
                elif event_type == 'resource_depletion':
                    icon = "⚡"
                    color = "orange"
                elif event_type == 'road_closure':
                    icon = "🚧"
                    color = "yellow"
                elif event_type == 'civilian_discovery':
                    icon = "👤"
                    color = "green"
                else:
                    icon = "ℹ️"
                    color = "blue"
                
                st.markdown(
                    f"<div style='background-color: {color}20; padding: 10px; "
                    f"border-radius: 5px; margin: 5px 0;'>"
                    f"{icon} <b>Step {step}:</b> {description}"
                    f"</div>", 
                    unsafe_allow_html=True
                )
        else:
            st.info("No events recorded yet")