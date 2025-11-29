#!/usr/bin/env python3
"""
Legend component for the Disaster Response AI dashboard.
Displays a colour legend mapping cell types to their visual colours.
"""
import streamlit as st
import yaml
import os

class Legend:
    """Render a horizontal colour legend based on config.yaml mappings."""

    def __init__(self, config_path: str = None):
        # Determine config file location
        if config_path is None:
            # Assume config.yaml is at project root (two levels up from this file)
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            config_path = os.path.join(base_dir, "config.yaml")
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)
        except Exception as e:
            st.error(f"❌ Failed to load config for legend: {e}")
            self.cfg = {}

    def render(self):
        """Render the legend using Streamlit components."""
        if not self.cfg:
            return
        # Get mappings
        cell_types = self.cfg.get("environment", {}).get("cell_types", {})
        colours = self.cfg.get("visualization", {}).get("colors", {})
        # Build ordered list of (label, rgb)
        items = []
        for name, idx in cell_types.items():
            # colours dict uses numeric keys as strings
            colour = colours.get(str(idx))
            if colour:
                items.append((name.title(), colour))
        if not items:
            return
        # Render horizontally
        st.markdown("**Map Legend**")
        cols = st.columns(len(items))
        for col, (label, rgb) in zip(cols, items):
            r, g, b = rgb
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            # Use HTML for colour swatch
            swatch_html = f"""
            <div style='display:flex;align-items:center'>
                <div style='width:20px;height:20px;background:{hex_color};border:1px solid #555;margin-right:5px'></div>
                <span>{label}</span>
            </div>
            """
            col.markdown(swatch_html, unsafe_allow_html=True)
