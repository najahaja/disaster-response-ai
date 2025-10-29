#!/usr/bin/env python3
"""
Main training script for Disaster Response AI
Week 4: Enhanced training with multiple frameworks
"""
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train import main

if __name__ == "__main__":
    main()