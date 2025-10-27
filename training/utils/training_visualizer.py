import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List

class TrainingVisualizer:
    """
    Utility class for visualizing training progress and results
    """
    
    def __init__(self):
        self.fig = None
        self.axs = None
    
    def plot_training_curves(self, training_data: Dict[str, List[float]]):
        """
        Plot training curves (rewards, losses, etc.)
        """
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot episode rewards
        if 'episode_rewards' in training_data:
            self.axs[0, 0].plot(training_data['episode_rewards'])
            self.axs[0, 0].set_title('Episode Rewards')
            self.axs[0, 0].set_xlabel('Episode')
            self.axs[0, 0].set_ylabel('Total Reward')
            self.axs[0, 0].grid(True)
        
        # Plot episode lengths
        if 'episode_lengths' in training_data:
            self.axs[0, 1].plot(training_data['episode_lengths'])
            self.axs[0, 1].set_title('Episode Lengths')
            self.axs[0, 1].set_xlabel('Episode')
            self.axs[0, 1].set_ylabel('Steps')
            self.axs[0, 1].grid(True)
        
        # Plot civilian rescue rate
        if 'rescue_rates' in training_data:
            self.axs[1, 0].plot(training_data['rescue_rates'])
            self.axs[1, 0].set_title('Civilian Rescue Rate')
            self.axs[1, 0].set_xlabel('Episode')
            self.axs[1, 0].set_ylabel('Rescue Rate')
            self.axs[1, 0].grid(True)
        
        # Plot agent collaboration
        if 'collaboration_scores' in training_data:
            self.axs[1, 1].plot(training_data['collaboration_scores'])
            self.axs[1, 1].set_title('Agent Collaboration')
            self.axs[1, 1].set_xlabel('Episode')
            self.axs[1, 1].set_ylabel('Collaboration Score')
            self.axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_agent_performance(self, agent_data: Dict[str, Dict]):
        """
        Plot individual agent performance
        """
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        agent_names = list(agent_data.keys())
        
        # Plot rewards by agent
        rewards = [agent_data[name].get('total_reward', 0) for name in agent_names]
        axs[0, 0].bar(agent_names, rewards)
        axs[0, 0].set_title('Total Rewards by Agent')
        axs[0, 0].set_ylabel('Reward')
        
        # Plot civilians rescued by agent
        rescues = [agent_data[name].get('civilians_rescued', 0) for name in agent_names]
        axs[0, 1].bar(agent_names, rescues)
        axs[0, 1].set_title('Civilians Rescued by Agent')
        axs[0, 1].set_ylabel('Civilians Rescued')
        
        # Plot steps taken by agent
        steps = [agent_data[name].get('steps_taken', 0) for name in agent_names]
        axs[1, 0].bar(agent_names, steps)
        axs[1, 0].set_title('Steps Taken by Agent')
        axs[1, 0].set_ylabel('Steps')
        
        # Plot efficiency (civilians per step)
        efficiency = [r/s if s > 0 else 0 for r, s in zip(rescues, steps)]
        axs[1, 1].bar(agent_names, efficiency)
        axs[1, 1].set_title('Efficiency (Civilians/Step)')
        axs[1, 1].set_ylabel('Efficiency')
        
        plt.tight_layout()
        plt.show()
    
    def save_plots(self, filename: str):
        """Save current plots to file"""
        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved as {filename}")