#!/usr/bin/env python3
"""
Main training script for Disaster Response AI - Week 2
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train import TrainingManager

def main():
    """Main training function"""
    print("🚀 Starting Disaster Response AI Training - Week 2")
    print("=" * 60)
    
    # Initialize training manager
    trainer = TrainingManager("training/configs/base_config.yaml")
    
    print("🤖 Training Options:")
    print("1. Custom Training (Basic)")
    print("2. Stable-Baselines3 Training (Advanced)")
    print("3. Evaluate Existing Model")
    
    choice = input("🎯 Choose training method (1-3): ").strip()
    
    if choice == "1":
        print("\n🎯 Starting Custom Training...")
        trainer.train_custom(num_episodes=100)  # Reduced for testing
        
    elif choice == "2":
        print("\n🎯 Starting Stable-Baselines3 Training...")
        trainer.train_with_sb3()
        
    elif choice == "3":
        print("\n🎯 Starting Evaluation...")
        model_name = input("Enter model name: ").strip()
        trainer.load_model(model_name)
        trainer.evaluate()
        
    else:
        print("❌ Invalid choice")
        return
    
    # Visualize results
    print("\n📊 Generating training visualizations...")
    trainer.visualize_training()
    
    # Save model
    if trainer.model is not None:
        save_choice = input("💾 Save model? (y/n): ").strip().lower()
        if save_choice == 'y':
            model_name = input("Enter model name: ").strip()
            trainer.save_model(model_name)
    
    print("\n🎉 Week 2 Training Completed!")
    print("🤖 Your agents are now learning to coordinate!")
    print("🚀 Ready for Week 3: Advanced Features!")

if __name__ == "__main__":
    main()