#!/usr/bin/env python3
"""
Test script for the animation system.
Demonstrates how to run a simulation with animation recording.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import Simulation
from htn_v3 import _build_units, _build_enemy_units
from ga_optimizer import build_friendly_units
from log import logger


def test_animation_system(sequence=None, seed=None):
    """Test the animation system with a short simulation run."""
    
    print("Testing Animation System")
    print("=" * 50)
    
    # Build units
    config = {
        "FriendlyTankGroup": 2,
        "FriendlyInfantryGroup": 3,
        "FriendlyAntiTankGroup": 1,
        "FriendlyArtilleryGroup": 1
    }
    
    friendly_units, enemy_units = _build_units(config)
    enemy_units = _build_enemy_units()
    
    print(f"Created {len(friendly_units)} friendly units and {len(enemy_units)} enemy units")
    
    # Create simulation with animation enabled
    sim = Simulation(
        friendly_units=friendly_units,
        enemy_units=enemy_units,
        visualize=False,  # Disable real-time visualization
        plan_name="Animation Test",
        enable_animation=True,  # Enable animation recording
        animation_dir="test_animations"
    )
    
    # Set attack sequence if provided
    if sequence:
        sim.attack_sequence = sequence
        print(f"Using custom attack sequence: {sequence}")
    
    # Set seed if provided
    if seed is not None:
        import random
        random.seed(seed)
        print(f"Using seed: {seed}")
    
    print("Starting simulation with animation recording...")
    
    # Run simulation for a limited number of steps
    result = sim.run(max_steps=50)
    
    print(f"Simulation completed!")
    print(f"Result: {result}")
    
    # Check if animation was created
    if sim.animator:
        animation_dir = sim.animator.run_dir
        print(f"\nAnimation saved to: {animation_dir}")
        print(f"Open {animation_dir / 'animation.html'} in your browser to view the animation")
        
        # List files created
        print(f"\nFiles created:")
        for file_path in animation_dir.rglob("*"):
            if file_path.is_file():
                print(f"  {file_path.relative_to(animation_dir)}")
    
    return result


def test_visualization_with_animation(sequence=None, seed=None):
    """Test both visualization and animation together."""
    
    print("\nTesting Visualization + Animation")
    print("=" * 50)
    
    # Build units
    config = {
        "FriendlyTankGroup": 1,
        "FriendlyInfantryGroup": 2,
        "FriendlyAntiTankGroup": 1
    }
    
    friendly_units, enemy_units = _build_units(config)
    enemy_units = _build_enemy_units()
    
    # Create simulation with both visualization and animation
    sim = Simulation(
        friendly_units=friendly_units,
        enemy_units=enemy_units,
        visualize=True,  # Enable real-time visualization
        plan_name="Visual + Animation Test",
        enable_animation=True,  # Also enable animation recording
        animation_dir="test_animations"
    )
    
    # Set attack sequence if provided
    if sequence:
        sim.attack_sequence = sequence
        print(f"Using custom attack sequence: {sequence}")
    
    # Set seed if provided
    if seed is not None:
        import random
        random.seed(seed)
        print(f"Using seed: {seed}")
    
    print("Starting simulation with both visualization and animation...")
    print("Press 'p' to pause/resume visualization")
    
    # Run simulation for a limited number of steps
    result = sim.run(max_steps=30)
    
    print(f"Simulation completed!")
    print(f"Result: {result}")
    
    if sim.animator:
        animation_dir = sim.animator.run_dir
        print(f"\nAnimation saved to: {animation_dir}")
        print(f"Open {animation_dir / 'animation.html'} in your browser to view the animation")
    
    return result


if __name__ == "__main__":
    try:
        # Test 1: Animation only
        print("Test 1: Animation Only")
        result1 = test_animation_system()
        
        # Test 2: Visualization + Animation
        print("\nTest 2: Visualization + Animation")
        result2 = test_visualization_with_animation()
        
        # Test 3: With custom sequence and seed
        print("\nTest 3: Custom Sequence and Seed")
        custom_sequence = [
            ["EnemyTankGroup1", "ConsolidateAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
            ["EnemyTankGroup2", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]]
        ]
        result3 = test_animation_system(sequence=custom_sequence, seed=12345)
        
        print("\nAll tests completed successfully!")
        print(f"Test 1 result: {result1}")
        print(f"Test 2 result: {result2}")
        print(f"Test 3 result: {result3}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
