#!/usr/bin/env python3
"""
Example script showing how to use the animation system.
This demonstrates how to run simulations with animation recording.
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


def run_simulation_with_animation(sequence=None, seed=None):
    """
    Example: Run a simulation with animation recording.
    This will save images of every step and create a rewindable HTML animation.
    
    Args:
        sequence: Optional attack sequence to use
        seed: Optional random seed for reproducible results
    """
    
    print("Running Simulation with Animation")
    print("=" * 50)
    
    # Configure your friendly units
    config = {
        "FriendlyTankGroup": 2,        # 2 tank groups
        "FriendlyInfantryGroup": 3,    # 3 infantry groups  
        "FriendlyAntiTankGroup": 1,    # 1 anti-tank group
        "FriendlyArtilleryGroup": 1    # 1 artillery group
    }
    
    # Build units
    friendly_units, enemy_units = _build_units(config)
    enemy_units = _build_enemy_units()
    
    print(f"Created {len(friendly_units)} friendly units and {len(enemy_units)} enemy units")
    
    # Create simulation with animation enabled
    sim = Simulation(
        friendly_units=friendly_units,
        enemy_units=enemy_units,
        visualize=False,                    # Disable real-time visualization (faster)
        plan_name="Example Animation Run",   # Name for this run
        enable_animation=True,              # Enable animation recording
        animation_dir="example_animations"  # Directory to save animations
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
    
    # Run simulation
    result = sim.run(max_steps=100)
    
    print(f"Simulation completed!")
    print(f"Final Score: {result['score']}")
    print(f"Friendly Health: {result['health']}")
    print(f"Enemy Health: {result['enemy_health']}")
    print(f"Outpost Secured: {result['outpost_secured']}")
    print(f"Steps Taken: {result['steps_taken']}")
    
    # Animation information
    if sim.animator:
        animation_dir = sim.animator.run_dir
        print(f"\n🎬 Animation created!")
        print(f"📁 Location: {animation_dir}")
        print(f"🌐 Open in browser: {animation_dir / 'animation.html'}")
        print(f"📊 Total steps recorded: {sim.animator.metadata['total_steps']}")
        
        # List the files created
        print(f"\n📋 Files created:")
        for file_path in sorted(animation_dir.rglob("*")):
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"  {file_path.relative_to(animation_dir)} ({size:,} bytes)")
    
    return result


def run_simulation_with_visualization_and_animation(sequence=None, seed=None):
    """
    Example: Run a simulation with both real-time visualization and animation recording.
    This is useful for watching the simulation in real-time while also recording it.
    
    Args:
        sequence: Optional attack sequence to use
        seed: Optional random seed for reproducible results
    """
    
    print("\nRunning Simulation with Visualization + Animation")
    print("=" * 50)
    
    # Smaller configuration for faster demo
    config = {
        "FriendlyTankGroup": 1,
        "FriendlyInfantryGroup": 2,
        "FriendlyAntiTankGroup": 1
    }
    
    # Build units
    friendly_units, enemy_units = _build_units(config)
    enemy_units = _build_enemy_units()
    
    # Create simulation with both visualization and animation
    sim = Simulation(
        friendly_units=friendly_units,
        enemy_units=enemy_units,
        visualize=True,                     # Enable real-time visualization
        plan_name="Visual + Animation Demo", # Name for this run
        enable_animation=True,              # Also enable animation recording
        animation_dir="example_animations"  # Directory to save animations
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
    print("💡 Press 'p' to pause/resume the visualization")
    print("🎬 Animation is being recorded in the background")
    
    # Run simulation
    result = sim.run(max_steps=50)
    
    print(f"Simulation completed!")
    print(f"Final Score: {result['score']}")
    
    if sim.animator:
        animation_dir = sim.animator.run_dir
        print(f"\n🎬 Animation saved to: {animation_dir}")
        print(f"🌐 Open: {animation_dir / 'animation.html'}")
    
    return result


def run_multiple_animations(sequences=None, seeds=None):
    """
    Example: Run multiple simulations and compare their animations.
    
    Args:
        sequences: Optional list of attack sequences (one per config)
        seeds: Optional list of seeds (one per config)
    """
    
    print("\nRunning Multiple Animation Comparisons")
    print("=" * 50)
    
    # Different configurations to compare
    configs = [
        {
            "name": "Tank Heavy",
            "config": {
                "FriendlyTankGroup": 3,
                "FriendlyInfantryGroup": 1,
                "FriendlyAntiTankGroup": 1
            }
        },
        {
            "name": "Balanced",
            "config": {
                "FriendlyTankGroup": 1,
                "FriendlyInfantryGroup": 2,
                "FriendlyAntiTankGroup": 1,
                "FriendlyArtilleryGroup": 1
            }
        },
        {
            "name": "Infantry Heavy", 
            "config": {
                "FriendlyTankGroup": 1,
                "FriendlyInfantryGroup": 4,
                "FriendlyAntiTankGroup": 1
            }
        }
    ]
    
    results = []
    
    for i, config_info in enumerate(configs):
        print(f"\nRunning {config_info['name']} configuration...")
        
        # Build units
        friendly_units, enemy_units = _build_units(config_info['config'])
        enemy_units = _build_enemy_units()
        
        # Create simulation
        sim = Simulation(
            friendly_units=friendly_units,
            enemy_units=enemy_units,
            visualize=False,
            plan_name=config_info['name'],
            enable_animation=True,
            animation_dir="comparison_animations"
        )
        
        # Set attack sequence if provided
        if sequences and i < len(sequences):
            sim.attack_sequence = sequences[i]
            print(f"  Using custom attack sequence: {sequences[i]}")
        
        # Set seed if provided
        if seeds and i < len(seeds):
            import random
            random.seed(seeds[i])
            print(f"  Using seed: {seeds[i]}")
        
        # Run simulation
        result = sim.run(max_steps=75)
        result['config_name'] = config_info['name']
        results.append(result)
        
        print(f"  Score: {result['score']}")
        print(f"  Animation: {sim.animator.run_dir / 'animation.html'}")
    
    # Summary
    print(f"\n📊 Comparison Summary:")
    print(f"{'Configuration':<15} {'Score':<8} {'Health':<8} {'Steps':<8}")
    print("-" * 45)
    for result in results:
        print(f"{result['config_name']:<15} {result['score']:<8.1f} {result['health']:<8} {result['steps_taken']:<8}")
    
    return results


def run_custom_sequence_and_seed_example():
    """
    Example: Run simulations with custom attack sequences and specific seeds.
    This demonstrates how to use the sequence and seed parameters.
    """
    
    print("\n🎯 Custom Sequence and Seed Example")
    print("=" * 50)
    
    # Example attack sequence from your robustness.json
    custom_sequence = [
        ["EnemyTankGroup1", "ConsolidateAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
        ["EnemyTankGroup2", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
        ["EnemyInfantryGroup1", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
        ["EnemyAntiTankGroup1", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]]
    ]
    
    # Test with different seeds
    test_seeds = [12345, 67890, 11111]
    
    print("Testing the same sequence with different seeds...")
    
    for i, seed in enumerate(test_seeds):
        print(f"\n--- Test {i+1}: Seed {seed} ---")
        
        # Run with custom sequence and seed
        result = run_simulation_with_animation(
            sequence=custom_sequence,
            seed=seed
        )
        
        print(f"Result with seed {seed}: Score={result['score']}, Health={result['health']}")
    
    print(f"\n✅ Custom sequence and seed testing completed!")
    print(f"💡 Notice how different seeds can produce different results even with the same sequence!")


def run_sequence_comparison():
    """
    Example: Compare different attack sequences with the same seed.
    """
    
    print("\n🔄 Sequence Comparison Example")
    print("=" * 50)
    
    # Different attack sequences to compare
    sequences = [
        # Sequence 1: All ConsolidateAttack
        [
            ["EnemyTankGroup1", "ConsolidateAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
            ["EnemyTankGroup2", "ConsolidateAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
            ["EnemyInfantryGroup1", "ConsolidateAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
            ["EnemyAntiTankGroup1", "ConsolidateAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]]
        ],
        # Sequence 2: All FlankAttack
        [
            ["EnemyTankGroup1", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
            ["EnemyTankGroup2", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
            ["EnemyInfantryGroup1", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
            ["EnemyAntiTankGroup1", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]]
        ],
        # Sequence 3: Mixed approach
        [
            ["EnemyTankGroup1", "ConsolidateAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
            ["EnemyTankGroup2", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
            ["EnemyInfantryGroup1", "ConsolidateAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
            ["EnemyAntiTankGroup1", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]]
        ]
    ]
    
    sequence_names = ["All Consolidate", "All Flank", "Mixed Strategy"]
    fixed_seed = 99999  # Use same seed for fair comparison
    
    print(f"Comparing {len(sequences)} different attack sequences with seed {fixed_seed}")
    
    results = []
    
    for i, (sequence, name) in enumerate(zip(sequences, sequence_names)):
        print(f"\n--- {name} Strategy ---")
        
        # Create a custom plan name
        plan_name = f"Sequence_Comparison_{name.replace(' ', '_')}"
        
        # Configure units
        config = {
            "FriendlyTankGroup": 1,
            "FriendlyInfantryGroup": 2,
            "FriendlyAntiTankGroup": 1,
            "FriendlyArtilleryGroup": 1
        }
        
        # Build units
        friendly_units, enemy_units = _build_units(config)
        enemy_units = _build_enemy_units()
        
        # Create simulation
        sim = Simulation(
            friendly_units=friendly_units,
            enemy_units=enemy_units,
            visualize=False,
            plan_name=plan_name,
            enable_animation=True,
            animation_dir="sequence_comparison_animations"
        )
        
        # Set custom sequence and seed
        sim.attack_sequence = sequence
        import random
        random.seed(fixed_seed)
        
        print(f"Using sequence: {sequence}")
        print(f"Using seed: {fixed_seed}")
        
        # Run simulation
        result = sim.run(max_steps=75)
        result['sequence_name'] = name
        results.append(result)
        
        print(f"Result: Score={result['score']}, Health={result['health']}, Steps={result['steps_taken']}")
        print(f"Animation: {sim.animator.run_dir / 'animation.html'}")
    
    # Summary
    print(f"\n📊 Sequence Comparison Summary:")
    print(f"{'Strategy':<15} {'Score':<8} {'Health':<8} {'Steps':<8}")
    print("-" * 45)
    for result in results:
        print(f"{result['sequence_name']:<15} {result['score']:<8.1f} {result['health']:<8} {result['steps_taken']:<8}")
    
    return results


if __name__ == "__main__":
    try:
        print("🎬 Animation System Examples")
        print("=" * 50)
        
        # Example 1: Basic animation
        print("\n1️⃣ Basic Animation Example")
        result1 = run_simulation_with_animation()
        
        # Example 2: Visualization + Animation
        print("\n2️⃣ Visualization + Animation Example")
        result2 = run_simulation_with_visualization_and_animation()
        
        # Example 3: Multiple comparisons
        print("\n3️⃣ Multiple Animation Comparison")
        results = run_multiple_animations()
        
        # Example 4: Custom sequence and seed
        print("\n4️⃣ Custom Sequence and Seed Example")
        run_custom_sequence_and_seed_example()
        
        # Example 5: Sequence comparison
        print("\n5️⃣ Sequence Comparison Example")
        sequence_results = run_sequence_comparison()
        
        print(f"\n✅ All examples completed successfully!")
        print(f"\n💡 Tips:")
        print(f"   • Open the HTML files in your browser to view animations")
        print(f"   • Use the controls to play, pause, rewind, and adjust speed")
        print(f"   • Check the metadata.json files for detailed simulation data")
        print(f"   • Images are saved in the 'images' subdirectory")
        print(f"   • Use custom sequences and seeds for reproducible experiments")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
