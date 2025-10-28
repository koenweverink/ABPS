#!/usr/bin/env python3
"""
Script for running simulations with custom attack sequences and specific seeds.
This allows you to easily test specific scenarios and reproduce results.
"""

import os
import sys
import json
import argparse

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from log import logger


def load_sequence_from_json(json_file, sequence_key="sequence"):
    """
    Load an attack sequence from a JSON file.
    
    Args:
        json_file: Path to JSON file
        sequence_key: Key in JSON that contains the sequence
    
    Returns:
        Attack sequence list
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if sequence_key in data:
            return data[sequence_key]
        else:
            print(f"Warning: Key '{sequence_key}' not found in {json_file}")
            print(f"Available keys: {list(data.keys())}")
            return None
    except Exception as e:
        print(f"Error loading sequence from {json_file}: {e}")
        return None


def run_custom_simulation(config, sequence=None, seed=None, max_steps=100,
                         visualize=False, plan_name="Roster B With Time Penalty",
                         animation_dir="custom_animations"):
    """
    Run a simulation with custom parameters.

    Args:
        config: Unit configuration dictionary
        sequence: Optional attack sequence
        seed: Optional random seed
        max_steps: Maximum simulation steps
        visualize: Whether to show real-time visualization
        plan_name: Name for this simulation run
        animation_dir: Directory to save animations

    Returns:
        Simulation result dictionary
    """

    print(f"Running Custom Simulation: {plan_name}")
    print("=" * 60)

    import random
    import numpy as np

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"Using seed: {seed}")
    else:
        print("Using random seed")

    from simulation import Simulation
    from htn_v3 import _build_units, _build_enemy_units

    # Build units
    friendly_units, enemy_units = _build_units(config)
    enemy_units = _build_enemy_units()
    
    print(f"Unit Configuration:")
    for unit_type, count in config.items():
        print(f"   {unit_type}: {count}")
    print(f"   Total friendly units: {len(friendly_units)}")
    print(f"   Total enemy units: {len(enemy_units)}")
    
    # Create simulation
    sim = Simulation(
        friendly_units=friendly_units,
        enemy_units=enemy_units,
        visualize=visualize,
        plan_name=plan_name,
        enable_animation=True,
        animation_dir=animation_dir
    )
    
    # Set attack sequence if provided
    if sequence:
        sim.attack_sequence = sequence
        print(f"Attack Sequence:")
        for i, (enemy, attack_type, units) in enumerate(sequence):
            print(f"   {i+1}. {enemy} -> {attack_type} with {units}")
    else:
        print("Using default attack sequence")
    
    print(f"Max steps: {max_steps}")
    print(f"Visualization: {'Enabled' if visualize else 'Disabled'}")
    print()
    
    # Run simulation
    print("Starting simulation...")
    result = sim.run(max_steps=max_steps)
    
    # Display results
    print(f"\nSimulation Results:")
    print(f"   Score: {result['score']:.1f}")
    print(f"   Friendly Health: {result['health']}")
    print(f"   Enemy Health: {result['enemy_health']}")
    print(f"   Outpost Secured: {result['outpost_secured']}")
    print(f"   Steps Taken: {result['steps_taken']}")
    
    # Animation information
    if sim.animator:
        animation_dir = sim.animator.run_dir
        print(f"\nAnimation created!")
        print(f"   Location: {animation_dir}")
        print(f"   Open in browser: {animation_dir / 'animation.html'}")
        print(f"   Total steps recorded: {sim.animator.metadata['total_steps']}")
    
    return result


def main():
    """Main function with command line argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Run simulation animations with custom sequences and seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_custom_animation.py
  
  # Run with specific seed
  python run_custom_animation.py --seed 12345
  
  # Run with sequence from JSON file
  python run_custom_animation.py --sequence-file sequence_ex1.json
  
  # Run with custom unit configuration
  python run_custom_animation.py --tanks 3 --infantry 2 --anti-tank 1
  
  # Run with visualization enabled
  python run_custom_animation.py --visualize --seed 99999
  
  # Run with custom sequence and seed
  python run_custom_animation.py --sequence-file sequence_ex1.json --seed 12345 --max-steps 150
        """
    )
    
    # Unit configuration arguments
    parser.add_argument('--tanks', type=int, default=2, 
                       help='Number of tank groups (default: 2)')
    parser.add_argument('--infantry', type=int, default=3,
                       help='Number of infantry groups (default: 3)')
    parser.add_argument('--anti-tank', type=int, default=1,
                       help='Number of anti-tank groups (default: 1)')
    parser.add_argument('--artillery', type=int, default=1,
                       help='Number of artillery groups (default: 1)')
    
    # Simulation parameters
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible results')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum simulation steps (default: 100)')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable real-time visualization')
    
    # Sequence options
    parser.add_argument('--sequence-file', type=str, default=None,
                       help='JSON file containing attack sequence')
    parser.add_argument('--sequence-key', type=str, default='sequence',
                       help='Key in JSON file containing sequence (default: "sequence")')
    
    # Output options
    parser.add_argument('--plan-name', type=str, default='Roster B With Time Penalty',
                       help='Name for this simulation run')
    parser.add_argument('--animation-dir', type=str, default='custom_animations',
                       help='Directory to save animations')
    
    args = parser.parse_args()
    
    try:
        # Build unit configuration
        config = {
            "FriendlyTankGroup": args.tanks,
            "FriendlyInfantryGroup": args.infantry,
            "FriendlyAntiTankGroup": args.anti_tank,
            "FriendlyArtilleryGroup": args.artillery
        }
        
        # Load sequence if provided
        sequence = None
        if args.sequence_file:
            sequence = load_sequence_from_json(args.sequence_file, args.sequence_key)
            if sequence is None:
                print(f"Failed to load sequence from {args.sequence_file}")
                return 1
        
        # Run simulation
        result = run_custom_simulation(
            config=config,
            sequence=sequence,
            seed=args.seed,
            max_steps=args.max_steps,
            visualize=args.visualize,
            plan_name=args.plan_name,
            animation_dir=args.animation_dir
        )
        
        print(f"\nSimulation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_quick_examples():
    """Run some quick examples to demonstrate the functionality."""
    
    print("Quick Examples")
    print("=" * 50)
    
    # Example 1: Default run
    print("\nDefault Configuration")
    config1 = {
        "FriendlyTankGroup": 2,
        "FriendlyInfantryGroup": 3,
        "FriendlyAntiTankGroup": 1,
        "FriendlyArtilleryGroup": 1
    }
    result1 = run_custom_simulation(config1, plan_name="Default Config")
    
    # Example 2: With specific seed
    print("\nWith Specific Seed")
    result2 = run_custom_simulation(
        config1, 
        seed=12345, 
        plan_name="Seed 12345"
    )
    
    # Example 3: Different configuration
    print("\nTank Heavy Configuration")
    config3 = {
        "FriendlyTankGroup": 4,
        "FriendlyInfantryGroup": 1,
        "FriendlyAntiTankGroup": 1,
        "FriendlyArtilleryGroup": 0
    }
    result3 = run_custom_simulation(
        config3, 
        seed=12345,  # Same seed for comparison
        plan_name="Tank Heavy"
    )
    
    # Summary
    print(f"\nQuick Examples Summary:")
    print(f"{'Configuration':<15} {'Score':<8} {'Health':<8} {'Steps':<8}")
    print("-" * 45)
    print(f"{'Default Config':<15} {result1['score']:<8.1f} {result1['health']:<8} {result1['steps_taken']:<8}")
    print(f"{'Seed 12345':<15} {result2['score']:<8.1f} {result2['health']:<8} {result2['steps_taken']:<8}")
    print(f"{'Tank Heavy':<15} {result3['score']:<8.1f} {result3['health']:<8} {result3['steps_taken']:<8}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run quick examples
        run_quick_examples()
    else:
        # Run with command line arguments
        exit_code = main()
        sys.exit(exit_code)
