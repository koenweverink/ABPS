# Simulation Animation System

This system automatically saves images of every step during simulation runs and creates rewindable HTML animations that you can view in your browser.

## Features

- **Automatic Image Capture**: Saves a high-quality image of every simulation step
- **Rewindable HTML Animation**: Interactive web-based viewer with play/pause/rewind controls
- **Real-time Data Display**: Shows unit health, positions, and simulation state for each step
- **Keyboard Controls**: Use arrow keys, spacebar, and other shortcuts for navigation
- **Multiple Animation Support**: Can run multiple simulations and compare their animations
- **Metadata Export**: Saves detailed simulation data and results

## Quick Start

### Basic Usage

```python
from simulation import Simulation
from htn_v3 import _build_units, _build_enemy_units

# Configure your units
config = {
    "FriendlyTankGroup": 2,
    "FriendlyInfantryGroup": 3,
    "FriendlyAntiTankGroup": 1,
    "FriendlyArtilleryGroup": 1
}

# Build units
friendly_units, enemy_units = _build_units(config)
enemy_units = _build_enemy_units()

# Create simulation with animation enabled
sim = Simulation(
    friendly_units=friendly_units,
    enemy_units=enemy_units,
    visualize=False,                    # Disable real-time visualization (faster)
    plan_name="My Animation Run",       # Name for this run
    enable_animation=True,              # Enable animation recording
    animation_dir="my_animations"       # Directory to save animations
)

# Optional: Set custom attack sequence
custom_sequence = [
    ["EnemyTankGroup1", "ConsolidateAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
    ["EnemyTankGroup2", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]]
]
sim.attack_sequence = custom_sequence

# Optional: Set specific seed for reproducible results
import random
random.seed(12345)

# Run simulation
result = sim.run(max_steps=100)

# Animation is automatically saved!
print(f"Animation saved to: {sim.animator.run_dir}")
print(f"Open in browser: {sim.animator.run_dir / 'animation.html'}")
```

### With Real-time Visualization

```python
# Enable both visualization and animation
sim = Simulation(
    friendly_units=friendly_units,
    enemy_units=enemy_units,
    visualize=True,                     # Show real-time visualization
    plan_name="Visual + Animation",     # Name for this run
    enable_animation=True,              # Also record animation
    animation_dir="my_animations"       # Directory to save animations
)

# Run simulation (you can watch it in real-time AND record it)
result = sim.run(max_steps=100)
```

## Animation Viewer Features

The HTML animation viewer includes:

### Controls
- **Play/Pause**: Click the play button or press spacebar
- **Step Navigation**: Use Previous/Next buttons or arrow keys
- **Jump to Start/End**: Use First/Last buttons or Home/End keys
- **Speed Control**: Adjust playback speed with the slider
- **Loop Mode**: Enable/disable looping at the end

### Information Display
- **Step Counter**: Current step and total steps
- **Unit Health**: Real-time friendly and enemy health
- **Unit Counts**: Number of alive friendly and enemy units
- **Mission Status**: Whether the outpost was secured
- **Step Details**: Detailed information for the current step

### Keyboard Shortcuts
- `Space`: Play/Pause
- `←` / `→`: Previous/Next step
- `Home`: Jump to first step
- `End`: Jump to last step

## File Structure

Each animation run creates a directory with:

```
animation_run_name/
├── animation.html          # Main animation viewer
├── metadata.json          # Simulation metadata and results
├── summary.txt            # Text summary of the run
└── images/                # Individual step images
    ├── step_0001.png
    ├── step_0002.png
    ├── step_0003.png
    └── ...
```

## Examples

### Run the Example Scripts

```bash
# Basic animation test
python test_animation.py

# Comprehensive examples
python example_animation_usage.py

# Custom sequence and seed examples
python run_custom_animation.py
```

### Command Line Usage

```bash
# Run with default settings
python run_custom_animation.py

# Run with specific seed
python run_custom_animation.py --seed 12345

# Run with sequence from JSON file
python run_custom_animation.py --sequence-file robustness.json

# Run with custom unit configuration
python run_custom_animation.py --tanks 3 --infantry 2 --anti-tank 1

# Run with visualization enabled
python run_custom_animation.py --visualize --seed 99999

# Run with custom sequence and seed
python run_custom_animation.py --sequence-file robustness.json --seed 12345 --max-steps 150
```

### Example Output

```
🎬 Animation System Examples
==================================================

1️⃣ Basic Animation Example
Created 7 friendly units and 4 enemy units
Starting simulation with animation recording...
Saved step 1 image: step_0001.png
Saved step 2 image: step_0002.png
...
Simulation completed!
Final Score: -20.0
Friendly Health: 606
Enemy Health: 144
Outpost Secured: False
Steps Taken: 50

🎬 Animation created!
📁 Location: example_animations/Example Animation Run
🌐 Open in browser: example_animations/Example Animation Run/animation.html
📊 Total steps recorded: 51
```

## Configuration Options

### Simulation Parameters

- `visualize`: Enable/disable real-time visualization window
- `enable_animation`: Enable/disable animation recording
- `animation_dir`: Directory to save animation files
- `plan_name`: Name for this animation run (used in filenames)

### Animation Settings

The animation system automatically configures:
- **Image Format**: PNG with high quality (150 DPI)
- **Frame Rate**: 2 FPS (adjustable in viewer)
- **Image Size**: 15x8 inches (optimized for web viewing)
- **Auto-play**: Enabled by default
- **Loop**: Enabled by default

## Advanced Usage

### Custom Attack Sequences

```python
# Define a custom attack sequence
custom_sequence = [
    ["EnemyTankGroup1", "ConsolidateAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
    ["EnemyTankGroup2", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
    ["EnemyInfantryGroup1", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]],
    ["EnemyAntiTankGroup1", "FlankAttack", ["FriendlyAntiTankGroup_0", "FriendlyArtilleryGroup_0"]]
]

# Apply the sequence to simulation
sim.attack_sequence = custom_sequence
```

### Reproducible Results with Seeds

```python
# Set a specific seed for reproducible results
import random
random.seed(12345)

# Or use the seed in your simulation setup
sim = Simulation(
    friendly_units=friendly_units,
    enemy_units=enemy_units,
    enable_animation=True,
    plan_name="Reproducible Run"
)
# Set seed before running
random.seed(12345)
result = sim.run(max_steps=100)
```

### Loading Sequences from JSON Files

```python
import json

# Load sequence from JSON file
with open('robustness.json', 'r') as f:
    data = json.load(f)
    sequence = data['sequence']

# Use the loaded sequence
sim.attack_sequence = sequence
```

### Custom Animation Directory

```python
# Save animations to a specific location
sim = Simulation(
    friendly_units=friendly_units,
    enemy_units=enemy_units,
    enable_animation=True,
    animation_dir="/path/to/my/animations"  # Custom directory
)
```

### Multiple Animation Comparison

```python
# Run multiple simulations and compare
configs = [
    {"name": "Tank Heavy", "config": {"FriendlyTankGroup": 3, "FriendlyInfantryGroup": 1}},
    {"name": "Balanced", "config": {"FriendlyTankGroup": 1, "FriendlyInfantryGroup": 2}},
    {"name": "Infantry Heavy", "config": {"FriendlyTankGroup": 1, "FriendlyInfantryGroup": 4}}
]

for config_info in configs:
    friendly_units, enemy_units = _build_units(config_info["config"])
    sim = Simulation(
        friendly_units=friendly_units,
        enemy_units=enemy_units,
        enable_animation=True,
        plan_name=config_info["name"],
        animation_dir="comparison_animations"
    )
    result = sim.run(max_steps=100)
```

## Troubleshooting

### Common Issues

1. **No images saved**: Make sure `enable_animation=True` and the simulation actually runs steps
2. **Animation not loading**: Check that all files are in the same directory and open `animation.html` in a web browser
3. **Slow performance**: Disable real-time visualization (`visualize=False`) for faster recording
4. **Large file sizes**: Reduce `max_steps` or adjust image quality settings in the animation system

### Performance Tips

- Use `visualize=False` for faster animation recording
- Limit `max_steps` for shorter animations
- Use smaller unit configurations for testing
- Close other applications to free up memory

## Technical Details

### Image Format
- **Format**: PNG
- **DPI**: 150 (high quality)
- **Size**: 15x8 inches
- **Compression**: Optimized for web viewing

### Browser Compatibility
- **Chrome**: Full support
- **Firefox**: Full support  
- **Safari**: Full support
- **Edge**: Full support

### File Sizes
- Typical step image: ~200-500 KB
- 100-step animation: ~20-50 MB total
- HTML viewer: ~50 KB

## Integration with Existing Code

The animation system is designed to be non-intrusive:

- **Backward Compatible**: Existing simulations work unchanged
- **Optional**: Only enabled when `enable_animation=True`
- **Non-blocking**: Doesn't interfere with simulation logic
- **Configurable**: Can be enabled/disabled per simulation

Simply add the animation parameters to your existing `Simulation()` calls to start recording animations!
