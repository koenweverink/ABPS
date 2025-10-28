"""
Animation system for simulation runs.
Saves images of every step and creates rewindable animations.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import base64
from io import BytesIO


class SimulationAnimator:
    """
    Handles saving images of every simulation step and creating rewindable animations.
    """
    
    def __init__(self, sim, output_dir: str = "animations", run_name: Optional[str] = None):
        """
        Initialize the animation system.
        
        Args:
            sim: Simulation instance
            output_dir: Directory to save animation files
            run_name: Name for this animation run (auto-generated if None)
        """
        self.sim = sim
        self.output_dir = Path(output_dir)
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.output_dir / self.run_name
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.run_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        # Animation metadata
        self.metadata = {
            "run_name": self.run_name,
            "start_time": datetime.now().isoformat(),
            "total_steps": 0,
            "simulation_config": {},
            "step_data": []
        }
        
        # Image saving settings
        self.image_format = "png"
        self.image_dpi = 150
        self.image_quality = 95
        
        # Animation settings
        self.frame_rate = 2  # frames per second
        self.auto_play = True
        self.loop = True
        
        print(f"Animation system initialized: {self.run_dir}")
    
    def save_step_image(self, step: int, additional_data: Optional[Dict[str, Any]] = None):
        """
        Save an image of the current simulation state.
        
        Args:
            step: Current step number
            additional_data: Additional data to save with this step
        """
        if not hasattr(self.sim, 'plotter') or not self.sim.plotter:
            print("Warning: No plotter available for animation")
            return
        
        # Update the plotter to ensure it's current
        self.sim.plotter.update()
        
        # Save the current figure
        filename = f"step_{step:04d}.{self.image_format}"
        filepath = self.images_dir / filename
        
        # Save with high quality
        self.sim.plotter.fig.savefig(
            filepath,
            dpi=self.image_dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            format=self.image_format
        )
        
        # Store step metadata
        step_info = {
            "step": step,
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "simulation_data": {
                "friendly_health": sum(u.state.get("health", 0) for u in self.sim.friendly_units),
                "enemy_health": sum(e.state.get("health", 0) for e in self.sim.enemy_units if e.state.get("enemy_alive", False)),
                "alive_friendlies": len([u for u in self.sim.friendly_units if u.state.get("health", 0) > 0]),
                "alive_enemies": len([e for e in self.sim.enemy_units if e.state.get("enemy_alive", False)]),
                "outpost_secured": any(u.state.get("outpost_secured", False) for u in self.sim.friendly_units)
            }
        }
        
        if additional_data:
            step_info.update(additional_data)
        
        self.metadata["step_data"].append(step_info)
        self.metadata["total_steps"] = step + 1
        
        print(f"Saved step {step} image: {filename}")
    
    def finalize_animation(self, simulation_result: Optional[Dict[str, Any]] = None):
        """
        Finalize the animation and create the viewer.
        
        Args:
            simulation_result: Final simulation results
        """
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["simulation_result"] = simulation_result or {}
        
        # Save metadata
        metadata_file = self.run_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Create HTML animation viewer
        self._create_html_viewer()
        
        # Create a summary
        self._create_summary()
        
        # Generate Gantt chart
        self._create_gantt_chart()
        
        print(f"Animation finalized: {self.run_dir}")
        print(f"Total steps: {self.metadata['total_steps']}")
        print(f"View animation: {self.run_dir / 'animation.html'}")
        print(f"Gantt chart: {self.run_dir / 'gantt_chart.png'}")
    
    def _create_html_viewer(self):
        """Create an HTML-based animation viewer with rewindable controls."""
        
        # Get list of image files
        image_files = sorted([f for f in self.images_dir.glob(f"*.{self.image_format}")])
        
        if not image_files:
            print("No images found to create animation")
            return
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulation Animation - {self.run_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .controls {{
            padding: 20px;
            background-color: #ecf0f1;
            border-bottom: 1px solid #bdc3c7;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        .control-group label {{
            font-weight: bold;
            min-width: 100px;
        }}
        button {{
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }}
        .play-btn {{
            background-color: #27ae60;
            color: white;
        }}
        .play-btn:hover {{
            background-color: #229954;
        }}
        .play-btn.playing {{
            background-color: #e74c3c;
        }}
        .play-btn.playing:hover {{
            background-color: #c0392b;
        }}
        .step-btn {{
            background-color: #3498db;
            color: white;
        }}
        .step-btn:hover {{
            background-color: #2980b9;
        }}
        .step-btn:disabled {{
            background-color: #bdc3c7;
            cursor: not-allowed;
        }}
        input[type="range"] {{
            flex: 1;
            min-width: 200px;
        }}
        .info-panel {{
            display: flex;
            gap: 20px;
            padding: 15px 20px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        .info-item {{
            flex: 1;
        }}
        .info-item h4 {{
            margin: 0 0 5px 0;
            color: #495057;
            font-size: 14px;
        }}
        .info-item .value {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .animation-container {{
            text-align: center;
            padding: 20px;
            background-color: white;
        }}
        .animation-image {{
            max-width: 100%;
            height: auto;
            border: 2px solid #dee2e6;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .step-info {{
            margin-top: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .loading {{
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Simulation Animation</h1>
            <p>{self.run_name} - {self.metadata['total_steps']} steps</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>Playback:</label>
                <button id="playBtn" class="play-btn">Play</button>
                <button id="prevBtn" class="step-btn">Previous</button>
                <button id="nextBtn" class="step-btn">Next</button>
                <button id="firstBtn" class="step-btn">First</button>
                <button id="lastBtn" class="step-btn">Last</button>
            </div>
            
            <div class="control-group">
                <label>Step:</label>
                <input type="range" id="stepSlider" min="0" max="{len(image_files)-1}" value="0">
                <span id="stepDisplay">0 / {len(image_files)-1}</span>
            </div>
            
            <div class="control-group">
                <label>Speed:</label>
                <input type="range" id="speedSlider" min="1" max="10" value="{self.frame_rate}">
                <span id="speedDisplay">{self.frame_rate} fps</span>
            </div>
            
            <div class="control-group">
                <label>Loop:</label>
                <input type="checkbox" id="loopCheckbox" {'checked' if self.loop else ''}>
            </div>
        </div>
        
        <div class="info-panel">
            <div class="info-item">
                <h4>Friendly Health</h4>
                <div class="value" id="friendlyHealth">-</div>
            </div>
            <div class="info-item">
                <h4>Enemy Health</h4>
                <div class="value" id="enemyHealth">-</div>
            </div>
            <div class="info-item">
                <h4>Alive Friendlies</h4>
                <div class="value" id="aliveFriendlies">-</div>
            </div>
            <div class="info-item">
                <h4>Alive Enemies</h4>
                <div class="value" id="aliveEnemies">-</div>
            </div>
            <div class="info-item">
                <h4>Outpost Secured</h4>
                <div class="value" id="outpostSecured">-</div>
            </div>
            <div class="info-item">
                <h4>Gantt Chart</h4>
                <div class="value"><a href="gantt_chart.png" target="_blank" style="color: #3498db; text-decoration: none;">View Chart</a></div>
            </div>
        </div>
        
        <div class="animation-container">
            <div id="loading" class="loading">Loading animation...</div>
            <img id="animationImage" class="animation-image" style="display: none;">
            <div id="stepInfo" class="step-info" style="display: none;"></div>
        </div>
    </div>

    <script>
        // Animation data
        const totalSteps = {len(image_files)};
        const stepData = {json.dumps(self.metadata['step_data'])};
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;
        let currentSpeed = {self.frame_rate};
        
        // DOM elements
        const playBtn = document.getElementById('playBtn');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const firstBtn = document.getElementById('firstBtn');
        const lastBtn = document.getElementById('lastBtn');
        const stepSlider = document.getElementById('stepSlider');
        const stepDisplay = document.getElementById('stepDisplay');
        const speedSlider = document.getElementById('speedSlider');
        const speedDisplay = document.getElementById('speedDisplay');
        const loopCheckbox = document.getElementById('loopCheckbox');
        const animationImage = document.getElementById('animationImage');
        const loading = document.getElementById('loading');
        const stepInfo = document.getElementById('stepInfo');
        
        // Info display elements
        const friendlyHealth = document.getElementById('friendlyHealth');
        const enemyHealth = document.getElementById('enemyHealth');
        const aliveFriendlies = document.getElementById('aliveFriendlies');
        const aliveEnemies = document.getElementById('aliveEnemies');
        const outpostSecured = document.getElementById('outpostSecured');
        
        // Initialize
        function init() {{
            loading.style.display = 'none';
            animationImage.style.display = 'block';
            stepInfo.style.display = 'block';
            updateDisplay();
            updateButtons();
        }}
        
        // Update display
        function updateDisplay() {{
            if (currentStep >= 0 && currentStep < totalSteps) {{
                const filename = `step_${{String(currentStep).padStart(4, '0')}}.{self.image_format}`;
                animationImage.src = `images/${{filename}}`;
                
                stepDisplay.textContent = `${{currentStep}} / ${{totalSteps - 1}}`;
                stepSlider.value = currentStep;
                
                // Update step info
                if (stepData[currentStep]) {{
                    const data = stepData[currentStep].simulation_data;
                    friendlyHealth.textContent = Math.round(data.friendly_health);
                    enemyHealth.textContent = Math.round(data.enemy_health);
                    aliveFriendlies.textContent = data.alive_friendlies;
                    aliveEnemies.textContent = data.alive_enemies;
                    outpostSecured.textContent = data.outpost_secured ? 'Yes' : 'No';
                    
                    stepInfo.innerHTML = `
                        <strong>Step ${{currentStep}}</strong><br>
                        Friendly Health: ${{Math.round(data.friendly_health)}}<br>
                        Enemy Health: ${{Math.round(data.enemy_health)}}<br>
                        Alive Friendlies: ${{data.alive_friendlies}}<br>
                        Alive Enemies: ${{data.alive_enemies}}<br>
                        Outpost Secured: ${{data.outpost_secured ? 'Yes' : 'No'}}
                    `;
                }}
            }}
        }}
        
        // Update button states
        function updateButtons() {{
            prevBtn.disabled = currentStep <= 0;
            nextBtn.disabled = currentStep >= totalSteps - 1;
            firstBtn.disabled = currentStep <= 0;
            lastBtn.disabled = currentStep >= totalSteps - 1;
        }}
        
        // Play/pause animation
        function togglePlay() {{
            if (isPlaying) {{
                pause();
            }} else {{
                play();
            }}
        }}
        
        function play() {{
            isPlaying = true;
            playBtn.textContent = 'Pause';
            playBtn.classList.add('playing');
            
            playInterval = setInterval(() => {{
                if (currentStep < totalSteps - 1) {{
                    currentStep++;
                    updateDisplay();
                    updateButtons();
                }} else if (loopCheckbox.checked) {{
                    currentStep = 0;
                    updateDisplay();
                    updateButtons();
                }} else {{
                    pause();
                }}
            }}, 1000 / currentSpeed);
        }}
        
        function pause() {{
            isPlaying = false;
            playBtn.textContent = 'Play';
            playBtn.classList.remove('playing');
            
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}
        
        // Navigation functions
        function goToStep(step) {{
            if (step >= 0 && step < totalSteps) {{
                currentStep = step;
                updateDisplay();
                updateButtons();
            }}
        }}
        
        function previousStep() {{
            if (currentStep > 0) {{
                goToStep(currentStep - 1);
            }}
        }}
        
        function nextStep() {{
            if (currentStep < totalSteps - 1) {{
                goToStep(currentStep + 1);
            }}
        }}
        
        function firstStep() {{
            goToStep(0);
        }}
        
        function lastStep() {{
            goToStep(totalSteps - 1);
        }}
        
        // Event listeners
        playBtn.addEventListener('click', togglePlay);
        prevBtn.addEventListener('click', previousStep);
        nextBtn.addEventListener('click', nextStep);
        firstBtn.addEventListener('click', firstStep);
        lastBtn.addEventListener('click', lastStep);
        
        stepSlider.addEventListener('input', (e) => {{
            goToStep(parseInt(e.target.value));
        }});
        
        speedSlider.addEventListener('input', (e) => {{
            currentSpeed = parseInt(e.target.value);
            speedDisplay.textContent = `${{currentSpeed}} fps`;
            
            if (isPlaying) {{
                pause();
                play();
            }}
        }});
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {{
            switch(e.key) {{
                case ' ':
                    e.preventDefault();
                    togglePlay();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    previousStep();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    nextStep();
                    break;
                case 'Home':
                    e.preventDefault();
                    firstStep();
                    break;
                case 'End':
                    e.preventDefault();
                    lastStep();
                    break;
            }}
        }});
        
        // Initialize when page loads
        window.addEventListener('load', init);
    </script>
</body>
</html>
        """
        
        # Write HTML file
        html_file = self.run_dir / "animation.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _create_summary(self):
        """Create a text summary of the animation."""
        summary_file = self.run_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Simulation Animation Summary\n")
            f.write(f"==========================\n\n")
            f.write(f"Run Name: {self.run_name}\n")
            f.write(f"Start Time: {self.metadata['start_time']}\n")
            f.write(f"End Time: {self.metadata['end_time']}\n")
            f.write(f"Total Steps: {self.metadata['total_steps']}\n\n")
            
            if self.metadata.get('simulation_result'):
                result = self.metadata['simulation_result']
                f.write(f"Simulation Results:\n")
                f.write(f"  Score: {result.get('score', 'N/A')}\n")
                f.write(f"  Final Friendly Health: {result.get('health', 'N/A')}\n")
                f.write(f"  Final Enemy Health: {result.get('enemy_health', 'N/A')}\n")
                f.write(f"  Outpost Secured: {result.get('outpost_secured', 'N/A')}\n")
                f.write(f"  Steps Taken: {result.get('steps_taken', 'N/A')}\n\n")
            
            f.write(f"Files:\n")
            f.write(f"  Animation Viewer: animation.html\n")
            f.write(f"  Images Directory: images/\n")
            f.write(f"  Metadata: metadata.json\n")
            f.write(f"  Summary: summary.txt\n")
            f.write(f"  Gantt Chart: gantt_chart.png\n")
    
    def _create_gantt_chart(self):
        """Create a Gantt chart from the simulation data."""
        try:
            # Import gantt module
            from gantt import generate_gantt_chart
            
            # Get all units (friendly and enemy) for the Gantt chart. Prefer
            # the full unit lists (which include units that may have been
            # removed from active play) if the simulation exposes them.
            all_units = []
            sources = []
            if hasattr(self.sim, 'all_friendly_units'):
                sources.append(self.sim.all_friendly_units)
            elif hasattr(self.sim, 'friendly_units'):
                sources.append(self.sim.friendly_units)

            if hasattr(self.sim, 'all_enemy_units'):
                sources.append(self.sim.all_enemy_units)
            elif hasattr(self.sim, 'enemy_units'):
                sources.append(self.sim.enemy_units)

            # Only include units that track task logs to avoid errors with
            # simple enemy units that don't record actions.
            for unit_list in sources:
                all_units.extend([u for u in unit_list if hasattr(u, 'task_log')])
            
            if all_units:
                # Generate Gantt chart
                gantt_filename = self.run_dir / "gantt_chart.png"
                generate_gantt_chart(
                    all_units, 
                    filename=str(gantt_filename),
                    figsize=(20, 14),
                    label_mode='acronym',
                    include_target_key=True,
                    save_target_key_txt=True,
                    fontsize=11,
                    bar_height=0.8
                )
                print(f"Generated Gantt chart: {gantt_filename}")
            else:
                print("No units found for Gantt chart generation")
                
        except ImportError:
            print("Warning: gantt module not found, skipping Gantt chart generation")
        except Exception as e:
            print(f"Warning: Failed to generate Gantt chart: {e}")
    
    def cleanup(self):
        """Clean up temporary files if needed."""
        # This could be used to clean up temporary files
        pass


def create_animation_from_simulation(sim, output_dir: str = "animations", run_name: Optional[str] = None) -> SimulationAnimator:
    """
    Create an animation from an existing simulation.
    
    Args:
        sim: Simulation instance
        output_dir: Directory to save animation files
        run_name: Name for this animation run
    
    Returns:
        SimulationAnimator instance
    """
    animator = SimulationAnimator(sim, output_dir, run_name)
    
    # If simulation has already run, we can't capture individual steps
    # But we can create a static image
    if hasattr(sim, 'plotter') and sim.plotter:
        animator.save_step_image(0, {"note": "Final simulation state"})
        animator.finalize_animation(sim.evaluate_plan() if hasattr(sim, 'evaluate_plan') else {})
    
    return animator
