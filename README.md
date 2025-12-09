# 3D Rocket Trajectory Simulator

A Python script that simulates the 3D flight path of a powered rocket, factoring in variable gravity, variable mass, a dynamic atmosphere, aerodynamic drag, wind, and thrust.

## Features
-   Models forces like gravity, drag, and wind.
-   Uses a time-step simulation to calculate the flight path.
-   Considers when the engine is on and calculates thrust
-   Simulates variable mass as fuel is consumed
-   Calculates gravity at each time step
-   Finds the drag coefficient based on mach number at each time step
-   Finds air density, temperature, and pressure at each altitude
-   Generates a detailed plot of the trajectory using Matplotlib.

## How to Run
1.  Clone this repository.
2.  Create and activate a virtual environment.
3.  Install the required packages: `pip install -r requirements.txt`
4.  Run the simulation: `python3 main.py`

## Testing
This project includes a rigorous physics validation suite to ensure simulation accuracy.

### Physics Validation
Tests individual physics formulas (ISA model, gravity, drag, thrust vectors) against known physical laws:
```bash
python3 physics_tests.py
```

### Simulation Scenarios
Runs the simulator through various configurations (vertical launch, optimal range, wind effects) to verify trajectory behavior:
```bash
python3 simulation_tests.py
```
