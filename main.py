'''Main file for running the application.'''
import math
import config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from ISA_model import get_atmosphere

# Constant Attributes:

# Earth attributes
G_EARTH = 6.67430e-11        # Gravitational constant (N·m^2/kg^2)
M_EARTH = 5.97219e24         # Mass of Earth (kg)
R_EARTH = 6371000            # Average radius of Earth (m)

# Atmosphere Attributes
GAMMA = 1.4                  # Heat capacity ratio for air
R = 287.058                  # Specific gas constant for air (J/(kg*K))

# Drag Coefficient Lookup table
MACH_DATA = [
    0.0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0
]
DC_DATA = [
    0.28, 0.27, 0.29, 0.38, 0.60, 0.58, 0.55, 0.47, 0.41, 0.35, 0.32, 0.30
]


# Get settings from config file
launch_angle, launch_altitude, time_step, area, dry_mass, propellant_mass, fuel_consumption_rate, I_sp, wind_vx = (
    config.launch_angle,
    config.launch_altitude,
    config.time_step,
    config.area,
    config.dry_mass,
    config.propellant_mass,
    config.fuel_consumption_rate,
    config.I_sp,
    config.wind_vx,
)

angle_rad = math.radians(launch_angle)  # Convert angle to radians

# Rocket starts with 0 velocity at the launchpad
v_x = 0
v_y = 0

# Initialize rocket's position
x = 0
y = 0 + launch_altitude
t = 0

# Lists to store rocket's position data
x_coords = []
y_coords = []

# Find the total possible runtime of the engine
burn_time = propellant_mass / fuel_consumption_rate

# Find the exhaust velocity of the engine
v_ex = I_sp * 9.80665

while y >= 0:  # Condition: rocket is on or above ground
    x_coords.append(x)
    y_coords.append(y)

    # Relative Velocity
    v_rel_x = v_x - wind_vx
    v_rel_y = v_y - 0

    # Speed
    mag_rel_v = math.sqrt(v_rel_x**2 + v_rel_y**2)

    # International Standard Atmosphere (ISA) model
    temperature, pressure, rho = get_atmosphere(y)

    # Variable gravity
    gravity = (G_EARTH * M_EARTH) / (R_EARTH + y)**2

    # Variable Drag Coefficient
    speed_of_sound = math.sqrt(GAMMA * R * temperature)
    mach_number = mag_rel_v / speed_of_sound
    drag_coefficient = np.interp(mach_number, MACH_DATA, DC_DATA)

    # Drag
    if mag_rel_v > 0:
        F_drag = 0.5 * rho * mag_rel_v**2 * drag_coefficient * area
        drag_x = -F_drag * (v_rel_x / mag_rel_v)
        drag_y = -F_drag * (v_rel_y / mag_rel_v)
    else:
        drag_x, drag_y = 0

# Mass & Thrust
    if t < burn_time:
        f_thrust = v_ex * fuel_consumption_rate
        f_thrust_x = f_thrust * math.cos(angle_rad)
        f_thrust_y = f_thrust * math.sin(angle_rad)

        propellant_mass = propellant_mass - fuel_consumption_rate * time_step
        total_mass = propellant_mass + dry_mass

    else:
        f_thrust, f_thrust_x, f_thrust_y = 0, 0, 0
        propellant_mass = 0
        total_mass = dry_mass

    # Forces
    net_force_x = f_thrust_x + 0.0 + drag_x
    net_force_y = f_thrust_y - (total_mass * gravity) + drag_y

    # Acceleration
    acceleration_x = net_force_x/total_mass
    acceleration_y = net_force_y/total_mass

    # Update Velocity
    v_x = v_x + acceleration_x * time_step
    v_y = v_y + acceleration_y * time_step

    # Update Position
    x = x + v_x * time_step
    y = y + v_y * time_step

    # Time Step
    t = t + time_step

# Set a style for later
plt.style.use('dark_background')
NEON_COLOR = 'cyan'

# Initiate plot
fig, ax = plt.subplots()

# Labels
ax.set_title("Powered Rocket Trajectory")
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Altitude (m)")

# Set limits w/ padding
ax.set_xlim(0, max(x_coords) * 1.1)
ax.set_ylim(0, max(y_coords) * 1.1)

# Animation objects (three for a neon effect)
line_glow, = ax.plot([], [], color=NEON_COLOR, linewidth=5, alpha=0.3)
line_main, = ax.plot([], [], color=NEON_COLOR, linewidth=2, alpha=0.7)
line_core, = ax.plot([], [], color=NEON_COLOR, linewidth=1, alpha=1.0)

# Arrow Properties
arrowprops = dict(
    facecolor=NEON_COLOR,
    edgecolor=NEON_COLOR,
    shrink=0.05,
    width=1,
    headwidth=8
)
# Create the single arrow object, initially invisible at (0,0)
arrow = ax.annotate(
    '',
    xy=(0, 0),
    xytext=(0, 0),
    arrowprops=arrowprops,
)

# Function to update the graph for the animation to play out


def update(frame):

    # Set data for all three lines
    line_glow.set_data(x_coords[:frame], y_coords[:frame])
    line_main.set_data(x_coords[:frame], y_coords[:frame])
    line_core.set_data(x_coords[:frame], y_coords[:frame])

    # Arrow
    if frame > 0:
        # Get the new tip and tail coordinates
        tip_x = x_coords[frame]
        tip_y = y_coords[frame]
        tail_x = x_coords[frame - 1]  # One frame behind
        tail_y = y_coords[frame - 1]  # One frame behind

        # Set the arrow's new positions
        arrow.xy = (tip_x, tip_y)
        arrow.set_position((tail_x, tail_y))


# The actual animation
anim = FuncAnimation(fig, update, frames=len(x_coords),
                     interval=7.5, blit=False, repeat=False)

# Check if flight is long enough for graph features
if len(x_coords) > 1:

    # Highlight the maximum (apogee) point
    apogee_y = max(y_coords)
    apogee_index = y_coords.index(apogee_y)
    apogee_x = x_coords[apogee_index]
    plt.text(apogee_x, apogee_y,
             f' Apogee: {apogee_y:.2f} m', va='bottom', ha='center')

    # Mark the launch & impact points
    plt.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Launch')
    plt.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='Impact')

    # Label the launch angle
    ax = plt.gca()  # Get current axes

    # Create an arc patch
    arc_radius = max(x_coords) * 0.05
    launch_arc = patches.Arc(
        (x_coords[0], y_coords[0]),  # Center of the arc
        width=arc_radius*2,  # Width of the arc's ellipse
        height=arc_radius*2,  # Height of the arc's ellipse
        angle=0,  # Rotation of the arc's ellipse
        theta1=0,  # Start angle of the arc
        theta2=launch_angle,  # End angle of the arc
        color='orange',
        linewidth=2,
        linestyle='--'
    )

    # Add the arc to the plot
    ax.add_patch(launch_arc)

    # Add a text label for the angle
    plt.text(arc_radius * 1.1, launch_altitude +
             arc_radius * 0.4, f'{launch_angle}°')


# Add a grid
ax.grid(True)

plt.show()
