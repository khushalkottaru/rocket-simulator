'''Main file for running the application.'''
import math
import config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from ISA_model import get_atmosphere
from mpl_toolkits.mplot3d import Axes3D

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
launch_pitch, launch_azimuth, launch_altitude, time_step, area, dry_mass, propellant_mass, fuel_consumption_rate, I_sp, wind_speed, wind_direction = (
    config.launch_pitch,
    config.launch_azimuth,
    config.launch_altitude,
    config.time_step,
    config.area,
    config.dry_mass,
    config.propellant_mass,
    config.fuel_consumption_rate,
    config.I_sp,
    config.wind_speed,
    config.wind_direction
)

# Convert angles to radians
pitch_rad = math.radians(launch_pitch)
azimuth_rad = math.radians(launch_azimuth)
# Wind is "from", so vector is opposite
wind_dir_rad = math.radians(wind_direction + 180)

# Calculate Wind Vector
w_x = wind_speed * math.sin(wind_dir_rad)
w_y = wind_speed * math.cos(wind_dir_rad)
w_z = 0  # Assumed horizontal wind

# Rocket starts with 0 velocity at the launchpad
v_x = 0
v_y = 0
v_z = 0

# Initialize rocket's position (x, y are ground plane, z is altitude)
x = 0
y = 0
z = 0 + launch_altitude
t = 0

# Lists to store rocket's position data
x_coords = []
y_coords = []
z_coords = []

# Find the total possible runtime of the engine
burn_time = propellant_mass / fuel_consumption_rate

# Find the exhaust velocity of the engine
v_ex = I_sp * 9.80665

while z >= 0:  # Condition: rocket is on or above ground
    x_coords.append(x)
    y_coords.append(y)
    z_coords.append(z)

    # Relative Velocity
    v_rel_x = v_x - w_x
    v_rel_y = v_y - w_y
    v_rel_z = v_z - w_z

    # Speed
    mag_rel_v = math.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)

    # International Standard Atmosphere (ISA) model
    temperature, pressure, rho = get_atmosphere(z)

    # Variable gravity
    gravity = (G_EARTH * M_EARTH) / (R_EARTH + z)**2

    # Variable Drag Coefficient
    speed_of_sound = math.sqrt(GAMMA * R * temperature)
    mach_number = mag_rel_v / speed_of_sound
    drag_coefficient = np.interp(mach_number, MACH_DATA, DC_DATA)

    # Drag
    if mag_rel_v > 0:
        F_drag = 0.5 * rho * mag_rel_v**2 * drag_coefficient * area
        drag_x = -F_drag * (v_rel_x / mag_rel_v)
        drag_y = -F_drag * (v_rel_y / mag_rel_v)
        drag_z = -F_drag * (v_rel_z / mag_rel_v)
    else:
        drag_x, drag_y, drag_z = 0, 0, 0

# Mass & Thrust
    if t < burn_time:
        f_thrust = v_ex * fuel_consumption_rate

        # Thrust components
        # Vertical component
        f_thrust_z = f_thrust * math.sin(pitch_rad)
        # Horizontal component
        f_thrust_h = f_thrust * math.cos(pitch_rad)

        f_thrust_x = f_thrust_h * math.sin(azimuth_rad)
        f_thrust_y = f_thrust_h * math.cos(azimuth_rad)

        propellant_mass = propellant_mass - fuel_consumption_rate * time_step
        total_mass = propellant_mass + dry_mass

    else:
        f_thrust, f_thrust_x, f_thrust_y, f_thrust_z = 0, 0, 0, 0
        propellant_mass = 0
        total_mass = dry_mass

    # Forces (Gravity acts only on Z)
    net_force_x = f_thrust_x + drag_x
    net_force_y = f_thrust_y + drag_y
    net_force_z = f_thrust_z - (total_mass * gravity) + drag_z

    # Acceleration
    acceleration_x = net_force_x/total_mass
    acceleration_y = net_force_y/total_mass
    acceleration_z = net_force_z/total_mass

    # Update Velocity
    v_x = v_x + acceleration_x * time_step
    v_y = v_y + acceleration_y * time_step
    v_z = v_z + acceleration_z * time_step

    # Update Position
    x = x + v_x * time_step
    y = y + v_y * time_step
    z = z + v_z * time_step

    # Time Step
    t = t + time_step

# Set a style for later
plt.style.use('dark_background')
NEON_COLOR = 'cyan'

# Initiate plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Labels
ax.set_title("Powered Rocket Trajectory (3D)")
ax.set_xlabel("X Distance (m)")
ax.set_ylabel("Y Distance (m)")
ax.set_zlabel("Altitude (m)")

# Set limits w/ padding
ax.set_xlim(min(x_coords) * 1.1, max(x_coords) * 1.1)
ax.set_ylim(min(y_coords) * 1.1, max(y_coords) * 1.1)
ax.set_zlim(0, max(z_coords) * 1.1)

# Animation objects (three for a neon effect)
line_glow, = ax.plot([], [], [], color=NEON_COLOR, linewidth=5, alpha=0.3)
line_main, = ax.plot([], [], [], color=NEON_COLOR, linewidth=2, alpha=0.7)
line_core, = ax.plot([], [], [], color=NEON_COLOR, linewidth=1, alpha=1.0)

# Create a point object for the rocket
rocket_marker, = ax.plot([], [], [], marker='^',
                         color=NEON_COLOR, markersize=10)

# Function to update the graph for the animation to play out


def update(frame):
    # Set data for all three lines
    # 3D plot requires setting x,y data then z properties separately
    current_x = x_coords[:frame]
    current_y = y_coords[:frame]
    current_z = z_coords[:frame]

    line_glow.set_data(current_x, current_y)
    line_glow.set_3d_properties(current_z)

    line_main.set_data(current_x, current_y)
    line_main.set_3d_properties(current_z)

    line_core.set_data(current_x, current_y)
    line_core.set_3d_properties(current_z)

    # Rocket Marker
    if frame > 0:
        rocket_marker.set_data([x_coords[frame]], [y_coords[frame]])
        rocket_marker.set_3d_properties([z_coords[frame]])


# The actual animation
anim = FuncAnimation(fig, update, frames=len(x_coords),
                     interval=7.5, blit=False, repeat=False)

# Check if flight is long enough for graph features
if len(x_coords) > 1:

    # Highlight the maximum (apogee) point
    apogee_z = max(z_coords)
    apogee_index = z_coords.index(apogee_z)
    apogee_x = x_coords[apogee_index]
    apogee_y = y_coords[apogee_index]

    ax.text(apogee_x, apogee_y, apogee_z,
            f' Apogee: {apogee_z:.2f} m', color='white')

    # Mark the launch & impact points
    ax.scatter([x_coords[0]], [y_coords[0]], [z_coords[0]],
               color='green', s=100, label='Launch')
    ax.scatter([x_coords[-1]], [y_coords[-1]], [z_coords[-1]],
               color='red', s=100, label='Impact')

# Add a grid
ax.grid(True)

plt.show()
