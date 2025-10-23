'''Main file for running the application.'''
import math
import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Get settings from config file
gravity, launch_angle, launch_altitude, time_step, area, dry_mass, propellant_mass, fuel_consumption_rate, I_sp, drag_coefficient, rho, wind_vx = (
    config.gravity,
    config.launch_angle,
    config.launch_altitude,
    config.time_step,
    config.area,
    config.dry_mass,
    config.propellant_mass,
    config.fuel_consumption_rate,
    config.I_sp,
    config.drag_coefficient,
    config.rho,
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

plt.plot(x_coords, y_coords)

# Labels
plt.title("Powered Rocket Trajectory")
plt.xlabel("Distance (m)")
plt.ylabel("Altitude (m)")

# Check if flight is long enough for graph features
if len(x_coords) > 1:

    # Arrow
    plt.annotate(
        '', xy=(x_coords[-1], y_coords[-1]),
        xytext=(x_coords[-2], y_coords[-2]),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8)
    )

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
        color='blue',
        linewidth=2,
        linestyle='--'
    )

    # Add the arc to the plot
    ax.add_patch(launch_arc)

    # Add a text label for the angle
    plt.text(arc_radius * 1.1, launch_altitude +
             arc_radius * 0.4, f'{launch_angle}°')


# Scaling
plt.axis('equal')
plt.grid(True)

plt.show()
