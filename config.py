'''File to store user's settings.'''

# Input desired values below:

# Environment
gravity = 9.81  # m/s^2
time_step = 0.1  # Interval between calculations (s)
rho = 1.225  # Air density (kg/m^3)
wind_vx = -50  # (m/s), (+) indicates tailwind & (-) indicates headwind

# Launch
launch_angle = 45  # Angle at which rocket is launched (deg.)
launch_altitude = 0  # Height of launch (m)

# Rocket
area = 0.1  # Cross-sectional area (m^2)
dry_mass = 50  # kg
propellant_mass = 75  # kg
fuel_consumption_rate = 5.0  # kg/s, assuming max thrust whenever engine is on
I_sp = 250  # Specific impulse (s)
drag_coefficient = 0.5  # Dimensionless
