'''File to store user's settings.'''

# Input desired values below:

# Environment
time_step = 0.1  # Interval between calculations (s)
wind_vx = -10  # (m/s), (+) indicates tailwind & (-) indicates headwind

# Note: air density & gravity will be calculated automatically @ different altitudes

# Launch
launch_angle = 45  # Angle at which rocket is launched (deg.)
launch_altitude = 0  # Height of launch (m)

# Rocket
area = 0.1  # Cross-sectional area (m^2)
dry_mass = 50  # kg
propellant_mass = 75  # kg
fuel_consumption_rate = 5.0  # kg/s, assuming max thrust whenever engine is on
I_sp = 250  # Specific impulse (s)

# Automatically calculate: Air density (rho), gravity, drag coefficient
