'''File to store user's settings.'''

# Input desired values below:

# Environment
time_step = 0.1  # Interval between calculations (s)
wind_speed = 10  # (m/s)
# Direction wind is coming FROM (degrees). 0 = North, 90 = East.
wind_direction = 0

# Note: air density & gravity will be calculated automatically @ different altitudes

# Launch
launch_pitch = 45  # Angle of elevation (deg). 90 = vertical.
launch_azimuth = 0  # Compass heading (deg). 0 = North, 90 = East.
launch_altitude = 0  # Height of launch (m)

# Rocket
area = 0.1  # Cross-sectional area (m^2)
dry_mass = 50  # kg
propellant_mass = 75  # kg
fuel_consumption_rate = 5.0  # kg/s, assuming max thrust whenever engine is on
I_sp = 250  # Specific impulse (s)

# Automatically calculate: Air density (rho), gravity, drag coefficient
