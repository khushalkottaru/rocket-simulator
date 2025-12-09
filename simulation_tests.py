'''
Multi-Configuration Simulation Tests for Rocket Simulator

This script runs the simulator with different configurations to test
physics accuracy under various conditions.
'''

import math
import sys
import numpy as np

# Import the ISA model
from ISA_model import get_atmosphere

# Constants from the simulator
G_EARTH = 6.67430e-11
M_EARTH = 5.97219e24
R_EARTH = 6371000
GAMMA = 1.4
R = 287.058

# Drag Coefficient Lookup table
MACH_DATA = [0.0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]
DC_DATA = [0.28, 0.27, 0.29, 0.38, 0.60,
           0.58, 0.55, 0.47, 0.41, 0.35, 0.32, 0.30]


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def simulate_rocket(config, verbose=True):
    """
    Run a rocket simulation with the given configuration.
    Returns trajectory data and key metrics.
    """
    # Unpack configuration
    launch_pitch = config.get('launch_pitch', 90)
    launch_azimuth = config.get('launch_azimuth', 0)
    launch_altitude = config.get('launch_altitude', 0)
    time_step = config.get('time_step', 0.1)
    area = config.get('area', 0.1)
    dry_mass = config.get('dry_mass', 50)
    propellant_mass = config.get('propellant_mass', 75)
    fuel_consumption_rate = config.get('fuel_consumption_rate', 5.0)
    I_sp = config.get('I_sp', 250)
    wind_speed = config.get('wind_speed', 0)
    wind_direction = config.get('wind_direction', 0)

    # Convert angles to radians
    pitch_rad = math.radians(launch_pitch)
    azimuth_rad = math.radians(launch_azimuth)
    wind_dir_rad = math.radians(wind_direction + 180)

    # Wind Vector
    w_x = wind_speed * math.sin(wind_dir_rad)
    w_y = wind_speed * math.cos(wind_dir_rad)
    w_z = 0

    # Initial conditions
    v_x, v_y, v_z = 0, 0, 0
    x, y, z = 0, 0, launch_altitude
    t = 0

    # Data storage
    x_coords, y_coords, z_coords = [], [], []
    velocities = []
    times = []

    # Burn parameters
    initial_propellant = propellant_mass
    burn_time = propellant_mass / fuel_consumption_rate
    v_ex = I_sp * 9.80665

    # Metrics
    max_velocity = 0
    max_acceleration = 0
    max_mach = 0

    while z >= 0:
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        times.append(t)

        # Current velocity magnitude
        mag_v = math.sqrt(v_x**2 + v_y**2 + v_z**2)
        velocities.append(mag_v)
        max_velocity = max(max_velocity, mag_v)

        # Relative velocity (considering wind)
        v_rel_x = v_x - w_x
        v_rel_y = v_y - w_y
        v_rel_z = v_z - w_z
        mag_rel_v = math.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)

        # Atmosphere
        temperature, pressure, rho = get_atmosphere(z)

        # Gravity
        gravity = (G_EARTH * M_EARTH) / (R_EARTH + z)**2

        # Speed of sound and Mach
        speed_of_sound = math.sqrt(GAMMA * R * temperature)
        mach_number = mag_rel_v / speed_of_sound
        max_mach = max(max_mach, mach_number)
        drag_coefficient = np.interp(mach_number, MACH_DATA, DC_DATA)

        # Drag force
        if mag_rel_v > 0:
            F_drag = 0.5 * rho * mag_rel_v**2 * drag_coefficient * area
            drag_x = -F_drag * (v_rel_x / mag_rel_v)
            drag_y = -F_drag * (v_rel_y / mag_rel_v)
            drag_z = -F_drag * (v_rel_z / mag_rel_v)
        else:
            drag_x, drag_y, drag_z = 0, 0, 0

        # Thrust and mass
        if t < burn_time:
            f_thrust = v_ex * fuel_consumption_rate
            f_thrust_z = f_thrust * math.sin(pitch_rad)
            f_thrust_h = f_thrust * math.cos(pitch_rad)
            f_thrust_x = f_thrust_h * math.sin(azimuth_rad)
            f_thrust_y = f_thrust_h * math.cos(azimuth_rad)

            propellant_mass = initial_propellant - fuel_consumption_rate * t
            total_mass = propellant_mass + dry_mass
        else:
            f_thrust = f_thrust_x = f_thrust_y = f_thrust_z = 0
            propellant_mass = 0
            total_mass = dry_mass

        # Net forces
        net_force_x = f_thrust_x + drag_x
        net_force_y = f_thrust_y + drag_y
        net_force_z = f_thrust_z - (total_mass * gravity) + drag_z

        # Acceleration
        a_x = net_force_x / total_mass
        a_y = net_force_y / total_mass
        a_z = net_force_z / total_mass
        mag_a = math.sqrt(a_x**2 + a_y**2 + a_z**2)
        max_acceleration = max(max_acceleration, mag_a)

        # Update velocity
        v_x += a_x * time_step
        v_y += a_y * time_step
        v_z += a_z * time_step

        # Update position
        x += v_x * time_step
        y += v_y * time_step
        z += v_z * time_step

        # Time step
        t += time_step

    # Calculate metrics
    apogee = max(z_coords)
    apogee_idx = z_coords.index(apogee)
    time_to_apogee = times[apogee_idx]
    total_flight_time = times[-1]
    range_x = x_coords[-1]
    range_y = y_coords[-1]
    ground_range = math.sqrt(range_x**2 + range_y**2)
    impact_velocity = velocities[-1] if velocities else 0

    results = {
        'apogee': apogee,
        'time_to_apogee': time_to_apogee,
        'total_flight_time': total_flight_time,
        'ground_range': ground_range,
        'range_x': range_x,
        'range_y': range_y,
        'max_velocity': max_velocity,
        'max_acceleration': max_acceleration,
        'max_mach': max_mach,
        'impact_velocity': impact_velocity,
        'burn_time': burn_time,
        'trajectory': (x_coords, y_coords, z_coords, times)
    }

    if verbose:
        print(f"\n  📊 Results:")
        print(f"     Apogee:           {apogee:.1f} m ({apogee/1000:.2f} km)")
        print(f"     Time to apogee:   {time_to_apogee:.1f} s")
        print(f"     Total flight:     {total_flight_time:.1f} s")
        print(
            f"     Ground range:     {ground_range:.1f} m ({ground_range/1000:.2f} km)")
        print(
            f"     Max velocity:     {max_velocity:.1f} m/s (Mach {max_mach:.2f})")
        print(
            f"     Max acceleration: {max_acceleration:.1f} m/s² ({max_acceleration/9.81:.1f} g)")
        print(f"     Impact velocity:  {impact_velocity:.1f} m/s")

    return results


# =============================================================================
# Test Configuration 1: Vertical Launch (Pure ballistic test)
# =============================================================================
def test_vertical_launch():
    print_header("TEST 1: Vertical Launch (90° pitch)")

    print("\n  Config: Straight up launch to test maximum altitude")
    print("  Expected: Symmetric trajectory, landing near launch point")

    config = {
        'launch_pitch': 90,
        'launch_azimuth': 0,
        'time_step': 0.05,
        'dry_mass': 50,
        'propellant_mass': 75,
        'fuel_consumption_rate': 5.0,
        'I_sp': 250,
        'area': 0.1,
        'wind_speed': 0,
    }

    results = simulate_rocket(config)

    # Physics checks
    print("\n  🔬 Physics Validation:")

    # Check that landing is near launch point
    if results['ground_range'] < 10:
        print(
            f"     ✅ Symmetric trajectory: landed {results['ground_range']:.2f} m from launch")
    else:
        print(
            f"     ⚠️  Unexpected drift: landed {results['ground_range']:.2f} m from launch")

    # Check that apogee time is roughly half of flight time
    ratio = results['time_to_apogee'] / results['total_flight_time']
    if 0.4 < ratio < 0.6:
        print(
            f"     ✅ Apogee timing: {ratio:.2f} of total flight (expected ~0.5)")
    else:
        print(
            f"     ⚠️  Asymmetric timing: apogee at {ratio:.2f} of total flight")

    # Theoretical delta-v calculation
    v_ex = 250 * 9.80665
    delta_v_th = v_ex * math.log(125/50)
    print(f"     📐 Theoretical Δv: {delta_v_th:.0f} m/s")
    print(f"     📐 Achieved max velocity: {results['max_velocity']:.0f} m/s")
    print(
        f"     📐 Gravity losses: ~{delta_v_th - results['max_velocity']:.0f} m/s")

    return results


# =============================================================================
# Test Configuration 2: 45° Optimal Angle Test
# =============================================================================
def test_45_degree_launch():
    print_header("TEST 2: 45° Launch Angle")

    print("\n  Config: Classic 45° launch for range")
    print("  Theory: 45° is optimal for vacuum; with drag, optimal is lower")

    config = {
        'launch_pitch': 45,
        'launch_azimuth': 0,
        'time_step': 0.05,
        'dry_mass': 50,
        'propellant_mass': 75,
        'fuel_consumption_rate': 5.0,
        'I_sp': 250,
        'area': 0.1,
        'wind_speed': 0,
    }

    results = simulate_rocket(config)

    print("\n  🔬 Physics Validation:")
    # With drag air and powered flight, 45° won't give max range
    # But should give a good balance
    print(f"     📐 Range achieved: {results['ground_range']/1000:.2f} km")
    print(f"     📐 Apogee: {results['apogee']/1000:.2f} km")

    return results


# =============================================================================
# Test Configuration 3: Compare Launch Angles
# =============================================================================
def test_launch_angles():
    print_header("TEST 3: Launch Angle Comparison")

    print("\n  Testing angles from 20° to 80° to find optimal range")

    angles = [20, 30, 40, 45, 50, 60, 70, 80]
    results_by_angle = {}

    base_config = {
        'time_step': 0.1,
        'dry_mass': 50,
        'propellant_mass': 75,
        'fuel_consumption_rate': 5.0,
        'I_sp': 250,
        'area': 0.1,
        'wind_speed': 0,
    }

    print(
        f"\n  {'Angle':>6} | {'Apogee (km)':>12} | {'Range (km)':>11} | {'Max Mach':>9}")
    print("  " + "-" * 50)

    for angle in angles:
        config = base_config.copy()
        config['launch_pitch'] = angle
        results = simulate_rocket(config, verbose=False)
        results_by_angle[angle] = results
        print(
            f"  {angle:>5}° | {results['apogee']/1000:>11.2f} | {results['ground_range']/1000:>10.2f} | {results['max_mach']:>8.2f}")

    # Find optimal
    best_angle = max(results_by_angle,
                     key=lambda a: results_by_angle[a]['ground_range'])
    print(f"\n  🏆 Best range achieved at {best_angle}° pitch")

    # Physics validation
    print("\n  🔬 Physics Validation:")
    if best_angle < 45:
        print(
            f"     ✅ Optimal angle < 45° ({best_angle}°) - consistent with drag effects")
    elif best_angle == 45:
        print(f"     ⚠️  Optimal at exactly 45° - drag may not be significant enough")
    else:
        print(
            f"     ❌ Optimal > 45° ({best_angle}°) - unexpected for atmosphere with drag")

    return results_by_angle


# =============================================================================
# Test Configuration 4: Wind Effects
# =============================================================================
def test_wind_effects():
    print_header("TEST 4: Wind Effects on Trajectory")

    print("\n  Testing how wind affects trajectory")

    base_config = {
        'launch_pitch': 45,
        'launch_azimuth': 0,  # North
        'time_step': 0.1,
        'dry_mass': 50,
        'propellant_mass': 75,
        'fuel_consumption_rate': 5.0,
        'I_sp': 250,
        'area': 0.1,
    }

    print(
        f"\n  {'Wind':>20} | {'Range X (m)':>12} | {'Range Y (m)':>12} | {'Total (km)':>11}")
    print("  " + "-" * 60)

    wind_cases = [
        (0, 0, "No wind"),
        (10, 0, "10 m/s from North"),
        (10, 90, "10 m/s from East"),
        (10, 180, "10 m/s from South"),
        (20, 0, "20 m/s from North"),
    ]

    results_by_wind = {}

    for speed, direction, desc in wind_cases:
        config = base_config.copy()
        config['wind_speed'] = speed
        config['wind_direction'] = direction
        results = simulate_rocket(config, verbose=False)
        results_by_wind[desc] = results

        print(
            f"  {desc:>20} | {results['range_x']:>11.1f} | {results['range_y']:>11.1f} | {results['ground_range']/1000:>10.2f}")

    # Physics validation
    print("\n  🔬 Physics Validation:")
    no_wind = results_by_wind["No wind"]
    north_wind = results_by_wind["10 m/s from North"]

    # Wind from north should push rocket south (negative y)
    if north_wind['range_y'] < no_wind['range_y']:
        print("     ✅ North wind correctly decreases Y range (pushes south)")
    else:
        print("     ❌ Wind effect incorrect - should reduce Y range")

    return results_by_wind


# =============================================================================
# Test Configuration 5: Mass Ratio Effects
# =============================================================================
def test_mass_ratio():
    print_header("TEST 5: Mass Ratio Effects")

    print("\n  Testing Tsiolkovsky equation: Δv = v_ex * ln(m0/mf)")

    v_ex = 250 * 9.80665  # ~2451 m/s
    dry_mass = 50

    print(f"\n  {'Prop Mass':>10} | {'Mass Ratio':>11} | {'Th. Δv':>10} | {'Apogee':>10} | {'Max V':>10}")
    print("  " + "-" * 60)

    propellant_masses = [25, 50, 75, 100, 150, 200]

    for prop_mass in propellant_masses:
        config = {
            'launch_pitch': 90,  # Vertical for simplicity
            'time_step': 0.1,
            'dry_mass': dry_mass,
            'propellant_mass': prop_mass,
            'fuel_consumption_rate': 5.0,
            'I_sp': 250,
            'area': 0.1,
            'wind_speed': 0,
        }

        total_mass = dry_mass + prop_mass
        mass_ratio = total_mass / dry_mass
        theoretical_dv = v_ex * math.log(mass_ratio)

        results = simulate_rocket(config, verbose=False)

        print(
            f"  {prop_mass:>9} kg | {mass_ratio:>10.2f} | {theoretical_dv:>9.0f} m/s | {results['apogee']/1000:>9.2f} km | {results['max_velocity']:>9.0f} m/s")

    print("\n  🔬 Physics Validation:")
    print("     As propellant increases, max velocity should approach theoretical Δv")
    print("     Gravity losses decrease relative fraction with higher mass ratio")

    return True


# =============================================================================
# Test Configuration 6: Time Step Sensitivity
# =============================================================================
def test_timestep_sensitivity():
    print_header("TEST 6: Time Step Sensitivity (Numerical Stability)")

    print("\n  Testing if results converge with smaller time steps")

    time_steps = [1.0, 0.5, 0.1, 0.05, 0.01]

    print(
        f"\n  {'dt (s)':>8} | {'Apogee (m)':>12} | {'Flight Time':>12} | {'Range (m)':>12}")
    print("  " + "-" * 55)

    results_by_dt = {}

    for dt in time_steps:
        config = {
            'launch_pitch': 45,
            'time_step': dt,
            'dry_mass': 50,
            'propellant_mass': 75,
            'fuel_consumption_rate': 5.0,
            'I_sp': 250,
            'area': 0.1,
            'wind_speed': 0,
        }

        results = simulate_rocket(config, verbose=False)
        results_by_dt[dt] = results

        print(
            f"  {dt:>7} | {results['apogee']:>11.1f} | {results['total_flight_time']:>11.1f} | {results['ground_range']:>11.1f}")

    # Check convergence
    print("\n  🔬 Convergence Analysis:")
    apogee_01 = results_by_dt[0.1]['apogee']
    apogee_001 = results_by_dt[0.01]['apogee']
    diff = abs(apogee_01 - apogee_001) / apogee_001 * 100

    if diff < 1:
        print(
            f"     ✅ Good convergence: dt=0.1 vs dt=0.01 differ by {diff:.2f}%")
    elif diff < 5:
        print(f"     ⚠️  Acceptable convergence: {diff:.2f}% difference")
    else:
        print(
            f"     ❌ Poor convergence: {diff:.2f}% difference - consider smaller dt")

    return results_by_dt


# =============================================================================
# Test Configuration 7: High Altitude Effects
# =============================================================================
def test_high_altitude():
    print_header("TEST 7: High Altitude Physics")

    print("\n  Testing a powerful rocket that reaches high altitude")
    print("  Should show decreasing gravity and atmospheric density effects")

    # More powerful rocket configuration
    config = {
        'launch_pitch': 90,
        'time_step': 0.1,
        'dry_mass': 50,
        'propellant_mass': 300,  # More fuel
        'fuel_consumption_rate': 10.0,  # Higher thrust
        'I_sp': 300,  # Better engine
        'area': 0.1,
        'wind_speed': 0,
    }

    results = simulate_rocket(config)

    print("\n  🔬 Physics Validation:")

    if results['apogee'] > 100000:  # Above Karman line
        print(
            f"     ✅ Reached space (>{100}km): {results['apogee']/1000:.1f} km")
    else:
        print(
            f"     ℹ️  Apogee: {results['apogee']/1000:.1f} km (sub-orbital)")

    # Check that max Mach > 1 (supersonic)
    if results['max_mach'] > 1:
        print(f"     ✅ Achieved supersonic: Mach {results['max_mach']:.2f}")
    else:
        print(f"     ℹ️  Subsonic flight: Mach {results['max_mach']:.2f}")

    return results


# =============================================================================
# Test Configuration 8: 3D Trajectory (Azimuth Test)
# =============================================================================
def test_3d_trajectory():
    print_header("TEST 8: 3D Trajectory (Azimuth Direction)")

    print("\n  Testing different azimuth angles at 45° pitch")

    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]

    print(
        f"\n  {'Azimuth':>8} | {'Direction':>12} | {'X Range':>12} | {'Y Range':>12}")
    print("  " + "-" * 52)

    directions = ["North", "NE", "East", "SE", "South", "SW", "West", "NW"]

    for az, dir_name in zip(azimuths, directions):
        config = {
            'launch_pitch': 45,
            'launch_azimuth': az,
            'time_step': 0.1,
            'dry_mass': 50,
            'propellant_mass': 75,
            'fuel_consumption_rate': 5.0,
            'I_sp': 250,
            'area': 0.1,
            'wind_speed': 0,
        }

        results = simulate_rocket(config, verbose=False)
        print(
            f"  {az:>7}° | {dir_name:>12} | {results['range_x']/1000:>10.2f} km | {results['range_y']/1000:>10.2f} km")

    print("\n  🔬 Physics Validation:")
    print("     All directions should have same total range (rotational symmetry)")

    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_all_simulation_tests():
    print("\n" + "╔" + "═" * 70 + "╗")
    print("║  ROCKET SIMULATOR - MULTI-CONFIGURATION PHYSICS TESTS" + " " * 16 + "║")
    print("╚" + "═" * 70 + "╝")

    test_vertical_launch()
    test_45_degree_launch()
    test_launch_angles()
    test_wind_effects()
    test_mass_ratio()
    test_timestep_sensitivity()
    test_high_altitude()
    test_3d_trajectory()

    print("\n" + "=" * 70)
    print("  ALL SIMULATION TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_simulation_tests()
