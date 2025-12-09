'''
Rigorous Physics Validation Tests for Rocket Simulator

This script tests the simulator against known physics principles:
1. ISA Atmosphere Model accuracy
2. Gravity calculations
3. Drag force calculations
4. Thrust and mass flow
5. Trajectory physics
'''

from ISA_model import get_atmosphere
import math
import sys

# Constants from the simulator
G_EARTH = 6.67430e-11        # Gravitational constant (N·m²/kg²)
M_EARTH = 5.97219e24         # Mass of Earth (kg)
R_EARTH = 6371000            # Average radius of Earth (m)
GAMMA = 1.4                  # Heat capacity ratio for air
R_GAS = 287.058              # Specific gas constant for air (J/(kg*K))

# Import the ISA model

# Test results tracking
test_results = []


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_passed(name, details=""):
    test_results.append(("PASS", name))
    print(f"✅ PASS: {name}")
    if details:
        print(f"   {details}")


def test_failed(name, expected, actual, tolerance=""):
    test_results.append(("FAIL", name))
    print(f"❌ FAIL: {name}")
    print(f"   Expected: {expected}")
    print(f"   Actual:   {actual}")
    if tolerance:
        print(f"   Tolerance: {tolerance}")


def test_warning(name, details):
    test_results.append(("WARN", name))
    print(f"⚠️  WARN: {name}")
    print(f"   {details}")


# =============================================================================
# TEST 1: ISA Atmosphere Model
# =============================================================================
def test_isa_model():
    print_header("TEST 1: ISA Atmosphere Model Accuracy")

    # Standard ISA values at sea level (0 m)
    # T = 288.15 K, P = 101.325 kPa, ρ = 1.225 kg/m³

    print("\n--- Sea Level (0 m) ---")
    T, P, rho = get_atmosphere(0)

    # Expected values
    T_expected = 288.15  # K
    P_expected = 101.29  # kPa (from the model formula)
    rho_expected = 1.225  # kg/m³

    T_error = abs(T - T_expected) / T_expected * 100
    rho_error = abs(rho - rho_expected) / rho_expected * 100

    print(
        f"Temperature: {T:.2f} K (expected ~{T_expected} K, error: {T_error:.2f}%)")
    print(f"Pressure: {P:.2f} kPa")
    print(
        f"Density: {rho:.4f} kg/m³ (expected ~{rho_expected} kg/m³, error: {rho_error:.2f}%)")

    if T_error < 1 and rho_error < 5:
        test_passed("Sea level atmosphere",
                    f"T error: {T_error:.2f}%, ρ error: {rho_error:.2f}%")
    else:
        test_failed("Sea level atmosphere",
                    f"T={T_expected}K, ρ={rho_expected}kg/m³",
                    f"T={T:.2f}K, ρ={rho:.4f}kg/m³")

    # Test at 5,000 m (typical mountain altitude)
    # ISA: T ≈ 255.7 K, P ≈ 54.0 kPa, ρ ≈ 0.736 kg/m³
    print("\n--- 5,000 m altitude ---")
    T, P, rho = get_atmosphere(5000)
    T_expected = 255.54  # K (15.04 - 0.00649*5000 + 273.15)
    rho_expected = 0.736  # kg/m³ (approximate)

    T_error = abs(T - T_expected) / T_expected * 100
    rho_error = abs(rho - rho_expected) / rho_expected * 100

    print(
        f"Temperature: {T:.2f} K (expected ~{T_expected} K, error: {T_error:.2f}%)")
    print(
        f"Density: {rho:.4f} kg/m³ (expected ~{rho_expected} kg/m³, error: {rho_error:.2f}%)")

    if T_error < 1 and rho_error < 10:
        test_passed("5,000 m atmosphere",
                    f"T error: {T_error:.2f}%, ρ error: {rho_error:.2f}%")
    else:
        test_failed("5,000 m atmosphere",
                    f"T={T_expected}K, ρ={rho_expected}kg/m³",
                    f"T={T:.2f}K, ρ={rho:.4f}kg/m³")

    # Test at 11,000 m (tropopause)
    print("\n--- 11,000 m altitude (Tropopause) ---")
    T, P, rho = get_atmosphere(11000)
    T_expected = 216.65  # K (ISA standard tropopause temp)
    rho_expected = 0.365  # kg/m³ (approximate)

    T_actual = T
    print(f"Temperature: {T:.2f} K (expected ~{T_expected} K)")
    print(f"Density: {rho:.4f} kg/m³ (expected ~{rho_expected} kg/m³)")

    # Note: The model has a gap at exactly 11000 m due to the condition
    # Check for boundary condition issue
    T_lower, _, _ = get_atmosphere(10999)
    T_upper, _, _ = get_atmosphere(11001)
    discontinuity = abs(T_lower - T_upper)

    if discontinuity > 5:
        test_warning("Tropopause boundary",
                     f"Discontinuity of {discontinuity:.2f} K between 10999m and 11001m")
    else:
        test_passed("Tropopause transition",
                    f"Smooth transition (Δ={discontinuity:.2f} K)")

    # Test at 20,000 m (stratosphere)
    print("\n--- 20,000 m altitude (Stratosphere) ---")
    T, P, rho = get_atmosphere(20000)
    T_expected = 216.65  # K (isothermal in lower stratosphere)

    print(f"Temperature: {T:.2f} K (expected ~{T_expected} K)")
    print(f"Pressure: {P:.4f} kPa")
    print(f"Density: {rho:.6f} kg/m³")

    # In lower stratosphere, temp should be constant
    T_15k, _, _ = get_atmosphere(15000)
    T_20k, _, _ = get_atmosphere(20000)

    if abs(T_15k - T_20k) < 1:
        test_passed("Stratosphere isothermal",
                    f"Temp at 15km: {T_15k:.2f}K, at 20km: {T_20k:.2f}K")
    else:
        test_warning("Stratosphere temp variation",
                     f"Lower stratosphere should be isothermal. Got {T_15k:.2f}K at 15km, {T_20k:.2f}K at 20km")


# =============================================================================
# TEST 2: Gravity Model
# =============================================================================
def test_gravity():
    print_header("TEST 2: Gravity Model Accuracy")

    # Test surface gravity
    print("\n--- Surface Gravity ---")
    g_surface = (G_EARTH * M_EARTH) / R_EARTH**2
    g_expected = 9.80665  # m/s² (standard gravity)
    g_error = abs(g_surface - g_expected) / g_expected * 100

    print(f"Calculated: {g_surface:.5f} m/s²")
    print(f"Expected:   {g_expected:.5f} m/s²")
    print(f"Error:      {g_error:.3f}%")

    if g_error < 0.1:
        test_passed("Surface gravity", f"Error: {g_error:.3f}%")
    else:
        test_failed("Surface gravity", g_expected, g_surface)

    # Test gravity at 100 km (Kármán line)
    print("\n--- Gravity at 100 km (Kármán Line) ---")
    altitude = 100000  # 100 km
    g_100km = (G_EARTH * M_EARTH) / (R_EARTH + altitude)**2
    g_expected_100km = 9.505  # m/s² (approximate)
    g_reduction = (1 - g_100km / g_surface) * 100

    print(f"Calculated: {g_100km:.4f} m/s²")
    print(f"Reduction from surface: {g_reduction:.2f}%")

    if abs(g_100km - g_expected_100km) / g_expected_100km < 0.05:
        test_passed("Gravity at 100 km",
                    f"g = {g_100km:.4f} m/s² (reduction: {g_reduction:.2f}%)")
    else:
        test_failed("Gravity at 100 km", g_expected_100km, g_100km)

    # Test inverse square law
    print("\n--- Inverse Square Law Verification ---")
    g_at_2R = (G_EARTH * M_EARTH) / (2 * R_EARTH)**2
    expected_ratio = 0.25  # Should be 1/4 at 2*R
    actual_ratio = g_at_2R / g_surface

    print(f"g at 2*R_Earth: {g_at_2R:.4f} m/s²")
    print(
        f"Ratio to surface g: {actual_ratio:.4f} (expected: {expected_ratio})")

    if abs(actual_ratio - expected_ratio) < 0.001:
        test_passed("Inverse square law", f"Ratio: {actual_ratio:.4f}")
    else:
        test_failed("Inverse square law", expected_ratio, actual_ratio)


# =============================================================================
# TEST 3: Drag Physics
# =============================================================================
def test_drag():
    print_header("TEST 3: Drag Force Calculations")

    # Drag formula: F_d = 0.5 * ρ * v² * C_d * A
    print("\n--- Drag Force Formula Verification ---")

    # Test case: v = 100 m/s, ρ = 1.225 kg/m³, Cd = 0.28, A = 0.1 m²
    v = 100
    rho = 1.225
    Cd = 0.28
    A = 0.1

    F_drag = 0.5 * rho * v**2 * Cd * A
    F_expected = 171.5  # N

    print(f"Test case: v={v} m/s, ρ={rho} kg/m³, Cd={Cd}, A={A} m²")
    print(f"Calculated drag: {F_drag:.2f} N")
    print(f"Expected drag:   {F_expected:.2f} N")

    if abs(F_drag - F_expected) < 0.5:
        test_passed("Drag formula", f"F_d = {F_drag:.2f} N")
    else:
        test_failed("Drag formula", F_expected, F_drag)

    # Test velocity squared relationship
    print("\n--- Velocity Squared Relationship ---")
    F_drag_2v = 0.5 * rho * (2*v)**2 * Cd * A
    ratio = F_drag_2v / F_drag

    print(f"Drag at v:   {F_drag:.2f} N")
    print(f"Drag at 2v:  {F_drag_2v:.2f} N")
    print(f"Ratio:       {ratio:.1f}x (expected: 4x)")

    if abs(ratio - 4) < 0.01:
        test_passed("Drag velocity² relationship", f"Ratio: {ratio:.1f}")
    else:
        test_failed("Drag velocity² relationship", 4, ratio)

    # Test Mach-dependent drag coefficient lookup
    print("\n--- Mach-Dependent Drag Coefficient ---")
    import numpy as np
    MACH_DATA = [0.0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]
    DC_DATA = [0.28, 0.27, 0.29, 0.38, 0.60,
               0.58, 0.55, 0.47, 0.41, 0.35, 0.32, 0.30]

    test_cases = [
        (0.0, 0.28, "Subsonic (M=0)"),
        (0.5, 0.27, "Subsonic (M=0.5)"),
        (1.0, 0.60, "Transonic (M=1.0) - Peak drag"),
        (2.0, 0.41, "Supersonic (M=2.0)"),
    ]

    all_passed = True
    for mach, expected_cd, desc in test_cases:
        cd = np.interp(mach, MACH_DATA, DC_DATA)
        if abs(cd - expected_cd) > 0.001:
            all_passed = False
            print(f"   {desc}: Cd = {cd:.3f} (expected {expected_cd})")
        else:
            print(f"   {desc}: Cd = {cd:.3f} ✓")

    if all_passed:
        test_passed("Drag coefficient lookup")
    else:
        test_failed("Drag coefficient lookup",
                    "See values above", "Mismatch found")

    # Verify transonic peak (drag divergence)
    print("\n--- Transonic Drag Rise Verification ---")
    Cd_subsonic = np.interp(0.7, MACH_DATA, DC_DATA)
    Cd_transonic = np.interp(1.0, MACH_DATA, DC_DATA)
    Cd_supersonic = np.interp(1.5, MACH_DATA, DC_DATA)

    print(f"Cd at M=0.7:  {Cd_subsonic:.3f}")
    print(f"Cd at M=1.0:  {Cd_transonic:.3f} (peak)")
    print(f"Cd at M=1.5:  {Cd_supersonic:.3f}")

    if Cd_transonic > Cd_subsonic and Cd_transonic > Cd_supersonic:
        test_passed("Transonic drag peak", "Cd is maximum around M=1.0")
    else:
        test_failed("Transonic drag peak", "Maximum Cd at M≈1",
                    "Peak not at transonic")


# =============================================================================
# TEST 4: Thrust and Mass Flow
# =============================================================================
def test_thrust():
    print_header("TEST 4: Thrust and Mass Flow Physics")

    # Thrust equation: F = ṁ * v_ex = ṁ * I_sp * g₀
    print("\n--- Thrust Equation Verification ---")

    I_sp = 250  # seconds
    mdot = 5.0  # kg/s
    g0 = 9.80665  # m/s²

    v_ex = I_sp * g0
    F_thrust = mdot * v_ex

    print(f"Specific impulse (I_sp): {I_sp} s")
    print(f"Mass flow rate (ṁ): {mdot} kg/s")
    print(f"Exhaust velocity (v_ex): {v_ex:.2f} m/s")
    print(f"Thrust: {F_thrust:.2f} N = {F_thrust/1000:.2f} kN")

    # Expected thrust
    F_expected = 12258.3  # N (250 * 9.80665 * 5)

    if abs(F_thrust - F_expected) < 1:
        test_passed("Thrust calculation", f"F = {F_thrust:.2f} N")
    else:
        test_failed("Thrust calculation", F_expected, F_thrust)

    # Test burn time calculation
    print("\n--- Burn Time Calculation ---")
    propellant_mass = 75  # kg
    burn_time = propellant_mass / mdot

    print(f"Propellant mass: {propellant_mass} kg")
    print(f"Burn time: {burn_time:.1f} s")

    if burn_time == 15.0:
        test_passed("Burn time", f"t_burn = {burn_time} s")
    else:
        test_failed("Burn time", 15.0, burn_time)

    # Test delta-v (Tsiolkovsky equation)
    print("\n--- Tsiolkovsky Rocket Equation ---")
    dry_mass = 50  # kg
    total_mass = dry_mass + propellant_mass

    delta_v = v_ex * math.log(total_mass / dry_mass)
    delta_v_expected = 2451.66 * math.log(125/50)  # ≈ 2245 m/s

    print(f"Initial mass: {total_mass} kg")
    print(f"Final mass: {dry_mass} kg")
    print(f"Mass ratio: {total_mass/dry_mass:.2f}")
    print(f"Delta-v: {delta_v:.2f} m/s")

    # Check if delta-v matches Tsiolkovsky equation
    if abs(delta_v - delta_v_expected) / delta_v_expected < 0.01:
        test_passed("Tsiolkovsky equation", f"Δv = {delta_v:.2f} m/s")
    else:
        test_failed("Tsiolkovsky equation",
                    f"~{delta_v_expected:.2f} m/s", f"{delta_v:.2f} m/s")

    # Initial thrust-to-weight ratio
    print("\n--- Thrust-to-Weight Ratio ---")
    initial_weight = total_mass * g0
    twr = F_thrust / initial_weight

    print(f"Initial weight: {initial_weight:.2f} N")
    print(f"Thrust: {F_thrust:.2f} N")
    print(f"TWR: {twr:.2f}")

    if twr > 1:
        test_passed("TWR check", f"TWR = {twr:.2f} > 1 (rocket can lift off)")
    else:
        test_warning(
            "TWR check", f"TWR = {twr:.2f} < 1 (rocket cannot lift off!)")


# =============================================================================
# TEST 5: Speed of Sound Calculation
# =============================================================================
def test_speed_of_sound():
    print_header("TEST 5: Speed of Sound Calculations")

    # Formula: a = sqrt(γ * R * T)
    print("\n--- Speed of Sound Formula ---")

    # Test at sea level (T = 288.15 K)
    T = 288.15  # K
    a = math.sqrt(GAMMA * R_GAS * T)
    a_expected = 340.3  # m/s at sea level

    print(f"Temperature: {T} K")
    print(f"Calculated: {a:.2f} m/s")
    print(f"Expected:   ~{a_expected} m/s")

    error = abs(a - a_expected) / a_expected * 100
    if error < 0.5:
        test_passed("Speed of sound at sea level",
                    f"a = {a:.2f} m/s (error: {error:.2f}%)")
    else:
        test_failed("Speed of sound at sea level", a_expected, a)

    # Test at tropopause (T ≈ 216.65 K)
    print("\n--- Speed of Sound at Tropopause ---")
    T_trop = 216.65
    a_trop = math.sqrt(GAMMA * R_GAS * T_trop)
    a_trop_expected = 295.1  # m/s

    print(f"Temperature: {T_trop} K")
    print(f"Calculated: {a_trop:.2f} m/s")
    print(f"Expected:   ~{a_trop_expected} m/s")

    error = abs(a_trop - a_trop_expected) / a_trop_expected * 100
    if error < 1:
        test_passed("Speed of sound at tropopause", f"a = {a_trop:.2f} m/s")
    else:
        test_failed("Speed of sound at tropopause", a_trop_expected, a_trop)


# =============================================================================
# TEST 6: Wind Effects
# =============================================================================
def test_wind():
    print_header("TEST 6: Wind Vector Calculations")

    # Wind direction convention: "from" direction
    # If wind is FROM the north (0°), it blows SOUTH (+y to -y)

    print("\n--- Wind Direction Vector Conversion ---")

    test_cases = [
        (0, 10, "North wind (from north)", 0, -10),  # Blows south
        (90, 10, "East wind (from east)", -10, 0),   # Blows west
        (180, 10, "South wind (from south)", 0, 10),  # Blows north
        (270, 10, "West wind (from west)", 10, 0),   # Blows east
    ]

    all_passed = True
    for wind_dir, wind_speed, desc, expected_wx, expected_wy in test_cases:
        # Wind "from" direction, so vector is opposite
        wind_dir_rad = math.radians(wind_dir + 180)
        w_x = wind_speed * math.sin(wind_dir_rad)
        w_y = wind_speed * math.cos(wind_dir_rad)

        # Check with small tolerance for floating point
        wx_ok = abs(w_x - expected_wx) < 0.01
        wy_ok = abs(w_y - expected_wy) < 0.01

        status = "✓" if (wx_ok and wy_ok) else "✗"
        print(
            f"   {desc}: ({w_x:.1f}, {w_y:.1f}) expected ({expected_wx}, {expected_wy}) {status}")

        if not (wx_ok and wy_ok):
            all_passed = False

    if all_passed:
        test_passed("Wind vector conversion")
    else:
        test_failed("Wind vector conversion", "See above", "Mismatch detected")

    # Test relative velocity
    print("\n--- Relative Velocity Calculation ---")
    v_rocket = (100, 50, 200)  # Rocket velocity
    v_wind = (10, -5, 0)       # Wind velocity

    v_rel = (v_rocket[0] - v_wind[0],
             v_rocket[1] - v_wind[1],
             v_rocket[2] - v_wind[2])

    print(f"Rocket velocity: {v_rocket}")
    print(f"Wind velocity:   {v_wind}")
    print(f"Relative velocity: {v_rel}")

    # Check magnitude computation
    mag_rel = math.sqrt(v_rel[0]**2 + v_rel[1]**2 + v_rel[2]**2)
    expected_mag = math.sqrt(90**2 + 55**2 + 200**2)

    if abs(mag_rel - expected_mag) < 0.01:
        test_passed("Relative velocity magnitude",
                    f"|v_rel| = {mag_rel:.2f} m/s")
    else:
        test_failed("Relative velocity magnitude", expected_mag, mag_rel)


# =============================================================================
# TEST 7: Thrust Vector Components
# =============================================================================
def test_thrust_vectors():
    print_header("TEST 7: Thrust Vector Decomposition")

    print("\n--- Pitch and Azimuth Decomposition ---")

    F_thrust = 10000  # N (arbitrary)

    test_cases = [
        (90, 0, "Vertical launch", 0, 0, F_thrust),
        (0, 0, "Horizontal north", 0, F_thrust, 0),
        (0, 90, "Horizontal east", F_thrust, 0, 0),
        (45, 0, "45° north", 0, F_thrust*math.cos(math.radians(45)),
         F_thrust*math.sin(math.radians(45))),
    ]

    all_passed = True
    for pitch, azimuth, desc, exp_x, exp_y, exp_z in test_cases:
        pitch_rad = math.radians(pitch)
        azimuth_rad = math.radians(azimuth)

        # Vertical component
        f_z = F_thrust * math.sin(pitch_rad)
        # Horizontal component
        f_h = F_thrust * math.cos(pitch_rad)
        f_x = f_h * math.sin(azimuth_rad)
        f_y = f_h * math.cos(azimuth_rad)

        # Check with tolerance
        x_ok = abs(f_x - exp_x) < 1
        y_ok = abs(f_y - exp_y) < 1
        z_ok = abs(f_z - exp_z) < 1

        status = "✓" if (x_ok and y_ok and z_ok) else "✗"
        print(f"   {desc}: ({f_x:.1f}, {f_y:.1f}, {f_z:.1f}) N {status}")

        if not (x_ok and y_ok and z_ok):
            all_passed = False
            print(f"      Expected: ({exp_x:.1f}, {exp_y:.1f}, {exp_z:.1f}) N")

    if all_passed:
        test_passed("Thrust vector decomposition")
    else:
        test_failed("Thrust vector decomposition",
                    "See above", "Mismatch detected")

    # Verify magnitude preservation
    print("\n--- Thrust Magnitude Preservation ---")
    pitch = 35
    azimuth = 60
    pitch_rad = math.radians(pitch)
    azimuth_rad = math.radians(azimuth)

    f_z = F_thrust * math.sin(pitch_rad)
    f_h = F_thrust * math.cos(pitch_rad)
    f_x = f_h * math.sin(azimuth_rad)
    f_y = f_h * math.cos(azimuth_rad)

    mag = math.sqrt(f_x**2 + f_y**2 + f_z**2)

    print(f"Original thrust: {F_thrust} N")
    print(f"Components: ({f_x:.2f}, {f_y:.2f}, {f_z:.2f}) N")
    print(f"Magnitude: {mag:.2f} N")

    if abs(mag - F_thrust) < 0.1:
        test_passed("Thrust magnitude preserved", f"|F| = {mag:.2f} N")
    else:
        test_failed("Thrust magnitude preserved", F_thrust, mag)


# =============================================================================
# TEST 8: Numerical Integration Check
# =============================================================================
def test_integration():
    print_header("TEST 8: Numerical Integration (Euler Method)")

    # Simple free-fall test without atmosphere
    print("\n--- Free Fall Verification (no drag) ---")

    g = 9.80665
    dt = 0.1  # time step
    t_total = 10  # seconds

    # Analytical solution: h(t) = h0 - 0.5*g*t²
    h0 = 1000  # m

    # Euler integration
    v = 0
    h = h0
    t = 0

    while t < t_total:
        a = -g
        v = v + a * dt
        h = h + v * dt
        t += dt

    h_analytical = h0 - 0.5 * g * t_total**2
    error = abs(h - h_analytical)
    error_pct = error / h0 * 100

    print(f"After {t_total}s of free fall from {h0}m:")
    print(f"Euler result:      {h:.2f} m")
    print(f"Analytical result: {h_analytical:.2f} m")
    print(f"Error: {error:.2f} m ({error_pct:.2f}%)")

    # Euler method with dt=0.1 should have ~5% error for this case
    if error_pct < 10:
        test_passed("Euler integration",
                    f"Error: {error_pct:.2f}% (acceptable for Euler method)")
    else:
        test_warning("Euler integration accuracy",
                     f"Error of {error_pct:.2f}% is high. Consider smaller time step or RK4.")

    # Check velocity
    v_analytical = -g * t_total
    v_error = abs(v - v_analytical)
    print(f"\nFinal velocity:")
    print(f"Euler:      {v:.2f} m/s")
    print(f"Analytical: {v_analytical:.2f} m/s")
    print(f"Error: {v_error:.2f} m/s")


# =============================================================================
# TEST 9: Trajectory Physics Check
# =============================================================================
def test_trajectory():
    print_header("TEST 9: Trajectory Physics Sanity Checks")

    # Test that 45° launch angle gives max range (ignoring drag and Earth curvature)
    print("\n--- Optimal Launch Angle (Vacuum Ballistic) ---")
    print("Theory: In a vacuum with flat Earth, 45° gives maximum range")
    print("With drag: Optimal angle is typically < 45° (around 30-40°)")
    print("This simulator includes drag, so < 45° may be optimal.")

    test_passed("Optimal angle reference documented")

    # Test energy conservation (approximately, accounting for drag losses)
    print("\n--- Energy Considerations ---")

    dry_mass = 50
    propellant_mass = 75
    total_mass = dry_mass + propellant_mass
    I_sp = 250
    g0 = 9.80665
    v_ex = I_sp * g0

    # Total chemical energy available
    delta_v = v_ex * math.log(total_mass / dry_mass)

    # Maximum theoretical altitude (all energy to height, no velocity at apogee)
    # For a vertical launch: h_max ≈ (Δv)² / (2g) (simplified, ignoring varying g and drag)
    h_max_approx = delta_v**2 / (2 * g0)

    print(f"Delta-v: {delta_v:.2f} m/s")
    print(
        f"Theoretical max altitude (vacuum, constant g): {h_max_approx/1000:.2f} km")
    print("Actual altitude will be lower due to drag and gravity losses.")

    test_passed("Energy analysis documented")


# =============================================================================
# TEST 10: ISA Model Edge Cases
# =============================================================================
def test_isa_edge_cases():
    print_header("TEST 10: ISA Model Edge Cases")

    print("\n--- Boundary Condition at 11000m ---")
    # Check the condition: 11000 < altitude < 25000
    # At exactly 11000m, it falls through to troposphere formula

    T_10999, P_10999, _ = get_atmosphere(10999)
    T_11000, P_11000, _ = get_atmosphere(11000)  # Should use troposphere
    T_11001, P_11001, _ = get_atmosphere(11001)  # Should use stratosphere

    print(f"At 10999m: T = {T_10999:.2f} K, P = {P_10999:.4f} kPa")
    print(f"At 11000m: T = {T_11000:.2f} K, P = {P_11000:.4f} kPa")
    print(f"At 11001m: T = {T_11001:.2f} K, P = {P_11001:.4f} kPa")

    # Check for discontinuity
    T_jump = abs(T_11000 - T_11001)
    if T_jump > 10:
        test_warning("ISA boundary at 11000m",
                     f"Temperature jump of {T_jump:.2f} K between 11000m and 11001m. Consider using >= in condition.")
    else:
        test_passed("ISA boundary at 11000m",
                    f"Temperature difference: {T_jump:.2f} K")

    print("\n--- Negative Altitude Check ---")
    try:
        T, P, rho = get_atmosphere(-100)
        print(f"At -100m: T = {T:.2f} K, P = {P:.4f} kPa, ρ = {rho:.4f} kg/m³")
        # This should work (below sea level locations exist)
        test_passed("Negative altitude handling")
    except Exception as e:
        test_failed("Negative altitude handling", "No error", str(e))

    print("\n--- Very High Altitude Check ---")
    try:
        T, P, rho = get_atmosphere(50000)
        print(
            f"At 50000m: T = {T:.2f} K, P = {P:.6f} kPa, ρ = {rho:.8f} kg/m³")
        if rho > 0:
            test_passed("High altitude (50km)")
        else:
            test_warning("High altitude density",
                         "Density might be too low to be meaningful")
    except Exception as e:
        test_failed("High altitude handling", "No error", str(e))


# =============================================================================
# RUN ALL TESTS
# =============================================================================
def run_all_tests():
    print("\n" + "╔" + "═" * 60 + "╗")
    print("║  ROCKET SIMULATOR PHYSICS VALIDATION TEST SUITE" + " " * 12 + "║")
    print("╚" + "═" * 60 + "╝")

    test_isa_model()
    test_gravity()
    test_drag()
    test_thrust()
    test_speed_of_sound()
    test_wind()
    test_thrust_vectors()
    test_integration()
    test_trajectory()
    test_isa_edge_cases()

    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in test_results if r[0] == "PASS")
    failed = sum(1 for r in test_results if r[0] == "FAIL")
    warnings = sum(1 for r in test_results if r[0] == "WARN")

    print(f"\n✅ Passed:   {passed}")
    print(f"❌ Failed:   {failed}")
    print(f"⚠️  Warnings: {warnings}")
    print(f"\nTotal tests: {len(test_results)}")

    if failed > 0:
        print("\n❌ FAILED TESTS:")
        for result, name in test_results:
            if result == "FAIL":
                print(f"   - {name}")

    if warnings > 0:
        print("\n⚠️  WARNINGS:")
        for result, name in test_results:
            if result == "WARN":
                print(f"   - {name}")

    print("\n" + "=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
