'''A function to find temperature, pressure, and air density from altitude using the ISA model.'''
import math


def get_atmosphere(altitude):
    if altitude > 25000:  # Upper Stratosphere
        temperature = -131.21 + 0.00299 * altitude
        pressure = 2.488 * ((temperature + 273.1) / 216.6) ** -11.388
    elif 11000 < altitude < 25000:  # Lower Stratosphere
        temperature = -56.46
        pressure = 22.65 * (math.e ** (1.73 - 0.000157 * altitude))
    else:  # Troposphere
        temperature = 15.04 - 0.00649 * altitude
        pressure = 101.29 * ((temperature + 273.1) / (288.08)) ** 5.256

    rho = pressure / (0.2869 * (temperature + 273.1))

    temperature += 273.15   # Convert temperature to Kelvins from Celsius

    return temperature, pressure, rho
