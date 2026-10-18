# Copyright (c) 2025 The Adirondax Team.

"""
Physical Constants in cgs units
"""

constants = {
    "speed_of_light": 2.998e10,  # cm/s
    "gravitational_constant": 6.6743e-8,  # cm^3 g^-1 s^-2
    "planck_constant": 6.6261e-27,  # erg s
    "boltzmann_constant": 1.380649e-16,  # erg K^-1
    "radiation_constant": 7.5657e-15,  # erg cm^-3 K^-4
    "electron_volt": 1.602176634e-12,  # erg
    "electron_charge": 4.8032e-10,  # statC
    "electron_mass": 9.1094e-28,  # g
    "proton_mass": 1.6726e-24,  # g
    "neutron_mass": 1.6749e-24,  # g
    "atomic_mass_unit": 1.6605e-24,  # g
    "stefan_boltzmann_constant": 5.6704e-5,  # erg cm^-2 s^-1 K^-4
    "avogadro_number": 6.02214076e23,  # mol^-1
    "gas_constant": 8.3145e7,  # erg mol^-1 K^-1
    "faraday_constant": 2.8926e14,  # statC mol^-1
    "rydberg_constant": 1.09737316e5,  # cm^-1
}

# dimensions of each constant, as powers of (mass, length, time, temperature)
dimensions = {
    "speed_of_light": (0, 1, -1, 0),
    "gravitational_constant": (-1, 3, -2, 0),
    "planck_constant": (1, 2, -1, 0),
    "boltzmann_constant": (1, 2, -2, -1),
    "radiation_constant": (1, -1, -2, -4),
    "electron_volt": (1, 2, -2, 0),
    "electron_charge": (0.5, 1.5, -1, 0),
    "electron_mass": (1, 0, 0, 0),
    "proton_mass": (1, 0, 0, 0),
    "neutron_mass": (1, 0, 0, 0),
    "atomic_mass_unit": (1, 0, 0, 0),
    "stefan_boltzmann_constant": (1, 0, -3, -4),
    "avogadro_number": (0, 0, 0, 0),
    "gas_constant": (1, 2, -2, -1),
    "faraday_constant": (0.5, 1.5, -1, 0),
    "rydberg_constant": (0, -1, 0, 0),
}


def code_constants(units, overrides=None):
    """
    Return the physical constants in code units.

    Parameters
    ----------
    units: dict
      The size of the code units of 'mass', 'length', 'time' and 'temperature',
      in g, cm, s and K.
    overrides: dict
      Constants to set directly, in code units, in place of their physical
      values (None entries are ignored). Useful for problems posed in terms of
      dimensionless numbers, such as the ratio of the speed of light to the
      sound speed.

    Returns
    -------
    constants: dict
      Every constant of adirondax.constants, in code units.
    """

    scale = [units[k] for k in ("mass", "length", "time", "temperature")]

    converted = {}
    for name, value in constants.items():
        for s, p in zip(scale, dimensions[name]):
            value = value / s**p
        converted[name] = value

    for name, value in (overrides or {}).items():
        if value is not None:
            converted[name] = value

    return converted
