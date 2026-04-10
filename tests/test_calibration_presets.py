from __future__ import annotations

import math

from exorl.core.generator import PRESETS
from exorl.core.planet import AtmosphereComposition


def test_earth_mars_venus_moon_calibration_signals() -> None:
    earth = PRESETS["earth"]()
    mars = PRESETS["mars"]()
    venus = PRESETS["venus"]()
    moon = PRESETS["moon"]()

    # Radius checks vs known values in generator presets.
    assert math.isclose(earth.radius, 6_371_000, rel_tol=0.0, abs_tol=1_000)
    assert math.isclose(mars.radius, 3_389_500, rel_tol=0.0, abs_tol=1_000)
    assert math.isclose(venus.radius, 6_051_800, rel_tol=0.0, abs_tol=1_000)
    assert math.isclose(moon.radius, 1_737_400, rel_tol=0.0, abs_tol=1_000)

    # Atmospheric calibration fingerprints.
    assert earth.atmosphere.enabled
    assert earth.atmosphere.composition == AtmosphereComposition.EARTH_LIKE
    assert mars.atmosphere.enabled
    assert mars.atmosphere.composition == AtmosphereComposition.CO2_THIN
    assert venus.atmosphere.enabled
    assert venus.atmosphere.composition == AtmosphereComposition.CO2_THICK
    assert not moon.atmosphere.enabled

    # Relative gravity ordering should be stable for these presets.
    assert venus.surface_gravity > mars.surface_gravity
    assert earth.surface_gravity > moon.surface_gravity
