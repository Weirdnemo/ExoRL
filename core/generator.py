"""
generator.py — Procedural planet generation with presets and full toggleability.
"""

from __future__ import annotations
import random
import math
from typing import Optional

from core.planet import (
    Planet, AtmosphereConfig, TerrainConfig, MagneticFieldConfig,
    OblatenessConfig, MoonConfig, AtmosphereComposition, TerrainType,
    MagneticFieldStrength, G, R_EARTH, M_EARTH
)


# ── Preset atmospheres by composition ─────────────────────────────────────────
ATMOSPHERE_PRESETS: dict[AtmosphereComposition, dict] = {
    AtmosphereComposition.NONE: dict(
        enabled=False, surface_pressure=0, surface_density=0,
        scale_height=1, surface_temp=50
    ),
    AtmosphereComposition.CO2_THIN: dict(   # Mars-like
        enabled=True, surface_pressure=636, surface_density=0.015,
        scale_height=10_800, surface_temp=210, lapse_rate=0.004
    ),
    AtmosphereComposition.CO2_THICK: dict(  # Venus-like
        enabled=True, surface_pressure=9_200_000, surface_density=65.0,
        scale_height=15_000, surface_temp=737, lapse_rate=0.0078
    ),
    AtmosphereComposition.EARTH_LIKE: dict(
        enabled=True, surface_pressure=101_325, surface_density=1.225,
        scale_height=8_500, surface_temp=288, lapse_rate=0.0065
    ),
    AtmosphereComposition.NITROGEN: dict(
        enabled=True, surface_pressure=101_325, surface_density=1.16,
        scale_height=9_000, surface_temp=300, lapse_rate=0.006
    ),
    AtmosphereComposition.HYDROGEN: dict(   # gas giant envelope
        enabled=True, surface_pressure=100_000, surface_density=0.09,
        scale_height=27_000, surface_temp=165, lapse_rate=0.002
    ),
    AtmosphereComposition.METHANE: dict(    # Titan-like
        enabled=True, surface_pressure=146_700, surface_density=5.3,
        scale_height=40_000, surface_temp=94, lapse_rate=0.0009
    ),
}

# ── Named planet presets ───────────────────────────────────────────────────────
def preset_earth() -> Planet:
    return Planet(
        name="Earth",
        radius=R_EARTH, mass=M_EARTH,
        rotation_period=86_164.1,
        atmosphere=AtmosphereConfig(**ATMOSPHERE_PRESETS[AtmosphereComposition.EARTH_LIKE],
                                    composition=AtmosphereComposition.EARTH_LIKE),
        terrain=TerrainConfig(terrain_type=TerrainType.OCEANIC, max_elevation=8_848,
                              min_elevation=-11_000),
        magnetic_field=MagneticFieldConfig(enabled=True,
                                           strength=MagneticFieldStrength.MEDIUM,
                                           radiation_belt_enabled=True),
        oblateness=OblatenessConfig(enabled=True, J2=1.08263e-3, flattening=1/298.257),
        moons=MoonConfig(enabled=True, count=1, mass_fraction=0.01230,
                         orbit_radius=384_400e3),
    )

def preset_mars() -> Planet:
    return Planet(
        name="Mars",
        radius=3_389_500, mass=6.4171e23,
        rotation_period=88_642,
        atmosphere=AtmosphereConfig(**ATMOSPHERE_PRESETS[AtmosphereComposition.CO2_THIN],
                                    composition=AtmosphereComposition.CO2_THIN),
        terrain=TerrainConfig(terrain_type=TerrainType.CRATERED, max_elevation=21_900,
                              min_elevation=-7_200),
        magnetic_field=MagneticFieldConfig(enabled=False),
        oblateness=OblatenessConfig(enabled=True, J2=1.960e-3, flattening=1/169.8),
        moons=MoonConfig(enabled=True, count=2, mass_fraction=1.8e-8,
                         orbit_radius=9_376e3),
    )

def preset_venus() -> Planet:
    return Planet(
        name="Venus",
        radius=6_051_800, mass=4.8675e24,
        rotation_period=243.0 * 86400,
        atmosphere=AtmosphereConfig(**ATMOSPHERE_PRESETS[AtmosphereComposition.CO2_THICK],
                                    composition=AtmosphereComposition.CO2_THICK),
        terrain=TerrainConfig(terrain_type=TerrainType.VOLCANIC),
        magnetic_field=MagneticFieldConfig(enabled=False),
        oblateness=OblatenessConfig(enabled=False),
        moons=MoonConfig(enabled=False),
    )

def preset_moon() -> Planet:
    return Planet(
        name="Moon",
        radius=1_737_400, mass=7.342e22,
        rotation_period=27.32 * 86400,
        atmosphere=AtmosphereConfig(enabled=False),
        terrain=TerrainConfig(terrain_type=TerrainType.CRATERED, max_elevation=10_786,
                              min_elevation=-9_000),
        magnetic_field=MagneticFieldConfig(enabled=False),
        oblateness=OblatenessConfig(enabled=True, J2=2.027e-4),
        moons=MoonConfig(enabled=False),
    )

def preset_titan() -> Planet:
    return Planet(
        name="Titan",
        radius=2_574_730, mass=1.3452e23,
        rotation_period=15.94 * 86400,
        atmosphere=AtmosphereConfig(**ATMOSPHERE_PRESETS[AtmosphereComposition.METHANE],
                                    composition=AtmosphereComposition.METHANE),
        terrain=TerrainConfig(terrain_type=TerrainType.OCEANIC),
        magnetic_field=MagneticFieldConfig(enabled=False),
        oblateness=OblatenessConfig(enabled=False),
        moons=MoonConfig(enabled=False),
    )

PRESETS = {
    "earth": preset_earth,
    "mars":  preset_mars,
    "venus": preset_venus,
    "moon":  preset_moon,
    "titan": preset_titan,
}


# ── Random planet generator ───────────────────────────────────────────────────
class PlanetGenerator:
    """
    Procedurally generate planets with individually toggleable features.
    Every feature has its own on/off flag; the generator respects them all.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.seed = seed

    # ── Feature toggle registry ───────────────────────────────────────────────
    def generate(
        self,
        name: str = "Random-Planet",
        # Size constraints
        radius_range: tuple[float, float] = (0.1 * R_EARTH, 4.0 * R_EARTH),
        density_range: tuple[float, float] = (1_500, 8_000),  # [kg/m³]
        # Feature toggles
        atmosphere_enabled: bool = True,
        rotation_enabled: bool = True,
        terrain_enabled: bool = True,
        magnetic_field_enabled: bool = False,
        oblateness_enabled: bool = False,
        moons_enabled: bool = False,
        # Atmosphere options
        atmosphere_composition: Optional[AtmosphereComposition] = None,
        allow_no_atmosphere: bool = True,
        # Rotation options
        retrograde_allowed: bool = True,
        tidally_locked_allowed: bool = True,
        # Terrain options
        terrain_type: Optional[TerrainType] = None,
        # Seed passthrough
        terrain_seed: Optional[int] = None,
    ) -> Planet:
        rng = self.rng

        # ── Size / Mass ────────────────────────────────────────────────────────
        radius = rng.uniform(*radius_range)
        density = rng.uniform(*density_range)
        volume = (4/3) * math.pi * radius**3
        mass = density * volume

        # ── Rotation ──────────────────────────────────────────────────────────
        if rotation_enabled:
            # Log-uniform period between 4h and 300 days
            log_min = math.log(4 * 3600)
            log_max = math.log(300 * 86400)
            period = math.exp(rng.uniform(log_min, log_max))
            # Tidal locking for very small/close bodies (heuristic)
            if tidally_locked_allowed and rng.random() < 0.1:
                period = rng.uniform(10 * 86400, 100 * 86400)
            # Retrograde rotation
            if retrograde_allowed and rng.random() < 0.15:
                period = -period  # negative = retrograde convention
        else:
            period = 86_400.0

        # ── Atmosphere ────────────────────────────────────────────────────────
        if atmosphere_enabled:
            # No-atm probability scales with small bodies
            no_atm_prob = 0.3 if (allow_no_atmosphere and radius < 0.4 * R_EARTH) else 0.05
            if allow_no_atmosphere and rng.random() < no_atm_prob:
                atm = AtmosphereConfig(enabled=False)
            else:
                comp = atmosphere_composition or rng.choice(list(AtmosphereComposition)[1:])  # skip NONE
                base = dict(ATMOSPHERE_PRESETS.get(comp,
                            ATMOSPHERE_PRESETS[AtmosphereComposition.EARTH_LIKE]))
                # Randomize within ±30% of preset values
                def jitter(v, pct=0.3):
                    return v * rng.uniform(1 - pct, 1 + pct)
                atm = AtmosphereConfig(
                    enabled=True,
                    composition=comp,
                    scale_height=jitter(base["scale_height"]),
                    surface_pressure=jitter(base["surface_pressure"]),
                    surface_density=jitter(base["surface_density"]),
                    surface_temp=jitter(base["surface_temp"], 0.15),
                    lapse_rate=jitter(base.get("lapse_rate", 0.006), 0.2),
                    drag_coeff_multiplier=rng.uniform(0.8, 1.5),
                    wind_enabled=rng.random() < 0.4,
                    wind_speed_mps=rng.uniform(0, 150),
                    wind_direction_deg=rng.uniform(0, 360),
                )
        else:
            atm = AtmosphereConfig(enabled=False)

        # ── Terrain ───────────────────────────────────────────────────────────
        if terrain_enabled:
            t_type = terrain_type or rng.choice(list(TerrainType)[:-1])  # skip RANDOM
            max_elev = rng.uniform(1_000, 30_000)
            terrain = TerrainConfig(
                enabled=True,
                terrain_type=t_type,
                max_elevation=max_elev,
                min_elevation=-rng.uniform(2_000, max_elev),
                roughness=rng.uniform(0.1, 0.95),
                seed=terrain_seed or rng.randint(0, 2**32 - 1),
            )
        else:
            terrain = TerrainConfig(enabled=False)

        # ── Magnetic field ────────────────────────────────────────────────────
        if magnetic_field_enabled:
            strength = rng.choice(list(MagneticFieldStrength))
            mag = MagneticFieldConfig(
                enabled=True,
                strength=strength,
                tilt_deg=rng.uniform(0, 30),
                radiation_belt_enabled=(strength != MagneticFieldStrength.NONE
                                        and rng.random() < 0.5),
                inner_belt_altitude=rng.uniform(500e3, 2_000e3),
                outer_belt_altitude=rng.uniform(10_000e3, 40_000e3),
            )
        else:
            mag = MagneticFieldConfig(enabled=False)

        # ── Oblateness ────────────────────────────────────────────────────────
        if oblateness_enabled:
            period_abs = abs(period)
            J2 = rng.uniform(1e-4, 2e-3) * (86400 / max(period_abs, 3600))**0.5
            obl = OblatenessConfig(
                enabled=True,
                J2=J2,
                J3=J2 * rng.uniform(-0.01, 0.01),
                flattening=J2 * rng.uniform(0.5, 1.5),
            )
        else:
            obl = OblatenessConfig(enabled=False)

        # ── Moons ─────────────────────────────────────────────────────────────
        if moons_enabled:
            n_moons = rng.choices([1, 2, 3, 4, 5], weights=[40, 25, 15, 10, 10])[0]
            moon_mass_frac = rng.uniform(1e-5, 0.02)
            moon_orbit = rng.uniform(5 * radius, 500 * radius)
            moon_cfg = MoonConfig(
                enabled=True,
                count=n_moons,
                mass_fraction=moon_mass_frac,
                orbit_radius=moon_orbit,
            )
        else:
            moon_cfg = MoonConfig(enabled=False)

        return Planet(
            name=name,
            radius=radius,
            mass=mass,
            rotation_enabled=rotation_enabled,
            rotation_period=abs(period),
            atmosphere=atm,
            terrain=terrain,
            magnetic_field=mag,
            oblateness=obl,
            moons=moon_cfg,
        )

    def batch(self, n: int, **kwargs) -> list[Planet]:
        """Generate n planets with identical toggle settings."""
        return [self.generate(name=f"Planet-{i+1:03d}", **kwargs) for i in range(n)]
