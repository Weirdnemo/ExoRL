"""
demo.py — Full sandbox demo: generate planets, run physics, visualize.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from planet_rl.core import (
    PRESETS,
    AeroConfig,
    OrbitalIntegrator,
    PlanetGenerator,
    SpacecraftState,
    ThrusterConfig,
    state_to_orbital_elements,
)
from planet_rl.visualization import (
    plot_atmosphere_profile,
    plot_atmosphere_single,
    plot_mission_telemetry,
    plot_planet_cross_section,
    plot_trajectory_2d,
)

# ── 1. Preset planets ─────────────────────────────────────────────────────────
print("=" * 60)
print("PRESET PLANETS")
print("=" * 60)
for name, factory in PRESETS.items():
    p = factory()
    print(p.summary())
    print()

# ── 2. Random planet generation ───────────────────────────────────────────────
print("=" * 60)
print("RANDOM PLANET GENERATION")
print("=" * 60)

gen = PlanetGenerator(seed=1337)

p_full = gen.generate(
    name="Zephyria",
    atmosphere_enabled=True,
    terrain_enabled=True,
    magnetic_field_enabled=True,
    oblateness_enabled=True,
    moons_enabled=True,
)
print("[Full-featured random planet]")
print(p_full.summary())
print()

p_no_atm = gen.generate(
    name="Barren-Alpha",
    atmosphere_enabled=False,
    terrain_enabled=True,
    magnetic_field_enabled=False,
    oblateness_enabled=False,
)
print("[No atmosphere]")
print(p_no_atm.summary())
print()

batch = gen.batch(5, atmosphere_enabled=True, terrain_enabled=True)
print("[Batch of 5 random planets]")
for p in batch:
    print(
        f"  {p.name:15s}  R={p.radius / 1e6:.2f}Mm  g={p.surface_gravity:.2f}m/s²"
        f"  atm={'Y' if p.atmosphere.enabled else 'N'}"
    )
print()

# ── 3. Orbital mechanics demo ─────────────────────────────────────────────────
print("=" * 60)
print("ORBITAL MECHANICS DEMO")
print("=" * 60)

target_planet = PRESETS["mars"]()
target_alt = 300_000

print(f"Planet: {target_planet.name}")
print(
    f"Circular orbit speed at {target_alt / 1e3:.0f} km: "
    f"{target_planet.circular_orbit_speed(target_alt) / 1e3:.3f} km/s"
)
print(f"Orbital period: {target_planet.circular_orbit_period(target_alt) / 60:.1f} min")
dv1, dv2 = target_planet.hohmann_delta_v(200_000, 500_000)
print(f"Hohmann 200->500 km: dv1={dv1:.1f} m/s, dv2={dv2:.1f} m/s")
print()

integrator = OrbitalIntegrator(
    planet=target_planet,
    thruster=ThrusterConfig(max_thrust=3000, Isp=320),
    aero=AeroConfig(enabled=target_planet.atmosphere.enabled),
)

approach_alt = target_alt + 200_000
v_circ = target_planet.circular_orbit_speed(approach_alt)
v0 = v_circ * 1.25

initial = SpacecraftState(
    x=target_planet.radius + approach_alt,
    y=0,
    z=0,
    vx=0,
    vy=v0,
    vz=0,
    mass=2000,
    dry_mass=500,
)

burn_direction = np.array([0, -1, 0])
schedule = [(0, 180, burn_direction * 2500)]
history = integrator.propagate(initial, duration=7200, dt=5, thrust_schedule=schedule)
print(f"Propagated {len(history)} steps ({len(history) * 5 / 60:.1f} min)")

final = history[-1]
elems = state_to_orbital_elements(final, target_planet.mu)
print("Final orbital elements:")
for k, v in elems.items():
    print(f"  {k}: {v:.4g}")
print()
