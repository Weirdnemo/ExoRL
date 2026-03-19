"""
demo.py — Full sandbox demo: generate planets, run physics, visualize.
"""

import sys, os
# Make sure core/ and visualization/ are importable from wherever you run this
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import (
    PlanetGenerator, PRESETS,
    OrbitalIntegrator, ThrusterConfig, AeroConfig, SpacecraftState,
    state_to_orbital_elements,
)
from visualization import (
    plot_planet_cross_section,
    plot_atmosphere_profile,
    plot_trajectory_2d,
    plot_mission_telemetry,
    plot_planet_comparison,
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

# All features ON
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

# Atmosphere OFF
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

# Batch of 5 planets
batch = gen.batch(5, atmosphere_enabled=True, terrain_enabled=True)
print("[Batch of 5 random planets]")
for p in batch:
    print(f"  {p.name:15s}  R={p.radius/1e6:.2f}Mm  g={p.surface_gravity:.2f}m/s²"
          f"  atm={'Y' if p.atmosphere.enabled else 'N'}")
print()

# ── 3. Orbital mechanics demo ─────────────────────────────────────────────────
print("=" * 60)
print("ORBITAL MECHANICS DEMO")
print("=" * 60)

target_planet = PRESETS["mars"]()
target_alt = 300_000  # 300 km

print(f"Planet: {target_planet.name}")
print(f"Circular orbit speed at {target_alt/1e3:.0f} km: "
      f"{target_planet.circular_orbit_speed(target_alt)/1e3:.3f} km/s")
print(f"Orbital period: {target_planet.circular_orbit_period(target_alt)/60:.1f} min")
dv1, dv2 = target_planet.hohmann_delta_v(200_000, 500_000)
print(f"Hohmann 200→500 km: Δv1={dv1:.1f} m/s, Δv2={dv2:.1f} m/s")
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
    x=target_planet.radius + approach_alt, y=0, z=0,
    vx=0, vy=v0, vz=0,
    mass=2000, dry_mass=500,
)

burn_direction = np.array([0, -1, 0])
schedule = [(0, 180, burn_direction * 2500)]
history = integrator.propagate(initial, duration=7200, dt=5, thrust_schedule=schedule)
print(f"Propagated {len(history)} steps ({len(history)*5/60:.1f} min)")

final = history[-1]
elems = state_to_orbital_elements(final, target_planet.mu)
print("Final orbital elements:")
for k, v in elems.items():
    print(f"  {k}: {v:.4g}")
print()

# ── 4. Visualizations ─────────────────────────────────────────────────────────
print("=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0d0d1a")

# Row 1: planet cross-sections
for idx, (name, factory) in enumerate(list(PRESETS.items())[:4]):
    ax = fig.add_subplot(3, 4, idx + 1)
    ax.set_facecolor("#08080f")
    plot_planet_cross_section(factory(), ax=ax)

# Row 2: atmosphere profiles
atm_planets = [PRESETS["earth"](), PRESETS["venus"](),
                PRESETS["mars"](),  PRESETS["titan"]()]
for idx, p in enumerate(atm_planets):
    ax = fig.add_subplot(3, 4, idx + 5)
    ax.set_facecolor("#141428")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="#aaa")
    plot_atmosphere_profile(p, ax=ax, max_altitude_km=100)

# Row 3 left: trajectory
ax_traj = fig.add_subplot(3, 4, (9, 10))
ax_traj.set_facecolor("#08080f")
plot_trajectory_2d(target_planet, history, ax=ax_traj,
                   target_altitude=target_alt, color_by="speed")

# Row 3 right: comparison bars
all_planets = [PRESETS[k]() for k in PRESETS]
names  = [p.name for p in all_planets]
gravs  = [p.surface_gravity for p in all_planets]
escvs  = [p.escape_velocity / 1e3 for p in all_planets]
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(names)))

ax_c1 = fig.add_subplot(3, 4, 11)
ax_c1.set_facecolor("#141428")
ax_c1.barh(names, gravs, color=colors)
ax_c1.set_xlabel("g (m/s²)", color="#ccc")
ax_c1.set_title("Surface Gravity", color="white", fontsize=9)
ax_c1.tick_params(colors="#aaa")

ax_c2 = fig.add_subplot(3, 4, 12)
ax_c2.set_facecolor("#141428")
ax_c2.barh(names, escvs, color=colors)
ax_c2.set_xlabel("v_esc (km/s)", color="#ccc")
ax_c2.set_title("Escape Velocity", color="white", fontsize=9)
ax_c2.tick_params(colors="#aaa")

for ax in [ax_c1, ax_c2]:
    for s in ax.spines.values():
        s.set_edgecolor("#333")

fig.suptitle("Planet Sandbox — Full Overview", color="white",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, "sandbox_overview.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
print("Saved: sandbox_overview.png")

fig2 = plot_mission_telemetry(history, target_planet, target_alt)
fig2.savefig(os.path.join(out_dir, "mission_telemetry.png"),
             dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
print("Saved: mission_telemetry.png")
print("\nDone! ✓")
