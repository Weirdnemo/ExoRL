"""
test_planets.py  —  Planet generation and visualisation.
Run from Planet-RL/:  python test_planets.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core import PlanetGenerator, PRESETS, AtmosphereComposition, TerrainType
from visualization import (
    plot_planet_cross_section, plot_atmosphere_profile,
    plot_planet_comparison, save_figure, apply_journal_style,
    WONG, W_BLUE, W_RED, W_GREEN, W_ORANGE,
    FT, FL, FK, FA, FG, LW,
)
from visualization.visualizer import _ax

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
apply_journal_style()

print("=" * 54)
print("Planet-RL — Generation & Visualisation Test")
print("=" * 54)


# ─────────────────────────────────────────────────────
# FIG 1a  — Preset planet cross-sections (1 row × 5)
# ─────────────────────────────────────────────────────
print("\n[1/8] Preset cross-sections ...")
presets = [PRESETS[k]() for k in PRESETS]

fig1a, axes1a = plt.subplots(1, 5, figsize=(7.08, 2.8),
    gridspec_kw=dict(wspace=0.05, left=0.01, right=0.99,
                     top=0.88, bottom=0.04))
fig1a.patch.set_facecolor("white")
ref1a = max(p.radius for p in presets)
for ax, p in zip(axes1a, presets):
    plot_planet_cross_section(p, ax=ax, ref_radius=ref1a)
fig1a.suptitle("Solar-system analogue presets", fontsize=FT+1, fontweight="bold")
save_figure(fig1a, "fig1a_preset_crosssec", output_dir=OUT)
plt.close(fig1a)


# ─────────────────────────────────────────────────────
# FIG 1b  — Preset atmosphere profiles (5 planet × 3 panel)
#   Each row = 1 planet, 3 panels side by side
# ─────────────────────────────────────────────────────
print("[2/8] Preset atmosphere profiles ...")

fig1b, axes1b = plt.subplots(5, 3, figsize=(5.5, 8.5),
    gridspec_kw=dict(hspace=0.65, wspace=0.12,
                     left=0.12, right=0.97,
                     top=0.95, bottom=0.06))
fig1b.patch.set_facecolor("white")

for row, planet in enumerate(presets):
    row_axes = [axes1b[row, 0], axes1b[row, 1], axes1b[row, 2]]
    plot_atmosphere_profile(planet, axes=row_axes, max_altitude_km=150)

fig1b.suptitle("Preset planet atmosphere profiles", fontsize=FT+1, fontweight="bold")
save_figure(fig1b, "fig1b_preset_atm", output_dir=OUT)
plt.close(fig1b)


# ─────────────────────────────────────────────────────
# FIG 2a  — 6 random planet cross-sections
# ─────────────────────────────────────────────────────
print("[3/8] Random planet cross-sections ...")
gen = PlanetGenerator(seed=2024)
randoms = [gen.generate(name=f"P-{chr(65+i)}",
                        atmosphere_enabled=True, terrain_enabled=True,
                        magnetic_field_enabled=True, oblateness_enabled=True,
                        moons_enabled=True)
           for i in range(6)]

for p in randoms:
    print(f"  {p.name}  R={p.radius/1e6:.1f}Mm  "
          f"g={p.surface_gravity:.1f}  atm={p.atmosphere.composition.name}")

fig2a, axes2a = plt.subplots(1, 6, figsize=(7.08, 2.8),
    gridspec_kw=dict(wspace=0.05, left=0.01, right=0.99,
                     top=0.88, bottom=0.04))
fig2a.patch.set_facecolor("white")
ref2a = max(p.radius for p in randoms)
for ax, p in zip(axes2a, randoms):
    plot_planet_cross_section(p, ax=ax, ref_radius=ref2a)
fig2a.suptitle("Random planets — all features enabled", fontsize=FT+1, fontweight="bold")
save_figure(fig2a, "fig2a_random_crosssec", output_dir=OUT)
plt.close(fig2a)


# ─────────────────────────────────────────────────────
# FIG 2b  — 6 random atmosphere profiles
# ─────────────────────────────────────────────────────
print("[4/8] Random atmosphere profiles ...")

fig2b, axes2b = plt.subplots(6, 3, figsize=(5.5, 10.0),
    gridspec_kw=dict(hspace=0.65, wspace=0.12,
                     left=0.12, right=0.97,
                     top=0.96, bottom=0.05))
fig2b.patch.set_facecolor("white")

for row, planet in enumerate(randoms):
    plot_atmosphere_profile(planet, axes=[axes2b[row,0], axes2b[row,1], axes2b[row,2]],
                            max_altitude_km=150)

fig2b.suptitle("Random planet atmosphere profiles", fontsize=FT+1, fontweight="bold")
save_figure(fig2b, "fig2b_random_atm", output_dir=OUT)
plt.close(fig2b)


# ─────────────────────────────────────────────────────
# FIG 3  — Atmosphere zoo (6 compositions, 2×3 grid of triplets)
#   Each cell in the grid = 3 atmosphere panels
#   Grid: 2 rows × 3 planet-columns, each planet-column = 3 axes
# ─────────────────────────────────────────────────────
print("[5/8] Atmosphere zoo ...")

comp_info = [
    (AtmosphereComposition.CO2_THIN,   "CO2 thin (Mars)"),
    (AtmosphereComposition.CO2_THICK,  "CO2 thick (Venus)"),
    (AtmosphereComposition.EARTH_LIKE, "N2/O2 (Earth)"),
    (AtmosphereComposition.NITROGEN,   "N2 inert"),
    (AtmosphereComposition.HYDROGEN,   "H2 gas giant"),
    (AtmosphereComposition.METHANE,    "CH4 Titan"),
]
gen_atm = PlanetGenerator(seed=7)
atm_planets = [
    gen_atm.generate(name=label, atmosphere_enabled=True,
                     atmosphere_composition=comp,
                     terrain_enabled=False, magnetic_field_enabled=False,
                     oblateness_enabled=False, moons_enabled=False)
    for comp, label in comp_info
]

# Layout: 6 rows × 3 cols (one row per planet, 3 panels per row)
fig3, axes3 = plt.subplots(6, 3, figsize=(5.5, 10.0),
    gridspec_kw=dict(hspace=0.65, wspace=0.12,
                     left=0.12, right=0.97,
                     top=0.96, bottom=0.05))
fig3.patch.set_facecolor("white")

for row, planet in enumerate(atm_planets):
    plot_atmosphere_profile(planet, axes=[axes3[row,0], axes3[row,1], axes3[row,2]],
                            max_altitude_km=180)

fig3.suptitle("Atmosphere profiles by composition type",
              fontsize=FT+1, fontweight="bold")
save_figure(fig3, "fig3_atm_zoo", output_dir=OUT)
plt.close(fig3)


# ─────────────────────────────────────────────────────
# FIG 4  — Airless worlds (cross-sections)
# ─────────────────────────────────────────────────────
print("[6/8] Airless worlds ...")

terrain_info = [
    (TerrainType.CRATERED,    "Cratered"),
    (TerrainType.MOUNTAINOUS, "Mountainous"),
    (TerrainType.VOLCANIC,    "Volcanic"),
    (TerrainType.FLAT,        "Flat"),
]
gen_bare = PlanetGenerator(seed=99)
bare_planets = [
    gen_bare.generate(name=name, atmosphere_enabled=False,
                      terrain_enabled=True, terrain_type=ttype,
                      magnetic_field_enabled=False,
                      oblateness_enabled=False, moons_enabled=False)
    for ttype, name in terrain_info
]

fig4, axes4 = plt.subplots(1, 4, figsize=(7.08, 2.8),
    gridspec_kw=dict(wspace=0.05, left=0.01, right=0.99,
                     top=0.88, bottom=0.04))
fig4.patch.set_facecolor("white")
ref4 = max(p.radius for p in bare_planets)
for ax, planet in zip(axes4, bare_planets):
    plot_planet_cross_section(planet, ax=ax, ref_radius=ref4)
fig4.suptitle("Airless worlds — terrain archetypes", fontsize=FT+1, fontweight="bold")
save_figure(fig4, "fig4_airless", output_dir=OUT)
plt.close(fig4)


# ─────────────────────────────────────────────────────
# FIG 5  — Feature toggle
# ─────────────────────────────────────────────────────
print("[7/8] Feature toggle ...")

toggle_configs = [
    ("Atm only",
     dict(atmosphere_enabled=True,  terrain_enabled=False,
          magnetic_field_enabled=False, oblateness_enabled=False, moons_enabled=False)),
    ("+ Terrain",
     dict(atmosphere_enabled=True,  terrain_enabled=True,
          magnetic_field_enabled=False, oblateness_enabled=False, moons_enabled=False)),
    ("+ Mag",
     dict(atmosphere_enabled=True,  terrain_enabled=True,
          magnetic_field_enabled=True,  oblateness_enabled=False, moons_enabled=False)),
    ("+ J2 + Moons",
     dict(atmosphere_enabled=True,  terrain_enabled=True,
          magnetic_field_enabled=True,  oblateness_enabled=True, moons_enabled=True)),
]

fig5, axes5 = plt.subplots(1, 4, figsize=(7.08, 2.8),
    gridspec_kw=dict(wspace=0.05, left=0.01, right=0.99,
                     top=0.88, bottom=0.04))
fig5.patch.set_facecolor("white")
toggle_planets = []
for label, kwargs in toggle_configs:
    g = PlanetGenerator(seed=42)
    p = g.generate(name=label, radius_range=(0.85*6.371e6, 1.15*6.371e6),
                   density_range=(4800, 5800), **kwargs)
    toggle_planets.append(p)
ref5 = max(p.radius for p in toggle_planets)
for ax, p in zip(axes5, toggle_planets):
    plot_planet_cross_section(p, ax=ax, ref_radius=ref5)
fig5.suptitle("Feature toggle — identical planet seed", fontsize=FT+1, fontweight="bold")
save_figure(fig5, "fig5_toggle", output_dir=OUT)
plt.close(fig5)


# ─────────────────────────────────────────────────────
# FIG 6  — Batch statistics
# ─────────────────────────────────────────────────────
print("[8/8] Batch statistics ...")

gen_b = PlanetGenerator(seed=1337)
batch = gen_b.batch(50, atmosphere_enabled=True, terrain_enabled=True,
                    magnetic_field_enabled=True, oblateness_enabled=True,
                    moons_enabled=True)

radii   = np.array([p.radius/6.371e6 for p in batch])
gravs   = np.array([p.surface_gravity for p in batch])
escvs   = np.array([p.escape_velocity/1e3 for p in batch])
dens    = np.array([p.mean_density for p in batch])
atm_d   = np.array([p.atmosphere.surface_density
                    if p.atmosphere.enabled else 0.0 for p in batch])
n_moons = np.array([p.moons.count if p.moons.enabled else 0 for p in batch])
has_mag = np.array([p.magnetic_field.enabled for p in batch])

fig6, axes6 = plt.subplots(2, 3, figsize=(7.08, 4.5),
    gridspec_kw=dict(hspace=0.54, wspace=0.40,
                     left=0.10, right=0.97,
                     top=0.91, bottom=0.12))
fig6.patch.set_facecolor("white")

def _hist(ax, data, xlabel, title, color):
    _ax(ax)
    ax.hist(data, bins=12, color=color, edgecolor="white", lw=0.4)
    mu = data.mean()
    ax.axvline(mu, color=W_RED, lw=1.2, ls="--", label=f"μ={mu:.2f}")
    ax.set_xlabel(xlabel, fontsize=FL, labelpad=2)
    ax.set_ylabel("Count", fontsize=FL, labelpad=2)
    ax.set_title(title, fontsize=FT, fontweight="bold", pad=3)
    ax.tick_params(labelsize=FK)
    ax.legend(fontsize=FG, framealpha=0.9, edgecolor="#cccccc", fancybox=False)

_hist(axes6[0,0], radii, "Radius (R⊕)",    "(a) Radius",       W_BLUE)
_hist(axes6[0,1], gravs, "g (m/s²)",       "(b) Gravity",      W_ORANGE)
_hist(axes6[0,2], escvs, "v_esc (km/s)",   "(c) Escape vel.",  W_GREEN)
_hist(axes6[1,0], dens,  "Density (kg/m³)","(d) Mean density", W_RED)
_hist(axes6[1,1], atm_d, "ρ₀ (kg/m³)",    "(e) Atm. density", W_BLUE)

ax_m = axes6[1,2]
_ax(ax_m)
unique, counts = np.unique(n_moons, return_counts=True)
ax_m.bar(unique, counts, color=[WONG[i % len(WONG)] for i in range(len(unique))],
         edgecolor="white", lw=0.4, width=0.6)
ax_m.set_xlabel("Moon count", fontsize=FL, labelpad=2)
ax_m.set_ylabel("Count", fontsize=FL, labelpad=2)
ax_m.set_title("(f) Moons", fontsize=FT, fontweight="bold", pad=3)
ax_m.tick_params(labelsize=FK)
ax_m.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

fig6.suptitle(f"Batch statistics — 50 random planets  "
              f"(magnetosphere: {has_mag.mean()*100:.0f}%)",
              fontsize=FT+1, fontweight="bold")
save_figure(fig6, "fig6_batch", output_dir=OUT)
plt.close(fig6)


print()
print("=" * 54)
print(f"Done — 8 figures saved to ./{OUT}/")
print("=" * 54)
print(f"  Radius  {radii.min():.2f}–{radii.max():.2f} R⊕  (μ={radii.mean():.2f})")
print(f"  Gravity {gravs.min():.1f}–{gravs.max():.1f} m/s²")
print(f"  v_esc   {escvs.min():.1f}–{escvs.max():.1f} km/s")
print(f"  Moons   max {n_moons.max()}, μ={n_moons.mean():.1f}")
print(f"  Mag     {has_mag.sum()}/50 planets")