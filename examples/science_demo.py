"""
science_demo.py  —  Planet-RL Science Toolkit: Full Feature Demo
=================================================================
Figures produced
----------------
fig01_solar_system_comparison.png  — 5 presets cross-sections + interior pie charts
fig02_interior_profiles.png        — Earth vs Mars: MoI, J2, B-field, heat flux
fig03_star_habitable_zones.png     — 6 stars: HZ bands + XUV flux comparison
fig04_atmosphere_science.png       — Multi-layer profiles + Jeans escape + greenhouse
fig05_habitability_radar.png       — 10-factor radar + scores for 6 worlds
fig06_orbital_mechanics.png        — J2 precession, frozen orbit, sun-sync design
fig07_ground_track_coverage.png    — 3-day ground track + coverage heat map
fig08_surface_energy.png           — Global insolation + temperature maps (3 seasons)
fig09_tidal_dynamics.png           — Heating vs distance, Roche zones, locking time
fig10_mission_design.png           — ΔV budget, aerobraking passes, porkchop grid
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# ── science imports ───────────────────────────────────────────────────────────
from planet_rl.core import (
    AU,
    PRESETS,
    STAR_PRESETS,
    AtmosphereComposition,
    InteriorConfig,
    PlanetGenerator,
    SpectralType,
    Star,
    TerrainType,
    interior_from_bulk_density,
    star_eps_eridani,
    star_kepler452,
    star_proxima_centauri,
    star_sun,
    star_tau_ceti,
    star_trappist1,
)
from planet_rl.core.atmosphere_science import (
    STANDARD_COMPOSITIONS,
    GreenhouseModel,
    JeansEscape,
    MultiLayerAtmosphere,
    analyse_atmosphere,
)
from planet_rl.core.ground_track import (
    compute_coverage_map,
    find_passes,
    propagate_ground_track,
)
from planet_rl.core.habitability import (
    assess_habitability,
    composition_class,
    size_class,
)
from planet_rl.core.mission import (
    GravityAssist,
    build_mission_dv_budget,
    orbital_insertion_dv,
    plan_aerobraking,
    porkchop_data,
)
from planet_rl.core.orbital_analysis import (
    DragLifetime,
    FrozenOrbit,
    J2Analysis,
    OrbitDesign,
    RepeatGroundTrack,
    StationKeeping,
    SunSynchronousOrbit,
    circular_speed,
    orbital_period,
)
from planet_rl.core.surface_energy import (
    compute_insolation_map,
    compute_temperature_map,
    permanent_shadow_latitude_deg,
    surface_energy_balance,
)
from planet_rl.core.tidal import (
    RocheLimit,
    TidalHeating,
    TidalLocking,
    analyse_tidal,
)
from planet_rl.visualization import (
    FA,
    FG,
    FK,
    FL,
    FT,
    LW,
    LW2,
    W_BLACK,
    W_BLUE,
    W_GREEN,
    W_ORANGE,
    W_PINK,
    W_RED,
    W_SKY,
    W_YELLOW,
    WONG,
    apply_journal_style,
    plot_atmosphere_profile,
    plot_planet_cross_section,
    save_figure,
)
from planet_rl.visualization.visualizer import _ax

# ── output directory ──────────────────────────────────────────────────────────
OUT = "figures/science_figures"
os.makedirs(OUT, exist_ok=True)
apply_journal_style()

# ── shared planet objects ─────────────────────────────────────────────────────
earth = PRESETS["earth"]()
mars = PRESETS["mars"]()
venus = PRESETS["venus"]()
moon = PRESETS["moon"]()
titan = PRESETS["titan"]()
sun = star_sun()

earth.interior = InteriorConfig.earth_like()
mars.interior = InteriorConfig.mars_like()
venus.interior = interior_from_bulk_density(5243)
moon.interior = interior_from_bulk_density(3346)
titan.interior = InteriorConfig.ocean_world()

earth.star_context = sun
earth.orbital_distance_m = 1.000 * AU
mars.star_context = sun
mars.orbital_distance_m = 1.524 * AU
venus.star_context = sun
venus.orbital_distance_m = 0.723 * AU
moon.star_context = sun
moon.orbital_distance_m = 1.000 * AU
titan.star_context = sun
titan.orbital_distance_m = 9.537 * AU

ALL_PLANETS = [earth, mars, venus, moon, titan]
COLORS6 = [W_BLUE, W_RED, W_ORANGE, "#888888", W_PINK, W_SKY]

print("=" * 60)
print("Planet-RL  Science Feature Demo")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 01 — Solar System Presets: cross-sections + interior breakdown
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/10] Solar system overview …")

fig1 = plt.figure(figsize=(7.08, 5.0))
fig1.patch.set_facecolor("white")
gs1 = gridspec.GridSpec(
    2,
    5,
    figure=fig1,
    height_ratios=[1.8, 1.0],
    hspace=0.35,
    wspace=0.10,
    left=0.02,
    right=0.98,
    top=0.87,
    bottom=0.08,
)

ref_r = max(p.radius for p in ALL_PLANETS)
for i, planet in enumerate(ALL_PLANETS):
    ax_cs = fig1.add_subplot(gs1[0, i])
    plot_planet_cross_section(planet, ax=ax_cs, ref_radius=ref_r)

# Bottom row: interior composition pie charts
PIE_COLS = {
    "iron_solid": W_RED,
    "iron_liquid": "#E86060",
    "iron_sulfide": "#D07070",
    "perovskite": "#C4956A",
    "silicate_mix": "#A88060",
    "olivine": "#B09878",
    "basalt": "#8A7060",
    "granite": "#BCA898",
    "water_ice": W_SKY,
    "liquid_water": W_BLUE,
    "high_pressure_ice": "#80B8D0",
    "other": "#AAAAAA",
}

for i, planet in enumerate(ALL_PLANETS):
    ax_pie = fig1.add_subplot(gs1[1, i])
    ax_pie.set_facecolor("white")
    if planet.interior and planet.interior.enabled and planet.interior.layers:
        layers = sorted(planet.interior.layers, key=lambda l: l.outer_radius_frac)
        inner_r = 0.0
        sizes, colors, labels = [], [], []
        for lyr in layers:
            r_frac = lyr.outer_radius_frac**3 - inner_r**3
            inner_r = lyr.outer_radius_frac
            sizes.append(r_frac)
            colors.append(PIE_COLS.get(lyr.material, "#AAAAAA"))
            labels.append(lyr.name)
        wedges, _ = ax_pie.pie(
            sizes,
            colors=colors,
            startangle=90,
            wedgeprops=dict(linewidth=0.4, edgecolor="white"),
        )
        ax_pie.set_title(planet.name, fontsize=FA, pad=2)
    else:
        ax_pie.text(
            0.5,
            0.5,
            "no\nmodel",
            ha="center",
            va="center",
            fontsize=FA,
            color="#999999",
            transform=ax_pie.transAxes,
        )
        ax_pie.axis("off")
    ax_pie.set_aspect("equal")

fig1.suptitle(
    "Solar system analogues — cross-sections & interior structure",
    fontsize=FT + 1,
    fontweight="bold",
)
save_figure(fig1, "fig01_solar_system_comparison", OUT)
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 02 — Interior model: derived quantities for all 5 planets
# ══════════════════════════════════════════════════════════════════════════════
print("[2/10] Interior profiles …")

fig2, axes2 = plt.subplots(
    2,
    2,
    figsize=(7.08, 4.8),
    gridspec_kw=dict(
        hspace=0.50, wspace=0.40, left=0.10, right=0.97, top=0.87, bottom=0.11
    ),
)
fig2.patch.set_facecolor("white")
names = [p.name for p in ALL_PLANETS]
colors = [WONG[i] for i in range(len(ALL_PLANETS))]

# MoI factor
moi_vals = [p.derived_MoI() for p in ALL_PLANETS]
ax = axes2[0, 0]
_ax(ax)
bars = ax.bar(names, moi_vals, color=colors, edgecolor="white", width=0.6)
ax.axhline(0.4, color=W_BLACK, lw=LW2, ls="--", label="Uniform sphere (0.40)")
ax.axhline(0.331, color=W_BLUE, lw=LW2, ls=":", label="Earth actual (0.331)")
ax.set_ylabel("MoI factor  C/(MR²)", fontsize=FL)
ax.set_ylim(0, 0.45)
ax.set_title("(a) Moment of inertia factor", fontsize=FT, fontweight="bold", pad=3)
ax.legend(fontsize=FG - 1, framealpha=0.9)
ax.tick_params(labelsize=FK)
for b, v in zip(bars, moi_vals):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v + 0.003,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=FA,
    )

# Derived J2
j2_vals = [p.derived_J2() * 1000 for p in ALL_PLANETS]  # ×10³
ax = axes2[0, 1]
_ax(ax)
bars2 = ax.bar(names, j2_vals, color=colors, edgecolor="white", width=0.6)
ax.set_ylabel("J₂  (×10⁻³)", fontsize=FL)
ax.set_title("(b) Derived J₂ gravity harmonic", fontsize=FT, fontweight="bold", pad=3)
ax.tick_params(labelsize=FK)
for b, v in zip(bars2, j2_vals):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v + 0.02,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=FA,
    )

# Surface B-field
b_vals = [p.derived_magnetic_field_T() * 1e6 for p in ALL_PLANETS]  # μT
ax = axes2[1, 0]
_ax(ax)
bars3 = ax.bar(names, b_vals, color=colors, edgecolor="white", width=0.6)
ax.set_ylabel("B-field (μT)", fontsize=FL)
ax.set_title(
    "(c) Derived surface magnetic field", fontsize=FT, fontweight="bold", pad=3
)
ax.tick_params(labelsize=FK)
for b, v in zip(bars3, b_vals):
    ax.text(
        b.get_x() + b.get_width() / 2,
        max(v, 0) + 0.5,
        f"{v:.1f}",
        ha="center",
        va="bottom",
        fontsize=FA,
    )

# Radiogenic heat flux
hf_vals = [p.derived_heat_flux() * 1000 for p in ALL_PLANETS]  # mW/m²
ax = axes2[1, 1]
_ax(ax)
bars4 = ax.bar(names, hf_vals, color=colors, edgecolor="white", width=0.6)
ax.axhline(30, color=W_BLACK, lw=LW2, ls="--", label="Earth reference 30 mW/m²")
ax.set_ylabel("Heat flux (mW/m²)", fontsize=FL)
ax.set_title("(d) Radiogenic surface heat flux", fontsize=FT, fontweight="bold", pad=3)
ax.legend(fontsize=FG - 1, framealpha=0.9)
ax.tick_params(labelsize=FK)
for b, v in zip(bars4, hf_vals):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v + 0.3,
        f"{v:.1f}",
        ha="center",
        va="bottom",
        fontsize=FA,
    )

fig2.suptitle(
    "Interior model: physically derived planetary properties",
    fontsize=FT + 1,
    fontweight="bold",
)
save_figure(fig2, "fig02_interior_profiles", OUT)
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 03 — Stars: habitable zones + XUV comparison
# ══════════════════════════════════════════════════════════════════════════════
print("[3/10] Stellar habitable zones …")

stars = [
    sun,
    star_tau_ceti(),
    star_kepler452(),
    star_eps_eridani(),
    star_proxima_centauri(),
    star_trappist1(),
]
star_colors = [W_ORANGE, W_YELLOW, "#FFD080", W_RED, "#FF6060", "#FF4040"]

fig3 = plt.figure(figsize=(7.08, 5.0))
fig3.patch.set_facecolor("white")
gs3 = gridspec.GridSpec(
    1, 2, figure=fig3, wspace=0.40, left=0.10, right=0.97, top=0.87, bottom=0.12
)

# Left: HZ bands as horizontal bars
ax3a = fig3.add_subplot(gs3[0])
_ax(ax3a)
for i, (star, sc) in enumerate(zip(stars, star_colors)):
    y = len(stars) - 1 - i
    hz_in = star.hz_inner_au
    hz_out = star.hz_outer_au
    hz_in_opt = star.hz_inner_optimistic_m / AU
    hz_out_opt = star.hz_outer_optimistic_m / AU
    # Optimistic extension (lighter)
    ax3a.barh(
        y, hz_out_opt - hz_in_opt, left=hz_in_opt, height=0.55, color=sc, alpha=0.25
    )
    # Conservative HZ (solid)
    ax3a.barh(
        y,
        hz_out - hz_in,
        left=hz_in,
        height=0.55,
        color=sc,
        alpha=0.80,
        label=star.name,
    )
    # Reference planets
    for pname, pdist in [("Venus", 0.723), ("Earth", 1.0), ("Mars", 1.524)]:
        if hz_in_opt * 0.5 <= pdist <= hz_out_opt * 2:
            ax3a.plot(pdist, y, "o", color=W_BLACK, ms=3, zorder=5)
            if i == 0:
                ax3a.text(
                    pdist,
                    y + 0.35,
                    pname,
                    ha="center",
                    fontsize=FA - 0.5,
                    color=W_BLACK,
                )

ax3a.set_yticks(range(len(stars)))
ax3a.set_yticklabels([s.name for s in reversed(stars)], fontsize=FK)
ax3a.set_xlabel("Orbital distance (AU)", fontsize=FL)
ax3a.set_title(
    "(a) Habitable zones (conservative + optimistic)",
    fontsize=FT,
    fontweight="bold",
    pad=3,
)
ax3a.set_xlim(0, 2.2)

# Right: XUV flux at 1 AU equivalent distance
ax3b = fig3.add_subplot(gs3[1])
_ax(ax3b)
xuv_vals = [s.xuv_luminosity / s.luminosity * 1e6 for s in stars]  # ppm of bolometric
x_pos = range(len(stars))
bars_x = ax3b.barh(
    list(range(len(stars))),
    [s.xuv_luminosity / s.luminosity * 1e3 for s in stars],
    color=star_colors,
    edgecolor="white",
    height=0.6,
)
ax3b.set_yticks(range(len(stars)))
ax3b.set_yticklabels([s.name for s in stars], fontsize=FK)
ax3b.set_xlabel("L_XUV / L_bol  (×10⁻³)", fontsize=FL)
ax3b.set_title(
    "(b) XUV/bolometric luminosity ratio", fontsize=FT, fontweight="bold", pad=3
)
ax3b.axvline(1.0, color=W_BLACK, lw=LW2, ls="--", label="Sun reference")
ax3b.legend(fontsize=FG - 1)

fig3.suptitle(
    "Stellar environments: habitable zones and high-energy radiation",
    fontsize=FT + 1,
    fontweight="bold",
)
save_figure(fig3, "fig03_star_habitable_zones", OUT)
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 04 — Atmosphere science: profiles + Jeans escape + greenhouse
# ══════════════════════════════════════════════════════════════════════════════
print("[4/10] Atmosphere science …")

fig4 = plt.figure(figsize=(7.08, 5.8))
fig4.patch.set_facecolor("white")
gs4 = gridspec.GridSpec(
    2,
    3,
    figure=fig4,
    hspace=0.55,
    wspace=0.38,
    left=0.10,
    right=0.97,
    top=0.87,
    bottom=0.10,
)

# Top row: multi-layer atmosphere profiles for Earth, Venus, Titan
atm_planets = [
    (earth, "Earth", W_BLUE),
    (venus, "Venus", W_ORANGE),
    (titan, "Titan", W_PINK),
]
for col, (planet, pname, pc) in enumerate(atm_planets):
    ax = fig4.add_subplot(gs4[0, col])
    _ax(ax)
    multi = MultiLayerAtmosphere.from_atmosphere_config(planet.atmosphere, planet)
    max_alt = 150_000
    alts = np.linspace(0, max_alt, 400)
    temps = [multi.temperature_at(h) for h in alts]
    rhos = [multi.density_at(h) for h in alts]
    ax2 = ax.twiny()
    ax.plot(temps, alts / 1e3, color=W_GREEN, lw=LW, label="T (K)")
    ax2.plot(
        np.array(rhos) * 1000, alts / 1e3, color=pc, lw=LW, ls="--", label="ρ (g/m³)"
    )
    ax.set_ylabel("Altitude (km)" if col == 0 else "", fontsize=FL)
    ax.set_xlabel("Temperature (K)", fontsize=FL - 0.5, labelpad=1)
    ax2.set_xlabel("Density (g/m³)", fontsize=FL - 0.5, labelpad=1)
    ax.set_ylim(0, max_alt / 1e3)
    ax.set_title(f"({chr(97 + col)}) {pname}", fontsize=FT, fontweight="bold", pad=3)
    ax.tick_params(labelsize=FK)
    ax2.tick_params(labelsize=FK)

# Bottom left: Jeans escape Λ for bulk species across all planets
ax4d = fig4.add_subplot(gs4[1, 0])
_ax(ax4d)
species_show = ["H2", "H2O", "CH4", "N2", "CO2"]
planet_names = [p.name for p in ALL_PLANETS]
x = np.arange(len(planet_names))
w = 0.15
for si, sp in enumerate(species_show):
    lams = []
    for planet in ALL_PLANETS:
        v_esc = planet.escape_velocity
        T_exo = (
            planet.atmosphere.surface_temp * 2.5 if planet.atmosphere.enabled else 500
        )
        lam = JeansEscape.lambda_parameter(sp, v_esc, T_exo)
        lams.append(min(lam, 200))
    ax4d.bar(
        x + si * w,
        lams,
        width=w,
        label=sp,
        color=WONG[si % len(WONG)],
        edgecolor="white",
    )
ax4d.axhline(20, color=W_BLACK, lw=LW2, ls="--", label="λ=20 retention limit")
ax4d.set_xticks(x + w * 2)
ax4d.set_xticklabels(planet_names, fontsize=FK, rotation=20)
ax4d.set_ylabel("Jeans parameter λ", fontsize=FL)
ax4d.set_title("(d) Jeans escape parameter", fontsize=FT, fontweight="bold", pad=3)
ax4d.legend(fontsize=FG - 1, ncol=2, framealpha=0.9)
ax4d.tick_params(labelsize=FK)

# Bottom middle: greenhouse warming vs CO2 partial pressure
ax4e = fig4.add_subplot(gs4[1, 1])
_ax(ax4e)
p_co2_range = np.logspace(-2, 7, 200)  # Pa
dT_vals = [GreenhouseModel.co2_forcing_K(p) for p in p_co2_range]
ax4e.semilogx(p_co2_range, dT_vals, color=W_RED, lw=LW)
for label, pco2, color in [
    ("Mars\n(636 Pa CO₂)", 636 * 0.953, W_ORANGE),
    ("Earth\n(28 Pa CO₂)", 28.3, W_BLUE),
    ("Venus\n(8.9 MPa CO₂)", 9.2e6 * 0.965, W_RED),
]:
    dT_pt = GreenhouseModel.co2_forcing_K(pco2)
    ax4e.axvline(pco2, color=color, lw=0.8, ls=":", alpha=0.7)
    ax4e.plot(pco2, dT_pt, "o", color=color, ms=5, zorder=5)
    ax4e.text(pco2 * 1.5, dT_pt + 8, label, fontsize=FA - 0.5, color=color)
ax4e.set_xlabel("CO₂ partial pressure (Pa)", fontsize=FL)
ax4e.set_ylabel("Greenhouse forcing ΔT (K)", fontsize=FL)
ax4e.set_title("(e) CO₂ greenhouse forcing", fontsize=FT, fontweight="bold", pad=3)
ax4e.tick_params(labelsize=FK)

# Bottom right: surface temperature vs equilibrium temperature
ax4f = fig4.add_subplot(gs4[1, 2])
_ax(ax4f)
for pi, (planet, pcolor) in enumerate(zip(ALL_PLANETS, WONG)):
    if not planet.atmosphere.enabled or not planet.star_context:
        continue
    aa = analyse_atmosphere(planet, planet.star_context, planet.orbital_distance_m)
    if aa.get("enabled"):
        t_eq = aa["equilibrium_temp_K"]
        t_sf = aa["surface_temp_K"]
        ax4f.plot([t_eq], [t_sf], "o", color=pcolor, ms=7, zorder=5, label=planet.name)
        ax4f.annotate(
            planet.name,
            (t_eq, t_sf),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=FA - 0.5,
            color=pcolor,
        )
t_range = np.linspace(50, 500, 100)
ax4f.plot(t_range, t_range, "--", color=W_BLACK, lw=LW2, label="T_surf = T_eq")
ax4f.set_xlabel("Equilibrium T (K)", fontsize=FL)
ax4f.set_ylabel("Surface T (K)", fontsize=FL)
ax4f.set_title("(f) Greenhouse amplification", fontsize=FT, fontweight="bold", pad=3)
ax4f.legend(fontsize=FG - 1, framealpha=0.9)
ax4f.tick_params(labelsize=FK)

fig4.suptitle(
    "Atmospheric science: structure, escape, and greenhouse physics",
    fontsize=FT + 1,
    fontweight="bold",
)
save_figure(fig4, "fig04_atmosphere_science", OUT)
plt.close(fig4)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 05 — Habitability assessment: radar charts + score table
# ══════════════════════════════════════════════════════════════════════════════
print("[5/10] Habitability assessment …")

hab_planets = [earth, mars, venus, moon, titan]
hab_names = [p.name for p in hab_planets]
# Add a random generated "interesting" planet
gen = PlanetGenerator(seed=77)
super_e = gen.generate(
    "Super-Earth",
    atmosphere_enabled=True,
    radius_range=(1.2 * 6.371e6, 1.6 * 6.371e6),
    density_range=(4500, 5500),
    magnetic_field_enabled=True,
    oblateness_enabled=True,
    moons_enabled=False,
)
super_e.interior = interior_from_bulk_density(super_e.mean_density)
super_e.star_context = sun
super_e.orbital_distance_m = 1.1 * AU
hab_planets.append(super_e)
hab_names.append("Super-Earth")

assessments = [
    assess_habitability(p, p.star_context, p.orbital_distance_m) for p in hab_planets
]

factor_names = list(assessments[0].factors.keys())
factor_short = [
    "Stellar\ntype",
    "Stellar\nage",
    "HZ\npos.",
    "Surf.\ntemp",
    "Liq.\nwater",
    "Atm.\nretain",
    "Mag.\nshield",
    "Tidal\nlock",
    "Interior",
    "Size",
]

fig5 = plt.figure(figsize=(7.08, 5.6))
fig5.patch.set_facecolor("white")
gs5 = gridspec.GridSpec(
    2,
    3,
    figure=fig5,
    hspace=0.70,
    wspace=0.42,
    left=0.08,
    right=0.97,
    top=0.83,
    bottom=0.08,
)

ha_colors = [W_BLUE, W_RED, W_ORANGE, "#888888", W_PINK, W_GREEN]

for pi, (ha, hc) in enumerate(zip(assessments, ha_colors)):
    row, col = divmod(pi, 3)
    ax = fig5.add_subplot(gs5[row, col], polar=True)
    N = len(factor_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    scores = [s for s, _ in ha.factors.values()]
    scores += scores[:1]
    ax.plot(angles, scores, color=hc, lw=LW, zorder=3)
    ax.fill(angles, scores, color=hc, alpha=0.20, zorder=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(factor_short, size=FA - 1.0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], size=0)
    ax.grid(color="#dddddd", lw=0.5)
    ax.set_facecolor("white")
    grade_col = {
        "A": W_GREEN,
        "B": W_BLUE,
        "C": W_ORANGE,
        "D": W_RED,
        "F": "#888888",
    }.get(ha.grade, W_BLACK)
    ax.set_title(
        f"{ha.planet_name}\nscore={ha.overall_score:.2f}  grade={ha.grade}",
        fontsize=FA + 0.5,
        fontweight="bold",
        pad=8,
        color=grade_col,
    )

fig5.suptitle(
    "Habitability assessment: ten-factor radar comparison",
    fontsize=FT + 1,
    fontweight="bold",
)
save_figure(fig5, "fig05_habitability_radar", OUT)
plt.close(fig5)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 06 — Orbital mechanics: J2 rates, frozen orbit, sun-sync design
# ══════════════════════════════════════════════════════════════════════════════
print("[6/10] Orbital mechanics …")

T_earth_year = sun.orbital_period(1.0 * AU)
T_mars_year = sun.orbital_period(1.524 * AU)

fig6, axes6 = plt.subplots(
    2,
    3,
    figsize=(7.08, 4.8),
    gridspec_kw=dict(
        hspace=0.52, wspace=0.42, left=0.10, right=0.97, top=0.87, bottom=0.11
    ),
)
fig6.patch.set_facecolor("white")

# (a) RAAN precession rate vs altitude for 4 inclinations — Earth
ax = axes6[0, 0]
_ax(ax)
alts = np.linspace(200, 1000, 200)
incs = [30, 60, 90, 98]
linestyles = ["-", "--", "-.", ":"]
for inc, ls in zip(incs, linestyles):
    rates = [
        J2Analysis.nodal_precession_rate_deg_day(
            earth, (earth.radius + h * 1e3), math.radians(inc)
        )
        for h in alts
    ]
    ax.plot(alts, rates, lw=LW, ls=ls, label=f"{inc}°")
ax.axhline(0.9856, color=W_BLACK, lw=LW2, ls="--", alpha=0.5, label="Sun-sync rate")
ax.set_xlabel("Altitude (km)", fontsize=FL)
ax.set_ylabel("dΩ/dt  (°/day)", fontsize=FL)
ax.set_title("(a) J2 nodal precession (Earth)", fontsize=FT, fontweight="bold", pad=3)
ax.legend(fontsize=FG - 1, framealpha=0.9, ncol=2)
ax.tick_params(labelsize=FK)

# (b) Sun-synchronous inclination vs altitude for Earth and Mars
ax = axes6[0, 1]
_ax(ax)
alts2 = np.linspace(200, 900, 200)
ss_earth = []
ss_mars = []
for h in alts2:
    ie = SunSynchronousOrbit.sun_sync_inclination(earth, h * 1e3, T_earth_year)
    im = SunSynchronousOrbit.sun_sync_inclination(mars, h * 1e3, T_mars_year)
    ss_earth.append(ie if ie else float("nan"))
    ss_mars.append(im if im else float("nan"))
ax.plot(alts2, ss_earth, color=W_BLUE, lw=LW, label="Earth")
ax.plot(alts2, ss_mars, color=W_RED, lw=LW, label="Mars")
ax.set_xlabel("Altitude (km)", fontsize=FL)
ax.set_ylabel("Sun-sync inclination (°)", fontsize=FL)
ax.set_title("(b) Sun-sync inclination", fontsize=FT, fontweight="bold", pad=3)
ax.legend(fontsize=FG, framealpha=0.9)
ax.tick_params(labelsize=FK)

# (c) Frozen orbit eccentricity vs inclination at 500 km (Earth)
ax = axes6[0, 2]
_ax(ax)
inc_range = np.linspace(55, 140, 200)
frz_eccs = [
    FrozenOrbit.frozen_eccentricity(earth, earth.radius + 500e3, math.radians(i))
    for i in inc_range
]
ax.plot(inc_range, np.array(frz_eccs) * 1000, color=W_GREEN, lw=LW)
ax.axvline(63.4, color=W_ORANGE, lw=LW2, ls="--", label="Critical inc. 63.4°")
ax.axvline(116.6, color=W_ORANGE, lw=LW2, ls="--")
ax.set_xlabel("Inclination (°)", fontsize=FL)
ax.set_ylabel("Frozen eccentricity (×10⁻³)", fontsize=FL)
ax.set_title(
    "(c) Frozen orbit eccentricity (500 km)", fontsize=FT, fontweight="bold", pad=3
)
ax.legend(fontsize=FG)
ax.tick_params(labelsize=FK)

# (d) Drag lifetime vs altitude for different B values — Earth
ax = axes6[1, 0]
_ax(ax)
alts3 = np.linspace(250, 800, 200)
b_vals = [50, 100, 200]
b_labels = ["B=50 (CubeSat)", "B=100 (small sc.)", "B=200 (large sc.)"]
for bv, bl, lss in zip(b_vals, b_labels, ["-", "--", "-."]):
    lifetimes = [DragLifetime.lifetime_years(earth, h * 1e3, 1000, bv) for h in alts3]
    lifetimes = [min(l, 1000) for l in lifetimes]
    ax.semilogy(alts3, lifetimes, lw=LW, ls=lss, label=bl)
ax.axhline(1, color=W_BLACK, lw=LW2, ls=":", label="1-yr threshold")
ax.set_xlabel("Altitude (km)", fontsize=FL)
ax.set_ylabel("Drag lifetime (yr)", fontsize=FL)
ax.set_title(
    "(d) Atmospheric drag lifetime (Earth)", fontsize=FT, fontweight="bold", pad=3
)
ax.legend(fontsize=FG - 1, framealpha=0.9)
ax.tick_params(labelsize=FK)

# (e) Station-keeping ΔV per year vs altitude
ax = axes6[1, 1]
_ax(ax)
alts4 = np.linspace(250, 700, 100)
sk_vals = [
    StationKeeping.total_annual_budget(earth, h, 98.0, 100, T_earth_year)[
        "total_dv_m_s_yr"
    ]
    for h in alts4
]
sk_vals = [min(v, 50) for v in sk_vals]
ax.plot(alts4, sk_vals, color=W_ORANGE, lw=LW)
ax.set_xlabel("Altitude (km)", fontsize=FL)
ax.set_ylabel("Station-keeping ΔV (m/s/yr)", fontsize=FL)
ax.set_title("(e) Annual station-keeping cost", fontsize=FT, fontweight="bold", pad=3)
ax.tick_params(labelsize=FK)

# (f) Repeat ground-track solutions (altitude vs orbits/day)
ax = axes6[1, 2]
_ax(ax)
repeats = RepeatGroundTrack.find_repeat_orbits(earth, (300, 900), max_days=25)
if repeats:
    r_alts = [r["alt_km"] for r in repeats]
    r_opd = [r["orbits_per_day"] for r in repeats]
    r_days = [r["n_days"] for r in repeats]
    sc = ax.scatter(
        r_alts, r_opd, c=r_days, cmap="viridis_r", s=12, alpha=0.8, zorder=3
    )
    cb = fig6.colorbar(sc, ax=ax, shrink=0.8, pad=0.03)
    cb.set_label("Repeat period (days)", fontsize=FA)
    cb.ax.tick_params(labelsize=FA)
ax.set_xlabel("Altitude (km)", fontsize=FL)
ax.set_ylabel("Orbits per day", fontsize=FL)
ax.set_title("(f) Repeat ground-track solutions", fontsize=FT, fontweight="bold", pad=3)
ax.tick_params(labelsize=FK)

fig6.suptitle(
    "Orbital mechanics: J2 perturbations, frozen orbits, mission design",
    fontsize=FT + 1,
    fontweight="bold",
)
save_figure(fig6, "fig06_orbital_mechanics", OUT)
plt.close(fig6)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 07 — Ground track and coverage map
# ══════════════════════════════════════════════════════════════════════════════
print("[7/10] Ground track and coverage …")

track = propagate_ground_track(
    earth, 500_000, 98.0, duration_s=3 * 86400, dt_s=120, include_j2=True
)
cov = compute_coverage_map(
    earth, track, swath_width_km=120, lat_res_deg=2.0, lon_res_deg=2.0
)

fig7, axes7 = plt.subplots(
    1,
    2,
    figsize=(7.08, 3.4),
    gridspec_kw=dict(wspace=0.35, left=0.07, right=0.97, top=0.85, bottom=0.14),
)
fig7.patch.set_facecolor("white")

# Ground track (first 3 orbits for clarity)
ax = axes7[0]
ax.set_facecolor("#E8F4FD")
for sp in ax.spines.values():
    sp.set_edgecolor("#444444")
    sp.set_linewidth(0.8)
# Plot track colour-coded by time
lats = np.array([p.lat_deg for p in track])
lons = np.array([p.lon_deg for p in track])
times = np.array([p.time_s for p in track]) / 3600  # hours

# Downsample for speed
step = max(1, len(track) // 2000)
lats_s = lats[::step]
lons_s = lons[::step]
times_s = times[::step]

sc7 = ax.scatter(lons_s, lats_s, c=times_s, cmap="plasma", s=0.8, alpha=0.8, zorder=3)
cb7 = fig7.colorbar(sc7, ax=ax, shrink=0.8, pad=0.03)
cb7.set_label("Time (hr)", fontsize=FA)
cb7.ax.tick_params(labelsize=FA)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xticks(np.arange(-180, 181, 60))
ax.set_yticks(np.arange(-90, 91, 30))
ax.axhline(0, color="white", lw=0.5, alpha=0.5)
ax.set_xlabel("Longitude (°)", fontsize=FL)
ax.set_ylabel("Latitude (°)", fontsize=FL)
ax.set_title(
    f"(a) 3-day ground track (500 km / 98°)", fontsize=FT, fontweight="bold", pad=3
)
ax.tick_params(labelsize=FK)

# Coverage map
ax = axes7[1]
_ax(ax)
coverage_pct = (cov.grid > 0).astype(float)
lons_c = cov.lon_centres()
lats_c = cov.lat_centres()
lon_g, lat_g = np.meshgrid(lons_c, lats_c)
im = ax.pcolormesh(
    lon_g, lat_g, np.minimum(cov.grid, 5), cmap="Blues", shading="auto", vmin=0, vmax=5
)
cb8 = fig7.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
cb8.set_label("Observation count", fontsize=FA)
cb8.ax.tick_params(labelsize=FA)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude (°)", fontsize=FL)
ax.set_ylabel("Latitude (°)", fontsize=FL)
ax.set_title(
    f"(b) Surface coverage ({cov.coverage_fraction() * 100:.0f}% in 3 days, 120 km swath)",
    fontsize=FT,
    fontweight="bold",
    pad=3,
)
ax.tick_params(labelsize=FK)

fig7.suptitle(
    "Ground track propagation and surface coverage analysis",
    fontsize=FT + 1,
    fontweight="bold",
)
save_figure(fig7, "fig07_ground_track_coverage", OUT)
plt.close(fig7)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 08 — Surface energy: insolation + temperature maps
# ══════════════════════════════════════════════════════════════════════════════
print("[8/10] Surface energy maps …")

S_earth = sun.flux_at_distance(1.0 * AU)

fig8, axes8 = plt.subplots(
    2,
    3,
    figsize=(7.08, 4.8),
    gridspec_kw=dict(
        hspace=0.45, wspace=0.40, left=0.08, right=0.97, top=0.87, bottom=0.10
    ),
)
fig8.patch.set_facecolor("white")

seasons = [(0.0, "Northern solstice"), (0.25, "Equinox"), (0.5, "Southern solstice")]
T_insets = []

for col, (phase, label) in enumerate(seasons):
    ins = compute_insolation_map(earth, S_earth, 23.5, phase, time_average=True)
    T = compute_temperature_map(
        ins, bond_albedo=0.3, emissivity=0.95, greenhouse_dT_K=25, thermal_inertia=800
    )
    T_insets.append(T)
    lats = ins.lat_deg
    lons = ins.lon_deg
    lon_g, lat_g = np.meshgrid(lons, lats)

    ax_ins = axes8[0, col]
    _ax(ax_ins)
    im1 = ax_ins.pcolormesh(
        lon_g, lat_g, ins.data_W_m2, cmap="YlOrRd", shading="auto", vmin=0, vmax=500
    )
    cb = fig8.colorbar(im1, ax=ax_ins, shrink=0.85, pad=0.03)
    cb.set_label("W/m²", fontsize=FA)
    cb.ax.tick_params(labelsize=FA)
    ax_ins.set_title(f"(insolation) {label}", fontsize=FA, pad=2)
    ax_ins.set_xlabel("Lon (°)", fontsize=FA)
    if col == 0:
        ax_ins.set_ylabel("Lat (°)", fontsize=FA)
    ax_ins.tick_params(labelsize=FA - 1)

    ax_T = axes8[1, col]
    _ax(ax_T)
    im2 = ax_T.pcolormesh(
        lon_g, lat_g, T.data_K, cmap="RdYlBu_r", shading="auto", vmin=210, vmax=320
    )
    cb2 = fig8.colorbar(im2, ax=ax_T, shrink=0.85, pad=0.03)
    cb2.set_label("K", fontsize=FA)
    cb2.ax.tick_params(labelsize=FA)
    ax_T.set_title(f"(temperature) {label}", fontsize=FA, pad=2)
    ax_T.set_xlabel("Lon (°)", fontsize=FA)
    if col == 0:
        ax_T.set_ylabel("Lat (°)", fontsize=FA)
    ax_T.tick_params(labelsize=FA - 1)

    # Habitable zone contour (273-373 K)
    ax_T.contour(
        lon_g,
        lat_g,
        T.data_K,
        levels=[273, 373],
        colors=["white"],
        linewidths=0.8,
        linestyles="--",
    )

fig8.suptitle(
    "Surface energy balance: insolation and temperature maps (Earth, obliquity 23.5°)",
    fontsize=FT + 1,
    fontweight="bold",
)
save_figure(fig8, "fig08_surface_energy", OUT)
plt.close(fig8)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 09 — Tidal dynamics
# ══════════════════════════════════════════════════════════════════════════════
print("[9/10] Tidal dynamics …")

fig9, axes9 = plt.subplots(
    2,
    2,
    figsize=(7.08, 4.8),
    gridspec_kw=dict(
        hspace=0.50, wspace=0.40, left=0.11, right=0.97, top=0.87, bottom=0.11
    ),
)
fig9.patch.set_facecolor("white")

# (a) Tidal heating vs orbital distance for Io-like moons around different planets
ax = axes9[0, 0]
_ax(ax)
M_Io = 8.93e22
R_Io = 1.821e6
e_Io = 0.004
dist_range = np.logspace(4.8, 6.5, 200) * 1e3  # 60,000 km to 3M km

for planet, pcolor, plabel in [(earth, W_BLUE, "Earth"), (mars, W_RED, "Mars")]:
    heats = [
        TidalHeating.heating_rate_W(R_Io, M_Io, planet.mass, d, e_Io)
        for d in dist_range
    ]
    ax.loglog(dist_range / 1e3, heats, color=pcolor, lw=LW, label=plabel)

# Jupiter reference
heats_jup = [
    TidalHeating.heating_rate_W(R_Io, M_Io, 1.898e27, d, e_Io) for d in dist_range
]
ax.loglog(dist_range / 1e3, heats_jup, color=W_ORANGE, lw=LW, label="Jupiter")
ax.axhline(1e14, color=W_BLACK, lw=LW2, ls="--", label="Io observed (~10¹⁴ W)")
ax.set_xlabel("Orbital distance (km)", fontsize=FL)
ax.set_ylabel("Tidal heating (W)", fontsize=FL)
ax.set_title(
    "(a) Tidal heating vs distance (Io-like moon)",
    fontsize=FT,
    fontweight="bold",
    pad=3,
)
ax.legend(fontsize=FG - 1, framealpha=0.9)
ax.tick_params(labelsize=FK)

# (b) Tidal locking timescale vs orbital distance
ax = axes9[0, 1]
_ax(ax)
dists_lock = np.logspace(9.5, 12, 200)  # 3 to 10,000 AU in m
for star_obj, sc, sl in [
    (sun, W_ORANGE, "Sun"),
    (star_proxima_centauri(), W_RED, "Proxima"),
]:
    lock_times = [
        TidalLocking.locking_timescale_gyr(6.371e6, 5.972e24, star_obj.mass, d)
        for d in dists_lock
    ]
    lock_times = [min(t, 1000) for t in lock_times]
    ax.loglog(dists_lock / AU, lock_times, color=sc, lw=LW, label=sl)
ax.axhline(4.5, color=W_BLACK, lw=LW2, ls="--", label="Solar age (4.5 Gyr)")
ax.axvline(1.0, color=W_BLUE, lw=0.8, ls=":", alpha=0.7, label="1 AU")
ax.set_xlabel("Orbital distance (AU)", fontsize=FL)
ax.set_ylabel("Locking timescale (Gyr)", fontsize=FL)
ax.set_title("(b) Tidal locking timescale", fontsize=FT, fontweight="bold", pad=3)
ax.legend(fontsize=FG - 1, framealpha=0.9)
ax.tick_params(labelsize=FK)

# (c) Roche limits for solar system planets
ax = axes9[1, 0]
_ax(ax)
roche_planets = [earth, mars, venus]
sat_densities = [1000, 2000, 3000, 5000]
x_pos = np.arange(len(roche_planets))
w = 0.2
for si, rho_sat in enumerate(sat_densities):
    rochef = [
        RocheLimit.fluid_satellite(p.radius, p.mean_density, rho_sat) / 1e3
        for p in roche_planets
    ]
    ax.bar(
        x_pos + si * w,
        rochef,
        width=w,
        label=f"ρ_sat={rho_sat} kg/m³",
        color=WONG[si],
        edgecolor="white",
    )
ax.set_xticks(x_pos + w * 1.5)
ax.set_xticklabels([p.name for p in roche_planets], fontsize=FK)
ax.set_ylabel("Roche limit (km)", fontsize=FL)
ax.set_title("(c) Fluid Roche limits", fontsize=FT, fontweight="bold", pad=3)
ax.legend(fontsize=FG - 1, framealpha=0.9)
ax.tick_params(labelsize=FK)

# (d) Subsurface ocean heating threshold — needed eccentricity
ax = axes9[1, 1]
_ax(ax)
planet_objs = [earth, mars]
R_moon = 1.5e6
M_moon = 1e22  # generic moon
target_flux = 0.05  # W/m²  (Europa threshold for liquid water)
for planet, pc, pn in [(earth, W_BLUE, "Earth"), (mars, W_RED, "Mars")]:
    dists2 = np.linspace(2 * planet.radius, 30 * planet.radius, 200)
    e_needed = [
        TidalHeating.equilibrium_eccentricity_for_target_flux(
            R_moon, M_moon, planet.mass, d, target_flux
        )
        for d in dists2
    ]
    e_needed = np.array([min(e, 0.5) for e in e_needed])
    ax.semilogy(dists2 / planet.radius, e_needed, color=pc, lw=LW, label=pn)
ax.axhline(0.004, color=W_BLACK, lw=LW2, ls="--", label="Io eccentricity (0.004)")
ax.set_xlabel("Orbital distance (R_planet)", fontsize=FL)
ax.set_ylabel("Required eccentricity", fontsize=FL)
ax.set_title(
    "(d) Eccentricity for subsurface ocean", fontsize=FT, fontweight="bold", pad=3
)
ax.legend(fontsize=FG - 1, framealpha=0.9)
ax.tick_params(labelsize=FK)

fig9.suptitle(
    "Tidal dynamics: heating, locking, Roche limits, and ocean worlds",
    fontsize=FT + 1,
    fontweight="bold",
)
save_figure(fig9, "fig09_tidal_dynamics", OUT)
plt.close(fig9)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Mission design
# ══════════════════════════════════════════════════════════════════════════════
print("[10/10] Mission design …")

fig10, axes10 = plt.subplots(
    2,
    3,
    figsize=(7.5, 5.2),
    gridspec_kw=dict(
        hspace=0.60, wspace=0.48, left=0.09, right=0.97, top=0.87, bottom=0.10
    ),
)
fig10.patch.set_facecolor("white")

# (a) Orbital insertion ΔV vs arrival v_inf for Earth/Mars/Venus
ax = axes10[0, 0]
_ax(ax)
vinf_range = np.linspace(0.5, 8.0, 100)
for planet, pc, pn in [
    (earth, W_BLUE, "Earth"),
    (mars, W_RED, "Mars"),
    (venus, W_ORANGE, "Venus"),
]:
    dvs = [
        orbital_insertion_dv(planet, v, 300)["dv_total_m_s"] / 1e3 for v in vinf_range
    ]
    ax.plot(vinf_range, dvs, color=pc, lw=LW, label=pn)
ax.set_xlabel("Arrival v∞ (km/s)", fontsize=FL)
ax.set_ylabel("Insertion ΔV (km/s)", fontsize=FL)
ax.set_title("(a) Orbital insertion ΔV", fontsize=FT, fontweight="bold", pad=3)
ax.legend(fontsize=FG, framealpha=0.9)
ax.tick_params(labelsize=FK)

# (b) Aerobraking campaign — apoapsis vs pass number
ax = axes10[0, 1]
_ax(ax)
ab = plan_aerobraking(
    mars, initial_apoapsis_km=35000, target_apoapsis_km=400, periapsis_altitude_km=115
)
if ab.passes:
    pass_nos = [p.pass_number for p in ab.passes]
    apos = [p.apoapsis_before_km for p in ab.passes]
    heats = [p.peak_heating_W_m2 for p in ab.passes]
    ax.plot(pass_nos, apos, color=W_RED, lw=LW, label="Apoapsis altitude")
    ax.set_xlabel("Aerobraking pass number", fontsize=FL)
    ax.set_ylabel("Apoapsis altitude (km)", fontsize=FL)
    ax2b = ax.twinx()
    ax2b.plot(pass_nos, heats, color=W_ORANGE, lw=LW, ls="--", label="Peak heating")
    ax2b.set_ylabel("Peak heat flux (W/m²)", fontsize=FL - 0.5, color=W_ORANGE)
    ax2b.tick_params(labelsize=FK, colors=W_ORANGE)
ax.set_title(
    f"(b) Mars aerobraking ({len(ab.passes)} passes)",
    fontsize=FT,
    fontweight="bold",
    pad=3,
)
ax.tick_params(labelsize=FK)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=FG - 1, framealpha=0.9)

# (c) ΔV budget breakdown (stacked bar) for missions to Earth/Mars/Venus
ax = axes10[0, 2]
_ax(ax)
mission_planets = [(earth, 3.0, "Earth"), (mars, 2.5, "Mars"), (venus, 4.0, "Venus")]
budget_components = {}
for planet, vinf, pname in mission_planets:
    b = build_mission_dv_budget(
        planet,
        sun,
        planet.orbital_distance_m,
        vinf,
        300,
        use_aerobraking=(planet == mars),
        station_keeping_years=5,
    )
    for entry in b.entries:
        if entry["name"] not in budget_components:
            budget_components[entry["name"]] = {}
        budget_components[entry["name"]][pname] = entry["dv_m_s"]
pnames = ["Earth", "Mars", "Venus"]
bottom = np.zeros(3)
for ci, (comp_name, vals_dict) in enumerate(budget_components.items()):
    heights = [vals_dict.get(pn, 0) for pn in pnames]
    ax.bar(
        pnames,
        heights,
        bottom=bottom,
        color=WONG[ci % len(WONG)],
        edgecolor="white",
        label=comp_name[:22],
    )
    bottom += np.array(heights)
ax.set_ylabel("ΔV (m/s)", fontsize=FL)
ax.set_title("(c) 5-yr mission ΔV budget", fontsize=FT, fontweight="bold", pad=3)
ax.legend(fontsize=FA - 0.5, framealpha=0.9, bbox_to_anchor=(1.02, 1), loc="upper left")
ax.tick_params(labelsize=FK)

# (d) Gravity assist: bending angle vs v_inf for different periapsis altitudes
ax = axes10[1, 0]
_ax(ax)
vinf2 = np.linspace(0.5, 10, 100)
peris = [100, 500, 1000]
for peri, ls in zip(peris, ["-", "--", "-."]):
    bends = [math.degrees(GravityAssist.bending_angle(mars, v, peri)) for v in vinf2]
    ax.plot(vinf2, bends, lw=LW, ls=ls, label=f"peri={peri} km")
ax.set_xlabel("v∞ (km/s)", fontsize=FL)
ax.set_ylabel("Bending angle (°)", fontsize=FL)
ax.set_title("(d) Mars gravity assist bending", fontsize=FT, fontweight="bold", pad=3)
ax.legend(fontsize=FG, framealpha=0.9)
ax.tick_params(labelsize=FK)

# (e) Insertion ΔV vs target orbit altitude for Mars
ax = axes10[1, 1]
_ax(ax)
alt_range2 = np.linspace(100, 1000, 100)
for vinf_val, vc in [(1.5, W_BLUE), (2.5, W_GREEN), (4.0, W_RED)]:
    dvs2 = [orbital_insertion_dv(mars, vinf_val, h)["dv_total_m_s"] for h in alt_range2]
    ax.plot(alt_range2, dvs2, color=vc, lw=LW, label=f"v∞={vinf_val} km/s")
ax.set_xlabel("Target altitude (km)", fontsize=FL)
ax.set_ylabel("Total insertion ΔV (m/s)", fontsize=FL)
ax.set_title(
    "(e) Mars insertion vs orbit altitude", fontsize=FT, fontweight="bold", pad=3
)
ax.legend(fontsize=FG, framealpha=0.9)
ax.tick_params(labelsize=FK)

# (f) Porkchop: C3 vs departure/arrival date using Lambert solver
ax = axes10[1, 2]
_ax(ax)
from planet_rl.core.launch_window import PorkchopData as _PC

_r_e = 1.0 * AU
_r_m = 1.524 * AU
_dep = np.linspace(0, 780, 40)
_arr = np.linspace(150, 930, 40)
_pc = _PC.compute(
    _r_e, _r_m, _dep, _arr, dep_name="Earth", arr_name="Mars", min_tof_days=60
)
_C3 = np.where(_pc.valid, _pc.c3, np.nan)
_DEP, _ARR = np.meshgrid(_dep, _arr, indexing="ij")
im10 = ax.contourf(
    _DEP, _ARR, _C3, levels=np.linspace(5, 30, 14), cmap="RdYlGn_r", extend="max"
)
ax.contour(
    _DEP, _ARR, _C3, levels=[8, 10, 15, 20], colors="white", linewidths=0.5, alpha=0.6
)
cb10 = fig10.colorbar(im10, ax=ax, shrink=0.9, pad=0.03)
cb10.set_label("C3  (km²/s²)", fontsize=FA)
cb10.ax.tick_params(labelsize=FA)
ax.set_xlabel("Departure day", fontsize=FL)
ax.set_ylabel("Arrival day", fontsize=FL)
ax.set_title("(f) Earth→Mars porkchop", fontsize=FT, fontweight="bold", pad=3)
ax.tick_params(labelsize=FK)

fig10.suptitle(
    "Mission design: insertion ΔV, aerobraking, gravity assists, porkchop",
    fontsize=FT + 1,
    fontweight="bold",
)
save_figure(fig10, "fig10_mission_design", OUT)
plt.close(fig10)


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print(f"  All figures saved to ./{OUT}/")
print("=" * 60)

print("\n  Figures produced:")
for fn in sorted(os.listdir(OUT)):
    if fn.endswith(".png"):
        size_kb = os.path.getsize(os.path.join(OUT, fn)) // 1024
        print(f"    {fn:<45s}  {size_kb:4d} kB")

# ── Quick numeric summary ─────────────────────────────────────────────────────
print("\n  Key science outputs:")
ha_e = assess_habitability(earth, sun, 1.0 * AU)
ha_m = assess_habitability(mars, sun, 1.524 * AU)
print(f"    Earth habitability score : {ha_e.overall_score:.3f}  (grade {ha_e.grade})")
print(f"    Mars  habitability score : {ha_m.overall_score:.3f}  (grade {ha_m.grade})")
print(f"    Earth MoI factor         : {earth.derived_MoI():.4f}  (observed 0.3307)")
print(f"    Earth derived J2         : {earth.derived_J2():.4e}  (observed 1.083e-3)")
print(f"    Sun HZ conservative      : {sun.hz_inner_au:.3f}–{sun.hz_outer_au:.3f} AU")
print(
    f"    Mars insertion (2.5 km/s): {orbital_insertion_dv(mars, 2.5, 300)['dv_total_m_s']:.0f} m/s"
)
ab2 = plan_aerobraking(mars, 35000, 400, 115)
print(f"    Mars aerobraking passes  : {ab2.total_passes}")
print(
    f"    Earth 400km drag life    : {DragLifetime.lifetime_years(earth, 400_000, 1000, 100):.1f} yr"
)
cov_pct = cov.coverage_fraction() * 100
print(f"    3-day coverage (500km)   : {cov_pct:.1f}%")
seb = surface_energy_balance(earth, sun, 1.0 * AU, obliquity_deg=23.5)
print(f"    Earth global mean T      : {seb['global_mean_T_K']:.1f} K")
print(f"    Earth habitable area     : {seb['habitable_fraction'] * 100:.1f}%")
io = TidalHeating.io_analogue_heating(1.898e27, 421_800e3, 0.0041)
print(f"    Io tidal heating         : {io:.2e} W  (observed ~1e14 W)")

print("\n  Done.")
