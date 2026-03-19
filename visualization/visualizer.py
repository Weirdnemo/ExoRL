"""
visualizer.py  —  Planet-RL publication figures.

Rules that prevent ALL overlap/clipping:
  1. No twiny axes.
  2. No nested GridSpec. Every subplot is added with a flat index.
  3. Every figure sets left/right/top/bottom/hspace/wspace explicitly.
  4. Atmosphere profiles are ALWAYS 3 separate side-by-side subplots,
     never squeezed into a shared cell.
  5. Cross-section diagrams live in their own figure row.
  6. Text inside axes only — no annotations outside the axes box.
"""

from __future__ import annotations
import math, os
from typing import Optional, Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from core.planet import Planet
from core.physics import SpacecraftState

# ── Wong (2011) colorblind-safe palette ───────────────────────────────────────
W_BLUE    = "#0072B2"
W_ORANGE  = "#E69F00"
W_GREEN   = "#009E73"
W_RED     = "#D55E00"
W_PINK    = "#CC79A7"
W_SKY     = "#56B4E9"
W_YELLOW  = "#F0E442"
W_BLACK   = "#000000"
WONG      = [W_BLUE, W_RED, W_GREEN, W_ORANGE, W_PINK, W_SKY, W_YELLOW, W_BLACK]
WONG_CYCLE = WONG  # alias for backward compatibility

C_DENSITY  = W_RED
C_PRESSURE = W_BLUE
C_TEMP     = W_GREEN
C_ALT      = W_BLUE
C_SPEED    = W_RED
C_FUEL     = W_GREEN
C_HEAT     = W_ORANGE
C_TARGET   = W_BLACK

# ── Typography ────────────────────────────────────────────────────────────────
FT = 9    # title
FL = 8    # axis label
FK = 7    # tick label
FG = 7    # legend
FA = 6.5  # annotation
LW = 1.4  # main line weight
LW2= 0.9  # reference line weight

# ── Terrain / atmosphere colours ─────────────────────────────────────────────
ATM_COL = {
    "NONE":       "#d9d9d9",
    "CO2_THICK":  "#fdae61",
    "CO2_THIN":   "#fee08b",
    "NITROGEN":   "#abd9e9",
    "EARTH_LIKE": "#74add1",
    "HYDROGEN":   "#ffffbf",
    "METHANE":    "#c994c7",
    "CUSTOM":     "#cccccc",
}
TER_COL = {
    "FLAT":        "#b8cfa0",
    "CRATERED":    "#c4b49a",
    "MOUNTAINOUS": "#a89070",
    "OCEANIC":     "#7eb8d4",
    "VOLCANIC":    "#c08060",
    "RANDOM":      "#bbbbbb",
}

# ── Global style ──────────────────────────────────────────────────────────────
def apply_journal_style():
    matplotlib.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#444444",
        "axes.linewidth":    0.8,
        "axes.labelsize":    FL,
        "axes.titlesize":    FT,
        "axes.titleweight":  "bold",
        "axes.titlepad":     4,
        "axes.prop_cycle":   matplotlib.cycler(color=WONG),
        "axes.grid":         True,
        "grid.color":        "#e4e4e4",
        "grid.linewidth":    0.5,
        "grid.linestyle":    "--",
        "xtick.labelsize":   FK,
        "ytick.labelsize":   FK,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "legend.fontsize":   FG,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#cccccc",
        "legend.fancybox":   False,
        "lines.linewidth":   LW,
        "font.family":       "serif",
        "font.size":         FL,
        "mathtext.fontset":  "dejavuserif",
        "savefig.dpi":       300,
        "savefig.facecolor": "white",
        "pdf.fonttype":      42,
    })

apply_journal_style()


def _ax(ax):
    """Apply clean style to one axes."""
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_edgecolor("#444444")
        s.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FK, length=3, width=0.8)
    ax.grid(True, color="#e4e4e4", lw=0.5, ls="--")


def save_figure(fig, filename, output_dir=".", dpi_png=300, formats=("png","pdf")):
    """Save as raster PNG and vector PDF."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(output_dir, f"{filename}.{fmt}")
        kw = dict(bbox_inches="tight", facecolor="white")
        if fmt == "png":
            kw["dpi"] = dpi_png
        fig.savefig(path, format=fmt, **kw)
        print(f"  Saved: {path}")
        paths.append(path)
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# PLANET CROSS-SECTION  (single axes, no text outside bounds)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_planet_cross_section(planet: Planet, ax=None, ref_radius: float = None):
    """
    Draw a schematic cross-section of a planet.

    Parameters
    ----------
    ref_radius : float, optional
        Reference radius (metres) used to scale this planet relative to
        others drawn in the same figure.  All planets in a group should
        receive the same ref_radius = max(p.radius for p in group).
        When None (default, standalone figure) the planet fills the axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.8))
        fig.patch.set_facecolor("white")

    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.axis("off")

    comp  = planet.atmosphere.composition.name if planet.atmosphere.enabled else "NONE"
    atm_c = ATM_COL.get(comp, "#cccccc")
    ter_c = TER_COL.get(
        planet.terrain.terrain_type.name if planet.terrain.enabled else "FLAT",
        "#bbbbbb")

    # Scale factor: planet radius relative to the reference (largest) planet.
    # ref_radius=None → standalone, fills the axes (scale=1.0).
    if ref_radius is None or ref_radius <= 0:
        scale = 1.0
    else:
        scale = planet.radius / ref_radius   # 0 < scale <= 1.0

    # Atmosphere halos (radii are in normalised units × scale)
    if planet.atmosphere.enabled:
        max_alt = planet.atmosphere.scale_height * 5
        atm_frac = max_alt / planet.radius   # fractional thickness
        for i in range(6, 0, -1):
            frac = i / 6
            r = scale * (1.0 + atm_frac * frac)
            ax.add_patch(plt.Circle((0, 0), r, color=atm_c,
                                    alpha=0.06 + 0.16*(1-frac)**1.5, zorder=2))
        ax.add_patch(plt.Circle((0, 0), scale * 1.03, color=atm_c,
                                 alpha=0.4, fill=False, lw=1.5, zorder=3))

    # Surface disc
    ax.add_patch(plt.Circle((0, 0), scale, fc=ter_c, ec="#555555", lw=0.6, zorder=5))
    ax.add_patch(plt.Circle((-0.2*scale, 0.25*scale), 0.50*scale,
                             color="white", alpha=0.14, zorder=6))

    # Core
    cc = W_RED if planet.mean_density > 5000 else "#999999"
    ax.add_patch(plt.Circle((0, 0), 0.26*scale, fc=cc, ec="#333333",
                             lw=0.5, alpha=0.8, zorder=7))

    # Magnetic arcs
    if planet.magnetic_field.enabled:
        t = np.linspace(0, 2*math.pi, 200)
        for sc2, al in [(1.6, 0.45), (2.1, 0.25), (2.7, 0.12)]:
            ax.plot(sc2*scale*np.cos(t), sc2*scale*np.sin(t)*0.44,
                    color=W_BLUE, alpha=al, lw=0.8, zorder=4)

    # Moons
    if planet.moons.enabled:
        for i in range(min(planet.moons.count, 3)):
            ang = math.radians(i*120 + 30)
            ax.add_patch(plt.Circle(
                (2.2*scale*math.cos(ang), 2.2*scale*math.sin(ang)*0.5),
                0.09*scale, fc="#aaaaaa", ec="#555555", lw=0.4, zorder=5))

    tags = []
    if planet.atmosphere.enabled:
        tags.append(comp.lower().replace("_"," "))
    if planet.magnetic_field.enabled:
        tags.append("mag")
    if planet.oblateness.enabled:
        tags.append("J2")
    if planet.moons.enabled:
        tags.append(f"{planet.moons.count}mn")

    # Title above axes (no collision with drawing)
    ax.set_title(planet.name, fontsize=FT, fontweight="bold",
                 color="#111111", pad=4)

    # Stats and tags anchored to fixed positions below the disc
    ax.text(0, -1.75,
            f"R={planet.radius/1e3:,.0f} km   g={planet.surface_gravity:.1f}",
            ha="center", va="center",
            fontsize=FA-0.5, color="#444444", zorder=10)
    if tags:
        ax.text(0, -2.20, " · ".join(tags),
                ha="center", va="center",
                fontsize=FA-1.0, color="#777777", style="italic", zorder=10)

    # Fixed world-space limits so all planets in a row share the same frame
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-2.55, 1.7)
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# ATMOSPHERE PROFILE  —  always 3 separate axes, no twiny ever
# ═══════════════════════════════════════════════════════════════════════════════

def plot_atmosphere_profile(planet: Planet, axes=None, max_altitude_km=150):
    """
    Draw density / pressure / temperature each on their own axis.
    Pass axes = list/array of exactly 3 Axes, or None to create a new figure.
    Only the leftmost axis shows the y-label; the others suppress it.
    """
    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.8),
                                 sharey=True, squeeze=True)
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.10, right=0.97,
                            top=0.82, bottom=0.22, wspace=0.12)

    axes = list(axes)

    if not planet.atmosphere.enabled:
        for a in axes:
            a.set_visible(False)
        axes[1].set_visible(True)
        axes[1].set_facecolor("white")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].text(0.5, 0.5, "No atmosphere",
                     transform=axes[1].transAxes,
                     ha="center", va="center",
                     fontsize=FL, color="#aaaaaa", style="italic")
        axes[1].set_title(planet.name, fontsize=FT,
                          fontweight="bold", pad=4)
        return axes

    alts   = np.linspace(0, max_altitude_km * 1e3, 500)
    alts_km = alts / 1e3
    rho    = np.array([planet.atmosphere.density_at_altitude(h)      for h in alts])
    pres   = np.array([planet.atmosphere.pressure_at_altitude(h)/1e3 for h in alts])
    temp   = np.array([planet.atmosphere.temperature_at_altitude(h)  for h in alts])

    data   = [rho, pres, temp]
    colors = [C_DENSITY, C_PRESSURE, C_TEMP]
    xlabels= ["Density (kg/m³)", "Pressure (kPa)", "Temp (K)"]
    H_km   = planet.atmosphere.scale_height / 1e3

    for i, (ax, d, color, xlabel) in enumerate(zip(axes, data, colors, xlabels)):
        _ax(ax)
        ax.fill_betweenx(alts_km, d, alpha=0.10, color=color)
        ax.plot(d, alts_km, color=color, lw=LW)
        # Short x-labels, rotated ticks to prevent crowding
        ax.set_xlabel(xlabel, fontsize=FL-0.5, labelpad=1)
        ax.tick_params(axis='x', labelsize=FK-0.5, rotation=30)
        ax.set_ylim(0, max_altitude_km)

        # y-axis label and tick labels: ONLY on the first (leftmost) panel
        if i == 0:
            ax.set_ylabel("Altitude (km)", fontsize=FL, labelpad=2)
        else:
            # Hide y-axis completely on panels 1 and 2
            ax.set_ylabel("")
            ax.yaxis.set_visible(False)

        # Scale height dotted lines — no text labels to avoid clipping
        for mult in [1, 2, 3]:
            h = H_km * mult
            if 0 < h < max_altitude_km:
                ax.axhline(h, color="#cccccc", lw=0.5, ls=":", zorder=0)

        # Tight x limits
        ax.set_xlim(left=0)
        xmax = d.max()
        ax.set_xlim(0, xmax * 1.08)

    # Title on middle panel only
    axes[1].set_title(planet.name, fontsize=FT, fontweight="bold", pad=4)

    # Surface summary as a single clean line inside the left axes
    axes[0].text(0.97, 0.97,
                 f"P₀={planet.atmosphere.surface_pressure:.3g} Pa\n"
                 f"T₀={planet.atmosphere.surface_temp:.0f} K\n"
                 f"H={planet.atmosphere.scale_height/1e3:.1f} km",
                 transform=axes[0].transAxes,
                 ha="right", va="top",
                 fontsize=FA, color="#555555",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white",
                           ec="#cccccc", lw=0.5))

    return axes


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY 2D
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trajectory_2d(planet: Planet, trajectory, ax=None,
                       target_altitude=200_000, color_by="speed"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        fig.patch.set_facecolor("white")

    ax.set_facecolor("#f6f6f6")
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_edgecolor("#999999")
        sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    R     = planet.radius
    comp  = planet.atmosphere.composition.name if planet.atmosphere.enabled else "NONE"
    ter_c = TER_COL.get(
        planet.terrain.terrain_type.name if planet.terrain.enabled else "FLAT", "#bbb")
    atm_c = ATM_COL.get(comp, "#cccccc")

    if planet.atmosphere.enabled:
        ax.add_patch(plt.Circle((0,0), 1.08, color=atm_c, alpha=0.28, zorder=2))
    ax.add_patch(plt.Circle((0,0), 1.0, fc=ter_c, ec="#555555", lw=0.6, zorder=3))

    r_tgt = (R + target_altitude) / R
    ax.add_patch(plt.Circle((0,0), r_tgt, fill=False,
                             color=C_TARGET, ls="--", lw=LW2, zorder=4,
                             label=f"Target {target_altitude/1e3:.0f} km"))

    xs = np.array([s.x / R for s in trajectory])
    ys = np.array([s.y / R for s in trajectory])

    if color_by == "speed":
        vals   = np.array([s.speed for s in trajectory]) / 1e3
        cmap   = matplotlib.colormaps["plasma"]
        clabel = "Speed (km/s)"
    else:
        vals   = np.array([s.fuel_mass for s in trajectory])
        cmap   = matplotlib.colormaps["RdYlGn"]
        clabel = "Fuel (kg)"

    norm  = matplotlib.colors.Normalize(vals.min(), vals.max())
    cvals = norm(vals)
    for i in range(len(xs)-1):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=cmap(cvals[i]), lw=1.2, alpha=0.9, zorder=5)

    ax.plot(xs[0],  ys[0],  "o", color=W_GREEN, ms=5, zorder=10, label="Start",
            mec="#333", mew=0.4)
    ax.plot(xs[-1], ys[-1], "s", color=W_RED,   ms=5, zorder=10, label="End",
            mec="#333", mew=0.4)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, shrink=0.65, pad=0.03, aspect=20)
    cb.set_label(clabel, fontsize=FL)
    cb.ax.tick_params(labelsize=FK)

    lim = max(abs(xs).max(), abs(ys).max()) * 1.18
    ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
    ax.set_xlabel("x / Rp", fontsize=FL)
    ax.set_ylabel("y / Rp", fontsize=FL)
    ax.tick_params(labelsize=FK)
    ax.set_title(f"{planet.name} — trajectory", fontsize=FT, fontweight="bold", pad=5)
    ax.legend(loc="upper left", fontsize=FG, framealpha=0.92,
              edgecolor="#cccccc", fancybox=False)
    ax.grid(True, color="#dddddd", lw=0.5, ls="--")
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mission_telemetry(trajectory, planet: Planet,
                           target_altitude=200_000, figsize=(7.0, 4.5)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.46, wspace=0.32,
                        left=0.10, right=0.97,
                        top=0.90, bottom=0.12)

    times  = np.array([s.time                   for s in trajectory])
    alts   = np.array([s.radius - planet.radius  for s in trajectory]) / 1e3
    speeds = np.array([s.speed                   for s in trajectory]) / 1e3
    fuels  = np.array([s.fuel_mass               for s in trajectory])
    heats  = np.array([s.heat_load               for s in trajectory])
    t_min  = times / 60.0

    rows = [
        (axes[0,0], alts,   C_ALT,   "Altitude (km)",    "(a) Altitude",
         target_altitude/1e3, f"{target_altitude/1e3:.0f} km"),
        (axes[0,1], speeds, C_SPEED, "Speed (km/s)",     "(b) Speed",
         planet.circular_orbit_speed(target_altitude)/1e3, "v_circ"),
        (axes[1,0], fuels,  C_FUEL,  "Propellant (kg)",  "(c) Propellant", None, None),
        (axes[1,1], heats,  C_HEAT,  "Heat load (J/m²)", "(d) Aeroheating", None, None),
    ]

    for ax, y, color, ylabel, title, hline, hlabel in rows:
        _ax(ax)
        ax.fill_between(t_min, y, alpha=0.12, color=color)
        ax.plot(t_min, y, color=color, lw=LW)
        if hline is not None:
            ax.axhline(hline, color=C_TARGET, ls="--", lw=LW2, label=hlabel)
            ax.legend(fontsize=FG, framealpha=0.9, edgecolor="#cccccc", fancybox=False)
        ax.set_xlabel("Time (min)", fontsize=FL, labelpad=2)
        ax.set_ylabel(ylabel,       fontsize=FL, labelpad=2)
        ax.set_title(title,         fontsize=FT, fontweight="bold", pad=3)
        ax.tick_params(labelsize=FK)

    fig.suptitle(f"Orbital insertion telemetry — {planet.name}",
                 fontsize=FT+1, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PLANET COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def plot_planet_comparison(planets, figsize=(7.0, 3.2)):
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(wspace=0.50, left=0.08, right=0.97,
                        top=0.83, bottom=0.28)

    names  = [p.name for p in planets]
    colors = [WONG[i % len(WONG)] for i in range(len(planets))]

    props = [
        ("Radius (R⊕)",    [p.radius/6.371e6        for p in planets]),
        ("g (m/s²)",       [p.surface_gravity        for p in planets]),
        ("v_esc (km/s)",   [p.escape_velocity/1e3    for p in planets]),
        ("ρ₀ (kg/m³)",    [p.atmosphere.surface_density
                            if p.atmosphere.enabled else 0 for p in planets]),
    ]

    for ax, (ylabel, vals) in zip(axes, props):
        _ax(ax)
        bars = ax.bar(range(len(names)), vals, color=colors,
                      edgecolor="white", lw=0.5, width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=38, ha="right",
                           fontsize=FK, rotation_mode="anchor")
        ax.set_ylabel(ylabel, fontsize=FL, labelpad=2)
        ax.tick_params(axis="y", labelsize=FK)
        ax.grid(True, axis="y", color="#e4e4e4", lw=0.5, ls="--")
        ax.grid(False, axis="x")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.03,
                    f"{val:.2g}", ha="center", va="bottom",
                    fontsize=FA, color="#333333")

    fig.suptitle("Planet physical properties",
                 fontsize=FT+1, fontweight="bold")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — Interplanetary transfer visualisations
# ═════════════════════════════════════════════════════════════════════════════

def plot_heliocentric_transfer(
        transfer_trajectory,           # (N,6) array [x,y,z,vx,vy,vz] in m
        departure_radius_m,            # departure planet orbital radius [m]
        arrival_radius_m,              # arrival planet orbital radius [m]
        tof_s,                         # time of flight [s]
        departure_name="Departure",
        arrival_name="Arrival",
        star_name="Sun",
        additional_orbits=None,        # list of (radius_m, name, color) for other planets
        agent_departure_day=None,      # optional: agent's chosen departure day
        ax=None,
        figsize=(6.5, 6.5),
        show_velocity_arrows=True,
        mu_star=None,
):
    """
    Plot the heliocentric transfer orbit in the ecliptic plane.

    Shows:
    - Star at centre
    - Departure and arrival planet orbits (full circles)
    - Transfer arc (coloured by speed)
    - Planet positions at departure and arrival
    - Departure/arrival velocity arrows
    - Optional: additional planet orbits for context

    Parameters
    ----------
    transfer_trajectory : (N,6) array from KeplerPropagator.orbit_at_time()
                          columns: [x, y, z, vx, vy, vz]  in SI
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyArrowPatch, Circle
    AU_m = 1.495978707e11

    if mu_star is None:
        mu_star = 6.674e-11 * 1.989e30

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    _ax(ax)
    ax.set_aspect("equal")
    ax.set_facecolor("#F8F9FA")

    traj = np.asarray(transfer_trajectory)
    x_au = traj[:, 0] / AU_m
    y_au = traj[:, 1] / AU_m
    speeds = np.linalg.norm(traj[:, 3:6], axis=1)

    # ── Background planet orbits ──────────────────────────────────────────────
    theta_ring = np.linspace(0, 2*np.pi, 500)
    for r_m, name, col in (additional_orbits or []):
        r_au = r_m / AU_m
        ax.plot(r_au*np.cos(theta_ring), r_au*np.sin(theta_ring),
                color=col, lw=0.6, ls="--", alpha=0.4, zorder=1)

    r_dep_au = departure_radius_m / AU_m
    r_arr_au = arrival_radius_m / AU_m
    ax.plot(r_dep_au*np.cos(theta_ring), r_dep_au*np.sin(theta_ring),
            color=W_BLUE, lw=0.9, ls="-", alpha=0.5, zorder=2,
            label=f"{departure_name} orbit")
    ax.plot(r_arr_au*np.cos(theta_ring), r_arr_au*np.sin(theta_ring),
            color=W_RED, lw=0.9, ls="-", alpha=0.5, zorder=2,
            label=f"{arrival_name} orbit")

    # ── Transfer arc (coloured by speed) ──────────────────────────────────────
    from matplotlib.collections import LineCollection
    points  = np.array([x_au, y_au]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    norm    = mcolors.Normalize(vmin=speeds.min(), vmax=speeds.max())
    cmap    = plt.cm.plasma
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2.0,
                        zorder=4, alpha=0.9)
    lc.set_array(speeds[:-1])
    ax.add_collection(lc)

    # Colorbar for speed
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("Transfer speed (m/s)", fontsize=FA)
    cbar.ax.tick_params(labelsize=FA-1)

    # ── Planet positions at departure and arrival ─────────────────────────────
    dep_pos = traj[0,  :2] / AU_m
    arr_pos = traj[-1, :2] / AU_m

    ax.plot(*dep_pos, "o", color=W_BLUE,   ms=9, zorder=6,
            label=f"{departure_name} (dep)")
    ax.plot(*arr_pos, "o", color=W_RED,    ms=9, zorder=6,
            label=f"{arrival_name} (arr)")

    # Labels
    offset = 0.06
    ax.text(dep_pos[0]+offset, dep_pos[1]+offset, departure_name,
            fontsize=FA+0.5, color=W_BLUE, fontweight="bold", zorder=7)
    ax.text(arr_pos[0]+offset, arr_pos[1]+offset, arrival_name,
            fontsize=FA+0.5, color=W_RED,  fontweight="bold", zorder=7)

    # ── Velocity arrows ───────────────────────────────────────────────────────
    if show_velocity_arrows:
        dep_v = traj[0,  3:5]
        arr_v = traj[-1, 3:5]
        v_scale = r_dep_au / (np.linalg.norm(dep_v) * 15)
        ax.annotate("", xytext=tuple(dep_pos),
                    xy=(dep_pos[0]+dep_v[0]*v_scale, dep_pos[1]+dep_v[1]*v_scale),
                    arrowprops=dict(arrowstyle="->", color=W_BLUE, lw=1.5),
                    zorder=7)
        v_scale2 = r_arr_au / (np.linalg.norm(arr_v) * 15)
        ax.annotate("", xytext=tuple(arr_pos),
                    xy=(arr_pos[0]+arr_v[0]*v_scale2, arr_pos[1]+arr_v[1]*v_scale2),
                    arrowprops=dict(arrowstyle="->", color=W_RED, lw=1.5),
                    zorder=7)

    # ── Star ──────────────────────────────────────────────────────────────────
    ax.plot(0, 0, "*", color=W_YELLOW, ms=14, markeredgecolor=W_ORANGE,
            markeredgewidth=0.5, zorder=5, label=star_name)

    # ── Direction of flight markers ───────────────────────────────────────────
    mid_idx = len(traj) // 2
    mid_pos = traj[mid_idx, :2] / AU_m
    mid_v   = traj[mid_idx, 3:5]
    ax.annotate("", xytext=tuple(mid_pos),
                xy=(mid_pos[0]+mid_v[0]*r_dep_au/(np.linalg.norm(mid_v)*20),
                    mid_pos[1]+mid_v[1]*r_dep_au/(np.linalg.norm(mid_v)*20)),
                arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.0),
                zorder=5)

    # ── Labels and decoration ─────────────────────────────────────────────────
    max_r = max(r_dep_au, r_arr_au) * 1.15
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_xlabel("x (AU)", fontsize=FL)
    ax.set_ylabel("y (AU)", fontsize=FL)
    ax.tick_params(labelsize=FK)
    ax.legend(fontsize=FG, loc="upper right", framealpha=0.9,
              handlelength=1.2, borderpad=0.5)

    tof_days = tof_s / 86400
    ax.set_title(
        f"Heliocentric transfer: {departure_name} → {arrival_name}  "
        f"(ToF = {tof_days:.0f} d)",
        fontsize=FT, fontweight="bold", pad=4
    )

    if standalone:
        return ax.get_figure()
    return ax


def plot_porkchop(porkchop_data,
                   ax=None,
                   figsize=(6.5, 5.5),
                   max_c3=40.0,
                   max_vinf_arr=8.0,
                   quantity="c3",               # "c3", "vinf_arr", or "tof"
                   contour_levels=None,
                   best_window=None,             # LaunchWindow object to mark
                   agent_choice=None,            # (dep_idx, arr_idx) to mark
                   colormap="RdYlGn_r",
                   show_tof_contours=True,
):
    """
    Plot a porkchop diagram (C3 or arrival v∞ over departure × arrival date grid).

    The classic mission design tool. Green valleys = optimal launch windows.

    Parameters
    ----------
    porkchop_data : PorkchopData object
    quantity      : "c3"       → plot launch energy C3 [km²/s²]
                    "vinf_arr" → plot arrival v∞ [km/s]
                    "tof"      → plot time of flight [days]
    best_window   : LaunchWindow to mark with a star
    agent_choice  : (dep_idx, arr_idx) tuple to mark with a circle
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    pc = porkchop_data
    dep = pc.departure_days
    arr = pc.arrival_days

    if quantity == "c3":
        data  = np.where(pc.valid, pc.c3, np.nan)
        label = "C3 (km²/s²)"
        vmin, vmax = 0, max_c3
        cmap_use = colormap
    elif quantity == "vinf_arr":
        data  = np.where(pc.valid, pc.vinf_arr, np.nan)
        label = "Arrival v∞ (km/s)"
        vmin, vmax = 0, max_vinf_arr
        cmap_use = colormap
    else:
        data  = np.where(pc.valid, pc.tof, np.nan)
        label = "Time of flight (days)"
        vmin  = np.nanmin(data) if not np.all(np.isnan(data)) else 0
        vmax  = np.nanmax(data) if not np.all(np.isnan(data)) else 500
        cmap_use = "viridis"

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    _ax(ax)

    DEP, ARR = np.meshgrid(dep, arr, indexing="ij")

    # Filled contour
    if contour_levels is None:
        n_levels = 16
        contour_levels = np.linspace(vmin, vmax, n_levels)

    cf = ax.contourf(DEP, ARR, data,
                      levels=contour_levels, cmap=cmap_use,
                      extend="max", alpha=0.90)
    ax.contour(DEP, ARR, data,
               levels=contour_levels[::2], colors="white",
               linewidths=0.5, alpha=0.6)

    # ToF contours overlaid
    if show_tof_contours and quantity != "tof":
        tof_data = np.where(pc.valid, pc.tof, np.nan)
        tof_min  = np.nanmin(tof_data) if not np.all(np.isnan(tof_data)) else 100
        tof_max  = np.nanmax(tof_data) if not np.all(np.isnan(tof_data)) else 500
        tof_lvls = np.arange(int(tof_min//50)*50, tof_max+50, 50)
        ct = ax.contour(DEP, ARR, tof_data,
                         levels=tof_lvls, colors="#333333",
                         linewidths=0.7, linestyles="--", alpha=0.5)
        ax.clabel(ct, inline=True, fontsize=FA-1, fmt="%d d")

    # Colorbar
    cbar = ax.get_figure().colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(label, fontsize=FL)
    cbar.ax.tick_params(labelsize=FK)

    # Mark best window
    if best_window is not None:
        ax.plot(best_window.departure_day, best_window.arrival_day,
                "*", ms=14, color=W_YELLOW, markeredgecolor=W_BLACK,
                markeredgewidth=0.8, zorder=10,
                label=f"Best: C3={best_window.c3_km2_s2:.1f} km²/s²")
        ax.legend(fontsize=FG, loc="upper left", framealpha=0.9)

    # Mark agent choice
    if agent_choice is not None:
        dep_idx, arr_idx = agent_choice
        if 0 <= dep_idx < len(dep) and 0 <= arr_idx < len(arr):
            ax.plot(dep[dep_idx], arr[arr_idx],
                    "o", ms=10, color=W_ORANGE,
                    markeredgecolor=W_BLACK, markeredgewidth=0.8,
                    zorder=10, label="Agent choice")
            ax.legend(fontsize=FG, loc="upper left", framealpha=0.9)

    ax.set_xlabel(f"Departure day (from epoch)", fontsize=FL)
    ax.set_ylabel(f"Arrival day (from epoch)",   fontsize=FL)
    ax.tick_params(labelsize=FK)

    title_map = {"c3": "Launch energy (C3)", "vinf_arr": "Arrival v∞",
                 "tof": "Time of flight"}
    ax.set_title(
        f"Porkchop: {pc.departure_planet} → {pc.arrival_planet}  —  "
        f"{title_map.get(quantity, quantity)}",
        fontsize=FT, fontweight="bold", pad=4
    )

    if standalone:
        return ax.get_figure()
    return ax


def plot_soi_approach(approach_trajectory,
                       planet,
                       soi_radius_m,
                       v_inf_m_s,
                       periapsis_alt_km=300.0,
                       ax=None,
                       figsize=(5.5, 5.5),
                       show_capture_orbit=True,
                       target_alt_km=300.0,
):
    """
    Zoom-in plot of the hyperbolic approach and capture within the planet's SOI.

    Shows:
    - Planet and SOI boundary
    - Incoming hyperbolic trajectory
    - Capture burn location (at periapsis)
    - Resulting capture/science orbit

    Parameters
    ----------
    approach_trajectory : (N,6) array of spacecraft state in planet-centred frame [m, m/s]
    planet              : Planet object
    soi_radius_m        : sphere of influence radius [m]
    v_inf_m_s           : arrival hyperbolic excess speed [m/s]
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    traj = np.asarray(approach_trajectory)
    x_km = traj[:, 0] / 1e3
    y_km = traj[:, 1] / 1e3
    R    = planet.radius / 1e3      # km
    soi  = soi_radius_m / 1e3      # km
    mu   = 6.674e-11 * planet.mass

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    _ax(ax)
    ax.set_aspect("equal")
    ax.set_facecolor("#0D1117")   # dark space background

    # ── SOI boundary ──────────────────────────────────────────────────────────
    theta = np.linspace(0, 2*np.pi, 500)
    ax.plot(soi*np.cos(theta), soi*np.sin(theta),
            color="#4466AA", lw=1.0, ls="--", alpha=0.6,
            label=f"SOI ({soi/1e3:.0f} Mm)")

    # ── Planet body ───────────────────────────────────────────────────────────
    planet_circle = plt.Circle((0, 0), R, color=W_ORANGE, alpha=0.85, zorder=5)
    ax.add_patch(planet_circle)
    ax.text(0, 0, planet.name, ha="center", va="center",
            fontsize=FA, color="white", fontweight="bold", zorder=6)

    # ── Hyperbolic approach trajectory ────────────────────────────────────────
    speeds = np.linalg.norm(traj[:, 3:6], axis=1)
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    points = np.array([x_km, y_km]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    norm   = mcolors.Normalize(vmin=speeds.min(), vmax=speeds.max())
    lc = LineCollection(segs, cmap="hot", norm=norm, lw=2.0, zorder=4)
    lc.set_array(speeds[:-1])
    ax.add_collection(lc)

    # ── Periapsis marker (capture burn location) ──────────────────────────────
    r_peri_km = R + periapsis_alt_km
    # Find closest approach in trajectory
    dists = np.linalg.norm(traj[:, :2], axis=1) / 1e3
    peri_idx = np.argmin(dists)
    peri_x, peri_y = x_km[peri_idx], y_km[peri_idx]

    ax.plot(peri_x, peri_y, "v", ms=10, color=W_GREEN, zorder=8,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate(f"  Capture burn\n  alt={periapsis_alt_km:.0f} km",
                (peri_x, peri_y), fontsize=FA-1, color=W_GREEN,
                xytext=(peri_x + R*0.3, peri_y - R*0.2), zorder=9)

    # ── Capture orbit ─────────────────────────────────────────────────────────
    if show_capture_orbit:
        r_tgt_km = R + target_alt_km
        # Show circular science orbit
        ax.plot(r_tgt_km*np.cos(theta), r_tgt_km*np.sin(theta),
                color=W_GREEN, lw=1.2, ls=":", alpha=0.7,
                label=f"Science orbit ({target_alt_km:.0f} km)")

    # ── v∞ annotation ─────────────────────────────────────────────────────────
    ax.text(0.03, 0.97,
            f"v∞ = {v_inf_m_s/1e3:.2f} km/s\nR_SOI = {soi/1e3:.0f} Mm",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=FA, color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", alpha=0.8))

    # ── Axes ──────────────────────────────────────────────────────────────────
    # Set limits to show the full trajectory with planet visible
    traj_x_max = np.abs(x_km).max() * 1.1
    traj_y_ext = max(np.abs(y_km).max(), R * 3) * 1.2
    lim_x = max(traj_x_max, R * 5)
    lim_y = max(traj_y_ext, R * 5)
    ax.set_xlim(-lim_x * 0.15, lim_x)
    ax.set_ylim(-lim_y, lim_y)
    ax.set_xlabel("x (km)", fontsize=FL, color="white")
    ax.set_ylabel("y (km)", fontsize=FL, color="white")
    ax.tick_params(labelsize=FK, colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.legend(fontsize=FG, loc="lower right", framealpha=0.7,
              facecolor="#1a1a2e", edgecolor="#444444",
              labelcolor="white")
    ax.set_title(
        f"SOI approach: {planet.name}  (v∞={v_inf_m_s/1e3:.2f} km/s)",
        fontsize=FT, fontweight="bold", pad=4, color="#DDDDDD"
    )

    if standalone:
        return ax.get_figure()
    return ax


def plot_transfer_dashboard(
        departure_planet,
        arrival_planet,
        porkchop_data,
        transfer_trajectory,
        tof_s,
        v_inf_dep_m_s,
        v_inf_arr_m_s,
        approach_trajectory=None,
        star_name="Sun",
        best_window=None,
        agent_choice=None,
        output_dir=".",
        filename="transfer_dashboard",
        additional_orbits=None,
        soi_radius_arr_m=None,
):
    """
    Full 4-panel transfer dashboard combining all three plot types.

    Panels:
    (a) Top-left:  Heliocentric transfer orbit
    (b) Top-right: Porkchop C3 plot
    (c) Bottom-left:  Porkchop v∞ arrival plot
    (d) Bottom-right: SOI approach zoom (if approach_trajectory provided)

    Parameters
    ----------
    departure_planet   : Planet object (departure body)
    arrival_planet     : Planet object (target body)
    porkchop_data      : PorkchopData object
    transfer_trajectory: (N,6) heliocentric trajectory array [m, m/s]
    tof_s              : time of flight [s]
    v_inf_dep_m_s      : departure v∞ [m/s]
    v_inf_arr_m_s      : arrival v∞ [m/s]
    approach_trajectory: (N,6) planetocentric trajectory [m, m/s] (optional)
    soi_radius_arr_m   : SOI radius at arrival planet [m]
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    AU_m = 1.495978707e11

    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                             hspace=0.38, wspace=0.30,
                             left=0.07, right=0.97,
                             top=0.94, bottom=0.06)

    # ── Panel (a): Heliocentric transfer ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    plot_heliocentric_transfer(
        transfer_trajectory,
        departure_planet.orbital_distance_m,
        arrival_planet.orbital_distance_m,
        tof_s,
        departure_name=departure_planet.name,
        arrival_name=arrival_planet.name,
        star_name=star_name,
        additional_orbits=additional_orbits,
        ax=ax_a,
        show_velocity_arrows=True,
    )

    # ── Panel (b): Porkchop C3 ────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    plot_porkchop(porkchop_data, ax=ax_b, quantity="c3",
                   best_window=best_window, agent_choice=agent_choice)

    # ── Panel (c): Porkchop v∞ arrival ───────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    plot_porkchop(porkchop_data, ax=ax_c, quantity="vinf_arr",
                   show_tof_contours=False)

    # ── Panel (d): SOI approach or mission summary ────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    if approach_trajectory is not None and soi_radius_arr_m is not None:
        plot_soi_approach(
            approach_trajectory, arrival_planet,
            soi_radius_arr_m, v_inf_arr_m_s,
            ax=ax_d,
            target_alt_km=300.0,
        )
    else:
        # Summary text panel
        _ax(ax_d); ax_d.axis("off")
        summary_lines = [
            f"Mission: {departure_planet.name} → {arrival_planet.name}",
            "",
            f"Time of flight:     {tof_s/86400:.0f} days",
            f"Departure v∞:       {v_inf_dep_m_s/1e3:.3f} km/s",
            f"Arrival v∞:         {v_inf_arr_m_s/1e3:.3f} km/s",
            f"C3:                 {(v_inf_dep_m_s/1e3)**2:.2f} km²/s²",
            "",
        ]
        if best_window is not None:
            summary_lines += [
                "Optimal window:",
                f"  Depart day {best_window.departure_day:.0f}",
                f"  Arrive day {best_window.arrival_day:.0f}",
                f"  Best C3: {best_window.c3_km2_s2:.2f} km²/s²",
                f"  v∞ arr:  {best_window.vinf_arr_km_s:.3f} km/s",
            ]
        ax_d.text(0.08, 0.92, "\n".join(summary_lines),
                  transform=ax_d.transAxes,
                  va="top", ha="left",
                  fontsize=FL-0.5, family="monospace",
                  color=W_BLACK)
        ax_d.set_title("Mission summary", fontsize=FT, fontweight="bold", pad=4)

    fig.suptitle(
        f"Interplanetary mission: {departure_planet.name} → {arrival_planet.name}",
        fontsize=FT+2, fontweight="bold"
    )

    save_figure(fig, filename, output_dir)
    return fig