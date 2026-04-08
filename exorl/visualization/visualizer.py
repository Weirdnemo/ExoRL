"""
visualizer.py  —  ExoRL publication figures.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from exorl.core.physics import SpacecraftState
from exorl.core.planet import Planet

# ── Wong (2011) colorblind-safe palette ───────────────────────────────────────
W_BLUE = "#0072B2"
W_ORANGE = "#E69F00"
W_GREEN = "#009E73"
W_RED = "#D55E00"
W_PINK = "#CC79A7"
W_SKY = "#56B4E9"
W_YELLOW = "#F0E442"
W_BLACK = "#000000"
WONG = [W_BLUE, W_RED, W_GREEN, W_ORANGE, W_PINK, W_SKY, W_YELLOW, W_BLACK]
WONG_CYCLE = WONG

C_DENSITY = W_RED
C_PRESSURE = W_BLUE
C_TEMP = W_GREEN
C_ALT = W_BLUE
C_SPEED = W_RED
C_FUEL = W_GREEN
C_HEAT = W_ORANGE
C_TARGET = W_BLACK

# ── Typography ────────────────────────────────────────────────────────────────
FT = 9  # title
FL = 8  # axis label
FK = 7  # tick label
FG = 7  # legend
FA = 6.5  # annotation
LW = 1.4  # main line weight
LW2 = 0.9  # reference line weight

# ── Terrain / atmosphere colours ─────────────────────────────────────────────
ATM_COL = {
    "NONE": "#d9d9d9",
    "CO2_THICK": "#fdae61",
    "CO2_THIN": "#fee08b",
    "NITROGEN": "#abd9e9",
    "EARTH_LIKE": "#74add1",
    "HYDROGEN": "#ffffbf",
    "METHANE": "#c994c7",
    "CUSTOM": "#cccccc",
}
TER_COL = {
    "FLAT": "#b8cfa0",
    "CRATERED": "#c4b49a",
    "MOUNTAINOUS": "#a89070",
    "OCEANIC": "#7eb8d4",
    "VOLCANIC": "#c08060",
    "RANDOM": "#bbbbbb",
}


# ── Global style ──────────────────────────────────────────────────────────────
def apply_journal_style():
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.labelsize": FL,
            "axes.titlesize": FT,
            "axes.titleweight": "bold",
            "axes.titlepad": 4,
            "axes.prop_cycle": matplotlib.cycler(color=WONG),
            "axes.grid": True,
            "grid.color": "#e4e4e4",
            "grid.linewidth": 0.5,
            "grid.linestyle": "--",
            "xtick.labelsize": FK,
            "ytick.labelsize": FK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": FG,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
            "legend.fancybox": False,
            "lines.linewidth": LW,
            "font.family": "serif",
            "font.size": FL,
            "mathtext.fontset": "dejavuserif",
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


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


def save_figure(fig, filename, output_dir=".", dpi_png=300, formats=("png", "pdf")):
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
# PLANET CROSS-SECTION
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

    comp = planet.atmosphere.composition.name if planet.atmosphere.enabled else "NONE"
    atm_c = ATM_COL.get(comp, "#cccccc")
    ter_c = TER_COL.get(
        planet.terrain.terrain_type.name if planet.terrain.enabled else "FLAT",
        "#bbbbbb",
    )

    if ref_radius is None or ref_radius <= 0:
        scale = 1.0
    else:
        scale = planet.radius / ref_radius

    # Atmosphere halos (radii are in normalised units × scale)
    if planet.atmosphere.enabled:
        max_alt = planet.atmosphere.scale_height * 5
        atm_frac = max_alt / planet.radius
        for i in range(6, 0, -1):
            frac = i / 6
            r = scale * (1.0 + atm_frac * frac)
            ax.add_patch(
                plt.Circle(
                    (0, 0),
                    r,
                    color=atm_c,
                    alpha=0.06 + 0.16 * (1 - frac) ** 1.5,
                    zorder=2,
                )
            )
        ax.add_patch(
            plt.Circle(
                (0, 0),
                scale * 1.03,
                color=atm_c,
                alpha=0.4,
                fill=False,
                lw=1.5,
                zorder=3,
            )
        )

    # Surface disc
    ax.add_patch(plt.Circle((0, 0), scale, fc=ter_c, ec="#555555", lw=0.6, zorder=5))
    ax.add_patch(
        plt.Circle(
            (-0.2 * scale, 0.25 * scale),
            0.50 * scale,
            color="white",
            alpha=0.14,
            zorder=6,
        )
    )

    # Core
    cc = W_RED if planet.mean_density > 5000 else "#999999"
    ax.add_patch(
        plt.Circle(
            (0, 0), 0.26 * scale, fc=cc, ec="#333333", lw=0.5, alpha=0.8, zorder=7
        )
    )

    # Magnetic arcs
    if planet.magnetic_field.enabled:
        t = np.linspace(0, 2 * math.pi, 200)
        for sc2, al in [(1.6, 0.45), (2.1, 0.25), (2.7, 0.12)]:
            ax.plot(
                sc2 * scale * np.cos(t),
                sc2 * scale * np.sin(t) * 0.44,
                color=W_BLUE,
                alpha=al,
                lw=0.8,
                zorder=4,
            )

    # Moons
    if planet.moons.enabled:
        for i in range(min(planet.moons.count, 3)):
            ang = math.radians(i * 120 + 30)
            ax.add_patch(
                plt.Circle(
                    (2.2 * scale * math.cos(ang), 2.2 * scale * math.sin(ang) * 0.5),
                    0.09 * scale,
                    fc="#aaaaaa",
                    ec="#555555",
                    lw=0.4,
                    zorder=5,
                )
            )

    tags = []
    if planet.atmosphere.enabled:
        tags.append(comp.lower().replace("_", " "))
    if planet.magnetic_field.enabled:
        tags.append("mag")
    if planet.oblateness.enabled:
        tags.append("J2")
    if planet.moons.enabled:
        tags.append(f"{planet.moons.count}mn")

    ax.set_title(planet.name, fontsize=FT, fontweight="bold", color="#111111", pad=4)

    ax.text(
        0,
        -1.75,
        f"R={planet.radius / 1e3:,.0f} km   g={planet.surface_gravity:.1f}",
        ha="center",
        va="center",
        fontsize=FA - 0.5,
        color="#444444",
        zorder=10,
    )
    if tags:
        ax.text(
            0,
            -2.20,
            " · ".join(tags),
            ha="center",
            va="center",
            fontsize=FA - 1.0,
            color="#777777",
            style="italic",
            zorder=10,
        )

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
        fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.8), sharey=True, squeeze=True)
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.22, wspace=0.12)

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
        axes[1].text(
            0.5,
            0.5,
            "No atmosphere",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=FL,
            color="#aaaaaa",
            style="italic",
        )
        axes[1].set_title(planet.name, fontsize=FT, fontweight="bold", pad=4)
        return axes

    alts = np.linspace(0, max_altitude_km * 1e3, 500)
    alts_km = alts / 1e3
    rho = np.array([planet.atmosphere.density_at_altitude(h) for h in alts])
    pres = np.array([planet.atmosphere.pressure_at_altitude(h) / 1e3 for h in alts])
    temp = np.array([planet.atmosphere.temperature_at_altitude(h) for h in alts])

    data = [rho, pres, temp]
    colors = [C_DENSITY, C_PRESSURE, C_TEMP]
    xlabels = ["Density (kg/m³)", "Pressure (kPa)", "Temp (K)"]
    H_km = planet.atmosphere.scale_height / 1e3

    for i, (ax, d, color, xlabel) in enumerate(zip(axes, data, colors, xlabels)):
        _ax(ax)
        ax.fill_betweenx(alts_km, d, alpha=0.10, color=color)
        ax.plot(d, alts_km, color=color, lw=LW)
        ax.set_xlabel(xlabel, fontsize=FL - 0.5, labelpad=1)
        ax.tick_params(axis="x", labelsize=FK - 0.5, rotation=30)
        ax.set_ylim(0, max_altitude_km)

        if i == 0:
            ax.set_ylabel("Altitude (km)", fontsize=FL, labelpad=2)
        else:
            ax.set_ylabel("")
            ax.yaxis.set_visible(False)

        for mult in [1, 2, 3]:
            h = H_km * mult
            if 0 < h < max_altitude_km:
                ax.axhline(h, color="#cccccc", lw=0.5, ls=":", zorder=0)

        ax.set_xlim(left=0)
        xmax = d.max()
        ax.set_xlim(0, xmax * 1.08)

    axes[1].set_title(planet.name, fontsize=FT, fontweight="bold", pad=4)

    axes[0].text(
        0.97,
        0.97,
        f"P₀={planet.atmosphere.surface_pressure:.3g} Pa\n"
        f"T₀={planet.atmosphere.surface_temp:.0f} K\n"
        f"H={planet.atmosphere.scale_height / 1e3:.1f} km",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=FA,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5),
    )

    return axes


# ═══════════════════════════════════════════════════════════════════════════════
# ATMOSPHERE SINGLE — single-axis density profile (backwards-compatible helper)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_atmosphere_single(
    planet: Planet, ax=None, max_altitude_km=150, quantity="density"
):
    """
    Draw a single atmosphere profile panel on one axis.
    Backwards-compatible replacement for the old plot_atmosphere_profile(ax=ax) API.

    Parameters
    ----------
    planet            : Planet object
    ax                : single matplotlib Axes, or None to create a new figure
    max_altitude_km   : altitude ceiling in km
    quantity          : "density" | "pressure" | "temperature"
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(3.0, 3.5))
        fig.patch.set_facecolor("white")

    _ax(ax)

    if not planet.atmosphere.enabled:
        ax.text(
            0.5,
            0.5,
            "No atmosphere",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FL,
            color="#aaaaaa",
            style="italic",
        )
        ax.set_title(planet.name, fontsize=FT, fontweight="bold", pad=3)
        if own_fig:
            return ax.get_figure()
        return ax

    alts = np.linspace(0, max_altitude_km * 1e3, 400)
    alts_km = alts / 1e3

    if quantity == "density":
        vals = np.array([planet.atmosphere.density_at_altitude(h) for h in alts])
        xlabel = "Density (kg/m³)"
        color = C_DENSITY
    elif quantity == "pressure":
        vals = np.array([planet.atmosphere.pressure_at_altitude(h) / 1e3 for h in alts])
        xlabel = "Pressure (kPa)"
        color = C_PRESSURE
    else:
        vals = np.array([planet.atmosphere.temperature_at_altitude(h) for h in alts])
        xlabel = "Temp (K)"
        color = C_TEMP

    ax.fill_betweenx(alts_km, vals, alpha=0.12, color=color)
    ax.plot(vals, alts_km, color=color, lw=LW)
    ax.set_xlim(left=0, right=vals.max() * 1.08)
    ax.set_ylim(0, max_altitude_km)
    ax.set_xlabel(xlabel, fontsize=FL - 0.5, labelpad=1)
    ax.set_ylabel("Altitude (km)", fontsize=FL, labelpad=2)
    ax.tick_params(labelsize=FK - 0.5)

    H_km = planet.atmosphere.scale_height / 1e3
    for mult in [1, 2, 3]:
        h = H_km * mult
        if 0 < h < max_altitude_km:
            ax.axhline(h, color="#cccccc", lw=0.5, ls=":", zorder=0)

    ax.set_title(planet.name, fontsize=FT, fontweight="bold", pad=3)

    if own_fig:
        return ax.get_figure()
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY 2D
# ═══════════════════════════════════════════════════════════════════════════════


def plot_trajectory_2d(
    planet: Planet, trajectory, ax=None, target_altitude=200_000, color_by="speed"
):
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

    R = planet.radius
    comp = planet.atmosphere.composition.name if planet.atmosphere.enabled else "NONE"
    ter_c = TER_COL.get(
        planet.terrain.terrain_type.name if planet.terrain.enabled else "FLAT", "#bbb"
    )
    atm_c = ATM_COL.get(comp, "#cccccc")

    if planet.atmosphere.enabled:
        ax.add_patch(plt.Circle((0, 0), 1.08, color=atm_c, alpha=0.28, zorder=2))
    ax.add_patch(plt.Circle((0, 0), 1.0, fc=ter_c, ec="#555555", lw=0.6, zorder=3))

    r_tgt = (R + target_altitude) / R
    ax.add_patch(
        plt.Circle(
            (0, 0),
            r_tgt,
            fill=False,
            color=C_TARGET,
            ls="--",
            lw=LW2,
            zorder=4,
            label=f"Target {target_altitude / 1e3:.0f} km",
        )
    )

    xs = np.array([s.x / R for s in trajectory])
    ys = np.array([s.y / R for s in trajectory])

    if color_by == "speed":
        vals = np.array([s.speed for s in trajectory]) / 1e3
        cmap = matplotlib.colormaps["plasma"]
        clabel = "Speed (km/s)"
    else:
        vals = np.array([s.fuel_mass for s in trajectory])
        cmap = matplotlib.colormaps["RdYlGn"]
        clabel = "Fuel (kg)"

    norm = matplotlib.colors.Normalize(vals.min(), vals.max())
    cvals = norm(vals)
    for i in range(len(xs) - 1):
        ax.plot(
            [xs[i], xs[i + 1]],
            [ys[i], ys[i + 1]],
            color=cmap(cvals[i]),
            lw=1.2,
            alpha=0.9,
            zorder=5,
        )

    ax.plot(
        xs[0],
        ys[0],
        "o",
        color=W_GREEN,
        ms=5,
        zorder=10,
        label="Start",
        mec="#333",
        mew=0.4,
    )
    ax.plot(
        xs[-1],
        ys[-1],
        "s",
        color=W_RED,
        ms=5,
        zorder=10,
        label="End",
        mec="#333",
        mew=0.4,
    )

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, shrink=0.65, pad=0.03, aspect=20)
    cb.set_label(clabel, fontsize=FL)
    cb.ax.tick_params(labelsize=FK)

    lim = max(abs(xs).max(), abs(ys).max()) * 1.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x / Rp", fontsize=FL)
    ax.set_ylabel("y / Rp", fontsize=FL)
    ax.tick_params(labelsize=FK)
    ax.set_title(f"{planet.name} — trajectory", fontsize=FT, fontweight="bold", pad=5)
    ax.legend(
        loc="upper left",
        fontsize=FG,
        framealpha=0.92,
        edgecolor="#cccccc",
        fancybox=False,
    )
    ax.grid(True, color="#dddddd", lw=0.5, ls="--")
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════════


def plot_mission_telemetry(
    trajectory, planet: Planet, target_altitude=200_000, figsize=(7.0, 4.5)
):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        hspace=0.46, wspace=0.32, left=0.10, right=0.97, top=0.90, bottom=0.12
    )

    times = np.array([s.time for s in trajectory])
    alts = np.array([s.radius - planet.radius for s in trajectory]) / 1e3
    speeds = np.array([s.speed for s in trajectory]) / 1e3
    fuels = np.array([s.fuel_mass for s in trajectory])
    heats = np.array([s.heat_load for s in trajectory])
    t_min = times / 60.0

    rows = [
        (
            axes[0, 0],
            alts,
            C_ALT,
            "Altitude (km)",
            "(a) Altitude",
            target_altitude / 1e3,
            f"{target_altitude / 1e3:.0f} km",
        ),
        (
            axes[0, 1],
            speeds,
            C_SPEED,
            "Speed (km/s)",
            "(b) Speed",
            planet.circular_orbit_speed(target_altitude) / 1e3,
            "v_circ",
        ),
        (axes[1, 0], fuels, C_FUEL, "Propellant (kg)", "(c) Propellant", None, None),
        (axes[1, 1], heats, C_HEAT, "Heat load (J/m²)", "(d) Aeroheating", None, None),
    ]

    for ax, y, color, ylabel, title, hline, hlabel in rows:
        _ax(ax)
        ax.fill_between(t_min, y, alpha=0.12, color=color)
        ax.plot(t_min, y, color=color, lw=LW)
        if hline is not None:
            ax.axhline(hline, color=C_TARGET, ls="--", lw=LW2, label=hlabel)
            ax.legend(fontsize=FG, framealpha=0.9, edgecolor="#cccccc", fancybox=False)
        ax.set_xlabel("Time (min)", fontsize=FL, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=FL, labelpad=2)
        ax.set_title(title, fontsize=FT, fontweight="bold", pad=3)
        ax.tick_params(labelsize=FK)

    fig.suptitle(
        f"Orbital insertion telemetry — {planet.name}",
        fontsize=FT + 1,
        fontweight="bold",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PLANET COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════


def plot_planet_comparison(planets, figsize=(7.0, 3.2)):
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(wspace=0.50, left=0.08, right=0.97, top=0.83, bottom=0.28)

    names = [p.name for p in planets]
    colors = [WONG[i % len(WONG)] for i in range(len(planets))]

    props = [
        ("Radius (R⊕)", [p.radius / 6.371e6 for p in planets]),
        ("g (m/s²)", [p.surface_gravity for p in planets]),
        ("v_esc (km/s)", [p.escape_velocity / 1e3 for p in planets]),
        (
            "ρ₀ (kg/m³)",
            [
                p.atmosphere.surface_density if p.atmosphere.enabled else 0
                for p in planets
            ],
        ),
    ]

    for ax, (ylabel, vals) in zip(axes, props):
        _ax(ax)
        bars = ax.bar(
            range(len(names)), vals, color=colors, edgecolor="white", lw=0.5, width=0.6
        )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(
            names, rotation=38, ha="right", fontsize=FK, rotation_mode="anchor"
        )
        ax.set_ylabel(ylabel, fontsize=FL, labelpad=2)
        ax.tick_params(axis="y", labelsize=FK)
        ax.grid(True, axis="y", color="#e4e4e4", lw=0.5, ls="--")
        ax.grid(False, axis="x")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.03,
                f"{val:.2g}",
                ha="center",
                va="bottom",
                fontsize=FA,
                color="#333333",
            )

    fig.suptitle("Planet physical properties", fontsize=FT + 1, fontweight="bold")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — Interplanetary transfer visualisations
# ═════════════════════════════════════════════════════════════════════════════


# =============================================================================
# Phase 2 — Interplanetary transfer visualisations
# =============================================================================


def plot_heliocentric_transfer(
    transfer_trajectory,
    departure_radius_m,
    arrival_radius_m,
    tof_s,
    departure_name="Departure",
    arrival_name="Arrival",
    star_name="Sun",
    additional_orbits=None,
    ax=None,
    figsize=(6.0, 6.0),
    show_velocity_arrows=True,
    mu_star=None,
):
    """
    Transfer orbit in the heliocentric ecliptic plane.
    Clean publication style matching the existing science figures.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    AU_m = 1.495978707e11
    if mu_star is None:
        mu_star = 6.674e-11 * 1.989e30

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=figsize,
            gridspec_kw=dict(left=0.10, right=0.87, top=0.92, bottom=0.10),
        )
        fig.patch.set_facecolor("white")
    _ax(ax)
    ax.set_aspect("equal")

    traj = np.asarray(transfer_trajectory)
    x_au = traj[:, 0] / AU_m
    y_au = traj[:, 1] / AU_m
    speeds = np.linalg.norm(traj[:, 3:6], axis=1)

    theta = np.linspace(0, 2 * np.pi, 600)
    r_dep = departure_radius_m / AU_m
    r_arr = arrival_radius_m / AU_m

    if additional_orbits:
        for r_m, name, col in additional_orbits:
            r = r_m / AU_m
            ax.plot(
                r * np.cos(theta),
                r * np.sin(theta),
                color="#BBBBBB",
                lw=0.7,
                ls="--",
                zorder=1,
            )
            ax.text(
                r * 0.707 + 0.03,
                r * 0.707 + 0.03,
                name,
                fontsize=FA - 1,
                color="#999999",
            )

    ax.plot(
        r_dep * np.cos(theta),
        r_dep * np.sin(theta),
        color=W_BLUE,
        lw=0.9,
        ls="-",
        alpha=0.4,
        zorder=2,
    )
    ax.plot(
        r_arr * np.cos(theta),
        r_arr * np.sin(theta),
        color=W_RED,
        lw=0.9,
        ls="-",
        alpha=0.4,
        zorder=2,
    )

    points = np.array([x_au, y_au]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = mcolors.Normalize(vmin=speeds.min(), vmax=speeds.max())
    lc = LineCollection(
        segs, cmap="plasma", norm=norm, linewidth=1.8, zorder=4, alpha=0.95
    )
    lc.set_array(speeds[:-1])
    ax.add_collection(lc)

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, shrink=0.50, pad=0.02, aspect=25)
    cbar.set_label("Speed (m/s)", fontsize=FA)
    cbar.ax.tick_params(labelsize=FA - 1)

    dep_xy = traj[0, :2] / AU_m
    arr_xy = traj[-1, :2] / AU_m
    ax.plot(*dep_xy, "o", color=W_BLUE, ms=7, zorder=6, clip_on=False)
    ax.plot(*arr_xy, "o", color=W_RED, ms=7, zorder=6, clip_on=False)

    off = 0.055
    ax.text(
        dep_xy[0] + off,
        dep_xy[1] + off,
        departure_name,
        fontsize=FA,
        color=W_BLUE,
        fontweight="bold",
    )
    ax.text(
        arr_xy[0] + off,
        arr_xy[1] + off,
        arrival_name,
        fontsize=FA,
        color=W_RED,
        fontweight="bold",
    )

    ax.plot(
        0,
        0,
        "o",
        color=W_ORANGE,
        ms=8,
        zorder=5,
        markeredgecolor=W_BLACK,
        markeredgewidth=0.4,
    )
    ax.text(0.04, 0.04, star_name, fontsize=FA, color=W_ORANGE)

    mid = len(traj) // 2
    mp = traj[mid, :2] / AU_m
    mv = traj[mid, 3:5]
    mv_au = mv / (np.linalg.norm(mv) + 1e-30) * 0.06
    ax.annotate(
        "",
        xy=(mp[0] + mv_au[0], mp[1] + mv_au[1]),
        xytext=mp,
        arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.9, mutation_scale=8),
    )

    if show_velocity_arrows:
        for pos, vel, col in [
            (dep_xy, traj[0, 3:5], W_BLUE),
            (arr_xy, traj[-1, 3:5], W_RED),
        ]:
            scale = r_dep / (np.linalg.norm(vel) + 1e-30) * 0.18
            dv = vel * scale
            ax.annotate(
                "",
                xy=(pos[0] + dv[0], pos[1] + dv[1]),
                xytext=pos,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=7),
            )

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=W_BLUE, lw=1.2, label=f"{departure_name} orbit"),
        Line2D([0], [0], color=W_RED, lw=1.2, label=f"{arrival_name} orbit"),
    ]
    ax.legend(
        handles=handles, fontsize=FG, loc="lower left", frameon=False, handlelength=1.5
    )

    # Axes
    lim = max(r_dep, r_arr) * 1.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x  (AU)", fontsize=FL)
    ax.set_ylabel("y  (AU)", fontsize=FL)
    ax.tick_params(labelsize=FK)
    ax.set_title(
        f"Heliocentric transfer:  {departure_name} → {arrival_name}"
        f"   (ToF = {tof_s / 86400:.0f} d)",
        fontsize=FT,
        fontweight="bold",
        pad=4,
    )

    if standalone:
        return ax.get_figure()
    return ax


def plot_porkchop(
    porkchop_data,
    ax=None,
    figsize=(6.2, 5.2),
    max_c3=30.0,
    max_vinf_arr=8.0,
    quantity="c3",
    contour_levels=None,
    best_window=None,
    agent_choice=None,
    colormap="RdYlGn_r",
    show_tof_contours=True,
):
    """
    Porkchop plot — clean white-background style matching science figures.
    No emoji markers. Denser grid for smooth contours.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    pc = porkchop_data
    dep = pc.departure_days
    arr = pc.arrival_days

    if quantity == "c3":
        data = np.where(pc.valid, pc.c3, np.nan)
        label = "C3  (km$^2$ s$^{-2}$)"
        vmin, vmax = 0, max_c3
    elif quantity == "vinf_arr":
        data = np.where(pc.valid, pc.vinf_arr, np.nan)
        label = r"Arrival $v_\infty$  (km/s)"
        vmin, vmax = 0, max_vinf_arr
    else:
        data = np.where(pc.valid, pc.tof, np.nan)
        label = "Time of flight  (days)"
        vmin = float(np.nanmin(data)) if not np.all(np.isnan(data)) else 0
        vmax = float(np.nanmax(data)) if not np.all(np.isnan(data)) else 500
        colormap = "viridis"

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=figsize,
            gridspec_kw=dict(left=0.12, right=0.88, top=0.91, bottom=0.11),
        )
        fig.patch.set_facecolor("white")
    _ax(ax)

    DEP, ARR = np.meshgrid(dep, arr, indexing="ij")

    if contour_levels is None:
        contour_levels = np.linspace(vmin, vmax, 18)

    cf = ax.contourf(
        DEP, ARR, data, levels=contour_levels, cmap=colormap, extend="max", alpha=0.88
    )
    ax.contour(
        DEP,
        ARR,
        data,
        levels=contour_levels[1::2],
        colors="white",
        linewidths=0.5,
        alpha=0.5,
    )

    if show_tof_contours and quantity != "tof":
        tof_d = np.where(pc.valid, pc.tof, np.nan)
        if not np.all(np.isnan(tof_d)):
            t_min = float(np.nanmin(tof_d))
            t_max = float(np.nanmax(tof_d))
            step = 50 if (t_max - t_min) > 200 else 25
            lvls = np.arange(int(t_min // step) * step + step, t_max, step)
            ct = ax.contour(
                DEP,
                ARR,
                tof_d,
                levels=lvls,
                colors="#333333",
                linewidths=0.6,
                linestyles="--",
                alpha=0.55,
            )
            ax.clabel(ct, inline=True, fontsize=FA - 1.5, fmt="%g d", inline_spacing=2)

    cbar = ax.get_figure().colorbar(cf, ax=ax, shrink=0.88, pad=0.02, aspect=28)
    cbar.set_label(label, fontsize=FL)
    cbar.ax.tick_params(labelsize=FK)

    if best_window is not None:
        ax.plot(
            best_window.departure_day,
            best_window.arrival_day,
            "o",
            ms=7,
            color=W_BLACK,
            markerfacecolor=W_YELLOW,
            markeredgewidth=0.8,
            zorder=10,
        )
        ax.text(
            best_window.departure_day + 8,
            best_window.arrival_day + 8,
            f"C3 = {best_window.c3_km2_s2:.1f}",
            fontsize=FA,
            color=W_BLACK,
        )

    if agent_choice is not None:
        di, ai = agent_choice
        if 0 <= di < len(dep) and 0 <= ai < len(arr):
            ax.plot(
                dep[di],
                arr[ai],
                "D",
                ms=6,
                color=W_ORANGE,
                markeredgecolor=W_BLACK,
                markeredgewidth=0.8,
                zorder=10,
            )

    ax.set_xlabel("Departure day  (from epoch)", fontsize=FL)
    ax.set_ylabel("Arrival day  (from epoch)", fontsize=FL)
    ax.tick_params(labelsize=FK)

    titles = {
        "c3": "Launch energy  (C3)",
        "vinf_arr": "Arrival excess speed",
        "tof": "Time of flight",
    }
    ax.set_title(
        f"{pc.departure_planet} → {pc.arrival_planet}"
        f"   —   {titles.get(quantity, quantity)}",
        fontsize=FT,
        fontweight="bold",
        pad=4,
    )

    if standalone:
        return ax.get_figure()
    return ax


def plot_soi_approach(
    approach_trajectory,
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
    Hyperbolic approach and capture within the planet SOI.

    Recomputes the approach arc internally, clipped to 50x planet radius
    so the gravitational curve is clearly visible regardless of planet size.
    The external approach_trajectory argument is still accepted for API
    compatibility but the display is driven by the internal recomputation.
    """
    import math as _math

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    G_si = 6.674e-11
    mu_planet = G_si * planet.mass
    R_km = planet.radius / 1e3

    # Hyperbola parameters
    r_peri = planet.radius + periapsis_alt_km * 1e3
    a_hyp = -mu_planet / v_inf_m_s**2
    e_hyp = 1.0 + r_peri / abs(a_hyp)
    p_hyp = abs(a_hyp) * (e_hyp**2 - 1)

    r_clip = 50.0 * planet.radius
    nu_clip_cos = max(-1.0, min(1.0, (p_hyp / r_clip - 1.0) / e_hyp))
    nu_clip = _math.acos(nu_clip_cos) if abs(nu_clip_cos) < 1 else 1.2

    nu_vals = np.linspace(-nu_clip, min(nu_clip * 0.6, 0.8), 400)
    xs_km, ys_km, spds = [], [], []
    for nu in nu_vals:
        r_nu = p_hyp / (1 + e_hyp * _math.cos(nu))
        x = r_nu * _math.cos(nu) / 1e3
        y = r_nu * _math.sin(nu) / 1e3
        spd = _math.sqrt(v_inf_m_s**2 + 2 * mu_planet / r_nu)
        xs_km.append(x)
        ys_km.append(y)
        spds.append(spd)

    xs_km = np.array(xs_km)
    ys_km = np.array(ys_km)
    spds = np.array(spds)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=figsize,
            gridspec_kw=dict(left=0.13, right=0.93, top=0.91, bottom=0.11),
        )
        fig.patch.set_facecolor("white")
    _ax(ax)
    ax.set_aspect("equal")

    # Planet body
    pcirc = plt.Circle((0, 0), R_km, color=W_ORANGE, alpha=0.80, zorder=5)
    ax.add_patch(pcirc)
    ax.text(
        0,
        0,
        planet.name,
        ha="center",
        va="center",
        fontsize=FA,
        color="white",
        fontweight="bold",
        zorder=6,
    )

    # Science / capture orbit
    if show_capture_orbit:
        theta = np.linspace(0, 2 * np.pi, 400)
        r_tgt_km = (planet.radius + target_alt_km * 1e3) / 1e3
        ax.plot(
            r_tgt_km * np.cos(theta),
            r_tgt_km * np.sin(theta),
            color=W_GREEN,
            lw=1.0,
            ls=":",
            zorder=3,
            label=f"Orbit  ({target_alt_km:.0f} km)",
        )

    points = np.array([xs_km, ys_km]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = mcolors.Normalize(vmin=spds.min(), vmax=spds.max())
    lc = LineCollection(segs, cmap="plasma", norm=norm, lw=1.8, zorder=4, alpha=0.95)
    lc.set_array(spds[:-1])
    ax.add_collection(lc)

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=22)
    cbar.set_label("Speed  (m/s)", fontsize=FA)
    cbar.ax.tick_params(labelsize=FA - 1)

    peri_km = r_peri / 1e3
    ax.plot(
        peri_km,
        0,
        "v",
        ms=7,
        color=W_GREEN,
        markeredgecolor=W_BLACK,
        markeredgewidth=0.5,
        zorder=8,
    )
    ax.text(
        peri_km + R_km * 0.6,
        R_km * 0.4,
        f"Capture burn\n{periapsis_alt_km:.0f} km alt",
        fontsize=FA - 0.5,
        color=W_GREEN,
        va="bottom",
    )

    # Annotation
    ax.text(
        0.03,
        0.97,
        r"$v_\infty$" + f" = {v_inf_m_s / 1e3:.2f} km/s",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=FA,
        color=W_BLACK,
        bbox=dict(
            boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", lw=0.6, alpha=0.9
        ),
    )

    pad = r_clip / 1e3 * 1.15
    x_lo = xs_km.min() - pad * 0.05
    x_hi = xs_km.max() + pad * 0.15
    y_span = max(abs(ys_km.min()), abs(ys_km.max())) + pad * 0.15
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-y_span, y_span)

    ax.set_xlabel("x  (km)", fontsize=FL)
    ax.set_ylabel("y  (km)", fontsize=FL)
    ax.tick_params(labelsize=FK)
    ax.legend(fontsize=FG, loc="lower right", frameon=False)
    ax.set_title(
        f"SOI approach:  {planet.name}   (v$_\\infty$ = {v_inf_m_s / 1e3:.2f} km/s)",
        fontsize=FT,
        fontweight="bold",
        pad=4,
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
    4-panel mission dashboard. White background, consistent style.
    (a) Heliocentric transfer orbit
    (b) Porkchop C3
    (c) Porkchop arrival v_inf
    (d) Mission ΔV breakdown — stacked bar + capture ΔV vs v∞ curve
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(12.0, 9.5))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.40,
        wspace=0.32,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.07,
    )

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

    ax_b = fig.add_subplot(gs[0, 1])
    plot_porkchop(
        porkchop_data,
        ax=ax_b,
        quantity="c3",
        best_window=best_window,
        agent_choice=agent_choice,
        max_c3=25,
    )

    ax_c = fig.add_subplot(gs[1, 0])
    plot_porkchop(
        porkchop_data,
        ax=ax_c,
        quantity="vinf_arr",
        show_tof_contours=True,
        max_vinf_arr=7,
    )

    G_si = 6.674e-11
    try:
        from exorl.core.soi import HyperbolicArrival, HyperbolicDeparture

        mu_dep = G_si * departure_planet.mass
        mu_arr = G_si * arrival_planet.mass
        r_park_dep = departure_planet.radius + 300_000
        r_park_arr = arrival_planet.radius + 300_000

        vinf_dep = best_window.vinf_dep_m_s if best_window else v_inf_dep_m_s
        vinf_arr = best_window.vinf_arr_m_s if best_window else v_inf_arr_m_s

        dep_obj = HyperbolicDeparture(
            vinf_dep, 300_000, departure_planet.radius, mu_dep
        )
        arr_obj = HyperbolicArrival(
            vinf_arr, 300_000, arrival_planet.radius, mu_arr, target_alt_m=300_000
        )
        sk_dv = 50.0

        dv_tli = dep_obj.delta_v_m_s
        dv_cap = arr_obj.dv_capture_m_s
        dv_sk = sk_dv
        dv_total = dv_tli + dv_cap + dv_sk

        # v∞ curve
        vinf_range = np.linspace(1.5, 6.0, 60)
        dv_cap_curve = []
        for v in vinf_range:
            a = HyperbolicArrival(
                v * 1e3, 300_000, arrival_planet.radius, mu_arr, target_alt_m=300_000
            )
            dv_cap_curve.append(a.dv_capture_m_s)
        dv_cap_curve = np.array(dv_cap_curve)

        gs_d = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[1, 1], wspace=0.45, hspace=0
        )
        ax_d1 = fig.add_subplot(gs_d[0, 0])
        ax_d2 = fig.add_subplot(gs_d[0, 1])
        _ax(ax_d1)
        _ax(ax_d2)

        full_labels = ["TLI", "MOI\ncapture", "Station-\nkeeping"]
        values = [dv_tli, dv_cap, dv_sk]
        colors = [W_BLUE, W_RED, W_ORANGE]
        left = 0
        bar_h = 0.45
        for v, c, lbl in zip(values, colors, full_labels):
            ax_d1.barh(
                0,
                v / 1000,
                left=left / 1000,
                height=bar_h,
                color=c,
                alpha=0.85,
                edgecolor="white",
                lw=0.8,
            )
            mid = (left + v / 2) / 1000
            if v / dv_total > 0.05:  # only label if wide enough
                ax_d1.text(
                    mid,
                    0,
                    f"{lbl}\n{v / 1000:.2f}",
                    ha="center",
                    va="center",
                    fontsize=FA - 1.5,
                    color="white",
                    fontweight="bold",
                )
            left += v

        ax_d1.text(
            dv_total / 1000 * 0.5,
            -0.40,
            f"Station-keeping (5 yr): {dv_sk:.0f} m/s",
            ha="center",
            va="center",
            fontsize=FA - 1.5,
            color=W_ORANGE,
        )

        ax_d1.set_xlim(0, dv_total / 1000 * 1.04)
        ax_d1.set_ylim(-0.65, 0.65)
        ax_d1.set_yticks([])
        ax_d1.set_xlabel("ΔV  (km/s)", fontsize=FL)
        ax_d1.tick_params(axis="x", labelsize=FK)
        ax_d1.set_title(
            f"Mission ΔV  —  total {dv_total / 1000:.2f} km/s",
            fontsize=FT,
            fontweight="bold",
            pad=4,
        )

        ax_d2.plot(vinf_range, dv_cap_curve / 1000, color=W_RED, lw=1.6)
        ax_d2.axvline(vinf_arr / 1e3, color=W_ORANGE, lw=1.0, ls="--", alpha=0.8)
        ax_d2.plot(
            vinf_arr / 1e3,
            arr_obj.dv_capture_m_s / 1000,
            "o",
            ms=5,
            color=W_ORANGE,
            markeredgecolor=W_BLACK,
            markeredgewidth=0.5,
        )
        ax_d2.text(
            vinf_arr / 1e3 + 0.12,
            arr_obj.dv_capture_m_s / 1000,
            "Best\nwindow",
            fontsize=FA - 1,
            color=W_ORANGE,
            va="center",
        )
        ax_d2.set_xlabel(r"Arrival $v_\infty$  (km/s)", fontsize=FL)
        ax_d2.set_ylabel("Capture ΔV  (km/s)", fontsize=FL)
        ax_d2.tick_params(labelsize=FK)
        ax_d2.set_title(
            "Capture cost vs arrival speed", fontsize=FT, fontweight="bold", pad=4
        )

    except Exception:
        ax_d = fig.add_subplot(gs[1, 1])
        _ax(ax_d)
        ax_d.axis("off")
        if best_window:
            txt = (
                f"Best window\n"
                f"  Depart day {best_window.departure_day:.0f}\n"
                f"  Arrive day {best_window.arrival_day:.0f}\n"
                f"  C3   {best_window.c3_km2_s2:.2f} km²/s²\n"
                f"  v∞   {best_window.vinf_arr_km_s:.3f} km/s"
            )
            ax_d.text(
                0.08,
                0.88,
                txt,
                transform=ax_d.transAxes,
                va="top",
                fontsize=FL - 0.5,
                family="monospace",
                color=W_BLACK,
            )

    fig.suptitle(
        f"Interplanetary mission:  {departure_planet.name} → {arrival_planet.name}",
        fontsize=FT + 2,
        fontweight="bold",
    )

    save_figure(fig, filename, output_dir)
    return fig


# =============================================================================
# Population statistics plots
# =============================================================================


def plot_mass_radius(
    population,
    ax=None,
    figsize=(6.5, 5.5),
    show_composition_lines=True,
    colour_by="hab_score",
    highlight_solar=True,
):
    """
    Mass-radius diagram with Zeng (2013) composition curves.
    Planets coloured by habitability score (default) or composition.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    from exorl.core.population import composition_radius

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=figsize,
            gridspec_kw=dict(left=0.11, right=0.88, top=0.91, bottom=0.11),
        )
        fig.patch.set_facecolor("white")
    _ax(ax)

    recs = population.records
    masses = np.array([r.mass_earth for r in recs])
    radii = np.array([r.radius_earth for r in recs])

    # Colour mapping
    if colour_by == "hab_score":
        c_vals = np.array([r.hab_score for r in recs])
        cmap = "RdYlGn"
        clabel = "Habitability score"
        vmin, vmax = 0.0, 1.0
    elif colour_by == "density":
        c_vals = np.array([r.density_kg_m3 / 1000 for r in recs])
        cmap = "plasma"
        clabel = "Density (g/cm³)"
        vmin, vmax = 1.0, 12.0
    else:
        c_vals = np.array([r.hab_score for r in recs])
        cmap = "RdYlGn"
        clabel = "Habitability score"
        vmin, vmax = 0.0, 1.0

    sc = ax.scatter(
        masses,
        radii,
        c=c_vals,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=12,
        alpha=0.65,
        linewidths=0,
        zorder=3,
    )
    cbar = ax.get_figure().colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(clabel, fontsize=FA)
    cbar.ax.tick_params(labelsize=FA - 1)

    # Composition curves
    if show_composition_lines:
        m_range = np.logspace(
            np.log10(max(masses.min() * 0.8, 0.05)), np.log10(masses.max() * 1.2), 200
        )
        styles = [
            ("iron", "#AA3333", "-", "Pure iron"),
            ("rocky", "#885522", "--", "Rocky  (Zeng 2013)"),
            ("water50", "#3366AA", "-.", "50% water"),
            ("water", "#2255CC", ":", "Pure water"),
        ]
        for comp, col, ls, lbl in styles:
            r_vals = [composition_radius(m, comp) for m in m_range]
            ax.plot(
                m_range,
                r_vals,
                color=col,
                lw=1.1,
                ls=ls,
                alpha=0.75,
                label=lbl,
                zorder=2,
            )

    # Solar system reference points
    if highlight_solar:
        solar = [
            ("Earth", 1.00, 1.00, W_BLUE),
            ("Venus", 0.815, 0.950, W_ORANGE),
            ("Mars", 0.107, 0.532, W_RED),
            ("Moon", 0.0123, 0.272, "#888888"),
        ]
        for name, m, r, col in solar:
            ax.plot(
                m,
                r,
                "D",
                ms=6,
                color=col,
                markeredgecolor=W_BLACK,
                markeredgewidth=0.5,
                zorder=5,
            )
            ax.text(
                m * 1.08,
                r,
                name,
                fontsize=FA - 0.5,
                color=col,
                va="center",
                fontweight="bold",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mass  (M⊕)", fontsize=FL)
    ax.set_ylabel("Radius  (R⊕)", fontsize=FL)
    ax.tick_params(labelsize=FK)
    if show_composition_lines:
        ax.legend(fontsize=FG, loc="upper left", frameon=False, handlelength=1.5)
    ax.set_title(
        f"Mass-radius diagram  (n = {len(recs)})", fontsize=FT, fontweight="bold", pad=4
    )

    if standalone:
        return ax.get_figure()
    return ax


def plot_habitability_distribution(population, ax=None, figsize=(6.5, 4.5)):
    """
    Habitability score histogram with grade annotations.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=figsize,
            gridspec_kw=dict(left=0.10, right=0.97, top=0.91, bottom=0.12),
        )
        fig.patch.set_facecolor("white")
    _ax(ax)

    scores = np.array([r.hab_score for r in population.records])
    n = len(scores)

    # Histogram
    bins = np.linspace(0, 1, 26)
    counts, edges = np.histogram(scores, bins=bins)

    grade_colors = {
        "A": W_GREEN,
        "B": W_BLUE,
        "C": W_ORANGE,
        "D": W_RED,
        "F": "#888888",
    }
    grade_bounds = {"A": 0.75, "B": 0.60, "C": 0.45, "D": 0.30, "F": 0.0}

    def grade_color(score):
        if score >= 0.75:
            return W_GREEN
        if score >= 0.60:
            return W_BLUE
        if score >= 0.45:
            return W_ORANGE
        if score >= 0.30:
            return W_RED
        return "#888888"

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mid = (lo + hi) / 2
        ax.bar(
            lo,
            counts[i],
            width=(hi - lo) * 0.92,
            align="edge",
            color=grade_color(mid),
            alpha=0.80,
            edgecolor="white",
            lw=0.3,
        )

    # Grade boundary lines
    for grade, thresh in grade_bounds.items():
        if thresh > 0:
            ax.axvline(thresh, color="#444444", lw=0.8, ls="--", alpha=0.6)
            ax.text(
                thresh + 0.01,
                ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 5,
                f"Grade {grade}",
                fontsize=FA - 0.5,
                color="#444444",
                va="top",
            )

    # Stats annotation
    mean_s = scores.mean()
    n_hab = int(np.sum(scores > 0.5))
    ax.axvline(mean_s, color=W_BLACK, lw=1.2, ls="-", alpha=0.7)
    ax.text(
        mean_s + 0.01,
        0.95,
        f"mean = {mean_s:.3f}",
        transform=ax.get_xaxis_transform(),
        fontsize=FA,
        color=W_BLACK,
        va="top",
    )

    ax.set_xlabel("Habitability score", fontsize=FL)
    ax.set_ylabel("Count", fontsize=FL)
    ax.tick_params(labelsize=FK)
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Habitability distribution  (n={n},  {n_hab} score > 0.5  "
        f"[{100 * n_hab / n:.0f}%])",
        fontsize=FT,
        fontweight="bold",
        pad=4,
    )

    if standalone:
        return ax.get_figure()
    return ax


def plot_correlation_heatmap(population, ax=None, figsize=(7.0, 6.0)):
    """
    Pearson correlation heatmap between key planetary properties.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=figsize,
            gridspec_kw=dict(left=0.18, right=0.97, top=0.97, bottom=0.18),
        )
        fig.patch.set_facecolor("white")
    _ax(ax)

    corr, keys = population.correlation_matrix()

    labels = {
        "mass": "Mass",
        "radius": "Radius",
        "density": "Density",
        "gravity": "Gravity",
        "j2": "J₂",
        "b_field": "B-field",
        "heat_flux": "Heat flux",
        "moi": "MoI",
        "P_srf": "Atm pressure",
        "T_surf": "T surface",
        "hab_score": "Habitability",
        "transit_ppm": "Transit depth",
        "rv_K": "RV amplitude",
        "dist_au": "Orb. distance",
    }
    display = [labels.get(k, k) for k in keys]
    n = len(keys)

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.get_figure().colorbar(
        im, ax=ax, shrink=0.75, pad=0.02, label="Pearson r", fraction=0.03
    )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display, rotation=45, ha="right", fontsize=FA - 0.5)
    ax.set_yticklabels(display, fontsize=FA - 0.5)

    # Annotate cells with |r| > 0.4
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            if abs(v) > 0.4 and i != j:
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=FA - 1.5,
                    color="white" if abs(v) > 0.7 else W_BLACK,
                )

    ax.set_title("Property correlation matrix", fontsize=FT, fontweight="bold", pad=4)

    if standalone:
        return ax.get_figure()
    return ax


def plot_population_dashboard(
    population, output_dir=".", filename="population_dashboard"
):
    """
    4-panel population dashboard:
    (a) Mass-radius diagram
    (b) Habitability distribution
    (c) Correlation heatmap
    (d) RL training context — key stats for the agent
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(13.0, 10.0))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.40,
        wspace=0.35,
        left=0.08,
        right=0.97,
        top=0.93,
        bottom=0.08,
    )

    plot_mass_radius(population, ax=fig.add_subplot(gs[0, 0]))
    plot_habitability_distribution(population, ax=fig.add_subplot(gs[0, 1]))
    plot_correlation_heatmap(population, ax=fig.add_subplot(gs[1, 0]))

    ax_d = fig.add_subplot(gs[1, 1])
    _ax(ax_d)
    ax_d.axis("off")

    stats = population.rl_training_stats()
    a = population.arrays()
    lines = [
        f"RL training distribution  (n = {stats['n_total']})",
        "",
        f"Mass range (5th–95th %ile)",
        f"  {stats['mass_p5']:.3f} – {stats['mass_p95']:.2f}  M⊕",
        "",
        f"Radius range (5th–95th %ile)",
        f"  {stats['radius_p5']:.3f} – {stats['radius_p95']:.2f}  R⊕",
        "",
        f"Gravity range (5th–95th %ile)",
        f"  {stats['gravity_p5']:.2f} – {stats['gravity_p95']:.2f}  m/s²",
        "",
        f"Habitability",
        f"  Mean score:  {stats['hab_score_mean']:.3f}",
        f"  Std:         {stats['hab_score_std']:.3f}",
        f"  Score > 0.5: {100 * stats['frac_habitable']:.1f}%",
        f"  In HZ:       {100 * stats['frac_in_hz']:.1f}%",
        "",
        f"Physics diversity",
        f"  Median J2:   {stats['j2_median']:.2e}",
        f"  Median B:    {stats['b_field_median_uT']:.1f} μT",
        f"  Has dynamo:  {100 * stats['frac_with_dynamo']:.1f}%",
        "",
        f"Observation proxy",
        f"  Transit ppm (median): {stats['transit_ppm_median']:.0f}",
    ]

    comp_counts = {}
    for r in population.records:
        comp_counts[r.composition] = comp_counts.get(r.composition, 0) + 1
    lines.append("")
    lines.append("Composition breakdown")
    for comp, cnt in sorted(comp_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {comp:<12s} {100 * cnt / len(population):.1f}%")

    ax_d.text(
        0.04,
        0.97,
        "\n".join(lines),
        transform=ax_d.transAxes,
        va="top",
        fontsize=FA - 0.5,
        family="monospace",
        color=W_BLACK,
        linespacing=1.35,
    )
    ax_d.set_title("RL training context", fontsize=FT, fontweight="bold", pad=4)

    fig.suptitle("Planet population statistics", fontsize=FT + 2, fontweight="bold")
    save_figure(fig, filename, output_dir)
    return fig
