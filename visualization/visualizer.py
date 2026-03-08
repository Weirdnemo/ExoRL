"""
visualizer.py — 2D/3D visualization for planets and trajectories.
Uses matplotlib only (no heavy deps).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from core.planet import Planet
from core.physics import SpacecraftState


# ── Atmosphere composition → colour palette ───────────────────────────────────
ATM_COLORS = {
    "NONE":       ("#1a1a2e", "#0f0f1a"),
    "CO2_THICK":  ("#d4a76a", "#c47a3a"),
    "CO2_THIN":   ("#c1754a", "#a0522d"),
    "NITROGEN":   ("#5b9bd5", "#2e6fa3"),
    "EARTH_LIKE": ("#4fa3e0", "#2980b9"),
    "HYDROGEN":   ("#e8d5a3", "#c4a46b"),
    "METHANE":    ("#a87dc2", "#7b5b9e"),
    "CUSTOM":     ("#888888", "#555555"),
}

TERRAIN_COLORS = {
    "FLAT":        "#8b9d6a",
    "CRATERED":    "#9e8e7e",
    "MOUNTAINOUS": "#7a6a5a",
    "OCEANIC":     "#4a7fa5",
    "VOLCANIC":    "#8b4513",
    "RANDOM":      "#888888",
}


def plot_planet_cross_section(
    planet: Planet,
    ax: Optional[plt.Axes] = None,
    show_atmosphere_layers: bool = True,
    n_atm_rings: int = 6,
) -> plt.Axes:
    """Draw a 2D cross-section of the planet with atmosphere rings."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_facecolor("#08080f")

    comp_name = planet.atmosphere.composition.name if planet.atmosphere.enabled else "NONE"
    atm_outer, atm_inner = ATM_COLORS.get(comp_name, ("#888888", "#555555"))
    terrain_color = TERRAIN_COLORS.get(
        planet.terrain.terrain_type.name if planet.terrain.enabled else "FLAT",
        "#888888"
    )

    # ── Atmosphere ────────────────────────────────────────────────────────────
    if planet.atmosphere.enabled and show_atmosphere_layers:
        # Stack translucent rings from outermost inward
        max_atm_alt = planet.atmosphere.scale_height * 5
        for i in range(n_atm_rings, 0, -1):
            frac = i / n_atm_rings
            ring_r = (planet.radius + max_atm_alt * frac) / planet.radius
            alpha = 0.08 + 0.06 * (1 - frac)
            circle = plt.Circle((0, 0), ring_r, color=atm_outer, alpha=alpha)
            ax.add_patch(circle)

    # ── Planet surface ────────────────────────────────────────────────────────
    planet_circle = plt.Circle((0, 0), 1.0, color=terrain_color, zorder=5)
    ax.add_patch(planet_circle)

    # ── Core ──────────────────────────────────────────────────────────────────
    core_r = 0.3
    core_circle = plt.Circle((0, 0), core_r,
                              color="#e85c2a" if planet.mean_density > 5000 else "#888",
                              zorder=6, alpha=0.6)
    ax.add_patch(core_circle)

    # ── Magnetic field lines (simple arcs) ────────────────────────────────────
    if planet.magnetic_field.enabled:
        theta = np.linspace(0, 2 * math.pi, 100)
        for scale in [1.8, 2.5]:
            mx = scale * np.cos(theta)
            my = scale * np.sin(theta) * 0.5
            ax.plot(mx, my, color="#5bc8ff", alpha=0.2, lw=0.8, zorder=4)

    # ── Scale bar ─────────────────────────────────────────────────────────────
    r_km = planet.radius / 1e3
    scale_label = f"R = {r_km:.0f} km  |  g = {planet.surface_gravity:.2f} m/s²"
    ax.set_title(f"{planet.name}\n{scale_label}", color="white", fontsize=11, pad=10)
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    ax.axis("off")
    return ax


def plot_atmosphere_profile(
    planet: Planet,
    ax: Optional[plt.Axes] = None,
    max_altitude_km: float = 200,
) -> plt.Axes:
    """Altitude vs. density / pressure / temperature profiles."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if not planet.atmosphere.enabled:
        ax.text(0.5, 0.5, "No atmosphere", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
        return ax

    alts = np.linspace(0, max_altitude_km * 1e3, 500)
    density  = [planet.atmosphere.density_at_altitude(h) for h in alts]
    pressure = [planet.atmosphere.pressure_at_altitude(h) / 1e3 for h in alts]
    temp     = [planet.atmosphere.temperature_at_altitude(h) for h in alts]
    alts_km  = alts / 1e3

    ax2 = ax.twiny()
    ax3 = ax.twiny()
    ax3.spines["top"].set_position(("axes", 1.15))

    l1, = ax.plot(density, alts_km, color="#e07b3a", lw=2, label="Density (kg/m³)")
    l2, = ax2.plot(pressure, alts_km, color="#4fa3e0", lw=2, label="Pressure (kPa)")
    l3, = ax3.plot(temp, alts_km, color="#7dd67a", lw=2, label="Temperature (K)")

    ax.set_ylabel("Altitude (km)", color="white")
    ax.set_xlabel("Density (kg/m³)", color="#e07b3a")
    ax2.set_xlabel("Pressure (kPa)", color="#4fa3e0")
    ax3.set_xlabel("Temperature (K)", color="#7dd67a")
    ax.set_title(f"{planet.name} – Atmosphere Profile", color="white", pad=30)

    lines = [l1, l2, l3]
    ax.legend(lines, [l.get_label() for l in lines], loc="lower right",
              framealpha=0.3, labelcolor="white")
    return ax


def plot_trajectory_2d(
    planet: Planet,
    trajectory: list[SpacecraftState],
    ax: Optional[plt.Axes] = None,
    target_altitude: float = 200_000,
    color_by: str = "speed",
) -> plt.Axes:
    """Top-down 2D trajectory plot (XY plane)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.set_facecolor("#08080f")
    ax.set_aspect("equal")

    # Planet
    r_plot = planet.radius
    planet_circle = plt.Circle((0, 0), r_plot / r_plot, color="#4a7a5a",
                                zorder=5, label="Planet")
    ax.add_patch(planet_circle)

    # Target orbit circle
    r_target = (planet.radius + target_altitude) / r_plot
    target_circle = plt.Circle((0, 0), r_target, fill=False,
                                color="#ffcc44", ls="--", lw=1, zorder=4,
                                label=f"Target orbit ({target_altitude/1e3:.0f} km)")
    ax.add_patch(target_circle)

    # Trajectory
    xs = np.array([s.x / r_plot for s in trajectory])
    ys = np.array([s.y / r_plot for s in trajectory])

    if color_by == "speed":
        speeds = np.array([s.speed for s in trajectory])
        c_vals = speeds / (speeds.max() + 1e-9)
        cmap = plt.cm.plasma
    elif color_by == "fuel":
        fuels = np.array([s.fuel_mass for s in trajectory])
        c_vals = fuels / (fuels.max() + 1e-9)
        cmap = plt.cm.RdYlGn
    else:
        c_vals = np.linspace(0, 1, len(xs))
        cmap = plt.cm.viridis

    # Scatter with colour
    for i in range(len(xs) - 1):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=cmap(c_vals[i]), lw=1.2, alpha=0.85)

    ax.plot(xs[0], ys[0], "go", ms=8, zorder=10, label="Start")
    ax.plot(xs[-1], ys[-1], "r*", ms=10, zorder=10, label="End")

    lim = max(np.abs(xs).max(), np.abs(ys).max()) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"{planet.name} – Trajectory (XY)", color="white")
    ax.legend(loc="upper right", framealpha=0.3, labelcolor="white")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    return ax


def plot_mission_telemetry(
    trajectory: list[SpacecraftState],
    planet: Planet,
    target_altitude: float = 200_000,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """4-panel telemetry dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor("#0d0d1a")
    for ax in axes.flat:
        ax.set_facecolor("#141428")
        ax.tick_params(colors="#aaa")
        for s in ax.spines.values():
            s.set_edgecolor("#333")

    times = np.array([s.time for s in trajectory])
    alts  = np.array([s.radius - planet.radius for s in trajectory]) / 1e3  # km
    speeds = np.array([s.speed for s in trajectory]) / 1e3   # km/s
    fuels  = np.array([s.fuel_mass for s in trajectory])
    heats  = np.array([s.heat_load for s in trajectory])

    t_min = times / 60

    # Altitude
    axes[0, 0].plot(t_min, alts, color="#4fa3e0", lw=1.5)
    axes[0, 0].axhline(target_altitude / 1e3, color="#ffcc44", ls="--", lw=1, label="Target")
    axes[0, 0].set_ylabel("Altitude (km)", color="#ccc")
    axes[0, 0].set_title("Altitude", color="white")
    axes[0, 0].legend(framealpha=0.3, labelcolor="white")

    # Speed
    v_target = planet.circular_orbit_speed(target_altitude) / 1e3
    axes[0, 1].plot(t_min, speeds, color="#e07b3a", lw=1.5)
    axes[0, 1].axhline(v_target, color="#ffcc44", ls="--", lw=1, label="Target v_circ")
    axes[0, 1].set_ylabel("Speed (km/s)", color="#ccc")
    axes[0, 1].set_title("Speed", color="white")
    axes[0, 1].legend(framealpha=0.3, labelcolor="white")

    # Fuel
    axes[1, 0].fill_between(t_min, fuels, color="#7dd67a", alpha=0.4)
    axes[1, 0].plot(t_min, fuels, color="#7dd67a", lw=1.5)
    axes[1, 0].set_ylabel("Fuel mass (kg)", color="#ccc")
    axes[1, 0].set_xlabel("Time (min)", color="#aaa")
    axes[1, 0].set_title("Fuel Remaining", color="white")

    # Heat
    axes[1, 1].plot(t_min, heats, color="#e85c2a", lw=1.5)
    axes[1, 1].set_ylabel("Heat load (J/m²)", color="#ccc")
    axes[1, 1].set_xlabel("Time (min)", color="#aaa")
    axes[1, 1].set_title("Aeroheating", color="white")

    fig.suptitle(f"Mission Telemetry — {planet.name}", color="white",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_planet_comparison(
    planets: list[Planet],
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Bar chart comparing key properties of multiple planets."""
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.patch.set_facecolor("#0d0d1a")
    for ax in axes:
        ax.set_facecolor("#141428")
        ax.tick_params(colors="#aaa")

    names = [p.name for p in planets]
    colors = plt.cm.tab20(np.linspace(0, 1, len(planets)))

    props = [
        ("Radius (R⊕)", [p.radius / 6.371e6 for p in planets]),
        ("Surface g (m/s²)", [p.surface_gravity for p in planets]),
        ("Escape v (km/s)", [p.escape_velocity / 1e3 for p in planets]),
        ("Atm density (kg/m³)", [
            p.atmosphere.surface_density if p.atmosphere.enabled else 0
            for p in planets
        ]),
    ]

    for ax, (label, vals) in zip(axes, props):
        bars = ax.bar(names, vals, color=colors)
        ax.set_title(label, color="white", fontsize=10)
        ax.set_xticklabels(names, rotation=30, ha="right", color="#ccc", fontsize=8)
        for s in ax.spines.values():
            s.set_edgecolor("#333")

    fig.suptitle("Planet Comparison", color="white", fontsize=13)
    fig.tight_layout()
    return fig
