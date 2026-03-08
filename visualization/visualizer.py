"""
visualizer.py — Publication-quality figures for Planet-RL sandbox.

Design targets:
  - Light theme suitable for journals (Nature, AIAA, IEEE, Elsevier)
  - Wong (2011) colorblind-safe 8-colour palette
  - Line weights and font sizes tuned for single-column (88 mm) and
    double-column (180 mm) journal layouts
  - save_figure() helper exports PNG @ 300 dpi and PDF (vector) simultaneously

Reference:
  Wong, B. (2011). Points of view: Color blindness. Nature Methods, 8(6), 441.
"""

from __future__ import annotations
import math
import os
import numpy as np
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from core.planet import Planet
from core.physics import SpacecraftState


# ═══════════════════════════════════════════════════════════════════════════════
# PALETTE & STYLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Wong (2011) colorblind-safe palette — 8 colours, works in grayscale print
WONG = {
    "black":        "#000000",
    "orange":       "#E69F00",
    "sky_blue":     "#56B4E9",
    "green":        "#009E73",
    "yellow":       "#F0E442",
    "blue":         "#0072B2",
    "vermillion":   "#D55E00",
    "pink":         "#CC79A7",
}
WONG_CYCLE = [
    WONG["blue"], WONG["vermillion"], WONG["green"],
    WONG["orange"], WONG["pink"], WONG["sky_blue"],
    WONG["yellow"], WONG["black"],
]

# Semantic colour assignments
C_DENSITY   = WONG["vermillion"]   # orange-red  — density curves
C_PRESSURE  = WONG["blue"]         # deep blue   — pressure curves
C_TEMP      = WONG["green"]        # teal-green  — temperature curves
C_ALT       = WONG["blue"]         # altitude panel
C_SPEED     = WONG["vermillion"]   # speed panel
C_FUEL      = WONG["green"]        # fuel panel
C_HEAT      = WONG["orange"]       # heat panel
C_TARGET    = WONG["black"]        # target reference lines (dashed)
C_FILL      = WONG["sky_blue"]     # fill alpha regions

# Paper layout sizes (inches) — matches common journal column widths
SINGLE_COL  = (3.46, 2.60)   # 88 mm wide  — 1 journal column
DOUBLE_COL  = (7.08, 4.72)   # 180 mm wide — 2 journal columns
FULL_PAGE   = (7.08, 9.00)   # full page figure
SQUARE_SM   = (3.46, 3.46)   # square single-column
SQUARE_LG   = (5.00, 5.00)   # larger square

# Typography — minimum sizes for legibility at column width
FONT_TITLE  = 9
FONT_LABEL  = 8
FONT_TICK   = 7
FONT_LEGEND = 7
FONT_ANNOT  = 6.5
LW_MAIN     = 1.5    # main data lines
LW_REF      = 1.0    # reference / guide lines
LW_THIN     = 0.6    # decorative / secondary
MS_MARKER   = 4      # marker size

# ── Terrain and atmosphere colour maps (muted, print-friendly) ────────────────
ATM_COLORS = {
    "NONE":        ("#d9d9d9", "#bdbdbd"),
    "CO2_THICK":   ("#fdae61", "#f46d43"),
    "CO2_THIN":    ("#fee08b", "#fdae61"),
    "NITROGEN":    ("#abd9e9", "#74add1"),
    "EARTH_LIKE":  ("#74add1", "#4575b4"),
    "HYDROGEN":    ("#ffffbf", "#fee090"),
    "METHANE":     ("#d9d9d9", "#bababa"),
    "CUSTOM":      ("#cccccc", "#999999"),
}

TERRAIN_COLORS = {
    "FLAT":        "#b8cfa0",
    "CRATERED":    "#c4b49a",
    "MOUNTAINOUS": "#a89070",
    "OCEANIC":     "#7eb8d4",
    "VOLCANIC":    "#c08060",
    "RANDOM":      "#aaaaaa",
}


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def apply_journal_style() -> None:
    """
    Apply a journal-ready matplotlib rcParams.
    Call once at the top of your script, or it is called automatically
    by each plot function.
    """
    plt.rcParams.update({
        # Figure
        "figure.facecolor":        "white",
        "figure.dpi":              150,
        # Axes
        "axes.facecolor":          "white",
        "axes.edgecolor":          "#333333",
        "axes.linewidth":          0.8,
        "axes.labelsize":          FONT_LABEL,
        "axes.labelcolor":         "#000000",
        "axes.titlesize":          FONT_TITLE,
        "axes.titleweight":        "bold",
        "axes.titlepad":           5,
        "axes.prop_cycle":         matplotlib.cycler(color=WONG_CYCLE),
        # Grid
        "axes.grid":               True,
        "grid.color":              "#e0e0e0",
        "grid.linewidth":          0.5,
        "grid.linestyle":          "--",
        "grid.alpha":              0.8,
        # Ticks
        "xtick.labelsize":         FONT_TICK,
        "ytick.labelsize":         FONT_TICK,
        "xtick.color":             "#333333",
        "ytick.color":             "#333333",
        "xtick.direction":         "out",
        "ytick.direction":         "out",
        "xtick.major.width":       0.8,
        "ytick.major.width":       0.8,
        "xtick.major.size":        3.0,
        "ytick.major.size":        3.0,
        # Legend
        "legend.fontsize":         FONT_LEGEND,
        "legend.framealpha":       0.92,
        "legend.edgecolor":        "#cccccc",
        "legend.fancybox":         False,
        "legend.borderpad":        0.4,
        "legend.labelspacing":     0.3,
        # Lines
        "lines.linewidth":         LW_MAIN,
        "lines.markersize":        MS_MARKER,
        # Font — Liberation Serif / DejaVu Sans fallback (no external deps)
        "font.family":             "serif",
        "font.size":               FONT_LABEL,
        "mathtext.fontset":        "dejavuserif",
        # Save
        "savefig.dpi":             300,
        "savefig.bbox":            "tight",
        "savefig.facecolor":       "white",
        "pdf.fonttype":            42,   # embed fonts as Type 42 (TrueType) in PDF
        "ps.fonttype":             42,
    })

# Apply on import
apply_journal_style()


def _style_ax(ax: plt.Axes, minor_ticks: bool = True) -> None:
    """Reinforce journal style on a single axes (safe to call after twiny etc.)."""
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(0.8)
    ax.tick_params(which="major", length=3, width=0.8, colors="#333333",
                   labelsize=FONT_TICK)
    if minor_ticks:
        ax.minorticks_on()
        ax.tick_params(which="minor", length=1.5, width=0.5, colors="#555555")
    ax.grid(True, which="major", color="#e0e0e0", lw=0.5, ls="--", alpha=0.8)
    ax.grid(False, which="minor")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str = ".",
    dpi_png: int = 300,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    """
    Save a figure in publication formats.

    Parameters
    ----------
    fig        : matplotlib Figure to save
    filename   : base filename WITHOUT extension  (e.g. "fig1_atmosphere")
    output_dir : destination folder (created if missing)
    dpi_png    : raster DPI — 300 for most journals, 600 for IEEE
    formats    : tuple of formats to write; "pdf" produces vector output

    Returns
    -------
    List of paths written.

    Usage
    -----
    save_figure(fig, "fig2_orbit", output_dir="figures", dpi_png=300)
    # → figures/fig2_orbit.png   (300 dpi raster)
    # → figures/fig2_orbit.pdf   (vector, fonts embedded)
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        dpi = dpi_png if fmt == "png" else None
        path = os.path.join(output_dir, f"{filename}.{fmt}")
        kwargs = dict(bbox_inches="tight", facecolor="white")
        if dpi:
            kwargs["dpi"] = dpi
        fig.savefig(path, format=fmt, **kwargs)
        paths.append(path)
        print(f"  Saved: {path}")
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# PLANET CROSS-SECTION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_planet_cross_section(
    planet: Planet,
    ax: Optional[plt.Axes] = None,
    show_atmosphere_layers: bool = True,
    n_atm_rings: int = 6,
    figsize: tuple = SQUARE_SM,
) -> plt.Axes:
    """
    Schematic cross-section of a planet.
    Suitable for a single journal column.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")

    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.axis("off")

    comp_name = (planet.atmosphere.composition.name
                 if planet.atmosphere.enabled else "NONE")
    atm_outer, atm_inner = ATM_COLORS.get(comp_name, ("#cccccc", "#aaaaaa"))
    terrain_color = TERRAIN_COLORS.get(
        planet.terrain.terrain_type.name if planet.terrain.enabled else "FLAT",
        "#aaaaaa"
    )

    # ── Atmosphere layers ─────────────────────────────────────────────────────
    if planet.atmosphere.enabled and show_atmosphere_layers:
        max_atm_alt = planet.atmosphere.scale_height * 5
        for i in range(n_atm_rings, 0, -1):
            frac = i / n_atm_rings
            ring_r = (planet.radius + max_atm_alt * frac) / planet.radius
            alpha = 0.08 + 0.22 * (1 - frac) ** 1.8
            ax.add_patch(plt.Circle((0, 0), ring_r, color=atm_outer,
                                    alpha=alpha, zorder=2))
        # Limb glow
        ax.add_patch(plt.Circle((0, 0), 1.035, color=atm_outer, alpha=0.55,
                                 fill=False, linewidth=1.8, zorder=3))

    # ── Surface ───────────────────────────────────────────────────────────────
    ax.add_patch(plt.Circle((0, 0), 1.0, color=terrain_color,
                             zorder=5, linewidth=0.8,
                             edgecolor="#555555"))
    # Lighting highlight (top-left)
    ax.add_patch(plt.Circle((-0.22, 0.28), 0.55, color="white",
                             alpha=0.18, zorder=6))

    # ── Core ──────────────────────────────────────────────────────────────────
    core_c = WONG["vermillion"] if planet.mean_density > 5000 else "#999999"
    ax.add_patch(plt.Circle((0, 0), 0.27, color=core_c,
                             zorder=7, alpha=0.85, linewidth=0.5,
                             edgecolor="#333333"))
    ax.add_patch(plt.Circle((0, 0), 0.12, color="white",
                             zorder=8, alpha=0.30))

    # ── Magnetic field lines ──────────────────────────────────────────────────
    if planet.magnetic_field.enabled:
        theta = np.linspace(0, 2 * math.pi, 200)
        for scale, alpha, lw in [(1.65, 0.55, 1.0),
                                  (2.20, 0.30, 0.7),
                                  (2.85, 0.15, 0.5)]:
            ax.plot(scale * np.cos(theta),
                    scale * np.sin(theta) * 0.44,
                    color=WONG["blue"], alpha=alpha, lw=lw, zorder=4)

    # ── Moons ─────────────────────────────────────────────────────────────────
    if planet.moons.enabled and planet.moons.count > 0:
        for i in range(min(planet.moons.count, 3)):
            angle = math.radians(i * 120 + 30)
            mx = 2.35 * math.cos(angle)
            my = 2.35 * math.sin(angle) * 0.50
            ax.add_patch(plt.Circle((mx, my), 0.11, color="#aaaaaa",
                                     zorder=5, linewidth=0.5,
                                     edgecolor="#555555"))

    # ── Labels ────────────────────────────────────────────────────────────────
    ax.text(0, 2.68, planet.name, ha="center", va="center",
            fontsize=FONT_TITLE, fontweight="bold", color="#111111", zorder=10)
    ax.text(0, 2.38,
            f"$R$ = {planet.radius/1e3:,.0f} km"
            f"   $g$ = {planet.surface_gravity:.2f} m s$^{{-2}}$",
            ha="center", va="center", fontsize=FONT_ANNOT,
            color="#333333", zorder=10)

    # Feature annotation strip
    tags = []
    if planet.atmosphere.enabled:
        tags.append(comp_name.replace("_", " ").lower())
    if planet.magnetic_field.enabled:
        tags.append("magnetosphere")
    if planet.oblateness.enabled:
        tags.append(f"$J_2$={planet.oblateness.J2:.1e}")
    if planet.moons.enabled:
        n = planet.moons.count
        tags.append(f"{n} moon{'s' if n != 1 else ''}")
    if tags:
        ax.text(0, -2.58, " · ".join(tags), ha="center", va="center",
                fontsize=FONT_ANNOT, color="#555555",
                style="italic", zorder=10)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.85, 3.05)
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# ATMOSPHERE PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_atmosphere_profile(
    planet: Planet,
    ax: Optional[plt.Axes] = None,
    max_altitude_km: float = 200,
    figsize: tuple = SINGLE_COL,
) -> plt.Axes:
    """
    Altitude vs. density / pressure / temperature.
    Three x-axes share one y-axis — each line colour matches its axis label.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(top=0.78, bottom=0.18)

    _style_ax(ax)

    if not planet.atmosphere.enabled:
        ax.text(0.5, 0.5, "No atmosphere",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_LABEL, color="#777777", style="italic")
        ax.set_title(planet.name, fontsize=FONT_TITLE, fontweight="bold", pad=5)
        return ax

    alts     = np.linspace(0, max_altitude_km * 1e3, 600)
    density  = np.array([planet.atmosphere.density_at_altitude(h)    for h in alts])
    pressure = np.array([planet.atmosphere.pressure_at_altitude(h) / 1e3 for h in alts])
    temp     = np.array([planet.atmosphere.temperature_at_altitude(h) for h in alts])
    alts_km  = alts / 1e3

    # Three x-axes
    ax2 = ax.twiny()
    ax3 = ax.twiny()
    ax3.spines["top"].set_position(("axes", 1.22))

    for a in (ax, ax2, ax3):
        _style_ax(a, minor_ticks=False)
        a.grid(False)

    # Shared horizontal grid on the primary axis only
    ax.grid(True, axis="y", color="#e0e0e0", lw=0.5, ls="--", alpha=0.8)
    ax.grid(True, axis="x", color="#e0e0e0", lw=0.5, ls="--", alpha=0.8)

    # Fill under density for visual anchor
    ax.fill_betweenx(alts_km, density, alpha=0.10, color=C_DENSITY)

    l1, = ax.plot(density,  alts_km, color=C_DENSITY,  lw=LW_MAIN,
                  label="Density")
    l2, = ax2.plot(pressure, alts_km, color=C_PRESSURE, lw=LW_MAIN,
                   label="Pressure", ls=(0, (5, 2)))
    l3, = ax3.plot(temp,     alts_km, color=C_TEMP,     lw=LW_MAIN,
                   label="Temperature", ls=(0, (2, 2)))

    # Axis labels — colour-coded
    ax.set_ylabel("Altitude (km)", fontsize=FONT_LABEL)
    ax.set_xlabel(r"Density (kg m$^{-3}$)",   color=C_DENSITY,  fontsize=FONT_LABEL, labelpad=3)
    ax2.set_xlabel("Pressure (kPa)",           color=C_PRESSURE, fontsize=FONT_LABEL, labelpad=3)
    ax3.set_xlabel("Temperature (K)",          color=C_TEMP,     fontsize=FONT_LABEL, labelpad=3)

    ax.tick_params(axis="x",  colors=C_DENSITY,  labelsize=FONT_TICK)
    ax2.tick_params(axis="x", colors=C_PRESSURE, labelsize=FONT_TICK)
    ax3.tick_params(axis="x", colors=C_TEMP,     labelsize=FONT_TICK)
    ax.tick_params(axis="y",  labelsize=FONT_TICK)

    ax.set_title(f"{planet.name} — Atmosphere Profile",
                 fontsize=FONT_TITLE, fontweight="bold", pad=42)

    # Legend — custom handles to avoid twiny duplication
    handles = [
        Line2D([0], [0], color=C_DENSITY,  lw=LW_MAIN,
               label=r"$\rho$ — Density (kg m$^{-3}$)"),
        Line2D([0], [0], color=C_PRESSURE, lw=LW_MAIN, ls=(0, (5, 2)),
               label="$P$ — Pressure (kPa)"),
        Line2D([0], [0], color=C_TEMP,     lw=LW_MAIN, ls=(0, (2, 2)),
               label="$T$ — Temperature (K)"),
    ]
    ax.legend(handles=handles, loc="upper right",
              fontsize=FONT_LEGEND, framealpha=0.92,
              edgecolor="#cccccc", fancybox=False)

    # Surface conditions footnote
    ax.annotate(
        f"Surface:  $P_0$ = {planet.atmosphere.surface_pressure:.0f} Pa"
        f",  $T_0$ = {planet.atmosphere.surface_temp:.0f} K"
        f",  $H$ = {planet.atmosphere.scale_height/1e3:.1f} km",
        xy=(0.5, -0.20), xycoords="axes fraction",
        ha="center", va="top", fontsize=FONT_ANNOT, color="#555555",
        style="italic",
    )

    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY 2D
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trajectory_2d(
    planet: Planet,
    trajectory: list[SpacecraftState],
    ax: Optional[plt.Axes] = None,
    target_altitude: float = 200_000,
    color_by: str = "speed",
    figsize: tuple = SQUARE_LG,
) -> plt.Axes:
    """
    Top-down (XY plane) trajectory plot.
    Trajectory is coloured by speed or fuel.
    Includes a colourbar for the metric.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")

    ax.set_facecolor("#f7f7f7")   # very light grey — differentiates space from paper
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_edgecolor("#999999")
        spine.set_linewidth(0.8)

    r_plot     = planet.radius
    comp_name  = (planet.atmosphere.composition.name
                  if planet.atmosphere.enabled else "NONE")
    terrain_c  = TERRAIN_COLORS.get(
        planet.terrain.terrain_type.name if planet.terrain.enabled else "FLAT", "#aaa")
    atm_c, _   = ATM_COLORS.get(comp_name, ("#cccccc", "#aaaaaa"))

    # Atmosphere halo
    if planet.atmosphere.enabled:
        ax.add_patch(plt.Circle((0, 0), 1.10, color=atm_c, alpha=0.35, zorder=2))

    # Planet disc
    ax.add_patch(plt.Circle((0, 0), 1.0, color=terrain_c, zorder=3,
                             linewidth=0.8, edgecolor="#555555"))

    # Target orbit
    r_tgt = (planet.radius + target_altitude) / r_plot
    ax.add_patch(plt.Circle((0, 0), r_tgt, fill=False,
                             color=C_TARGET, ls="--", lw=LW_REF, zorder=4,
                             label=f"Target orbit ({target_altitude/1e3:.0f} km)"))

    # Trajectory
    xs = np.array([s.x / r_plot for s in trajectory])
    ys = np.array([s.y / r_plot for s in trajectory])

    if color_by == "speed":
        vals  = np.array([s.speed for s in trajectory])
        cmap  = matplotlib.colormaps["plasma"]
        clabel = r"Speed (km s$^{-1}$)"
        scale  = 1e-3
    elif color_by == "fuel":
        vals  = np.array([s.fuel_mass for s in trajectory])
        cmap  = matplotlib.colormaps["RdYlGn"]
        clabel = "Fuel mass (kg)"
        scale  = 1.0
    else:
        vals  = np.linspace(0, 1, len(xs))
        cmap  = matplotlib.colormaps["viridis"]
        clabel = "Progress"
        scale  = 1.0

    v_plot  = vals * scale
    norm    = matplotlib.colors.Normalize(v_plot.min(), v_plot.max())
    c_vals  = norm(v_plot)

    for i in range(len(xs) - 1):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=cmap(c_vals[i]), lw=1.4, alpha=0.9, zorder=5,
                solid_capstyle="round")

    ax.plot(xs[0],  ys[0],  "o", color=WONG["green"],      ms=MS_MARKER + 1,
            zorder=10, label="Departure", markeredgecolor="#333", markeredgewidth=0.5)
    ax.plot(xs[-1], ys[-1], "s", color=WONG["vermillion"],  ms=MS_MARKER + 1,
            zorder=10, label="Arrival",   markeredgecolor="#333", markeredgewidth=0.5)

    # Colourbar
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, shrink=0.65, pad=0.02, aspect=20)
    cbar.set_label(clabel, fontsize=FONT_LABEL)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    cbar.outline.set_edgecolor("#999999")
    cbar.outline.set_linewidth(0.8)

    lim = max(np.abs(xs).max(), np.abs(ys).max()) * 1.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("$x$ / $R_p$", fontsize=FONT_LABEL)
    ax.set_ylabel("$y$ / $R_p$", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_title(f"{planet.name} — Trajectory (equatorial plane)",
                 fontsize=FONT_TITLE, fontweight="bold", pad=6)
    ax.legend(loc="upper left", fontsize=FONT_LEGEND,
              framealpha=0.92, edgecolor="#cccccc", fancybox=False)
    ax.grid(True, color="#e0e0e0", lw=0.5, ls="--", alpha=0.7)
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION TELEMETRY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mission_telemetry(
    trajectory: list[SpacecraftState],
    planet: Planet,
    target_altitude: float = 200_000,
    figsize: tuple = DOUBLE_COL,
) -> plt.Figure:
    """
    Four-panel telemetry dashboard.
    Sized for a double-column journal figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor("white")

    for ax in axes.flat:
        _style_ax(ax)

    times  = np.array([s.time                       for s in trajectory])
    alts   = np.array([s.radius - planet.radius      for s in trajectory]) / 1e3
    speeds = np.array([s.speed                       for s in trajectory]) / 1e3
    fuels  = np.array([s.fuel_mass                   for s in trajectory])
    heats  = np.array([s.heat_load                   for s in trajectory])
    t_min  = times / 60

    def _panel(ax, x, y, color, ylabel, title,
               hline=None, hline_label=None, fill=True):
        if fill:
            ax.fill_between(x, y, alpha=0.12, color=color)
        ax.plot(x, y, color=color, lw=LW_MAIN)
        if hline is not None:
            ax.axhline(hline, color=C_TARGET, ls="--", lw=LW_REF,
                       label=hline_label)
            ax.legend(fontsize=FONT_LEGEND, framealpha=0.92,
                      edgecolor="#cccccc", fancybox=False)
        ax.set_xlabel("Time (min)", fontsize=FONT_LABEL)
        ax.set_ylabel(ylabel,       fontsize=FONT_LABEL)
        ax.set_title(title,         fontsize=FONT_TITLE, fontweight="bold", pad=4)
        ax.tick_params(labelsize=FONT_TICK)

    _panel(axes[0, 0], t_min, alts,   C_ALT,
           "Altitude (km)",        "(a) Altitude",
           target_altitude / 1e3,  f"{target_altitude/1e3:.0f} km target")

    _panel(axes[0, 1], t_min, speeds, C_SPEED,
           r"Speed (km s$^{-1}$)", "(b) Speed",
           planet.circular_orbit_speed(target_altitude) / 1e3,
           r"$v_\mathrm{circ}$")

    _panel(axes[1, 0], t_min, fuels,  C_FUEL,
           "Propellant mass (kg)", "(c) Propellant remaining")

    _panel(axes[1, 1], t_min, heats,  C_HEAT,
           r"Heat load (J m$^{-2}$)", "(d) Aeroheating",
           fill=False)

    fig.suptitle(f"Orbital insertion telemetry — {planet.name}",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PLANET COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def plot_planet_comparison(
    planets: list[Planet],
    figsize: tuple = DOUBLE_COL,
) -> plt.Figure:
    """
    Horizontal bar chart comparing key physical properties.
    Uses Wong palette cycling for bars.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.patch.set_facecolor("white")

    for ax in axes:
        _style_ax(ax)

    names  = [p.name for p in planets]
    colors = [WONG_CYCLE[i % len(WONG_CYCLE)] for i in range(len(planets))]

    props = [
        ("(a) Radius ($R_\oplus$)",
         [p.radius / 6.371e6 for p in planets]),
        ("(b) Surface $g$ (m s$^{-2}$)",
         [p.surface_gravity for p in planets]),
        (r"(c) $v_\mathrm{esc}$ (km s$^{-1}$)",
         [p.escape_velocity / 1e3 for p in planets]),
        (r"(d) $\rho_\mathrm{atm,0}$ (kg m$^{-3}$)",
         [p.atmosphere.surface_density
          if p.atmosphere.enabled else 0 for p in planets]),
    ]

    for ax, (label, vals) in zip(axes, props):
        bars = ax.barh(names, vals, color=colors,
                       edgecolor="#ffffff", linewidth=0.5, height=0.55)
        ax.set_title(label, fontsize=FONT_LABEL, fontweight="bold", pad=4)
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.tick_params(axis="x", labelsize=FONT_TICK)
        ax.invert_yaxis()
        ax.grid(True, axis="x", color="#e0e0e0", lw=0.5, ls="--", alpha=0.8)
        ax.grid(False, axis="y")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(val + ax.get_xlim()[1] * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2g}", va="center", fontsize=FONT_ANNOT,
                    color="#333333")

    fig.suptitle("Planet physical properties — comparison",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=1.5)
    return fig