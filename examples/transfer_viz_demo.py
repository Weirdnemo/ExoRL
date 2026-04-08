"""
transfer_viz_demo.py — visualisation test for interplanetary transfers.

Produces four figures:
  fig11_heliocentric_transfer.png  — Earth-Mars transfer arc in ecliptic plane
  fig12_porkchop_c3.png            — C3 porkchop plot over one synodic period
  fig13_porkchop_vinf.png          — Arrival v∞ porkchop
  fig14_transfer_dashboard.png     — 4-panel mission dashboard

Run from Planet-RL root:  python transfer_viz_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math

import matplotlib
import numpy as np

matplotlib.use("Agg")

from planet_rl.core import (
    AU,
    MU_SUN,
    PRESETS,
    HyperbolicArrival,
    KeplerPropagator,
    LambertSolver,
    LaunchDecisionSpace,
    LaunchWindow,
    PorkchopData,
    SphereOfInfluence,
    laplace_soi_radius,
    planet_state,
    star_sun,
)
from planet_rl.visualization import (
    W_BLUE,
    W_GREEN,
    W_ORANGE,
    W_PINK,
    W_RED,
    apply_journal_style,
    plot_heliocentric_transfer,
    plot_porkchop,
    plot_soi_approach,
    plot_transfer_dashboard,
    save_figure,
)

OUT = "figures/science_figures"
os.makedirs(OUT, exist_ok=True)
apply_journal_style()

sun = star_sun()
earth = PRESETS["earth"]()
mars = PRESETS["mars"]()
venus = PRESETS["venus"]()

earth.orbital_distance_m = 1.000 * AU
mars.orbital_distance_m = 1.524 * AU
venus.orbital_distance_m = 0.723 * AU

G = 6.674e-11

print("=" * 55)
print("Phase 2 — Transfer Visualisation Demo")
print("=" * 55)

# ─────────────────────────────────────────────────────────────────────────────
# Build transfer trajectory
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing Earth→Mars transfer trajectory …")

solver = LambertSolver(MU_SUN)
prop = KeplerPropagator(MU_SUN)
tof_s = 260 * 86400
r1v, v1p = planet_state(AU, 0.0)
r2v, v2p = planet_state(1.524 * AU, tof_s)

v1s, v2s = solver.solve(r1v, r2v, tof_s)
if v1s is None:
    r2v, v2p = planet_state(1.524 * AU, tof_s, initial_phase_rad=0.01)
    v1s, v2s = solver.solve(r1v, r2v, tof_s)

vinf_dep = float(np.linalg.norm(v1s - v1p))
vinf_arr = float(np.linalg.norm(v2s - v2p))
print(f"  v∞ departure: {vinf_dep / 1e3:.3f} km/s")
print(f"  v∞ arrival:   {vinf_arr / 1e3:.3f} km/s")
print(f"  C3:           {(vinf_dep / 1e3) ** 2:.2f} km²/s²")

times_s = np.linspace(0, tof_s, 300)
transfer_traj = prop.orbit_at_time(r1v, v1s, times_s)

# ─────────────────────────────────────────────────────────────────────────────
# Build porkchop grid (30×30, one synodic period)
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing 30×30 porkchop grid (780-day window) …")

dep_days = np.linspace(0, 780, 30)
arr_days = np.linspace(150, 980, 30)

pc = PorkchopData.compute(
    AU,
    1.524 * AU,
    dep_days,
    arr_days,
    dep_name="Earth",
    arr_name="Mars",
    min_tof_days=60,
)
print(pc.summary())

best_win = pc.best_window(max_c3=25, max_vinf_arr=6, min_tof_days=100)
if best_win:
    print(
        f"  Best window: dep={best_win.departure_day:.0f}d  "
        f"arr={best_win.arrival_day:.0f}d  "
        f"C3={best_win.c3_km2_s2:.2f}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Build SOI approach trajectory (planet-centred hyperbolic approach)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding Mars SOI approach trajectory …")

mu_mars = G * mars.mass
soi_mars_m = laplace_soi_radius(mars.mass, 1.524 * AU)

v_inf_mars = best_win.vinf_arr_m_s if best_win else vinf_arr
r_peri = mars.radius + 300_000

# Hyperbola geometry
a_hyp = -mu_mars / v_inf_mars**2
e_hyp = 1.0 + r_peri / abs(a_hyp)
p_hyp = abs(a_hyp) * (e_hyp**2 - 1)

nu_soi_cos = (p_hyp / soi_mars_m - 1.0) / e_hyp
nu_soi_cos = max(-1.0, min(1.0, nu_soi_cos))
nu_soi = -math.acos(nu_soi_cos)
nu_arr = 0.5

nu_vals = np.linspace(nu_soi, nu_arr, 400)

approach_positions = []
approach_velocities = []
for nu in nu_vals:
    r_nu = p_hyp / (1 + e_hyp * math.cos(nu))
    if r_nu > soi_mars_m * 1.05:
        continue
    x = r_nu * math.cos(nu)
    y = r_nu * math.sin(nu)
    v_mag = math.sqrt(mu_mars * (2 / r_nu - 1 / a_hyp))
    fpa = math.atan2(e_hyp * math.sin(nu), 1 + e_hyp * math.cos(nu))
    theta = nu + math.pi / 2 - fpa
    vx = -v_mag * math.sin(nu + fpa)
    vy = v_mag * math.cos(nu + fpa)
    approach_positions.append([x, y, 0])
    approach_velocities.append([vx, vy, 0])

approach_traj = np.hstack([np.array(approach_positions), np.array(approach_velocities)])
print(f"  Approach trajectory: {len(approach_traj)} points")
print(
    f"  SOI entry distance:  {np.linalg.norm(approach_traj[0, :3]) / 1e6:.0f} Mm  (SOI={soi_mars_m / 1e6:.0f} Mm)"
)
print(
    f"  Periapsis distance:  {np.linalg.norm(approach_traj[np.argmin(np.linalg.norm(approach_traj[:, :3], axis=1))][:3]) / 1e3:.0f} km  (target {r_peri / 1e3:.0f} km)"
)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 — Heliocentric transfer arc
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Heliocentric transfer plot …")

fig11 = plot_heliocentric_transfer(
    transfer_traj,
    AU,
    1.524 * AU,
    tof_s,
    departure_name="Earth",
    arrival_name="Mars",
    star_name="Sun",
    additional_orbits=[
        (0.723 * AU, "Venus", W_ORANGE),
    ],
    show_velocity_arrows=True,
    figsize=(6.5, 6.5),
)
save_figure(fig11, "fig11_heliocentric_transfer", OUT)
import matplotlib.pyplot as plt

plt.close(fig11)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 — Porkchop C3
# ─────────────────────────────────────────────────────────────────────────────
print("[2/4] Porkchop C3 plot …")

space = LaunchDecisionSpace(
    AU,
    1.524 * AU,
    n_dep=12,
    n_arr=12,
    window_duration_days=780,
    min_tof_days=100,
    max_tof_days=450,
)
agent_dep_idx, agent_arr_idx = space.best_action()
agent_cost = space.cost(agent_dep_idx, agent_arr_idx)
print(f"  Agent best action: dep_idx={agent_dep_idx}, arr_idx={agent_arr_idx}")
print(f"  Agent C3={agent_cost['c3']:.2f}  v∞_arr={agent_cost['vinf_arr']:.3f}")

fig12 = plot_porkchop(
    pc,
    quantity="c3",
    best_window=best_win,
    figsize=(6.5, 5.5),
    max_c3=30,
)
save_figure(fig12, "fig12_porkchop_c3", OUT)
plt.close(fig12)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 13 — Porkchop arrival v∞
# ─────────────────────────────────────────────────────────────────────────────
print("[3/4] Porkchop arrival v∞ plot …")

fig13 = plot_porkchop(
    pc,
    quantity="vinf_arr",
    best_window=best_win,
    figsize=(6.5, 5.5),
    max_vinf_arr=7,
    show_tof_contours=True,
    colormap="RdYlGn_r",
)
save_figure(fig13, "fig13_porkchop_vinf", OUT)
plt.close(fig13)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 14 — 4-panel dashboard
# ─────────────────────────────────────────────────────────────────────────────
print("[4/4] Transfer dashboard …")

fig14 = plot_transfer_dashboard(
    departure_planet=earth,
    arrival_planet=mars,
    porkchop_data=pc,
    transfer_trajectory=transfer_traj,
    tof_s=tof_s,
    v_inf_dep_m_s=vinf_dep,
    v_inf_arr_m_s=v_inf_mars,
    approach_trajectory=approach_traj,
    soi_radius_arr_m=soi_mars_m,
    star_name="Sun",
    best_window=best_win,
    additional_orbits=[(0.723 * AU, "Venus", W_ORANGE)],
    output_dir=OUT,
    filename="fig14_transfer_dashboard",
)
plt.close(fig14)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Figures saved to ./science_figures/")
print("=" * 55)
for fn in sorted(os.listdir(OUT)):
    if fn.startswith("fig1") and fn.endswith(".png"):
        kb = os.path.getsize(os.path.join(OUT, fn)) // 1024
        print(f"  {fn:<45s}  {kb:4d} kB")

print("\nKey numbers:")
print(f"  Transfer ToF:     {tof_s / 86400:.0f} days")
print(f"  Departure v∞:     {vinf_dep / 1e3:.3f} km/s")
print(f"  Arrival v∞:       {vinf_arr / 1e3:.3f} km/s")
print(f"  C3:               {(vinf_dep / 1e3) ** 2:.2f} km²/s²")
print(f"  Porkchop valid:   {pc.valid.sum()} / {pc.valid.size} cells")
print(f"  SOI Mars:         {soi_mars_m / 1e6:.0f} Mm")
if best_win:
    print(f"  Best window C3:   {best_win.c3_km2_s2:.2f} km²/s²")
    print(f"  Best v∞ arr:      {best_win.vinf_arr_km_s:.3f} km/s")
print("\nDone.")
