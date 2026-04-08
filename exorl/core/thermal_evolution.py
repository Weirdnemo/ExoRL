"""
thermal_evolution.py — Planetary thermal history and age-dependent habitability.

Integrates the planetary heat equation from formation (t=0) to the current
stellar age, tracking:

  1. Total heat flux Q(t) = Q_radiogenic(t) + Q_secular(t)
  2. Core temperature T_core(t) → when does the outer core solidify?
  3. Dynamo lifetime: [t=0, t_dynamo_off]
  4. Atmospheric pressure history: volcanic outgassing − escape
  5. Surface temperature history: T_surf(t) = T_eq(t) + ΔT_GH(P_atm(t))
  6. Habitability window: time intervals where T_surf ∈ [273,373] K and P > 6 mbar

Physical model
--------------
  Q_radio(t)   = Q_radio_0 × exp(−λ_eff × t)           [W]
  Q_secular(t) = Q_sec_0 × (t / t_ref)^(−0.5)          [W]
  Q_total(t)   = Q_radio(t) + Q_secular(t)              [W]

  Core cooling:
    dT_core/dt = −Q_total / (M_core × c_p)
    Dynamo active while T_core > T_solidus

  Atmosphere:
    dP/dt = f_outgas × (Q_total/A) − f_escape × XUV(t) / (v_esc² P)
    Outgassing proportional to heat flux; escape proportional to XUV
    XUV(t) ∝ t^(−1.23) (Ribas et al. 2005)

References
----------
  Stevenson (1983) — secular cooling scaling
  Schubert, Turcotte & Olson (2001) — planetary interiors
  Ribas et al. (2005) — XUV evolution
  Sleep (2000) — volcanic outgassing scaling
  Lammer et al. (2009) — atmospheric escape

Usage
-----
    from exorl.core.thermal_evolution import ThermalEvolution

    evol = ThermalEvolution(planet, star)
    history = evol.run(dt_myr=10)

    print(f"Dynamo active for {evol.dynamo_lifetime_gyr:.2f} Gyr")
    print(f"Habitable for    {evol.habitable_duration_gyr:.2f} Gyr")
    print(f"Habitable window: {evol.habitable_window_gyr}")

    fig = evol.plot()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
G_SI = 6.674_30e-11  # m³ kg⁻¹ s⁻²
M_EARTH = 5.972e24  # kg
R_EARTH = 6.371e6  # m
AU = 1.495_978_707e11  # m
GYR_S = 3.156e16  # s per Gyr
SIGMA_SB = 5.670_374e-8  # W m⁻² K⁻⁴

# ── Material parameters ───────────────────────────────────────────────────────
C_P_CORE = 840.0  # J kg⁻¹ K⁻¹  specific heat of liquid iron outer core
T_CORE_INIT = 6500.0  # K  primordial core temperature (after magma ocean)
T_CORE_SOLIDUS = 4200.0  # K  temperature at which outer core fully solidifies
# Exponential cooling model: T_core(t) = T0 * exp(-t / tau_core)
# tau_core = TAU_CORE_EARTH * (M/M_E)^0.80 / (R/R_E)^0.50
# Calibrated: Earth tau=17 Gyr (still active), Mars tau=3.9 Gyr (off at ~1 Gyr)
TAU_CORE_EARTH = 17.0  # Gyr  Earth core cooling timescale

# ── Radiogenic decay constants ────────────────────────────────────────────────
LAMBDA_EFF = 0.16  # Gyr⁻¹  effective decay constant (BSE mix of U,Th,K)

# ── Secular cooling ───────────────────────────────────────────────────────────
Q_SEC_EARTH = 49e-3  # W/m²  secular heat flux for Earth today (at 4.5 Gyr)
T_AGE_REF = 4.5  # Gyr

# ── Atmospheric outgassing / escape ───────────────────────────────────────────
# Outgassing rate: proportional to mantle heat flux
# Calibrated: Earth at present → 0.001 Pa/yr net gain (very slow)
OUTGAS_COEFF = 3.0e-4  # Pa m² / W  (Pa gained per W/m² of heat flux per yr)

# Escape rate: XUV-driven hydrodynamic escape
# XUV flux scales as (t/t0)^(-1.23)  (Ribas et al. 2005)
# Escape coefficient calibrated to Mars atmosphere loss (~5 bar → 6 mbar in 4 Gyr)
ESCAPE_COEFF = 1.8e-6  # Pa / (W/m² × Gyr)  per bar of atmospheric pressure
XUV_REF_FLUX = 0.05  # W/m²  XUV at 1 AU, t=100 Myr  (relative to bolometric)
XUV_DECAY = 1.23  # power-law exponent for XUV decay

# Minimum atmospheric pressure (atmospheric floor, can't go below)
P_MIN_PA = 100.0  # Pa  (~1 mbar)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot at one time step
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ThermalSnapshot:
    """State of a planet at one point in its thermal history."""

    age_gyr: float
    heat_flux_Wm2: float  # total surface heat flux [W/m²]
    q_radio: float  # radiogenic component [W/m²]
    q_secular: float  # secular cooling component [W/m²]
    T_core_K: float  # core temperature [K]
    dynamo_active: bool  # is the dynamo operating?
    P_atm_Pa: float  # surface atmospheric pressure [Pa]
    T_surf_K: float  # surface temperature [K]
    T_eq_K: float  # equilibrium temperature [K]
    dT_GH_K: float  # greenhouse warming [K]
    xuv_flux: float  # XUV flux [W/m²]
    habitable: bool  # T_surf in [273,373] and P > 6 mbar


# ─────────────────────────────────────────────────────────────────────────────
# Thermal evolution engine
# ─────────────────────────────────────────────────────────────────────────────


class ThermalEvolution:
    """
    Integrates a planet's thermal history from formation to its current age.

    Parameters
    ----------
    planet      : ExoRL Planet object
    star        : Star object (for equilibrium temperature and XUV history)
    orbital_dist: orbital distance [m] (overrides planet.orbital_distance_m)
    age_gyr     : integration endpoint [Gyr] (default: 4.5 Gyr)
    """

    def __init__(
        self, planet, star=None, orbital_dist: float = None, age_gyr: float = None
    ):
        self.planet = planet
        self.star = star or getattr(planet, "star_context", None)
        self.orbital_dist = (
            orbital_dist or getattr(planet, "orbital_distance_m", None) or AU
        )
        self.age_gyr = (
            age_gyr or getattr(getattr(self.star, None, None), "age", 4.5) or 4.5
        )

        # Planet-derived parameters
        self.R = planet.radius
        self.M = planet.mass
        self.A_surf = 4 * math.pi * self.R**2

        # Core parameters (scaled from Earth)
        self.core_mass_frac = self._estimate_core_frac()
        self.M_core = self.M * self.core_mass_frac
        self.T_core_0 = T_CORE_INIT * min((self.M / M_EARTH) ** 0.03, 1.15)
        # Exponential cooling timescale: tau ∝ M^0.8 / R^0.5
        M_frac = self.M / M_EARTH
        R_frac = self.R / R_EARTH
        self.tau_core_gyr = TAU_CORE_EARTH * (M_frac**0.80) / max(R_frac**0.50, 0.1)

        # Initial atmospheric pressure
        self.P_atm_0 = max(
            float(getattr(planet.atmosphere, "surface_pressure", 101325)), P_MIN_PA
        )

        # History storage
        self.history: list[ThermalSnapshot] = []

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _estimate_core_frac(self) -> float:
        """Iron core mass fraction from bulk density."""
        rho = self.planet.mean_density
        # Empirical fit to interior models (Zeng & Sasselov 2013):
        # rocky Earth-like: ~32%, lower density → smaller core
        rho_rocky = 5514.0
        core_frac = 0.325 * (rho / rho_rocky) ** 2.5
        return max(0.05, min(0.50, core_frac))

    def _radiogenic_flux(self, age_gyr: float) -> float:
        """Radiogenic heat flux [W/m²] at age t."""
        # Present-day radiogenic for this planet (scaled from Earth)
        q_radio_now = (
            49e-3 * (self.M / M_EARTH) ** 0.7 * 0.44
        )  # 44% of total is radiogenic at 4.5 Gyr
        # Scale back in time using decay constant
        dt = self.age_gyr - age_gyr
        return q_radio_now * math.exp(LAMBDA_EFF * dt)

    def _secular_flux(self, age_gyr: float) -> float:
        """Secular cooling heat flux [W/m²] at age t."""
        q_sec_now = (
            Q_SEC_EARTH
            * (self.M / M_EARTH) ** 0.7
            * (T_AGE_REF / max(self.age_gyr, 0.5)) ** 0.5
        )
        if age_gyr < 0.01:
            return q_sec_now * 10.0
        return q_sec_now * (self.age_gyr / age_gyr) ** 0.5

    def _xuv_flux(self, age_gyr: float) -> float:
        """
        XUV flux at the planet [W/m²] at age t.
        Scaled from Ribas et al. (2005) power-law with stellar luminosity.
        """
        L_bol = (
            self.star.luminosity
            if self.star and hasattr(self.star, "luminosity")
            else 3.828e26
        )
        t = max(age_gyr, 0.01)
        # Reference XUV at 1 AU, t=0.1 Gyr
        xuv_1au = XUV_REF_FLUX * (0.1 / t) ** XUV_DECAY
        # Scale to orbital distance
        return xuv_1au * (AU / self.orbital_dist) ** 2 * (L_bol / 3.828e26) ** 0.5

    def _eq_temperature(self, age_gyr: float, bond_albedo: float = 0.3) -> float:
        """Equilibrium temperature [K] at age t."""
        if self.star and hasattr(self.star, "luminosity"):
            L = self.star.luminosity
            # Stellar luminosity was lower in the past: L ∝ t^0.4 at early times
            # (Gough 1981 solar evolution model)
            t_age = getattr(self.star, "age", 4.5) or 4.5
            if t_age > 0 and age_gyr < t_age:
                L_factor = 0.7 + 0.3 * (age_gyr / t_age) ** 0.4
            else:
                L_factor = 1.0
            L_eff = L * L_factor
        else:
            L_eff = 3.828e26
        flux = L_eff / (4 * math.pi * self.orbital_dist**2)
        return (flux * (1 - bond_albedo) / (4 * SIGMA_SB)) ** 0.25

    def _greenhouse_warming(
        self, P_atm_Pa: float, T_eq: float, comp_name: str = "EARTH_LIKE"
    ) -> float:
        """Greenhouse warming [K] for given atmospheric state."""
        try:
            from exorl.core.atmosphere_science import (
                STANDARD_COMPOSITIONS,
                GreenhouseModel,
            )

            comp = STANDARD_COMPOSITIONS.get(comp_name, {"CO2": 0.0004, "N2": 0.78})
            T_surf = GreenhouseModel.surface_temperature(T_eq, comp, P_atm_Pa)
            return T_surf - T_eq
        except Exception:
            # Fallback: simple CO2 forcing
            return 33.0 * math.log(1 + P_atm_Pa / 101325) / math.log(2)

    # ── Integration ────────────────────────────────────────────────────────────

    def run(
        self, dt_myr: float = 10.0, bond_albedo: float = 0.3
    ) -> list[ThermalSnapshot]:
        """
        Integrate thermal history from t=0 to t=age_gyr.

        Parameters
        ----------
        dt_myr      : timestep [Myr]
        bond_albedo : Bond albedo for equilibrium temperature

        Returns
        -------
        List of ThermalSnapshot at each timestep.
        """
        dt_gyr = dt_myr / 1000.0
        ages = np.arange(0.01, self.age_gyr + dt_gyr, dt_gyr)

        # State variables  (T_core computed analytically, P_atm integrated)
        T_core = self.T_core_0  # initial value only, overwritten each step
        P_atm = self.P_atm_0
        comp_nm = getattr(getattr(self.planet, "atmosphere", None), "composition", None)
        comp_name = (
            comp_nm.name if comp_nm and hasattr(comp_nm, "name") else "EARTH_LIKE"
        )

        self.history = []

        for age in ages:
            q_radio = self._radiogenic_flux(age)
            q_sec = self._secular_flux(age)
            q_total = q_radio + q_sec
            xuv = self._xuv_flux(age)

            # Core cooling — exponential model
            # T_core(t) = T0 * exp(-t / tau_core)
            # More physically accurate than Euler integration for this simplified model
            T_core = self.T_core_0 * math.exp(-age / self.tau_core_gyr)
            T_core = max(T_CORE_SOLIDUS * 0.5, T_core)
            # Core thermal check
            core_hot = (T_core > T_CORE_SOLIDUS) and (self.core_mass_frac > 0.03)
            # Rotation check: slow rotators can't sustain a dipole (Rossby number)
            rot_hr = abs(self.planet.rotation_period) / 3600.0
            from exorl.core.interior import DYNAMO_MAX_PERIOD_HR

            rotation_ok = rot_hr < DYNAMO_MAX_PERIOD_HR
            dynamo = core_hot and rotation_ok

            # Atmospheric evolution
            # Outgassing: mantle volcanism proportional to heat flux
            dP_outgas = OUTGAS_COEFF * q_total * dt_gyr * 1000  # Pa per Gyr → per Myr
            # Escape: XUV-driven (depends on pressure: dense atm = more resilient)
            shield = 1.0 + 0.5 * dynamo  # magnetic field reduces escape
            dP_escape = (
                ESCAPE_COEFF * xuv * dt_gyr * 1000 * (P_atm / 101325) / shield
            )  # stronger at higher P
            P_atm = max(P_MIN_PA, P_atm + dP_outgas - dP_escape)

            # Surface temperature
            T_eq = self._eq_temperature(age, bond_albedo)
            dT_GH = self._greenhouse_warming(P_atm, T_eq, comp_name)
            T_surf = T_eq + dT_GH

            # Habitability check
            habitable = (
                273.0 <= T_surf <= 373.0
                and P_atm >= 600.0  # > 6 mbar (water triple point)
            )

            snap = ThermalSnapshot(
                age_gyr=float(age),
                heat_flux_Wm2=float(q_total),
                q_radio=float(q_radio),
                q_secular=float(q_sec),
                T_core_K=float(T_core),
                dynamo_active=bool(dynamo),
                P_atm_Pa=float(P_atm),
                T_surf_K=float(T_surf),
                T_eq_K=float(T_eq),
                dT_GH_K=float(dT_GH),
                xuv_flux=float(xuv),
                habitable=bool(habitable),
            )
            self.history.append(snap)

        return self.history

    # ── Derived quantities ─────────────────────────────────────────────────────

    @property
    def dynamo_lifetime_gyr(self) -> float:
        """How long (Gyr) the dynamo was active."""
        if not self.history:
            return 0.0
        return sum(s.dynamo_active for s in self.history) * (
            self.history[1].age_gyr - self.history[0].age_gyr
            if len(self.history) > 1
            else 0.01
        )

    @property
    def dynamo_turnoff_gyr(self) -> Optional[float]:
        """Age (Gyr) when the dynamo switched off, or None if still active."""
        if not self.history:
            return None
        was_on = self.history[0].dynamo_active
        for snap in self.history[1:]:
            if was_on and not snap.dynamo_active:
                return snap.age_gyr
            was_on = snap.dynamo_active
        return None  # never turned off

    @property
    def habitable_duration_gyr(self) -> float:
        """Total cumulative time (Gyr) the planet was habitable."""
        if not self.history:
            return 0.0
        dt = (
            self.history[1].age_gyr - self.history[0].age_gyr
            if len(self.history) > 1
            else 0.01
        )
        return sum(s.habitable for s in self.history) * dt

    @property
    def habitable_window_gyr(self) -> list[tuple[float, float]]:
        """
        List of (start, end) Gyr intervals when the planet was habitable.
        Returns empty list if never habitable.
        """
        windows = []
        in_window = False
        t_start = 0.0
        for snap in self.history:
            if snap.habitable and not in_window:
                in_window = True
                t_start = snap.age_gyr
            elif not snap.habitable and in_window:
                in_window = False
                windows.append((t_start, snap.age_gyr))
        if in_window:
            windows.append((t_start, self.history[-1].age_gyr))
        return windows

    @property
    def current_snapshot(self) -> Optional[ThermalSnapshot]:
        """The final snapshot (current state)."""
        return self.history[-1] if self.history else None

    def arrays(self) -> dict:
        """Return history as dict of numpy arrays for plotting."""
        if not self.history:
            return {}
        return {
            "age": np.array([s.age_gyr for s in self.history]),
            "heat_flux": np.array([s.heat_flux_Wm2 for s in self.history]) * 1000,
            "q_radio": np.array([s.q_radio for s in self.history]) * 1000,
            "q_secular": np.array([s.q_secular for s in self.history]) * 1000,
            "T_core": np.array([s.T_core_K for s in self.history]),
            "dynamo": np.array([s.dynamo_active for s in self.history]),
            "P_atm_bar": np.array([s.P_atm_Pa for s in self.history]) / 1e5,
            "T_surf": np.array([s.T_surf_K for s in self.history]),
            "T_eq": np.array([s.T_eq_K for s in self.history]),
            "dT_GH": np.array([s.dT_GH_K for s in self.history]),
            "xuv": np.array([s.xuv_flux for s in self.history]),
            "habitable": np.array([s.habitable for s in self.history]),
        }

    def summary(self) -> str:
        if not self.history:
            return "ThermalEvolution: not yet run. Call .run() first."
        cur = self.current_snapshot
        dynoff = self.dynamo_turnoff_gyr
        windows = self.habitable_window_gyr
        lines = [
            f"ThermalEvolution  ({self.planet.name}, {self.age_gyr:.1f} Gyr)",
            f"  Current heat flux : {cur.heat_flux_Wm2 * 1000:.1f} mW/m²",
            f"  Current T_core    : {cur.T_core_K:.0f} K",
            f"  Current T_surf    : {cur.T_surf_K:.1f} K",
            f"  Current P_atm     : {cur.P_atm_Pa / 1e5:.3f} bar",
            f"  Dynamo lifetime   : {self.dynamo_lifetime_gyr:.2f} Gyr",
            f"  Dynamo turnoff    : {f'{dynoff:.2f} Gyr' if dynoff else 'still active'}",
            f"  Habitable total   : {self.habitable_duration_gyr:.2f} Gyr",
            f"  Habitable windows : {[(f'{a:.1f}', f'{b:.1f}') for a, b in windows] or 'none'}",
        ]
        return "\n".join(lines)

    # ── Plotting ───────────────────────────────────────────────────────────────

    def plot(self, figsize=(10.0, 7.0), title: str = None):
        """
        4-panel evolution figure:
          (a) Heat flux (radiogenic + secular)
          (b) Core temperature + dynamo state
          (c) Atmospheric pressure
          (d) Surface temperature + habitability window
        """
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        from visualization.visualizer import (
            FA,
            FK,
            FL,
            FT,
            LW,
            W_BLACK,
            W_BLUE,
            W_GREEN,
            W_ORANGE,
            W_RED,
            _ax,
        )

        arr = self.arrays()
        if not arr:
            raise RuntimeError("Call .run() before .plot()")

        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("white")
        gs = gridspec.GridSpec(
            2,
            2,
            figure=fig,
            hspace=0.42,
            wspace=0.35,
            left=0.09,
            right=0.97,
            top=0.91,
            bottom=0.10,
        )

        t = arr["age"]

        # (a) Heat flux
        ax = fig.add_subplot(gs[0, 0])
        _ax(ax)
        ax.fill_between(t, arr["q_radio"], alpha=0.3, color=W_RED, label="Radiogenic")
        ax.fill_between(
            t,
            arr["q_radio"],
            arr["heat_flux"],
            alpha=0.3,
            color=W_BLUE,
            label="Secular",
        )
        ax.plot(t, arr["heat_flux"], color=W_BLACK, lw=1.2)
        ax.set_xlabel("Age  (Gyr)", fontsize=FL)
        ax.set_ylabel("Heat flux  (mW/m²)", fontsize=FL)
        ax.legend(fontsize=FA - 1, frameon=False)
        ax.set_title("(a) Internal heat flux", fontsize=FT, fontweight="bold", pad=4)
        ax.tick_params(labelsize=FK)

        # (b) Core temperature + dynamo
        ax = fig.add_subplot(gs[0, 1])
        _ax(ax)
        dynamo_on = arr["dynamo"].astype(float)
        ax.plot(t, arr["T_core"] / 1000, color=W_ORANGE, lw=1.4)
        ax.axhline(T_CORE_SOLIDUS / 1000, color="#888888", lw=0.8, ls="--", alpha=0.7)
        ax.text(
            t[1],
            T_CORE_SOLIDUS / 1000 * 1.01,
            "Solidification",
            fontsize=FA - 1.5,
            color="#666666",
        )
        # Shade dynamo-active region
        ax.fill_between(
            t,
            0,
            arr["T_core"] / 1000,
            where=arr["dynamo"],
            alpha=0.12,
            color=W_BLUE,
            label="Dynamo active",
        )
        ax.set_xlabel("Age  (Gyr)", fontsize=FL)
        ax.set_ylabel("Core temperature  (10³ K)", fontsize=FL)
        ax.legend(fontsize=FA - 1, frameon=False)
        ax.set_title(
            "(b) Core temperature & dynamo", fontsize=FT, fontweight="bold", pad=4
        )
        ax.tick_params(labelsize=FK)

        # (c) Atmospheric pressure
        ax = fig.add_subplot(gs[1, 0])
        _ax(ax)
        ax.semilogy(t, arr["P_atm_bar"], color=W_GREEN, lw=1.4)
        ax.axhline(6e-3, color="#888888", lw=0.8, ls="--", alpha=0.7)
        ax.text(
            t[1], 6e-3 * 1.2, "Water triple point", fontsize=FA - 1.5, color="#666666"
        )
        ax.set_xlabel("Age  (Gyr)", fontsize=FL)
        ax.set_ylabel("Surface pressure  (bar)", fontsize=FL)
        ax.set_title("(c) Atmospheric pressure", fontsize=FT, fontweight="bold", pad=4)
        ax.tick_params(labelsize=FK)

        # (d) Surface temperature + habitability
        ax = fig.add_subplot(gs[1, 1])
        _ax(ax)
        ax.plot(
            t,
            arr["T_eq"],
            color=W_ORANGE,
            lw=1.0,
            ls="--",
            alpha=0.7,
            label="T_eq (no GH)",
        )
        ax.plot(t, arr["T_surf"], color=W_BLACK, lw=1.4, label="T_surf")
        ax.fill_between(
            t, 273, 373, alpha=0.08, color=W_GREEN, label="Liquid water range"
        )
        # Shade habitable intervals
        ax.fill_between(
            t,
            t * 0 + 240,
            t * 0 + 390,
            where=arr["habitable"],
            alpha=0.25,
            color=W_GREEN,
        )
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(100, 500)
        ax.axhline(273, color=W_GREEN, lw=0.6, alpha=0.6)
        ax.axhline(373, color=W_RED, lw=0.6, alpha=0.6)
        ax.set_xlabel("Age  (Gyr)", fontsize=FL)
        ax.set_ylabel("Surface temperature  (K)", fontsize=FL)
        ax.legend(fontsize=FA - 1, frameon=False)
        ax.set_title("(d) Surface temperature", fontsize=FT, fontweight="bold", pad=4)
        ax.tick_params(labelsize=FK)

        fig.suptitle(
            title
            or f"Thermal evolution:  {self.planet.name}  "
            f"(dynamo {self.dynamo_lifetime_gyr:.1f} Gyr, "
            f"habitable {self.habitable_duration_gyr:.1f} Gyr)",
            fontsize=FT + 1,
            fontweight="bold",
        )
        return fig
