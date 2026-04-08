"""
power.py — Spacecraft solar power and energy budget.

Models the power available to a spacecraft in planetary orbit as a function
of orbital altitude, stellar flux, eclipse fraction, and solar panel parameters.

Physical model
--------------
    P_solar  = η × A × S × cos(θ)         [W]  peak panel output
    P_avg    = P_solar × f_sunlit           [W]  orbit-averaged power
    f_sunlit = 1 - arcsin(R_p / r) / π     [-]  sunlit fraction (cylindrical shadow)
    S        = L_star / (4π d²)            [W/m²] flux at orbital distance

    Battery energy stored during sunlit pass:
    E_bat = (P_solar - P_bus) × t_sunlit    [Wh]
    Discharge during eclipse:
    E_eclipse = P_bus × t_eclipse           [Wh]

    Power deficit: max(0, E_eclipse - E_bat)

References
----------
    Wertz & Larson — Space Mission Engineering (2011)
    Fortescue, Stark & Swinerd — Spacecraft Systems Engineering (2011)

Usage
-----
    from exorl.core.power import PowerModel, SolarPanelConfig

    panel  = SolarPanelConfig(area_m2=10.0, efficiency=0.28)
    power  = PowerModel(planet, star, altitude_m=300_000, panel=panel)

    print(power.solar_flux_W_m2)        # W/m² at this orbit
    print(power.eclipse_fraction)       # fraction of orbit in shadow
    print(power.peak_power_W)           # peak panel output [W]
    print(power.avg_power_W)            # orbit-averaged power [W]
    print(power.power_deficit_W(300))   # deficit if bus needs 300W [W]
    print(power.battery_capacity_Wh(300, dod=0.8))  # battery for 300W bus [Wh]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
SOLAR_CONSTANT = 1361.0  # W/m²  at 1 AU
AU = 1.495_978_707e11  # m
STEFAN_BOLTZMANN = 5.670_374e-8  # W m⁻² K⁻⁴


# ─────────────────────────────────────────────────────────────────────────────
# Solar panel configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SolarPanelConfig:
    """
    Physical parameters of the spacecraft's solar panels.

    Attributes
    ----------
    area_m2         : total panel area [m²]
    efficiency      : solar-to-electrical conversion efficiency (0–1)
    degradation_pct : annual degradation [% per year]
    packing_factor  : fraction of gross area that is active cells (0–1)
    pointing_loss   : cosine loss from non-perfect sun-pointing (0–1)
    """

    area_m2: float = 10.0
    efficiency: float = 0.28  # 28% — current triple-junction GaAs
    degradation_pct: float = 2.5  # % per year
    packing_factor: float = 0.85  # ~85% of panel area is active
    pointing_loss: float = 0.98  # near-perfect pointing

    def effective_efficiency(self, mission_age_yr: float = 0.0) -> float:
        """Efficiency accounting for radiation degradation."""
        deg = (1 - self.degradation_pct / 100) ** mission_age_yr
        return self.efficiency * self.packing_factor * self.pointing_loss * deg

    def effective_area(self) -> float:
        """Active cell area [m²]."""
        return self.area_m2 * self.packing_factor


# ─────────────────────────────────────────────────────────────────────────────
# Power model
# ─────────────────────────────────────────────────────────────────────────────


class PowerModel:
    """
    Spacecraft solar power budget for a given orbit and stellar environment.

    Parameters
    ----------
    planet          : Planet object (provides radius, orbital_distance_m, star)
    altitude_m      : orbital altitude above planet surface [m]
    inclination_deg : orbital inclination [degrees]  (affects eclipse duration)
    panel           : SolarPanelConfig
    mission_age_yr  : mission age for degradation calculation
    star            : Star object (overrides planet.star_context if given)
    orbital_dist_m  : distance from star [m] (overrides planet.orbital_distance_m)
    """

    def __init__(
        self,
        planet,
        altitude_m: float = 300_000,
        inclination_deg: float = 98.0,
        panel: SolarPanelConfig = None,
        mission_age_yr: float = 0.0,
        star=None,
        orbital_dist_m: float = None,
    ):

        self.planet = planet
        self.altitude_m = altitude_m
        self.inclination_deg = inclination_deg
        self.panel = panel or SolarPanelConfig()
        self.mission_age_yr = mission_age_yr

        # Resolve star and orbital distance
        self._star = star or getattr(planet, "star_context", None)
        self._orbital_dist = (
            orbital_dist_m or getattr(planet, "orbital_distance_m", None) or AU
        )

    # ── Geometry ──────────────────────────────────────────────────────────────

    @property
    def orbital_radius_m(self) -> float:
        """Orbital radius from planet centre [m]."""
        return self.planet.radius + self.altitude_m

    @property
    def orbital_period_s(self) -> float:
        """Orbital period [s]."""
        return 2 * math.pi * math.sqrt(self.orbital_radius_m**3 / self.planet.mu)

    @property
    def eclipse_fraction(self) -> float:
        """
        Fraction of each orbit spent in the planet's shadow.

        Uses the standard cylindrical shadow model (Wertz & Larson 2011):
            f_eclipse = arcsin(R_planet / r_orbit) / π

        This gives the worst-case (maximum) eclipse fraction for a circular
        orbit, which occurs for equatorial orbits. It is independent of
        inclination to first order and is the standard conservative estimate
        used in mission power analysis.

        Calibration: Earth 300km orbit → 0.404 (40.4% eclipse per orbit) ✓
        """
        r = self.orbital_radius_m
        R = self.planet.radius
        if r <= R:
            return 1.0
        sin_rho = R / r
        if sin_rho >= 1.0:
            return 1.0
        return math.asin(sin_rho) / math.pi

    @property
    def sunlit_fraction(self) -> float:
        """Fraction of orbit in sunlight."""
        return 1.0 - self.eclipse_fraction

    @property
    def eclipse_duration_s(self) -> float:
        """Eclipse duration per orbit [s]."""
        return self.eclipse_fraction * self.orbital_period_s

    @property
    def sunlit_duration_s(self) -> float:
        """Sunlit duration per orbit [s]."""
        return self.sunlit_fraction * self.orbital_period_s

    # ── Power ─────────────────────────────────────────────────────────────────

    @property
    def solar_flux_W_m2(self) -> float:
        """
        Solar flux at the spacecraft's orbital distance [W/m²].
        Uses attached star's luminosity if available, otherwise scales from
        the solar constant by (1 AU / d)².
        """
        if self._star is not None and hasattr(self._star, "flux_at_distance"):
            return float(self._star.flux_at_distance(self._orbital_dist))
        # Fallback: scale from solar constant
        d_au = self._orbital_dist / AU
        return SOLAR_CONSTANT / (d_au**2)

    @property
    def peak_power_W(self) -> float:
        """Peak panel output (full sun, no eclipse) [W]."""
        eff = self.panel.effective_efficiency(self.mission_age_yr)
        return eff * self.panel.area_m2 * self.solar_flux_W_m2

    @property
    def avg_power_W(self) -> float:
        """Orbit-averaged power output [W]."""
        return self.peak_power_W * self.sunlit_fraction

    def power_deficit_W(self, bus_power_W: float) -> float:
        """
        Power deficit [W] when the spacecraft bus demands bus_power_W.
        Positive means the battery must make up the difference.
        Zero means panels alone can sustain the bus.
        """
        return max(0.0, bus_power_W - self.avg_power_W)

    def battery_capacity_Wh(
        self, bus_power_W: float, depth_of_discharge: float = 0.8
    ) -> float:
        """
        Minimum battery capacity [Wh] needed to sustain bus_power_W through
        the eclipse, given the depth-of-discharge limit.

        During sunlit phase: battery charges from panel surplus.
        During eclipse:      battery discharges to supply the bus.
        """
        # Energy needed during eclipse
        E_eclipse = bus_power_W * self.eclipse_duration_s / 3600  # Wh

        # Energy available from panels during sunlit phase (beyond bus load)
        E_surplus = max(
            0.0, (self.peak_power_W - bus_power_W) * self.sunlit_duration_s / 3600
        )

        # Battery must cover shortfall
        E_deficit = max(0.0, E_eclipse - E_surplus)

        # Account for depth-of-discharge limit
        return E_deficit / max(depth_of_discharge, 0.01)

    def is_power_positive(self, bus_power_W: float) -> bool:
        """True if orbit-averaged power exceeds bus demand."""
        return self.avg_power_W >= bus_power_W

    # ── Thermal ───────────────────────────────────────────────────────────────

    @property
    def equilibrium_temp_K(self) -> float:
        """
        Spacecraft equilibrium temperature [K] assuming grey-body radiation
        with absorptivity α = 0.9 (dark surface) and emissivity ε = 0.85.
        """
        alpha = 0.9
        eps = 0.85
        S = self.solar_flux_W_m2
        # Sphere: absorbs on πR², radiates over 4πR²
        T_eq = (alpha * S / (4 * eps * STEFAN_BOLTZMANN)) ** 0.25
        return T_eq

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, bus_power_W: float = 300.0) -> str:
        lines = [
            f"PowerModel  ({self.planet.name}, alt={self.altitude_m / 1e3:.0f}km, "
            f"inc={self.inclination_deg:.1f}°)",
            f"  Solar flux      : {self.solar_flux_W_m2:.0f} W/m²",
            f"  Eclipse frac    : {self.eclipse_fraction:.3f}  "
            f"({self.eclipse_duration_s / 60:.1f} min / "
            f"{self.orbital_period_s / 60:.1f} min)",
            f"  Peak power      : {self.peak_power_W:.0f} W",
            f"  Avg power       : {self.avg_power_W:.0f} W",
            f"  Bus demand      : {bus_power_W:.0f} W",
            f"  Power positive  : {self.is_power_positive(bus_power_W)}",
            f"  Battery needed  : {self.battery_capacity_Wh(bus_power_W):.0f} Wh",
            f"  Equil. temp     : {self.equilibrium_temp_K:.0f} K",
        ]
        return "\n".join(lines)

    # ── RL observation vector ─────────────────────────────────────────────────

    def obs_vector(self, bus_power_W: float = 300.0) -> "np.ndarray":
        """
        4-element normalised observation vector for RL agents:
          [solar_flux_norm, eclipse_frac, power_margin_norm, battery_norm]
        """
        import numpy as np

        flux_norm = min(self.solar_flux_W_m2 / SOLAR_CONSTANT, 5.0)
        eclipse = self.eclipse_fraction
        margin_norm = min(max(self.avg_power_W - bus_power_W, -500) / 500 + 0.5, 1.0)
        bat_norm = min(self.battery_capacity_Wh(bus_power_W) / 1000, 3.0)
        return np.array([flux_norm, eclipse, margin_norm, bat_norm], dtype=np.float32)
