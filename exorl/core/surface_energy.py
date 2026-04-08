"""
surface_energy.py — Surface energy balance and temperature mapping.

Computes the spatially and temporally resolved surface temperature
distribution on a planet, given:
  - Stellar flux at the planet's orbit
  - Bond albedo (global or spatially varying)
  - Thermal inertia (controls diurnal temperature swing)
  - Obliquity (axial tilt — drives seasons)
  - Atmospheric greenhouse warming

The key physical insight: a planet's surface temperature distribution
is NOT uniform. The day/night contrast, polar vs equatorial temperatures,
and permanent shadow regions at the poles all depend on obliquity, thermal
inertia, and orbital position.

All SI units. Angles in degrees unless labelled _rad.

References:
  Pierrehumbert (2010) — planetary energy balance
  Kieffer & Titus (2001) — Mars thermal inertia
  Vasavada et al. (2012) — Moon thermal model
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

SIGMA_SB = 5.670_374e-8  # Stefan-Boltzmann [W m⁻² K⁻⁴]
AU = 1.495_978_707e11


@dataclass
class InsolationMap:
    """
    Global map of time-averaged stellar flux at the surface [W/m²].
    Shape: (n_lat, n_lon).
    """

    data_W_m2: np.ndarray  # (n_lat, n_lon)
    lat_deg: np.ndarray  # (n_lat,)
    lon_deg: np.ndarray  # (n_lon,)
    description: str = ""

    @property
    def global_mean(self) -> float:
        """Flux averaged over the globe (weighted by cos lat)."""
        weights = np.abs(np.cos(np.radians(self.lat_deg)))
        w2d = weights[:, np.newaxis] * np.ones((1, self.data_W_m2.shape[1]))
        return float(np.average(self.data_W_m2, weights=w2d))

    @property
    def max_flux(self) -> float:
        return float(self.data_W_m2.max())

    @property
    def min_flux(self) -> float:
        return float(self.data_W_m2.min())

    def summary(self) -> str:
        return (
            f"Insolation map {self.data_W_m2.shape}\n"
            f"  Global mean : {self.global_mean:.1f} W/m²\n"
            f"  Max (noon)  : {self.max_flux:.1f} W/m²\n"
            f"  Min (polar) : {self.min_flux:.1f} W/m²\n"
        )


@dataclass
class TemperatureMap:
    """
    Global surface temperature map [K].
    Shape: (n_lat, n_lon).
    """

    data_K: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    description: str = ""

    @property
    def global_mean_K(self) -> float:
        weights = np.abs(np.cos(np.radians(self.lat_deg)))
        w2d = weights[:, np.newaxis] * np.ones((1, self.data_K.shape[1]))
        return float(np.average(self.data_K, weights=w2d))

    @property
    def equatorial_mean_K(self) -> float:
        """Mean temperature within 10° of equator."""
        eq_mask = np.abs(self.lat_deg) <= 10
        return float(self.data_K[eq_mask, :].mean())

    @property
    def polar_mean_K(self) -> float:
        """Mean temperature poleward of 70°."""
        pol_mask = np.abs(self.lat_deg) >= 70
        return float(self.data_K[pol_mask, :].mean())

    @property
    def day_night_contrast_K(self) -> float:
        """Max minus min temperature (proxy for day/night contrast)."""
        return float(self.data_K.max() - self.data_K.min())

    @property
    def habitable_area_fraction(self) -> float:
        """
        Fraction of surface with T between 273 and 373 K
        (liquid water range at 1 atm).
        Weighted by cos(lat) for spherical area.
        """
        weights = np.abs(np.cos(np.radians(self.lat_deg)))
        hab = ((self.data_K >= 273) & (self.data_K <= 373)).astype(float)
        w2d = weights[:, np.newaxis] * np.ones(self.data_K.shape[1])
        return float((hab * w2d).sum() / w2d.sum())

    def summary(self) -> str:
        return (
            f"Temperature map {self.data_K.shape}\n"
            f"  Global mean    : {self.global_mean_K:.1f} K  ({self.global_mean_K - 273.15:.1f} °C)\n"
            f"  Equatorial mean: {self.equatorial_mean_K:.1f} K\n"
            f"  Polar mean     : {self.polar_mean_K:.1f} K\n"
            f"  Day-night range: {self.day_night_contrast_K:.1f} K\n"
            f"  Habitable area : {self.habitable_area_fraction * 100:.1f}%\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Insolation calculation
# ─────────────────────────────────────────────────────────────────────────────


def compute_insolation_map(
    planet,
    stellar_flux_W_m2: float,
    obliquity_deg: float = 0.0,
    orbital_phase: float = 0.5,
    lat_res_deg: float = 5.0,
    lon_res_deg: float = 5.0,
    time_average: bool = True,
) -> InsolationMap:
    """
    Compute the stellar flux at every surface point.

    Parameters
    ----------
    stellar_flux_W_m2 : total stellar flux at the planet's orbit [W/m²]
                        (from star.flux_at_distance(orbital_distance))
    obliquity_deg     : axial tilt [°] — 0 = no seasons, 23.5 = Earth-like
    orbital_phase     : 0=northern solstice, 0.25=equinox, 0.5=southern solstice, 0.75=equinox
    time_average      : if True, average over one full planet rotation (diurnal average)
                        if False, compute instantaneous noon flux

    Returns
    -------
    InsolationMap with stellar_flux [W/m²] at each grid point.
    For time_average=True: this is the daily-mean flux, not noon flux.
    """
    lats = np.arange(-90 + lat_res_deg / 2, 90, lat_res_deg)
    lons = np.arange(-180 + lon_res_deg / 2, 180, lon_res_deg)
    n_lat, n_lon = len(lats), len(lons)

    # Sub-solar latitude (declination) based on orbital phase and obliquity
    # At northern solstice (phase=0): sub-solar lat = +obliquity
    # At equinox: sub-solar lat = 0
    delta_sol = obliquity_deg * math.cos(
        2 * math.pi * orbital_phase
    )  # solar declination [°]
    delta_rad = delta_sol * math.pi / 180

    data = np.zeros((n_lat, n_lon))

    for i, lat in enumerate(lats):
        phi = lat * math.pi / 180

        if time_average:
            # Daily-mean insolation at this latitude (averaged over full rotation)
            # Formula: S_daily = (S/π) × (H × sin(φ)sin(δ) + cos(φ)cos(δ)sin(H))
            # where H is the hour angle at sunrise/sunset
            cos_H0 = -math.tan(phi) * math.tan(delta_rad)
            if cos_H0 < -1:
                # Polar day (continuous sunlight)
                H0 = math.pi
                S_daily = stellar_flux_W_m2 * (math.sin(phi) * math.sin(delta_rad))
            elif cos_H0 > 1:
                # Polar night
                S_daily = 0.0
                H0 = 0.0
            else:
                H0 = math.acos(cos_H0)
                S_daily = (stellar_flux_W_m2 / math.pi) * (
                    H0 * math.sin(phi) * math.sin(delta_rad)
                    + math.cos(phi) * math.cos(delta_rad) * math.sin(H0)
                )
            S_daily = max(0.0, S_daily)
            data[i, :] = S_daily  # longitude-independent for daily average
        else:
            # Instantaneous flux at each longitude
            for j, lon in enumerate(lons):
                # Hour angle (lon relative to sub-solar longitude)
                H = lon * math.pi / 180  # approximate: sub-solar at lon=0
                cos_z = math.sin(phi) * math.sin(delta_rad) + math.cos(phi) * math.cos(
                    delta_rad
                ) * math.cos(H)
                data[i, j] = max(0.0, stellar_flux_W_m2 * cos_z)

    desc = f"{'Daily-mean' if time_average else 'Instantaneous'} insolation, obliquity={obliquity_deg}°"
    return InsolationMap(data_W_m2=data, lat_deg=lats, lon_deg=lons, description=desc)


# ─────────────────────────────────────────────────────────────────────────────
# Surface temperature model
# ─────────────────────────────────────────────────────────────────────────────


def compute_temperature_map(
    insolation: InsolationMap,
    bond_albedo: float = 0.3,
    emissivity: float = 0.95,
    greenhouse_dT_K: float = 0.0,
    thermal_inertia: float = 1000.0,
) -> TemperatureMap:
    """
    Compute surface temperature from insolation and thermal properties.

    For the daily-mean insolation case:
        T = ((S(1-A)) / (ε σ))^0.25 + ΔT_greenhouse

    For time-varying cases, thermal inertia controls the phase lag and
    amplitude of the diurnal temperature cycle.

    Parameters
    ----------
    bond_albedo       : fraction of incoming light reflected [0–1]
    emissivity        : infrared emissivity of the surface [0–1]
    greenhouse_dT_K   : greenhouse warming to add [K]
    thermal_inertia   : J m⁻² K⁻¹ s⁻¹/²
                        Low  (100):  dusty regolith — extreme day/night swings (Moon: ~50)
                        Med  (500):  rocky surface  — moderate swings (Mars: ~200)
                        High (2000): ocean/ice      — small diurnal swing (Earth: ~1000)

    Returns
    -------
    TemperatureMap
    """
    S = insolation.data_W_m2

    # Radiative equilibrium temperature at each point
    absorbed = S * (1 - bond_albedo)

    # For very low thermal inertia: temperature tracks insolation closely
    # For high thermal inertia: temperature converges to the diurnal mean
    T_rad = np.where(
        absorbed > 0,
        (absorbed / (emissivity * SIGMA_SB)) ** 0.25,
        50.0,  # night-time minimum (cosmic background + residual)
    )

    # Thermal inertia smoothing: high inertia → less extreme
    # Reference: Mars TI~200 has ~100 K diurnal range; Earth ocean ~0
    # We apply a blending: T_actual = T_mean + (T_rad - T_mean) × f_TI
    # where f_TI = exp(-TI/TI_ref) — fraction of swing preserved
    TI_ref = 800.0  # J m⁻² K⁻¹ s⁻¹/²  — reference thermal inertia
    f_ti = math.exp(-thermal_inertia / TI_ref)

    # Mean temperature per latitude (for inertia smoothing)
    T_lat_mean = T_rad.mean(axis=1, keepdims=True)
    T_smoothed = T_lat_mean + (T_rad - T_lat_mean) * f_ti

    # Add greenhouse warming
    T_final = T_smoothed + greenhouse_dT_K
    T_final = np.maximum(T_final, 2.7)  # CMB lower bound

    desc = (
        f"Surface temperature: albedo={bond_albedo}, TI={thermal_inertia:.0f}, "
        f"ΔT_GH={greenhouse_dT_K:.1f} K"
    )
    return TemperatureMap(
        data_K=T_final,
        lat_deg=insolation.lat_deg,
        lon_deg=insolation.lon_deg,
        description=desc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Permanent shadow regions
# ─────────────────────────────────────────────────────────────────────────────


def permanent_shadow_latitude_deg(obliquity_deg: float) -> float:
    """
    The poleward latitude beyond which the sun never rises, integrated
    over a full orbital year. These are cold traps for water ice.

    For Earth (23.5°): no permanent shadow at orbital time-scales.
    For Moon (1.5°): shadow inside craters within ~87-90° lat.
    For Uranus (97.9°): extreme — equatorial regions have permanent shadow.

    Returns the poleward latitude [°] of the permanent shadow boundary.
    At this latitude and beyond, solar flux = 0 averaged over the year.

    Approximate: shadow_lat ≈ 90 - obliquity  (for obliquity < 90°)
    """
    if obliquity_deg <= 0:
        return 90.0  # no tilt — no polar shadow
    if obliquity_deg >= 90:
        # Retrograde or extreme — shadow at low latitudes
        return max(0.0, 180.0 - obliquity_deg - 90.0)
    return 90.0 - obliquity_deg


def has_permanent_polar_ice(
    planet,
    star,
    orbital_distance_m: float,
    obliquity_deg: float,
    bond_albedo: float = 0.3,
) -> bool:
    """
    Simple test: are the polar permanent-shadow regions cold enough
    to permanently trap water ice?

    Ice is stable if T < ~120 K (water sublimation rate negligible).
    """
    shadow_lat = permanent_shadow_latitude_deg(obliquity_deg)
    if shadow_lat >= 90:
        return False  # no permanent shadow

    # Temperature at the shadow latitude (no direct sunlight → T from re-radiation)
    # Approximate: T_shadow ≈ T_equilibrium × (fraction_not_illuminated)^0.25
    S_total = star.flux_at_distance(orbital_distance_m)
    T_eq = star.equilibrium_temperature(orbital_distance_m, bond_albedo)
    # Polar shadow gets ~0% direct insolation → warms only by IR from surroundings
    # Rough estimate: T_shadow ≈ 0.3 × T_eq
    T_shadow = 0.3 * T_eq
    return T_shadow < 120.0  # K


# ─────────────────────────────────────────────────────────────────────────────
# Full surface energy balance — convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────


def surface_energy_balance(
    planet,
    star=None,
    orbital_distance_m: float = None,
    obliquity_deg: float = None,
    bond_albedo: float = 0.3,
    thermal_inertia: float = None,
    greenhouse_dT_K: float = None,
    lat_res_deg: float = 5.0,
    lon_res_deg: float = 5.0,
) -> dict:
    """
    Full surface energy balance: insolation → temperature map.

    Resolves parameters from planet attributes where possible.

    Returns
    -------
    dict with:
        insolation_map    : InsolationMap
        temperature_map   : TemperatureMap (annual mean)
        temperature_solstice : TemperatureMap (northern solstice)
        temperature_equinox  : TemperatureMap (equinox)
        permanent_shadow_lat : float [°]
        has_polar_ice     : bool
        global_mean_T_K   : float
        habitable_fraction : float
    """
    # Resolve star and orbit
    if star is None and hasattr(planet, "star_context"):
        star = planet.star_context
    if orbital_distance_m is None and hasattr(planet, "orbital_distance_m"):
        orbital_distance_m = planet.orbital_distance_m
    if star is None or orbital_distance_m is None:
        raise ValueError(
            "Provide star and orbital_distance_m, or set planet.star_context "
            "and planet.orbital_distance_m"
        )

    S = star.flux_at_distance(orbital_distance_m)

    # Obliquity: use provided, or planet rotation period proxy, or 0
    if obliquity_deg is None:
        obliquity_deg = 23.5  # assume Earth-like if unknown

    # Thermal inertia: default based on terrain type
    if thermal_inertia is None:
        if planet.terrain.enabled:
            terrain_TI = {
                "FLAT": 300,
                "CRATERED": 150,
                "MOUNTAINOUS": 400,
                "OCEANIC": 2000,
                "VOLCANIC": 600,
                "RANDOM": 400,
            }
            thermal_inertia = terrain_TI.get(planet.terrain.terrain_type.name, 400)
        else:
            thermal_inertia = 400  # generic rocky

    # Greenhouse warming
    if greenhouse_dT_K is None:
        if planet.atmosphere.enabled:
            from exorl.core.atmosphere_science import (
                STANDARD_COMPOSITIONS,
                GreenhouseModel,
            )

            comp_name = planet.atmosphere.composition.name
            comp = dict(STANDARD_COMPOSITIONS.get(comp_name, {"N2": 1.0}))
            total = sum(comp.values())
            comp = {k: v / total for k, v in comp.items()} if total > 0 else comp
            T_eq = star.equilibrium_temperature(orbital_distance_m, bond_albedo)
            greenhouse_dT_K = GreenhouseModel.total_greenhouse_warming_K(
                comp, planet.atmosphere.surface_pressure, T_eq
            )
        else:
            greenhouse_dT_K = 0.0

    # Annual-mean insolation (orbital phase averaged over 0, 0.25, 0.5, 0.75)
    annual_data = np.zeros((int(180 / lat_res_deg), int(360 / lon_res_deg)))
    for phase in [0.0, 0.25, 0.5, 0.75]:
        ins = compute_insolation_map(
            planet, S, obliquity_deg, phase, lat_res_deg, lon_res_deg, time_average=True
        )
        annual_data += ins.data_W_m2
    annual_data /= 4

    ins_mean = InsolationMap(
        data_W_m2=annual_data,
        lat_deg=ins.lat_deg,
        lon_deg=ins.lon_deg,
        description=f"Annual-mean insolation ({star.name}, {orbital_distance_m / AU:.3f} AU)",
    )

    T_mean = compute_temperature_map(
        ins_mean,
        bond_albedo=bond_albedo,
        emissivity=0.95,
        greenhouse_dT_K=greenhouse_dT_K,
        thermal_inertia=thermal_inertia,
    )

    # Solstice and equinox maps
    ins_sol = compute_insolation_map(
        planet, S, obliquity_deg, 0.0, lat_res_deg, lon_res_deg, True
    )
    T_sol = compute_temperature_map(
        ins_sol, bond_albedo, 0.95, greenhouse_dT_K, thermal_inertia
    )

    ins_eq = compute_insolation_map(
        planet, S, obliquity_deg, 0.25, lat_res_deg, lon_res_deg, True
    )
    T_eq_map = compute_temperature_map(
        ins_eq, bond_albedo, 0.95, greenhouse_dT_K, thermal_inertia
    )

    shadow_lat = permanent_shadow_latitude_deg(obliquity_deg)
    has_ice = has_permanent_polar_ice(
        planet, star, orbital_distance_m, obliquity_deg, bond_albedo
    )

    return {
        "insolation_map": ins_mean,
        "temperature_map": T_mean,
        "temperature_solstice": T_sol,
        "temperature_equinox": T_eq_map,
        "permanent_shadow_lat_deg": shadow_lat,
        "has_polar_ice": has_ice,
        "global_mean_T_K": T_mean.global_mean_K,
        "equatorial_mean_T_K": T_mean.equatorial_mean_K,
        "polar_mean_T_K": T_mean.polar_mean_K,
        "habitable_fraction": T_mean.habitable_area_fraction,
        "stellar_flux_W_m2": S,
        "greenhouse_dT_K": greenhouse_dT_K,
        "obliquity_deg": obliquity_deg,
        "thermal_inertia": thermal_inertia,
    }
