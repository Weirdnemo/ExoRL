"""
orbital_analysis.py — Orbital mechanics analysis for science mission design.

Answers the questions a mission designer actually asks:
  - What inclination gives a sun-synchronous orbit around this planet?
  - What eccentricity freezes the orbit so it doesn't drift?
  - How long before atmospheric drag decays the orbit?
  - What is the J2-induced precession rate at my science altitude?
  - How much ΔV per year does station-keeping cost?

All SI units unless noted. Angles in radians unless labelled _deg.

References:
  Vallado (2013) "Fundamentals of Astrodynamics and Applications"
  Liu (1974) — frozen orbit condition
  Cutting et al. (1978) — sun-synchronous design
  King-Hele (1987) — drag lifetime estimation
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

# ── Physical constants ────────────────────────────────────────────────────────
G        = 6.674_30e-11   # m³ kg⁻¹ s⁻²
TWO_PI   = 2 * math.pi
DEG      = math.pi / 180  # multiply to convert deg → rad
RAD      = 180 / math.pi  # multiply to convert rad → deg


# ─────────────────────────────────────────────────────────────────────────────
# Orbital elements helpers
# ─────────────────────────────────────────────────────────────────────────────

def semi_major_axis_from_altitude(planet_radius_m: float,
                                   altitude_m: float) -> float:
    """Semi-major axis for a circular orbit at given altitude [m]."""
    return planet_radius_m + altitude_m


def mean_motion(semi_major_axis_m: float, mu: float) -> float:
    """Mean motion n = sqrt(mu/a³) [rad/s]."""
    return math.sqrt(mu / semi_major_axis_m**3)


def orbital_period(semi_major_axis_m: float, mu: float) -> float:
    """Orbital period T = 2π/n [s]."""
    return TWO_PI / mean_motion(semi_major_axis_m, mu)


def circular_speed(semi_major_axis_m: float, mu: float) -> float:
    """Circular orbital speed v = sqrt(mu/a) [m/s]."""
    return math.sqrt(mu / semi_major_axis_m)


# ─────────────────────────────────────────────────────────────────────────────
# J2 perturbations
# ─────────────────────────────────────────────────────────────────────────────

class J2Analysis:
    """
    First-order secular effects of J2 oblateness on orbital elements.

    All rates are secular (averaged over one orbit) — they represent
    the long-term drift, not the short-period oscillations.
    """

    @staticmethod
    def _validate(planet) -> tuple[float, float, float]:
        """Return (mu, J2, R_p) from planet; raise if J2 not available."""
        mu = G * planet.mass
        if hasattr(planet, "derived_J2"):
            J2 = planet.derived_J2()
        elif planet.oblateness.enabled:
            J2 = planet.oblateness.J2
        else:
            J2 = 0.0
        return mu, J2, planet.radius

    @staticmethod
    def nodal_precession_rate(planet,
                               semi_major_axis_m: float,
                               inclination_rad: float,
                               eccentricity: float = 0.0) -> float:
        """
        Rate of change of the right ascension of ascending node (RAAN)
        dΩ/dt [rad/s] due to J2.

        Positive inclination < 90° → Ω decreases (westward precession).
        Retrograde orbit (i > 90°) → Ω increases (eastward).

        Formula (Vallado 4th ed. Eq. 9-41):
            dΩ/dt = -(3/2) n J2 (R_p/p)² cos(i)
        where p = a(1-e²) is the semi-latus rectum.
        """
        mu, J2, R_p = J2Analysis._validate(planet)
        if J2 == 0:
            return 0.0
        n = mean_motion(semi_major_axis_m, mu)
        p = semi_major_axis_m * (1 - eccentricity**2)
        return -(3/2) * n * J2 * (R_p / p)**2 * math.cos(inclination_rad)

    @staticmethod
    def nodal_precession_rate_deg_day(planet,
                                       semi_major_axis_m: float,
                                       inclination_rad: float,
                                       eccentricity: float = 0.0) -> float:
        """dΩ/dt in degrees per day."""
        rate_rad_s = J2Analysis.nodal_precession_rate(
            planet, semi_major_axis_m, inclination_rad, eccentricity
        )
        return rate_rad_s * RAD * 86400  # rad/s → deg/day

    @staticmethod
    def apsidal_precession_rate(planet,
                                 semi_major_axis_m: float,
                                 inclination_rad: float,
                                 eccentricity: float = 0.0) -> float:
        """
        Rate of change of argument of periapsis dω/dt [rad/s] due to J2.

            dω/dt = (3/4) n J2 (R_p/p)² (5cos²i - 1)

        Frozen orbit condition: dω/dt = 0 when cos²i = 1/5 → i ≈ 63.43°
        (the critical inclination — used by Molniya orbits).
        """
        mu, J2, R_p = J2Analysis._validate(planet)
        if J2 == 0:
            return 0.0
        n = mean_motion(semi_major_axis_m, mu)
        p = semi_major_axis_m * (1 - eccentricity**2)
        return (3/4) * n * J2 * (R_p / p)**2 * (5 * math.cos(inclination_rad)**2 - 1)

    @staticmethod
    def critical_inclination_deg() -> float:
        """
        The inclination at which apsidal precession vanishes regardless of J2.
        i_crit = arccos(1/√5) ≈ 63.435° and its supplement 116.565°.
        Used for Molniya-type highly elliptical orbits.
        """
        return math.acos(1 / math.sqrt(5)) * RAD  # ≈ 63.435°

    @staticmethod
    def mean_motion_with_J2(planet,
                             semi_major_axis_m: float,
                             inclination_rad: float,
                             eccentricity: float = 0.0) -> float:
        """
        Effective mean motion including J2 correction [rad/s].
        The actual orbital period differs slightly from the Keplerian value.
        """
        mu, J2, R_p = J2Analysis._validate(planet)
        n0 = mean_motion(semi_major_axis_m, mu)
        if J2 == 0:
            return n0
        p = semi_major_axis_m * (1 - eccentricity**2)
        # First-order correction (Brouwer theory)
        correction = 1 + (3/2) * J2 * (R_p / p)**2 * (
            1 - (3/2) * math.sin(inclination_rad)**2
        ) * math.sqrt(1 - eccentricity**2)
        return n0 * correction

    @staticmethod
    def secular_rates_summary(planet,
                               altitude_km: float,
                               inclination_deg: float,
                               eccentricity: float = 0.0) -> dict:
        """
        Return all J2 secular rates for a given orbit.
        Convenient wrapper for reporting.
        """
        a   = semi_major_axis_from_altitude(planet.radius, altitude_km * 1e3)
        inc = inclination_deg * DEG
        return {
            "semi_major_axis_km":      a / 1e3,
            "altitude_km":             altitude_km,
            "inclination_deg":         inclination_deg,
            "eccentricity":            eccentricity,
            "dOmega_dt_deg_day":       J2Analysis.nodal_precession_rate_deg_day(planet, a, inc, eccentricity),
            "domega_dt_deg_day":       J2Analysis.apsidal_precession_rate(planet, a, inc, eccentricity) * RAD * 86400,
            "orbital_period_min":      orbital_period(a, G * planet.mass) / 60,
            "critical_inclination_deg": J2Analysis.critical_inclination_deg(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sun-synchronous orbit
# ─────────────────────────────────────────────────────────────────────────────

class SunSynchronousOrbit:
    """
    Design a sun-synchronous orbit around any planet.

    A sun-synchronous orbit precesses eastward at exactly the rate the
    planet orbits its star, so every pass over a surface point occurs
    at the same local solar time. Essential for imaging missions.

    The required nodal precession rate is:
        dΩ/dt = (2π / orbital_year)  [rad/s]

    Setting this equal to the J2 nodal precession formula and solving for
    inclination gives the sun-synchronous inclination.
    """

    @staticmethod
    def required_precession_rate(stellar_orbital_period_s: float) -> float:
        """
        Required nodal precession rate for sun-sync [rad/s].
        = 2π / T_year  (positive = eastward)
        """
        return TWO_PI / stellar_orbital_period_s

    @staticmethod
    def sun_sync_inclination(planet,
                              altitude_m: float,
                              stellar_orbital_period_s: float,
                              eccentricity: float = 0.0) -> Optional[float]:
        """
        Required inclination for a sun-synchronous orbit [degrees].

        Solves:
            -(3/2) n J2 (R/p)² cos(i) = 2π / T_year

        for i. Returns None if no solution exists (planet too small or slow).

        Parameters
        ----------
        stellar_orbital_period_s : orbital period of the planet around its star [s]
        """
        mu, J2, R_p = J2Analysis._validate(planet)
        if J2 == 0:
            return None
        a   = planet.radius + altitude_m
        n   = mean_motion(a, mu)
        p   = a * (1 - eccentricity**2)
        req = TWO_PI / stellar_orbital_period_s   # required dΩ/dt (positive)

        # cos(i) = -req / (-(3/2) n J2 (R/p)²)
        #        = req / ((3/2) n J2 (R/p)²)
        denominator = (3/2) * n * J2 * (R_p / p)**2
        if denominator == 0:
            return None
        # Sun-sync needs EASTWARD (positive) precession = +2π/T_year
        # Since dΩ/dt = -(3/2)nJ2(R/p)²cos(i), positive dΩ/dt requires cos(i)<0 → i>90°
        # So: cos(i) = -req / denominator
        cos_i = -req / denominator
        if abs(cos_i) > 1.0:
            return None   # no sun-sync solution at this altitude
        return math.acos(cos_i) * RAD  # degrees (will be > 90 for retrograde)

    @staticmethod
    def sun_sync_altitude_range(planet,
                                 stellar_orbital_period_s: float,
                                 inclination_range_deg: tuple = (95.0, 108.0)
                                 ) -> tuple[Optional[float], Optional[float]]:
        """
        Altitude range [km] over which sun-synchronous orbits exist for a
        given inclination range.

        Solves for altitude at the min and max inclination bounds.
        Returns (alt_min_km, alt_max_km) — smaller inclination → higher altitude.
        """
        mu, J2, R_p = J2Analysis._validate(planet)
        if J2 == 0:
            return None, None

        req   = TWO_PI / stellar_orbital_period_s
        alts  = []
        for inc_deg in inclination_range_deg:
            cos_i = math.cos(inc_deg * DEG)
            if cos_i == 0:
                continue
            # n × (R/a)² = req / (-(3/2) J2 cos_i) — solve for a
            # n = sqrt(mu/a³), so sqrt(mu/a³) × (R/a)² = C → mu^0.5 R² a^(-3.5) = C
            # a^3.5 = mu^0.5 R² / C
            C   = req / (-(3/2) * J2 * cos_i)
            if C <= 0:
                continue
            a35 = math.sqrt(mu) * R_p**2 / C
            a   = a35 ** (1/3.5)
            alt_km = (a - R_p) / 1e3
            if alt_km > 0:
                alts.append(alt_km)
        if len(alts) < 2:
            return (alts[0] if alts else None, None)
        return (min(alts), max(alts))

    @staticmethod
    def local_solar_time_drift(planet,
                                altitude_m: float,
                                actual_inclination_deg: float,
                                stellar_orbital_period_s: float,
                                eccentricity: float = 0.0) -> float:
        """
        Drift of local solar time per day [min/day] if the orbit is not
        exactly sun-synchronous.

        Positive = ascending node drifts later (eastward orbit drifts afternoon).
        Zero = perfectly sun-synchronous.
        """
        dOmega_actual = J2Analysis.nodal_precession_rate(
            planet, planet.radius + altitude_m,
            actual_inclination_deg * DEG, eccentricity
        )
        dOmega_required = TWO_PI / stellar_orbital_period_s
        # Difference in rad/s → min/day
        diff_rad_s = dOmega_actual - dOmega_required
        # 1 rad/day × (1440 min/day) / (2π rad/day)
        return diff_rad_s * 86400 * 1440 / TWO_PI


# ─────────────────────────────────────────────────────────────────────────────
# Frozen orbit
# ─────────────────────────────────────────────────────────────────────────────

class FrozenOrbit:
    """
    A frozen orbit is one where the orbital elements (especially eccentricity
    and argument of periapsis) remain constant on average.

    For J2+J3 perturbations, a frozen orbit satisfies:
        ω = 90° (periapsis over the pole) and
        e = -J3 R_p sin(i) / (2 J2 a (1-e²))

    Practical consequence: the spacecraft always passes over the same latitude
    at the same altitude — critical for radar altimeters, gravity field mapping,
    and repeat-pass interferometry.

    Used by: LRO, MRO, GRACE, ICESat, Envisat, ERS, Sentinel-3.
    """

    @staticmethod
    def frozen_eccentricity(planet,
                             semi_major_axis_m: float,
                             inclination_rad: float) -> float:
        """
        The eccentricity at which apsidal drift vanishes (frozen condition).

            e_frozen = -J3 R_p sin(i) / (2 J2 a (1-e²))

        Solved iteratively (e appears on both sides).

        Returns 0.0 if J2 or J3 are not available.
        """
        mu, J2, R_p = J2Analysis._validate(planet)
        if J2 == 0:
            return 0.0

        # Get J3
        if planet.oblateness.enabled:
            J3 = planet.oblateness.J3
        else:
            J3 = -J2 * 0.002  # typical J3/J2 ratio for rocky planets

        a   = semi_major_axis_m
        sin_i = math.sin(inclination_rad)

        # Iterative solution: e = f(e)
        e = 0.001   # initial guess
        for _ in range(50):
            e_new = -J3 * R_p * sin_i / (2 * J2 * a * (1 - e**2))
            e_new = max(0.0, min(e_new, 0.2))   # physical clamp
            if abs(e_new - e) < 1e-10:
                break
            e = 0.5 * e + 0.5 * e_new
        return max(0.0, e)

    @staticmethod
    def frozen_orbit_params(planet,
                             altitude_km: float,
                             inclination_deg: float) -> dict:
        """
        Full frozen orbit design for a given altitude and inclination.
        Returns a dict with all design parameters.
        """
        mu  = G * planet.mass
        a   = semi_major_axis_from_altitude(planet.radius, altitude_km * 1e3)
        inc = inclination_deg * DEG
        e_f = FrozenOrbit.frozen_eccentricity(planet, a, inc)

        # Periapsis and apoapsis altitudes
        r_peri = a * (1 - e_f) - planet.radius
        r_apo  = a * (1 + e_f) - planet.radius

        return {
            "altitude_km":       altitude_km,
            "inclination_deg":   inclination_deg,
            "frozen_ecc":        e_f,
            "frozen_omega_deg":  90.0,   # always 90° for frozen condition
            "periapsis_alt_km":  r_peri / 1e3,
            "apoapsis_alt_km":   r_apo  / 1e3,
            "alt_variation_km":  (r_apo - r_peri) / 1e3,
            "orbital_period_min": orbital_period(a, mu) / 60,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Atmospheric drag lifetime
# ─────────────────────────────────────────────────────────────────────────────

class DragLifetime:
    """
    Estimate how long a spacecraft orbit lasts before atmospheric drag
    causes it to re-enter.

    Uses the King-Hele (1987) analytic approximation which is accurate
    to ~20% for circular orbits and moderate eccentricities.
    """

    @staticmethod
    def lifetime_years(planet,
                        altitude_m: float,
                        spacecraft_mass_kg: float,
                        ballistic_coeff_kg_m2: float,
                        eccentricity: float = 0.0) -> float:
        """
        Estimated drag lifetime [years] for a circular-ish orbit.

        Uses an effective density model that accounts for the thermospheric
        scale height (~60 km) at high altitudes, which is very different
        from the tropospheric scale height (~8.5 km) used in the atmosphere model.

        King-Hele (1987): τ = B × H_eff / (ρ × v)

        At low altitudes (< 200 km): use atmosphere model density directly.
        At high altitudes: use empirical exponential with H_eff ~ 60 km.
        """
        if not planet.atmosphere.enabled:
            return float("inf")

        a   = planet.radius + altitude_m
        mu  = G * planet.mass
        v   = circular_speed(a, mu)
        P   = planet.atmosphere.surface_pressure
        g   = G * planet.mass / planet.radius**2
        H_tropo = planet.atmosphere.scale_height  # tropospheric scale height

        # At altitudes above ~6× scale height, the single-exponential model
        # drastically underestimates density because the thermosphere has a
        # much longer scale height (~50-70 km for Earth).
        # 
        # We use a two-regime model:
        # 1. Low altitude (h < h_homo): single exponential from surface
        # 2. High altitude: thermospheric model calibrated to Earth's NRLMSISE-00:
        #    ρ_Earth(200km) ≈ 2.5e-10 kg/m³, H_thermo(200km) ≈ 50 km
        # 
        # For non-Earth atmospheres: scale by surface pressure ratio
        # (denser atmosphere → denser thermosphere at same fraction of scale height).

        # Earth reference (solar mean activity)
        EARTH_RHO_200  = 2.5e-10   # kg/m³ at 200 km altitude
        EARTH_H_THERMO = 52_000    # m — thermospheric scale height at 200 km
        EARTH_PSRF     = 101_325   # Pa
        EARTH_G        = 9.81      # m/s²

        h_homo = 8 * H_tropo   # ~ homopause: where single-exp breaks down

        if altitude_m <= h_homo:
            rho   = planet.atmosphere.density_at_altitude(altitude_m)
            H_eff = H_tropo
        else:
            # Thermospheric model
            # Scale density by (P_surface / P_earth) and (g_earth / g) for scale height
            P_ratio = P / EARTH_PSRF
            g_ratio = EARTH_G / g
            H_eff   = EARTH_H_THERMO * g_ratio   # scale height scales inversely with g

            # Reference density at h_homo, scaled from Earth 200 km reference
            alt_ref = 200_000  # 200 km reference altitude
            if altitude_m <= alt_ref:
                # Between h_homo and 200km: interpolate
                rho = EARTH_RHO_200 * P_ratio * math.exp(
                    -(altitude_m - alt_ref) / H_eff
                )
            else:
                rho = EARTH_RHO_200 * P_ratio * math.exp(
                    -(altitude_m - alt_ref) / H_eff
                )

        if rho < 1e-25:
            return float("inf")

        # Correct circular-orbit decay formula (Vallado 2013):
        # Δa per orbit = -2π ρ a² / B
        # Lifetime ≈ altitude / (|Δa/orbit| × orbits/yr)
        # This gives physically correct results (~2-5 yr at 400 km for Earth)
        da_per_orbit = 2 * math.pi * rho * a**2 / ballistic_coeff_kg_m2
        if da_per_orbit < 1e-9:
            return float("inf")
        T_orbit = TWO_PI / mean_motion(a, mu)
        da_per_year = da_per_orbit * (365.25 * 86400 / T_orbit)   # m/yr
        # Lifetime to decay from current altitude to ~50 km (reentry)
        # Use exponential density growth during descent: effective lifetime × 1/2
        lifetime_s = (altitude_m - 50_000) / da_per_year * 365.25 * 86400 * 0.5
        return max(0.0, lifetime_s / (365.25 * 86400))

    @staticmethod
    def minimum_safe_altitude_km(planet,
                                   spacecraft_mass_kg: float,
                                   ballistic_coeff_kg_m2: float,
                                   min_lifetime_years: float = 1.0) -> float:
        """
        The minimum altitude [km] at which the orbit survives for at least
        min_lifetime_years given the spacecraft's ballistic coefficient.

        Searches from 50 km to 2000 km in 1 km steps.
        Returns NaN if not found (planet has no appreciable atmosphere).
        """
        if not planet.atmosphere.enabled:
            return 0.0   # airless — any altitude is safe

        for alt_km in range(50, 2001, 1):
            tau = DragLifetime.lifetime_years(
                planet, alt_km * 1e3,
                spacecraft_mass_kg, ballistic_coeff_kg_m2
            )
            if tau >= min_lifetime_years:
                return float(alt_km)
        return float("nan")

    @staticmethod
    def decay_rate_km_per_day(planet,
                               altitude_m: float,
                               ballistic_coeff_kg_m2: float) -> float:
        """
        Orbital altitude decay rate [km/day] from atmospheric drag.
        da/dt = -2π a² ρ / B  (circular orbit approximation)
        """
        if not planet.atmosphere.enabled:
            return 0.0
        a   = planet.radius + altitude_m
        rho = planet.atmosphere.density_at_altitude(altitude_m)
        B   = ballistic_coeff_kg_m2
        da_dt = -TWO_PI * a**2 * rho / B   # m/s
        return da_dt * 86400 / 1e3   # km/day (negative)


# ─────────────────────────────────────────────────────────────────────────────
# Station-keeping budget
# ─────────────────────────────────────────────────────────────────────────────

class StationKeeping:
    """
    Estimate the ΔV budget needed to maintain an orbit against perturbations.
    Includes drag makeup, J2 RAAN control, and eccentricity maintenance.
    """

    @staticmethod
    def drag_makeup_dv_per_year(planet,
                                 altitude_m: float,
                                 ballistic_coeff_kg_m2: float) -> float:
        """
        ΔV per year [m/s/year] to compensate atmospheric drag.

        ΔV_drag ≈ (2π / T) × Δa_year  (Gauss variational equation, circular)
        """
        if not planet.atmosphere.enabled:
            return 0.0
        decay_km_day = abs(DragLifetime.decay_rate_km_per_day(
            planet, altitude_m, ballistic_coeff_kg_m2
        ))
        decay_m_year = decay_km_day * 1e3 * 365.25   # m/year

        # ΔV to raise orbit by Δa: ΔV = 0.5 × Δa × n  (circular orbit)
        a  = planet.radius + altitude_m
        n  = mean_motion(a, G * planet.mass)
        return 0.5 * decay_m_year * n   # m/s/year

    @staticmethod
    def raan_control_dv_per_year(planet,
                                  altitude_m: float,
                                  inclination_deg: float,
                                  target_raan_rate_deg_day: float = 0.0) -> float:
        """
        ΔV per year [m/s/year] to maintain a specific RAAN precession rate.
        Set target_raan_rate_deg_day = 0 for an inertially fixed orbit,
        or pass the required sun-sync rate for sun-synchronous maintenance.

        Uses out-of-plane manoeuvre approximation: ΔV = |Δn| × v × T_year / T_orbit
        (rough estimate — actual depends on manoeuvre timing and strategy).
        """
        a   = planet.radius + altitude_m
        mu  = G * planet.mass
        v   = circular_speed(a, mu)
        T   = orbital_period(a, mu)
        inc = inclination_deg * DEG

        actual_rate = J2Analysis.nodal_precession_rate_deg_day(
            planet, a, inc
        )
        delta_rate_deg_day = abs(actual_rate - target_raan_rate_deg_day)
        delta_rate_rad_s   = delta_rate_deg_day / RAD / 86400

        # ΔV ~ Δ(dΩ/dt) × a / n per orbit → scale to yearly
        # More accurate: out-of-plane ΔV to correct RAAN drift
        # ΔV_raan ≈ 2 v sin(Δθ/2) ≈ v × Δθ  for small angles
        # Δθ over one year = delta_rate_rad_s × 365.25 × 86400
        delta_omega_year = abs(delta_rate_rad_s) * 365.25 * 86400  # rad/year
        # Out-of-plane manoeuvre ΔV = v × δθ / 2 (two half-impulses)
        # For typical station-keeping, much less than this — correction per orbit:
        # ΔV_per_orbit = v × |Δθ_per_orbit| / 2
        # But realistically, RAAN is corrected by orbit lowering/raising, not out-of-plane
        # Cost: ~0.1-10 m/s/year is typical for minor RAAN corrections
        # Cap at physically reasonable value
        v_circ = circular_speed(a, mu)
        dv_year = v_circ * delta_omega_year / (2 * math.pi)  # fraction of one orbit
        return min(dv_year, 200.0)  # cap at 200 m/s/year (mission-stopping threshold)

    @staticmethod
    def total_annual_budget(planet,
                             altitude_km: float,
                             inclination_deg: float,
                             ballistic_coeff_kg_m2: float = 100.0,
                             stellar_orbital_period_s: float = None) -> dict:
        """
        Full annual ΔV budget breakdown.

        Parameters
        ----------
        stellar_orbital_period_s : if provided, target sun-synchronous RAAN rate
        """
        alt_m = altitude_km * 1e3
        dv_drag = StationKeeping.drag_makeup_dv_per_year(
            planet, alt_m, ballistic_coeff_kg_m2
        )

        target_rate = 0.0
        if stellar_orbital_period_s:
            target_rate = TWO_PI / stellar_orbital_period_s * RAD * 86400  # deg/day

        dv_raan = StationKeeping.raan_control_dv_per_year(
            planet, alt_m, inclination_deg, target_rate
        )

        total = dv_drag + dv_raan
        return {
            "drag_dv_m_s_yr":    dv_drag,
            "raan_dv_m_s_yr":    dv_raan,
            "total_dv_m_s_yr":   total,
            "drag_fraction":     dv_drag / max(total, 1e-9),
            "mission_10yr_dv":   total * 10,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Repeat ground track
# ─────────────────────────────────────────────────────────────────────────────

class RepeatGroundTrack:
    """
    Design a repeat ground-track orbit: the satellite passes over every
    surface point exactly once in every N_days days (after K orbits).

    Condition: K × T_orbit = N_days × T_planet_rotation  (exact repeat)

    Used by: Landsat (16-day), Sentinel-2 (10-day), ICESat-2 (91-day).
    """

    @staticmethod
    def repeat_semi_major_axis(planet,
                                k_orbits: int,
                                n_days: int,
                                inclination_rad: float = None) -> float:
        """
        Semi-major axis [m] for a K-orbit / N-day repeat ground track.
        Ignores J2 correction (first approximation).

        T_orbit = N_days × T_rotation / K
        a = (mu × T_orbit² / 4π²)^(1/3)
        """
        mu   = G * planet.mass
        T_rot = planet.rotation_period
        T_orb = n_days * T_rot / k_orbits
        return (mu * T_orb**2 / (4 * math.pi**2)) ** (1/3)

    @staticmethod
    def find_repeat_orbits(planet,
                            altitude_range_km: tuple = (200, 800),
                            max_days: int = 30,
                            max_orbits_per_day: int = 20) -> list[dict]:
        """
        Find all exact repeat ground-track solutions within an altitude window.

        Returns list of dicts sorted by altitude.
        """
        mu  = G * planet.mass
        results = []
        for n_days in range(1, max_days + 1):
            for k_orbits in range(n_days, n_days * max_orbits_per_day + 1):
                a = RepeatGroundTrack.repeat_semi_major_axis(planet, k_orbits, n_days)
                alt_km = (a - planet.radius) / 1e3
                if altitude_range_km[0] <= alt_km <= altitude_range_km[1]:
                    T_orb = orbital_period(a, mu)
                    results.append({
                        "k_orbits":       k_orbits,
                        "n_days":         n_days,
                        "alt_km":         round(alt_km, 1),
                        "period_min":     round(T_orb / 60, 2),
                        "orbits_per_day": round(k_orbits / n_days, 3),
                    })
        results.sort(key=lambda x: x["alt_km"])
        return results

    @staticmethod
    def equatorial_track_spacing_km(planet,
                                     k_orbits: int,
                                     n_days: int) -> float:
        """
        Spacing between adjacent ground tracks at the equator [km].
        spacing = 2π R_planet / K
        """
        return TWO_PI * planet.radius / k_orbits / 1e3


# ─────────────────────────────────────────────────────────────────────────────
# OrbitalDesign — all-in-one design summary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrbitDesign:
    """
    A complete orbit design for a science mission.
    Bundles the chosen parameters with all derived quantities.
    """
    planet_name:            str
    altitude_km:            float
    inclination_deg:        float
    eccentricity:           float = 0.0
    ballistic_coeff_kg_m2:  float = 100.0
    stellar_orbital_period_s: Optional[float] = None   # for sun-sync

    # Populated by compute()
    _data: dict = field(default_factory=dict, repr=False)

    def compute(self, planet) -> "OrbitDesign":
        """Compute all derived quantities for this orbit design."""
        mu  = G * planet.mass
        a   = semi_major_axis_from_altitude(planet.radius, self.altitude_km * 1e3)
        inc = self.inclination_deg * DEG
        e   = self.eccentricity

        j2_rates = J2Analysis.secular_rates_summary(
            planet, self.altitude_km, self.inclination_deg, e
        )
        frozen = FrozenOrbit.frozen_orbit_params(
            planet, self.altitude_km, self.inclination_deg
        )
        lifetime = DragLifetime.lifetime_years(
            planet, self.altitude_km * 1e3,
            spacecraft_mass_kg=1000.0,
            ballistic_coeff_kg_m2=self.ballistic_coeff_kg_m2
        )
        sk = StationKeeping.total_annual_budget(
            planet, self.altitude_km, self.inclination_deg,
            self.ballistic_coeff_kg_m2, self.stellar_orbital_period_s
        )

        ss_inc = None
        if self.stellar_orbital_period_s:
            ss_inc = SunSynchronousOrbit.sun_sync_inclination(
                planet, self.altitude_km * 1e3,
                self.stellar_orbital_period_s, e
            )

        self._data = {
            "semi_major_axis_km":   a / 1e3,
            "orbital_period_min":   orbital_period(a, mu) / 60,
            "circular_speed_km_s":  circular_speed(a, mu) / 1e3,
            "j2_rates":             j2_rates,
            "frozen_orbit":         frozen,
            "drag_lifetime_yr":     lifetime,
            "station_keeping":      sk,
            "sun_sync_inclination_deg": ss_inc,
            "is_sun_synchronous":   (
                ss_inc is not None and
                abs(ss_inc - self.inclination_deg) < 0.5
            ),
        }
        return self

    def report(self) -> str:
        d = self._data
        if not d:
            return "Call .compute(planet) first."
        j2  = d["j2_rates"]
        frz = d["frozen_orbit"]
        sk  = d["station_keeping"]

        ss_line = ""
        if d["sun_sync_inclination_deg"] is not None:
            ss_line = (
                f"\n  Sun-sync inclination : {d['sun_sync_inclination_deg']:.2f}°  "
                f"({'✓ this orbit' if d['is_sun_synchronous'] else '✗ not sun-sync'})"
            )

        return (
            f"═══ Orbit design: {self.planet_name} @ {self.altitude_km:.0f} km ═══\n"
            f"  Semi-major axis      : {d['semi_major_axis_km']:.1f} km\n"
            f"  Inclination          : {self.inclination_deg:.2f}°\n"
            f"  Eccentricity         : {self.eccentricity:.5f}\n"
            f"  Orbital period       : {d['orbital_period_min']:.2f} min\n"
            f"  Circular speed       : {d['circular_speed_km_s']:.3f} km/s\n"
            f"\n  ── J2 secular rates ──\n"
            f"  RAAN precession      : {j2['dOmega_dt_deg_day']:+.4f} °/day\n"
            f"  Apsidal precession   : {j2['domega_dt_deg_day']:+.4f} °/day\n"
            f"  Critical inclination : {j2['critical_inclination_deg']:.2f}°\n"
            f"{ss_line}\n"
            f"\n  ── Frozen orbit ──\n"
            f"  Frozen eccentricity  : {frz['frozen_ecc']:.6f}\n"
            f"  Alt. variation       : ±{frz['alt_variation_km']/2:.1f} km\n"
            f"\n  ── Drag & station-keeping ──\n"
            f"  Drag lifetime        : {d['drag_lifetime_yr']:.1f} yr\n"
            f"  Annual drag ΔV       : {sk['drag_dv_m_s_yr']:.2f} m/s/yr\n"
            f"  Annual RAAN ΔV       : {sk['raan_dv_m_s_yr']:.2f} m/s/yr\n"
            f"  Total annual ΔV      : {sk['total_dv_m_s_yr']:.2f} m/s/yr\n"
            f"  10-yr mission ΔV     : {sk['mission_10yr_dv']:.1f} m/s\n"
        )
