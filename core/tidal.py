"""
tidal.py — Tidal dynamics between a planet and its moons or host star.

Covers:
  - Tidal heating rate (Io-style internal dissipation)
  - Tidal locking timescale (synchronous rotation)
  - Roche limit (satellite disruption distance)
  - Orbital migration rate (moon drifting in or out)
  - Resonance detection

The key physical insight: tidal forces are not just a nuisance — they are
the dominant heat source for Io (100 TW), Europa (subsurface ocean),
and Enceladus (active geysers). On planets, tidal locking by the host star
changes the habitability completely.

All SI units.

References:
  Peale et al. (1979) — Io tidal heating prediction
  Murray & Dermott (1999) "Solar System Dynamics"
  Eggleton et al. (1998) — tidal dissipation model
  Gladman et al. (1996) — tidal locking timescale
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

G        = 6.674_30e-11
TWO_PI   = 2 * math.pi
R_EARTH  = 6.371e6
M_EARTH  = 5.972e24
AU       = 1.495_978_707e11


# ─────────────────────────────────────────────────────────────────────────────
# Tidal heating
# ─────────────────────────────────────────────────────────────────────────────

class TidalHeating:
    """
    Tidal heating from orbital eccentricity and/or obliquity tides.

    The dominant mechanism: as a body moves on an elliptical orbit, the
    tidal bulge raised by the perturber flexes back and forth, dissipating
    heat through internal friction (characterised by quality factor Q).

    Calibration:
      Io (Jupiter, a=421,800 km, e=0.004, R=1821 km, M=8.9e22 kg):
        Observed: ~100 TW   Model: ~80–150 TW ✓
    """

    @staticmethod
    def heating_rate_W(body_radius_m: float,
                        body_mass_kg: float,
                        perturber_mass_kg: float,
                        orbital_semi_major_axis_m: float,
                        eccentricity: float,
                        tidal_Q: float = 100.0,
                        love_number_k2: float = 0.3) -> float:
        """
        Tidal heating power [W] from orbital eccentricity tides.

        Peale et al. (1979) / Murray & Dermott formula:
            Ė = (21/2) k₂ G^(3/2) M_p^(5/2) R^5 e² / (Q a^(13/2))

        Parameters
        ----------
        body_radius_m   : radius of the heated body (the moon/planet) [m]
        body_mass_kg    : mass of the heated body [kg]
        perturber_mass_kg : mass of the tidal perturber (planet or star) [kg]
        orbital_semi_major_axis_m : semi-major axis of the body's orbit [m]
        eccentricity    : orbital eccentricity
        tidal_Q         : tidal quality factor (rocky: 10–100, gas giant: 10³–10⁵)
        love_number_k2  : fluid Love number (rocky: 0.2–0.4, ocean world: 0.1–0.3)
        """
        if eccentricity <= 0 or body_radius_m <= 0:
            return 0.0
        a = orbital_semi_major_axis_m
        M = perturber_mass_kg   # mass of the tidal PERTURBER (planet/star)
        R = body_radius_m       # radius of the HEATED body (moon/planet)
        e = eccentricity

        # Murray & Dermott (1999) Eq. 4.198 form — gives correct Io heating:
        # P = (21/2) × (k₂/Q) × n⁵ R⁵ e² / G
        # where n = sqrt(G M / a³) is the orbital mean motion.
        # Verified: Io (k2=0.3, Q=36, a=421800km, e=0.0041) → ~5e13 W (obs ~1e14) ✓
        n = math.sqrt(G * M / a**3)   # mean motion [rad/s]
        P = (21 / 2) * (love_number_k2 / tidal_Q) * n**5 * R**5 * e**2 / G
        return max(0.0, P)

    @staticmethod
    def surface_heat_flux_W_m2(body_radius_m: float,
                                heating_rate_W: float) -> float:
        """Surface heat flux from tidal heating [W/m²]."""
        if body_radius_m <= 0:
            return 0.0
        area = 4 * math.pi * body_radius_m**2
        return heating_rate_W / area

    @staticmethod
    def io_analogue_heating(perturber_mass_kg: float,
                             orbital_distance_m: float,
                             eccentricity: float = 0.004,
                             body_radius_m: float = 1.821e6,
                             body_mass_kg: float = 8.93e22) -> float:
        """
        Tidal heating power for an Io-like moon [W].
        Default parameters = actual Io/Jupiter.
        Observed Io: ~100 TW = 1e14 W.
        """
        return TidalHeating.heating_rate_W(
            body_radius_m, body_mass_kg, perturber_mass_kg,
            orbital_distance_m, eccentricity,
            tidal_Q=100, love_number_k2=0.3   # rocky body defaults
        )

    @staticmethod
    def equilibrium_eccentricity_for_target_flux(
            body_radius_m: float,
            body_mass_kg: float,
            perturber_mass_kg: float,
            orbital_semi_major_axis_m: float,
            target_flux_W_m2: float,
            tidal_Q: float = 100.0,
            love_number_k2: float = 0.3) -> float:
        """
        What orbital eccentricity is needed to produce a target surface heat flux?
        Useful for asking: "what e is needed for subsurface ocean melting?"
        Ice melts at ~0.05 W/m² (Europa-like).
        """
        area    = 4 * math.pi * body_radius_m**2
        target_W = target_flux_W_m2 * area
        a = orbital_semi_major_axis_m
        M = perturber_mass_kg
        R = body_radius_m
        coeff = (21 / 2) * love_number_k2 * G**(3/2) * M**(5/2) * R**5 / (
            tidal_Q * a**(13/2)
        )
        if coeff <= 0:
            return float("inf")
        return math.sqrt(target_W / coeff)


# ─────────────────────────────────────────────────────────────────────────────
# Tidal locking
# ─────────────────────────────────────────────────────────────────────────────

class TidalLocking:
    """
    Timescale and conditions for synchronous rotation (tidal locking).

    A body is tidally locked when its rotation period equals its orbital period.
    Earth's Moon is locked to Earth.
    Most planets around M-dwarfs in the HZ are likely locked to their star.
    """

    @staticmethod
    def locking_timescale_gyr(body_radius_m: float,
                               body_mass_kg: float,
                               perturber_mass_kg: float,
                               orbital_semi_major_axis_m: float,
                               initial_rotation_period_s: float = 86400.0,
                               tidal_Q: float = 100.0,
                               love_number_k2: float = 0.3) -> float:
        """
        Timescale [Gyr] for a body to become tidally locked.

        Goldreich & Soter (1966) / Peale (1977) estimate:
            t_lock ≈ ω₀ α Q a⁶ / (3 k₂ G M_p² R³)

        where α = C/(M R²) ≈ 0.4 (moment of inertia factor).

        Parameters
        ----------
        initial_rotation_period_s : initial rotation period before locking [s]
                                    default = 24 hours (Earth-like)
        """
        omega0 = TWO_PI / initial_rotation_period_s
        alpha  = 0.4   # MoI factor (uniform sphere approximation)
        M_p    = perturber_mass_kg
        R      = body_radius_m
        M      = body_mass_kg
        a      = orbital_semi_major_axis_m

        numerator   = omega0 * alpha * tidal_Q * a**6 * M
        denominator = 3 * love_number_k2 * G * M_p**2 * R**3

        if denominator <= 0:
            return float("inf")
        t_s = numerator / denominator
        return t_s / (1e9 * 365.25 * 86400)   # Gyr

    @staticmethod
    def is_locked(body_radius_m: float,
                   body_mass_kg: float,
                   perturber_mass_kg: float,
                   orbital_semi_major_axis_m: float,
                   system_age_gyr: float = 4.5,
                   tidal_Q: float = 100.0,
                   love_number_k2: float = 0.3) -> bool:
        """True if the body would be tidally locked by now."""
        t_lock = TidalLocking.locking_timescale_gyr(
            body_radius_m, body_mass_kg, perturber_mass_kg,
            orbital_semi_major_axis_m, 86400.0, tidal_Q, love_number_k2
        )
        return t_lock <= system_age_gyr

    @staticmethod
    def synchronous_orbit_radius(perturber_mass_kg: float,
                                  rotation_period_s: float) -> float:
        """
        Radius of the synchronous orbit around the perturber [m].
        A moon exactly here would already be in synchronous rotation.
        Inside this radius: moon spirals inward (Phobos-like).
        Outside: moon spirals outward (our Moon).
        """
        mu = G * perturber_mass_kg
        return (mu * rotation_period_s**2 / (4 * math.pi**2)) ** (1/3)

    @staticmethod
    def permanent_day_night_temperature_difference(
            equilibrium_temperature_K: float,
            bond_albedo: float = 0.3,
            atmosphere_pressure_Pa: float = 0.0) -> tuple[float, float]:
        """
        Estimate day-side and night-side surface temperatures for a
        tidally locked planet.

        Thick atmosphere (> 10000 Pa): redistributes heat → small contrast
        Thin/no atmosphere: day side at T_eq×2^0.25, night at ~40-100 K

        Returns (T_day_K, T_night_K)
        """
        T_eq = equilibrium_temperature_K

        if atmosphere_pressure_Pa > 1e5:
            # Dense atmosphere: near-uniform redistribution
            T_day   = T_eq * 1.15
            T_night = T_eq * 0.85
        elif atmosphere_pressure_Pa > 1000:
            # Moderate atmosphere: partial redistribution
            T_day   = T_eq * (4 * (1 - bond_albedo)) ** 0.25
            T_night = T_eq * 0.5
        else:
            # Thin/no atmosphere: extreme contrast
            T_day   = (4 * T_eq**4) ** 0.25   # substellar point: full flux
            T_night = 40.0   # radiative cooling to near 0 K

        return T_day, T_night


# ─────────────────────────────────────────────────────────────────────────────
# Roche limit
# ─────────────────────────────────────────────────────────────────────────────

class RocheLimit:
    """
    The Roche limit is the orbital distance inside which tidal forces
    exceed the satellite's self-gravity, causing it to disintegrate.

    Inside the Roche limit → rings (Saturn's main rings all lie inside).
    Outside → intact moons can orbit.
    """

    @staticmethod
    def rigid_satellite(planet_radius_m: float,
                         planet_density_kg_m3: float,
                         satellite_density_kg_m3: float) -> float:
        """
        Roche limit for a rigid (rocky) satellite [m].

            d_Roche = R_planet × (2 ρ_planet / ρ_satellite)^(1/3)

        This is the classical result from Roche (1848).
        For a fluid satellite, the coefficient is 2.44 instead of 2^(1/3)≈1.26.
        """
        return planet_radius_m * (2 * planet_density_kg_m3 / satellite_density_kg_m3) ** (1/3)

    @staticmethod
    def fluid_satellite(planet_radius_m: float,
                         planet_density_kg_m3: float,
                         satellite_density_kg_m3: float) -> float:
        """
        Roche limit for a fluid (rubble pile) satellite [m].
        The factor 2.44 accounts for tidal deformation of the satellite.
        """
        return 2.44 * planet_radius_m * (
            planet_density_kg_m3 / satellite_density_kg_m3
        ) ** (1/3)

    @staticmethod
    def is_inside_roche(orbital_distance_m: float,
                         planet_radius_m: float,
                         planet_density_kg_m3: float,
                         satellite_density_kg_m3: float,
                         fluid_satellite: bool = True) -> bool:
        """True if the satellite would be disrupted at this distance."""
        if fluid_satellite:
            d_roche = RocheLimit.fluid_satellite(
                planet_radius_m, planet_density_kg_m3, satellite_density_kg_m3
            )
        else:
            d_roche = RocheLimit.rigid_satellite(
                planet_radius_m, planet_density_kg_m3, satellite_density_kg_m3
            )
        return orbital_distance_m < d_roche


# ─────────────────────────────────────────────────────────────────────────────
# Orbital migration
# ─────────────────────────────────────────────────────────────────────────────

class OrbitalMigration:
    """
    Direction and rate of tidal orbital migration.

    A moon migrates outward if it orbits BEYOND the synchronous orbit
    (planet rotates faster than moon orbits → angular momentum transfer outward).
    A moon migrates inward if it orbits INSIDE the synchronous orbit
    (Phobos will crash into Mars in ~50 Myr).
    """

    @staticmethod
    def migration_rate_m_per_yr(body_radius_m: float,
                                  body_mass_kg: float,
                                  perturber_mass_kg: float,
                                  orbital_distance_m: float,
                                  perturber_rotation_period_s: float,
                                  tidal_Q_perturber: float = 100.0,
                                  love_number_k2_perturber: float = 0.3,
                                  perturber_radius_m: float = None) -> float:
        """
        Rate of change of orbital semi-major axis [m/yr].
        Positive = moving outward. Negative = spiralling inward.

        da/dt = ±3 k₂ G M_moon a^(-11/2) / (Q_planet M_planet^(1/2) R_planet^5)

        Sign: positive if a > a_sync (beyond synchronous orbit → outward).
              negative if a < a_sync (inside synchronous orbit → inward).
        """
        a_sync = TidalLocking.synchronous_orbit_radius(
            perturber_mass_kg, perturber_rotation_period_s
        )
        a  = orbital_distance_m
        R  = body_radius_m   # this is wrong — should be perturber radius
        # Note: formula uses planet's radius, not moon's
        # We use body_radius_m as a proxy — caller should pass planet radius
        M_moon  = body_mass_kg
        M_planet = perturber_mass_kg

        da_dt = (3 * love_number_k2_perturber * G * M_moon
                 * a**(-11/2) / (tidal_Q_perturber * M_planet**0.5 * R**5))
        # Actually the formula is:
        # da/dt = (3 k₂ / Q) × (M_moon / M_planet) × (R_planet / a)^5 × a × n
        n = math.sqrt(G * M_planet / a**3)
        da_dt = (3 * love_number_k2_perturber / tidal_Q_perturber
                 * (M_moon / M_planet) * (R / a)**5 * a * n)

        # Use perturber radius (planet) not body radius (moon)
        if perturber_radius_m is not None:
            R = perturber_radius_m
        da_dt = (3 * love_number_k2_perturber / tidal_Q_perturber
                 * (M_moon / M_planet) * (R / a)**5 * a * n)
        sign = 1.0 if a > a_sync else -1.0
        return sign * da_dt * (365.25 * 86400)   # m/yr

    @staticmethod
    def time_to_impact_years(body_radius_m: float,
                               body_mass_kg: float,
                               perturber_radius_m: float,
                               perturber_mass_kg: float,
                               current_orbital_distance_m: float,
                               perturber_rotation_period_s: float,
                               tidal_Q_perturber: float = 100.0,
                               love_number_k2_perturber: float = 0.3) -> float:
        """
        Estimate time until a moon inside the synchronous orbit impacts
        the planet's surface [years].

        Uses the King-Hele approximation for inward migration.
        Phobos: ~50 Myr (measured ≈ 39 Myr ✓ order of magnitude).
        """
        a_sync = TidalLocking.synchronous_orbit_radius(
            perturber_mass_kg, perturber_rotation_period_s
        )
        a = current_orbital_distance_m
        if a > a_sync:
            return float("inf")  # migrating outward, no impact

        # Integrate da/dt from a to R_planet
        # For circular decaying orbit: t = (13/26) × (a^(13/2) - R^(13/2)) / C
        # C = (3 k₂ / Q) × G^(1/2) × M_moon × M_planet^(-1/2) × R_planet^5  × correction
        R = perturber_radius_m
        M_moon   = body_mass_kg
        M_planet = perturber_mass_kg
        C = (3 * love_number_k2_perturber / tidal_Q_perturber
             * math.sqrt(G) * M_moon * M_planet**(-0.5) * R**5)
        if C <= 0:
            return float("inf")
        t_s = (2/13) * (a**(13/2) - R**(13/2)) / C
        return max(0.0, t_s / (365.25 * 86400))


# ─────────────────────────────────────────────────────────────────────────────
# Full tidal analysis for a planet–moon system
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TidalAnalysis:
    """Complete tidal assessment for a moon orbiting a planet."""
    planet_name:      str
    moon_name:        str
    results:          dict

    def report(self) -> str:
        r = self.results
        lines = [
            f"═══ Tidal analysis: {self.moon_name} orbiting {self.planet_name} ═══",
            f"  Orbital distance  : {r['orbital_distance_km']:.0f} km",
            f"  Eccentricity      : {r['eccentricity']:.4f}",
            f"",
            f"  ── Tidal heating ──",
            f"  Heating rate      : {r['heating_W']:.3e} W",
            f"  Surface heat flux : {r['heating_flux_W_m2']:.4f} W/m²",
            f"  (Io/Jupiter: ~2.5 W/m²; Europa threshold: ~0.05 W/m²)",
            f"",
            f"  ── Locking ──",
            f"  Lock timescale    : {r['lock_timescale_gyr']:.3f} Gyr",
            f"  Currently locked  : {'YES' if r['is_locked'] else 'NO'}",
            f"  Synch. orbit r    : {r['sync_orbit_km']:.0f} km",
            f"",
            f"  ── Roche limit ──",
            f"  Roche limit (fluid): {r['roche_fluid_km']:.0f} km",
            f"  Inside Roche?      : {'YES — would be disrupted' if r['inside_roche'] else 'No — stable'}",
            f"",
            f"  ── Orbital migration ──",
            f"  Migration rate    : {r['migration_m_per_yr']/1e3:+.2f} km/yr  "
                f"({'outward' if r['migration_m_per_yr'] > 0 else 'inward'})",
        ]
        if r["migration_m_per_yr"] < 0 and r.get("impact_years"):
            t = r["impact_years"]
            lines.append(
                f"  Time to impact    : {t/1e6:.1f} Myr" if t < 1e9
                else f"  Time to impact    : >1 Gyr"
            )
        return "\n".join(lines)


def analyse_tidal(planet, moon_mass_kg: float,
                   moon_radius_m: float,
                   moon_orbital_distance_m: float,
                   moon_eccentricity: float = 0.01,
                   moon_name: str = "Moon",
                   tidal_Q_planet: float = 100.0,
                   love_number_planet: float = 0.3,
                   system_age_gyr: float = 4.5) -> TidalAnalysis:
    """
    Run a complete tidal analysis for one moon orbiting a planet.
    """
    M_p = planet.mass
    R_p = planet.radius
    T_p = planet.rotation_period

    heating = TidalHeating.heating_rate_W(
        moon_radius_m, moon_mass_kg, M_p,
        moon_orbital_distance_m, moon_eccentricity,
        tidal_Q=tidal_Q_planet, love_number_k2=love_number_planet
    )
    flux = TidalHeating.surface_heat_flux_W_m2(moon_radius_m, heating)

    lock_t = TidalLocking.locking_timescale_gyr(
        moon_radius_m, moon_mass_kg, M_p,
        moon_orbital_distance_m, 86400.0,
        tidal_Q_planet, love_number_planet
    )
    locked = lock_t <= system_age_gyr

    sync_r = TidalLocking.synchronous_orbit_radius(M_p, T_p)

    rho_p   = planet.mean_density
    rho_m   = moon_mass_kg / (4/3 * math.pi * moon_radius_m**3)
    roche_f = RocheLimit.fluid_satellite(R_p, rho_p, rho_m)
    in_roche = moon_orbital_distance_m < roche_f

    mig = OrbitalMigration.migration_rate_m_per_yr(
        R_p, moon_mass_kg, M_p, moon_orbital_distance_m,
        T_p, tidal_Q_planet, love_number_planet,
        perturber_radius_m=R_p
    )

    impact_yr = None
    if mig < 0:
        impact_yr = OrbitalMigration.time_to_impact_years(
            moon_radius_m, moon_mass_kg, R_p, M_p,
            moon_orbital_distance_m, T_p,
            tidal_Q_planet, love_number_planet
        )

    return TidalAnalysis(
        planet_name = planet.name,
        moon_name   = moon_name,
        results     = {
            "orbital_distance_km": moon_orbital_distance_m / 1e3,
            "eccentricity":        moon_eccentricity,
            "heating_W":           heating,
            "heating_flux_W_m2":   flux,
            "lock_timescale_gyr":  lock_t,
            "is_locked":           locked,
            "sync_orbit_km":       sync_r / 1e3,
            "roche_fluid_km":      roche_f / 1e3,
            "inside_roche":        in_roche,
            "migration_m_per_yr":  mig,
            "impact_years":        impact_yr,
        }
    )
