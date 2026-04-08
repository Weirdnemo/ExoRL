"""
habitability.py — Planetary habitability assessment.

Produces a structured, scored habitability report for any Planet + Star + orbit.

The assessment covers:
  1. Stellar context     — star type, age, XUV activity, main sequence lifetime
  2. Orbital context     — HZ position, tidal locking, orbital eccentricity
  3. Surface temperature — equilibrium + greenhouse → actual T_surface
  4. Liquid water window — is liquid water stable at the surface?
  5. Atmospheric retention — Jeans escape timescales per species
  6. Magnetic protection  — is the planet shielded from stellar wind stripping?
  7. Interior activity   — geologic/volcanic activity driving atmosphere replenishment
  8. Size class          — does the planet mass allow long-term retention?

Each factor is scored 0–1. The overall habitability index is the geometric mean
(so any factor near zero pulls the whole score down — all conditions must be met).

This is NOT a probability of life — it is a dimensionless index that summarises
how Earth-like the conditions are. A score of 1.0 = identical to Earth.
A score of 0.5 means the planet is borderline habitable.

All SI units unless noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from planet_rl.core.atmosphere_science import (
    STANDARD_COMPOSITIONS,
    GreenhouseModel,
    JeansEscape,
    analyse_atmosphere,
)

# ── Physical constants ────────────────────────────────────────────────────────
R_GAS = 8.314_462
k_B = 1.380_649e-23
G = 6.674_30e-11
AU = 1.495_978_707e11
R_EARTH = 6.371e6
M_EARTH = 5.972e24


# ── Planetary size classification ─────────────────────────────────────────────
def size_class(planet) -> str:
    """Classify planet by radius relative to Earth."""
    r = planet.radius / R_EARTH
    if r < 0.3:
        return "Dwarf body"
    if r < 0.8:
        return "Sub-Earth"
    if r < 1.4:
        return "Earth-sized"
    if r < 2.0:
        return "Super-Earth"
    if r < 4.0:
        return "Mini-Neptune"
    if r < 8.0:
        return "Ice giant"
    return "Gas giant"


def composition_class(planet) -> str:
    """Infer bulk composition class from mean density."""
    rho = planet.mean_density
    if rho < 2000:
        return "Icy / volatile-rich"
    if rho < 3500:
        return "Rocky low-density"
    if rho < 4500:
        return "Rocky medium (Mars-like)"
    if rho < 6000:
        return "Rocky Earth-like"
    if rho < 8000:
        return "Iron-rich rocky"
    return "Ultra-dense / exotic"


# ─────────────────────────────────────────────────────────────────────────────
# Individual scoring functions  (each returns 0.0 – 1.0)
# ─────────────────────────────────────────────────────────────────────────────


def score_stellar_type(star) -> tuple[float, str]:
    """
    Score based on stellar type.
    G and K stars score highest — long-lived, stable, moderate UV.
    M dwarfs: habitable zones are tidally locked and flare-prone → reduced score.
    F stars: more UV, shorter lifetime.
    A/B/O stars: insufficient main sequence lifetime for life to develop.
    """
    from planet_rl.core.star import SpectralType

    T = star.spectral_type
    scores = {
        SpectralType.O: 0.00,
        SpectralType.B: 0.00,
        SpectralType.A: 0.05,
        SpectralType.F: 0.60,
        SpectralType.G: 1.00,
        SpectralType.K: 0.95,
        SpectralType.M: 0.45,
        SpectralType.L: 0.02,
        SpectralType.T: 0.00,
    }
    score = scores.get(T, 0.5)
    if T in (SpectralType.G, SpectralType.K):
        note = f"Ideal stellar type ({T.name}) — stable, long-lived"
    elif T == SpectralType.M:
        note = "M dwarf — HZ planets likely tidally locked; high flare activity"
    elif T == SpectralType.F:
        note = "F star — higher UV and shorter main sequence (~4 Gyr)"
    else:
        note = f"{T.name} star — unsuitable for long-duration biology"
    return score, note


def score_stellar_age(star) -> tuple[float, str]:
    """
    Stellar age relative to a minimum needed for complex life (~1 Gyr)
    and the remaining main sequence lifetime.
    """
    min_age = 1.0  # Gyr minimum to develop life
    if star.age < min_age:
        score = star.age / min_age * 0.3
        note = f"Star too young ({star.age:.1f} Gyr) — insufficient time for life"
    elif star.remaining_lifetime_gyr < 1.0:
        score = max(0.0, star.remaining_lifetime_gyr) / 1.0 * 0.4
        note = f"Star near end of main sequence ({star.remaining_lifetime_gyr:.1f} Gyr remaining)"
    else:
        score = min(1.0, star.age / 4.5)  # peaks at solar age
        note = f"Stellar age {star.age:.1f} Gyr — suitable for complex life"
    return score, note


def score_habitable_zone(planet, star, orbital_distance_m: float) -> tuple[float, str]:
    """
    Score based on position within the habitable zone.
    Peak (1.0) at the centre of the HZ.
    0.0 outside the optimistic HZ bounds.
    """
    hz_in = star.hz_inner_m
    hz_out = star.hz_outer_m
    hz_in_opt = star.hz_inner_optimistic_m
    hz_out_opt = star.hz_outer_optimistic_m
    d_au = orbital_distance_m / AU

    if hz_in <= orbital_distance_m <= hz_out:
        # Inside conservative HZ
        hz_frac = star.hz_fraction(orbital_distance_m)
        # Peak score at hz_frac = 0.5 (middle of HZ)
        score = 1.0 - 2.0 * abs(hz_frac - 0.5)
        score = max(0.3, score)  # anywhere in conservative HZ is at least 0.3
        note = f"Inside conservative HZ at {d_au:.3f} AU (HZ position={hz_frac:.2f})"
    elif hz_in_opt <= orbital_distance_m <= hz_out_opt:
        # In optimistic extension
        score = 0.25
        note = f"In optimistic HZ extension ({d_au:.3f} AU) — marginal conditions"
    elif orbital_distance_m < hz_in_opt:
        # Too hot
        excess = (hz_in - orbital_distance_m) / hz_in
        score = max(0.0, 0.1 - excess)
        note = f"Inside inner HZ edge at {d_au:.3f} AU — likely too hot"
    else:
        # Too cold
        excess = (orbital_distance_m - hz_out) / hz_out
        score = max(0.0, 0.1 - excess)
        note = f"Outside outer HZ edge at {d_au:.3f} AU — likely too cold"
    return score, note


def score_surface_temperature(T_surface_K: float) -> tuple[float, str]:
    """
    Score based on surface temperature.
    Optimal: 270–315 K (liquid water stable).
    Steep falloff outside 200–400 K.
    """
    T_opt_lo, T_opt_hi = 270, 315
    T_hab_lo, T_hab_hi = 200, 400

    if T_opt_lo <= T_surface_K <= T_opt_hi:
        score = 1.0
        note = (
            f"Optimal surface temperature ({T_surface_K:.0f} K) — liquid water stable"
        )
    elif T_hab_lo <= T_surface_K < T_opt_lo:
        score = (T_surface_K - T_hab_lo) / (T_opt_lo - T_hab_lo)
        note = f"Cold surface ({T_surface_K:.0f} K) — water may be frozen"
    elif T_opt_hi < T_surface_K <= T_hab_hi:
        score = (T_hab_hi - T_surface_K) / (T_hab_hi - T_opt_hi)
        note = f"Warm surface ({T_surface_K:.0f} K) — approaching heat stress"
    else:
        score = 0.0
        if T_surface_K < T_hab_lo:
            note = f"Too cold ({T_surface_K:.0f} K) — water permanently frozen"
        else:
            note = f"Too hot ({T_surface_K:.0f} K) — water vapour or runaway greenhouse"
    return max(0.0, min(1.0, score)), note


def score_liquid_water(
    T_surface_K: float, surface_pressure_Pa: float
) -> tuple[float, str]:
    """
    Score based on whether liquid water is thermodynamically stable.
    Requires: 273.15 K < T < 647.1 K (critical point) AND P > 611 Pa (triple point).
    """
    T_melt = 273.15  # K
    T_crit = 647.1  # K  (supercritical above this)
    P_trip = 611.7  # Pa (triple point pressure)

    if surface_pressure_Pa < P_trip:
        score = 0.0
        note = f"Pressure too low ({surface_pressure_Pa:.0f} Pa < 612 Pa) — water sublimates"
    elif T_surface_K < T_melt:
        # Below freezing — partial credit if close
        deficit = T_melt - T_surface_K
        score = max(0.0, 0.2 - 0.01 * deficit)
        note = f"Below freezing ({T_surface_K:.0f} K) — surface water frozen"
    elif T_surface_K > T_crit:
        score = 0.0
        note = f"Above critical point ({T_surface_K:.0f} K) — water supercritical"
    else:
        # Liquid water stable — score peaks at ~290 K
        score = 1.0 - 0.5 * ((T_surface_K - 293) / 200) ** 2
        score = max(0.1, min(1.0, score))
        note = f"Liquid water stable at {T_surface_K:.0f} K, {surface_pressure_Pa / 1e5:.2f} bar"
    return score, note


def score_atmospheric_retention(
    planet, star=None, orbital_distance_m: float = None
) -> tuple[float, str]:
    """
    Score based on the planet's ability to retain its atmosphere over Gyr.
    Uses Jeans escape timescales for the bulk species.
    """
    if not planet.atmosphere.enabled:
        return 0.1, "No atmosphere — bare rock"

    jeans = JeansEscape.all_species_assessment(planet)
    if not jeans:
        return 0.5, "Unable to assess escape (no composition data)"

    # Use N2 or CO2 as the bulk retentive species
    # H2 and He escape easily — don't penalise if they're trace
    bulk_species = ["N2", "CO2", "CH4", "H2O"]
    n2_retained = True
    worst_lambda = float("inf")
    worst_species = None

    for sp in bulk_species:
        if sp in jeans:
            lam = jeans[sp]["lambda"]
            if lam < worst_lambda:
                worst_lambda = lam
                worst_species = sp

    if worst_lambda == float("inf"):
        return 0.7, "Bulk species not assessed"

    # Score based on Jeans parameter
    # λ > 40: essentially no escape — score 1.0
    # λ = 20: slow but significant — score 0.6
    # λ = 10: rapid escape — score 0.2
    # λ < 5: hydrodynamic blowoff — score 0.0
    if worst_lambda > 40:
        score = 1.0
        note = f"Stable retention (λ={worst_lambda:.0f} for {worst_species})"
    elif worst_lambda > 20:
        score = 0.5 + 0.5 * (worst_lambda - 20) / 20
        note = f"Moderate retention (λ={worst_lambda:.1f} for {worst_species})"
    elif worst_lambda > 10:
        score = 0.2 + 0.3 * (worst_lambda - 10) / 10
        note = f"Slow escape (λ={worst_lambda:.1f} for {worst_species})"
    elif worst_lambda > 5:
        score = 0.05 * (worst_lambda - 5) / 5
        note = f"Rapid escape (λ={worst_lambda:.1f} for {worst_species})"
    else:
        score = 0.0
        note = f"Hydrodynamic blowoff (λ={worst_lambda:.1f}) — atmosphere unsustainable"

    return score, note


def score_magnetic_protection(planet) -> tuple[float, str]:
    """
    Score based on magnetic field strength — protection from:
    1. Solar wind stripping of atmosphere
    2. Surface cosmic ray flux (relevant for biology)
    """
    # Get B in μT
    if hasattr(planet, "derived_magnetic_field_T"):
        B_T = planet.derived_magnetic_field_T()
    elif planet.magnetic_field.enabled:
        from planet_rl.core.planet import MagneticFieldStrength

        B_map = {
            MagneticFieldStrength.NONE: 0.0,
            MagneticFieldStrength.WEAK: 3e-6,
            MagneticFieldStrength.MEDIUM: 3e-5,
            MagneticFieldStrength.STRONG: 4e-4,
        }
        B_T = B_map.get(planet.magnetic_field.strength, 0.0)
    else:
        B_T = 0.0

    B_uT = B_T * 1e6

    # Earth: 30 μT → score 0.9 (not perfect — reversals, polar gaps)
    # Mars:  < 0.1 μT → score 0.05 (stripped)
    # Jupiter: 400 μT → score 1.0 but radiation belts add risk → 0.8

    if B_uT < 0.1:
        score = 0.05
        note = f"Negligible field ({B_uT:.2f} μT) — unshielded from solar wind"
    elif B_uT < 3.0:
        score = 0.15
        note = f"Weak field ({B_uT:.1f} μT) — partial atmospheric shielding"
    elif B_uT < 15:
        score = 0.55
        note = f"Moderate field ({B_uT:.1f} μT) — reasonable shielding"
    elif B_uT < 100:
        score = 0.85
        note = f"Strong Earth-like field ({B_uT:.1f} μT) — good atmospheric protection"
    else:
        score = 0.70  # very strong field → radiation belts add biological hazard
        note = f"Very strong field ({B_uT:.1f} μT) — intense radiation belts"
    return score, note


def score_tidal_locking(
    planet, star=None, orbital_distance_m: float = None
) -> tuple[float, str]:
    """
    Score penalising likely tidal locking.
    Locked planets have extreme permanent day/night temperature contrasts.
    Some models suggest atmospheric redistribution can still permit habitability,
    so we don't zero out — but it's a significant penalty.
    """
    if star is None or orbital_distance_m is None:
        if hasattr(planet, "star_context") and planet.star_context:
            star = planet.star_context
            orbital_distance_m = planet.orbital_distance_m
        else:
            return 0.8, "Tidal locking not assessed (no stellar context)"

    if star.is_tidally_locked(orbital_distance_m, planet.mass, planet.radius):
        d_au = orbital_distance_m / AU
        score = 0.35
        note = (
            f"Likely tidally locked at {d_au:.3f} AU — "
            f"permanent day/night temperature contrast"
        )
    else:
        score = 1.0
        note = "Not tidally locked — day/night cycle present"
    return score, note


def score_interior_activity(planet) -> tuple[float, str]:
    """
    Score based on geological activity.
    Active geology drives:
    - Volcanic outgassing (replenishes atmosphere)
    - Carbon-silicate cycle (long-term climate stabiliser)
    - Plate tectonics (nutrient cycling for life)

    A geologically dead planet loses its atmosphere and its climate thermostat.
    """
    from planet_rl.core.interior import ConvectionState

    if not (
        hasattr(planet, "interior") and planet.interior and planet.interior.enabled
    ):
        # Fall back to magnetic field proxy (active field → active interior)
        if planet.magnetic_field.enabled:
            return 0.6, "Magnetic field present (geological activity inferred)"
        return 0.4, "Interior model not available — assuming moderate activity"

    state = planet.interior.convection_state(planet.radius, planet.mass)
    hf = (
        planet.interior.radiogenic_heat_flux(planet.radius, planet.mass) * 1000
    )  # mW/m²

    if state == ConvectionState.VIGOROUS:
        score = 0.95
        note = f"Vigorous mantle convection (likely plate tectonics); heat flux={hf:.1f} mW/m²"
    elif state == ConvectionState.SLUGGISH:
        score = 0.70
        note = f"Sluggish convection (stagnant lid forming); heat flux={hf:.1f} mW/m²"
    elif state == ConvectionState.STAGNANT_LID:
        score = 0.40
        note = f"Stagnant lid regime (Venus-like); heat flux={hf:.1f} mW/m²"
    else:  # SHUTDOWN
        score = 0.10
        note = f"Interior shutdown — no volcanic outgassing; heat flux={hf:.1f} mW/m²"
    return score, note


def score_size_class(planet) -> tuple[float, str]:
    """
    Score based on planet size.
    Too small: cannot retain atmosphere long-term (Moon, Mars).
    Too large: likely mini-Neptune with thick H2 envelope (not rocky surface).
    Optimal: 0.5–2.0 R⊕ (Earth-sized to modest Super-Earth).
    """
    r = planet.radius / R_EARTH

    if r < 0.3:
        score = 0.05
        note = f"Too small ({r:.2f} R⊕) — cannot retain atmosphere"
    elif r < 0.7:
        score = 0.4 + 0.6 * (r - 0.3) / 0.4
        note = f"Small rocky body ({r:.2f} R⊕) — marginal atmospheric retention"
    elif r <= 1.6:
        score = 1.0
        note = f"Optimal size for rocky habitable world ({r:.2f} R⊕)"
    elif r <= 2.5:
        score = 1.0 - 0.5 * (r - 1.6) / 0.9
        note = f"Large Super-Earth ({r:.2f} R⊕) — may have thick H₂ envelope"
    elif r <= 4.0:
        score = 0.2
        note = (
            f"Mini-Neptune size ({r:.2f} R⊕) — likely volatile-rich, no solid surface"
        )
    else:
        score = 0.0
        note = f"Gas/ice giant ({r:.2f} R⊕) — no solid surface"
    return max(0.0, min(1.0, score)), note


# ─────────────────────────────────────────────────────────────────────────────
# HabitabilityAssessment  —  the main result object
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HabitabilityAssessment:
    """
    Complete habitability assessment for a Planet + Star + orbit.

    Attributes
    ----------
    overall_score   : float [0–1]  geometric mean of all factor scores
    factors         : dict[str, (score, note)]
    surface_temp_K  : derived surface temperature [K]
    T_eq_K          : equilibrium temperature [K]
    greenhouse_dT_K : greenhouse warming contribution [K]
    atmosphere_analysis : full dict from analyse_atmosphere()
    """

    planet_name: str
    star_name: str
    orbital_distance_au: float

    factors: dict  # {factor_name: (score, note)}
    overall_score: float

    surface_temp_K: float
    T_eq_K: float
    greenhouse_dT_K: float

    size_class: str
    composition_class: str
    atmosphere_analysis: dict

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def is_potentially_habitable(self) -> bool:
        """True if overall score > 0.3 (a generous threshold)."""
        return self.overall_score > 0.30

    @property
    def is_earth_like(self) -> bool:
        """True if overall score > 0.65 — broadly Earth-comparable."""
        return self.overall_score > 0.65

    @property
    def grade(self) -> str:
        """Letter grade for the overall assessment."""
        s = self.overall_score
        if s >= 0.80:
            return "A"
        if s >= 0.65:
            return "B"
        if s >= 0.45:
            return "C"
        if s >= 0.25:
            return "D"
        return "F"

    # ── Report ────────────────────────────────────────────────────────────────
    def report(self) -> str:
        """
        Full habitability report — a doctor-style write-up of the assessment.
        """
        lines = [
            "═" * 62,
            f"  HABITABILITY ASSESSMENT — {self.planet_name}",
            "═" * 62,
            f"  Star              : {self.star_name}",
            f"  Orbital distance  : {self.orbital_distance_au:.3f} AU",
            f"  Size class        : {self.size_class}",
            f"  Composition       : {self.composition_class}",
            "",
            f"  Overall score     : {self.overall_score:.3f} / 1.000  [Grade: {self.grade}]",
            f"  Potentially hab.  : {'YES' if self.is_potentially_habitable else 'NO'}",
            f"  Earth-like        : {'YES' if self.is_earth_like else 'NO'}",
            "",
            "  ── Thermal environment ──────────────────────────────────",
            f"  Equilibrium temp  : {self.T_eq_K:.1f} K  ({self.T_eq_K - 273.15:.1f} °C)",
            f"  Greenhouse ΔT     : +{self.greenhouse_dT_K:.1f} K",
            f"  Surface temp      : {self.surface_temp_K:.1f} K  ({self.surface_temp_K - 273.15:.1f} °C)",
        ]

        if self.atmosphere_analysis.get("enabled"):
            aa = self.atmosphere_analysis
            comp = aa.get("composition", {})
            lines += [
                "",
                "  ── Atmosphere ──────────────────────────────────────────",
                f"  Mean molar mass   : {aa.get('mean_molar_mass_g_mol', 0):.1f} g/mol",
                f"  Scale height      : {aa.get('scale_height_km', 0):.1f} km",
                f"  Atmospheric mass  : {aa.get('atmospheric_mass_kg', 0):.3e} kg",
                f"  Runaway GH        : {'YES — Venus-like state' if aa.get('runaway_greenhouse') else 'No'}",
                "  Composition (top species):",
            ]
            top5 = sorted(comp.items(), key=lambda x: -x[1])[:5]
            for sp, frac in top5:
                lines.append(f"    {sp:6s}  {frac * 100:6.2f} %")

            jeans = aa.get("jeans_escape", {})
            if jeans:
                lines.append("  Jeans escape (bulk species):")
                for sp in ["N2", "CO2", "CH4", "H2O", "H2"]:
                    if sp in jeans:
                        j = jeans[sp]
                        t = j["timescale_gyr"]
                        t_str = f"{t:.1f} Gyr" if t < 1e10 else "stable (>100 Gyr)"
                        lines.append(
                            f"    {sp:6s}  λ={j['lambda']:6.1f}  "
                            f"timescale={t_str}  "
                            f"retained: {'✓' if j['retained_4gyr'] else '✗'}"
                        )

        lines += [
            "",
            "  ── Factor scores ───────────────────────────────────────────",
        ]
        for name, (score, note) in self.factors.items():
            bar = "█" * int(score * 12) + "░" * (12 - int(score * 12))
            lines.append(f"  {bar}  {score:.2f}  {name}")
            lines.append(f"          └─ {note}")

        lines += [
            "",
            "  ── Interpretation ──────────────────────────────────────────",
        ]
        lines.append(self._interpretation())
        lines.append("═" * 62)
        return "\n".join(lines)

    def _interpretation(self) -> str:
        """Write a paragraph-form interpretation of the assessment."""
        s = self.overall_score
        T = self.surface_temp_K
        aa = self.atmosphere_analysis

        if s >= 0.75:
            verdict = (
                f"  {self.planet_name} is a strong habitability candidate with conditions "
                f"broadly similar to Earth. Surface temperature of {T:.0f} K supports "
                f"liquid water, the atmosphere is well-retained, and the star provides "
                f"a stable electromagnetic environment."
            )
        elif s >= 0.50:
            verdict = (
                f"  {self.planet_name} has marginal but potentially habitable conditions. "
                f"Several factors score below Earth-standard — see the factor table above. "
                f"Habitability would depend sensitively on feedback processes not captured "
                f"in this simplified model."
            )
        elif s >= 0.25:
            verdict = (
                f"  {self.planet_name} is likely uninhabitable under Earth-centric definitions "
                f"but has some interesting conditions. Surface temperature {T:.0f} K and an "
                f"overall score of {s:.2f} put it in the 'marginal' category. "
                f"Exotic life forms tolerant of extreme conditions cannot be excluded."
            )
        else:
            verdict = (
                f"  {self.planet_name} is uninhabitable as assessed. The overall score of "
                f"{s:.2f} reflects one or more critical failures (temperature, atmosphere "
                f"loss, wrong stellar type, or wrong orbital position)."
            )
        return verdict

    def summary_line(self) -> str:
        """Single-line summary for tables and comparisons."""
        return (
            f"{self.planet_name:20s}  "
            f"score={self.overall_score:.3f}  "
            f"grade={self.grade}  "
            f"T_surf={self.surface_temp_K:.0f}K  "
            f"{'habitable' if self.is_potentially_habitable else 'uninhabitable'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def assess_habitability(
    planet, star=None, orbital_distance_m: float = None, bond_albedo: float = 0.3
) -> HabitabilityAssessment:
    """
    Compute a full HabitabilityAssessment for a planet.

    Parameters
    ----------
    planet              : Planet object
    star                : Star object (uses planet.star_context if None)
    orbital_distance_m  : orbital distance [m] (uses planet.orbital_distance_m if None)
    bond_albedo         : bond albedo for T_eq calculation (default 0.3)

    Returns
    -------
    HabitabilityAssessment
    """
    # Resolve star and distance
    if star is None and hasattr(planet, "star_context"):
        star = planet.star_context
    if orbital_distance_m is None and hasattr(planet, "orbital_distance_m"):
        orbital_distance_m = planet.orbital_distance_m

    if star is None:
        raise ValueError(
            "star must be provided either directly or via planet.star_context"
        )
    if orbital_distance_m is None:
        raise ValueError(
            "orbital_distance_m must be provided either directly or via "
            "planet.orbital_distance_m"
        )

    d_au = orbital_distance_m / AU

    # ── Atmosphere analysis ────────────────────────────────────────────────────
    atm_analysis = analyse_atmosphere(
        planet,
        star=star,
        orbital_distance_m=orbital_distance_m,
        bond_albedo=bond_albedo,
    )

    T_eq = atm_analysis.get(
        "equilibrium_temp_K",
        star.equilibrium_temperature(orbital_distance_m, bond_albedo),
    )
    dT_GH = atm_analysis.get("greenhouse_dT_K", 0.0)
    T_srf = atm_analysis.get("surface_temp_K", T_eq + dT_GH)
    P_srf = planet.atmosphere.surface_pressure if planet.atmosphere.enabled else 0.0

    # ── Score each factor ──────────────────────────────────────────────────────
    f_star_type = score_stellar_type(star)
    f_star_age = score_stellar_age(star)
    f_hz = score_habitable_zone(planet, star, orbital_distance_m)
    f_temp = score_surface_temperature(T_srf)
    f_water = score_liquid_water(T_srf, P_srf)
    f_retention = score_atmospheric_retention(planet, star, orbital_distance_m)
    f_magnetic = score_magnetic_protection(planet)
    f_tidal = score_tidal_locking(planet, star, orbital_distance_m)
    f_interior = score_interior_activity(planet)
    f_size = score_size_class(planet)

    factors = {
        "Stellar type": f_star_type,
        "Stellar age": f_star_age,
        "Habitable zone": f_hz,
        "Surface temperature": f_temp,
        "Liquid water": f_water,
        "Atm. retention": f_retention,
        "Magnetic shield": f_magnetic,
        "Tidal locking": f_tidal,
        "Interior activity": f_interior,
        "Planet size": f_size,
    }

    # ── Geometric mean overall score ───────────────────────────────────────────
    scores = [s for s, _ in factors.values()]
    # Geometric mean with a small floor (0.01) so that even a critical failure
    # doesn't zero out all differentiation. A planet with one factor at 0.01
    # and everything else at 1.0 scores 0.01^(1/10) ≈ 0.63 — still marked
    # as potentially habitable but clearly penalised.
    # A planet with multiple critical failures correctly scores near 0.
    FLOOR = 0.01
    clamped = [max(s, FLOOR) for s in scores]
    log_sum = sum(math.log(s) for s in clamped)
    overall = math.exp(log_sum / len(clamped))
    overall = max(0.0, min(1.0, overall))

    return HabitabilityAssessment(
        planet_name=planet.name,
        star_name=star.name,
        orbital_distance_au=d_au,
        factors=factors,
        overall_score=overall,
        surface_temp_K=T_srf,
        T_eq_K=T_eq,
        greenhouse_dT_K=dT_GH,
        size_class=size_class(planet),
        composition_class=composition_class(planet),
        atmosphere_analysis=atm_analysis,
    )
