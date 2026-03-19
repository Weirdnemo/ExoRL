"""
observation.py — Observational signatures of planets.

Answers the questions that connect the planetary model to real telescopes:
  - What is the transit depth of this planet across its star?
  - What radial velocity signal does it induce on the star?
  - With a 10m telescope, what S/N does a transit observation achieve?
  - What does the transmission spectrum look like?
  - How many transits per year are observable?

These are the quantities that exoplanet surveys actually report.
Every planet in the model now has an "observable fingerprint".

All SI unless noted.

References:
  Seager & Mallén-Ornelas (2003) — transit geometry
  Lovis & Fischer (2010) — radial velocity technique
  Winn (2010) "Transits and Occultations" in Exoplanets (Seager ed.)
  Tinetti et al. (2007) — transmission spectroscopy
  Kempton et al. (2018) — Transmission Spectroscopy Metric (TSM)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
G       = 6.674_30e-11
SIGMA   = 5.670_374e-8
AU      = 1.495_978_707e11
R_SUN   = 6.957e8
M_SUN   = 1.989e30
R_EARTH = 6.371e6
M_EARTH = 5.972e24
k_B     = 1.380_649e-23
h_PLANCK = 6.626_070e-34
c_LIGHT  = 2.997_924e8
N_AVO   = 6.022_141e23

# ── Stellar flux (Johnson V-band) reference ───────────────────────────────────
# Used for photon noise calculation
FLUX_VEGA_V_PHOT = 3.64e-9   # W m⁻² μm⁻¹  (Vega at V-band)


# ─────────────────────────────────────────────────────────────────────────────
# Transit geometry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitSignal:
    """Complete transit observable for a planet-star system."""
    planet_name:          str
    star_name:            str
    orbital_distance_au:  float

    # Geometry
    transit_depth:        float    # (R_p/R_*)²  [dimensionless, ppm × 1e-6]
    transit_depth_ppm:    float    # transit_depth × 1e6  [ppm]
    impact_parameter:     float    # b = (a/R_*) cos(i)  [stellar radii]
    transit_duration_hr:  float    # T_14 [hours]
    ingress_duration_min: float    # T_12 [minutes]
    transit_prob:         float    # geometric probability of transit [0–1]
    transits_per_year:    float    # number of transit opportunities per year

    # Photometry
    snr_per_transit:      float    # S/N for one transit event
    snr_10_transits:      float    # S/N combining 10 transits

    # Radial velocity
    rv_semi_amplitude_m_s: float   # K [m/s] RV semi-amplitude

    # Detectability
    tsm:                  float    # Transmission Spectroscopy Metric (Kempton+2018)
    detectable_jwst:      bool     # TSM > 10 (conservative JWST threshold)
    detectable_ground:    bool     # approx. detectable from ground

    @property
    def orbital_distance_m(self) -> float:
        return self.orbital_distance_au * AU

    def report(self) -> str:
        detect_s = []
        if self.detectable_jwst:
            detect_s.append("JWST atmosphere study")
        if self.detectable_ground:
            detect_s.append("ground-based transit")
        detect_str = " + ".join(detect_s) if detect_s else "challenging with current instruments"

        return (
            f"═══ Transit signal: {self.planet_name} / {self.star_name} ═══\n"
            f"\n"
            f"  ── Transit geometry ──\n"
            f"  Transit depth          : {self.transit_depth_ppm:.1f} ppm  "
                f"({self.transit_depth_ppm/10000:.4f}%)\n"
            f"  Transit duration       : {self.transit_duration_hr:.2f} hr\n"
            f"  Ingress/egress         : {self.ingress_duration_min:.1f} min\n"
            f"  Geometric transit prob : {self.transit_prob*100:.2f}%\n"
            f"  Transits per year      : {self.transits_per_year:.1f}\n"
            f"\n"
            f"  ── Photometric S/N (2m telescope, V-band) ──\n"
            f"  S/N per transit        : {self.snr_per_transit:.1f}\n"
            f"  S/N (10 transits)      : {self.snr_10_transits:.1f}\n"
            f"\n"
            f"  ── Radial velocity ──\n"
            f"  RV semi-amplitude K    : {self.rv_semi_amplitude_m_s:.2f} m/s\n"
            f"  (Earth: 0.09 m/s; Hot Jupiter: ~100 m/s)\n"
            f"\n"
            f"  ── Spectroscopy ──\n"
            f"  TSM (Kempton 2018)     : {self.tsm:.1f}\n"
            f"  Detectable for         : {detect_str}\n"
        )


# ── Transit depth ─────────────────────────────────────────────────────────────

def transit_depth(planet_radius_m: float, star_radius_m: float) -> float:
    """
    Transit depth δ = (R_p / R_*)² [dimensionless].

    This is the fractional decrease in stellar flux during a central transit.
    Earth across Sun: (6.371e6 / 6.957e8)² = 8.4×10⁻⁵ = 84 ppm.
    Hot Jupiter (R=1.3 R_J) across Sun: ~1.5%.

    Note: actual observed depth depends on limb darkening, impact parameter,
    and stellar contamination. This is the geometric (first-order) depth.
    """
    return (planet_radius_m / star_radius_m) ** 2


def transit_depth_ppm(planet_radius_m: float, star_radius_m: float) -> float:
    """Transit depth in parts per million [ppm]."""
    return transit_depth(planet_radius_m, star_radius_m) * 1e6


# ── Transit duration ──────────────────────────────────────────────────────────

def transit_duration(planet_radius_m: float,
                      star_radius_m: float,
                      orbital_distance_m: float,
                      orbital_period_s: float,
                      impact_parameter: float = 0.0) -> float:
    """
    Total transit duration T_14 (first to fourth contact) [seconds].

    T_14 = (P / π) × arcsin( R_* / a × sqrt((1 + k)² - b²) )

    where k = R_p/R_* and b is the impact parameter.
    Valid for b < 1 - k (partial/total transits). Returns 0 for grazing/no transit.

    For a central transit (b=0): T_14 ≈ (2 R_* / v_orb) × sqrt(1 + k²)
    Earth across Sun: ~13 hours ✓
    """
    k    = planet_radius_m / star_radius_m
    arg  = (1.0 + k)**2 - impact_parameter**2
    if arg < 0:
        return 0.0   # no transit
    sin_arg = (star_radius_m / orbital_distance_m) * math.sqrt(arg)
    sin_arg = min(1.0, sin_arg)
    return (orbital_period_s / math.pi) * math.asin(sin_arg)


def transit_ingress_duration(planet_radius_m: float,
                               star_radius_m: float,
                               orbital_distance_m: float,
                               orbital_period_s: float,
                               impact_parameter: float = 0.0) -> float:
    """
    Transit ingress/egress duration T_12 (first to second contact) [seconds].

    T_12 = (P / π) × arcsin( R_* / a × sqrt((1 - k)² - b²) ) × ... correction
    Simplified: T_12 ≈ T_14 × k / sqrt(1 + k)
    """
    k   = planet_radius_m / star_radius_m
    T14 = transit_duration(planet_radius_m, star_radius_m,
                            orbital_distance_m, orbital_period_s, impact_parameter)
    if T14 == 0:
        return 0.0
    return T14 * k / (1.0 + k)


# ── Geometric transit probability ─────────────────────────────────────────────

def geometric_transit_probability(star_radius_m: float,
                                    orbital_distance_m: float,
                                    eccentricity: float = 0.0) -> float:
    """
    Probability that a randomly oriented orbit shows a transit [0–1].

    p = (R_* + R_p) / a × (1 + e_p sin ω) / (1 - e²)

    For circular orbits (e=0): p = R_* / a  (approximately, ignoring R_p)
    Earth: p = R_⊙ / 1 AU = 6.957e8 / 1.496e11 = 0.46%

    With eccentricity: high-eccentricity orbits have enhanced probability
    during the close approach phase.
    """
    r_eff = star_radius_m  # approximate: ignore R_p for geometric probability
    ecc_factor = (1.0 + eccentricity) / (1.0 - eccentricity**2)
    return (r_eff / orbital_distance_m) * ecc_factor


# ─────────────────────────────────────────────────────────────────────────────
# Radial velocity
# ─────────────────────────────────────────────────────────────────────────────

def rv_semi_amplitude(planet_mass_kg: float,
                       star_mass_kg: float,
                       orbital_period_s: float,
                       eccentricity: float = 0.0,
                       inclination_rad: float = math.pi / 2) -> float:
    """
    Radial velocity semi-amplitude K [m/s] induced by the planet on the star.

    K = (2π G / P)^(1/3) × (M_p sin i) / (M_* + M_p)^(2/3) × 1/sqrt(1-e²)

    Examples (circular orbits, edge-on):
      Earth:       K = 0.089 m/s   (requires ESPRESSO-class spectrograph)
      Venus:       K = 0.086 m/s
      Jupiter:     K = 12.5  m/s   (easily detectable)
      51 Peg b:    K = 55.9  m/s   (first exoplanet RV detection)
      Hot Jupiter: K ~ 50–200 m/s
    """
    if orbital_period_s <= 0:
        return 0.0
    factor = (2 * math.pi * G / orbital_period_s) ** (1/3)
    mp_sini = planet_mass_kg * math.sin(inclination_rad)
    M_tot = star_mass_kg + planet_mass_kg
    K = factor * mp_sini / M_tot**(2/3) / math.sqrt(1 - eccentricity**2)
    return K


def minimum_detectable_mass_kg(K_rms_m_s: float,
                                 star_mass_kg: float,
                                 orbital_period_s: float) -> float:
    """
    Minimum detectable planet mass [kg] for a given RV precision.

    Inverts rv_semi_amplitude() assuming edge-on orbit, circular.
    K_rms_m_s: instrument precision in m/s (e.g., HARPS = 1 m/s, ESPRESSO = 0.1 m/s)
    """
    factor = (2 * math.pi * G / orbital_period_s) ** (1/3)
    return K_rms_m_s * star_mass_kg**(2/3) / factor


# ─────────────────────────────────────────────────────────────────────────────
# Photometric S/N
# ─────────────────────────────────────────────────────────────────────────────

def photon_noise_floor_ppm(star_apparent_magnitude: float,
                             telescope_aperture_m: float,
                             exposure_time_s: float,
                             bandpass_nm: float = 100.0,
                             efficiency: float = 0.30) -> float:
    """
    Photon noise limit for transit photometry [ppm].

    σ_phot = 1 / sqrt(N_phot)

    where N_phot is the number of detected photons during the exposure.
    Uses a V-band reference flux and scales by the star's magnitude.

    Parameters
    ----------
    star_apparent_magnitude : V-band apparent magnitude of the star
    telescope_aperture_m    : primary mirror diameter [m]
    exposure_time_s         : exposure time [s]
    bandpass_nm             : photometric bandpass [nm]
    efficiency              : total throughput (optics + detector + atmosphere)

    Returns
    -------
    1-σ photon noise per data point [ppm]
    """
    # Reference: Vega at V=0 has ~3640 Jy = 3.64e-9 W/m²/μm
    # Convert to photon flux: N = F × A × t × Δλ / (hc/λ)
    lambda_V  = 550e-9    # m — centre of V band
    # FLUX_VEGA_V_PHOT is W m⁻² μm⁻¹; bandpass must be in μm (not nm)
    F0_phot   = FLUX_VEGA_V_PHOT * bandpass_nm * 1e-3  # W/m² (bandpass_nm×1e-3 = μm)
    F_star    = F0_phot * 10 ** (-star_apparent_magnitude / 2.5)

    # Collecting area
    A = math.pi * (telescope_aperture_m / 2) ** 2

    # Photon energy at V-band
    E_photon = h_PLANCK * c_LIGHT / lambda_V

    N_photons = F_star * A * exposure_time_s * efficiency / E_photon
    if N_photons <= 0:
        return float("inf")
    return 1e6 / math.sqrt(N_photons)   # ppm


def transit_snr(transit_depth_ppm: float,
                 star_magnitude_V: float,
                 transit_duration_hr: float,
                 telescope_aperture_m: float = 2.0,
                 exposure_time_s: float = 30.0,
                 read_noise_ppm: float = 50.0,
                 systematic_floor_ppm: float = 10.0) -> float:
    """
    Signal-to-noise ratio for one transit observation.

    S/N = δ / σ_total
    where δ is the transit depth and σ_total combines:
      - photon noise (σ_phot)
      - read noise
      - systematic floor (instrumental + atmospheric)

    Observing strategy: bin all exposures within the transit window.
    """
    if transit_duration_hr <= 0:
        return 0.0
    T_transit_s = transit_duration_hr * 3600

    # Number of exposures in transit
    n_exp = max(1, int(T_transit_s / exposure_time_s))

    # Noise per exposure [ppm]
    sigma_per_exp = math.sqrt(
        photon_noise_floor_ppm(star_magnitude_V, telescope_aperture_m, exposure_time_s)**2
        + read_noise_ppm**2
    )

    # Combined noise for all in-transit exposures (sqrt(N) improvement)
    sigma_transit = math.sqrt(sigma_per_exp**2 / n_exp + systematic_floor_ppm**2)

    if sigma_transit <= 0:
        return float("inf")
    return transit_depth_ppm / sigma_transit


# ─────────────────────────────────────────────────────────────────────────────
# Transmission spectroscopy
# ─────────────────────────────────────────────────────────────────────────────

def atmospheric_scale_height(mean_molar_mass_kg_mol: float,
                               surface_gravity_m_s2: float,
                               equilibrium_temperature_K: float) -> float:
    """
    Atmospheric scale height H = k_B T / (μ g) [m].

    Determines the height of each spectral feature in transmission.
    Large H → strong spectral features → easier to detect.
    """
    return k_B * equilibrium_temperature_K / (
        mean_molar_mass_kg_mol / N_AVO * surface_gravity_m_s2
    )


def transmission_spectral_feature_depth_ppm(planet_radius_m: float,
                                              star_radius_m: float,
                                              scale_height_m: float,
                                              n_scale_heights: float = 5.0) -> float:
    """
    Amplitude of spectral absorption features in transmission [ppm].

    Each molecular absorption feature spans ~N_H scale heights in the
    transmission spectrum. The feature depth is:

        Δδ = 2 R_p N_H H / R_*²

    where H is the scale height and N_H ≈ 5 (empirical).

    Earth: H=8.5 km, N_H=5 → Δδ = 2×6371×5×8.5 / (695700)² ≈ 0.9 ppm
    (This matches the ~1 ppm features observed for Earth by lunar eclipse)
    """
    delta = 2 * planet_radius_m * n_scale_heights * scale_height_m / star_radius_m**2
    return delta * 1e6   # ppm


def transmission_spectroscopy_metric(planet_radius_m: float,
                                      star_radius_m: float,
                                      planet_mass_kg: float,
                                      equilibrium_temperature_K: float,
                                      star_magnitude_J: float,
                                      mean_molar_mass_g_mol: float = 18.0  # kept for API compat
                                      ) -> float:
    """
    Transmission Spectroscopy Metric (TSM) from Kempton et al. (2018).

    A figure of merit predicting the S/N achievable with JWST NIRISS for
    atmospheric characterisation via transmission spectroscopy.

    TSM = C × (R_p³ T_eq) / (M_p R_*²) × 10^(-m_J/5)

    where C is a size-dependent scale factor (Table 1 of Kempton+2018).

    Rules of thumb:
      TSM > 90    : excellent (top priority for atmosphere study)
      TSM > 10    : good (JWST can characterise in ~10 transits)
      TSM < 10    : poor (marginal or not feasible for JWST)
      TSM < 1     : not feasible

    Note: TSM is only defined for planets < 4 R⊕ (below the giant planet boundary).
    """
    R_E = R_EARTH
    M_E = M_EARTH
    R_p_earth = planet_radius_m / R_E
    M_p_earth = planet_mass_kg  / M_E

    # Scale factor by size class (Kempton+2018 Table 1)
    if R_p_earth < 1.5:
        C = 0.190
    elif R_p_earth < 2.75:
        C = 1.26
    elif R_p_earth < 4.0:
        C = 1.28
    else:
        C = 1.15   # sub-Neptunes/giants — less well calibrated

    # Standard Kempton+2018 TSM formula (no μ correction — C already encodes
    # the typical scale height for each size class)
    R_s_sun = star_radius_m / R_SUN

    tsm = (C * R_p_earth**3 * equilibrium_temperature_K
           / (M_p_earth * R_s_sun**2)
           * 10**(-star_magnitude_J / 5.0))
    return max(0.0, tsm)


# ─────────────────────────────────────────────────────────────────────────────
# Transmission spectrum (simplified line-list)
# ─────────────────────────────────────────────────────────────────────────────

# Approximate wavelengths [μm] of key molecular features accessible to JWST
SPECTRAL_FEATURES = {
    "H2O":  [1.15, 1.38, 1.87, 2.70, 3.20, 5.90, 6.70],    # water vapour
    "CO2":  [1.05, 1.21, 1.43, 1.60, 2.01, 2.69, 4.27],    # carbon dioxide
    "CH4":  [1.67, 2.31, 3.31, 3.83, 7.66],                 # methane
    "O3":   [0.55, 0.60, 0.61, 0.71, 9.60],                 # ozone (Chappuis)
    "O2":   [0.69, 0.76, 1.27],                             # oxygen A and B bands
    "N2O":  [1.28, 2.87, 3.90, 7.78, 8.58, 17.0],          # nitrous oxide
    "NH3":  [1.50, 2.00, 2.96, 6.14, 10.3],                 # ammonia (Titan)
    "SO2":  [4.00, 7.35, 8.66],                             # sulphur dioxide
    "H2":   [0.83, 1.24, 2.12],                             # molecular hydrogen
    "CO":   [1.57, 2.35, 4.67],                             # carbon monoxide
}

# Biosignature potential (Oxford 2018 classification)
BIOSIGNATURE_POTENTIAL = {
    "O2":  "strong — photosynthesis byproduct",
    "O3":  "strong — photolysis of O2 (proxy for O2)",
    "CH4": "moderate — biological AND geological",
    "N2O": "moderate — biological (denitrification)",
    "H2O": "neutral — essential but not diagnostic",
    "CO2": "neutral — geological and biological",
    "NH3": "weak — can be abiotic (Titan, gas giant)",
}


def transmission_spectrum(planet, star=None,
                            orbital_distance_m: float = None
                            ) -> dict:
    """
    Predict the transmission spectrum features for a planet.

    Returns a dict of molecular features visible in the atmosphere,
    with the estimated signal strength [ppm] for each major band.

    Uses the planet's atmosphere composition to select which molecules
    are present, then computes the feature depth from scale height.
    """
    from core.atmosphere_science import STANDARD_COMPOSITIONS, MOLAR_MASS
    import math as _math

    if star is None and hasattr(planet, "star_context"):
        star = planet.star_context
    if orbital_distance_m is None and hasattr(planet, "orbital_distance_m"):
        orbital_distance_m = planet.orbital_distance_m

    if not planet.atmosphere.enabled:
        return {"enabled": False, "features": {}}

    comp_name   = planet.atmosphere.composition.name
    composition = dict(STANDARD_COMPOSITIONS.get(comp_name, {"N2": 1.0}))
    total       = sum(composition.values())
    if total > 0:
        composition = {k: v/total for k, v in composition.items()}

    # Mean molar mass
    mm_g_mol  = sum(MOLAR_MASS.get(sp, 28.0) * f
                    for sp, f in composition.items())
    mm_kg_mol = mm_g_mol * 1e-3

    # Gravity and equilibrium temperature
    g = planet.surface_gravity
    if star and orbital_distance_m:
        T_eq = star.equilibrium_temperature(orbital_distance_m, 0.3)
    else:
        T_eq = planet.atmosphere.surface_temp * 0.85

    H   = atmospheric_scale_height(mm_kg_mol, g, T_eq)
    R_p = planet.radius
    R_s = star.radius if star else R_SUN

    # Feature depth in ppm
    depth_per_feature = transmission_spectral_feature_depth_ppm(R_p, R_s, H)

    # Which features are present?
    detected = {}
    for molecule, wavelengths in SPECTRAL_FEATURES.items():
        frac = composition.get(molecule, 0.0)
        if frac < 1e-7:
            continue   # trace amount — undetectable
        # Feature strength scales roughly as sqrt of mole fraction
        strength = depth_per_feature * _math.sqrt(frac)
        bio_note = BIOSIGNATURE_POTENTIAL.get(molecule, "")
        detected[molecule] = {
            "mole_fraction":     frac,
            "feature_depth_ppm": strength,
            "wavelengths_um":    wavelengths,
            "biosignature":      bio_note,
        }

    return {
        "enabled":          True,
        "composition":      composition,
        "scale_height_m":   H,
        "scale_height_km":  H / 1e3,
        "mean_molar_mass":  mm_g_mol,
        "T_eq_K":           T_eq,
        "depth_per_H_ppm":  depth_per_feature,
        "features":         detected,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point: full observational characterisation
# ─────────────────────────────────────────────────────────────────────────────

def characterise_observations(planet,
                                star=None,
                                orbital_distance_m: float = None,
                                eccentricity: float = 0.0,
                                inclination_deg: float = 90.0,
                                star_magnitude_V: float = None,
                                star_magnitude_J: float = None,
                                telescope_aperture_m: float = 2.0
                                ) -> TransitSignal:
    """
    Compute the full observational signature of a planet.

    Parameters
    ----------
    planet               : Planet object
    star                 : Star object (uses planet.star_context if None)
    orbital_distance_m   : orbital distance [m]
    eccentricity         : orbital eccentricity
    inclination_deg      : orbital inclination [°] — 90° = edge-on = maximum transit prob
    star_magnitude_V     : apparent V-band magnitude (estimated if not given)
    star_magnitude_J     : apparent J-band magnitude for TSM (estimated if not given)
    telescope_aperture_m : telescope diameter [m] for S/N calculation

    Returns
    -------
    TransitSignal with all observable quantities.
    """
    if star is None and hasattr(planet, "star_context"):
        star = planet.star_context
    if orbital_distance_m is None and hasattr(planet, "orbital_distance_m"):
        orbital_distance_m = planet.orbital_distance_m
    if star is None:
        raise ValueError("Provide star or set planet.star_context")
    if orbital_distance_m is None:
        raise ValueError("Provide orbital_distance_m or set planet.orbital_distance_m")

    d_AU = orbital_distance_m / AU
    inc  = math.radians(inclination_deg)

    # Orbital period
    mu_star = G * star.mass
    T_orb   = 2 * math.pi * math.sqrt(orbital_distance_m**3 / mu_star)

    # Transit geometry
    b   = (orbital_distance_m / star.radius) * math.cos(inc)
    k   = planet.radius / star.radius
    depth = transit_depth(planet.radius, star.radius)
    depth_ppm = depth * 1e6

    dur_s    = transit_duration(planet.radius, star.radius,
                                 orbital_distance_m, T_orb, b)
    ingress_s = transit_ingress_duration(planet.radius, star.radius,
                                          orbital_distance_m, T_orb, b)
    prob      = geometric_transit_probability(star.radius, orbital_distance_m, eccentricity)
    n_transits_yr = (365.25 * 86400 / T_orb) * prob

    # Estimate star magnitude from luminosity and distance
    if star_magnitude_V is None:
        # Assume star is at 10 pc for comparison; scale L and T for color
        # V-band luminosity proxy from T_eff and L
        L_ratio = star.luminosity / 3.828e26
        star_magnitude_V = 4.83 - 2.5 * math.log10(max(L_ratio, 1e-10))
        # Add colour correction for cool/hot stars
        if star.temperature < 4000:
            star_magnitude_V += 1.5   # M-dwarfs are fainter in V
        elif star.temperature < 5000:
            star_magnitude_V += 0.5

    if star_magnitude_J is None:
        # J-band approximately 1 mag brighter for K/M dwarfs, same for G
        if star.temperature < 4500:
            star_magnitude_J = star_magnitude_V - 2.0
        else:
            star_magnitude_J = star_magnitude_V - 0.5

    # S/N
    snr1  = transit_snr(depth_ppm, star_magnitude_V, dur_s / 3600,
                         telescope_aperture_m)
    snr10 = snr1 * math.sqrt(10)

    # RV semi-amplitude
    K = rv_semi_amplitude(planet.mass, star.mass, T_orb, eccentricity, inc)

    # Equilibrium temperature
    T_eq = star.equilibrium_temperature(orbital_distance_m, 0.3)

    # TSM — mean molar mass from atmosphere
    from core.atmosphere_science import STANDARD_COMPOSITIONS, MOLAR_MASS
    comp_name  = planet.atmosphere.composition.name if planet.atmosphere.enabled else "NONE"
    comp       = STANDARD_COMPOSITIONS.get(comp_name, {"N2": 1.0})
    total      = sum(comp.values())
    mm_g_mol   = (sum(MOLAR_MASS.get(sp, 28.0) * f for sp, f in comp.items()) / total
                  if total > 0 else 28.0)

    tsm = transmission_spectroscopy_metric(
        planet.radius, star.radius, planet.mass, T_eq,
        star_magnitude_J, mean_molar_mass_g_mol=mm_g_mol
    )

    # Detectability thresholds
    detectable_jwst   = tsm > 10 and depth_ppm > 0.5
    detectable_ground = depth_ppm > 1000 and snr1 > 5   # ~0.1% depth and S/N>5

    return TransitSignal(
        planet_name           = planet.name,
        star_name             = star.name,
        orbital_distance_au   = d_AU,
        transit_depth         = depth,
        transit_depth_ppm     = depth_ppm,
        impact_parameter      = b,
        transit_duration_hr   = dur_s / 3600,
        ingress_duration_min  = ingress_s / 60,
        transit_prob          = prob,
        transits_per_year     = n_transits_yr,
        snr_per_transit       = snr1,
        snr_10_transits       = snr10,
        rv_semi_amplitude_m_s = K,
        tsm                   = tsm,
        detectable_jwst       = detectable_jwst,
        detectable_ground     = detectable_ground,
    )
