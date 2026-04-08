"""
star.py — Stellar model for planetary science context.

Provides:
  - Stellar physical parameters (luminosity, temperature, radius, age)
  - Habitable zone boundaries (Kopparapu et al. 2013 empirical fits)
  - XUV / EUV flux (drives atmospheric escape)
  - Spectral classification
  - Tidal locking radius
  - Preset stars (Sun, Proxima Centauri, TRAPPIST-1, Tau Ceti, Kepler-452)

All SI units unless noted.

References:
  Kopparapu et al. 2013, ApJ 765, 131  — habitable zone limits
  Ribas et al. 2005 — XUV luminosity evolution for Sun-like stars
  Scalo et al. 2007 — UV environment of M-dwarfs
"""

from __future__ import annotations
import math
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

# ── Physical constants ────────────────────────────────────────────────────────
SIGMA_SB    = 5.670_374e-8   # W m⁻² K⁻⁴
L_SUN       = 3.828e26       # W — IAU nominal solar luminosity
R_SUN       = 6.957e8        # m
M_SUN       = 1.989e30       # kg
T_SUN       = 5_778.0        # K — effective surface temperature
G           = 6.674_30e-11   # m³ kg⁻¹ s⁻²
AU          = 1.495_978_707e11  # m — 1 astronomical unit


# ── Spectral classification ───────────────────────────────────────────────────
class SpectralType(Enum):
    O = auto()   # > 30 000 K  — hot, blue, very short-lived
    B = auto()   # 10–30 000 K
    A = auto()   # 7 500–10 000 K
    F = auto()   # 6 000–7 500 K
    G = auto()   # 5 200–6 000 K — Sun-like
    K = auto()   # 3 700–5 200 K
    M = auto()   # < 3 700 K   — red dwarf, most common
    L = auto()   # < 2 200 K   — brown dwarf boundary
    T = auto()   # < 1 300 K   — methane brown dwarf

    @classmethod
    def from_temperature(cls, T_eff: float) -> "SpectralType":
        """Classify by effective temperature."""
        if T_eff > 30_000: return cls.O
        if T_eff > 10_000: return cls.B
        if T_eff >  7_500: return cls.A
        if T_eff >  6_000: return cls.F
        if T_eff >  5_200: return cls.G
        if T_eff >  3_700: return cls.K
        if T_eff >  2_200: return cls.M
        if T_eff >  1_300: return cls.L
        return cls.T

    @property
    def habitable_fraction(self) -> float:
        """
        Qualitative fraction of this star type's planets likely to be
        in the habitable zone considering stellar activity and tidal locking.
        Not a rigorous probability — a useful summary.
        """
        return {
            SpectralType.O: 0.00,   # too hot, too short-lived
            SpectralType.B: 0.00,
            SpectralType.A: 0.02,   # UV hazard, short main sequence
            SpectralType.F: 0.10,   # reasonable but UV-active
            SpectralType.G: 0.25,   # Sun-like — best understood
            SpectralType.K: 0.35,   # long-lived, stable — actually best
            SpectralType.M: 0.12,   # HZ is tidally locked zone, flare hazard
            SpectralType.L: 0.00,
            SpectralType.T: 0.00,
        }[self]


# ── Habitable zone model (Kopparapu et al. 2013) ─────────────────────────────
# Empirical coefficients for HZ boundary flux S_eff [S⊙]
# S_eff = S_eff_sun + a·T* + b·T*² + c·T*³ + d·T*⁴
# where T* = T_eff - 5780 K

# Table 1 from Kopparapu+2013  (conservative HZ)
_HZ_COEFFS = {
    # limit         : (S_eff_sun,   a,          b,          c,          d)
    "runaway_greenhouse":  (1.0512, 1.3242e-4,  1.5418e-8, -7.9895e-12, -1.8328e-15),  # inner edge
    "maximum_greenhouse":  (0.3438, 5.8942e-5,  1.6558e-9, -3.0045e-12, -5.2983e-16),  # outer edge
    # optimistic extensions
    "recent_venus":        (1.7763, 1.4335e-4,  3.3954e-9, -7.6364e-12, -1.1950e-15),  # very inner
    "early_mars":          (0.3179, 5.4513e-5,  1.5313e-9, -2.7786e-12, -4.8997e-16),  # very outer
}

def _hz_flux(limit_key: str, T_eff: float) -> float:
    """Return S_eff [S⊙] for a given HZ boundary and stellar temperature."""
    S0, a, b, c, d = _HZ_COEFFS[limit_key]
    T = T_eff - 5_780.0
    return S0 + a*T + b*T**2 + c*T**3 + d*T**4

def _hz_distance(S_eff_sun: float, luminosity_W: float) -> float:
    """Convert S_eff [S⊙] to orbital distance [m]."""
    # S_eff = (L/L⊙) / d²[AU]²  → d[AU] = sqrt((L/L⊙) / S_eff)
    L_ratio = luminosity_W / L_SUN
    d_AU    = math.sqrt(L_ratio / S_eff_sun)
    return d_AU * AU


# ─────────────────────────────────────────────────────────────────────────────
# Star dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Star:
    """
    A star that provides the electromagnetic and gravitational context
    for planets orbiting it.

    Parameters
    ----------
    name         : catalogue name or label
    mass         : stellar mass [kg]
    luminosity   : bolometric luminosity [W]
    radius       : photospheric radius [m]
    temperature  : effective surface temperature [K]
    age          : stellar age [Gyr]
    metallicity  : [Fe/H] log-solar — affects planet formation
    """
    name:        str   = "Star"
    mass:        float = M_SUN
    luminosity:  float = L_SUN
    radius:      float = R_SUN
    temperature: float = T_SUN
    age:         float = 4.6      # Gyr
    metallicity: float = 0.0      # [Fe/H]

    # ── Derived scalar properties ─────────────────────────────────────────────
    @property
    def spectral_type(self) -> SpectralType:
        return SpectralType.from_temperature(self.temperature)

    @property
    def luminosity_solar(self) -> float:
        """Luminosity in units of L⊙."""
        return self.luminosity / L_SUN

    @property
    def mass_solar(self) -> float:
        """Mass in units of M⊙."""
        return self.mass / M_SUN

    @property
    def radius_solar(self) -> float:
        """Radius in units of R⊙."""
        return self.radius / R_SUN

    @property
    def mu(self) -> float:
        """Gravitational parameter [m³/s²]."""
        return G * self.mass

    # ── Main sequence lifetime ────────────────────────────────────────────────
    @property
    def main_sequence_lifetime_gyr(self) -> float:
        """
        Approximate main sequence lifetime [Gyr].
        t_MS ≈ 10 Gyr × (M/M⊙) / (L/L⊙)
        """
        return 10.0 * self.mass_solar / self.luminosity_solar

    @property
    def remaining_lifetime_gyr(self) -> float:
        """Remaining main sequence lifetime [Gyr]. Negative = already evolved off."""
        return self.main_sequence_lifetime_gyr - self.age

    # ── Flux at a given distance ───────────────────────────────────────────────
    def flux_at_distance(self, orbital_distance_m: float) -> float:
        """
        Bolometric stellar flux at orbital distance d [W/m²].
        S = L / (4π d²)
        Earth receives S⊙ = 1361 W/m².
        """
        if orbital_distance_m <= 0:
            return float("inf")
        return self.luminosity / (4 * math.pi * orbital_distance_m**2)

    def flux_solar_units(self, orbital_distance_m: float) -> float:
        """Flux in units of S⊙ (1 = Earth equivalent)."""
        return self.flux_at_distance(orbital_distance_m) / 1361.0

    # ── XUV / EUV flux ────────────────────────────────────────────────────────
    @property
    def xuv_luminosity(self) -> float:
        """
        X-ray + EUV luminosity [W] — drives photoionisation and atmospheric escape.

        For Sun-like stars, L_XUV / L_bol ≈ 10^(-3.5) at solar age.
        Young stars (< 0.1 Gyr) emit orders of magnitude more XUV.
        M-dwarfs emit proportionally more XUV and stay active for Gyr.

        Uses Ribas et al. (2005) age-activity relation for FGK stars,
        and Scalo et al. (2007) scaling for M-dwarfs.
        """
        T = self.spectral_type

        if T in (SpectralType.G, SpectralType.F, SpectralType.K):
            # Ribas et al. 2005: L_XUV ∝ t^(-1.23) for t > 0.1 Gyr
            t_sat  = 0.1    # Gyr — saturated phase before this
            L_sat  = self.luminosity * 1e-3   # L_XUV/L_bol ≈ 1e-3 at saturation
            if self.age <= t_sat:
                return L_sat
            return L_sat * (self.age / t_sat) ** (-1.23)

        elif T == SpectralType.M:
            # M-dwarfs: higher baseline, longer saturation phase (~1-3 Gyr)
            t_sat  = 2.0    # Gyr
            L_sat  = self.luminosity * 1e-2   # 1% XUV in saturation
            if self.age <= t_sat:
                return L_sat
            return L_sat * (self.age / t_sat) ** (-0.7)

        elif T in (SpectralType.A, SpectralType.B, SpectralType.O):
            # Hot stars: strong XUV from their birth
            return self.luminosity * 2e-4

        else:
            return self.luminosity * 1e-5   # cool / substellar

    def xuv_flux_at_distance(self, orbital_distance_m: float) -> float:
        """XUV flux at orbital distance [W/m²]."""
        if orbital_distance_m <= 0:
            return float("inf")
        return self.xuv_luminosity / (4 * math.pi * orbital_distance_m**2)

    # ── Habitable zone ────────────────────────────────────────────────────────
    @property
    def hz_inner_m(self) -> float:
        """
        Conservative habitable zone inner edge (runaway greenhouse) [m].
        Inside this distance, a planet like Earth would enter a Venus-like
        runaway greenhouse state.
        For the Sun: ~0.95 AU.
        """
        S_eff = _hz_flux("runaway_greenhouse", self.temperature)
        return _hz_distance(S_eff, self.luminosity)

    @property
    def hz_outer_m(self) -> float:
        """
        Conservative habitable zone outer edge (maximum greenhouse) [m].
        Beyond this, CO₂ clouds can no longer warm the surface above 273 K.
        For the Sun: ~1.67 AU.
        """
        S_eff = _hz_flux("maximum_greenhouse", self.temperature)
        return _hz_distance(S_eff, self.luminosity)

    @property
    def hz_inner_optimistic_m(self) -> float:
        """Optimistic HZ inner edge (recent Venus limit) [m]."""
        S_eff = _hz_flux("recent_venus", self.temperature)
        return _hz_distance(S_eff, self.luminosity)

    @property
    def hz_outer_optimistic_m(self) -> float:
        """Optimistic HZ outer edge (early Mars limit) [m]."""
        S_eff = _hz_flux("early_mars", self.temperature)
        return _hz_distance(S_eff, self.luminosity)

    @property
    def hz_inner_au(self) -> float:
        return self.hz_inner_m / AU

    @property
    def hz_outer_au(self) -> float:
        return self.hz_outer_m / AU

    def in_habitable_zone(self, orbital_distance_m: float,
                          conservative: bool = True) -> bool:
        """True if the given distance falls within the habitable zone."""
        if conservative:
            return self.hz_inner_m <= orbital_distance_m <= self.hz_outer_m
        else:
            return (self.hz_inner_optimistic_m <= orbital_distance_m
                    <= self.hz_outer_optimistic_m)

    def hz_fraction(self, orbital_distance_m: float) -> float:
        """
        Fractional position within the HZ (0 = inner edge, 1 = outer edge).
        Negative = inside inner edge (too hot).
        > 1     = outside outer edge (too cold).
        0.5     = middle of the HZ — ideal.
        """
        inner = self.hz_inner_m
        outer = self.hz_outer_m
        if outer <= inner:
            return 0.5
        return (orbital_distance_m - inner) / (outer - inner)

    # ── Tidal locking ─────────────────────────────────────────────────────────
    def tidal_locking_radius_m(self, planet_mass_kg: float,
                                planet_radius_m: float,
                                Q_factor: float = 100.0) -> float:
        """
        Orbital radius inside which a planet of given mass and radius
        would be tidally locked to this star within 4.5 Gyr [m].

        Uses Peale (1977) formula:
          a_lock ≈ 0.027 × (t × M_star / Q)^(1/6) × (m_p / M_star)^(1/6) × R_p^(3/6)

        Q_factor : tidal quality factor (~10 rocky, ~100 Earth, ~10⁵ gas giants)

        For the Sun: a_lock ≈ 0.1–0.2 AU for Earth-mass planets.
        For M-dwarfs: a_lock extends past the HZ for low-Q planets.
        """
        t_sec = 4.5e9 * 365.25 * 86400   # 4.5 Gyr in seconds
        # Simplified Peale formula (order-of-magnitude accurate)
        a = (0.027 * (t_sec * self.mass / Q_factor) ** (1/6)
             * (planet_mass_kg / self.mass) ** (1/6)
             * planet_radius_m ** (1/2))
        return a

    def is_tidally_locked(self, orbital_distance_m: float,
                           planet_mass_kg: float,
                           planet_radius_m: float,
                           Q_factor: float = 100.0) -> bool:
        """True if a planet at this distance would be tidally locked."""
        return orbital_distance_m < self.tidal_locking_radius_m(
            planet_mass_kg, planet_radius_m, Q_factor
        )

    # ── Orbital mechanics around this star ────────────────────────────────────
    def orbital_period(self, semi_major_axis_m: float) -> float:
        """Orbital period of a planet at given distance [s] (Kepler III)."""
        return 2 * math.pi * math.sqrt(semi_major_axis_m**3 / self.mu)

    def orbital_speed(self, semi_major_axis_m: float) -> float:
        """Mean orbital speed at given distance [m/s]."""
        return math.sqrt(self.mu / semi_major_axis_m)

    # ── Equilibrium temperature ────────────────────────────────────────────────
    def equilibrium_temperature(self, orbital_distance_m: float,
                                 bond_albedo: float = 0.3) -> float:
        """
        Planet equilibrium temperature assuming black-body re-radiation [K].
        T_eq = ((L(1-A)) / (16πσd²))^0.25

        This is the temperature before greenhouse warming.
        Earth: T_eq ≈ 255 K, actual surface ≈ 288 K (+33 K greenhouse).
        Venus: T_eq ≈ 226 K, actual surface ≈ 737 K (+511 K extreme greenhouse).
        """
        if orbital_distance_m <= 0:
            return float("inf")
        S = self.flux_at_distance(orbital_distance_m)
        return ((S * (1 - bond_albedo)) / (4 * SIGMA_SB)) ** 0.25

    # ── Summary ───────────────────────────────────────────────────────────────
    def summary(self) -> str:
        sp = self.spectral_type.name
        lines = [
            f"═══ {self.name} ═══",
            f"  Spectral type      : {sp}",
            f"  Mass               : {self.mass/M_SUN:.3f} M⊙  ({self.mass:.3e} kg)",
            f"  Luminosity         : {self.luminosity_solar:.4f} L⊙  ({self.luminosity:.3e} W)",
            f"  Radius             : {self.radius_solar:.3f} R⊙",
            f"  Effective temp     : {self.temperature:.0f} K",
            f"  Age                : {self.age:.1f} Gyr",
            f"  Metallicity        : {self.metallicity:+.2f} [Fe/H]",
            f"  MS lifetime        : {self.main_sequence_lifetime_gyr:.1f} Gyr",
            f"  XUV luminosity     : {self.xuv_luminosity:.3e} W",
            f"  HZ (conservative)  : {self.hz_inner_au:.2f} – {self.hz_outer_au:.2f} AU",
            f"  HZ (optimistic)    : {self.hz_inner_optimistic_m/AU:.2f} – "
                                    f"{self.hz_outer_optimistic_m/AU:.2f} AU",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Preset stars
# ─────────────────────────────────────────────────────────────────────────────
def star_sun() -> Star:
    """The Sun — G2V main sequence star."""
    return Star(
        name        = "Sun",
        mass        = M_SUN,
        luminosity  = L_SUN,
        radius      = R_SUN,
        temperature = 5_778,
        age         = 4.603,
        metallicity = 0.0,
    )

def star_proxima_centauri() -> Star:
    """Proxima Centauri — M5.5Ve flare star, host to Proxima b."""
    return Star(
        name        = "Proxima Centauri",
        mass        = 0.1221 * M_SUN,
        luminosity  = 0.001567 * L_SUN,
        radius      = 0.1542 * R_SUN,
        temperature = 3_042,
        age         = 4.85,
        metallicity = 0.21,
    )

def star_trappist1() -> Star:
    """TRAPPIST-1 — M8V ultra-cool dwarf, host to 7 known planets."""
    return Star(
        name        = "TRAPPIST-1",
        mass        = 0.0898 * M_SUN,
        luminosity  = 0.000553 * L_SUN,
        radius      = 0.1192 * R_SUN,
        temperature = 2_566,
        age         = 7.6,
        metallicity = 0.04,
    )

def star_tau_ceti() -> Star:
    """Tau Ceti — G8.5V nearby Sun-analogue with debris disk."""
    return Star(
        name        = "Tau Ceti",
        mass        = 0.783 * M_SUN,
        luminosity  = 0.488 * L_SUN,
        radius      = 0.793 * R_SUN,
        temperature = 5_344,
        age         = 5.8,
        metallicity = -0.52,
    )

def star_kepler452() -> Star:
    """Kepler-452 — G2V near-Sun-twin, host to Kepler-452b (Earth cousin)."""
    return Star(
        name        = "Kepler-452",
        mass        = 1.037 * M_SUN,
        luminosity  = 1.200 * L_SUN,
        radius      = 1.110 * R_SUN,
        temperature = 5_757,
        age         = 6.0,
        metallicity = 0.21,
    )

def star_alpha_centauri_a() -> Star:
    """Alpha Centauri A — G2V, brightest in nearest stellar system."""
    return Star(
        name        = "Alpha Centauri A",
        mass        = 1.100 * M_SUN,
        luminosity  = 1.519 * L_SUN,
        radius      = 1.227 * R_SUN,
        temperature = 5_790,
        age         = 5.3,
        metallicity = 0.20,
    )

def star_eps_eridani() -> Star:
    """Epsilon Eridani — K2V young active star with debris disk."""
    return Star(
        name        = "Epsilon Eridani",
        mass        = 0.770 * M_SUN,
        luminosity  = 0.340 * L_SUN,
        radius      = 0.735 * R_SUN,
        temperature = 5_084,
        age         = 0.8,
        metallicity = -0.09,
    )


# Convenience dict for programmatic access
STAR_PRESETS: dict[str, callable] = {
    "sun":               star_sun,
    "proxima":           star_proxima_centauri,
    "trappist1":         star_trappist1,
    "tau_ceti":          star_tau_ceti,
    "kepler452":         star_kepler452,
    "alpha_centauri_a":  star_alpha_centauri_a,
    "eps_eridani":       star_eps_eridani,
}
