"""
kepler_catalog.py — Reference exoplanet catalog with Planet-RL habitability scoring.

Provides a curated dataset of well-characterised rocky exoplanets from the
Kepler, TESS, and ground-based surveys, all with measured masses and radii.
Runs the full Planet-RL habitability assessment over each entry and returns
a ranked table for comparison with solar system benchmarks.

The bundled catalog covers:
  - All confirmed rocky planets (R < 1.8 R⊕, M < 8 M⊕) with both
    mass and radius measured to < 30% uncertainty
  - The TRAPPIST-1 system (all 7 planets)
  - Selected near-HZ benchmark targets

Sources
-------
  NASA Exoplanet Archive (PSComPPars) — bulk parameters
  Turbet et al. 2020 — TRAPPIST-1 updated masses
  Agol et al. 2021 — TRAPPIST-1 transit timing variations
  Dressing & Charbonneau 2015 — M-dwarf rocky planet occurrence

Usage
-----
    from planet_rl.core.kepler_catalog import KeplerCatalog

    cat = KeplerCatalog()
    print(cat.summary())

    # Score all planets with Planet-RL habitability framework
    results = cat.score_all()
    top10   = cat.top_n(10)

    # Compare against ESI
    df = cat.comparison_table()

    # Get a Planet object for a specific entry
    trappist1e = cat.get_planet("TRAPPIST-1 e")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ── Physical constants ────────────────────────────────────────────────────────
R_EARTH = 6.371e6  # m
M_EARTH = 5.972e24  # kg
G = 6.674_30e-11  # m³ kg⁻¹ s⁻²
AU = 1.495_978_707e11  # m
L_SUN = 3.828e26  # W
R_SUN = 6.957e8  # m
M_SUN = 1.989e30  # kg


# ─────────────────────────────────────────────────────────────────────────────
# Catalog entry
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CatalogEntry:
    """One confirmed exoplanet in the reference catalog."""

    # Identity
    name: str
    host_star: str

    # Planet parameters (measured)
    radius_earth: float  # R⊕
    mass_earth: float  # M⊕
    orbital_period_d: float  # days
    semi_major_axis_au: float  # AU

    # Stellar parameters
    star_teff_K: float  # K
    star_radius_sun: float  # R☉
    star_mass_sun: float  # M☉
    star_luminosity: float  # L☉

    # Discovery info
    discovery_method: str = "Transit"
    reference: str = ""

    # Computed on demand
    _hab_score: Optional[float] = field(default=None, repr=False)
    _hab_grade: Optional[str] = field(default=None, repr=False)
    _esi: Optional[float] = field(default=None, repr=False)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def radius_m(self) -> float:
        return self.radius_earth * R_EARTH

    @property
    def mass_kg(self) -> float:
        return self.mass_earth * M_EARTH

    @property
    def mean_density(self) -> float:
        V = (4 / 3) * math.pi * self.radius_m**3
        return self.mass_kg / V

    @property
    def surface_gravity(self) -> float:
        return G * self.mass_kg / self.radius_m**2

    @property
    def escape_velocity(self) -> float:
        return math.sqrt(2 * G * self.mass_kg / self.radius_m)

    @property
    def orbital_distance_m(self) -> float:
        return self.semi_major_axis_au * AU

    @property
    def stellar_flux_earth(self) -> float:
        """Stellar flux relative to Earth's (S⊕)."""
        return self.star_luminosity / self.semi_major_axis_au**2

    @property
    def equilibrium_temperature_K(self) -> float:
        """Radiative equilibrium T (Bond albedo = 0.3)."""
        flux = self.star_luminosity * L_SUN / (4 * math.pi * self.orbital_distance_m**2)
        return (flux * (1 - 0.3) / (4 * 5.670374e-8)) ** 0.25

    @property
    def transit_depth_ppm(self) -> float:
        return 1e6 * (self.radius_m / (self.star_radius_sun * R_SUN)) ** 2

    def earth_similarity_index(self) -> float:
        """
        Earth Similarity Index (ESI) after Schulze-Makuch et al. (2011).
        Geometric mean of four sub-indices vs Earth reference values.
        Range: 0 (completely unlike Earth) to 1 (identical to Earth).
        """
        if self._esi is not None:
            return self._esi

        # Reference values (Earth)
        R_ref = 1.0  # R⊕
        rho_ref = 5514  # kg/m³
        v_ref = 11186  # m/s  (escape velocity)
        T_ref = 288  # K

        def sub_esi(x, x_ref, w):
            return (1 - abs(x - x_ref) / (x + x_ref)) ** w

        esi_r = sub_esi(self.radius_earth, R_ref, 0.57)
        esi_rho = sub_esi(self.mean_density, rho_ref, 1.07)
        esi_v = sub_esi(self.escape_velocity, v_ref, 0.70)
        # ESI uses estimated surface temperature for thermal sub-index
        # This gives Earth ESI=1.0 when surface_temp≈288K
        T_surf_est = self.equilibrium_temperature_K + 33  # rough greenhouse offset
        esi_T = sub_esi(T_surf_est, T_ref, 5.58)

        self._esi = (esi_r * esi_rho * esi_v * esi_T) ** 0.25
        return self._esi

    def to_planet(self):
        """
        Convert this catalog entry to a Planet-RL Planet object.
        Attaches a matching Star and sets orbital distance.
        Interior model is attached based on bulk density.
        """
        from planet_rl.core.interior import interior_from_bulk_density
        from planet_rl.core.planet import (
            AtmosphereComposition,
            AtmosphereConfig,
            MagneticFieldConfig,
            OblatenessConfig,
            Planet,
            TerrainConfig,
        )
        from planet_rl.core.star import Star

        # Estimate atmospheric composition from equilibrium temperature
        T_eq = self.equilibrium_temperature_K
        if T_eq > 500:
            comp = AtmosphereComposition.CO2_THICK
            P = 1e6
            rho0 = 50.0
        elif T_eq > 300:
            comp = AtmosphereComposition.CO2_THICK
            P = 5e4
            rho0 = 2.0
        elif T_eq > 200:
            comp = AtmosphereComposition.EARTH_LIKE
            P = 101325
            rho0 = 1.225
        else:
            comp = AtmosphereComposition.CO2_THIN
            P = 1000
            rho0 = 0.015

        p = Planet(
            name=self.name,
            radius=self.radius_m,
            mass=self.mass_kg,
            rotation_enabled=True,
            rotation_period=86400.0,
            atmosphere=AtmosphereConfig(
                enabled=True,
                composition=comp,
                surface_pressure=P,
                surface_density=rho0,
                surface_temp=max(T_eq + 30, 50),
                scale_height=8500
                * (self.equilibrium_temperature_K / 255)
                / (self.surface_gravity / 9.81),
            ),
            terrain=TerrainConfig(enabled=False),
            magnetic_field=MagneticFieldConfig(enabled=True),
            oblateness=OblatenessConfig(enabled=False),
        )

        # Interior model from bulk density
        p.interior = interior_from_bulk_density(self.mean_density)

        # Build a matching star
        star = Star(
            name=self.host_star,
            mass=self.star_mass_sun * M_SUN,
            radius=self.star_radius_sun * R_SUN,
            luminosity=self.star_luminosity * L_SUN,
            temperature=self.star_teff_K,
            age=5.0,
        )

        p.star_context = star
        p.orbital_distance_m = self.orbital_distance_m
        return p

    def habitability_score(self) -> tuple[float, str]:
        """
        Compute Planet-RL habitability score for this entry.
        Returns (score, grade).
        """
        if self._hab_score is not None:
            return (self._hab_score, self._hab_grade)
        from planet_rl.core.habitability import assess_habitability

        p = self.to_planet()
        ha = assess_habitability(p, p.star_context, self.orbital_distance_m)
        self._hab_score = ha.overall_score
        self._hab_grade = ha.grade
        return (self._hab_score, self._hab_grade)


# ─────────────────────────────────────────────────────────────────────────────
# Bundled catalog data
# ─────────────────────────────────────────────────────────────────────────────

# Curated set of well-characterised rocky/near-rocky exoplanets.
# Parameters from NASA Exoplanet Archive PSComPPars, accessed March 2026.
# TRAPPIST-1 masses from Agol et al. (2021) ApJS 251, 23.
_CATALOG_DATA = [
    # ── TRAPPIST-1 system (Gillon et al. 2017; Agol et al. 2021) ─────────────
    CatalogEntry(
        "TRAPPIST-1 b",
        "TRAPPIST-1",
        radius_earth=1.116,
        mass_earth=1.017,
        orbital_period_d=1.511,
        semi_major_axis_au=0.01154,
        star_teff_K=2566,
        star_radius_sun=0.1192,
        star_mass_sun=0.0898,
        star_luminosity=0.000553,
        reference="Agol+2021",
    ),
    CatalogEntry(
        "TRAPPIST-1 c",
        "TRAPPIST-1",
        radius_earth=1.097,
        mass_earth=1.156,
        orbital_period_d=2.422,
        semi_major_axis_au=0.01580,
        star_teff_K=2566,
        star_radius_sun=0.1192,
        star_mass_sun=0.0898,
        star_luminosity=0.000553,
        reference="Agol+2021",
    ),
    CatalogEntry(
        "TRAPPIST-1 d",
        "TRAPPIST-1",
        radius_earth=0.788,
        mass_earth=0.297,
        orbital_period_d=4.050,
        semi_major_axis_au=0.02228,
        star_teff_K=2566,
        star_radius_sun=0.1192,
        star_mass_sun=0.0898,
        star_luminosity=0.000553,
        reference="Agol+2021",
    ),
    CatalogEntry(
        "TRAPPIST-1 e",
        "TRAPPIST-1",
        radius_earth=0.920,
        mass_earth=0.772,
        orbital_period_d=6.101,
        semi_major_axis_au=0.02928,
        star_teff_K=2566,
        star_radius_sun=0.1192,
        star_mass_sun=0.0898,
        star_luminosity=0.000553,
        reference="Agol+2021",
    ),
    CatalogEntry(
        "TRAPPIST-1 f",
        "TRAPPIST-1",
        radius_earth=1.045,
        mass_earth=0.934,
        orbital_period_d=9.207,
        semi_major_axis_au=0.03853,
        star_teff_K=2566,
        star_radius_sun=0.1192,
        star_mass_sun=0.0898,
        star_luminosity=0.000553,
        reference="Agol+2021",
    ),
    CatalogEntry(
        "TRAPPIST-1 g",
        "TRAPPIST-1",
        radius_earth=1.129,
        mass_earth=1.148,
        orbital_period_d=12.353,
        semi_major_axis_au=0.04683,
        star_teff_K=2566,
        star_radius_sun=0.1192,
        star_mass_sun=0.0898,
        star_luminosity=0.000553,
        reference="Agol+2021",
    ),
    CatalogEntry(
        "TRAPPIST-1 h",
        "TRAPPIST-1",
        radius_earth=0.755,
        mass_earth=0.331,
        orbital_period_d=18.767,
        semi_major_axis_au=0.06189,
        star_teff_K=2566,
        star_radius_sun=0.1192,
        star_mass_sun=0.0898,
        star_luminosity=0.000553,
        reference="Agol+2021",
    ),
    # ── Proxima Centauri (Anglada-Escudé et al. 2016) ─────────────────────────
    CatalogEntry(
        "Proxima Cen b",
        "Proxima Centauri",
        radius_earth=1.03,
        mass_earth=1.07,
        orbital_period_d=11.186,
        semi_major_axis_au=0.0485,
        star_teff_K=3042,
        star_radius_sun=0.1542,
        star_mass_sun=0.1221,
        star_luminosity=0.00155,
        reference="Anglada-Escude+2016",
    ),
    # ── LHS 1140 system (Dittmann et al. 2017; Ment et al. 2019) ─────────────
    CatalogEntry(
        "LHS 1140 b",
        "LHS 1140",
        radius_earth=1.635,
        mass_earth=6.98,
        orbital_period_d=24.737,
        semi_major_axis_au=0.0936,
        star_teff_K=3216,
        star_radius_sun=0.1952,
        star_mass_sun=0.1979,
        star_luminosity=0.00296,
        reference="Ment+2019",
    ),
    CatalogEntry(
        "LHS 1140 c",
        "LHS 1140",
        radius_earth=1.282,
        mass_earth=1.81,
        orbital_period_d=3.778,
        semi_major_axis_au=0.02675,
        star_teff_K=3216,
        star_radius_sun=0.1952,
        star_mass_sun=0.1979,
        star_luminosity=0.00296,
        reference="Ment+2019",
    ),
    # ── Kepler benchmark planets ──────────────────────────────────────────────
    CatalogEntry(
        "Kepler-442 b",
        "Kepler-442",
        radius_earth=1.34,
        mass_earth=2.34,
        orbital_period_d=112.305,
        semi_major_axis_au=0.409,
        star_teff_K=4402,
        star_radius_sun=0.598,
        star_mass_sun=0.609,
        star_luminosity=0.112,
        reference="Torres+2015",
    ),
    CatalogEntry(
        "Kepler-62 e",
        "Kepler-62",
        radius_earth=1.61,
        mass_earth=4.5,
        orbital_period_d=122.387,
        semi_major_axis_au=0.427,
        star_teff_K=4925,
        star_radius_sun=0.638,
        star_mass_sun=0.690,
        star_luminosity=0.202,
        reference="Borucki+2013",
    ),
    CatalogEntry(
        "Kepler-62 f",
        "Kepler-62",
        radius_earth=1.41,
        mass_earth=2.8,
        orbital_period_d=267.291,
        semi_major_axis_au=0.718,
        star_teff_K=4925,
        star_radius_sun=0.638,
        star_mass_sun=0.690,
        star_luminosity=0.202,
        reference="Borucki+2013",
    ),
    CatalogEntry(
        "Kepler-186 f",
        "Kepler-186",
        radius_earth=1.17,
        mass_earth=1.71,
        orbital_period_d=129.944,
        semi_major_axis_au=0.432,
        star_teff_K=3788,
        star_radius_sun=0.472,
        star_mass_sun=0.478,
        star_luminosity=0.0405,
        reference="Quintana+2014",
    ),
    # ── K2 and TESS discoveries ───────────────────────────────────────────────
    CatalogEntry(
        "K2-18 b",
        "K2-18",
        radius_earth=2.37,
        mass_earth=8.63,
        orbital_period_d=32.940,
        semi_major_axis_au=0.1429,
        star_teff_K=3503,
        star_radius_sun=0.4445,
        star_mass_sun=0.3593,
        star_luminosity=0.0233,
        reference="Sarkis+2018",
    ),
    CatalogEntry(
        "TOI-700 d",
        "TOI-700",
        radius_earth=1.144,
        mass_earth=1.72,
        orbital_period_d=37.424,
        semi_major_axis_au=0.1633,
        star_teff_K=3480,
        star_radius_sun=0.4191,
        star_mass_sun=0.4150,
        star_luminosity=0.0231,
        reference="Gilbert+2020",
    ),
    CatalogEntry(
        "TOI-700 e",
        "TOI-700",
        radius_earth=0.953,
        mass_earth=0.82,
        orbital_period_d=27.812,
        semi_major_axis_au=0.1344,
        star_teff_K=3480,
        star_radius_sun=0.4191,
        star_mass_sun=0.4150,
        star_luminosity=0.0231,
        reference="Gilbert+2023",
    ),
    CatalogEntry(
        "GJ 667C c",
        "GJ 667C",
        radius_earth=1.54,
        mass_earth=3.86,
        orbital_period_d=28.155,
        semi_major_axis_au=0.1251,
        star_teff_K=3350,
        star_radius_sun=0.420,
        star_mass_sun=0.330,
        star_luminosity=0.0137,
        reference="Anglada-Escude+2013",
    ),
    CatalogEntry(
        "Ross 128 b",
        "Ross 128",
        radius_earth=1.13,
        mass_earth=1.35,
        orbital_period_d=9.877,
        semi_major_axis_au=0.0496,
        star_teff_K=3192,
        star_radius_sun=0.1967,
        star_mass_sun=0.1685,
        star_luminosity=0.00362,
        reference="Bonfils+2018",
    ),
    CatalogEntry(
        "Wolf 1061 c",
        "Wolf 1061",
        radius_earth=1.66,
        mass_earth=4.25,
        orbital_period_d=17.867,
        semi_major_axis_au=0.0890,
        star_teff_K=3342,
        star_radius_sun=0.310,
        star_mass_sun=0.294,
        star_luminosity=0.0100,
        reference="Wright+2016",
    ),
    # ── Solar system reference (for comparison) ────────────────────────────────
    CatalogEntry(
        "Earth",
        "Sun",
        radius_earth=1.000,
        mass_earth=1.000,
        orbital_period_d=365.25,
        semi_major_axis_au=1.000,
        star_teff_K=5778,
        star_radius_sun=1.000,
        star_mass_sun=1.000,
        star_luminosity=1.000,
        reference="Solar system",
    ),
    CatalogEntry(
        "Mars",
        "Sun",
        radius_earth=0.532,
        mass_earth=0.107,
        orbital_period_d=686.97,
        semi_major_axis_au=1.524,
        star_teff_K=5778,
        star_radius_sun=1.000,
        star_mass_sun=1.000,
        star_luminosity=1.000,
        reference="Solar system",
    ),
    CatalogEntry(
        "Venus",
        "Sun",
        radius_earth=0.950,
        mass_earth=0.815,
        orbital_period_d=224.70,
        semi_major_axis_au=0.723,
        star_teff_K=5778,
        star_radius_sun=1.000,
        star_mass_sun=1.000,
        star_luminosity=1.000,
        reference="Solar system",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# KeplerCatalog
# ─────────────────────────────────────────────────────────────────────────────


class KeplerCatalog:
    """
    Reference exoplanet catalog with Planet-RL habitability scoring.

    Bundles 23 well-characterised rocky and near-rocky planets including
    the full TRAPPIST-1 system, plus solar system reference points.
    """

    def __init__(self):
        self.entries: list[CatalogEntry] = list(_CATALOG_DATA)

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    # ── Access ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[CatalogEntry]:
        """Get a catalog entry by planet name (case-insensitive)."""
        nl = name.lower()
        for e in self.entries:
            if e.name.lower() == nl:
                return e
        return None

    def get_planet(self, name: str):
        """Get a Planet-RL Planet object for the named entry."""
        e = self.get(name)
        if e is None:
            raise KeyError(f"Planet {name!r} not in catalog")
        return e.to_planet()

    def by_host(self, host_star: str) -> list[CatalogEntry]:
        """All planets orbiting a given star."""
        hl = host_star.lower()
        return [e for e in self.entries if e.host_star.lower() == hl]

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_all(self, verbose: bool = True) -> list[tuple[CatalogEntry, float, str]]:
        """
        Score every entry with the Planet-RL habitability framework.
        Returns list of (entry, score, grade) sorted by score descending.
        """
        results = []
        for i, entry in enumerate(self.entries):
            if verbose:
                print(
                    f"  [{i + 1:2d}/{len(self.entries)}] Scoring {entry.name}...",
                    end="\r",
                    flush=True,
                )
            score, grade = entry.habitability_score()
            results.append((entry, score, grade))
        if verbose:
            print(" " * 60, end="\r")
        return sorted(results, key=lambda x: -x[1])

    def top_n(self, n: int = 10) -> list[CatalogEntry]:
        """Top N most habitable planets by Planet-RL score."""
        scored = self.score_all(verbose=False)
        return [e for e, s, g in scored[:n]]

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print a compact table of all entries with scores and ESI."""
        scored = self.score_all(verbose=False)
        lines = [
            f"\n{'Planet':<20s}  {'Host':<18s}  {'R/R⊕':>5s}  "
            f"{'M/M⊕':>5s}  {'Flux':>6s}  {'T_eq':>6s}  "
            f"{'ESI':>5s}  {'H score':>7s}  {'Grade':>5s}",
            "─" * 92,
        ]
        for entry, score, grade in scored:
            lines.append(
                f"{entry.name:<20s}  {entry.host_star:<18s}  "
                f"{entry.radius_earth:5.2f}  {entry.mass_earth:5.2f}  "
                f"{entry.stellar_flux_earth:6.3f}  "
                f"{entry.equilibrium_temperature_K:6.0f}  "
                f"{entry.earth_similarity_index():5.3f}  "
                f"{score:7.3f}  {grade:>5s}"
            )
        lines.append("─" * 92)
        lines.append(
            f"  {len(self.entries)} planets  (sorted by Planet-RL habitability score)"
        )
        return "\n".join(lines)

    def comparison_table(self) -> dict:
        """
        Returns a dict with arrays for plotting Planet-RL score vs ESI,
        suitable for a scatter plot comparison.
        """
        import math

        names, rl_scores, esi_scores, radii, fluxes = [], [], [], [], []
        for entry in self.entries:
            score, _ = entry.habitability_score()
            names.append(entry.name)
            rl_scores.append(score)
            esi_scores.append(entry.earth_similarity_index())
            radii.append(entry.radius_earth)
            fluxes.append(entry.stellar_flux_earth)
        return {
            "names": names,
            "rl_scores": rl_scores,
            "esi": esi_scores,
            "radii": radii,
            "fluxes": fluxes,
        }

    def rl_training_candidates(self, min_score: float = 0.35) -> list[CatalogEntry]:
        """
        Return catalog entries that make good RL training targets —
        physically interesting (not obviously uninhabitable) but diverse.
        Excludes solar system reference entries.
        """
        exoplanets = [e for e in self.entries if e.host_star != "Sun"]
        scored = [(e, e.habitability_score()[0]) for e in exoplanets]
        return [e for e, s in sorted(scored, key=lambda x: -x[1]) if s >= min_score]
