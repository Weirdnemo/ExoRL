"""
population.py — Statistical analysis of procedurally generated planet populations.

Answers the questions that a single planet cannot:
  - What does the mass-radius distribution look like across 1000 random planets?
  - Where do they sit relative to iron / rocky / water composition curves?
  - What fraction are potentially habitable?
  - What physical properties most strongly predict habitability?
  - What does the RL training distribution actually look like?

Key entry points
----------------
    pop = PlanetPopulation.generate(n=1000, seed=0)
    pop.save("population.csv")

    fig = plot_mass_radius(pop)
    fig = plot_habitability_distribution(pop)
    fig = plot_correlation_heatmap(pop)
    fig = plot_population_dashboard(pop)

References
----------
Zeng & Sasselov (2013) — mass-radius composition curves
Fortney, Marley & Barnes (2007) — planetary radii
Kopparapu et al. (2013) — habitable zone boundaries
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
R_EARTH = 6.371e6
M_EARTH = 5.972e24
G = 6.674_30e-11
AU = 1.495_978_707e11

# ─────────────────────────────────────────────────────────────────────────────
# Composition curves (Zeng & Sasselov 2013)
# ─────────────────────────────────────────────────────────────────────────────


def composition_radius(mass_earth: float, composition: str) -> float:
    """
    Planet radius [Earth radii] for a given mass and bulk composition.

    Zeng & Sasselov (2013) power-law fits:
        pure iron:   R = 0.774  × M^0.295
        rocky:       R = 1.008  × M^0.279   (32.5% Fe core, 67.5% MgSiO3 mantle)
        50% water:   R = 1.262  × M^0.341   (50% H2O, 50% rocky)
        pure water:  R = 1.410  × M^0.393
    """
    fits = {
        "iron": (0.774, 0.295),
        "rocky": (1.008, 0.279),
        "water50": (1.262, 0.341),
        "water": (1.410, 0.393),
    }
    a, b = fits.get(composition, (1.008, 0.279))
    return a * max(mass_earth, 1e-4) ** b


def classify_composition(mass_earth: float, radius_earth: float) -> str:
    """
    Classify a planet's bulk composition from its mass and radius.
    Returns one of: 'iron', 'rocky', 'water-rich', 'gas-dwarf', 'unknown'.
    """
    r_iron = composition_radius(mass_earth, "iron")
    r_rocky = composition_radius(mass_earth, "rocky")
    r_water = composition_radius(mass_earth, "water")

    if radius_earth < r_iron * 1.05:
        return "iron"
    elif radius_earth < r_rocky * 1.10:
        return "rocky"
    elif radius_earth < r_water * 1.15:
        return "water-rich"
    elif radius_earth < 4.0:
        return "gas-dwarf"
    else:
        return "gas-giant"


# ─────────────────────────────────────────────────────────────────────────────
# Planet record — one row in the population table
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PlanetRecord:
    """All derived quantities for one planet in the population."""

    # Identity
    index: int
    name: str

    # Physical
    mass_earth: float  # mass / M_earth
    radius_earth: float  # radius / R_earth
    density_kg_m3: float
    gravity_ms2: float
    escape_vel_km_s: float

    # Interior
    j2: float
    b_field_uT: float
    heat_flux_mW: float  # mW/m²
    moi_factor: float  # C/MR²
    has_dynamo: bool

    # Atmosphere
    has_atm: bool
    surface_pressure_bar: float
    surface_temp_K: float
    greenhouse_dT_K: float
    atm_composition: str

    # Habitability
    hab_score: float
    hab_grade: str
    in_hz: bool
    star_name: str
    orbital_dist_au: float

    # Composition
    composition: str  # iron/rocky/water-rich/gas-dwarf/gas-giant

    # Observational
    transit_depth_ppm: float
    rv_K_ms: float  # RV semi-amplitude vs Sun-like star at 1 AU

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "mass_earth": self.mass_earth,
            "radius_earth": self.radius_earth,
            "density": self.density_kg_m3,
            "gravity": self.gravity_ms2,
            "escape_km_s": self.escape_vel_km_s,
            "j2": self.j2,
            "b_uT": self.b_field_uT,
            "heat_mW": self.heat_flux_mW,
            "moi": self.moi_factor,
            "has_dynamo": int(self.has_dynamo),
            "has_atm": int(self.has_atm),
            "P_bar": self.surface_pressure_bar,
            "T_surf_K": self.surface_temp_K,
            "dT_GH_K": self.greenhouse_dT_K,
            "hab_score": self.hab_score,
            "hab_grade": self.hab_grade,
            "in_hz": int(self.in_hz),
            "star": self.star_name,
            "dist_au": self.orbital_dist_au,
            "composition": self.composition,
            "transit_ppm": self.transit_depth_ppm,
            "rv_K_ms": self.rv_K_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Population
# ─────────────────────────────────────────────────────────────────────────────


class PlanetPopulation:
    """
    A collection of PlanetRecord objects with statistical analysis methods.

    Generate via PlanetPopulation.generate(n, seed).
    """

    def __init__(self, records: list[PlanetRecord]):
        self.records = records
        self._arr: Optional[dict] = None

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    # ── Generation ────────────────────────────────────────────────────────────

    @classmethod
    def generate(
        cls,
        n: int = 500,
        seed: int = 0,
        atmosphere_enabled: bool = True,
        oblateness_enabled: bool = True,
        magnetic_field_enabled: bool = True,
        attach_random_star: bool = True,
        verbose: bool = True,
    ) -> "PlanetPopulation":
        """
        Generate n planets and compute all derived quantities.

        Parameters
        ----------
        n                    : number of planets to generate
        seed                 : random seed for reproducibility
        atmosphere_enabled   : include atmospheres
        oblateness_enabled   : include J2 oblateness
        magnetic_field_enabled: include magnetic fields
        attach_random_star   : attach a random star + HZ orbital distance
        verbose              : print progress
        """
        from planet_rl.core.generator import PRESETS, PlanetGenerator
        from planet_rl.core.habitability import assess_habitability
        from planet_rl.core.interior import interior_from_bulk_density
        from planet_rl.core.observation import rv_semi_amplitude
        from planet_rl.core.observation import transit_depth_ppm as tdppm

        try:
            from planet_rl.core.star import STAR_PRESETS, star_sun

            _star_ok = True
        except ImportError:
            _star_ok = False

        try:
            from planet_rl.core.atmosphere_science import analyse_atmosphere

            _atm_ok = True
        except ImportError:
            _atm_ok = False

        gen = PlanetGenerator(seed=seed)
        rng = np.random.RandomState(seed)
        star_keys = list(STAR_PRESETS.keys()) if _star_ok else ["sun"]
        records = []

        for i in range(n):
            if verbose and i % 100 == 0:
                print(f"  Generating planet {i + 1}/{n}...")

            planet = gen.generate(
                atmosphere_enabled=atmosphere_enabled,
                oblateness_enabled=oblateness_enabled,
                magnetic_field_enabled=magnetic_field_enabled,
            )

            # Attach interior
            try:
                planet.interior = interior_from_bulk_density(planet.mean_density)
            except Exception:
                pass

            # Attach star + orbital distance
            star = None
            orbital_dist_au = 1.0
            if _star_ok and attach_random_star:
                try:
                    key = star_keys[rng.randint(0, len(star_keys))]
                    star = STAR_PRESETS[key]()
                    hz_mid = (star.hz_inner_m + star.hz_outer_m) / 2.0
                    spread = (star.hz_outer_m - star.hz_inner_m) * 0.5
                    orbital_dist_m = hz_mid + rng.uniform(-spread, spread)
                    orbital_dist_au = orbital_dist_m / AU
                    planet.star_context = star
                    planet.orbital_distance_m = orbital_dist_m
                except Exception:
                    star = None

            if star is None and _star_ok:
                star = star_sun()
                planet.star_context = star
                planet.orbital_distance_m = 1.0 * AU
                orbital_dist_au = 1.0

            # Interior quantities
            try:
                j2 = float(planet.derived_J2())
                b_uT = float(planet.derived_magnetic_field_T()) * 1e6
                heat_mW = float(planet.derived_heat_flux()) * 1000
                moi = float(planet.derived_MoI())
                has_dynamo = (
                    planet.interior.dynamo_active(planet.radius, planet.mass)
                    if planet.interior
                    else False
                )
            except Exception:
                j2 = planet.oblateness.J2 if planet.oblateness.enabled else 0.0
                b_uT = 0.0
                heat_mW = 0.0
                moi = 0.4
                has_dynamo = False

            # Atmosphere
            t_surf_K = 0.0
            dT_GH_K = 0.0
            P_bar = 0.0
            atm_comp_str = "none"
            if planet.atmosphere.enabled:
                P_bar = planet.atmosphere.surface_pressure / 1e5
                atm_comp_str = planet.atmosphere.composition.name
                if _atm_ok and star:
                    try:
                        aa = analyse_atmosphere(planet, star, planet.orbital_distance_m)
                        t_surf_K = float(aa.get("surface_temp_K", 0))
                        dT_GH_K = float(aa.get("greenhouse_dT_K", 0))
                    except Exception:
                        t_surf_K = float(planet.atmosphere.surface_temp)

            # Habitability
            hab_score = 0.0
            hab_grade = "?"
            in_hz = False
            if star:
                try:
                    ha = assess_habitability(planet, star, planet.orbital_distance_m)
                    hab_score = float(ha.overall_score)
                    hab_grade = ha.grade
                    in_hz = bool(star.in_habitable_zone(planet.orbital_distance_m))
                except Exception:
                    pass

            # Composition
            comp = classify_composition(planet.mass / M_EARTH, planet.radius / R_EARTH)

            # Observational (vs Sun-like star at orbital distance)
            td_ppm = 0.0
            rv_K = 0.0
            if star:
                try:
                    td_ppm = float(tdppm(planet.radius, star.radius))
                    mu_s = G * star.mass
                    T_orb = 2 * math.pi * math.sqrt(planet.orbital_distance_m**3 / mu_s)
                    rv_K = float(rv_semi_amplitude(planet.mass, star.mass, T_orb))
                except Exception:
                    pass

            records.append(
                PlanetRecord(
                    index=i,
                    name=planet.name,
                    mass_earth=planet.mass / M_EARTH,
                    radius_earth=planet.radius / R_EARTH,
                    density_kg_m3=planet.mean_density,
                    gravity_ms2=planet.surface_gravity,
                    escape_vel_km_s=planet.escape_velocity / 1e3,
                    j2=j2,
                    b_field_uT=b_uT,
                    heat_flux_mW=heat_mW,
                    moi_factor=moi,
                    has_dynamo=bool(has_dynamo),
                    has_atm=planet.atmosphere.enabled,
                    surface_pressure_bar=P_bar,
                    surface_temp_K=t_surf_K,
                    greenhouse_dT_K=dT_GH_K,
                    atm_composition=atm_comp_str,
                    hab_score=hab_score,
                    hab_grade=hab_grade,
                    in_hz=in_hz,
                    star_name=star.name if star else "none",
                    orbital_dist_au=orbital_dist_au,
                    composition=comp,
                    transit_depth_ppm=td_ppm,
                    rv_K_ms=rv_K,
                )
            )

        if verbose:
            print(f"  Done. {n} planets generated.")

        return cls(records)

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save population to CSV."""
        import csv

        if not self.records:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].to_dict().keys())
            writer.writeheader()
            for r in self.records:
                writer.writerow(r.to_dict())

    @classmethod
    def load(cls, path: str) -> "PlanetPopulation":
        """Load population from CSV."""
        import csv

        records = []
        with open(path, "r") as f:
            for i, row in enumerate(csv.DictReader(f)):
                records.append(
                    PlanetRecord(
                        index=int(row["index"]),
                        name=f"Planet-{i}",
                        mass_earth=float(row["mass_earth"]),
                        radius_earth=float(row["radius_earth"]),
                        density_kg_m3=float(row["density"]),
                        gravity_ms2=float(row["gravity"]),
                        escape_vel_km_s=float(row["escape_km_s"]),
                        j2=float(row["j2"]),
                        b_field_uT=float(row["b_uT"]),
                        heat_flux_mW=float(row["heat_mW"]),
                        moi_factor=float(row["moi"]),
                        has_dynamo=bool(int(row["has_dynamo"])),
                        has_atm=bool(int(row["has_atm"])),
                        surface_pressure_bar=float(row["P_bar"]),
                        surface_temp_K=float(row["T_surf_K"]),
                        greenhouse_dT_K=float(row["dT_GH_K"]),
                        atm_composition=row.get("atm_composition", ""),
                        hab_score=float(row["hab_score"]),
                        hab_grade=row["hab_grade"],
                        in_hz=bool(int(row["in_hz"])),
                        star_name=row["star"],
                        orbital_dist_au=float(row["dist_au"]),
                        composition=row["composition"],
                        transit_depth_ppm=float(row["transit_ppm"]),
                        rv_K_ms=float(row["rv_K_ms"]),
                    )
                )
        return cls(records)

    # ── Arrays ────────────────────────────────────────────────────────────────

    def arrays(self) -> dict:
        """Return all fields as numpy arrays (cached)."""
        if self._arr is not None:
            return self._arr
        self._arr = {
            "mass": np.array([r.mass_earth for r in self.records]),
            "radius": np.array([r.radius_earth for r in self.records]),
            "density": np.array([r.density_kg_m3 for r in self.records]),
            "gravity": np.array([r.gravity_ms2 for r in self.records]),
            "escape": np.array([r.escape_vel_km_s for r in self.records]),
            "j2": np.array([r.j2 for r in self.records]),
            "b_field": np.array([r.b_field_uT for r in self.records]),
            "heat_flux": np.array([r.heat_flux_mW for r in self.records]),
            "moi": np.array([r.moi_factor for r in self.records]),
            "hab_score": np.array([r.hab_score for r in self.records]),
            "P_srf": np.array([r.surface_pressure_bar for r in self.records]),
            "T_surf": np.array([r.surface_temp_K for r in self.records]),
            "dT_GH": np.array([r.greenhouse_dT_K for r in self.records]),
            "transit_ppm": np.array([r.transit_depth_ppm for r in self.records]),
            "rv_K": np.array([r.rv_K_ms for r in self.records]),
            "dist_au": np.array([r.orbital_dist_au for r in self.records]),
            "has_dynamo": np.array([r.has_dynamo for r in self.records]),
            "has_atm": np.array([r.has_atm for r in self.records]),
            "in_hz": np.array([r.in_hz for r in self.records]),
        }
        return self._arr

    # ── Summary statistics ────────────────────────────────────────────────────

    def summary(self) -> str:
        a = self.arrays()
        n = len(self.records)
        n_hab = int(np.sum(a["hab_score"] > 0.5))
        n_hz = int(np.sum(a["in_hz"]))
        n_dyn = int(np.sum(a["has_dynamo"]))
        n_atm = int(np.sum(a["has_atm"]))

        comp_counts = {}
        for r in self.records:
            comp_counts[r.composition] = comp_counts.get(r.composition, 0) + 1

        grade_counts = {}
        for r in self.records:
            grade_counts[r.hab_grade] = grade_counts.get(r.hab_grade, 0) + 1

        lines = [
            f"Planet population  (n = {n})",
            f"",
            f"  Mass             {a['mass'].min():.3f} – {a['mass'].max():.3f} M⊕   "
            f"median {np.median(a['mass']):.2f}",
            f"  Radius           {a['radius'].min():.3f} – {a['radius'].max():.3f} R⊕   "
            f"median {np.median(a['radius']):.2f}",
            f"  Density          {a['density'].min():.0f} – {a['density'].max():.0f} kg/m³",
            f"  Habitability     {a['hab_score'].min():.2f} – {a['hab_score'].max():.2f}   "
            f"mean {a['hab_score'].mean():.3f}",
            f"",
            f"  Potentially habitable (score > 0.5):  {n_hab} / {n}  ({100 * n_hab / n:.1f}%)",
            f"  In habitable zone:                    {n_hz} / {n}  ({100 * n_hz / n:.1f}%)",
            f"  Has active dynamo:                    {n_dyn} / {n}  ({100 * n_dyn / n:.1f}%)",
            f"  Has atmosphere:                       {n_atm} / {n}  ({100 * n_atm / n:.1f}%)",
            f"",
            f"  Composition breakdown:",
        ]
        for comp, cnt in sorted(comp_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {comp:<12s}  {cnt:4d}  ({100 * cnt / n:.1f}%)")
        lines.append(f"")
        lines.append(f"  Habitability grades:")
        for g in "ABCDF?":
            cnt = grade_counts.get(g, 0)
            if cnt > 0:
                lines.append(f"    Grade {g}:  {cnt:4d}  ({100 * cnt / n:.1f}%)")

        return "\n".join(lines)

    # ── Correlation matrix ────────────────────────────────────────────────────

    def correlation_matrix(self) -> tuple[np.ndarray, list[str]]:
        """
        Pearson correlation matrix between key numeric properties.
        Returns (corr_matrix, feature_names).
        """
        a = self.arrays()
        keys = [
            "mass",
            "radius",
            "density",
            "gravity",
            "j2",
            "b_field",
            "heat_flux",
            "moi",
            "P_srf",
            "T_surf",
            "hab_score",
            "transit_ppm",
            "rv_K",
            "dist_au",
        ]
        mat = np.column_stack([a[k] for k in keys])

        for j in range(mat.shape[1]):
            col = mat[:, j]
            bad = ~np.isfinite(col)
            if bad.any():
                med = np.nanmedian(col)
                mat[bad, j] = med if np.isfinite(med) else 0.0

        corr = np.corrcoef(mat.T)
        return corr, keys

    # ── RL training analysis ──────────────────────────────────────────────────

    def rl_training_stats(self) -> dict:
        """
        Statistics most relevant for RL training with OrbitalInsertionEnv.
        Quantifies what the agent will actually face during training.
        """
        a = self.arrays()
        return {
            "n_total": len(self.records),
            "n_habitable": int(np.sum(a["hab_score"] > 0.5)),
            "frac_habitable": float(np.mean(a["hab_score"] > 0.5)),
            "hab_score_mean": float(a["hab_score"].mean()),
            "hab_score_std": float(a["hab_score"].std()),
            "mass_p5": float(np.percentile(a["mass"], 5)),
            "mass_p95": float(np.percentile(a["mass"], 95)),
            "radius_p5": float(np.percentile(a["radius"], 5)),
            "radius_p95": float(np.percentile(a["radius"], 95)),
            "gravity_p5": float(np.percentile(a["gravity"], 5)),
            "gravity_p95": float(np.percentile(a["gravity"], 95)),
            "j2_median": float(np.median(a["j2"])),
            "j2_p95": float(np.percentile(a["j2"], 95)),
            "b_field_median_uT": float(np.median(a["b_field"])),
            "frac_with_dynamo": float(np.mean(a["has_dynamo"])),
            "frac_in_hz": float(np.mean(a["in_hz"])),
            "T_surf_median_K": float(np.median(a["T_surf"][a["T_surf"] > 0])),
            "transit_ppm_median": float(np.median(a["transit_ppm"])),
        }
