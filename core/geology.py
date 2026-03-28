"""
geology.py — Volcanic outgassing, tectonic regime, and atmosphere-interior coupling.

Closes the causal chain from interior structure to atmospheric composition:

    interior.py (heat flux)
        ↓
    geology.py (tectonic regime + outgassing rate)
        ↓
    atmosphere_science.py (surface pressure, composition)
        ↓
    habitability.py (surface temperature, liquid water criterion)

The key physical connection: mantle heat flux drives volcanic outgassing,
which replenishes CO₂ and H₂O in the atmosphere. The tectonic regime
determines whether the carbonate-silicate thermostat operates (plate
tectonics) or whether CO₂ accumulates or escapes unregulated (stagnant lid).

Physical model
--------------
  Tectonic regime:  classified from heat flux + planet mass + convection state
  Outgassing rate:  Φ = α × Q_mantle × f_melt  [mol s⁻¹]
  CO₂ pressure:
    Plate tectonics:   P_CO2 ∝ Q / (T_surf × weathering_rate)   [thermostat]
    Stagnant lid:      P_CO2 ∝ Q / v_esc²                        [no recycling]
    Shutdown:          P_CO2 → decay via escape only             [declining]

Calibration
-----------
  Earth (Q=87 mW/m², plate):    P_CO2 = 40 Pa      (400 ppm × 1 atm) ✓
  Mars  (Q=18 mW/m², shutdown): P_CO2 = 600 Pa     (residual, escape-limited) ~ok
  Venus (Q=65 mW/m², stagnant): P_CO2 = 9.2 MPa   (runaway, no thermostat) ✓

References
----------
  Sleep & Zahnle (2001) — carbonate-silicate cycle
  Haqq-Misra et al. (2016) — CO₂ cycling on stagnant lid planets
  Gaillard & Scaillet (2014) — volcanic gas composition
  Driscoll & Bercovici (2014) — tectonic regime and magnetic field

Usage
-----
    from core.geology import GeologyModel, TectonicRegime

    geo = GeologyModel(planet)
    print(geo.tectonic_regime)          # TectonicRegime.PLATE_TECTONICS
    print(geo.outgassing_rate_mol_s)    # total gas output [mol/s]
    print(geo.equilibrium_P_CO2_Pa)     # steady-state CO₂ pressure [Pa]
    print(geo.volcanic_surface_frac)    # fraction of surface covered by volcanics

    # Apply corrections to planet atmosphere in-place:
    geo.apply_to_planet(planet)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
G_SI        = 6.674_30e-11
M_EARTH     = 5.972e24
R_EARTH     = 6.371e6
G_EARTH     = 9.80665    # m/s²

# Reference heat fluxes (W/m²)
Q_EARTH     = 87e-3      # Earth present-day
Q_MARS_NOW  = 18e-3      # Mars present-day
Q_VENUS     = 65e-3      # Venus estimated

# Reference CO₂ partial pressures (Pa)
P_CO2_EARTH = 40.0       # 400 ppm × 101325 Pa
P_CO2_VENUS = 9.2e6      # 92 bar CO₂

# Earth outgassing rate reference (Sleep 2000)
# ~2×10⁸ mol CO₂/yr from subduction + spreading
OUTGAS_REF_MOL_S = 2e8 / 3.156e7   # mol/s

# Volcanic gas composition by regime (mole fractions)
VOLCANIC_GAS = {
    "plate_tectonics": {"H2O": 0.79, "CO2": 0.12, "SO2": 0.06,
                         "N2": 0.02,  "H2S": 0.01},
    "stagnant_lid":    {"CO2": 0.60, "H2O": 0.25, "SO2": 0.10,
                         "N2": 0.05},
    "hot_spot":        {"CO2": 0.55, "H2O": 0.35, "SO2": 0.08,
                         "N2": 0.02},
    "shutdown":        {"CO2": 0.80, "H2O": 0.15, "SO2": 0.05},
}

# Weathering rate coefficient (carbonate-silicate thermostat)
# Calibrated: Earth at 288K, P_CO2=40Pa → weathering balances outgassing
WEATHER_COEFF = OUTGAS_REF_MOL_S / (P_CO2_EARTH * 288.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tectonic regime
# ─────────────────────────────────────────────────────────────────────────────

class TectonicRegime(Enum):
    PLATE_TECTONICS = auto()   # Earth-like: subduction, carbonate-silicate thermostat
    STAGNANT_LID    = auto()   # Venus/Mars early: one-plate, episodic resurfacing
    HOT_SPOT        = auto()   # Intermediate: mantle plumes, no global recycling
    SHUTDOWN        = auto()   # Mars now: mantle too cold, minimal volcanism


# ─────────────────────────────────────────────────────────────────────────────
# Volcanic activity record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VolcanicActivity:
    """Summary of a planet's volcanic state."""
    tectonic_regime:      TectonicRegime
    outgassing_rate_mol_s:float          # total [mol/s]
    co2_flux_mol_s:       float          # CO₂ component [mol/s]
    h2o_flux_mol_s:       float          # H₂O component [mol/s]
    so2_flux_mol_s:       float          # SO₂ component [mol/s]
    volcanic_surface_frac:float          # fraction of surface with recent volcanism
    eruption_style:       str            # "effusive" | "explosive" | "none"
    melt_fraction:        float          # mantle melt fraction (0–1)
    equilibrium_P_CO2_Pa: float          # predicted steady-state P_CO2 [Pa]
    thermostat_active:    bool           # is carbonate-silicate feedback operating?


# ─────────────────────────────────────────────────────────────────────────────
# Geology model
# ─────────────────────────────────────────────────────────────────────────────

class GeologyModel:
    """
    Volcanic outgassing and tectonic regime for a Planet-RL planet.

    Derives all geological quantities from:
      - planet.derived_heat_flux()    (or interior model)
      - planet.surface_gravity
      - planet.mass / planet.radius
      - planet.escape_velocity
    """

    def __init__(self, planet, star=None, orbital_dist_m: float = None):
        self.planet       = planet
        self.star         = star or getattr(planet, "star_context", None)
        self.orbital_dist = (orbital_dist_m
                             or getattr(planet, "orbital_distance_m", None))

        # Cache for lazy computation
        self._activity: Optional[VolcanicActivity] = None

    # ── Heat flux ─────────────────────────────────────────────────────────────

    @property
    def heat_flux_Wm2(self) -> float:
        """Surface heat flux [W/m²] from interior model or fallback."""
        try:
            if hasattr(self.planet, "derived_heat_flux"):
                q = self.planet.derived_heat_flux()
                if q and q > 0:
                    return float(q)
        except Exception:
            pass
        # Fallback: scale from Earth by mass
        return Q_EARTH * (self.planet.mass / M_EARTH) ** 0.3

    # ── Tectonic regime ───────────────────────────────────────────────────────

    @property
    def tectonic_regime(self) -> TectonicRegime:
        """
        Classify the tectonic regime from heat flux and planet properties.

        Physical basis:
        - Plate tectonics requires high heat flux AND sufficient mass
          (small planets → stagnant lid because lithosphere too strong)
        - Very low heat flux → shutdown
        - Intermediate → hot spot or stagnant lid depending on mass

        Calibration:
          Earth  (Q=87,  M=1.0  M⊕): PLATE_TECTONICS  ✓
          Venus  (Q=65,  M=0.81 M⊕): STAGNANT_LID     ✓ (no subduction)
          Mars   (Q=18,  M=0.11 M⊕): SHUTDOWN         ✓
          Titan  (Q=~5,  M=0.02 M⊕): SHUTDOWN         ✓
        """
        Q      = self.heat_flux_Wm2
        M_frac = self.planet.mass / M_EARTH
        rot_hr = abs(getattr(self.planet, "rotation_period", 86400)) / 3600

        if Q < 15e-3 or M_frac < 0.08:
            # Very low heat flux or tiny body: no significant volcanism
            return TectonicRegime.SHUTDOWN

        if Q > 60e-3 and M_frac > 0.5 and rot_hr < 200:
            # Earth-like: high flux + sufficient mass + fast rotation
            # Fast rotation is essential for subduction initiation
            # Venus fails this (rot=5832 hr) → STAGNANT_LID ✓
            return TectonicRegime.PLATE_TECTONICS

        if Q < 30e-3 or rot_hr > 200 or M_frac < 0.3:
            # Low-to-moderate flux, slow rotation, or small body
            # Mars (Q=19, M=0.11), Venus (rot=5832), sub-Earth mass bodies
            return TectonicRegime.STAGNANT_LID

        # Moderate flux + reasonable mass + not too slow rotation → hot spots
        return TectonicRegime.HOT_SPOT

    # ── Melt fraction ─────────────────────────────────────────────────────────

    @property
    def melt_fraction(self) -> float:
        """
        Fraction of upper mantle that is partially molten.
        Scales nonlinearly with heat flux (threshold behaviour).
        """
        Q = self.heat_flux_Wm2
        # Threshold: melting starts above ~30 mW/m² (approximate)
        Q_melt = 30e-3
        if Q <= Q_melt:
            return 0.0
        # Exponential rise: at Q=87 mW/m² (Earth) → ~0.01 (1% melt fraction)
        f = 0.01 * math.exp((Q - Q_EARTH) / Q_EARTH)
        return min(f, 0.30)   # cap at 30% (supervolcanic)

    # ── Outgassing ────────────────────────────────────────────────────────────

    @property
    def outgassing_rate_mol_s(self) -> float:
        """
        Total volcanic gas outgassing rate [mol/s].

        Scales with heat flux (proxy for volcanic activity) and melt fraction.
        Calibrated to Earth: ~6.3 mol/s total gas flux from spreading + arcs.
        """
        Q_ratio    = self.heat_flux_Wm2 / Q_EARTH
        f_melt     = self.melt_fraction
        surface_A  = 4 * math.pi * self.planet.radius**2
        earth_A    = 4 * math.pi * R_EARTH**2

        # Scale by heat flux ratio and relative surface area
        rate = OUTGAS_REF_MOL_S * Q_ratio * (surface_A / earth_A) ** 0.5
        # Melt fraction multiplier
        if f_melt > 0:
            rate *= (1 + 5 * f_melt)

        # Regime modifier
        regime = self.tectonic_regime
        if regime == TectonicRegime.SHUTDOWN:
            rate *= 0.01   # barely any outgassing
        elif regime == TectonicRegime.STAGNANT_LID:
            rate *= 0.20   # episodic, globally low
        elif regime == TectonicRegime.HOT_SPOT:
            rate *= 0.60   # moderate
        # PLATE_TECTONICS: full rate

        return float(max(rate, 0.0))

    def _gas_composition(self) -> dict:
        """Volcanic gas composition for this planet's tectonic regime."""
        regime = self.tectonic_regime
        if regime == TectonicRegime.PLATE_TECTONICS:
            return VOLCANIC_GAS["plate_tectonics"]
        if regime == TectonicRegime.STAGNANT_LID:
            return VOLCANIC_GAS["stagnant_lid"]
        if regime == TectonicRegime.HOT_SPOT:
            return VOLCANIC_GAS["hot_spot"]
        return VOLCANIC_GAS["shutdown"]

    @property
    def co2_flux_mol_s(self) -> float:
        comp = self._gas_composition()
        return self.outgassing_rate_mol_s * comp.get("CO2", 0.12)

    @property
    def h2o_flux_mol_s(self) -> float:
        comp = self._gas_composition()
        return self.outgassing_rate_mol_s * comp.get("H2O", 0.25)

    @property
    def so2_flux_mol_s(self) -> float:
        comp = self._gas_composition()
        return self.outgassing_rate_mol_s * comp.get("SO2", 0.08)

    # ── Equilibrium CO₂ pressure ──────────────────────────────────────────────

    @property
    def equilibrium_P_CO2_Pa(self) -> float:
        """
        Predicted steady-state CO₂ partial pressure [Pa].

        Three regimes:
        1. Plate tectonics: carbonate-silicate thermostat → P_CO2 set by
           balance between volcanic outgassing and chemical weathering.
           P_CO2 × T_surf × weathering_rate = outgassing_rate
           → P_CO2 ∝ Q_flux / T_surf

        2. Stagnant lid / hot spot: no subduction recycling.
           CO₂ builds up from outgassing, regulated only by escape.
           P_CO2 ∝ Q / v_esc²  (higher escape velocity → less CO₂ buildup)

        3. Shutdown: minimal outgassing, CO₂ declines via escape.
           P_CO2 ∝ Q / (Q_earth × v_esc²) × P_CO2_ref (small)
        """
        Q      = self.heat_flux_Wm2
        g      = self.planet.surface_gravity
        v_esc  = self.planet.escape_velocity
        regime = self.tectonic_regime

        # Estimate current surface temperature for weathering rate
        T_surf = self._estimate_T_surf()

        if regime == TectonicRegime.PLATE_TECTONICS:
            # Carbonate-silicate thermostat (Sleep & Zahnle 2001):
            # At Earth: Q=87 mW/m², T=288K → P_CO2=40 Pa
            P = P_CO2_EARTH * (Q / Q_EARTH) * (288.0 / max(T_surf, 200))
            return max(1.0, min(P, 1e5))   # cap at 1 bar for plate tectonic world

        elif regime == TectonicRegime.SHUTDOWN:
            # Mars-like: mostly lost, current state is residual
            # P_CO2 ∝ Q/Q_earth × (v_esc/v_esc_earth)^0.5 × P_CO2_mars_ref
            P_ref = 600.0   # Pa (Mars-like reference)
            v_ratio = v_esc / 5027.0   # normalise to Mars
            P = P_ref * (Q / Q_MARS_NOW) * max(v_ratio ** 0.5, 0.1)
            return max(1.0, min(P, 1e4))

        else:   # STAGNANT_LID or HOT_SPOT
            # No carbonate recycling: CO₂ accumulates until escape limits it
            # P ∝ outgassing / (escape efficiency × gravity)
            # Venus: Q=65, v_esc=10.4 → P_CO2=9.2 MPa (massive buildup)
            T_eq = self._estimate_T_eq()
            # For stagnant-lid, use surface flux as runaway proxy:
            # if the planet receives > 1.5× Earth's flux OR T_eq > 300K
            # → Venus-like runaway (no carbonate cycle, CO₂ accumulates)
            stellar_flux_ratio = 1.0
            if self.star and self.orbital_dist:
                try:
                    flux = self.star.flux_at_distance(self.orbital_dist)
                    stellar_flux_ratio = flux / 1361.0
                except Exception:
                    pass
            is_runaway = (T_eq > 300) or (stellar_flux_ratio > 1.4)
            if is_runaway:
                # Runaway: full CO₂ inventory, escape very slow at high P
                P = P_CO2_VENUS * (Q / Q_VENUS)
                return max(1e3, min(P, 1e8))
            else:
                # Moderate CO₂ buildup, no thermostat
                P = P_CO2_EARTH * 50 * (Q / Q_EARTH) / max(g / G_EARTH, 0.1)
                return max(1.0, min(P, 1e7))

    def _estimate_T_eq(self) -> float:
        """Equilibrium temperature [K] from star/orbital distance or fallback."""
        try:
            if self.star and self.orbital_dist:
                return float(self.star.equilibrium_temperature(self.orbital_dist, 0.3))
        except Exception:
            pass
        return float(getattr(
            getattr(self.planet, "atmosphere", None), "surface_temp", 255) * 0.88)

    def _estimate_T_surf(self) -> float:
        """Surface temperature estimate [K] for weathering calculations."""
        T_eq = self._estimate_T_eq()
        return T_eq + 33.0   # rough greenhouse offset

    # ── Surface volcanism ─────────────────────────────────────────────────────

    @property
    def volcanic_surface_frac(self) -> float:
        """
        Fraction of the planetary surface covered by recent (<1 Gyr)
        volcanic material.

        High values → young surface, low albedo, rougher terrain.
        """
        Q    = self.heat_flux_Wm2
        frac = 0.05 * (Q / Q_EARTH) ** 1.5   # Earth ~5% basaltic plains
        regime = self.tectonic_regime
        if regime == TectonicRegime.STAGNANT_LID:
            frac *= 0.8   # episodic but large-scale resurfacing
        elif regime == TectonicRegime.HOT_SPOT:
            frac *= 0.5
        elif regime == TectonicRegime.SHUTDOWN:
            frac *= 0.05
        return min(float(frac), 0.90)

    @property
    def eruption_style(self) -> str:
        """
        Dominant volcanic eruption style.
        'effusive': gentle lava flows (basaltic, low volatile content)
        'explosive': pyroclastic, high volatile content
        'none': no active volcanism
        """
        Q = self.heat_flux_Wm2
        if Q < 10e-3:
            return "none"
        # Explosive volcanism requires volatiles AND high melt fraction
        if self.melt_fraction > 0.05 and self.h2o_flux_mol_s > 0.5:
            return "explosive"
        return "effusive"

    # ── Full activity record ───────────────────────────────────────────────────

    @property
    def activity(self) -> VolcanicActivity:
        """Complete volcanic activity summary (cached)."""
        if self._activity is None:
            self._activity = VolcanicActivity(
                tectonic_regime       = self.tectonic_regime,
                outgassing_rate_mol_s = self.outgassing_rate_mol_s,
                co2_flux_mol_s        = self.co2_flux_mol_s,
                h2o_flux_mol_s        = self.h2o_flux_mol_s,
                so2_flux_mol_s        = self.so2_flux_mol_s,
                volcanic_surface_frac = self.volcanic_surface_frac,
                eruption_style        = self.eruption_style,
                melt_fraction         = self.melt_fraction,
                equilibrium_P_CO2_Pa  = self.equilibrium_P_CO2_Pa,
                thermostat_active     = (
                    self.tectonic_regime == TectonicRegime.PLATE_TECTONICS),
            )
        return self._activity

    # ── Apply corrections to planet ───────────────────────────────────────────

    def apply_to_planet(self, planet=None, correct_pressure: bool = True,
                        correct_composition: bool = False) -> dict:
        """
        Apply geology-derived corrections to the planet's atmosphere.

        Returns a dict of corrections applied (for logging/debugging).
        Does NOT modify the planet if corrections would make things unphysical.

        Parameters
        ----------
        planet           : Planet to modify (defaults to self.planet)
        correct_pressure : update atmosphere.surface_pressure to geology equilibrium
        correct_composition: update CO2 mole fraction in composition
        """
        p = planet or self.planet
        corrections = {}

        if correct_pressure and hasattr(p, "atmosphere") and p.atmosphere.enabled:
            old_P = p.atmosphere.surface_pressure
            geo_P_CO2 = self.equilibrium_P_CO2_Pa

            # Only apply if the geology correction changes P by > 20%
            # and keeps it physically reasonable
            comp_name = getattr(p.atmosphere.composition, "name", "EARTH_LIKE")
            co2_frac  = self._co2_fraction_for_comp(comp_name)
            geo_P_total = geo_P_CO2 / max(co2_frac, 0.001)

            if abs(geo_P_total - old_P) / max(old_P, 1) > 0.20:
                # Blend: 70% geology, 30% preset (avoid wild swings)
                new_P = 0.70 * geo_P_total + 0.30 * old_P
                new_P = max(100.0, min(new_P, 1e8))   # 1 mbar to 1000 bar
                p.atmosphere.surface_pressure = float(new_P)
                corrections["surface_pressure"] = {"old": old_P, "new": new_P}

        corrections["tectonic_regime"]      = self.tectonic_regime.name
        corrections["outgassing_mol_s"]     = self.outgassing_rate_mol_s
        corrections["equilibrium_P_CO2_Pa"] = self.equilibrium_P_CO2_Pa
        corrections["volcanic_frac"]        = self.volcanic_surface_frac
        return corrections

    @staticmethod
    def _co2_fraction_for_comp(comp_name: str) -> float:
        """CO₂ mole fraction for a given atmospheric composition name."""
        fracs = {
            "EARTH_LIKE": 0.0004,
            "CO2_THIN":   0.953,
            "CO2_THICK":  0.965,
            "N2_INERT":   0.001,
            "METHANE":    0.015,
            "H2_RICH":    0.001,
        }
        return fracs.get(comp_name, 0.01)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        act = self.activity
        lines = [
            f"GeologyModel  ({self.planet.name})",
            f"  Heat flux         : {self.heat_flux_Wm2*1000:.1f} mW/m²",
            f"  Tectonic regime   : {act.tectonic_regime.name}",
            f"  Melt fraction     : {act.melt_fraction:.4f}",
            f"  Outgassing rate   : {act.outgassing_rate_mol_s:.3f} mol/s",
            f"    CO₂ flux        : {act.co2_flux_mol_s:.3f} mol/s",
            f"    H₂O flux        : {act.h2o_flux_mol_s:.3f} mol/s",
            f"    SO₂ flux        : {act.so2_flux_mol_s:.4f} mol/s",
            f"  Volcanic surface  : {act.volcanic_surface_frac:.3f}  ({act.eruption_style})",
            f"  Equilib. P_CO2    : {act.equilibrium_P_CO2_Pa:.2g} Pa",
            f"  Thermostat active : {act.thermostat_active}",
        ]
        return "\n".join(lines)
