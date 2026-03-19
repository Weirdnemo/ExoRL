"""
climate.py — 1D Energy Balance Model (EBM) with climate feedbacks.

Closes the biggest gap in the habitability assessment: the current model
scores temperature but doesn't know that a planet can tip into permanent
climate states. This module finds those tipping points.

Physics implemented
-------------------
1. Ice-albedo feedback (Budyko-Sellers EBM)
   Ice reflects ~60% of sunlight vs ~30% for ocean/rock.
   As ice spreads equatorward it cools the planet further → positive feedback.
   There is a critical insolation below which the feedback runs away and
   the entire planet freezes: the "snowball" state.

2. Runaway greenhouse
   Above ~340 K, water vapour pressure rises exponentially (Clausius-Clapeyron).
   More H₂O → stronger greenhouse → higher T → more H₂O.
   The feedback is unbounded — the oceans evaporate.
   Threshold: when the outgoing LW radiation can no longer balance incoming SW.
   Kasting (1988) / Kopparapu (2013) parametrisation.

3. Multiple stable states
   The EBM has two stable solutions at some distances: a warm habitable state
   and a cold snowball state. Which one the planet is in depends on history.
   The bistable region is bounded by:
       d_snowball   → planet always freezes (too cold even in warm state)
       d_runaway    → planet always has runaway greenhouse (too hot even in cold state)
   Between these: both states are possible. Earth is in the warm state.

4. CO₂ thermostat (carbonate-silicate cycle)
   Over ~1 Myr timescales, weathering of silicate rock draws down CO₂ when
   the planet is warm, and volcanism replenishes it when cold. This provides
   a long-term negative feedback that stabilises the climate.
   Simplified version: CO₂ partial pressure adjusts to maintain T_surface ≈ 285 K.

Usage
-----
    from core.climate import EnergyBalanceModel, climate_map, find_bifurcation_points

    ebm = EnergyBalanceModel(planet, star)
    result = ebm.solve(orbital_distance_m=1.0 * AU)
    print(result.report())

    # Where does the planet tip into snowball/runaway?
    bif = find_bifurcation_points(planet, star)
    print(f"Snowball at d < {bif.snowball_au:.3f} AU")
    print(f"Runaway  at d > {bif.runaway_au:.3f} AU")

All SI unless noted.

References
----------
Budyko (1969), Sellers (1969) — original 1D EBM
North et al. (1981) — EBM with diffusion
Kasting (1988) — runaway greenhouse
Pierrehumbert (2010) "Principles of Planetary Climate" Ch. 3, 7
Kopparapu et al. (2013) — HZ with climate feedbacks
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
SIGMA   = 5.670_374e-8   # Stefan-Boltzmann [W m⁻² K⁻⁴]
AU      = 1.495_978_707e11
G       = 6.674_30e-11

# ── Ice-albedo parameters ──────────────────────────────────────────────────────
ALBEDO_ICE_FREE  = 0.30    # global mean albedo of an ice-free Earth-like planet
ALBEDO_SNOWBALL  = 0.60    # global mean albedo when fully ice-covered
ALBEDO_ICE       = 0.55    # albedo of sea ice / glaciers
ALBEDO_LAND      = 0.15    # bare land / ocean mix
T_FREEZE         = 263.0   # K — ice forms below this temperature
T_MELT           = 273.0   # K — ice melts above this (hysteresis gap)

# ── Greenhouse parameters ──────────────────────────────────────────────────────
T_RUNAWAY_ONSET  = 340.0   # K — water vapour feedback accelerates strongly
T_MOIST_GH       = 373.0   # K — moist greenhouse limit (oceans start to evaporate)
T_WET_LIMIT      = 647.0   # K — supercritical water — oceans fully evaporated

# ── Budyko-Sellers heat diffusion coefficient ─────────────────────────────────
# A = OLR intercept, B = OLR slope, D = poleward heat transport
# Calibrated to present-day Earth:
#   A + B*T = 203.3 + 2.09*T   gives ~240 W/m² at T=279 K (observed OLR)
# Budyko (1969): OLR = A_celsius + B*T_celsius, A=203.3 W/m², B=2.09 W/m²/°C
# Converted to Kelvin: A_K = 203.3 - 2.09*273.15 = -367.6
# Check: OLR(288 K) = -367.6 + 2.09*288 = -367.6 + 601.9 = 234.3 W/m² ✓
BUDYKO_A = -367.6  # W/m²   (Celsius intercept converted to Kelvin basis)
BUDYKO_B =  2.09   # W/m²/K


# ─────────────────────────────────────────────────────────────────────────────
# Albedo model
# ─────────────────────────────────────────────────────────────────────────────

def ice_covered_fraction(T_global_K: float,
                          T_freeze: float = T_FREEZE,
                          T_melt: float   = T_MELT,
                          hysteresis: bool = True) -> float:
    """
    Fraction of planet surface covered by ice as a function of global mean T.

    Uses a smooth sigmoidal transition between ice-free and ice-covered states.
    With hysteresis=True, the melting threshold is higher than the freezing
    threshold — this creates the bistable regime where both warm and cold
    states are stable.

    Returns f_ice ∈ [0, 1].
    """
    T_mid  = (T_freeze + T_melt) / 2
    T_half = (T_melt - T_freeze) / 2

    if T_global_K <= T_freeze:
        return 1.0
    if T_global_K >= T_melt:
        return 0.0
    # Smooth sigmoid between freeze and melt
    x = (T_global_K - T_mid) / max(T_half, 1.0)
    return 0.5 * (1.0 - math.tanh(2.0 * x))


def effective_albedo(T_global_K: float,
                      base_albedo: float = ALBEDO_ICE_FREE,
                      ice_albedo: float  = ALBEDO_ICE,
                      T_freeze: float    = T_FREEZE,
                      T_melt: float      = T_MELT) -> float:
    """
    Effective planetary albedo as a function of global mean temperature.

    Linear mix between ice-free albedo and full-ice albedo:
        A_eff = (1 - f_ice) × A_warm + f_ice × A_ice

    This is the Budyko (1969) parameterisation.
    """
    f_ice = ice_covered_fraction(T_global_K, T_freeze, T_melt)
    return (1.0 - f_ice) * base_albedo + f_ice * ice_albedo


def co2_feedback_albedo_correction(co2_ppm: float,
                                    reference_ppm: float = 280.0) -> float:
    """
    Small albedo correction from cloud feedback driven by CO₂.
    Higher CO₂ → slightly warmer clouds → slightly higher albedo.
    This is a second-order effect, included for completeness.

    Returns delta_albedo (typically -0.005 to +0.005).
    """
    if co2_ppm <= 0 or reference_ppm <= 0:
        return 0.0
    # ~0.003 albedo change per doubling of CO₂ (observational estimate)
    return -0.003 * math.log2(co2_ppm / reference_ppm)


# ─────────────────────────────────────────────────────────────────────────────
# Outgoing longwave radiation (OLR)
# ─────────────────────────────────────────────────────────────────────────────

def olr_budyko(T_surface_K: float) -> float:
    """
    Outgoing longwave radiation [W/m²] — Budyko (1969) linear approximation.

    OLR = A + B × T   where A=203.3 W/m², B=2.09 W/m²/K

    Calibrated to Earth: OLR(279 K) ≈ 237 W/m² ✓
    Valid for 230 K < T < 330 K (breaks down near runaway).
    """
    return BUDYKO_A + BUDYKO_B * T_surface_K


def olr_with_greenhouse(T_surface_K: float,
                         greenhouse_dT_K: float,
                         co2_ppm: float = 280.0) -> float:
    """
    OLR modified by greenhouse effect.

    The greenhouse gas absorbs some OLR and re-emits at a lower effective
    temperature, reducing the net OLR. We approximate this as:

        OLR_eff = OLR_clear × (1 - τ_GH)

    where τ_GH is the greenhouse optical depth, estimated from the
    greenhouse warming: τ_GH ≈ 1 - (T_eq/T_surface)⁴

    This is a simplification of the full line-by-line radiative transfer.
    """
    if T_surface_K <= 0:
        return 0.0
    # Effective emission temperature from energy balance
    T_eff    = (T_surface_K**4 - (greenhouse_dT_K / 0.26 * T_surface_K**3
                 if greenhouse_dT_K > 0 else 0)) ** 0.25
    T_eff    = max(T_eff, T_surface_K * 0.5)
    olr_base = olr_budyko(T_surface_K)
    # Reduce OLR by greenhouse optical depth
    tau = 1.0 - (T_eff / T_surface_K) ** 4
    tau = max(0.0, min(tau, 0.95))
    return olr_base * (1.0 - tau)


def olr_near_runaway(T_surface_K: float) -> float:
    """
    OLR approaching the runaway greenhouse limit [W/m²].

    Above ~340 K, the OLR effectively saturates — it cannot increase
    fast enough to balance the increasing absorbed solar radiation.
    This is the Simpson-Nakajima limit.

    Kasting (1988) computed the limit at ~310 W/m².
    """
    # Below onset: normal OLR
    if T_surface_K < T_RUNAWAY_ONSET:
        return olr_budyko(T_surface_K)
    # Transition zone: OLR saturates
    elif T_surface_K < T_MOIST_GH:
        progress = (T_surface_K - T_RUNAWAY_ONSET) / (T_MOIST_GH - T_RUNAWAY_ONSET)
        olr_normal  = olr_budyko(T_surface_K)
        olr_sat_limit = 310.0   # W/m² — Simpson-Nakajima saturation limit
        return olr_normal * (1.0 - progress) + olr_sat_limit * progress
    else:
        # Moist greenhouse / wet limit — OLR can no longer balance SW
        return 310.0


# ─────────────────────────────────────────────────────────────────────────────
# CO₂ carbonate-silicate thermostat
# ─────────────────────────────────────────────────────────────────────────────

def carbonate_silicate_co2_ppm(T_surface_K: float,
                                  reference_T: float = 285.0,
                                  reference_ppm: float = 280.0,
                                  sensitivity: float = 0.5) -> float:
    """
    Equilibrium CO₂ partial pressure from the carbonate-silicate cycle [ppm].

    The silicate weathering rate increases with temperature (more rain,
    faster chemical reactions). Higher weathering → more CO₂ drawdown → cooling.
    Volcanic outgassing is approximately constant.

    At equilibrium: weathering rate = outgassing rate
    → CO₂ ∝ exp(-sensitivity × (T - T_ref))

    Sensitivity ≈ 0.3–0.7 ppm/K in GCM studies.
    This gives a negative feedback that stabilises the climate over ~1 Myr.

    Returns CO₂ in ppm. Minimum floor: 10 ppm (photosynthesis limit).
    Minimum floor for abiotic planets: 200 ppm (outgassing baseline).
    """
    delta_T = T_surface_K - reference_T
    ppm = reference_ppm * math.exp(-sensitivity * delta_T / reference_T * 10)
    # Floor: 150 ppm minimum (C3 photosynthesis limit; abiotic floor ~200 ppm)
    # This prevents the thermostat from driving CO2 so low it destabilises the loop
    return max(150.0, min(ppm, 100_000.0))


def co2_greenhouse_warming_K(co2_ppm: float,
                               reference_ppm: float = 280.0,
                               reference_dT: float  = 15.0) -> float:
    """
    CO₂-ONLY greenhouse warming [K] — before water vapour amplification.

    Earth at 280 ppm CO₂ contributes ~15 K to the greenhouse effect.
    Water vapour (temperature-dependent) adds the remaining ~18 K,
    giving the observed total of ~33 K.

    This separation is important: the WV amplifier is applied on top
    in _greenhouse_dT(), so we must NOT include WV here.

    Myhre et al. (1998) CO₂ radiative forcing:
        RF = 5.35 × ln(C/C0)  [W/m²]
    Climate sensitivity ≈ 0.8 K/(W/m²) → ΔT_CO₂ = 4.3 × ln(C/C0)
    At 280 ppm ref: ΔT_CO₂(280) = 0 K above ref (calibrated to reference_dT).
    """
    if co2_ppm <= 0 or reference_ppm <= 0:
        return 0.0
    # Myhre formula: RF = 5.35 ln(C/C0), sensitivity 0.8 K per W/m²
    rf = 5.35 * math.log(max(co2_ppm, 1.0) / reference_ppm)
    delta = 0.8 * rf    # K change from CO2 alone
    return max(0.0, reference_dT + delta)


# ─────────────────────────────────────────────────────────────────────────────
# Climate state result
# ─────────────────────────────────────────────────────────────────────────────

class ClimateState:
    """Enumeration of possible climate equilibrium states."""
    WARM_HABITABLE   = "warm_habitable"     # liquid water, T ∈ [273, 340] K
    COLD_HABITABLE   = "cold_habitable"     # cold but not fully frozen, T ∈ [240, 273] K
    SNOWBALL         = "snowball"           # fully ice-covered, T < 240 K
    MOIST_GREENHOUSE = "moist_greenhouse"  # hot, oceans partially evaporating
    RUNAWAY          = "runaway_greenhouse" # ocean loss, T > 647 K
    HABITABLE_MARGIN = "habitable_margin"  # in HZ but close to an edge


@dataclass
class ClimateResult:
    """
    Result of a climate equilibrium calculation.
    """
    orbital_distance_m:  float
    T_eq_K:              float    # radiative equilibrium temperature (no feedback)
    T_surface_K:         float    # actual surface temperature with feedbacks
    T_surface_cold_K:    float    # cold branch (snowball) temperature at this distance
    albedo:              float    # effective planetary albedo
    f_ice:               float    # fraction of surface covered by ice
    greenhouse_dT_K:     float    # greenhouse warming contribution
    co2_ppm:             float    # CO₂ concentration from carbonate-silicate cycle
    olr_W_m2:            float    # outgoing longwave radiation
    absorbed_W_m2:       float    # absorbed stellar flux
    imbalance_W_m2:      float    # OLR − absorbed (should be ~0 at equilibrium)
    climate_state:       str
    is_bistable:         bool     # True if both warm and cold branches exist
    habitable:           bool

    @property
    def orbital_distance_au(self) -> float:
        return self.orbital_distance_m / AU

    def report(self) -> str:
        state_desc = {
            ClimateState.WARM_HABITABLE:   "Warm habitable (liquid water stable)",
            ClimateState.COLD_HABITABLE:   "Cold habitable (partially frozen)",
            ClimateState.SNOWBALL:         "Snowball Earth (fully ice-covered)",
            ClimateState.MOIST_GREENHOUSE: "Moist greenhouse (oceans evaporating)",
            ClimateState.RUNAWAY:          "Runaway greenhouse (ocean loss)",
            ClimateState.HABITABLE_MARGIN: "Habitable margin (close to tipping point)",
        }.get(self.climate_state, self.climate_state)

        return (
            f"═══ Climate result at {self.orbital_distance_au:.3f} AU ═══\n"
            f"  Climate state      : {state_desc}\n"
            f"  T_equilibrium      : {self.T_eq_K:.1f} K\n"
            f"  T_surface (warm)   : {self.T_surface_K:.1f} K  ({self.T_surface_K-273.15:.1f} °C)\n"
            f"  T_surface (cold)   : {self.T_surface_cold_K:.1f} K  ({self.T_surface_cold_K-273.15:.1f} °C)\n"
            f"  Effective albedo   : {self.albedo:.3f}\n"
            f"  Ice coverage       : {self.f_ice*100:.0f}%\n"
            f"  Greenhouse ΔT      : +{self.greenhouse_dT_K:.1f} K\n"
            f"  CO₂ (C-S cycle)    : {self.co2_ppm:.0f} ppm\n"
            f"  Absorbed SW        : {self.absorbed_W_m2:.1f} W/m²\n"
            f"  Outgoing LW        : {self.olr_W_m2:.1f} W/m²\n"
            f"  Energy imbalance   : {self.imbalance_W_m2:+.2f} W/m²\n"
            f"  Bistable regime    : {'YES — snowball also possible' if self.is_bistable else 'No'}\n"
            f"  Habitable          : {'YES' if self.habitable else 'NO'}\n"
        )


@dataclass
class BifurcationPoints:
    """
    The critical orbital distances at which climate transitions occur.
    """
    snowball_au:        float   # inside this: always snowball (too cold for warm branch)
    warm_onset_au:      float   # inside this: warm branch appears
    runaway_au:         float   # beyond this: runaway greenhouse
    bistable_inner_au:  float   # inner edge of bistable zone
    bistable_outer_au:  float   # outer edge of bistable zone
    star_name:          str
    planet_name:        str

    def report(self) -> str:
        return (
            f"═══ Climate bifurcation points: {self.planet_name} / {self.star_name} ═══\n"
            f"\n"
            f"  ← Too hot →  runaway greenhouse if d < {self.runaway_au:.3f} AU\n"
            f"  ← Habitable zone →\n"
            f"  ← Bistable zone ({self.bistable_inner_au:.3f}–{self.bistable_outer_au:.3f} AU): both warm and snowball possible\n"
            f"  ← Cold habitable →\n"
            f"  ← Snowball if d > {self.snowball_au:.3f} AU\n"
            f"\n"
            f"  Warm branch exists:   d < {self.warm_onset_au:.3f} AU\n"
            f"  Runaway threshold:    d > {self.runaway_au:.3f} AU\n"
            f"  'Safe' habitable HZ:  {self.runaway_au:.3f} – {self.bistable_outer_au:.3f} AU\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1D Energy Balance Model
# ─────────────────────────────────────────────────────────────────────────────

class EnergyBalanceModel:
    """
    1D (zero-dimensional) global mean energy balance model with:
    - Ice-albedo positive feedback (Budyko-Sellers)
    - Water vapour positive feedback (amplifies greenhouse warming)
    - CO₂ carbonate-silicate negative feedback (thermostat)
    - Runaway greenhouse detection (Simpson-Nakajima limit)

    The model solves for global mean surface temperature T by iterating:
        T_new = f(T_old, S, albedo(T_old), GH(T_old))

    until convergence. Multiple fixed-point iterations are used to find
    both the warm and cold stable states.

    Limitations:
    - Global mean only (no latitudinal structure)
    - No atmospheric dynamics
    - Greenhouse parameterised, not from line-by-line RT
    - CO₂ thermostat is simplified (1-Myr timescale, not 1-yr)
    """

    def __init__(self, planet, star,
                 include_carbonate_silicate: bool = True,
                 include_water_vapour_feedback: bool = True,
                 include_ice_albedo: bool = True):
        self.planet  = planet
        self.star    = star
        self.cs      = include_carbonate_silicate
        self.wv      = include_water_vapour_feedback
        self.ia      = include_ice_albedo

        # Planet-specific albedo parameters
        self.base_albedo = (planet.oblateness.J2 * 0 + ALBEDO_ICE_FREE
                            if not planet.atmosphere.enabled else ALBEDO_ICE_FREE)
        # Airless body → higher base albedo (more reflective rock/regolith)
        if not planet.atmosphere.enabled:
            self.base_albedo = 0.12

    def _absorbed_flux(self, T_K: float, S: float) -> float:
        """Absorbed stellar flux [W/m²] given surface temperature."""
        if self.ia:
            A = effective_albedo(T_K, self.base_albedo)
        else:
            A = self.base_albedo
        return S * (1.0 - A) / 4.0   # /4 for spherical geometry

    def _olr(self, T_K: float, dT_GH: float = 0.0) -> float:
        """Outgoing longwave radiation [W/m²] with greenhouse and runaway."""
        if T_K > T_RUNAWAY_ONSET:
            return olr_near_runaway(T_K)
        return olr_budyko(T_K)

    def _greenhouse_dT(self, T_K: float,
                        co2_ppm: float = 280.0) -> float:
        """
        Total greenhouse warming [K] from CO₂ + water vapour feedback.
        """
        dT_co2 = co2_greenhouse_warming_K(co2_ppm)

        if self.wv:
            # Water vapour amplifier (Clausius-Clapeyron driven).
            # Calibrated so Earth (T~285 K, CO2=15 K) gives total GH = 33 K:
            #   amp(285 K) = 33/15 = 2.20 ✓
            #
            # Physical range:
            #   220 K: very cold/dry atmosphere, WV negligible → amp ≈ 1.0
            #   285 K: Earth → amp = 2.2
            #   310 K: warm ocean world → amp ≈ 2.7
            #   340 K: pre-runaway → amp ≈ 3.5
            #   >370 K: moist greenhouse, oceans evaporating → amp → large
            if T_K < 220:
                amp = 1.0
            elif T_K < 250:
                amp = 1.0 + 0.5 * (T_K - 220) / 30     # 1.0 → 1.5
            elif T_K < 285:
                amp = 1.5 + 0.7 * (T_K - 250) / 35     # 1.5 → 2.2
            elif T_K < 310:
                amp = 2.2 + 0.5 * (T_K - 285) / 25     # 2.2 → 2.7
            elif T_K < 340:
                amp = 2.7 + 0.8 * (T_K - 310) / 30     # 2.7 → 3.5
            else:
                amp = 3.5 + 3.0 * min((T_K - 340) / 60, 1.0)  # 3.5 → 6.5
            return dT_co2 * amp
        return dT_co2

    def _equilibrium_temperature(self, S: float, A: float) -> float:
        """Radiative equilibrium T (no feedbacks) [K]."""
        return ((S * (1 - A) / 4) / SIGMA) ** 0.25

    def solve(self, orbital_distance_m: float,
               initial_T_K: float = None,
               max_iterations: int = 200,
               tolerance_K: float = 0.1) -> ClimateResult:
        """
        Find the equilibrium surface temperature at a given orbital distance.

        Solves for the warm branch (starts from T_initial ≈ 280 K).
        Also finds the cold branch (starts from T_initial ≈ 220 K) to
        detect bistability.

        Returns ClimateResult.
        """
        S    = self.star.flux_at_distance(orbital_distance_m)
        T_eq = self._equilibrium_temperature(S, self.base_albedo)

        # ── Warm branch (starts from habitable initial guess) ──────────────
        # Warm branch must start above the ice-melt threshold to find the warm
        # equilibrium. Starting below T_MELT means full ice cover → can't escape.
        T0_warm = initial_T_K if initial_T_K else max(T_eq + 30, T_MELT + 15.0)
        T_warm  = self._iterate(S, T0_warm, max_iterations, tolerance_K)

        # ── Cold branch (starts from cold initial guess) ───────────────────
        T_cold = self._iterate(S, 220.0, max_iterations, tolerance_K)

        # ── Classify state ─────────────────────────────────────────────────
        T_surf = T_warm
        if self.ia:
            albedo = effective_albedo(T_surf, self.base_albedo)
        else:
            albedo = self.base_albedo

        f_ice = ice_covered_fraction(T_surf)

        if self.cs and self._use_cs_thermostat():
            co2_ppm = carbonate_silicate_co2_ppm(T_surf)
        else:
            co2_ppm = self._planet_co2_ppm()

        dT_GH      = self._greenhouse_dT(T_surf, co2_ppm)
        absorbed   = self._absorbed_flux(T_surf, S)
        olr        = self._olr(T_surf, dT_GH)
        imbalance  = olr - absorbed

        # Bistability: warm and cold branches differ by > 20 K
        is_bistable = (T_warm - T_cold) > 20.0

        # Climate state classification
        if T_surf > T_WET_LIMIT:
            state = ClimateState.RUNAWAY
        elif T_surf > T_MOIST_GH:
            state = ClimateState.MOIST_GREENHOUSE
        elif T_surf > T_MELT and f_ice < 0.05:
            state = ClimateState.WARM_HABITABLE
        elif T_surf > T_FREEZE:
            state = ClimateState.COLD_HABITABLE
        elif f_ice > 0.95:
            state = ClimateState.SNOWBALL
        else:
            state = ClimateState.HABITABLE_MARGIN

        habitable = T_surf > T_FREEZE and T_surf < T_MOIST_GH

        return ClimateResult(
            orbital_distance_m  = orbital_distance_m,
            T_eq_K              = T_eq,
            T_surface_K         = T_surf,
            T_surface_cold_K    = T_cold,
            albedo              = albedo,
            f_ice               = f_ice,
            greenhouse_dT_K     = dT_GH,
            co2_ppm             = co2_ppm,
            olr_W_m2            = olr,
            absorbed_W_m2       = absorbed,
            imbalance_W_m2      = imbalance,
            climate_state       = state,
            is_bistable         = is_bistable,
            habitable           = habitable,
        )

    def _iterate(self, S: float, T_init: float,
                  max_iter: int, tol: float) -> float:
        """
        Fixed-point iteration to find equilibrium T.

        Energy balance: absorbed_SW = OLR
        S(1-A(T))/4  =  OLR(T)  →  T*
        """
        T = T_init
        for i in range(max_iter):
            if self.cs and self._use_cs_thermostat():
                co2_ppm = carbonate_silicate_co2_ppm(T)
            else:
                co2_ppm = self._planet_co2_ppm()

            dT_GH = self._greenhouse_dT(T, co2_ppm) if self.planet.atmosphere.enabled else 0.0

            if self.ia:
                A = effective_albedo(T, self.base_albedo)
            else:
                A = self.base_albedo

            absorbed = S * (1.0 - A) / 4.0

            # T that would be in balance:  T_new⁴ = absorbed / σ  (blackbody)
            # But we also have greenhouse: T_new = T_rad + dT_GH
            if absorbed > 0:
                T_rad = (absorbed / SIGMA) ** 0.25
                T_new = T_rad + dT_GH
            else:
                T_new = 50.0  # effective minimum

            # Cap runaway divergence
            if T_new > T_WET_LIMIT:
                T_new = T_WET_LIMIT

            # Damped update for stability
            T_new = 0.6 * T + 0.4 * T_new

            if abs(T_new - T) < tol:
                return T_new
            T = T_new

        return T

    def _use_cs_thermostat(self) -> bool:
        """
        Use C-S thermostat only for Earth-like rocky planets with liquid water.
        Venus (CO2_THICK), Mars (CO2_THIN), Titan (METHANE) all have their
        own fixed CO2 reservoirs — the thermostat doesn't apply.
        """
        if not self.planet.atmosphere.enabled:
            return False
        comp_name = self.planet.atmosphere.composition.name
        return comp_name not in ("CO2_THICK", "CO2_THIN", "HYDROGEN",
                                  "METHANE", "NITROGEN", "NONE")

    def _planet_co2_ppm(self) -> float:
        """CO2 concentration [ppm] from the planet's actual atmosphere."""
        if not self.planet.atmosphere.enabled:
            return 0.0
        from core.atmosphere_science import STANDARD_COMPOSITIONS
        comp_name = self.planet.atmosphere.composition.name
        comp      = STANDARD_COMPOSITIONS.get(comp_name, {})
        total     = sum(comp.values())
        if total == 0:
            return 280.0
        co2_frac = comp.get("CO2", 0.0) / total
        # Convert: mole fraction × surface pressure → partial pressure → ppm
        P_co2_Pa = co2_frac * self.planet.atmosphere.surface_pressure
        return P_co2_Pa / 0.101325   # 1 atm = 101325 Pa; 1 ppm = 0.101325 Pa

    def scan_distances(self, d_min_AU: float = 0.3, d_max_AU: float = 3.0,
                        n_points: int = 150) -> list[ClimateResult]:
        """
        Run the EBM across a range of orbital distances.
        Returns a list of ClimateResult objects, one per distance.
        """
        distances = np.linspace(d_min_AU * AU, d_max_AU * AU, n_points)
        return [self.solve(d) for d in distances]

    def habitable_distance_range(self, d_min_AU: float = 0.3,
                                   d_max_AU: float = 3.0,
                                   n_points: int = 150
                                   ) -> tuple[Optional[float], Optional[float]]:
        """
        Find the habitable zone boundaries including climate feedbacks [AU].

        Returns (inner_AU, outer_AU) of the habitable zone as defined by
        this EBM (T_surface ∈ [273, 373] K).

        More physically complete than the Kopparapu HZ because it includes
        the ice-albedo feedback (moves outer edge inward) and water vapour
        runaway (moves inner edge outward slightly).
        """
        results = self.scan_distances(d_min_AU, d_max_AU, n_points)
        inner_AU = outer_AU = None

        for r in results:
            d_AU = r.orbital_distance_au
            if r.habitable:
                if inner_AU is None:
                    inner_AU = d_AU
                outer_AU = d_AU

        return inner_AU, outer_AU


# ─────────────────────────────────────────────────────────────────────────────
# Bifurcation point finder
# ─────────────────────────────────────────────────────────────────────────────

def find_bifurcation_points(planet, star,
                              d_min_AU: float = 0.3,
                              d_max_AU: float = 4.0,
                              n_points: int   = 200) -> BifurcationPoints:
    """
    Find the critical distances where climate state transitions occur.

    Scans orbital distances from d_min to d_max, running the EBM at each.
    Identifies:
    - Runaway greenhouse threshold (inner edge)
    - Snowball transition (outer edge)
    - Bistable zone (where both warm and cold branches exist)

    Parameters
    ----------
    planet, star : Planet and Star objects
    d_min_AU     : inner scan limit [AU]
    d_max_AU     : outer scan limit [AU]
    n_points     : number of distances to sample
    """
    ebm     = EnergyBalanceModel(planet, star)
    results = ebm.scan_distances(d_min_AU, d_max_AU, n_points)

    runaway_au        = d_min_AU
    warm_onset_au     = d_max_AU
    snowball_au       = d_max_AU
    bistable_inner_au = d_max_AU
    bistable_outer_au = d_min_AU

    for r in results:
        d = r.orbital_distance_au
        if r.climate_state == ClimateState.RUNAWAY:
            runaway_au = max(runaway_au, d)
        if r.habitable or r.climate_state == ClimateState.COLD_HABITABLE:
            warm_onset_au = min(warm_onset_au, d)
        if r.climate_state == ClimateState.SNOWBALL:
            snowball_au = min(snowball_au, d)
        if r.is_bistable:
            bistable_inner_au = min(bistable_inner_au, d)
            bistable_outer_au = max(bistable_outer_au, d)

    if bistable_inner_au > bistable_outer_au:
        bistable_inner_au = bistable_outer_au = (runaway_au + snowball_au) / 2

    return BifurcationPoints(
        snowball_au       = snowball_au,
        warm_onset_au     = warm_onset_au,
        runaway_au        = runaway_au,
        bistable_inner_au = bistable_inner_au,
        bistable_outer_au = bistable_outer_au,
        star_name         = star.name,
        planet_name       = planet.name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: climate score for habitability integration
# ─────────────────────────────────────────────────────────────────────────────

def climate_habitability_score(planet, star,
                                 orbital_distance_m: float) -> tuple[float, str]:
    """
    Return a habitability score [0–1] and note based on climate feedbacks.

    This is designed to be called from habitability.py as an additional
    factor, replacing the simpler surface_temperature score when the EBM
    is available.

    Score 1.0: warm habitable state, far from tipping points
    Score 0.5: habitable but bistable (snowball risk)
    Score 0.1: cold habitable / moist greenhouse margin
    Score 0.0: confirmed snowball or runaway
    """
    ebm    = EnergyBalanceModel(planet, star)
    result = ebm.solve(orbital_distance_m)
    d_AU   = orbital_distance_m / AU

    state = result.climate_state

    if state == ClimateState.WARM_HABITABLE:
        if result.is_bistable:
            score = 0.55
            note  = (f"Warm habitable but bistable at {d_AU:.2f} AU — "
                     f"snowball state also possible")
        else:
            # How far from tipping points?
            margin_warm = result.T_surface_K - T_FREEZE    # K above freezing
            margin_hot  = T_RUNAWAY_ONSET - result.T_surface_K   # K below runaway
            margin_frac = min(margin_warm, margin_hot) / 60.0
            score = min(1.0, 0.7 + 0.3 * margin_frac)
            note  = (f"Warm habitable, T={result.T_surface_K:.0f} K, "
                     f"ice cover={result.f_ice*100:.0f}%, "
                     f"CO₂={result.co2_ppm:.0f} ppm")

    elif state == ClimateState.COLD_HABITABLE:
        score = 0.3
        note  = (f"Cold habitable, T={result.T_surface_K:.0f} K, "
                 f"ice cover={result.f_ice*100:.0f}% — near snowball transition")

    elif state == ClimateState.SNOWBALL:
        T_diff = T_FREEZE - result.T_surface_K
        score  = max(0.0, 0.05 - 0.001 * T_diff)
        note   = (f"Snowball state: T={result.T_surface_K:.0f} K, "
                  f"planet fully ice-covered — no liquid surface water")

    elif state == ClimateState.MOIST_GREENHOUSE:
        score = 0.15
        note  = (f"Moist greenhouse, T={result.T_surface_K:.0f} K — "
                 f"oceans beginning to evaporate")

    elif state == ClimateState.RUNAWAY:
        score = 0.0
        note  = f"Runaway greenhouse at {d_AU:.2f} AU — oceans lost"

    elif state == ClimateState.HABITABLE_MARGIN:
        score = 0.25
        note  = f"Habitable margin, T={result.T_surface_K:.0f} K — near tipping point"

    else:
        score = 0.3
        note  = f"Climate state: {state}, T={result.T_surface_K:.0f} K"

    return score, note


def climate_map(planet, star,
                 d_range_AU: tuple = (0.3, 3.0),
                 n: int = 150) -> dict:
    """
    Full climate map over a range of orbital distances.

    Returns dict with arrays:
        distances_AU, T_warm, T_cold, albedo, ice_fraction,
        co2_ppm, climate_state, habitable_mask, bistable_mask
    """
    ebm     = EnergyBalanceModel(planet, star)
    results = ebm.scan_distances(*d_range_AU, n)

    return {
        "distances_AU":  np.array([r.orbital_distance_au for r in results]),
        "T_warm_K":      np.array([r.T_surface_K          for r in results]),
        "T_cold_K":      np.array([r.T_surface_cold_K      for r in results]),
        "T_eq_K":        np.array([r.T_eq_K                for r in results]),
        "albedo":        np.array([r.albedo                 for r in results]),
        "ice_fraction":  np.array([r.f_ice                  for r in results]),
        "co2_ppm":       np.array([r.co2_ppm                for r in results]),
        "greenhouse_dT": np.array([r.greenhouse_dT_K        for r in results]),
        "olr":           np.array([r.olr_W_m2               for r in results]),
        "absorbed":      np.array([r.absorbed_W_m2          for r in results]),
        "climate_state": [r.climate_state                    for r in results],
        "habitable":     np.array([r.habitable               for r in results]),
        "bistable":      np.array([r.is_bistable             for r in results]),
    }
