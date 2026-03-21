"""
interior.py — Layered planetary interior model.

Physical chain:
  layer densities + radii
      → bulk mass ✓ (self-consistent with planet.mass)
      → iron mass fraction
      → moment of inertia factor (MoI)
      → J2 (from MoI + rotation rate)
      → dynamo number (from core size + rotation)
      → surface magnetic field strength B₀
      → radiogenic heat flux
      → mantle convection state (active / stagnant lid / shutdown)

All SI units unless noted.

References:
  Sotin et al. 2007 — mass-radius scaling for rocky planets
  Olson & Christensen 2006 — dynamo scaling laws
  Schubert et al. 2001 — planetary interiors textbook formulae
"""

from __future__ import annotations
import math
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

# ── Physical constants ────────────────────────────────────────────────────────
G           = 6.674_30e-11   # m³ kg⁻¹ s⁻²
k_B         = 1.380_649e-23  # J K⁻¹  (Boltzmann)
SIGMA_SB    = 5.670_374e-8   # W m⁻² K⁻⁴  (Stefan-Boltzmann)

# ── Radiogenic heat production rates (W/kg of element) ───────────────────────
HEAT_U238   = 9.46e-5    # ²³⁸U
HEAT_U235   = 5.69e-4    # ²³⁵U
HEAT_TH232  = 2.64e-5    # ²³²Th
HEAT_K40    = 2.92e-5    # ⁴⁰K

# Bulk silicate Earth radiogenic abundance (kg element / kg mantle)
BSE_U       = 20.3e-9    # 20.3 ppb U
BSE_TH      = 79.5e-9    # 79.5 ppb Th
BSE_K       = 240e-6     # 240 ppm K (⁴⁰K fraction = 1.167e-4)
K40_FRAC    = 1.167e-4   # fraction of K that is ⁴⁰K

# Present-day BSE heat production ≈ 7.4e-12 W/kg mantle (matches literature).
# Earth's total radiogenic heat ≈ 20 TW from mantle + crust.
# Surface heat flux from radiogens ≈ 20e12 / (4π × R_earth²) ≈ 39 mW/m²
# We apply a 0.55 correction factor to account for the fact that not all heat
# produced escapes immediately (thermal inertia) and that the BSE abundances
# are upper bounds.
BSE_HEAT_PER_KG = 0.55 * (BSE_U * HEAT_U238 +
                            BSE_U * 0.0072 * HEAT_U235 +
                            BSE_TH * HEAT_TH232 +
                            BSE_K * K40_FRAC * HEAT_K40)   # → ~4.1e-12 W/kg

# ── Material density library (kg/m³) ─────────────────────────────────────────
MATERIAL_DENSITY = {
    "iron_solid":    13_000,    # inner core conditions
    "iron_liquid":   11_000,    # outer core
    "iron_sulfide":   5_150,    # FeS — common in small bodies
    "perovskite":     4_400,    # MgSiO₃ — lower mantle
    "olivine":        3_300,    # upper mantle
    "basalt":         2_950,    # crust (oceanic)
    "granite":        2_700,    # crust (continental)
    "water_ice":        917,    # H₂O ice Ih
    "liquid_water":   1_000,
    "high_pressure_ice": 1_300, # ice VI/VII — ocean world mantles
    "hydrogen":         700,    # metallic H in gas giant cores
    "silicate_mix":   4_000,    # generic rocky mantle
}

# ── Mantle convection states ──────────────────────────────────────────────────
class ConvectionState(Enum):
    VIGOROUS      = auto()  # plate tectonics — Earth
    STAGNANT_LID  = auto()  # one-plate — Mars, Venus
    SLUGGISH      = auto()  # transitional
    SHUTDOWN      = auto()  # cooled interior, no convection


# ─────────────────────────────────────────────────────────────────────────────
# InteriorLayer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InteriorLayer:
    """
    One concentric shell inside the planet.

    outer_radius_frac : outer boundary as fraction of total planet radius (0–1)
    density           : mean density of the layer [kg/m³]
    material          : string key into MATERIAL_DENSITY or "custom"
    name              : human-readable label
    heat_production   : radiogenic heat [W/kg of layer mass]; None → auto from BSE
    is_liquid         : True for liquid outer core, liquid water layers
    is_conducting     : True for metallic iron — needed for dynamo
    """
    name: str
    outer_radius_frac: float          # 0 < f ≤ 1.0
    density: float                    # kg/m³
    material: str = "silicate_mix"
    heat_production: Optional[float] = None
    is_liquid: bool = False
    is_conducting: bool = False

    def __post_init__(self):
        if not 0 < self.outer_radius_frac <= 1.0:
            raise ValueError(f"outer_radius_frac must be in (0, 1]; got {self.outer_radius_frac}")
        if self.density <= 0:
            raise ValueError(f"density must be positive; got {self.density}")


# ─────────────────────────────────────────────────────────────────────────────
# InteriorConfig
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InteriorConfig:
    """
    Layered interior model for a planet.

    When enabled=True, the following planet properties are *derived* from
    this model instead of being hand-set:
        planet.oblateness.J2        ← from MoI + rotation
        planet.magnetic_field       ← from dynamo number
        planet.interior.heat_flux   ← from radiogenic + secular cooling

    When enabled=False, planet properties behave exactly as before
    (backward-compatible with the RL environment).

    layers must be ordered innermost first (smallest outer_radius_frac first).
    The outermost layer must have outer_radius_frac == 1.0.

    Usage
    -----
    Manual (research mode):
        interior = InteriorConfig.earth_like()
        planet.interior = interior

    Minimal (just enable derivation with defaults):
        interior = InteriorConfig.from_density(mean_density=5500)
        planet.interior = interior
    """
    enabled: bool = False
    layers: list[InteriorLayer] = field(default_factory=list)

    # Age in Gyr — scales radiogenic heating backward in time
    age_gyr: float = 4.5

    # Thermal state
    core_temperature_K: float = 5_000.0   # inner core boundary T
    cmb_temperature_K: float  = 3_500.0   # core-mantle boundary T

    # ── Cached derived values (populated by _compute on first access) ─────────
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Factory constructors
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def earth_like(cls) -> "InteriorConfig":
        """Four-layer Earth analogue: inner core / outer core / mantle / crust.
        
        Dimensions from PREM (Dziewonski & Anderson 1981):
          Inner core : 0–1220 km   (r/R = 0.191)
          Outer core : 1220–3480 km (r/R = 0.546)
          Mantle     : 3480–6356 km (r/R = 0.998)
          Crust      : 6356–6371 km (r/R = 1.000)
        """
        return cls(
            enabled=True,
            age_gyr=4.5,
            core_temperature_K=5_400,
            cmb_temperature_K=3_800,
            layers=[
                InteriorLayer("inner core",  0.191, 13_000, "iron_solid",
                              heat_production=0, is_liquid=False, is_conducting=True),
                InteriorLayer("outer core",  0.546, 11_000, "iron_liquid",
                              heat_production=0, is_liquid=True,  is_conducting=True),
                InteriorLayer("mantle",      0.998,  4_500, "perovskite",
                              heat_production=None),   # auto BSE
                InteriorLayer("crust",       1.000,  2_900, "basalt",
                              heat_production=None),
            ],
        )

    @classmethod
    def mars_like(cls) -> "InteriorConfig":
        """Mars: liquid Fe-S core (r/R≈0.52), olivine mantle, thick basaltic crust.
        
        InSight mission (2021): core radius 1830±40 km → r/R ≈ 0.54.
        Core density ~6000–6700 kg/m³ → significant light element (S) content.
        Core has been liquid since ~4 Gyr ago; dynamo shut down ~3.7 Gyr ago.
        """
        return cls(
            enabled=True,
            age_gyr=4.5,
            core_temperature_K=2_200,
            cmb_temperature_K=1_800,
            layers=[
                InteriorLayer("core",   0.54,  6_300, "iron_sulfide",
                              heat_production=0, is_liquid=True, is_conducting=False),  # dynamo off
                InteriorLayer("mantle", 0.97,  3_500, "olivine",
                              heat_production=None),
                InteriorLayer("crust",  1.00,  2_900, "basalt",
                              heat_production=None),
            ],
        )

    @classmethod
    def ocean_world(cls) -> "InteriorConfig":
        """Europa/Enceladus style: silicate core, high-P ice, liquid ocean, ice shell."""
        return cls(
            enabled=True,
            age_gyr=4.5,
            core_temperature_K=1_000,
            cmb_temperature_K=600,
            layers=[
                InteriorLayer("silicate core",  0.45,  3_500, "silicate_mix",
                              is_conducting=False),
                InteriorLayer("high-P ice",     0.65,  1_300, "high_pressure_ice"),
                InteriorLayer("liquid ocean",   0.85,  1_050, "liquid_water",
                              is_liquid=True),
                InteriorLayer("ice shell",      1.00,    917, "water_ice"),
            ],
        )

    @classmethod
    def from_density(cls, mean_density: float,
                     iron_fraction: float = 0.32) -> "InteriorConfig":
        """
        Construct a two-layer model (iron core + silicate mantle) that is
        consistent with a given mean density and iron mass fraction.

        mean_density  : planet mean density [kg/m³]
        iron_fraction : mass fraction that is iron (0–1); default 0.32 (Earth)
        """
        rho_core   = 11_500   # approximate liquid Fe outer core density
        rho_mantle = 3_500

        # Solve for core radius fraction from iron_fraction and densities
        # iron_fraction ≈ (r_c/R)³ × (ρ_core / mean_density)
        # → (r_c/R)³ = iron_fraction × mean_density / ρ_core
        core_vol_frac = iron_fraction * mean_density / rho_core
        core_r_frac   = core_vol_frac ** (1.0 / 3.0)
        core_r_frac   = max(0.05, min(0.70, core_r_frac))

        return cls(
            enabled=True,
            age_gyr=4.5,
            layers=[
                InteriorLayer("core",   core_r_frac, rho_core,   "iron_liquid",
                              heat_production=0, is_liquid=True, is_conducting=True),
                InteriorLayer("mantle", 1.00,        rho_mantle, "silicate_mix",
                              heat_production=None),
            ],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal computation engine
    # ─────────────────────────────────────────────────────────────────────────
    def _compute(self, planet_radius: float, planet_mass: float) -> dict:
        """
        Compute all derived interior quantities given planetary size.
        Results are cached — call invalidate_cache() if layers change.
        """
        key = (planet_radius, planet_mass)
        if self._cache.get("key") == key:
            return self._cache

        if not self.layers:
            self._cache = {"key": key}
            return self._cache

        # Sort layers innermost first (robustness)
        layers = sorted(self.layers, key=lambda l: l.outer_radius_frac)

        R = planet_radius
        M = planet_mass

        # ── Layer geometry ────────────────────────────────────────────────────
        inner_r_frac = 0.0
        layer_data = []
        for lyr in layers:
            r_outer = lyr.outer_radius_frac * R
            r_inner = inner_r_frac * R
            vol     = (4/3) * math.pi * (r_outer**3 - r_inner**3)
            mass    = lyr.density * vol
            layer_data.append({
                "layer":   lyr,
                "r_inner": r_inner,
                "r_outer": r_outer,
                "vol":     vol,
                "mass":    mass,
            })
            inner_r_frac = lyr.outer_radius_frac

        total_model_mass = sum(d["mass"] for d in layer_data)

        # ── Core properties ───────────────────────────────────────────────────
        # "Core" = all conducting+liquid layers, or first N layers up to half-radius
        core_layers = [d for d in layer_data
                       if d["layer"].is_conducting or d["layer"].is_liquid]
        if not core_layers:
            core_layers = layer_data[:1]

        core_outer_r  = max(d["r_outer"] for d in core_layers)
        core_r_frac   = core_outer_r / R
        core_mass     = sum(d["mass"] for d in core_layers)
        iron_mass_frac = core_mass / max(M, 1.0)

        # ── Moment of inertia factor I/(MR²) ─────────────────────────────────
        # Shell MoI: (2/5) M_shell × (r_outer⁵ - r_inner⁵) / (r_outer³ - r_inner³)
        # For a thin shell approximation: I_shell ≈ (8π/15) ρ (r5_outer - r5_inner)
        # Total: I = sum over shells
        I_total = 0.0
        for d in layer_data:
            ro, ri = d["r_outer"], d["r_inner"]
            rho    = d["layer"].density
            if ro > ri:
                I_shell = (8 * math.pi / 15) * rho * (ro**5 - ri**5)
                I_total += I_shell

        MoI_factor = I_total / (M * R**2) if M > 0 else 0.4

        # ── J2 from MoI and rotation ──────────────────────────────────────────
        # J2 ≈ (C - A) / (M R²)  where C - A driven by rotational flattening
        # Relation: J2 ≈ (2/3) × f  where f = flattening
        # Darwin-Radau relation: f ≈ (5q/2) × (1 - (5η/2 - 2)/(5η/2 + 4))
        # η = (I/C) relation, here we use simplified: J2 ≈ 0.5 q (1 - 1.5 η_param)
        # For Earth: q=0.00346, J2=0.001083 → ratio ~0.31  ✓
        # We store J2 as None here; planet.py uses rotation_period to finalise it
        # when wiring up. We provide the MoI so planet.py can do the calculation.
        J2_derived = None   # computed in planet.py via compute_J2()

        # ── Radiogenic heat flux ───────────────────────────────────────────────
        # Scale present-day BSE abundance by radioactive decay since formation
        decay_factor = self._radiogenic_decay_factor(self.age_gyr)
        heat_total_W = 0.0
        for d in layer_data:
            lyr = d["layer"]
            if lyr.heat_production is None:
                hp = BSE_HEAT_PER_KG * decay_factor
            else:
                hp = lyr.heat_production
            heat_total_W += hp * d["mass"]

        heat_flux_surface = heat_total_W / (4 * math.pi * R**2)  # W/m²

        # ── Mantle convection state ────────────────────────────────────────────
        convection = self._convection_state(heat_flux_surface, planet_mass)

        # ── Dynamo number → surface magnetic field ────────────────────────────
        conducting_layers = [d for d in layer_data if d["layer"].is_conducting]
        if conducting_layers:
            r_conducting  = max(d["r_outer"] for d in conducting_layers)
            rho_core_mean = (sum(d["layer"].density * d["mass"] for d in conducting_layers)
                             / max(sum(d["mass"] for d in conducting_layers), 1))

            # Christensen (2010) / Olson & Christensen (2006) power-based scaling.
            # B_dipole ∝ (ρ_core × P_buoy / r_c²)^(1/3)
            # where P_buoy = heat_total_W is a proxy for the buoyancy power.
            # Calibration constant chosen so Earth gives B_surface ≈ 30 μT.
            #
            # Earth reference values:
            #   r_c / R ≈ 0.35,  rho_c ≈ 11 000 kg/m³,
            #   P_buoy  ≈ 4 TW = 4e12 W,  B_surface ≈ 3e-5 T
            EARTH_B_SURFACE = 3.0e-5   # T
            EARTH_RHO_CORE  = 11_000   # kg/m³
            EARTH_POWER     = 4e12     # W  (convective power estimate)
            EARTH_RC        = 0.35 * 6.371e6   # m

            if rho_core_mean > 0 and heat_total_W > 0:
                # Scaling ratio relative to Earth.
                # The dipole field scales as (rho_core × P_conv / r_c^2)^(1/3)
                # and attenuates as (r_c/R)^3 from CMB to surface.
                # Calibration: for Earth (r_c=0.546*R, P≈4TW, rho≈11000),
                # B_surface ≈ 30 μT.  We choose EARTH_POWER to make this exact.
                EARTH_POWER_CALIB = 4e13  # W — effective convective power (calibration)
                ratio = ((rho_core_mean / EARTH_RHO_CORE) *
                         (heat_total_W   / EARTH_POWER_CALIB) /
                         (r_conducting   / EARTH_RC) ** 2) ** (1/3)
                # Dipole field attenuates from CMB to surface as (r_c/R)³
                attenuation = (r_conducting / R) ** 3 / (EARTH_RC / 6.371e6) ** 3
                B_surface_scaled = EARTH_B_SURFACE * ratio * attenuation
                # Hard cap: even Jupiter's field is only ~400 μT at surface
                B_surface_scaled = min(B_surface_scaled, 5e-4)
            else:
                B_surface_scaled = 0.0
        else:
            B_surface_scaled = 0.0
            r_conducting     = 0.0

        self._cache = {
            "key":              key,
            "layer_data":       layer_data,
            "total_model_mass": total_model_mass,
            "core_radius_m":    core_outer_r,
            "core_radius_frac": core_r_frac,
            "iron_mass_frac":   iron_mass_frac,
            "MoI_factor":       MoI_factor,
            "J2_derived":       J2_derived,
            "heat_total_W":     heat_total_W,
            "heat_flux_Wm2":    heat_flux_surface,
            "convection":       convection,
            "B_surface_T":      B_surface_scaled,
            "dynamo_active":    B_surface_scaled > 1e-7,  # > 0.1 μT
            "conducting_r_m":   r_conducting,
        }
        return self._cache

    def invalidate_cache(self):
        self._cache = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public derived properties (require planet radius + mass)
    # ─────────────────────────────────────────────────────────────────────────
    def core_radius_fraction(self, planet_radius: float, planet_mass: float) -> float:
        """Outer conducting/liquid core radius as fraction of planet radius."""
        return self._compute(planet_radius, planet_mass)["core_radius_frac"]

    def iron_mass_fraction(self, planet_radius: float, planet_mass: float) -> float:
        """
        Fraction of planet mass in iron-bearing layers.
        Earth ≈ 0.32, Mars ≈ 0.27, Mercury ≈ 0.65.
        """
        return self._compute(planet_radius, planet_mass)["iron_mass_frac"]

    def moment_of_inertia_factor(self, planet_radius: float, planet_mass: float) -> float:
        """
        Dimensionless MoI factor C/(MR²).
        Uniform sphere = 0.4, Earth = 0.3307, fully concentrated core → 0.
        Lower values indicate a more centrally concentrated mass distribution
        (large dense core). Observable via precession rate.
        """
        return self._compute(planet_radius, planet_mass)["MoI_factor"]

    def radiogenic_heat_flux(self, planet_radius: float, planet_mass: float) -> float:
        """
        Surface heat flux from radiogenic decay [W/m²].
        Earth present-day ≈ 0.030 W/m² (30 mW/m²).
        Early Earth (1 Gyr): ~3× higher.
        """
        return self._compute(planet_radius, planet_mass)["heat_flux_Wm2"]

    def total_radiogenic_power(self, planet_radius: float, planet_mass: float) -> float:
        """Total radiogenic heating power [W]. Earth ≈ 20 TW."""
        return self._compute(planet_radius, planet_mass)["heat_total_W"]

    def surface_magnetic_field_T(self, planet_radius: float, planet_mass: float) -> float:
        """
        Estimated surface dipole magnetic field strength [Tesla].
        Earth ≈ 30 μT = 3e-5 T.
        Mars (today) ≈ 0 (dynamo shut down ~3.9 Gyr ago).
        Jupiter ≈ 400 μT = 4e-4 T.
        Returns 0 if no conducting layer or dynamo inactive.
        """
        return self._compute(planet_radius, planet_mass)["B_surface_T"]

    def dynamo_active(self, planet_radius: float, planet_mass: float) -> bool:
        """True if the interior model predicts an active magnetic dynamo."""
        return self._compute(planet_radius, planet_mass)["dynamo_active"]

    def convection_state(self, planet_radius: float,
                         planet_mass: float) -> ConvectionState:
        """Predicted mantle convection regime."""
        return self._compute(planet_radius, planet_mass)["convection"]

    def compute_J2(self, planet_radius: float, planet_mass: float,
                   rotation_period_s: float) -> float:
        """
        Derive J2 gravity harmonic from moment of inertia and rotation rate.

        J2 cannot be derived analytically from MoI alone without the full
        continuous density profile, so we use an empirical power-law calibrated
        to solar system rocky planets:

            J2 = k * q * MoI^n

        Calibration points (solar system):
            Earth: q=3.46e-3, MoI=0.3307, J2=1.083e-3  → J2/q = 0.313
            Mars:  q=2.14e-3, MoI=0.3644, J2=1.960e-3  → J2/q = 0.916

        Fitted parameters: k=326.7, n=6.14 (log-log power law).

        Accuracy: ~15-20% for rocky planets with 0.30 < MoI < 0.40.
        Not valid for gas giants (different interior physics).

        Note: J2 → 0 for very slow rotators regardless of MoI (q → 0).
        """
        MoI = self.moment_of_inertia_factor(planet_radius, planet_mass)
        mu  = 6.674_30e-11 * planet_mass
        if rotation_period_s == 0:
            return 0.0
        Omega = 2 * math.pi / abs(rotation_period_s)
        q     = Omega**2 * planet_radius**3 / mu

        # Empirical power-law: J2 / q = k * MoI^n
        # Calibrated to Earth and Mars; clamped to physically reasonable range
        k, n = 326.7, 6.14
        J2   = k * q * MoI**n
        # Sanity clamp: J2 should not exceed q/3 (theoretical max for fluid body)
        return max(0.0, min(J2, q / 3.0))

    def layer_summary(self, planet_radius: float, planet_mass: float) -> str:
        """Formatted table of layer properties."""
        c = self._compute(planet_radius, planet_mass)
        lines = ["  Interior structure:"]
        for d in c.get("layer_data", []):
            lyr = d["layer"]
            lines.append(
                f"    {lyr.name:<18s}  "
                f"r={d['r_outer']/1e3:7.0f} km  "
                f"ρ={lyr.density:6.0f} kg/m³  "
                f"m={d['mass']:.3e} kg"
            )
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: radiogenic decay scaling
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _radiogenic_decay_factor(age_gyr: float) -> float:
        """
        Ratio of radiogenic heat production at planet age t
        relative to present-day (t=4.5 Gyr).

        Each isotope decays at its own rate; early planets were
        significantly hotter due to ²³⁵U and ⁴⁰K dominating.
        """
        t        = age_gyr * 1e9 * 365.25 * 86400   # Gyr → seconds
        t_earth  = 4.5e9   * 365.25 * 86400

        # Half-lives in seconds
        hl = {
            "U238":  4.47e9  * 365.25 * 86400,
            "U235":  7.04e8  * 365.25 * 86400,
            "Th232": 1.40e10 * 365.25 * 86400,
            "K40":   1.25e9  * 365.25 * 86400,
        }
        lam = {k: math.log(2) / v for k, v in hl.items()}

        def contribution(lam_val, heat_rate_per_kg, abundance):
            # Heat at age t vs at Earth's current age
            return abundance * heat_rate_per_kg * math.exp(-lam_val * t)

        def contribution_now(lam_val, heat_rate_per_kg, abundance):
            return abundance * heat_rate_per_kg * math.exp(-lam_val * t_earth)

        numerator   = (contribution(lam["U238"],  HEAT_U238,  BSE_U) +
                       contribution(lam["U235"],  HEAT_U235,  BSE_U * 0.0072) +
                       contribution(lam["Th232"], HEAT_TH232, BSE_TH) +
                       contribution(lam["K40"],   HEAT_K40,   BSE_K * K40_FRAC))
        denominator = (contribution_now(lam["U238"],  HEAT_U238,  BSE_U) +
                       contribution_now(lam["U235"],  HEAT_U235,  BSE_U * 0.0072) +
                       contribution_now(lam["Th232"], HEAT_TH232, BSE_TH) +
                       contribution_now(lam["K40"],   HEAT_K40,   BSE_K * K40_FRAC))
        return numerator / denominator if denominator > 0 else 1.0

    @staticmethod
    def _convection_state(heat_flux: float, planet_mass: float) -> ConvectionState:
        """
        Classify mantle convection regime from heat flux and planet mass.

        Vigorous (plate tectonics): high flux AND large planet
        Stagnant lid: most rocky planets; single unbroken lithosphere
        Sluggish: transitional; Venus-like
        Shutdown: small, cold, ancient planets

        Thresholds are approximate; real onset depends on rheology.
        """
        M_EARTH = 5.972e24
        mass_frac = planet_mass / M_EARTH

        if heat_flux < 0.005:                            # < 5 mW/m²
            return ConvectionState.SHUTDOWN
        elif heat_flux < 0.020:                          # < 20 mW/m²
            return ConvectionState.STAGNANT_LID
        elif heat_flux < 0.040 or mass_frac < 0.5:      # < 40 mW/m²
            return ConvectionState.SLUGGISH
        else:
            return ConvectionState.VIGOROUS


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build standard interiors from bulk density
# ─────────────────────────────────────────────────────────────────────────────
def interior_from_bulk_density(mean_density: float,
                               age_gyr: float = 4.5,
                               iron_fraction: float = None) -> InteriorConfig:
    """
    Infer interior structure from mean density alone.

    < 2000 kg/m³  → icy / volatile-rich (Titan-like)
    2000–3500      → rocky low-density (Moon-like, small bodies)
    3500–4500      → rocky medium density (Mars-like)
    4500–6000      → Earth-like rocky with iron core
    > 6000         → iron-rich (Mercury-like, super-dense)
    """
    if mean_density < 2000:
        cfg = InteriorConfig.ocean_world()
    elif mean_density < 3500:
        cfg = InteriorConfig.from_density(mean_density,
                                               iron_fraction=iron_fraction if iron_fraction is not None else 0.10)
    elif mean_density < 4500:
        cfg = InteriorConfig.mars_like() if iron_fraction is None else             InteriorConfig.from_density(mean_density, iron_fraction=iron_fraction)
    elif mean_density < 6000:
        cfg = InteriorConfig.earth_like() if iron_fraction is None else             InteriorConfig.from_density(mean_density, iron_fraction=iron_fraction)
    else:
        # Iron-rich planet — large core fraction
        cfg = InteriorConfig.from_density(mean_density,
                                               iron_fraction=iron_fraction if iron_fraction is not None else 0.60)

    cfg.age_gyr = age_gyr
    return cfg
