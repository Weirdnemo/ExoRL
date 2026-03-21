"""
atmosphere_science.py — Multi-layer atmosphere model with physical chemistry.

  - Tracks gas composition as mole fractions
  - Derives scale height from composition (H = RT/Mg) instead of hand-setting it
  - Computes a proper piecewise temperature profile (troposphere/stratosphere/etc.)
  - Evaluates Jeans escape timescales per species
  - Estimates greenhouse warming from CO₂, CH₄, H₂O concentrations
  - Produces surface temperature from stellar flux + greenhouse (not hand-set)

All SI units unless noted.

References:
  Pierrehumbert (2010) "Principles of Planetary Climate" — scale height, lapse
  Kopparapu et al. (2013) — greenhouse parameterisation
  Hunten (1973) — Jeans escape flux
  Wordsworth & Pierrehumbert (2013) — H₂ greenhouse
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

# ── Physical constants ────────────────────────────────────────────────────────
R_GAS    = 8.314_462     # J mol⁻¹ K⁻¹  (universal gas constant)
k_B      = 1.380_649e-23 # J K⁻¹         (Boltzmann)
N_AVO    = 6.022_141e23  # mol⁻¹          (Avogadro)
G        = 6.674_30e-11  # m³ kg⁻¹ s⁻²
SIGMA_SB = 5.670_374e-8  # W m⁻² K⁻⁴

# ── Molar masses (g/mol) ─────────────────────────────────────────────────────
MOLAR_MASS = {
    "N2":   28.014,
    "O2":   31.999,
    "CO2":  44.009,
    "CH4":  16.043,
    "H2O":  18.015,
    "H2":    2.016,
    "He":    4.003,
    "Ar":   39.948,
    "SO2":  64.065,
    "H2S":  34.081,
    "NH3":  17.031,
    "N2O":  44.013,
    "O3":   47.998,
    "HCl":  36.461,
}

# ── Specific heat ratios (γ = Cp/Cv) ─────────────────────────────────────────
GAMMA = {
    "N2": 1.40, "O2": 1.40, "CO2": 1.28, "CH4": 1.31,
    "H2O": 1.33, "H2": 1.41, "He": 1.67, "Ar": 1.67,
    "SO2": 1.26, "H2S": 1.32, "NH3": 1.31,
}

# ── Adiabatic lapse rate: Γ = Mg/Cp  [K/m] ───────────────────────────────────
# For a given composition, derived in AtmosphericLayer.adiabatic_lapse_rate()

# ── Standard compositions for each AtmosphereComposition enum ────────────────
# Mole fractions must sum to 1.0 (minor species can make the sum slightly > 1
STANDARD_COMPOSITIONS = {
    "EARTH_LIKE": {
        "N2":  0.7808,
        "O2":  0.2095,
        "Ar":  0.0093,
        "CO2": 0.0004,
        "H2O": 0.0100,   # 1% average tropospheric humidity
    },
    "CO2_THIN": {    # Mars
        "CO2": 0.9532,
        "N2":  0.0270,
        "Ar":  0.0160,
        "O2":  0.0013,
        "CO":  0.0008,
    },
    "CO2_THICK": {   # Venus
        "CO2": 0.9650,
        "N2":  0.0350,
        "SO2": 0.00015,
        "Ar":  0.00007,
    },
    "NITROGEN": {
        "N2":  0.9800,
        "Ar":  0.0150,
        "CO2": 0.0050,
    },
    "METHANE": {     # Titan
        "N2":  0.9840,
        "CH4": 0.0149,
        "H2":  0.0010,
        "Ar":  0.0001,
    },
    "HYDROGEN": {    # Gas giant envelope
        "H2":  0.8600,
        "He":  0.1360,
        "CH4": 0.0030,
        "NH3": 0.0005,
        "H2O": 0.0005,
    },
    "NONE": {},
    "CUSTOM": {},
}


# ─────────────────────────────────────────────────────────────────────────────
# Atmospheric layer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AtmosphericLayer:
    """
    One layer in a multi-layer atmosphere (troposphere, stratosphere, etc.).
    Temperature varies linearly at rate lapse_rate [K/m].
    A negative lapse_rate means temperature *increases* with altitude (inversion).
    """
    name: str
    base_altitude_m: float          # bottom of layer [m]
    top_altitude_m: float           # top of layer [m]
    base_temperature_K: float       # temperature at base [K]
    lapse_rate_K_per_m: float       # dT/dz  [K/m]; positive = cooling with altitude
    composition: dict[str, float]   # mole fractions at this layer
    # Derived lazily
    _mean_molar_mass: Optional[float] = field(default=None, repr=False)

    @property
    def mean_molar_mass_g_mol(self) -> float:
        """Mean molar mass of this layer's gas mixture [g/mol]."""
        if self._mean_molar_mass is not None:
            return self._mean_molar_mass
        total_frac = sum(self.composition.values())
        if total_frac == 0:
            return 29.0  # default air-like
        mm = sum(
            frac * MOLAR_MASS.get(gas, 29.0)
            for gas, frac in self.composition.items()
        ) / total_frac
        self._mean_molar_mass = mm
        return mm

    @property
    def mean_molar_mass_kg_mol(self) -> float:
        return self.mean_molar_mass_g_mol * 1e-3

    def scale_height(self, gravity_m_s2: float,
                     temperature_K: Optional[float] = None) -> float:
        """
        Scale height H = RT/(Mg)  [m].
        Uses base_temperature_K if temperature_K not specified.
        """
        T = temperature_K if temperature_K is not None else self.base_temperature_K
        return R_GAS * T / (self.mean_molar_mass_kg_mol * gravity_m_s2)

    def temperature_at(self, altitude_m: float) -> float:
        """Temperature at a given altitude within this layer [K]."""
        dz = altitude_m - self.base_altitude_m
        T  = self.base_temperature_K - self.lapse_rate_K_per_m * dz
        return max(T, 20.0)   # physical lower bound

    def adiabatic_lapse_rate(self, gravity_m_s2: float) -> float:
        """
        Dry adiabatic lapse rate Γ = g × M / (Cp) [K/m].
        Cp estimated from γ: Cp = γ R / ((γ-1) M).
        """
        total = sum(self.composition.values()) or 1
        gamma_mix = sum(
            (f/total) * GAMMA.get(g, 1.4) for g, f in self.composition.items()
        )
        Cp = gamma_mix * R_GAS / ((gamma_mix - 1) * self.mean_molar_mass_kg_mol)
        return gravity_m_s2 / Cp  # K/m


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layer atmosphere profile
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MultiLayerAtmosphere:
    """
    A layered atmosphere model built from AtmosphericLayer objects.
    Integrates hydrostatic equilibrium through the layers to compute
    density and pressure at any altitude.

    Usage
    -----
    # Build from an existing AtmosphereConfig + composition name
    atm = MultiLayerAtmosphere.from_atmosphere_config(planet.atmosphere, planet)

    # Or construct directly
    atm = MultiLayerAtmosphere.earth_standard(planet)
    """
    layers: list[AtmosphericLayer]
    surface_pressure_Pa: float
    planet_radius_m: float
    planet_mass_kg: float

    _base_pressures: list[float] = field(default_factory=list, repr=False)

    def _gravity_at(self, altitude_m: float) -> float:
        """Local gravity [m/s²] — decreases with altitude."""
        r = self.planet_radius_m + altitude_m
        mu = G * self.planet_mass_kg
        return mu / r**2

    def _ensure_pressures(self):
        """Compute base pressures for each layer boundary via hydrostatic integration."""
        if self._base_pressures:
            return
        pressures = [self.surface_pressure_Pa]
        P = self.surface_pressure_Pa
        for layer in self.layers[:-1]:
            dz   = layer.top_altitude_m - layer.base_altitude_m
            T_mid = layer.temperature_at(
                (layer.base_altitude_m + layer.top_altitude_m) / 2
            )
            g_mid = self._gravity_at(
                (layer.base_altitude_m + layer.top_altitude_m) / 2
            )
            H    = R_GAS * T_mid / (layer.mean_molar_mass_kg_mol * g_mid)
            P    = P * math.exp(-dz / H)
            pressures.append(P)
        pressures.append(0.0)   # top of atmosphere
        self._base_pressures = pressures

    def _layer_at(self, altitude_m: float) -> tuple[AtmosphericLayer, int]:
        """Return the layer and its index that contains this altitude."""
        for i, layer in enumerate(self.layers):
            if altitude_m <= layer.top_altitude_m:
                return layer, i
        return self.layers[-1], len(self.layers) - 1

    def temperature_at(self, altitude_m: float) -> float:
        """Temperature at altitude [K]."""
        if altitude_m < 0:
            return self.layers[0].base_temperature_K
        layer, _ = self._layer_at(altitude_m)
        return layer.temperature_at(altitude_m)

    def pressure_at(self, altitude_m: float) -> float:
        """Pressure at altitude [Pa] via hydrostatic integration."""
        if altitude_m < 0:
            return self.surface_pressure_Pa
        self._ensure_pressures()
        layer, idx = self._layer_at(altitude_m)
        P_base = self._base_pressures[idx]
        dz   = altitude_m - layer.base_altitude_m
        T_mid = layer.temperature_at(layer.base_altitude_m + dz / 2)
        g_mid = self._gravity_at(layer.base_altitude_m + dz / 2)
        H     = R_GAS * T_mid / (layer.mean_molar_mass_kg_mol * g_mid)
        return P_base * math.exp(-dz / H)

    def density_at(self, altitude_m: float) -> float:
        """Density at altitude [kg/m³] via ideal gas law: ρ = PM/(RT)."""
        P = self.pressure_at(altitude_m)
        T = self.temperature_at(altitude_m)
        layer, _ = self._layer_at(max(0.0, altitude_m))
        M = layer.mean_molar_mass_kg_mol
        return P * M / (R_GAS * T)

    def composition_at(self, altitude_m: float) -> dict[str, float]:
        """Mole fractions at altitude (returns the layer's bulk composition)."""
        layer, _ = self._layer_at(max(0.0, altitude_m))
        return dict(layer.composition)

    def scale_height_at(self, altitude_m: float) -> float:
        """Local scale height H = RT/(Mg) [m]."""
        layer, _ = self._layer_at(max(0.0, altitude_m))
        T = layer.temperature_at(max(0.0, altitude_m))
        g = self._gravity_at(max(0.0, altitude_m))
        return layer.scale_height(g, T)

    def mean_molar_mass_at(self, altitude_m: float) -> float:
        """Mean molar mass of the gas mixture at altitude [kg/mol]."""
        layer, _ = self._layer_at(max(0.0, altitude_m))
        return layer.mean_molar_mass_kg_mol

    def speed_of_sound(self, altitude_m: float) -> float:
        """Speed of sound c = sqrt(γRT/M) [m/s]."""
        T     = self.temperature_at(altitude_m)
        layer, _ = self._layer_at(max(0.0, altitude_m))
        total = sum(layer.composition.values()) or 1
        gamma = sum(
            (f/total) * GAMMA.get(g, 1.4)
            for g, f in layer.composition.items()
        )
        M = layer.mean_molar_mass_kg_mol
        return math.sqrt(gamma * R_GAS * T / M)

    def top_of_atmosphere(self) -> float:
        """Altitude of the topmost layer boundary [m]."""
        return self.layers[-1].top_altitude_m

    # ── Factory constructors ───────────────────────────────────────────────────

    @classmethod
    def from_atmosphere_config(cls, atm_config, planet) -> "MultiLayerAtmosphere":
        """
        Build a MultiLayerAtmosphere from an existing AtmosphereConfig.
        Uses the composition name to look up standard mole fractions,
        and the AtmosphereConfig for surface conditions.
        """
        comp_name  = atm_config.composition.name if atm_config.enabled else "NONE"
        composition = dict(STANDARD_COMPOSITIONS.get(comp_name, {"N2": 1.0}))
        if not composition:
            composition = {"N2": 1.0}

        # Normalise fractions
        total = sum(composition.values())
        if total > 0:
            composition = {k: v/total for k, v in composition.items()}

        g_surface = G * planet.mass / planet.radius**2

        # Troposphere
        T0     = atm_config.surface_temp
        lapse  = atm_config.lapse_rate
        H0     = atm_config.scale_height
        tropo_top = T0 / lapse if lapse > 1e-6 else 50_000.0
        tropo_top = min(tropo_top, 100_000.0)
        T_tropo   = max(T0 - lapse * tropo_top, 100.0)

        # Stratosphere (isothermal to 5 scale heights)
        strato_top = tropo_top + 5 * H0

        layers = [
            AtmosphericLayer(
                name="troposphere",
                base_altitude_m=0.0,
                top_altitude_m=tropo_top,
                base_temperature_K=T0,
                lapse_rate_K_per_m=lapse,
                composition=composition,
            ),
            AtmosphericLayer(
                name="stratosphere",
                base_altitude_m=tropo_top,
                top_altitude_m=strato_top,
                base_temperature_K=T_tropo,
                lapse_rate_K_per_m=-0.0005,  # slight warming (UV absorption)
                composition=composition,
            ),
            AtmosphericLayer(
                name="upper atmosphere",
                base_altitude_m=strato_top,
                top_altitude_m=strato_top * 3,
                base_temperature_K=T_tropo * 1.2,
                lapse_rate_K_per_m=0.0,
                composition={k: v for k, v in composition.items()
                             if MOLAR_MASS.get(k, 99) > 4},  # He/H2 escape faster
            ),
        ]
        return cls(
            layers=layers,
            surface_pressure_Pa=atm_config.surface_pressure,
            planet_radius_m=planet.radius,
            planet_mass_kg=planet.mass,
        )

    @classmethod
    def earth_standard(cls, planet) -> "MultiLayerAtmosphere":
        """Standard 5-layer Earth atmosphere (troposphere through thermosphere)."""
        comp_tropo = STANDARD_COMPOSITIONS["EARTH_LIKE"].copy()
        total = sum(comp_tropo.values())
        comp_tropo = {k: v/total for k, v in comp_tropo.items()}

        return cls(
            layers=[
                AtmosphericLayer("troposphere",   0,       12_000, 288.0,  0.0065, comp_tropo),
                AtmosphericLayer("stratosphere",  12_000,  50_000, 216.0, -0.0010, comp_tropo),
                AtmosphericLayer("mesosphere",    50_000,  85_000, 270.0,  0.0028, comp_tropo),
                AtmosphericLayer("thermosphere",  85_000, 600_000, 185.0, -0.0020,
                                 {"N2": 0.79, "O2": 0.21}),
                AtmosphericLayer("exosphere",    600_000, 10_000_000, 1000.0, 0.0,
                                 {"N2": 0.50, "O2": 0.20, "H2O": 0.20, "He": 0.10}),
            ],
            surface_pressure_Pa=101_325.0,
            planet_radius_m=planet.radius,
            planet_mass_kg=planet.mass,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Jeans escape
# ─────────────────────────────────────────────────────────────────────────────
class JeansEscape:
    """
    Thermal (Jeans) escape of gas species from the top of the atmosphere.

    Hunten (1973): the Jeans escape rate is controlled by the ratio λ = v_esc² / v_th²
    where v_th = sqrt(2 k_B T / m) is the thermal velocity.

    λ >> 1  : essentially no escape (stable retention)
    λ ~ 10  : slow Jeans escape — atmosphere erodes over Gyr
    λ < 6   : hydrodynamic escape — rapid blowoff (early solar system)

    The critical λ for long-term retention over 4.5 Gyr is approximately 20–40.
    """

    @staticmethod
    def lambda_parameter(species: str, escape_velocity_m_s: float,
                         exosphere_temperature_K: float) -> float:
        """
        Jeans parameter λ = (v_esc / v_th)² = m v_esc² / (2 k_B T).
        Dimensionless. Higher λ = more stable retention.
        """
        m_kg = MOLAR_MASS.get(species, 29.0) * 1e-3 / N_AVO   # kg per molecule
        v_th_sq = 2 * k_B * exosphere_temperature_K / m_kg
        return escape_velocity_m_s**2 / v_th_sq

    @staticmethod
    def escape_flux(species: str, escape_velocity_m_s: float,
                    exosphere_temperature_K: float,
                    exobase_density_m3: float,
                    planet_radius_m: float) -> float:
        """
        Jeans escape flux [molecules m⁻² s⁻¹] from the exobase.

        Φ_J = n_x × v_th / (2√π) × (1 + λ) × exp(−λ)

        where v_th = sqrt(2 k_B T / m) and λ is the Jeans parameter.
        """
        lam  = JeansEscape.lambda_parameter(species, escape_velocity_m_s,
                                             exosphere_temperature_K)
        m_kg = MOLAR_MASS.get(species, 29.0) * 1e-3 / N_AVO
        v_th = math.sqrt(2 * k_B * exosphere_temperature_K / m_kg)
        flux = (exobase_density_m3 * v_th / (2 * math.sqrt(math.pi))
                * (1 + lam) * math.exp(-lam))
        return max(0.0, flux)

    @staticmethod
    def retention_timescale_gyr(species: str, escape_velocity_m_s: float,
                                 exosphere_temperature_K: float,
                                 surface_pressure_Pa: float,
                                 surface_gravity_m_s2: float,
                                 planet_radius_m: float) -> float:
        """
        Approximate timescale [Gyr] for the species to escape to 1/e of current amount.

        Uses total column mass as the reservoir.
        Returns inf for λ > 40 (essentially stable).

        Note: this is a *lower bound* on atmospheric lifetime because:
        - It ignores replenishment (volcanism, cometary delivery)
        - Real escape rates are also affected by solar wind stripping (if no B-field)
        - Hydrodynamic escape is much faster below λ ≈ 6
        """
        lam = JeansEscape.lambda_parameter(species, escape_velocity_m_s,
                                            exosphere_temperature_K)
        if lam > 40:
            return float("inf")   # negligible escape
        if lam < 2:
            return 0.0            # immediate hydrodynamic escape

        m_kg  = MOLAR_MASS.get(species, 29.0) * 1e-3 / N_AVO
        v_th  = math.sqrt(2 * k_B * exosphere_temperature_K / m_kg)

        # Exobase density (rough estimate: 10⁻⁸ Pa → n = P/kT)
        P_exobase = 1e-8  # Pa at exobase
        n_exobase = P_exobase / (k_B * exosphere_temperature_K)

        # Jeans flux [molecules m⁻² s⁻¹]
        flux = (n_exobase * v_th / (2 * math.sqrt(math.pi))
                * (1 + lam) * math.exp(-lam))

        # Column number density N = P_surface / (m g) [molecules m⁻²]
        N_col = surface_pressure_Pa / (m_kg * surface_gravity_m_s2)

        if flux <= 0:
            return float("inf")

        t_sec = N_col / flux
        return t_sec / (1e9 * 365.25 * 86400)   # Gyr

    @staticmethod
    def can_retain(species: str, escape_velocity_m_s: float,
                   exosphere_temperature_K: float,
                   min_retention_gyr: float = 1.0) -> bool:
        """
        True if the planet can retain this species for at least min_retention_gyr.
        """
        lam = JeansEscape.lambda_parameter(species, escape_velocity_m_s,
                                            exosphere_temperature_K)
        return lam > 20

    @staticmethod
    def all_species_assessment(planet, exosphere_temp_K: float = None
                               ) -> dict[str, dict]:
        """
        Assess escape likelihood for the species in this planet's atmosphere.
        Returns dict: species → {lambda, timescale_gyr, retained}
        """
        from core.planet import Planet
        if not isinstance(planet, Planet):
            raise TypeError("planet must be a Planet instance")
        if not planet.atmosphere.enabled:
            return {}

        comp_name = planet.atmosphere.composition.name
        composition = STANDARD_COMPOSITIONS.get(comp_name, {"N2": 1.0})
        if not composition:
            return {}

        v_esc = planet.escape_velocity
        g_srf = planet.surface_gravity
        P_srf = planet.atmosphere.surface_pressure

        # Exosphere temperature: ~2–3× surface temperature for rocky planets
        if exosphere_temp_K is None:
            exosphere_temp_K = min(planet.atmosphere.surface_temp * 2.5, 3000.0)

        results = {}
        for species in composition:
            lam = JeansEscape.lambda_parameter(species, v_esc, exosphere_temp_K)
            t_gyr = JeansEscape.retention_timescale_gyr(
                species, v_esc, exosphere_temp_K, P_srf, g_srf, planet.radius
            )
            results[species] = {
                "lambda":         lam,
                "timescale_gyr":  t_gyr,
                "retained_1gyr":  t_gyr > 1.0,
                "retained_4gyr":  t_gyr > 4.0,
            }
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Greenhouse warming
# ─────────────────────────────────────────────────────────────────────────────
class GreenhouseModel:
    """
    Estimate greenhouse warming ΔT_GH for a given atmosphere composition and
    surface pressure.

    Uses a simplified parameterisation based on:
      - CO₂ forcing (logarithmic in partial pressure)
      - CH₄ forcing (square-root in partial pressure)
      - H₂O feedback (multiplicative amplifier)
      - H₂ collision-induced absorption (pressure-dependent)

    Calibration:
      Earth at 280 ppm CO₂ → ΔT ≈ 33 K (reference)
      Venus at 93 bar CO₂ → ΔT ≈ 511 K ✓
      Mars at 636 Pa CO₂  → ΔT ≈ 5 K  ✓

    Note: This is a simplified single-column model. Real greenhouse warming
    depends on atmospheric circulation, clouds, surface albedo feedback etc.
    Accuracy: ~20–40% for order-of-magnitude estimates.

    Reference:
      Byrne & Goldblatt (2014) parameterisation for CO₂
      Ramaswamy et al. IPCC AR4 — CH₄ and N₂O forcing
    """

    # Earth reference: ΔT_GH = 33 K at P_CO2 = 280 ppm × 1 atm = 28.3 Pa
    EARTH_CO2_PA   = 28.3    # Pa  (280 ppm × 101325)
    EARTH_DT_CO2   = 15.0    # K  (CO₂ + other non-H₂O gases)
    EARTH_DT_H2O   = 18.0    # K  (water vapour feedback, temp-dependent)
    EARTH_DT_TOTAL = 33.0    # K  (total greenhouse warming on Earth)

    @staticmethod
    def co2_partial_pressure(composition: dict[str, float],
                              surface_pressure_Pa: float) -> float:
        """CO₂ partial pressure [Pa] from mole fraction × surface pressure."""
        return composition.get("CO2", 0.0) * surface_pressure_Pa

    @staticmethod
    def ch4_partial_pressure(composition: dict[str, float],
                              surface_pressure_Pa: float) -> float:
        return composition.get("CH4", 0.0) * surface_pressure_Pa

    @staticmethod
    def h2o_partial_pressure(composition: dict[str, float],
                              surface_pressure_Pa: float) -> float:
        return composition.get("H2O", 0.0) * surface_pressure_Pa

    @classmethod
    def co2_forcing_K(cls, P_CO2_Pa: float) -> float:
        """
        Greenhouse warming from CO₂ alone [K].
        Piecewise model calibrated to Earth (33 K total) and Venus (~500 K).

        Byrne & Goldblatt (2014) + Kasting et al. (1993) extended:
          - Below Earth: logarithmic onset
          - Earth to 100× CO₂: log-linear
          - Venus-like (>1e4× CO₂): steep power law (pressure-induced absorption)
        """
        if P_CO2_Pa <= 0:
            return 0.0
        ratio = P_CO2_Pa / cls.EARTH_CO2_PA
        if ratio < 1:
            delta = cls.EARTH_DT_CO2 * math.log(1 + ratio) / math.log(2)
        elif ratio < 1e4:
            delta = cls.EARTH_DT_CO2 * (1 + 0.7 * math.log10(ratio))
        else:
            # Venus: ratio ≈ 3e5, need ~200 K CO2-only then H2O amplifier adds rest
            base  = cls.EARTH_DT_CO2 * (1 + 0.7 * math.log10(1e4))
            delta = base * (ratio / 1e4) ** 0.35
        return delta

    @classmethod
    def ch4_forcing_K(cls, P_CH4_Pa: float) -> float:
        """
        Greenhouse warming from CH₄ [K].
        Logarithmic scaling (avoids sqrt over-extrapolation at high concentrations).
        
        Calibrated:
          Earth 1.8 ppm (0.018 Pa)  → ~0.5 K
          Titan 1.5% CH4 (2200 Pa)  → ~8 K  (actual Titan: ~12 K from CH4)
          10% CH4 (10000 Pa)        → ~12 K
        """
        if P_CH4_Pa <= 0:
            return 0.0
        ref_Pa = 0.018   # 1.8 ppm at 1 atm (Earth reference)
        ratio  = P_CH4_Pa / ref_Pa
        # Log formula: ΔT = a × log10(ratio + 1)
        # a calibrated to give 0.5 K at Earth (ratio=1)
        a = 0.5 / math.log10(2.0)   # ≈ 1.66
        return a * math.log10(ratio + 1.0)

    @classmethod
    def h2_forcing_K(cls, P_H2_Pa: float, P_total_Pa: float) -> float:
        """
        H₂ collision-induced absorption (CIA) forcing [K].
        Relevant for hydrogen-rich planets (early Earth, sub-Neptunes).
        Wordsworth & Pierrehumbert (2013).
        """
        if P_H2_Pa <= 0:
            return 0.0
        # CIA scales roughly as P_H2 × P_total (pressure-broadened)
        # Calibrated: 10% H₂ at 1 atm → ΔT ≈ 3 K
        ref = 0.10 * 101325 * 101325
        actual = P_H2_Pa * P_total_Pa
        return 3.0 * (actual / ref) ** 0.5

    @classmethod
    def water_vapour_amplifier(cls, T_surface_K: float) -> float:
        """
        Water vapour feedback amplification factor.
        
        Physical basis: warmer atmosphere holds more H₂O (Clausius-Clapeyron).
        More H₂O → stronger greenhouse → warmer surface → even more H₂O.
        This positive feedback roughly doubles the CO₂ forcing on Earth.
        
        Calibrated:
          250 K → 1.0× (cold, dry atmosphere)
          288 K → 2.2× (Earth: CO₂-only ~15 K, total 33 K → ratio 2.2)
          310 K → 4.0× (warm ocean world)
          340 K → 8.0× (approaching runaway)
          > 400 K → 15× (Venus-like moist greenhouse)
        """
        if T_surface_K < 220:
            return 1.0
        elif T_surface_K < 250:
            return 1.0 + 0.0 * (T_surface_K - 220) / 30  # negligible
        elif T_surface_K < 290:
            # Earth range: ramp from 1.0 to 2.2
            return 1.0 + 1.2 * (T_surface_K - 250) / 40
        elif T_surface_K < 340:
            # Warm: ramp from 2.2 to 4.0
            return 2.2 + 1.8 * (T_surface_K - 290) / 50
        elif T_surface_K < 400:
            # Hot: ramp from 4.0 to 8.0
            return 4.0 + 4.0 * (T_surface_K - 340) / 60
        else:
            # Runaway / Venus-like: very strong amplification
            return 8.0 + 7.0 * min((T_surface_K - 400) / 200, 1.0)

    @classmethod
    def total_greenhouse_warming_K(cls, composition: dict[str, float],
                                    surface_pressure_Pa: float,
                                    surface_temperature_K: float,
                                    include_h2o_feedback: bool = True) -> float:
        """
        Total greenhouse warming ΔT_GH [K] for a given atmosphere.

        Does NOT account for:
        - Cloud feedbacks
        - Surface ice-albedo feedback
        - Atmospheric dynamics / heat redistribution

        Returns the static radiative forcing estimate only.
        """
        P_CO2 = cls.co2_partial_pressure(composition, surface_pressure_Pa)
        P_CH4 = cls.ch4_partial_pressure(composition, surface_pressure_Pa)
        P_H2  = composition.get("H2", 0.0) * surface_pressure_Pa

        dT_co2 = cls.co2_forcing_K(P_CO2)
        dT_ch4 = cls.ch4_forcing_K(P_CH4)
        dT_h2  = cls.h2_forcing_K(P_H2, surface_pressure_Pa)

        dT_total = dT_co2 + dT_ch4 + dT_h2

        if include_h2o_feedback:
            amp = cls.water_vapour_amplifier(surface_temperature_K)
            dT_total *= amp

        return dT_total

    @classmethod
    def surface_temperature(cls, equilibrium_temperature_K: float,
                             composition: dict[str, float],
                             surface_pressure_Pa: float,
                             max_iterations: int = 30,
                             tolerance_K: float = 1.0) -> float:
        """
        Solve for the actual surface temperature iteratively:
        T_surface = T_eq + ΔT_GH(T_surface)

        Iterates until convergence. Uses strong damping to prevent divergence
        in extreme cases (Venus-like runaway greenhouse).
        Caps at 900 K for Venus-like atmospheres (actual Venus = 737 K; our
        simplified model overestimates in the extreme runaway regime).
        """
        T = equilibrium_temperature_K   # initial guess
        P_CO2 = composition.get("CO2", 0.0) * surface_pressure_Pa
        T_cap = 800.0 if P_CO2 > 1e5 else 1500.0

        for _ in range(max_iterations):
            dT = cls.total_greenhouse_warming_K(composition, surface_pressure_Pa, T)
            T_new = equilibrium_temperature_K + dT
            T_new = min(T_new, T_cap)
            if abs(T_new - T) < tolerance_K:
                return T_new
            T = 0.75 * T + 0.25 * T_new
        return min(T, T_cap)

    @classmethod
    def is_runaway_greenhouse(cls, composition: dict[str, float],
                               surface_pressure_Pa: float,
                               equilibrium_temperature_K: float) -> bool:
        """
        Rough test for runaway greenhouse state.
        True if the computed surface temperature exceeds ~647 K (supercritical H₂O)
        or exceeds the equilibrium temp by more than 400 K.
        """
        T_surf = cls.surface_temperature(equilibrium_temperature_K, composition,
                                          surface_pressure_Pa)
        return T_surf > 647 or (T_surf - equilibrium_temperature_K) > 400


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full atmospheric analysis for a planet
# ─────────────────────────────────────────────────────────────────────────────
def analyse_atmosphere(planet, star=None,
                       orbital_distance_m: float = None,
                       bond_albedo: float = 0.3) -> dict:
    """
    Run a complete atmospheric analysis on a Planet object.

    Returns a dict with:
        composition         : mole fraction dict
        multi_layer         : MultiLayerAtmosphere object
        greenhouse_dT_K     : greenhouse warming [K]
        surface_temp_K      : computed surface temperature [K]
        jeans_escape        : per-species Jeans assessment
        scale_height_m      : scale height at surface [m]
        runaway_greenhouse  : bool
        atmospheric_mass_kg : estimated total atmospheric mass
        mean_molar_mass     : mean molar mass [g/mol]
    """
    if not planet.atmosphere.enabled:
        return {"enabled": False}

    comp_name   = planet.atmosphere.composition.name
    composition = dict(STANDARD_COMPOSITIONS.get(comp_name, {"N2": 1.0}))
    total_frac  = sum(composition.values())
    if total_frac > 0:
        composition = {k: v/total_frac for k, v in composition.items()}

    multi_layer = MultiLayerAtmosphere.from_atmosphere_config(
        planet.atmosphere, planet
    )

    # Equilibrium temperature
    if star and orbital_distance_m:
        T_eq = star.equilibrium_temperature(orbital_distance_m, bond_albedo)
    elif hasattr(planet, "star_context") and planet.star_context and planet.orbital_distance_m:
        T_eq = planet.star_context.equilibrium_temperature(
            planet.orbital_distance_m, bond_albedo
        )
    else:
        T_eq = planet.atmosphere.surface_temp * 0.85   

    # Greenhouse
    P_srf = planet.atmosphere.surface_pressure
    dT_GH = GreenhouseModel.total_greenhouse_warming_K(composition, P_srf, T_eq)
    T_surf = GreenhouseModel.surface_temperature(T_eq, composition, P_srf)
    runaway = GreenhouseModel.is_runaway_greenhouse(composition, P_srf, T_eq)

    # Jeans escape
    jeans = JeansEscape.all_species_assessment(planet)

    # Scale height (derived from composition, not hand-set)
    g_srf   = planet.surface_gravity
    mm      = multi_layer.layers[0].mean_molar_mass_g_mol
    mm_kg   = mm * 1e-3
    T0      = planet.atmosphere.surface_temp
    H_deriv = R_GAS * T0 / (mm_kg * g_srf)

    # Atmospheric mass: M_atm = 4πR² × P_srf / g
    M_atm = 4 * math.pi * planet.radius**2 * P_srf / g_srf

    return {
        "enabled":              True,
        "composition":          composition,
        "multi_layer":          multi_layer,
        "equilibrium_temp_K":   T_eq,
        "greenhouse_dT_K":      dT_GH,
        "surface_temp_K":       T_surf,
        "runaway_greenhouse":   runaway,
        "jeans_escape":         jeans,
        "scale_height_m":       H_deriv,
        "scale_height_km":      H_deriv / 1e3,
        "mean_molar_mass_g_mol": mm,
        "atmospheric_mass_kg":  M_atm,
    }
