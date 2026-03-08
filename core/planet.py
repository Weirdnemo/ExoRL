"""
planet.py — Core planet data model and derived physics properties.
All SI units unless stated otherwise.
"""

from __future__ import annotations
import math
import dataclasses
from typing import Optional
from enum import Enum, auto


# ── Physical constants ────────────────────────────────────────────────────────
G  = 6.674_30e-11   # gravitational constant  [m³ kg⁻¹ s⁻²]
R_EARTH  = 6.371e6  # Earth radius            [m]
M_EARTH  = 5.972e24 # Earth mass              [kg]
ATM_EARTH = 101_325  # Earth sea-level pressure [Pa]


# ── Enums ─────────────────────────────────────────────────────────────────────
class AtmosphereComposition(Enum):
    NONE        = auto()   # vacuum
    CO2_THICK   = auto()   # Venus / early Mars style
    CO2_THIN    = auto()   # Mars style
    NITROGEN    = auto()   # Earth-like but inert
    EARTH_LIKE  = auto()   # N₂/O₂ mix
    HYDROGEN    = auto()   # Gas giant envelope
    METHANE     = auto()   # Titan-like
    CUSTOM      = auto()   # user-defined molar mass / heat cap

class TerrainType(Enum):
    FLAT        = auto()
    CRATERED    = auto()
    MOUNTAINOUS = auto()
    OCEANIC     = auto()
    VOLCANIC    = auto()
    RANDOM      = auto()

class MagneticFieldStrength(Enum):
    NONE   = auto()
    WEAK   = auto()   # 10% of Earth
    MEDIUM = auto()   # Earth-like
    STRONG = auto()   # Jupiter-like

# ── Atmosphere profile ────────────────────────────────────────────────────────
@dataclasses.dataclass
class AtmosphereConfig:
    enabled: bool = True
    composition: AtmosphereComposition = AtmosphereComposition.EARTH_LIKE
    scale_height: float = 8_500.0        # [m] density e-folding height
    surface_pressure: float = 101_325.0  # [Pa]
    surface_density: float = 1.225       # [kg/m³]
    surface_temp: float = 288.0          # [K]
    lapse_rate: float = 0.0065           # [K/m]  tropospheric lapse
    # aerodynamics
    drag_coeff_multiplier: float = 1.0   # global scale on Cd
    wind_enabled: bool = False
    wind_speed_mps: float = 0.0
    wind_direction_deg: float = 0.0

    def density_at_altitude(self, altitude: float) -> float:
        """Exponential atmosphere model ρ(h) = ρ₀ · exp(-h / H)"""
        if not self.enabled or altitude < 0:
            return self.surface_density if altitude <= 0 else 0.0
        return self.surface_density * math.exp(-altitude / self.scale_height)

    def pressure_at_altitude(self, altitude: float) -> float:
        if not self.enabled or altitude < 0:
            return self.surface_pressure if altitude <= 0 else 0.0
        return self.surface_pressure * math.exp(-altitude / self.scale_height)

    def temperature_at_altitude(self, altitude: float) -> float:
        """Simple linear lapse in troposphere, isothermal above."""
        if not self.enabled:
            return 2.7   # CMB
        tropo_top = self.surface_temp / self.lapse_rate if self.lapse_rate > 0 else 1e9
        if altitude <= tropo_top:
            return max(self.surface_temp - self.lapse_rate * altitude, 50.0)
        return self.surface_temp - self.lapse_rate * tropo_top

# ── Terrain config ────────────────────────────────────────────────────────────
@dataclasses.dataclass
class TerrainConfig:
    enabled: bool = True
    terrain_type: TerrainType = TerrainType.CRATERED
    max_elevation: float = 8_848.0   # [m]  highest peak
    min_elevation: float = -11_000.0 # [m]  deepest trench
    roughness: float = 0.5           # 0=glass-smooth, 1=extreme
    seed: int = 42

# ── Magnetic field config ─────────────────────────────────────────────────────
@dataclasses.dataclass
class MagneticFieldConfig:
    enabled: bool = False
    strength: MagneticFieldStrength = MagneticFieldStrength.MEDIUM
    tilt_deg: float = 11.5           # dipole tilt from rotation axis
    # radiation belt intensity (affects spacecraft charging)
    radiation_belt_enabled: bool = False
    inner_belt_altitude: float = 1_000e3   # [m]
    outer_belt_altitude: float = 25_000e3  # [m]

# ── Oblateness / J2 config ────────────────────────────────────────────────────
@dataclasses.dataclass
class OblatenessConfig:
    enabled: bool = False
    J2: float = 1.082_63e-3   # Earth's J2 (dimensionless)
    J3: float = -2.53e-6      # Earth's J3
    flattening: float = 1/298.257  # a-c / a

# ── Moon config ───────────────────────────────────────────────────────────────
@dataclasses.dataclass
class MoonConfig:
    enabled: bool = False
    count: int = 1
    mass_fraction: float = 0.012  # moon mass / planet mass  (Earth/Moon ≈ 0.0123)
    orbit_radius: float = 384_400e3  # [m]  semi-major axis

# ── Main Planet dataclass ─────────────────────────────────────────────────────
@dataclasses.dataclass
class Planet:
    # ── Identity ──
    name: str = "Planet-X"

    # ── Size ──
    radius: float = R_EARTH        # [m]
    mass: float   = M_EARTH        # [kg]

    # ── Rotation ──
    rotation_enabled: bool = True
    rotation_period: float = 86_400.0  # [s]  sidereal day

    # ── Sub-systems (all individually toggleable) ──
    atmosphere: AtmosphereConfig    = dataclasses.field(default_factory=AtmosphereConfig)
    terrain: TerrainConfig          = dataclasses.field(default_factory=TerrainConfig)
    magnetic_field: MagneticFieldConfig = dataclasses.field(default_factory=MagneticFieldConfig)
    oblateness: OblatenessConfig    = dataclasses.field(default_factory=OblatenessConfig)
    moons: MoonConfig               = dataclasses.field(default_factory=MoonConfig)

    # ── Derived (computed on access) ──────────────────────────────────────────
    @property
    def mu(self) -> float:
        """Gravitational parameter μ = GM  [m³/s²]"""
        return G * self.mass

    @property
    def surface_gravity(self) -> float:
        """g at surface  [m/s²]"""
        return self.mu / self.radius**2

    @property
    def escape_velocity(self) -> float:
        """v_esc = sqrt(2μ/r)  [m/s]"""
        return math.sqrt(2 * self.mu / self.radius)

    @property
    def first_cosmic_velocity(self) -> float:
        """Minimum circular orbit speed at surface  [m/s]"""
        return math.sqrt(self.mu / self.radius)

    @property
    def hill_sphere_radius(self, parent_mu: float = 1.327e20, orbit_radius: float = 1.496e11) -> float:
        """Approximate Hill sphere radius assuming Sun-like parent  [m]"""
        return orbit_radius * (self.mass / (3 * parent_mu / G)) ** (1/3)

    @property
    def surface_area(self) -> float:
        return 4 * math.pi * self.radius**2

    @property
    def volume(self) -> float:
        return (4/3) * math.pi * self.radius**3

    @property
    def mean_density(self) -> float:
        return self.mass / self.volume

    # ── Physics helpers ───────────────────────────────────────────────────────
    def gravity_at_altitude(self, altitude: float) -> float:
        """Point-mass gravity  [m/s²]"""
        r = self.radius + altitude
        return self.mu / r**2

    def gravity_vector_J2(self, r_vec: tuple[float, float, float]) -> tuple[float, float, float]:
        """
        Gravitational acceleration including J2 oblateness correction.
        r_vec: position in planet-centred inertial frame (ECI-like) [m]
        Returns acceleration vector [m/s²]
        """
        x, y, z = r_vec
        r = math.sqrt(x**2 + y**2 + z**2)
        mu = self.mu

        # Point-mass term
        ax = -mu * x / r**3
        ay = -mu * y / r**3
        az = -mu * z / r**3

        if self.oblateness.enabled:
            J2 = self.oblateness.J2
            Re = self.radius
            factor = (3/2) * J2 * mu * Re**2 / r**5
            ax += factor * x * (5 * z**2 / r**2 - 1)
            ay += factor * y * (5 * z**2 / r**2 - 1)
            az += factor * z * (5 * z**2 / r**2 - 3)

        return (ax, ay, az)

    def circular_orbit_speed(self, altitude: float) -> float:
        """Circular orbit velocity at given altitude  [m/s]"""
        return math.sqrt(self.mu / (self.radius + altitude))

    def circular_orbit_period(self, altitude: float) -> float:
        """Circular orbit period at given altitude  [s]"""
        a = self.radius + altitude
        return 2 * math.pi * math.sqrt(a**3 / self.mu)

    def hohmann_delta_v(self, alt1: float, alt2: float) -> tuple[float, float]:
        """
        Delta-V for a Hohmann transfer between two circular orbits.
        Returns (dv1, dv2) in [m/s].
        """
        r1 = self.radius + alt1
        r2 = self.radius + alt2
        v1 = math.sqrt(self.mu / r1)
        v2 = math.sqrt(self.mu / r2)
        v_trans_peri = math.sqrt(self.mu * (2/r1 - 1/((r1+r2)/2)))
        v_trans_apo  = math.sqrt(self.mu * (2/r2 - 1/((r1+r2)/2)))
        return (abs(v_trans_peri - v1), abs(v2 - v_trans_apo))

    def aerobraking_deceleration(self, altitude: float, speed: float,
                                  Cd: float = 2.2, area: float = 10.0,
                                  sc_mass: float = 1000.0) -> float:
        """
        Aerodynamic drag deceleration  [m/s²]
        F_drag = 0.5 * ρ * v² * Cd * A   →   a = F/m
        """
        if not self.atmosphere.enabled:
            return 0.0
        rho = self.atmosphere.density_at_altitude(altitude)
        Cd_eff = Cd * self.atmosphere.drag_coeff_multiplier
        return 0.5 * rho * speed**2 * Cd_eff * area / sc_mass

    def summary(self) -> str:
        lines = [
            f"═══ {self.name} ═══",
            f"  Radius          : {self.radius/1e3:.1f} km  ({self.radius/R_EARTH:.3f} R⊕)",
            f"  Mass            : {self.mass:.3e} kg  ({self.mass/M_EARTH:.3f} M⊕)",
            f"  Surface gravity : {self.surface_gravity:.3f} m/s²",
            f"  Escape velocity : {self.escape_velocity/1e3:.3f} km/s",
            f"  1st cosmic vel  : {self.first_cosmic_velocity/1e3:.3f} km/s",
            f"  Mean density    : {self.mean_density:.1f} kg/m³",
            f"  Rotation period : {'disabled' if not self.rotation_enabled else f'{self.rotation_period/3600:.2f} h'}",
            f"  Atmosphere      : {'ON' if self.atmosphere.enabled else 'OFF'}"
                + (f" | ρ₀={self.atmosphere.surface_density:.3f} kg/m³"
                   f" | H={self.atmosphere.scale_height/1e3:.1f} km"
                   if self.atmosphere.enabled else ""),
            f"  Terrain         : {'ON' if self.terrain.enabled else 'OFF'}"
                + (f" | {self.terrain.terrain_type.name}" if self.terrain.enabled else ""),
            f"  Magnetic field  : {'ON' if self.magnetic_field.enabled else 'OFF'}",
            f"  J2 oblateness   : {'ON' if self.oblateness.enabled else 'OFF'}"
                + (f" | J2={self.oblateness.J2:.4e}" if self.oblateness.enabled else ""),
            f"  Moons           : {'ON' if self.moons.enabled else 'OFF'}"
                + (f" | n={self.moons.count}" if self.moons.enabled else ""),
        ]
        return "\n".join(lines)
