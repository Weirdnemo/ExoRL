from core.planet import (
    Planet, AtmosphereConfig, TerrainConfig, MagneticFieldConfig,
    OblatenessConfig, MoonConfig,
    AtmosphereComposition, TerrainType, MagneticFieldStrength,
    G, R_EARTH, M_EARTH,
)
from core.generator import PlanetGenerator, PRESETS
from core.physics import (
    SpacecraftState, OrbitalIntegrator, ThrusterConfig, AeroConfig,
    state_to_orbital_elements,
)
from core.env import OrbitalInsertionEnv

from core.interior import (
    InteriorConfig, InteriorLayer, ConvectionState,
    interior_from_bulk_density, MATERIAL_DENSITY,
)
from core.star import (
    Star, SpectralType, STAR_PRESETS,
    star_sun, star_proxima_centauri, star_trappist1,
    star_tau_ceti, star_kepler452,
    star_alpha_centauri_a, star_eps_eridani,
    AU,
)

__all__ = [
    "Planet", "AtmosphereConfig", "TerrainConfig", "MagneticFieldConfig",
    "OblatenessConfig", "MoonConfig",
    "AtmosphereComposition", "TerrainType", "MagneticFieldStrength",
    "G", "R_EARTH", "M_EARTH",
    "PlanetGenerator", "PRESETS",
    "SpacecraftState", "OrbitalIntegrator", "ThrusterConfig", "AeroConfig",
    "state_to_orbital_elements",
    "OrbitalInsertionEnv",
    "InteriorConfig", "InteriorLayer", "ConvectionState",
    "interior_from_bulk_density", "MATERIAL_DENSITY",
    "Star", "SpectralType", "STAR_PRESETS",
    "star_sun", "star_proxima_centauri", "star_trappist1",
    "star_tau_ceti", "star_kepler452",
    "star_alpha_centauri_a", "star_eps_eridani",
    "AU",
]

# ── Step 2: Atmosphere science + Habitability ─────────────────────────────────
from core.atmosphere_science import (
    MultiLayerAtmosphere, AtmosphericLayer, JeansEscape, GreenhouseModel,
    analyse_atmosphere, STANDARD_COMPOSITIONS, MOLAR_MASS,
)
from core.habitability import (
    HabitabilityAssessment, assess_habitability,
    size_class, composition_class,
)

# ── Steps 3-5: Orbital, Ground Track, Surface Energy, Tidal, Mission ─────────
from core.orbital_analysis import (
    J2Analysis, SunSynchronousOrbit, FrozenOrbit,
    DragLifetime, StationKeeping, RepeatGroundTrack, OrbitDesign,
    semi_major_axis_from_altitude, orbital_period, circular_speed,
)
from core.ground_track import (
    propagate_ground_track, compute_coverage_map,
    coverage_analysis, time_to_full_coverage_days,
    mean_revisit_time_days, find_passes, GroundTrackPoint, CoverageMap,
)
from core.surface_energy import (
    compute_insolation_map, compute_temperature_map,
    surface_energy_balance, permanent_shadow_latitude_deg,
    InsolationMap, TemperatureMap,
)
from core.tidal import (
    TidalHeating, TidalLocking, RocheLimit, OrbitalMigration,
    analyse_tidal, TidalAnalysis,
)
from core.mission import (
    DeltaVBudget, AerobrakingCampaign, AerobrakingPass,
    orbital_insertion_dv, plan_aerobraking,
    lambert_solve, porkchop_data, GravityAssist,
    build_mission_dv_budget,
)

# ── Phase 1: Interplanetary physics ──────────────────────────────────────────
from core.heliocentric import (
    LambertSolver, KeplerPropagator, HeliocentricIntegrator,
    HelioState, planet_state, soi_radius, transfer_summary,
    MU_SUN,
)
from core.soi import (
    SphereOfInfluence, HyperbolicDeparture, HyperbolicArrival,
    HyperbolicOrbit, SOITransitionDetector, SOIEvent,
    laplace_soi_radius, hill_sphere_radius,
    gravity_assist_turn, patched_conic_budget,
)
from core.launch_window import (
    PorkchopData, LaunchWindow, LaunchDecisionSpace,
    synodic_period_days, orbital_period_days, compute_transfer,
)
from core.climate import (
    EnergyBalanceModel, ClimateResult, BifurcationPoints,
    find_bifurcation_points, climate_map, climate_habitability_score,
    ClimateState,
)
from core.observation import (
    characterise_observations, TransitSignal,
    transit_depth, transit_depth_ppm, transit_duration,
    geometric_transit_probability, rv_semi_amplitude,
    transmission_spectrum, transmission_spectroscopy_metric,
    atmospheric_scale_height,
)
