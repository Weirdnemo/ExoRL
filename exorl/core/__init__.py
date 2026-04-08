from exorl.core.env import OrbitalInsertionEnv
from exorl.core.generator import PRESETS, PlanetGenerator
from exorl.core.interior import (
    MATERIAL_DENSITY,
    ConvectionState,
    InteriorConfig,
    InteriorLayer,
    interior_from_bulk_density,
)
from exorl.core.physics import (
    AeroConfig,
    OrbitalIntegrator,
    SpacecraftState,
    ThrusterConfig,
    state_to_orbital_elements,
)
from exorl.core.planet import (
    M_EARTH,
    R_EARTH,
    AtmosphereComposition,
    AtmosphereConfig,
    G,
    MagneticFieldConfig,
    MagneticFieldStrength,
    MoonConfig,
    OblatenessConfig,
    Planet,
    TerrainConfig,
    TerrainType,
)
from exorl.core.star import (
    AU,
    STAR_PRESETS,
    SpectralType,
    Star,
    star_alpha_centauri_a,
    star_eps_eridani,
    star_kepler452,
    star_proxima_centauri,
    star_sun,
    star_tau_ceti,
    star_trappist1,
)

__all__ = [
    "Planet",
    "AtmosphereConfig",
    "TerrainConfig",
    "MagneticFieldConfig",
    "OblatenessConfig",
    "MoonConfig",
    "AtmosphereComposition",
    "TerrainType",
    "MagneticFieldStrength",
    "G",
    "R_EARTH",
    "M_EARTH",
    "PlanetGenerator",
    "PRESETS",
    "SpacecraftState",
    "OrbitalIntegrator",
    "ThrusterConfig",
    "AeroConfig",
    "state_to_orbital_elements",
    "OrbitalInsertionEnv",
    "InteriorConfig",
    "InteriorLayer",
    "ConvectionState",
    "interior_from_bulk_density",
    "MATERIAL_DENSITY",
    "Star",
    "SpectralType",
    "STAR_PRESETS",
    "star_sun",
    "star_proxima_centauri",
    "star_trappist1",
    "star_tau_ceti",
    "star_kepler452",
    "star_alpha_centauri_a",
    "star_eps_eridani",
    "AU",
]

# ── Step 2: Atmosphere science + Habitability ─────────────────────────────────
import exorl.core.planet_io  # triggers Planet.to_json / .from_json / .fingerprint patch
from exorl.core.atmosphere_science import (
    MOLAR_MASS,
    STANDARD_COMPOSITIONS,
    AtmosphericLayer,
    GreenhouseModel,
    JeansEscape,
    MultiLayerAtmosphere,
    analyse_atmosphere,
)
from exorl.core.climate import (
    BifurcationPoints,
    ClimateResult,
    ClimateState,
    EnergyBalanceModel,
    climate_habitability_score,
    climate_map,
    find_bifurcation_points,
)

# ── Geology and tectonic regime ───────────────────────────────────────────────
from exorl.core.geology import GeologyModel, TectonicRegime
from exorl.core.ground_track import (
    CoverageMap,
    GroundTrackPoint,
    compute_coverage_map,
    coverage_analysis,
    find_passes,
    mean_revisit_time_days,
    propagate_ground_track,
    time_to_full_coverage_days,
)
from exorl.core.habitability import (
    HabitabilityAssessment,
    assess_habitability,
    composition_class,
    size_class,
)

# ── Phase 1: Interplanetary physics ──────────────────────────────────────────
from exorl.core.heliocentric import (
    MU_SUN,
    HeliocentricIntegrator,
    HelioState,
    KeplerPropagator,
    LambertSolver,
    planet_state,
    soi_radius,
    transfer_summary,
)

# ── Interplanetary RL environment ─────────────────────────────────────────────
from exorl.core.interplanetary_env import InterplanetaryEnv, MissionConfig

# ── Reference exoplanet catalog ───────────────────────────────────────────────
from exorl.core.kepler_catalog import KeplerCatalog
from exorl.core.launch_window import (
    LaunchDecisionSpace,
    LaunchWindow,
    PorkchopData,
    compute_transfer,
    orbital_period_days,
    synodic_period_days,
)
from exorl.core.mission import (
    AerobrakingCampaign,
    AerobrakingPass,
    DeltaVBudget,
    GravityAssist,
    build_mission_dv_budget,
    lambert_solve,
    orbital_insertion_dv,
    plan_aerobraking,
    porkchop_data,
)
from exorl.core.observation import (
    TransitSignal,
    atmospheric_scale_height,
    characterise_observations,
    geometric_transit_probability,
    rv_semi_amplitude,
    transit_depth,
    transit_depth_ppm,
    transit_duration,
    transmission_spectroscopy_metric,
    transmission_spectrum,
)

# ── Steps 3-5: Orbital, Ground Track, Surface Energy, Tidal, Mission ─────────
from exorl.core.orbital_analysis import (
    DragLifetime,
    FrozenOrbit,
    J2Analysis,
    OrbitDesign,
    RepeatGroundTrack,
    StationKeeping,
    SunSynchronousOrbit,
    circular_speed,
    orbital_period,
    semi_major_axis_from_altitude,
)

# ── Planet I/O and serialisation ─────────────────────────────────────────────
from exorl.core.planet_io import (
    load_planet,
    planet_fingerprint,
    planet_from_json,
    planet_to_json,
    save_planet,
)
from exorl.core.soi import (
    HyperbolicArrival,
    HyperbolicDeparture,
    HyperbolicOrbit,
    SOIEvent,
    SOITransitionDetector,
    SphereOfInfluence,
    gravity_assist_turn,
    hill_sphere_radius,
    laplace_soi_radius,
    patched_conic_budget,
)
from exorl.core.surface_energy import (
    InsolationMap,
    TemperatureMap,
    compute_insolation_map,
    compute_temperature_map,
    permanent_shadow_latitude_deg,
    surface_energy_balance,
)
from exorl.core.tidal import (
    OrbitalMigration,
    RocheLimit,
    TidalAnalysis,
    TidalHeating,
    TidalLocking,
    analyse_tidal,
)
