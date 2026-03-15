# Planet-RL → Planetary Science Toolkit
## A Research Scientist's Wishlist

*Written from the perspective of a planetary scientist who needs this as a primary research tool, not a training environment. The current codebase is an excellent foundation. What follows is an honest gap analysis and a prioritised roadmap for turning it into something publishable-quality.*

---

## What We Have vs. What Science Needs

The current tool does three things well: it generates physically plausible planets, integrates spacecraft trajectories around them, and visualises the results cleanly. That's genuinely useful. But a working scientist needs to ask questions like:

- *Is this planet inside the habitable zone of its star?*
- *What is the minimum ΔV budget for a spacecraft to reach orbit, circularise, and survive aerocapture?*
- *How does the J2 precession rate at 300 km compare between Mars and a hypothetical Super-Earth with the same surface gravity?*
- *What is the ground track of a polar orbit after 10 days? Where are the observation gaps?*
- *Given a 5,000 kg spacecraft and a 10-year transfer from Earth, what is the maximum payload that arrives with enough fuel to insert?*

None of these are answerable yet. Below is a detailed breakdown of every missing capability, grouped by scientific domain, with concrete implementation notes.

---

## Priority 1 — The Planetary Interior Model

**What's missing and why it matters most.**

Right now `mass` and `radius` are independent inputs. In reality, a planet's internal structure is the *source* of almost every observable property: bulk density tells you the iron fraction, which tells you the magnetic dynamo strength, which tells you the atmosphere retention history, which tells you the surface conditions today. Everything is coupled.

### 1a. Interior Structure Model (`core/interior.py`)

A layered interior with differentiation:

```python
@dataclass
class InteriorLayer:
    name: str                    # "iron core", "silicate mantle", "ice shell"
    outer_radius_frac: float     # fraction of planet radius
    density: float               # kg/m³
    material: str                # "Fe", "FeS", "silicate", "water_ice", "H2"
    heat_production: float       # W/kg  (radiogenic heating)

@dataclass
class InteriorConfig:
    enabled: bool = False
    layers: list[InteriorLayer] = field(default_factory=list)

    @property
    def core_radius_fraction(self) -> float: ...
    @property
    def iron_mass_fraction(self) -> float: ...
    @property
    def moment_of_inertia_factor(self) -> float:
        """I / (MR²) — 0.4 = uniform, 0.33 = Earth, 0.22 = differentiated"""
    @property
    def radiogenic_heat_flux(self) -> float:
        """W/m² — drives volcanism, tectonics, magnetic dynamo"""
```

**Why this matters:**
- Moment of inertia factor (MoI) is directly observable via precession rate and is used to infer internal structure from orbital data. Right now J2 is a free parameter — it should *derive* from the MoI.
- A large iron core → strong magnetic dynamo → retained atmosphere → surface liquid water → potential habitability. This causal chain is scientifically essential.
- Radiogenic heat flux determines whether a planet is still geologically active (plate tectonics, volcanism, outgassing), which feeds the atmosphere composition model.

### 1b. Derived Properties from Interior

Once the interior model exists, these become computable instead of inputs:

```python
# Currently: planet.magnetic_field.strength = MagneticFieldStrength.MEDIUM (hand-set)
# Should be: derived from core size + rotation rate + radiogenic heating
planet.interior.dynamo_number        # dimensionless — predicts magnetic field generation
planet.interior.predicted_B_surface  # Tesla — surface field strength
planet.magnetic_field.strength       # auto-set from interior

# Currently: planet.oblateness.J2 = 1.08e-3 (hand-set)
# Should be: J2 = f(rotation_period, MoI_factor, equatorial_radius)
planet.interior.predicted_J2         # derived from rotation + MoI
```

---

## Priority 2 — Atmospheric Science

The current atmosphere is a single exponential scale height. This is adequate for drag modelling but useless for actual atmospheric science.

### 2a. Multi-Layer Atmosphere (`core/atmosphere_science.py`)

Real atmospheres have structure. Earth has troposphere, stratosphere, mesosphere, thermosphere. Each has different physics.

```python
@dataclass
class AtmosphericLayer:
    name: str                    # "troposphere", "stratosphere", etc.
    base_altitude: float         # m
    top_altitude: float          # m
    base_temperature: float      # K
    lapse_rate: float            # K/m (positive = cooling with altitude)
    composition: dict[str, float]  # {"N2": 0.78, "O2": 0.21, ...} mole fractions
    mean_molar_mass: float       # g/mol — determines scale height
    
@dataclass
class MultiLayerAtmosphere:
    layers: list[AtmosphericLayer]
    
    def density_at_altitude(self, h: float) -> float:
        """Full hydrostatic integration through all layers."""
    
    def temperature_at_altitude(self, h: float) -> float:
        """Correct piecewise temperature profile."""
    
    def mean_molar_mass_at_altitude(self, h: float) -> float:
        """Changes with photodissociation in upper atmosphere."""
    
    def speed_of_sound(self, h: float) -> float:
        """c = √(γRT/M) — needed for sonic aerobraking transitions."""
    
    def dynamic_viscosity(self, h: float) -> float:
        """Sutherland's law — needed for boundary layer drag at low Re."""
```

### 2b. Atmospheric Composition Tracking

The current `AtmosphereComposition` enum is a label, not physics. The composition should be a dictionary that feeds into everything:

```python
# Current: composition = AtmosphereComposition.CO2_THICK  (just a label)
# Proposed: actual mole fractions that drive derived quantities

composition = {
    "CO2": 0.965,
    "N2":  0.035,
    "SO2": 0.00015,
}

# These then determine:
mean_molar_mass = sum(frac * MOLAR_MASS[gas] for gas, frac in composition.items())
# → scale height H = RT/(Mg)  — now physically derived, not hand-set
# → greenhouse forcing GHF = f(CO2_ppm, CH4_ppm, H2O_ppm)  — for surface temp
# → atmospheric escape rate = f(EUV_flux, M_gas, T_exosphere)
```

### 2c. Atmospheric Loss and Evolution

The most interesting question in planetary science: *why did Mars lose its atmosphere?*

```python
class AtmosphericEscape:
    """Jeans escape, ion pickup, sputtering, hydrodynamic escape."""
    
    def jeans_escape_rate(self, planet, gas: str, T_exosphere: float) -> float:
        """
        Thermal escape rate for a given gas species.
        Rate ∝ exp(-v_esc² / v_thermal²)
        Hydrogen escapes from Mars; CO2 does not from Venus.
        """
    
    def ion_pickup_rate(self, planet, solar_wind_pressure: float) -> float:
        """
        Solar wind stripping rate — dominant on Mars (no magnetosphere).
        vs. negligible on Earth (strong magnetosphere deflects solar wind).
        """
    
    def characteristic_escape_time(self, planet, gas: str) -> float:
        """How many years until this gas species is half-depleted?"""
```

This directly connects the magnetic field model to the atmosphere model in a physically motivated way.

### 2d. Greenhouse Effect and Surface Temperature

Currently `surface_temp` is an input. It should be computable:

```python
def compute_surface_temperature(
    planet,
    stellar_luminosity: float,      # W  (e.g. 3.828e26 for Sun)
    orbital_distance: float,        # m
    albedo: float,                  # bond albedo (0=black body, 0.3=Earth)
) -> float:
    """
    T_eq = ((L(1-A)) / (16πσd²))^0.25  →  equilibrium temperature
    + greenhouse forcing from CO2/CH4/H2O composition
    + internal heat contribution from radiogenic heating
    Returns: actual surface temperature [K]
    """
```

This closes the loop: composition → greenhouse → surface temp → habitability assessment.

---

## Priority 3 — Orbital Mechanics (the big missing piece)

The integrator handles simple trajectories well. But a planetary scientist doing mission design needs far more.

### 3a. Full Keplerian Analysis

The current `state_to_orbital_elements()` is correct but we're missing the analysis layer built on top of it.

```python
class OrbitalAnalysis:
    
    def nodal_precession_rate(self, planet, semi_major_axis, inclination) -> float:
        """
        dΩ/dt due to J2 oblateness.
        For a Sun-synchronous orbit: dΩ/dt = 0.9856°/day (Earth precesses to match Sun).
        Used to design repeat ground tracks.
        """
    
    def apsidal_precession_rate(self, planet, semi_major_axis, eccentricity, inclination) -> float:
        """dω/dt due to J2."""
    
    def frozen_orbit_eccentricity(self, planet, semi_major_axis, inclination) -> float:
        """
        The eccentricity at which apsidal drift vanishes.
        LRO, MRO, and most Mars orbiters use frozen orbits.
        Critical for long-duration science missions — prevents orbit from
        slowly becoming more eccentric over months.
        """
    
    def sun_synchronous_inclination(self, planet, semi_major_axis,
                                    stellar_angular_velocity: float) -> float:
        """
        The inclination at which nodal precession exactly matches
        the planet's orbital motion around its star.
        Guarantees constant local solar time at each ground point.
        Essential for imaging missions.
        """
    
    def repeat_ground_track(self, planet, semi_major_axis,
                            inclination, n_orbits: int, n_days: int) -> bool:
        """Does this orbit produce a repeat ground track in n_orbits / n_days?"""
    
    def lifetime_estimate(self, planet, semi_major_axis) -> float:
        """
        Atmospheric drag lifetime of a circular orbit.
        τ ≈ m/(ρ·v·Cd·A) × H/v  [seconds]
        Tells you if your science orbit will decay in weeks or decades.
        """
```

### 3b. Ground Track and Coverage

This is what connects an orbit to actual science return.

```python
class GroundTrackCalculator:
    
    def propagate_ground_track(
        self,
        planet: Planet,
        state: SpacecraftState,
        duration: float,          # seconds
        dt: float = 60.0,
    ) -> list[tuple[float, float]]:
        """
        Return list of (latitude, longitude) over time.
        Accounts for planet rotation.
        """
    
    def coverage_map(
        self,
        planet: Planet,
        orbit_elements: dict,
        duration_days: float,
        grid_resolution_deg: float = 1.0,
    ) -> np.ndarray:
        """
        2D array [lat, lon] of observation count over mission duration.
        Shows gaps, polar holes, equatorial oversampling.
        Visualisable as a global coverage heatmap.
        """
    
    def time_to_full_coverage(
        self,
        planet: Planet,
        orbit_elements: dict,
        swath_width_km: float,   # instrument field of view on surface
    ) -> float:
        """Days until every point on the surface has been observed at least once."""
    
    def revisit_time(
        self,
        planet: Planet,
        orbit_elements: dict,
        target_lat: float,
        target_lon: float,
    ) -> float:
        """
        How many hours between successive passes over a specific surface target?
        Critical for monitoring active geology (volcanoes, dust storms, etc.)
        """
```

### 3c. Interplanetary Transfer

The current model is purely planet-centric. Real mission design starts with getting *to* the planet.

```python
@dataclass
class StarSystem:
    """The planetary system context."""
    stellar_mass: float          # kg
    stellar_luminosity: float    # W
    stellar_radius: float        # m
    stellar_temperature: float   # K  (spectral type follows from L, R)
    
    planets: list[PlanetaryBody]  # planet + its orbital parameters
    
@dataclass  
class PlanetaryBody:
    planet: Planet
    orbital_semi_major_axis: float    # m — distance from star
    orbital_eccentricity: float       # 0=circular
    orbital_inclination: float        # deg — to stellar equator
    orbital_period: float             # s — derived from Kepler III
    true_anomaly_epoch: float         # rad — position at t=0

class InterplanetaryCalculator:
    
    def porkchop_plot(
        self,
        origin: PlanetaryBody,
        destination: PlanetaryBody,
        departure_window: tuple[float, float],  # Julian dates
        arrival_window: tuple[float, float],
        grid_points: int = 100,
    ) -> dict:
        """
        The fundamental mission design tool.
        Returns C3 (launch energy) and arrival v_inf as a 2D grid
        over (departure date, arrival date).
        Minima identify optimal launch windows.
        Outputs suitable for a contour plot.
        """
    
    def lambert_solution(
        self,
        r1: np.ndarray,     # departure position [m]
        r2: np.ndarray,     # arrival position [m]
        tof: float,         # time of flight [s]
        mu_star: float,     # central body gravitational parameter
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve Lambert's problem: find the velocity vectors at r1 and r2
        for a ballistic transfer in time tof.
        This is the core of all interplanetary trajectory design.
        """
    
    def delta_v_budget(
        self,
        mission: "MissionProfile",
    ) -> dict[str, float]:
        """
        Itemised ΔV: launch, TCMs, orbit insertion burn, circularisation,
        plane change, deorbit. Returns dict with breakdown and total.
        """
```

### 3d. Gravity Assists

Any realistic deep-space mission uses gravity assists.

```python
class GravityAssist:
    
    def flyby_delta_v(
        self,
        planet: Planet,
        v_inf_in: np.ndarray,     # incoming velocity at infinity [m/s]
        periapsis_altitude: float, # closest approach altitude [m]
        plane_angle: float,        # rotation angle of the flyby plane
    ) -> np.ndarray:
        """
        Compute the outgoing velocity vector after a gravity assist.
        |v_inf| is conserved; direction changes by the bending angle.
        bending angle: δ = 2 arcsin(1 / (1 + r_p·v_inf²/μ))
        """
    
    def max_delta_v_assist(self, planet: Planet, v_inf: float) -> float:
        """Maximum possible ΔV from a single flyby (grazing trajectory)."""
    
    def resonant_flyby_sequence(
        self,
        planet: Planet,
        v_inf_target: float,
        n_flybys: int,
    ) -> list[dict]:
        """
        VVEJGA-style resonant flyby planning.
        Find the sequence of flyby altitudes that progressively pump up v_inf.
        """
```

---

## Priority 4 — Habitability and Planetary Classification

This is where the science becomes genuinely exciting and where no existing simple Python tool covers the ground.

### 4a. Stellar Environment (`core/stellar.py`)

```python
@dataclass
class Star:
    name: str
    mass: float                # kg
    luminosity: float          # W
    radius: float              # m
    temperature: float         # K
    age: float                 # Gyr
    spectral_type: str         # "G2V", "M4V", etc.
    metallicity: float         # [Fe/H]  — affects planet formation
    
    @property
    def habitable_zone_inner(self) -> float:
        """
        Runaway greenhouse limit [m].
        Venus is just inside Earth's HZ inner edge.
        Kopparapu et al. (2013) empirical fit.
        """
    
    @property
    def habitable_zone_outer(self) -> float:
        """
        Maximum greenhouse limit [m].
        CO2 clouds can't warm beyond this.
        """
    
    @property
    def xuv_luminosity(self) -> float:
        """
        Extreme ultraviolet + X-ray flux [W].
        Drives atmospheric photoionisation and escape.
        Young stars emit 100-1000× more XUV than today's Sun.
        """
    
    @property
    def euv_flux_at_distance(self, orbital_distance: float) -> float:
        """W/m² — the quantity that drives Jeans escape on any atmosphere."""

STAR_PRESETS = {
    "sun":       Star("Sun",   1.989e30, 3.828e26, 6.957e8, 5778, 4.6, "G2V"),
    "proxima":   Star("Proxima Cen", 2.4e29, 5.8e23, 1.07e8, 3042, 4.85, "M5.5Ve"),
    "trappist1": Star("TRAPPIST-1", 1.8e29, 5.2e23, 8.4e7,  2566, 7.6,  "M8V"),
    "kepler452": Star("Kepler-452", 2.0e30, 5.2e26, 7.4e8,  5757, 6.0,  "G2V"),
    "tau_ceti":  Star("Tau Ceti",   1.78e30, 2.0e26, 6.2e8, 5344, 5.8,  "G8.5V"),
}
```

### 4b. Habitability Assessment

```python
class HabitabilityAssessment:
    """
    A structured scoring system — not a binary yes/no.
    Each factor is scored 0–1 and the final score reflects
    the probability of a given property being life-compatible.
    """
    
    def __init__(self, planet: Planet, star: Star, orbital_distance: float):
        self.planet = planet
        self.star   = star
        self.d      = orbital_distance
    
    def in_habitable_zone(self) -> bool:
        """Is the planet between the HZ inner and outer edges?"""
    
    def surface_temperature_score(self) -> float:
        """
        0 = uninhabitable (T < 150 K or T > 500 K)
        1 = ideal (270–315 K)
        Accounts for greenhouse effect from composition.
        """
    
    def atmospheric_retention_score(self) -> float:
        """
        Can this planet hold an atmosphere over Gyr timescales?
        Key ratio: Jeans parameter λ = v_esc² / v_thermal²
        λ > 30 → stable retention (Earth, Venus)
        λ < 10 → rapid loss (early Mars, Titan marginal)
        """
    
    def magnetic_protection_score(self) -> float:
        """
        Does the magnetosphere shield the surface from cosmic rays?
        And shield the atmosphere from solar wind stripping?
        """
    
    def water_stability_score(self) -> float:
        """
        Is liquid water stable at the surface?
        Requires: 273 K < T < 647 K AND P > 611 Pa
        """
    
    def tidal_locking_penalty(self) -> float:
        """
        Reduction score if planet is tidally locked.
        Locked planets have extreme day/night temperature gradients.
        Relevant for all M-dwarf planets.
        """
    
    def overall_score(self) -> float:
        """Weighted geometric mean of all factors."""
    
    def report(self) -> str:
        """
        Human-readable habitability assessment.
        Like a doctor's report: numerical scores + written interpretation.
        """
```

### 4c. Planetary Classification System

```python
class PlanetaryClassification:
    """
    Classify planets by their dominant character, like a field guide.
    Based on mass-radius diagram position + atmospheric properties.
    """
    
    # Size classes
    ROCKY_DWARF     = "Rocky dwarf (< 0.1 R⊕)"       # Ceres, Pluto
    ROCKY_PLANET    = "Rocky planet (0.1–2.0 R⊕)"     # Earth, Mars, Venus, Super-Earth
    MINI_NEPTUNE    = "Mini-Neptune (2.0–4.0 R⊕)"     # The radius gap
    ICE_GIANT       = "Ice giant (4.0–8.0 R⊕)"        # Neptune, Uranus
    GAS_GIANT       = "Gas giant (> 8.0 R⊕)"          # Jupiter, Saturn
    
    # Atmosphere classes
    AIRLESS         = "Airless body"
    TENUOUS         = "Tenuous exosphere (< 10 Pa)"
    THIN_ROCKY      = "Thin rocky atmosphere (10 Pa – 0.1 atm)"
    EARTH_LIKE      = "Earth-like atmosphere"
    SUPER_VENUS     = "Super-Venus (runaway greenhouse)"
    HYDROGEN_RICH   = "H₂-rich (reducing atmosphere)"
    
    # Surface state
    FROZEN          = "Frozen surface (T < 200 K)"
    TEMPERATE       = "Temperate (200–400 K)"
    HOT             = "Hot (400–700 K)"
    MOLTEN          = "Molten surface (T > 700 K)"
    
    def classify(self, planet: Planet, star: Star, 
                 orbital_distance: float) -> dict[str, str]:
        """Return size class, atmosphere class, and surface state."""
```

---

## Priority 5 — Mission Design Tools

### 5a. Mission Profile (`core/mission.py`)

```python
@dataclass
class MissionProfile:
    """
    A complete end-to-end mission specification.
    From launch vehicle to final orbit — everything needed for a
    Phase A concept study.
    """
    name: str
    target: Planet
    star_system: StarSystem
    
    # Spacecraft
    launch_mass: float           # kg — total at launch
    propulsion: list[ThrusterConfig]  # can be multi-stage
    power_system: float          # W — solar or RTG
    
    # Mission phases
    phases: list[MissionPhase]
    
    def total_delta_v(self) -> float:
        """Sum of all burns."""
    
    def propellant_mass_required(self, payload_mass: float) -> float:
        """Tsiolkovsky rocket equation applied to full mission ΔV budget."""
    
    def arrival_mass(self) -> float:
        """Mass after all burns. Must be >= payload_mass."""
    
    def power_at_destination(self, orbital_distance: float) -> float:
        """Solar power scales as 1/d². RTG decays at ~1.8%/year."""

@dataclass
class MissionPhase:
    name: str                # "Launch", "TCM-1", "Orbit Insertion", etc.
    phase_type: str          # "coast", "burn", "aerobraking", "observation"
    duration: float          # s
    delta_v: float           # m/s (0 for coasts)
    thrust_config: ThrusterConfig
```

### 5b. Aerobraking Campaign Planner

Aerobraking is one of the most complex operational activities in planetary exploration. It deserves its own module.

```python
class AerobrakingCampaign:
    """
    Plan a multi-pass aerobraking sequence to lower an elliptical
    capture orbit to the science orbit, using atmospheric drag.
    This is how MRO, MAVEN, and Venus Express reached their science orbits.
    """
    
    def __init__(self, planet: Planet, spacecraft_config: dict):
        self.planet = planet
        self.sc     = spacecraft_config   # mass, Cd, area, heat_limit
    
    def periapsis_walk_sequence(
        self,
        initial_apoapsis: float,   # m — after capture burn
        target_apoapsis: float,    # m — science orbit
        periapsis_altitude: float, # m — drag pass altitude
        heat_limit: float,         # J/m² — maximum heat load per pass
        max_g_load: float,         # g — structural limit
    ) -> list[dict]:
        """
        Returns a list of drag passes, each with:
          - pass number
          - expected ΔV from drag
          - peak heating rate
          - peak deceleration
          - new apoapsis after pass
          - recommended apoapsis-raise manoeuvre (if any)
        """
    
    def safe_periapsis_band(self, apoapsis: float) -> tuple[float, float]:
        """
        The altitude range where drag is strong enough to be useful
        but not so strong that it violates heat or g limits.
        This narrows as the orbit shrinks.
        """
    
    def total_aerobraking_duration(self, sequence: list[dict]) -> float:
        """Estimated campaign duration in days."""
    
    def abort_raise_cost(self, current_periapsis: float) -> float:
        """
        ΔV to raise periapsis above the atmosphere if an anomaly occurs.
        Every aerobraking campaign needs this contingency budget.
        """
```

### 5c. Orbit Determination and Perturbation Budget

```python
class PerturbationBudget:
    """
    Quantify all sources of orbital perturbation over a mission lifetime.
    Essential for planning station-keeping budgets.
    """
    
    def j2_drift_per_day(self, planet, orbit_elements) -> dict[str, float]:
        """dΩ/dt, dω/dt, da/dt from J2 oblateness."""
    
    def atmospheric_drag_decay(self, planet, orbit_elements, 
                                sc_config) -> dict[str, float]:
        """da/dt, de/dt, lifetime estimate from drag."""
    
    def third_body_perturbation(self, planet, orbit_elements,
                                perturbing_body: Planet,
                                perturbing_distance: float) -> float:
        """
        Gravitational tug from moons, or from the star if in high orbit.
        Relevant for halo orbits, L1/L2 missions, and high lunar orbiters.
        """
    
    def solar_radiation_pressure(self, star, orbital_distance, 
                                  orbit_elements, sc_area, sc_mass) -> float:
        """
        da/dt from radiation pressure.
        Dominant perturbation for large solar sails and high-area spacecraft.
        """
    
    def station_keeping_budget(self, planet, orbit_elements,
                                mission_duration_years: float,
                                sc_config: dict) -> float:
        """
        Total ΔV to maintain the orbit against all perturbations.
        Typically 1-10 m/s/year for low orbiters.
        """
```

---

## Priority 6 — Instrument and Observation Modelling

A science mission isn't just an orbit — it's an orbit + an instrument. The observation geometry matters enormously.

### 6a. Observation Geometry

```python
class ObservationGeometry:
    """
    Given a spacecraft position and surface target, compute
    all the parameters that determine observation quality.
    """
    
    def ground_pixel_size(
        self,
        spacecraft_altitude: float,    # m
        instrument_ifov: float,        # rad — instantaneous field of view
    ) -> float:
        """Spatial resolution on the surface [m/pixel]."""
    
    def emission_angle(
        self,
        spacecraft_position: np.ndarray,
        surface_point: np.ndarray,
        surface_normal: np.ndarray,
    ) -> float:
        """Angle between surface normal and spacecraft direction [deg]."""
    
    def solar_incidence_angle(
        self,
        surface_point: np.ndarray,
        sun_direction: np.ndarray,
        surface_normal: np.ndarray,
    ) -> float:
        """Illumination angle at the surface. > 90° = night. [deg]"""
    
    def phase_angle(
        self,
        spacecraft_position: np.ndarray,
        surface_point: np.ndarray,
        sun_direction: np.ndarray,
    ) -> float:
        """
        Sun-target-spacecraft angle.
        Low phase = forward scatter (specular)
        High phase = backward scatter (useful for haze detection)
        """
    
    def signal_to_noise_estimate(
        self,
        solar_irradiance: float,      # W/m²/nm
        surface_reflectance: float,   # 0–1
        emission_angle: float,        # deg
        incidence_angle: float,       # deg
        integration_time: float,      # s
        instrument_params: dict,      # aperture, detector QE, etc.
    ) -> float:
        """Approximate SNR for a given observation geometry."""
    
    def limb_viewing_geometry(
        self,
        spacecraft_altitude: float,
        atmosphere_altitude: float,   # altitude being probed
    ) -> dict:
        """
        Geometry for limb-sounding observations of atmosphere.
        Line of sight length through the atmosphere layer.
        """
```

### 6b. Data Volume and Downlink

Real missions are data-volume constrained.

```python
class DataBudget:
    
    def data_rate_per_orbit(
        self,
        instrument_data_rate: float,   # bits/s
        duty_cycle: float,             # fraction of orbit spent observing
        orbit_period: float,           # s
    ) -> float:
        """Total raw data per orbit [bits]."""
    
    def downlink_window(
        self,
        planet: Planet,
        orbit_elements: dict,
        ground_station_lat: float,
        ground_station_lon: float,
        min_elevation_deg: float = 10.0,
    ) -> list[tuple[float, float]]:
        """List of (start, end) contact windows per day [s]."""
    
    def link_budget(
        self,
        transmitter_power_W: float,
        antenna_gain_dBi: float,
        distance_m: float,
        frequency_GHz: float,
        receiver_gain_dBi: float,
    ) -> float:
        """Received power and data rate via Friis transmission equation."""
```

---

## Priority 7 — N-body Dynamics

The current moon model is a static perturbation. Real moon dynamics — and the stability of the planet's orbit itself — require proper N-body integration.

### 7a. Full N-body Integrator (`core/nbody.py`)

```python
class NBodySystem:
    """
    Integrates gravitational interactions among all bodies simultaneously.
    Uses a symplectic integrator (Yoshida or Leapfrog) for energy conservation.
    """
    
    def __init__(self, bodies: list[dict]):
        """
        bodies: list of {"mass": float, "position": np.ndarray, "velocity": np.ndarray}
        """
    
    def step_leapfrog(self, dt: float) -> None:
        """Single leapfrog step. Symplectic — conserves energy exactly."""
    
    def propagate(self, duration: float, dt: float) -> list[dict]:
        """Full trajectory for all bodies."""
    
    def energy(self) -> float:
        """Total mechanical energy — should be constant (conservation check)."""
    
    def angular_momentum(self) -> np.ndarray:
        """Total angular momentum vector — also conserved."""

# Applications:
# 1. Proper moon orbits — evolving positions, not fixed perturbations
# 2. Binary planet systems (Pluto-Charon, Earth-Moon as two-body)
# 3. Planet-in-stellar-system orbital evolution
# 4. Lagrange point stability (L4/L5 Trojan asteroids)
# 5. Kozai-Lidov oscillations in hierarchical triple systems
```

### 7b. Tidal Evolution

```python
class TidalDynamics:
    """
    Tidal forces between a planet and its moons or host star.
    Drives orbital migration, circularisation, and tidal locking.
    """
    
    def tidal_locking_timescale(self, planet: Planet, 
                                 moon_mass: float, 
                                 moon_distance: float) -> float:
        """
        How long until the planet (or moon) becomes tidally locked?
        t_lock ∝ ω·a⁶·m / (M_body²·R_body³·Q)
        where Q is the tidal quality factor.
        """
    
    def tidal_heating_rate(self, planet: Planet,
                            moon_mass: float,
                            moon_distance: float,
                            eccentricity: float) -> float:
        """
        Internal heating from tidal flexing [W].
        Io (Jupiter's moon): 100 TW of tidal heating → most volcanically
        active body in solar system. Europa: tidal heating → subsurface ocean.
        """
    
    def roche_limit(self, planet: Planet, moon_density: float) -> float:
        """
        Below this distance, tidal forces exceed moon's self-gravity → rings.
        d_Roche = R_planet · (2 · ρ_planet / ρ_moon)^(1/3)
        """
    
    def orbital_migration_rate(self, planet: Planet,
                                moon_mass: float,
                                moon_distance: float) -> float:
        """
        da/dt for the moon due to tidal interaction.
        Moon migrates outward if above synchronous orbit (like our Moon).
        Moon migrates inward if below (like Phobos → will crash in ~50 Myr).
        """
```

---

## Priority 8 — Geological and Surface Science

### 8a. Surface Energy Balance

```python
class SurfaceEnergyBalance:
    """
    Compute surface temperatures across the globe,
    accounting for solar flux, albedo, emissivity, and
    atmospheric greenhouse effect.
    """
    
    def insolation_map(
        self,
        planet: Planet,
        star: Star,
        orbital_distance: float,
        obliquity_deg: float,      # axial tilt
        time_of_year: float,       # 0–1 (fraction of orbital period)
        lat_resolution: float = 5, # degrees
        lon_resolution: float = 5,
    ) -> np.ndarray:
        """
        2D [lat, lon] array of solar flux at the surface [W/m²].
        Accounts for day/night cycle, obliquity, orbital eccentricity.
        """
    
    def surface_temperature_map(
        self,
        planet: Planet,
        insolation: np.ndarray,
        thermal_inertia: float,   # J m⁻² K⁻¹ s⁻¹/²  — controls day/night swing
        albedo_map: np.ndarray,
        emissivity: float = 0.95,
    ) -> np.ndarray:
        """
        Surface temperature at each lat/lon point [K].
        High thermal inertia (rock, water) = small diurnal swing.
        Low thermal inertia (dust, fine regolith) = extreme swing (e.g. Mars).
        """
    
    def permanent_shadow_regions(
        self,
        planet: Planet,
        obliquity_deg: float,
    ) -> np.ndarray:
        """
        Identify areas that never receive sunlight (polar craters).
        These are cold traps for water ice — scientifically critical.
        """
```

### 8b. Volcanic and Outgassing Model

```python
class VolcanicOutgassing:
    """
    How does the interior thermal state drive atmospheric evolution?
    """
    
    def outgassing_rate(self, planet: Planet) -> dict[str, float]:
        """
        Current volcanic gas flux at the surface [mol/s].
        Returns {H2O, CO2, SO2, H2S, N2} based on interior heat flux
        and assumed mantle composition.
        """
    
    def cumulative_outgassing(
        self,
        planet: Planet,
        age_gyr: float,
    ) -> dict[str, float]:
        """
        Total gas delivered to atmosphere over planet's lifetime [kg].
        Compare to current atmospheric mass to estimate net escape.
        """
    
    def carbonate_silicate_cycle(
        self,
        planet: Planet,
        surface_temperature: float,
        ocean_present: bool,
    ) -> float:
        """
        The long-term CO2 thermostat:
        Warm → more rain → more weathering → more CO2 drawdown → cooling.
        Cold → less weathering → CO2 builds from volcanism → warming.
        Returns equilibrium CO2 partial pressure [Pa].
        Earth's ~400 ppm is the result of this balance.
        """
```

---

## Priority 9 — Statistical and Comparative Analysis

### 9a. Population Synthesis

The batch generator exists but we need analysis tools to make sense of the population.

```python
class PopulationAnalysis:
    
    def mass_radius_diagram(
        self,
        planets: list[Planet],
        annotate: bool = True,
    ) -> Figure:
        """
        The fundamental exoplanet diagram.
        Overplot theoretical interior composition lines:
        pure iron, Earth-like (32% Fe / 68% silicate), pure rock, 50% water, pure H2.
        A planet's position tells you its bulk composition.
        """
    
    def parameter_correlation_matrix(
        self,
        planets: list[Planet],
        parameters: list[str],
    ) -> Figure:
        """
        Pearson r correlation heatmap among all chosen parameters.
        Reveals physical relationships: does high J2 correlate with
        fast rotation in our sample? (It should, physically.)
        """
    
    def habitability_distribution(
        self,
        planets: list[Planet],
        star: Star,
        orbital_distances: list[float],
    ) -> Figure:
        """
        Distribution of habitability scores across the population.
        How rare are habitable worlds in this parameter space?
        """
    
    def comparative_table(
        self,
        planets: list[Planet],
        include_solar_analogues: bool = True,
    ) -> pd.DataFrame:
        """
        Tidy data table of all physical parameters.
        Ready to export as CSV for external analysis.
        Includes Earth/Mars/Venus rows for reference if requested.
        """
```

### 9b. Sensitivity Analysis

```python
class SensitivityAnalysis:
    """
    Understand which parameters most affect mission design outcomes.
    Essential for robust mission design under planet parameter uncertainty.
    """
    
    def delta_v_sensitivity(
        self,
        planet: Planet,
        parameter: str,           # e.g. "surface_density", "scale_height"
        variation_range: tuple,   # (min_fraction, max_fraction) of nominal
        n_samples: int = 50,
        mission_config: dict = None,
    ) -> Figure:
        """
        How much does the total mission ΔV change as one parameter varies?
        For an aerobraking mission: how sensitive is the ΔV budget to
        uncertainty in atmospheric density? (Answer: very.)
        """
    
    def monte_carlo_mission(
        self,
        nominal_planet: Planet,
        uncertainty_model: dict,   # parameter → std_dev
        n_realizations: int = 1000,
        mission_config: dict = None,
    ) -> dict:
        """
        Monte Carlo propagation of planet parameter uncertainty through
        to mission performance metrics.
        Returns distributions of: arrival ΔV, insertion duration,
        aerobraking passes required, science orbit lifetime, etc.
        """
```

---

## Priority 10 — Serialisation, I/O, and Reproducibility

This is unglamorous but scientifically critical. Every result needs to be reproducible.

### 10a. Planet Serialisation

```python
# Every planet should serialise to/from a clean, readable format
import json

# To JSON
planet_dict = planet.to_dict()    # all parameters, fully specified
json.dump(planet_dict, open("mars.json", "w"), indent=2)

# From JSON — exact reproduction
planet = Planet.from_dict(json.load(open("mars.json")))

# To YAML (more readable)
planet.to_yaml("mars.yaml")
planet = Planet.from_yaml("mars.yaml")

# Fingerprint for reproducibility
planet.fingerprint()   # → "sha256:3f7a..."  — changes if any parameter changes
```

Example JSON output:

```json
{
  "name": "Mars",
  "radius_m": 3389500.0,
  "mass_kg": 6.4171e+23,
  "rotation_period_s": 88642.0,
  "atmosphere": {
    "enabled": true,
    "composition": "CO2_THIN",
    "scale_height_m": 10800.0,
    "surface_pressure_Pa": 636.0,
    "surface_density_kg_m3": 0.015,
    "surface_temp_K": 210.0,
    "lapse_rate_K_m": 0.004
  },
  "oblateness": {
    "enabled": true,
    "J2": 0.00196,
    "flattening": 0.00589
  },
  ...
}
```

### 10b. Trajectory Serialisation

```python
# trajectories should be saveable as standard formats
trajectory.to_csv("insertion_burn.csv")
trajectory.to_hdf5("mission.h5", key="insertion")  # efficient for long runs

# Load and plot later
traj = Trajectory.from_csv("insertion_burn.csv")

# CCSDS OEM format for interoperability with other astrodynamics tools
trajectory.to_oem("spacecraft_001.oem")
```

### 10c. Simulation Reproducibility

```python
@dataclass
class SimulationRecord:
    """
    Complete record of a simulation run.
    Can be shared and exactly reproduced by anyone.
    """
    planet_fingerprint: str
    integrator_config: dict       # method, dt, tolerances
    initial_state: dict
    thrust_schedule: list
    random_seeds: dict
    planet_rl_version: str        # semver
    numpy_version: str
    timestamp: str                # ISO 8601
    
    def reproduce(self) -> list[SpacecraftState]:
        """Run the exact simulation again."""
```

---

## Priority 11 — What the Visualiser Needs

The current visualiser is good for what it does. Science needs more.

### 11a. Global Map Projections

```python
def plot_ground_track(
    planet: Planet,
    trajectory: list[SpacecraftState],
    projection: str = "mollweide",   # "mollweide", "robinson", "polar"
    duration_days: float = 1.0,
    show_terminator: bool = True,     # day/night boundary
    show_ground_stations: list = None,
) -> Figure:
    """
    Ground track on a global map with day/night terminator.
    This is what planetary scientists actually look at every day.
    """

def plot_coverage_map(
    coverage: np.ndarray,    # 2D [lat, lon] observation count
    planet: Planet,
    projection: str = "mollweide",
    colorbar_label: str = "Observation count",
) -> Figure:
    """
    Global heatmap of surface coverage.
    Shows observation gaps, polar holes, equatorial oversampling.
    """
```

### 11b. Porkchop Plot

```python
def plot_porkchop(
    c3_grid: np.ndarray,          # C3 [km²/s²] at each (departure, arrival) pair
    arrival_vinf_grid: np.ndarray, # arrival v_infinity grid
    departure_dates: np.ndarray,
    arrival_dates: np.ndarray,
    c3_contours: list = [10, 20, 40, 80],
    vinf_contours: list = [2, 4, 6, 8],
) -> Figure:
    """
    The canonical mission design plot.
    Contours of C3 (launch energy) and arrival v_inf vs. departure/arrival date.
    The intersection of low-C3 and low-v_inf regions identifies launch windows.
    """
```

### 11c. Mass-Radius Diagram

```python
def plot_mass_radius_diagram(
    planets: list[Planet],
    highlight: list[str] = None,   # planet names to annotate
    show_composition_lines: bool = True,
    show_solar_system: bool = True,
) -> Figure:
    """
    Standard exoplanet characterisation diagram.
    Overplot theoretical interior structure models as composition lines.
    """
```

### 11d. Orbit Evolution Plot

```python
def plot_orbit_evolution(
    element_history: list[dict],   # list of orbital element dicts over time
    elements: list[str] = ["a", "e", "i", "RAAN"],
) -> Figure:
    """
    How do orbital elements evolve over a long simulation?
    Shows J2 precession, drag decay, third-body oscillations.
    """
```

---

## What This Becomes

If all of the above were implemented, Planet-RL would stop being an RL sandbox and become something more like **a first-principles planetary science workbench** — sitting between quick-calculation tools like GMAT's simplified interfaces and full-scale tools like SPICE or OpenOrb.

The niche it would occupy: **hypothesis testing for exoplanet science and mission concept studies**, where you want physical rigour but not the multi-month learning curve of aerospace-grade simulation suites.

Concrete scientific workflows it would enable:

1. *"Given this newly discovered exoplanet's mass and radius, what interior structure is consistent? What is the likely surface temperature? Is it in the HZ?"*

2. *"If we send a 500 kg spacecraft on a minimum-energy trajectory, can it insert into a 300 km orbit and survive the aerobraking campaign?"*

3. *"How does J2 precession rate change across a population of 500 procedurally generated planets? Does the scatter match what we'd predict from the rotation rate distribution?"*

4. *"Design a sun-synchronous repeat ground-track orbit for a hypothetical Mars-like exoplanet with 1.5× Earth's radius."*

5. *"Run a Monte Carlo over atmospheric density uncertainty: what is the 95th percentile heat load during aerobraking insertion?"*

---

## Implementation Priority Order

If implementing from scratch, the order that builds maximum scientific value per unit of effort:

| Phase | What to build | Why now |
|---|---|---|
| **1** | Multi-layer atmosphere + composition-derived properties | Fixes the biggest physical gap; everything else depends on atmosphere being realistic |
| **2** | StarSystem + HabitabilityAssessment | Immediately enables the most common science question ("is it habitable?") |
| **3** | Ground track + coverage map | Turns orbit calculations into actual science return |
| **4** | OrbitalAnalysis (frozen orbit, sun-sync, J2 drift) | Essential for mission design; builds on existing J2 code |
| **5** | Planet serialisation (JSON/YAML + fingerprinting) | Enables reproducibility; should exist before serious science use |
| **6** | InteriorConfig (layered structure, MoI) | Closes the loop on derived magnetic field and J2 |
| **7** | MissionProfile + AerobrakingCampaign | Enables end-to-end mission concept studies |
| **8** | Lambert solver + porkchop plot | Interplanetary transfers; most complex to implement correctly |
| **9** | N-body integrator | Proper moon dynamics, tidal evolution |
| **10** | PopulationAnalysis + SensitivityAnalysis | Statistical science on top of everything else |

---

*This document describes direction, not a sprint plan. Each item here is a research module in its own right — implementing any one of them well would make the tool meaningfully more useful to a working planetary scientist.*
MARKDOWN
echo "Done — $(wc -l < /home/claude/planet_rl_flat/SCIENCE_ROADMAP.md) lines"