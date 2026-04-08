"""
env.py — Gymnasium-compatible RL environment for orbital insertion.

Upgraded to use the Planet-RL science stack:
  - J2 from interior model (derived_J2) when available, not hand-set
  - Atmosphere density from MultiLayerAtmosphere (layered, physical)
  - Star context attached so habitability can be scored
  - Extended observation vector (18 floats) with science quantities
  - Curriculum mode: sorts episode difficulty by habitability score
  - Science reward bonus: rewards frozen/sun-sync orbit design

Observation space (18 x float32)
─────────────────────────────────
Dynamic state (changes every step):
  [0]  altitude_norm          alt / target_alt             (0..~3)
  [1]  speed_norm             v / v_circular_target        (0..~2)
  [2]  flight_path_angle_norm fpa / pi                     (-1..1)
  [3]  eccentricity           orbital eccentricity         (0..~2)
  [4]  fuel_fraction          fuel_remaining / fuel_init   (0..1)
  [5]  heat_norm              heat_load / heat_limit       (0..1)

Planet context (constant per episode):
  [6]  radius_norm            R / R_earth                  (0..4)
  [7]  gravity_norm           g / 9.81                     (0..5)
  [8]  atm_density_norm       rho(alt) / 1.225             (0..10)
  [9]  target_alt_norm        target_alt / R               (0..0.3)
  [10] j2_norm                J2 x 1000                    (0..3)
  [11] mag_field_norm         B / 60e-6                    (0..2)
  [12] surface_pressure_norm  log(1+P_srf/1e5)/log(101)    (0..1)
  [13] habitability_norm      score                        (0..1)
  [14] star_type_norm         0=M, 0.5=K, 1=G              (0..1)
  [15] orbital_dist_norm      d_AU / 5                     (0..1)

Orbit design signals (constant per episode):
  [16] frozen_ecc_norm        frozen eccentricity x 100    (0..1)
  [17] ss_inc_norm            sun-sync inc / 180           (0..1)

Action space (3 x float32, all in [-1, 1])
  [0]  thrust_magnitude  ->  [0, max_thrust] N
  [1]  pitch             ->  [-pi/2, pi/2] rad
  [2]  yaw               ->  [-pi, pi] rad
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

    class gym:
        class Env:
            pass

    class spaces:
        @staticmethod
        def Box(*a, **kw):
            pass


from planet_rl.core.generator import PRESETS, PlanetGenerator
from planet_rl.core.physics import (
    AeroConfig,
    OrbitalIntegrator,
    SpacecraftState,
    ThrusterConfig,
)
from planet_rl.core.planet import Planet

# Science stack imports — graceful fallback if not available
try:
    from planet_rl.core.interior import InteriorConfig, interior_from_bulk_density

    _INTERIOR_OK = True
except ImportError:
    _INTERIOR_OK = False

try:
    from planet_rl.core.atmosphere_science import MultiLayerAtmosphere

    _ATM_OK = True
except ImportError:
    _ATM_OK = False

try:
    from planet_rl.core.star import AU as _AU
    from planet_rl.core.star import STAR_PRESETS, SpectralType

    _STAR_OK = True
except ImportError:
    _STAR_OK = False
    _AU = 1.496e11

try:
    from planet_rl.core.habitability import assess_habitability

    _HAB_OK = True
except ImportError:
    _HAB_OK = False

try:
    from planet_rl.core.orbital_analysis import FrozenOrbit, SunSynchronousOrbit

    _ORB_OK = True
except ImportError:
    _ORB_OK = False

OBS_DIM = 18


# ─────────────────────────────────────────────────────────────────────────────
# Science context — pre-computed once per episode
# ─────────────────────────────────────────────────────────────────────────────


class PlanetScienceContext:
    """Pre-computed science quantities for obs[10:18]. Cached for full episode."""

    def __init__(
        self,
        planet: Planet,
        target_altitude_m: float,
        star_year_s: float = 365.25 * 86400,
    ):
        self.planet = planet

        # J2
        if hasattr(planet, "derived_J2"):
            try:
                self.j2 = float(planet.derived_J2())
            except Exception:
                self.j2 = planet.oblateness.J2 if planet.oblateness.enabled else 0.0
        else:
            self.j2 = planet.oblateness.J2 if planet.oblateness.enabled else 0.0

        # Magnetic field
        if hasattr(planet, "derived_magnetic_field_T"):
            try:
                self.B_T = float(planet.derived_magnetic_field_T())
            except Exception:
                self.B_T = 0.0
        else:
            self.B_T = 0.0

        # Surface pressure
        self.P_srf = (
            planet.atmosphere.surface_pressure if planet.atmosphere.enabled else 0.0
        )

        # Habitability score
        self.hab_score = 0.5
        if (
            _HAB_OK
            and hasattr(planet, "star_context")
            and planet.star_context is not None
            and hasattr(planet, "orbital_distance_m")
            and planet.orbital_distance_m
        ):
            try:
                ha = assess_habitability(
                    planet, planet.star_context, planet.orbital_distance_m
                )
                self.hab_score = float(ha.overall_score)
            except Exception:
                pass

        # Star type
        self.star_type_norm = 0.7
        if (
            _STAR_OK
            and hasattr(planet, "star_context")
            and planet.star_context is not None
        ):
            mapping = {
                SpectralType.M: 0.0,
                SpectralType.K: 0.3,
                SpectralType.G: 0.7,
                SpectralType.F: 0.85,
                SpectralType.A: 1.0,
            }
            st = planet.star_context.spectral_type
            self.star_type_norm = float(mapping.get(st, 0.7))

        # Orbital distance
        self.orbital_dist_au = 1.0
        if hasattr(planet, "orbital_distance_m") and planet.orbital_distance_m:
            self.orbital_dist_au = planet.orbital_distance_m / _AU

        # Frozen orbit eccentricity
        self.frozen_ecc = 0.0
        if _ORB_OK and self.j2 > 0:
            try:
                fe = FrozenOrbit.frozen_eccentricity(
                    planet, planet.radius + target_altitude_m, math.radians(98.0)
                )
                self.frozen_ecc = float(fe) if fe is not None else 0.0
            except Exception:
                pass

        # Sun-sync inclination
        self.ss_inc_deg = 90.0
        if _ORB_OK and self.j2 > 0:
            try:
                ss = SunSynchronousOrbit.sun_sync_inclination(
                    planet, target_altitude_m, star_year_s
                )
                self.ss_inc_deg = float(ss) if ss is not None else 90.0
            except Exception:
                pass

    def to_obs_slice(self) -> np.ndarray:
        """Return obs[10:18] as float32 array."""
        return np.array(
            [
                min(self.j2 * 1000, 3.0),
                min(abs(self.B_T) / 60e-6, 2.0),
                math.log1p(self.P_srf / 1e5)
                / math.log1p(100.0),  # log-normalised: Earth=0.15, Venus=0.99
                float(self.hab_score),
                float(self.star_type_norm),
                min(self.orbital_dist_au / 5.0, 1.0),
                min(self.frozen_ecc * 100, 1.0),
                min(self.ss_inc_deg / 180.0, 1.0),
            ],
            dtype=np.float32,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Science atmosphere wrapper
# ─────────────────────────────────────────────────────────────────────────────


class ScienceAtmosphere:
    """Wraps MultiLayerAtmosphere; falls back to exponential model."""

    def __init__(self, planet: Planet):
        self.planet = planet
        self._multi = None
        if _ATM_OK and planet.atmosphere.enabled:
            try:
                self._multi = MultiLayerAtmosphere.from_atmosphere_config(
                    planet.atmosphere, planet
                )
            except Exception:
                pass

    def density_at(self, altitude_m: float) -> float:
        alt = max(0.0, altitude_m)
        if self._multi is not None:
            try:
                return float(self._multi.density_at(alt))
            except Exception:
                pass
        if self.planet.atmosphere.enabled:
            return float(self.planet.atmosphere.density_at_altitude(alt))
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Planet wiring helper
# ─────────────────────────────────────────────────────────────────────────────


def _attach_science_context(
    planet: Planet, rng: Optional[np.random.RandomState] = None
) -> Planet:
    """Attach interior model + star to a freshly generated planet."""
    # Interior
    if _INTERIOR_OK and (not hasattr(planet, "interior") or planet.interior is None):
        try:
            planet.interior = interior_from_bulk_density(planet.mean_density)
        except Exception:
            try:
                planet.interior = InteriorConfig.earth_like()
            except Exception:
                pass

    # Star + orbital distance
    if _STAR_OK and (
        not hasattr(planet, "star_context") or planet.star_context is None
    ):
        keys = list(STAR_PRESETS.keys())
        idx = rng.randint(0, len(keys) - 1) if rng is not None else 0
        key = keys[idx]
        try:
            star = STAR_PRESETS[key]()
            planet.star_context = star
            hz_mid = (star.hz_inner_m + star.hz_outer_m) / 2.0
            if rng is not None:
                spread = (star.hz_outer_m - star.hz_inner_m) * 0.3
                planet.orbital_distance_m = float(hz_mid + rng.uniform(-spread, spread))
            else:
                planet.orbital_distance_m = hz_mid
        except Exception:
            pass

    return planet


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum planet pool
# ─────────────────────────────────────────────────────────────────────────────


class CurriculumPool:
    """
    Pre-generates N planets, scores by habitability, serves easy->hard.
    Easy planets (high habitability score) = Earth-like, intuitive physics.
    Hard planets (low score) = exotic, challenging environments.
    """

    def __init__(
        self,
        generator: PlanetGenerator,
        gen_kwargs: dict,
        pool_size: int = 200,
        easy_first: bool = True,
    ):
        self.easy_first = easy_first
        self._pool: list[Planet] = []
        self._idx = 0
        self._build(generator, gen_kwargs, pool_size)

    def _build(self, generator, gen_kwargs, n):
        scored = []
        for _ in range(n):
            p = generator.generate(**gen_kwargs)
            p = _attach_science_context(p, generator.rng)
            score = 0.5
            if (
                _HAB_OK
                and hasattr(p, "star_context")
                and p.star_context
                and hasattr(p, "orbital_distance_m")
                and p.orbital_distance_m
            ):
                try:
                    ha = assess_habitability(p, p.star_context, p.orbital_distance_m)
                    score = float(ha.overall_score)
                except Exception:
                    pass
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=self.easy_first)
        self._pool = [p for _, p in scored]

    def next(self) -> Planet:
        p = self._pool[self._idx % len(self._pool)]
        self._idx += 1
        return p


# ─────────────────────────────────────────────────────────────────────────────
# Main environment
# ─────────────────────────────────────────────────────────────────────────────


class OrbitalInsertionEnv(gym.Env if GYM_AVAILABLE else object):
    """
    Orbital insertion environment wired to the Planet-RL science stack.

    Key upgrades:
      - Physics-consistent J2 from interior model
      - Multi-layer atmosphere density for accurate aerobraking
      - 18-dimensional observation with habitability and orbit design signals
      - Curriculum mode: easy -> hard by habitability score
      - Science reward bonus for orbit quality
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        planet: Optional[Planet] = None,
        planet_preset: Optional[str] = None,
        generator_seed: Optional[int] = None,
        randomize_planet: bool = True,
        atmosphere_enabled: bool = True,
        terrain_enabled: bool = True,
        magnetic_field_enabled: bool = True,
        oblateness_enabled: bool = True,
        moons_enabled: bool = False,
        # Science stack flags
        use_science_atmosphere: bool = True,
        use_science_j2: bool = True,
        attach_star: bool = True,
        # Curriculum
        curriculum_mode: bool = False,
        curriculum_pool_size: int = 200,
        curriculum_easy_first: bool = True,
        # Mission
        target_altitude: float = 300_000,
        target_eccentricity: float = 0.01,
        initial_altitude: float = 0,
        initial_speed_ratio: float = 1.08,  # ecc≈0.08; was 1.3 (ecc=0.69)
        # Spacecraft
        wet_mass: float = 1000.0,
        dry_mass: float = 300.0,
        max_thrust: float = 500.0,
        Isp: float = 320.0,
        # Simulation
        dt: float = 10.0,
        max_steps: int = 2000,
        heat_limit: float = 1e7,
        # Rewards
        reward_success: float = 100.0,
        reward_crash: float = -50.0,
        reward_fuel_weight: float = 0.01,
        reward_step_penalty: float = -0.01,
        reward_science_orbit: float = 10.0,
        # Observation
        obs_dim: int = OBS_DIM,
        render_mode: Optional[str] = None,
    ):
        self.randomize_planet = randomize_planet
        self.planet_preset = planet_preset
        self._fixed_planet = planet
        self.generator = PlanetGenerator(seed=generator_seed)

        self.gen_kwargs = dict(
            atmosphere_enabled=atmosphere_enabled,
            terrain_enabled=terrain_enabled,
            magnetic_field_enabled=magnetic_field_enabled,
            oblateness_enabled=oblateness_enabled,
            moons_enabled=moons_enabled,
        )

        self.use_science_atm = use_science_atmosphere and _ATM_OK
        self.use_science_j2 = use_science_j2 and _INTERIOR_OK
        self.attach_star = attach_star and _STAR_OK

        self._curriculum_pool: Optional[CurriculumPool] = None
        if curriculum_mode and randomize_planet:
            self._curriculum_pool = CurriculumPool(
                self.generator,
                self.gen_kwargs,
                pool_size=curriculum_pool_size,
                easy_first=curriculum_easy_first,
            )

        self.target_altitude = target_altitude
        self.target_ecc = target_eccentricity
        self.initial_altitude = initial_altitude
        self.initial_speed_ratio = initial_speed_ratio

        self.wet_mass = wet_mass
        self.dry_mass = dry_mass
        self.max_thrust = max_thrust
        self.Isp = Isp

        self.dt = dt
        self.max_steps = max_steps
        self.heat_limit = heat_limit

        self.reward_success = reward_success
        self.reward_crash = reward_crash
        self.reward_fuel_weight = reward_fuel_weight
        self.reward_step_penalty = reward_step_penalty
        self.reward_science_orbit = reward_science_orbit

        self.obs_dim = obs_dim
        self.render_mode = render_mode

        self.planet: Optional[Planet] = None
        self.state: Optional[SpacecraftState] = None
        self.integrator: Optional[OrbitalIntegrator] = None
        self._sci_atm: Optional[ScienceAtmosphere] = None
        self._sci_ctx: Optional[PlanetScienceContext] = None
        self._step_count = 0
        self._initial_fuel = wet_mass - dry_mass
        self._trajectory: list[SpacecraftState] = []

        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=np.zeros(self.obs_dim, dtype=np.float32),
                high=np.full(self.obs_dim, 10.0, dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

    # ── Planet selection ──────────────────────────────────────────────────────
    def _select_planet(self) -> Planet:
        if self._fixed_planet is not None:
            return self._fixed_planet
        if self._curriculum_pool is not None:
            return self._curriculum_pool.next()
        if self.planet_preset and self.planet_preset in PRESETS:
            p = PRESETS[self.planet_preset]()
        elif self.randomize_planet:
            p = self.generator.generate(**self.gen_kwargs)
        else:
            p = PRESETS["earth"]()

        if self.attach_star:
            p = _attach_science_context(p, self.generator.rng)

        # Override J2 with interior-derived value
        if self.use_science_j2 and p.oblateness.enabled and hasattr(p, "derived_J2"):
            try:
                j2 = float(p.derived_J2())
                if j2 > 0:
                    p.oblateness.J2 = j2
            except Exception:
                pass

        return p

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        sc = self.state
        p = self.planet
        alt = sc.radius - p.radius
        spd = sc.speed
        v_target = p.circular_orbit_speed(self.target_altitude)

        r_hat = sc.position / (sc.radius + 1e-9)
        v_hat = sc.velocity / (spd + 1e-9)
        fpa = math.asin(max(-1.0, min(1.0, float(np.dot(r_hat, v_hat)))))
        fuel_frac = sc.fuel_mass / (self._initial_fuel + 1e-9)

        if self._sci_atm is not None:
            atm_density = self._sci_atm.density_at(alt)
        elif p.atmosphere.enabled:
            atm_density = p.atmosphere.density_at_altitude(max(alt, 0.0))
        else:
            atm_density = 0.0

        r = sc.radius
        eps = spd**2 / 2.0 - p.mu / r
        h = float(np.linalg.norm(np.cross(sc.position, sc.velocity)))
        ecc = math.sqrt(max(0.0, 1.0 + 2.0 * eps * h**2 / p.mu**2)) if p.mu > 0 else 0.0

        obs_core = np.array(
            [
                alt / (self.target_altitude + 1e-9),
                spd / (v_target + 1e-9),
                fpa / math.pi,
                min(ecc, 2.0),
                float(fuel_frac),
                min(sc.heat_load / (self.heat_limit + 1e-9), 1.0),
                p.radius / 6.371e6,
                p.surface_gravity / 9.81,
                min(atm_density / 1.225, 10.0),
                self.target_altitude / (p.radius + 1e-9),
            ],
            dtype=np.float32,
        )

        if self.obs_dim == 10:
            return obs_core

        obs_sci = (
            self._sci_ctx.to_obs_slice()
            if self._sci_ctx is not None
            else np.zeros(8, dtype=np.float32)
        )
        return np.concatenate([obs_core, obs_sci])

    # ── Reward ────────────────────────────────────────────────────────────────
    def _compute_reward(
        self, obs: np.ndarray, done: bool, success: bool, crashed: bool
    ) -> float:
        """
        Lyapunov potential-based reward shaping (Ng et al. 1999).

        Potential: Φ(s) = -dv_to_circular(s) / dv_init
          → Φ = -1.0 at episode start (far from circular)
          → Φ = 0.0  at perfect circular orbit

        Shaped reward: r = γΦ(s') - Φ(s)
          → Positive whenever the spacecraft is making progress toward circular
          → Negative if moving away from circular
          → Guaranteed not to change the optimal policy (Ng et al.)

        This replaces the old altitude/speed difference shaping which
        penalised correct manoeuvres (the spacecraft climbs during a
        retrograde burn at periapsis, increasing -|alt - target| penalty).
        """
        dv_curr = (
            float(obs[1]) * self._dv_init
        )  # spd_norm * dv_init is approx dv_remaining
        # Better: use stored dv from obs or compute directly
        # obs[1] = speed/v_circ so dv = |speed - v_circ| = v_circ * |spd_norm - 1|
        alt_norm = float(obs[0])
        spd_norm = float(obs[1])
        ecc = float(obs[3])

        # ── Lyapunov shaping ──────────────────────────────────────────────────
        # dv_to_circular ≈ v_circ × |spd_norm - 1| (works well near circular)
        dv_to_circ = abs(spd_norm - 1.0) + ecc * 0.5  # normalised units
        phi_curr = -dv_to_circ
        phi_prev = -self._phi_prev

        gamma = 0.99
        shaped = gamma * phi_curr - phi_prev  # γΦ(s') - Φ(s)
        self._phi_prev = dv_to_circ  # store for next step

        # ── Progress bonus ────────────────────────────────────────────────────
        # Small additional reward whenever ecc decreases (encourages exploration)
        ecc_improvement = max(0.0, self._ecc_prev - ecc)
        progress = ecc_improvement * 2.0
        self._ecc_prev = ecc

        # ── Altitude proximity ────────────────────────────────────────────────
        # Mild bonus for being near target altitude (avoids drifting far away)
        alt_proximity = max(0.0, 1.0 - abs(alt_norm - 1.0)) * 0.1

        total = (
            shaped
            + progress
            + alt_proximity
            - self.reward_fuel_weight * (1.0 - float(obs[4]))
        )

        # ── Terminal rewards ──────────────────────────────────────────────────
        if success:
            total += self.reward_success
            if self.obs_dim >= 18 and self._sci_ctx is not None:
                ecc_err = abs(ecc - self._sci_ctx.frozen_ecc)
                total += (
                    self.reward_science_orbit * max(0.0, 1.0 - ecc_err / 0.01) * 0.5
                )
                total += self.reward_science_orbit * self._sci_ctx.hab_score * 0.5

        if crashed:
            total += self.reward_crash

        return float(total)

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.generator.rng.seed(seed)

        self.planet = self._select_planet()

        self._sci_atm = ScienceAtmosphere(self.planet) if self.use_science_atm else None

        if self.obs_dim >= 18:
            star_yr = 365.25 * 86400
            if (
                hasattr(self.planet, "star_context")
                and self.planet.star_context is not None
                and hasattr(self.planet, "orbital_distance_m")
                and self.planet.orbital_distance_m
            ):
                try:
                    star_yr = self.planet.star_context.orbital_period(
                        self.planet.orbital_distance_m
                    )
                except Exception:
                    pass
            self._sci_ctx = PlanetScienceContext(
                self.planet, self.target_altitude, star_yr
            )
        else:
            self._sci_ctx = None

        self.integrator = OrbitalIntegrator(
            planet=self.planet,
            thruster=ThrusterConfig(max_thrust=self.max_thrust, Isp=self.Isp),
            aero=AeroConfig(enabled=self.planet.atmosphere.enabled),
        )

        approach_alt = self.target_altitude + self.initial_altitude
        v0 = self.planet.circular_orbit_speed(approach_alt) * self.initial_speed_ratio

        self.state = SpacecraftState(
            x=self.planet.radius + approach_alt,
            y=0.0,
            z=0.0,
            vx=0.0,
            vy=v0,
            vz=0.0,
            mass=self.wet_mass,
            dry_mass=self.dry_mass,
        )
        self._step_count = 0
        self._initial_fuel = self.wet_mass - self.dry_mass
        self._trajectory = [self.state]

        # Reward shaping state (Lyapunov potential tracking)
        v_circ_0 = self.planet.circular_orbit_speed(approach_alt)
        self._dv_init = max(abs(v0 - v_circ_0), 1.0)
        ecc_0 = abs(self.initial_speed_ratio - 1.0) * 2.0  # approximate
        self._phi_prev = abs(self.initial_speed_ratio - 1.0) + ecc_0 * 0.5
        self._ecc_prev = ecc_0

        obs = self._get_obs()
        info = {
            "planet": self.planet.name,
            "j2": self.planet.oblateness.J2 if self.planet.oblateness.enabled else 0.0,
            "habitability": self._sci_ctx.hab_score if self._sci_ctx else 0.0,
            "star": (
                self.planet.star_context.name
                if hasattr(self.planet, "star_context") and self.planet.star_context
                else "none"
            ),
            "atm_model": (
                "multi-layer"
                if self._sci_atm and self._sci_atm._multi
                else "exponential"
            ),
        }
        return obs, info

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        t_mag = (float(action[0]) + 1.0) / 2.0 * self.max_thrust
        pitch = float(action[1]) * math.pi / 2.0
        yaw = float(action[2]) * math.pi

        r_hat = self.state.position / (self.state.radius + 1e-9)
        t_hat = np.cross(np.array([0.0, 0.0, 1.0]), r_hat)
        t_n = float(np.linalg.norm(t_hat))
        t_hat = t_hat / t_n if t_n > 1e-9 else np.array([0.0, 1.0, 0.0])
        n_hat = np.cross(r_hat, t_hat)

        thrust_vec = t_mag * (
            math.cos(pitch) * math.cos(yaw) * t_hat
            + math.cos(pitch) * math.sin(yaw) * n_hat
            + math.sin(pitch) * r_hat
        )

        # Patch atmosphere density with science model for this step
        _orig = None
        if self._sci_atm is not None and self._sci_atm._multi is not None:
            _orig = self.planet.atmosphere.density_at_altitude
            self.planet.atmosphere.density_at_altitude = self._sci_atm.density_at

        arr = self.state.to_array()
        arr_new = self.integrator.step_rk4(arr, thrust_vec, self.dt, self.state.time)
        self.state = SpacecraftState.from_array(
            arr_new, time=self.state.time + self.dt, dry_mass=self.dry_mass
        )

        if _orig is not None:
            self.planet.atmosphere.density_at_altitude = _orig

        self._step_count += 1
        self._trajectory.append(self.state)

        obs = self._get_obs()

        alt = self.state.radius - self.planet.radius
        crashed = alt < 0
        escaped = self.state.radius > 10.0 * (self.planet.radius + self.target_altitude)
        overheated = self.state.heat_load > self.heat_limit
        timeout = self._step_count >= self.max_steps
        no_fuel = self.state.fuel_mass < 1.0

        alt_err = abs(alt - self.target_altitude) / (self.target_altitude + 1e-9)
        ecc = float(obs[3])
        success = alt_err < 0.05 and ecc < 0.05 and not crashed

        terminated = crashed or escaped or overheated or success
        truncated = timeout or no_fuel

        reward = self._compute_reward(obs, terminated or truncated, success, crashed)

        info = {
            "altitude_m": alt,
            "speed_mps": self.state.speed,
            "eccentricity": ecc,
            "fuel_kg": self.state.fuel_mass,
            "heat_load": self.state.heat_load,
            "crashed": crashed,
            "escaped": escaped,
            "success": success,
            "step": self._step_count,
            "habitability": self._sci_ctx.hab_score if self._sci_ctx else 0.0,
        }
        return obs, reward, bool(terminated), bool(truncated), info

    def get_trajectory(self) -> list[SpacecraftState]:
        return self._trajectory

    def get_science_context(self) -> Optional[PlanetScienceContext]:
        return self._sci_ctx
