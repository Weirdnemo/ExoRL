"""
interplanetary_env.py — Full planet-to-planet mission RL environment.

This is the central research environment of Planet-RL. An agent executes
a complete interplanetary mission in three sequential phases:

  Phase A — Launch window selection
    The agent picks a departure and arrival date from a discretised
    porkchop grid. One decision step. Commits when action[2] > 0.

  Phase B — Heliocentric cruise
    The agent coasts (or fires mid-course corrections) from departure
    planet SOI exit to arrival planet SOI entry. One step per simulated
    day. The Kepler propagator handles the coast efficiently.

  Phase C — SOI capture and orbit insertion
    Once inside the target SOI, control passes to the orbital insertion
    physics (identical to OrbitalInsertionEnv). The agent fires burns
    to capture into a circular science orbit.

Observation space: 28 × float32
  [0]     phase_norm            current phase / 2
  [1-6]   window context        C3, v_inf_arr, tof, slots, valid
  [7-13]  heliocentric state    r, dist_to_target, v, angle, time, fuel, in_soi
  [14-23] planetocentric state  alt, v, fpa, ecc, fuel, heat, R, g, rho, tgt_alt
  [24-27] target planet context hab, mass, radius, atm_pressure

Action space: 4 × float32 in [-1, 1]
  Phase A:  [0]=dep_slot  [1]=arr_slot  [2]=commit(>0 triggers)  [3]=unused
  Phase B:  [0]=thrust_T  [1]=thrust_R  [2]=thrust_N  [3]=magnitude
  Phase C:  [0]=magnitude [1]=pitch     [2]=yaw        [3]=unused

Episode terminates when:
  - Phase C succeeds (orbit achieved)
  - Phase C crashes/escapes/overheats
  - Phase B times out (missed SOI)
  - max_episode_steps exceeded

Usage
-----
    env = InterplanetaryEnv(
        departure_planet_name = "earth",
        arrival_planet_name   = "mars",
    )
    obs, info = env.reset()
    print(info['phase'], info['departure_planet'], info['arrival_planet'])

    # Phase A: pick a window
    while info['phase'] == 'window':
        action = agent.act(obs)
        obs, reward, done, trunc, info = env.step(action)

    # Phase B: cruise
    while info['phase'] == 'cruise':
        obs, reward, done, trunc, info = env.step(agent.act(obs))

    # Phase C: capture
    while not (done or trunc):
        obs, reward, done, trunc, info = env.step(agent.act(obs))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


from planet_rl.core.generator import PRESETS
from planet_rl.core.heliocentric import (
    AU,
    MU_SUN,
    HeliocentricIntegrator,
    HelioState,
    KeplerPropagator,
    LambertSolver,
    planet_state,
)
from planet_rl.core.launch_window import (
    LaunchDecisionSpace,
    orbital_period_days,
    synodic_period_days,
)
from planet_rl.core.physics import (
    AeroConfig,
    OrbitalIntegrator,
    SpacecraftState,
    ThrusterConfig,
)
from planet_rl.core.planet import Planet
from planet_rl.core.soi import (
    HyperbolicArrival,
    SphereOfInfluence,
    laplace_soi_radius,
    patched_conic_budget,
)

G = 6.674_30e-11

# --- Optional science stack ---
try:
    from planet_rl.core.interior import interior_from_bulk_density

    _INTERIOR_OK = True
except ImportError:
    _INTERIOR_OK = False

try:
    from planet_rl.core.atmosphere_science import MultiLayerAtmosphere

    _ATM_OK = True
except ImportError:
    _ATM_OK = False

try:
    from planet_rl.core.star import STAR_PRESETS, star_sun

    _STAR_OK = True
except ImportError:
    _STAR_OK = False

    def star_sun():
        class _Sun:
            mass = 1.989e30
            luminosity = 3.828e26
            name = "Sun"
            hz_inner_m = 0.975 * AU
            hz_outer_m = 1.706 * AU

            def flux_at_distance(self, d):
                return 1361 * (AU / d) ** 2

            def orbital_period(self, r):
                return 2 * math.pi * math.sqrt(r**3 / (G * self.mass))

        return _Sun()


try:
    from planet_rl.core.habitability import assess_habitability

    _HAB_OK = True
except ImportError:
    _HAB_OK = False

try:
    from planet_rl.core.orbital_analysis import FrozenOrbit

    _ORB_OK = True
except ImportError:
    _ORB_OK = False

# Observation dimension
OBS_DIM = 28

# Phase constants
PHASE_WINDOW = 0
PHASE_CRUISE = 1
PHASE_CAPTURE = 2


# ─────────────────────────────────────────────────────────────────────────────
# Episode configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MissionConfig:
    """Fixed parameters for one episode."""

    departure_planet: Planet
    arrival_planet: Planet
    departure_radius_m: float
    arrival_radius_m: float
    mu_star: float
    soi_arrival_m: float
    synodic_days: float
    hab_score: float  # arrival planet habitability
    frozen_ecc: float  # frozen orbit eccentricity at target alt


# ─────────────────────────────────────────────────────────────────────────────
# Main environment
# ─────────────────────────────────────────────────────────────────────────────


class InterplanetaryEnv(gym.Env if GYM_AVAILABLE else object):
    """
    Full interplanetary mission environment: window → cruise → capture.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        # Planet selection
        departure_planet_name: str = "earth",
        arrival_planet_name: str = "mars",
        randomise_planets: bool = False,  # future: random pairs
        # Porkchop grid
        n_dep_slots: int = 20,
        n_arr_slots: int = 20,
        window_days: float = 780.0,
        min_tof_days: float = 80.0,
        max_tof_days: float = 500.0,
        # Spacecraft
        wet_mass: float = 1500.0,
        dry_mass: float = 400.0,
        max_thrust_N: float = 500.0,
        Isp_s: float = 320.0,
        # Cruise
        dt_cruise_s: float = 86400.0,  # 1 simulated day per step
        max_cruise_steps: int = 600,  # ~20 months max cruise
        mcc_thrust_N: float = 50.0,  # mid-course correction thrust
        # Capture (Phase C)
        target_alt_m: float = 300_000.0,
        dt_capture_s: float = 10.0,
        max_capture_steps: int = 2000,
        heat_limit: float = 1e7,
        # Rewards
        r_window_scale: float = 5.0,  # scale for C3/vinf penalty
        r_soi_reached: float = 20.0,  # bonus for reaching target SOI
        r_soi_missed: float = -30.0,  # penalty for missing
        r_capture_success: float = 100.0,
        r_capture_crash: float = -50.0,
        r_step_penalty: float = -0.005,
        r_fuel_weight: float = 0.02,
        r_hab_scale: float = 10.0,  # habitability bonus at capture
        # Misc
        seed: Optional[int] = None,
    ):
        self.dep_name = departure_planet_name
        self.arr_name = arrival_planet_name
        self.randomise_planets = randomise_planets

        self.n_dep = n_dep_slots
        self.n_arr = n_arr_slots
        self.window_days = window_days
        self.min_tof_days = min_tof_days
        self.max_tof_days = max_tof_days

        self.wet_mass = wet_mass
        self.dry_mass = dry_mass
        self.max_thrust = max_thrust_N
        self.Isp = Isp_s
        self.mcc_thrust = mcc_thrust_N

        self.dt_cruise = dt_cruise_s
        self.max_cruise = max_cruise_steps
        self.dt_capture = dt_capture_s
        self.max_capture = max_capture_steps
        self.target_alt = target_alt_m
        self.heat_limit = heat_limit

        self.r_window_scale = r_window_scale
        self.r_soi_reached = r_soi_reached
        self.r_soi_missed = r_soi_missed
        self.r_capture_success = r_capture_success
        self.r_capture_crash = r_capture_crash
        self.r_step_penalty = r_step_penalty
        self.r_fuel_weight = r_fuel_weight
        self.r_hab_scale = r_hab_scale

        self._rng = np.random.RandomState(seed)

        # Episode state
        self._phase = PHASE_WINDOW
        self._cfg: Optional[MissionConfig] = None
        self._space: Optional[LaunchDecisionSpace] = None
        self._dep_idx = 0
        self._arr_idx = 0
        self._window_committed = False
        self._sc_helio: Optional[HelioState] = None
        self._sc_planet: Optional[SpacecraftState] = None
        self._integrator: Optional[OrbitalIntegrator] = None
        self._helio_integ: Optional[HeliocentricIntegrator] = None
        self._cruise_step = 0
        self._capture_step = 0
        self._fuel_init = wet_mass - dry_mass
        self._vinf_arr_actual = 0.0
        self._total_dv_used = 0.0

        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=np.full(OBS_DIM, -2.0, dtype=np.float32),
                high=np.full(OBS_DIM, 100.0, dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=np.full(4, -1.0, dtype=np.float32),
                high=np.full(4, 1.0, dtype=np.float32),
                dtype=np.float32,
            )

    # ── Planet helpers ────────────────────────────────────────────────────────

    def _make_planet(self, name: str) -> Planet:
        planet = PRESETS[name]()
        if _INTERIOR_OK:
            try:
                planet.interior = interior_from_bulk_density(planet.mean_density)
            except Exception:
                pass
        if _STAR_OK:
            try:
                sun = star_sun()
                planet.star_context = sun
                dist_map = {
                    "earth": 1.000,
                    "mars": 1.524,
                    "venus": 0.723,
                    "moon": 1.000,
                    "titan": 9.537,
                }
                planet.orbital_distance_m = dist_map.get(name, 1.0) * AU
            except Exception:
                pass
        return planet

    def _hab_score(self, planet: Planet) -> float:
        if (
            _HAB_OK
            and hasattr(planet, "star_context")
            and planet.star_context
            and hasattr(planet, "orbital_distance_m")
            and planet.orbital_distance_m
        ):
            try:
                ha = assess_habitability(
                    planet, planet.star_context, planet.orbital_distance_m
                )
                return float(ha.overall_score)
            except Exception:
                pass
        return 0.5

    def _frozen_ecc(self, planet: Planet) -> float:
        if _ORB_OK and planet.oblateness.enabled:
            try:
                fe = FrozenOrbit.frozen_eccentricity(
                    planet, planet.radius + self.target_alt, math.radians(98.0)
                )
                return float(fe) if fe is not None else 0.0
            except Exception:
                pass
        return 0.0

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # Build planets
        dep = self._make_planet(self.dep_name)
        arr = self._make_planet(self.arr_name)

        r_dep = getattr(dep, "orbital_distance_m", 1.0 * AU)
        r_arr = getattr(arr, "orbital_distance_m", 1.524 * AU)

        soi_m = laplace_soi_radius(arr.mass, r_arr)
        syn_d = synodic_period_days(
            orbital_period_days(r_dep), orbital_period_days(r_arr)
        )

        self._cfg = MissionConfig(
            departure_planet=dep,
            arrival_planet=arr,
            departure_radius_m=r_dep,
            arrival_radius_m=r_arr,
            mu_star=MU_SUN,
            soi_arrival_m=soi_m,
            synodic_days=syn_d,
            hab_score=self._hab_score(arr),
            frozen_ecc=self._frozen_ecc(arr),
        )

        # Build porkchop decision space
        self._space = LaunchDecisionSpace(
            r_dep,
            r_arr,
            n_dep=self.n_dep,
            n_arr=self.n_arr,
            window_duration_days=self.window_days,
            min_tof_days=self.min_tof_days,
            max_tof_days=self.max_tof_days,
        )

        # Reset episode state
        self._phase = PHASE_WINDOW
        self._dep_idx = self.n_dep // 2
        self._arr_idx = self.n_arr // 2
        self._window_committed = False
        self._sc_helio = None
        self._sc_planet = None
        self._integrator = None
        self._helio_integ = HeliocentricIntegrator(MU_SUN, self.Isp)
        self._cruise_step = 0
        self._capture_step = 0
        self._fuel_init = self.wet_mass - self.dry_mass
        self._vinf_arr_actual = 0.0
        self._total_dv_used = 0.0
        self._tof_s = self.min_tof_days * 86400

        obs = self._get_obs()
        info = self._make_info()
        return obs, info

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        if self._phase == PHASE_WINDOW:
            return self._step_window(action)
        elif self._phase == PHASE_CRUISE:
            return self._step_cruise(action)
        else:
            return self._step_capture(action)

    # ── Phase A: Window selection ─────────────────────────────────────────────

    def _step_window(self, action):
        # Map continuous actions to slot indices
        di = int((action[0] + 1.0) / 2.0 * (self.n_dep - 1) + 0.5)
        ai = int((action[1] + 1.0) / 2.0 * (self.n_arr - 1) + 0.5)
        self._dep_idx = max(0, min(self.n_dep - 1, di))
        self._arr_idx = max(0, min(self.n_arr - 1, ai))

        # Commit when action[2] > 0
        if float(action[2]) > 0.0:
            return self._commit_window()

        cost = self._space.cost(self._dep_idx, self._arr_idx)
        if cost["valid"]:
            r = -0.01 * (cost["c3"] / 30.0 + cost["vinf_arr"] / 8.0)
        else:
            r = -0.05

        obs = self._get_obs()
        info = self._make_info()
        return obs, float(r), False, False, info

    def _commit_window(self):
        """Lock in the chosen window and initialise the cruise phase."""
        cost = self._space.cost(self._dep_idx, self._arr_idx)

        if not cost["valid"]:
            bi, bj = self._space.best_action()
            self._dep_idx = bi
            self._arr_idx = bj
            cost = self._space.cost(bi, bj)
            r_window = -self.r_window_scale * 2
        else:
            r_window = -self.r_window_scale * (
                cost["c3"] / 30.0 + cost["vinf_arr"] / 8.0
            )

        self._tof_s = cost["tof_days"] * 86400

        # Solve Lambert to get departure velocity
        cfg = self._cfg
        dep_t = self._space.departure_days[self._dep_idx] * 86400
        arr_t = self._space.arrival_days[self._arr_idx] * 86400

        r1v, v1p = planet_state(cfg.departure_radius_m, dep_t)
        r2v, v2p = planet_state(cfg.arrival_radius_m, arr_t)
        solver = LambertSolver(MU_SUN)
        v1s, v2s = solver.solve(r1v, r2v, self._tof_s)

        if v1s is None:
            v_dep = math.sqrt(MU_SUN / cfg.departure_radius_m)
            v1s = v1p + np.array([0.0, v_dep * 0.1, 0.0])
            v2s = v2p

        # Departure burn cost (fuel via Tsiolkovsky rocket equation)
        vinf_dep = float(np.linalg.norm(v1s - v1p))
        mu_dep = G * cfg.departure_planet.mass
        r_park = cfg.departure_planet.radius + 300_000.0
        v_hyp = math.sqrt(vinf_dep**2 + 2 * mu_dep / r_park)
        v_circ = math.sqrt(mu_dep / r_park)
        dv_dep = v_hyp - v_circ

        # Apply Tsiolkovsky — but cap mass consumption so spacecraft retains
        # enough fuel for capture at arrival (realistic propellant staging)
        # Real missions carry enough for the full mission, not just departure.
        mass_after = self.wet_mass * math.exp(-dv_dep / (self.Isp * 9.80665))
        # Reserve at least 60% of propellant for capture phase
        min_mass_after = self.dry_mass + 0.4 * (self.wet_mass - self.dry_mass)
        mass_after = max(min_mass_after, mass_after)
        self._total_dv_used += dv_dep

        # Initialise heliocentric spacecraft state
        self._sc_helio = HelioState(
            position=r1v.copy(),
            velocity=v1s.copy(),
            mass_kg=mass_after,
            dry_mass_kg=self.dry_mass,
            time_s=dep_t,
        )
        self._target_r2v = r2v.copy()
        self._target_v2p = v2p.copy()
        self._arrival_time_s = arr_t

        self._phase = PHASE_CRUISE
        self._cruise_step = 0
        self._window_committed = True

        obs = self._get_obs()
        info = self._make_info()
        info["window_c3"] = cost["c3"]
        info["window_vinf_arr"] = cost["vinf_arr"]
        info["dv_departure"] = dv_dep
        return obs, float(r_window), False, False, info

    # ── Phase B: Heliocentric cruise ──────────────────────────────────────────

    def _step_cruise(self, action):
        cfg = self._cfg
        sc = self._sc_helio

        # Mid-course correction thrust (small, optional)
        t_mag = max(0.0, float(action[3])) * self.mcc_thrust
        if t_mag > 1e-3:
            # Thrust in heliocentric RTN frame
            r_hat = sc.position / (float(np.linalg.norm(sc.position)) + 1e-30)
            v_hat = sc.velocity / (float(np.linalg.norm(sc.velocity)) + 1e-30)
            n_hat = np.cross(r_hat, v_hat)
            n_hat /= float(np.linalg.norm(n_hat)) + 1e-30
            thrust_vec = t_mag * (
                float(action[1]) * r_hat
                + float(action[0]) * v_hat
                + float(action[2]) * n_hat
            )
        else:
            thrust_vec = np.zeros(3)

        # Integrate one day
        fuel_before = sc.fuel_kg
        self._sc_helio = self._helio_integ.step(sc, self.dt_cruise, thrust_vec)
        fuel_burned = fuel_before - self._sc_helio.fuel_kg
        self._total_dv_used += fuel_burned * self.Isp * 9.80665 / max(sc.mass_kg, 1.0)

        self._cruise_step += 1

        # Check SOI entry
        target_pos, target_vel = planet_state(
            cfg.arrival_radius_m, self._sc_helio.time_s
        )
        dist = float(np.linalg.norm(self._sc_helio.position - target_pos))

        # Reward: small penalty per step + fuel cost
        r = self.r_step_penalty - self.r_fuel_weight * fuel_burned / max(
            1.0, self._fuel_init
        )

        in_soi = dist < cfg.soi_arrival_m
        timed_out = self._cruise_step >= self.max_cruise

        if in_soi:
            # Arrived at target SOI — transition to capture
            vinf_vec = self._sc_helio.velocity - target_vel
            self._vinf_arr_actual = float(np.linalg.norm(vinf_vec))
            r += self.r_soi_reached * max(0.0, 1.0 - self._vinf_arr_actual / 8000.0)
            self._phase = PHASE_CAPTURE
            self._capture_step = 0
            self._init_capture_phase(target_pos, target_vel)
            obs = self._get_obs()
            info = self._make_info()
            info["vinf_arrival"] = self._vinf_arr_actual
            return obs, float(r), False, False, info

        if timed_out:
            # Missed the target
            r += self.r_soi_missed
            obs = self._get_obs()
            info = self._make_info()
            info["timeout"] = True
            return obs, float(r), True, False, info

        obs = self._get_obs()
        info = self._make_info()
        return obs, float(r), False, False, info

    def _init_capture_phase(self, target_pos, target_vel):
        """Switch from heliocentric to planetocentric frame and init integrator."""
        cfg = self._cfg

        # Transform to planet frame
        r_planet = self._sc_helio.position - target_pos
        v_planet = self._sc_helio.velocity - target_vel

        # Place spacecraft at SOI entry with correct planetocentric state
        # Use magnitude from actual position, but initialise for capture
        r_mag = float(np.linalg.norm(r_planet))
        v_mag = self._vinf_arr_actual

        # Set up capture approach: place at 300 km periapsis approach altitude
        # Initialise state at a plausible capture approach position
        r_peri = cfg.arrival_planet.radius + 300_000.0

        # Use the actual planetocentric state from the heliocentric trajectory
        self._sc_planet = SpacecraftState(
            x=float(r_planet[0]),
            y=float(r_planet[1]),
            z=float(r_planet[2]),
            vx=float(v_planet[0]),
            vy=float(v_planet[1]),
            vz=float(v_planet[2]),
            mass=self._sc_helio.mass_kg,
            dry_mass=self.dry_mass,
        )

        # Build integrator around the arrival planet
        arr = cfg.arrival_planet
        self._integrator = OrbitalIntegrator(
            planet=arr,
            thruster=ThrusterConfig(max_thrust=self.max_thrust, Isp=self.Isp),
            aero=AeroConfig(enabled=arr.atmosphere.enabled),
        )

    # ── Phase C: Capture and insertion ────────────────────────────────────────

    def _step_capture(self, action):
        cfg = self._cfg
        arr = cfg.arrival_planet

        # Same thrust mapping as OrbitalInsertionEnv
        t_mag = (float(action[0]) + 1.0) / 2.0 * self.max_thrust
        pitch = float(action[1]) * math.pi / 2.0
        yaw = float(action[2]) * math.pi

        r_hat = self._sc_planet.position / (self._sc_planet.radius + 1e-9)
        t_hat = np.cross(np.array([0.0, 0.0, 1.0]), r_hat)
        t_n = float(np.linalg.norm(t_hat))
        t_hat = t_hat / t_n if t_n > 1e-9 else np.array([0.0, 1.0, 0.0])
        n_hat = np.cross(r_hat, t_hat)

        thrust_vec = t_mag * (
            math.cos(pitch) * math.cos(yaw) * t_hat
            + math.cos(pitch) * math.sin(yaw) * n_hat
            + math.sin(pitch) * r_hat
        )

        arr_old = self._sc_planet.to_array()
        arr_new = self._integrator.step_rk4(
            arr_old, thrust_vec, self.dt_capture, self._sc_planet.time
        )
        self._sc_planet = SpacecraftState.from_array(
            arr_new, time=self._sc_planet.time + self.dt_capture, dry_mass=self.dry_mass
        )
        self._capture_step += 1

        alt = self._sc_planet.radius - arr.radius
        spd = self._sc_planet.speed
        v_tgt = arr.circular_orbit_speed(self.target_alt)

        # Orbital elements
        r = self._sc_planet.radius
        eps = spd**2 / 2.0 - arr.mu / r
        h = float(
            np.linalg.norm(np.cross(self._sc_planet.position, self._sc_planet.velocity))
        )
        ecc = (
            math.sqrt(max(0.0, 1.0 + 2.0 * eps * h**2 / arr.mu**2))
            if arr.mu > 0
            else 0.0
        )

        # Terminal conditions
        crashed = alt < 0
        escaped = self._sc_planet.radius > 10 * (arr.radius + self.target_alt)
        overheated = self._sc_planet.heat_load > self.heat_limit
        no_fuel = self._sc_planet.fuel_mass < 1.0
        timeout = self._capture_step >= self.max_capture

        alt_err = abs(alt - self.target_alt) / (self.target_alt + 1e-9)
        success = alt_err < 0.05 and ecc < 0.05 and not crashed

        # Dense shaping reward
        r_step = (
            self.r_step_penalty
            - abs(alt / self.target_alt - 1.0) * 0.3
            - abs(spd / v_tgt - 1.0) * 0.3
            - ecc * 0.2
        )

        terminated = crashed or escaped or overheated or success
        truncated = timeout or no_fuel

        if success:
            r_step += self.r_capture_success
            # Frozen orbit quality bonus
            fe_err = abs(ecc - cfg.frozen_ecc)
            r_step += 5.0 * max(0.0, 1.0 - fe_err / 0.005)
            # Habitability bonus — more interesting target = more reward
            r_step += self.r_hab_scale * cfg.hab_score

        if crashed or escaped or overheated:
            r_step += self.r_capture_crash

        obs = self._get_obs()
        info = self._make_info()
        info.update(
            {
                "altitude_m": alt,
                "eccentricity": ecc,
                "speed_mps": spd,
                "fuel_kg": self._sc_planet.fuel_mass,
                "success": success,
                "crashed": crashed,
            }
        )
        return obs, float(r_step), terminated, truncated, info

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        cfg = self._cfg
        if cfg is None:
            return obs

        # [0] phase
        obs[0] = float(self._phase) / 2.0

        # [1-6] window context — current slot choice
        cost = self._space.cost(self._dep_idx, self._arr_idx)
        obs[1] = float(self._dep_idx) / max(1, self.n_dep - 1)
        obs[2] = float(self._arr_idx) / max(1, self.n_arr - 1)
        if cost["valid"]:
            obs[3] = min(cost["c3"] / 30.0, 1.0)
            obs[4] = min(cost["vinf_arr"] / 8.0, 1.0)
            obs[5] = min(cost["tof_days"] / self.max_tof_days, 1.0)
            obs[6] = 1.0
        else:
            obs[3] = 1.0
            obs[4] = 1.0
            obs[5] = 1.0
            obs[6] = 0.0

        # [7-13] heliocentric cruise state
        if self._sc_helio is not None:
            sc = self._sc_helio
            r = float(np.linalg.norm(sc.position))
            v = float(np.linalg.norm(sc.velocity))
            obs[7] = min(r / (5 * AU), 1.0)

            target_pos, _ = planet_state(cfg.arrival_radius_m, sc.time_s)
            dist = float(np.linalg.norm(sc.position - target_pos))
            obs[8] = min(dist / (5 * AU), 1.0)
            obs[9] = min(v / 35000.0, 1.0)

            # Angle to target
            if r > 0 and dist > 0:
                cos_a = float(
                    np.dot(sc.position, target_pos)
                    / (r * float(np.linalg.norm(target_pos)) + 1e-30)
                )
                obs[10] = math.acos(max(-1.0, min(1.0, cos_a))) / math.pi
            obs[11] = min(float(sc.time_s) / max(1.0, self._tof_s), 1.0)
            obs[12] = sc.fuel_kg / max(1.0, self._fuel_init)
            obs[13] = 1.0 if dist < cfg.soi_arrival_m else 0.0

        # [14-23] planetocentric capture state
        if self._sc_planet is not None and self._phase == PHASE_CAPTURE:
            sc = self._sc_planet
            arr = cfg.arrival_planet
            alt = sc.radius - arr.radius
            spd = sc.speed
            v_tgt = arr.circular_orbit_speed(self.target_alt)

            r_hat = sc.position / (sc.radius + 1e-9)
            v_hat = sc.velocity / (spd + 1e-9)
            fpa = math.asin(max(-1.0, min(1.0, float(np.dot(r_hat, v_hat)))))

            r = sc.radius
            eps = spd**2 / 2.0 - arr.mu / r
            h = float(np.linalg.norm(np.cross(sc.position, sc.velocity)))
            ecc = (
                math.sqrt(max(0.0, 1.0 + 2.0 * eps * h**2 / arr.mu**2))
                if arr.mu > 0
                else 0.0
            )

            rho = (
                arr.atmosphere.density_at_altitude(max(alt, 0.0))
                if arr.atmosphere.enabled
                else 0.0
            )

            obs[14] = alt / (self.target_alt + 1e-9)
            obs[15] = spd / (v_tgt + 1e-9)
            obs[16] = fpa / math.pi
            obs[17] = min(ecc, 2.0)
            obs[18] = sc.fuel_mass / max(1.0, self._fuel_init)
            obs[19] = min(sc.heat_load / (self.heat_limit + 1e-9), 1.0)
            obs[20] = arr.radius / 6.371e6
            obs[21] = arr.surface_gravity / 9.81
            obs[22] = min(rho / 1.225, 10.0)
            obs[23] = self.target_alt / (arr.radius + 1e-9)

        # [24-27] target planet context (constant)
        if cfg is not None:
            arr = cfg.arrival_planet
            obs[24] = float(cfg.hab_score)
            obs[25] = min(arr.mass / 5.972e24, 20.0)
            obs[26] = min(arr.radius / 6.371e6, 5.0)
            obs[27] = min(
                arr.atmosphere.surface_pressure / 1e5
                if arr.atmosphere.enabled
                else 0.0,
                100.0,
            )

        return obs.clip(-2.0, 100.0)

    # ── Info ──────────────────────────────────────────────────────────────────

    def _make_info(self) -> dict:
        phase_names = {
            PHASE_WINDOW: "window",
            PHASE_CRUISE: "cruise",
            PHASE_CAPTURE: "capture",
        }
        info = {
            "phase": phase_names.get(self._phase, "unknown"),
            "departure_planet": self.dep_name,
            "arrival_planet": self.arr_name,
            "cruise_step": self._cruise_step,
            "capture_step": self._capture_step,
            "total_dv_m_s": self._total_dv_used,
            "fuel_remaining": (
                self._sc_helio.fuel_kg
                if self._sc_helio
                else self._sc_planet.fuel_mass
                if self._sc_planet
                else self._fuel_init
            ),
            "habitability": self._cfg.hab_score if self._cfg else 0.0,
        }
        return info

    def get_mission_config(self) -> Optional[MissionConfig]:
        return self._cfg
