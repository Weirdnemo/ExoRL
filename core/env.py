"""
env.py — Gymnasium-compatible RL environment for orbital insertion.
Wraps Planet + OrbitalIntegrator with a clean obs/action interface.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    # Lightweight stub so the file is still importable without gymnasium
    class gym:
        class Env: pass
    class spaces:
        @staticmethod
        def Box(*a, **kw): pass
        @staticmethod
        def Dict(*a, **kw): pass

from core.planet import Planet
from core.generator import PlanetGenerator, PRESETS
from core.physics import SpacecraftState, OrbitalIntegrator, ThrusterConfig, AeroConfig


class OrbitalInsertionEnv(gym.Env if GYM_AVAILABLE else object):
    """
    Gymnasium environment for orbital insertion around procedurally
    generated (or preset) planets.

    Observation space:
        altitude_norm     : altitude / target_altitude            (0..~2)
        speed_norm        : current_speed / target_orbital_speed  (0..~2)
        flight_path_angle : radians                               (-π..π)
        eccentricity      : 0..~2
        fuel_fraction     : remaining fuel / initial fuel         (0..1)
        heat_norm         : heat_load / heat_limit                (0..1)
        planet_radius_norm : planet.radius / R_EARTH              (0..4)
        surface_grav_norm : planet.surface_gravity / 9.81         (0..~5)
        atm_density_norm  : atm density at current alt / 1.225    (0..~100)
        target_alt_norm   : target_altitude / planet.radius       (0..~1)

    Action space (continuous, all in [-1, 1]):
        thrust_magnitude  : mapped to [0, max_thrust]
        pitch             : thrust pitch angle  [-π/2, π/2]
        yaw               : thrust yaw angle    [-π, π]
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # ── Configuration ─────────────────────────────────────────────────────────
    def __init__(
        self,
        # Planet source
        planet: Optional[Planet] = None,       # fixed planet; None = randomize
        planet_preset: Optional[str] = None,   # "earth", "mars", etc.
        generator_seed: Optional[int] = None,
        # Feature toggles for random generation
        randomize_planet: bool = True,
        atmosphere_enabled: bool = True,
        terrain_enabled: bool = True,
        magnetic_field_enabled: bool = False,
        oblateness_enabled: bool = False,
        moons_enabled: bool = False,
        # Mission parameters
        target_altitude: float = 200_000,  # [m]
        target_eccentricity: float = 0.01, # desired eccentricity
        initial_altitude: float = 100_000, # above target (hyperbolic approach)
        initial_speed_ratio: float = 1.3,  # v0 / v_circular (approach speed)
        # Spacecraft
        wet_mass: float = 1000.0,
        dry_mass: float = 300.0,
        max_thrust: float = 500.0,
        Isp: float = 320.0,
        # Simulation
        dt: float = 10.0,            # integrator step [s]
        max_steps: int = 2000,
        heat_limit: float = 1e7,     # [J/m²] abort if exceeded
        # Rewards
        reward_success: float = 100.0,
        reward_crash: float = -50.0,
        reward_fuel_weight: float = 0.01,
        reward_step_penalty: float = -0.01,
        render_mode: Optional[str] = None,
    ):
        self.randomize_planet   = randomize_planet
        self.planet_preset      = planet_preset
        self._fixed_planet      = planet
        self.generator          = PlanetGenerator(seed=generator_seed)

        # Feature toggle dict passed to generator
        self.gen_kwargs = dict(
            atmosphere_enabled   = atmosphere_enabled,
            terrain_enabled      = terrain_enabled,
            magnetic_field_enabled = magnetic_field_enabled,
            oblateness_enabled   = oblateness_enabled,
            moons_enabled        = moons_enabled,
        )

        self.target_altitude    = target_altitude
        self.target_ecc         = target_eccentricity
        self.initial_altitude   = initial_altitude
        self.initial_speed_ratio = initial_speed_ratio

        self.wet_mass   = wet_mass
        self.dry_mass   = dry_mass
        self.max_thrust = max_thrust
        self.Isp        = Isp

        self.dt         = dt
        self.max_steps  = max_steps
        self.heat_limit = heat_limit

        self.reward_success     = reward_success
        self.reward_crash       = reward_crash
        self.reward_fuel_weight = reward_fuel_weight
        self.reward_step_penalty = reward_step_penalty
        self.render_mode        = render_mode

        # Internal state
        self.planet: Optional[Planet] = None
        self.state: Optional[SpacecraftState] = None
        self.integrator: Optional[OrbitalIntegrator] = None
        self._step_count = 0
        self._initial_fuel = wet_mass - dry_mass
        self._trajectory: list[SpacecraftState] = []

        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=np.zeros(10, dtype=np.float32),
                high=np.full(10, 10.0, dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=np.array([-1, -1, -1], dtype=np.float32),
                high=np.array([ 1,  1,  1], dtype=np.float32),
                dtype=np.float32,
            )

    # ── Planet selection ──────────────────────────────────────────────────────
    def _select_planet(self) -> Planet:
        if self._fixed_planet is not None:
            return self._fixed_planet
        if self.planet_preset and self.planet_preset in PRESETS:
            return PRESETS[self.planet_preset]()
        if self.randomize_planet:
            return self.generator.generate(**self.gen_kwargs)
        return PRESETS["earth"]()

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        sc = self.state
        p  = self.planet
        alt = sc.radius - p.radius
        spd = sc.speed
        v_target = p.circular_orbit_speed(self.target_altitude)

        # Flight path angle
        r_hat = sc.position / (sc.radius + 1e-9)
        v_hat = sc.velocity / (spd + 1e-9)
        fpa = math.asin(max(-1.0, min(1.0, np.dot(r_hat, v_hat))))

        fuel_frac = sc.fuel_mass / (self._initial_fuel + 1e-9)

        atm_density = p.atmosphere.density_at_altitude(max(alt, 0)) if p.atmosphere.enabled else 0.0

        # Rough eccentricity from vis-viva
        r = sc.radius
        eps = spd**2/2 - p.mu/r
        a = -p.mu / (2 * eps) if eps < 0 else float("inf")
        h_vec = np.cross(sc.position, sc.velocity)
        h = np.linalg.norm(h_vec)
        ecc = math.sqrt(max(0, 1 + 2 * eps * h**2 / p.mu**2)) if p.mu > 0 else 0

        obs = np.array([
            alt / (self.target_altitude + 1e-9),
            spd / (v_target + 1e-9),
            fpa / math.pi,
            min(ecc, 2.0),
            fuel_frac,
            min(sc.heat_load / (self.heat_limit + 1e-9), 1.0),
            p.radius / 6.371e6,
            p.surface_gravity / 9.81,
            min(atm_density / 1.225, 10.0),
            self.target_altitude / (p.radius + 1e-9),
        ], dtype=np.float32)

        return obs

    # ── Reward ────────────────────────────────────────────────────────────────
    def _compute_reward(self, obs: np.ndarray, done: bool, success: bool, crashed: bool) -> float:
        alt_norm = obs[0]
        spd_norm = obs[1]
        ecc      = obs[3]

        r_orbit = -abs(alt_norm - 1.0) * 0.5
        r_speed = -abs(spd_norm - 1.0) * 0.5
        r_ecc   = -ecc * 0.3
        r_fuel  = -self.reward_fuel_weight * (1.0 - obs[4])
        r_step  = self.reward_step_penalty

        total = r_orbit + r_speed + r_ecc + r_fuel + r_step

        if success:
            total += self.reward_success
        if crashed:
            total += self.reward_crash

        return float(total)

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.generator.rng.seed(seed)

        self.planet = self._select_planet()
        self.integrator = OrbitalIntegrator(
            planet=self.planet,
            thruster=ThrusterConfig(max_thrust=self.max_thrust, Isp=self.Isp),
            aero=AeroConfig(enabled=self.planet.atmosphere.enabled),
        )

        # Hyperbolic approach: place above target, flying tangentially but faster
        approach_alt = self.target_altitude + self.initial_altitude
        v_circ  = self.planet.circular_orbit_speed(approach_alt)
        v0 = v_circ * self.initial_speed_ratio

        self.state = SpacecraftState(
            x=self.planet.radius + approach_alt, y=0, z=0,
            vx=0, vy=v0, vz=0,
            mass=self.wet_mass, dry_mass=self.dry_mass,
        )
        self._step_count = 0
        self._initial_fuel = self.wet_mass - self.dry_mass
        self._trajectory = [self.state]

        obs = self._get_obs()
        return obs, {"planet": self.planet.name, "planet_summary": self.planet.summary()}

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        # Map action [-1,1]³ → thrust vector
        t_mag   = (action[0] + 1) / 2 * self.max_thrust
        pitch   = action[1] * math.pi / 2
        yaw     = action[2] * math.pi

        # Thrust in local RTN frame → inertial
        r_hat = self.state.position / (self.state.radius + 1e-9)
        t_hat = np.cross(np.array([0,0,1]), r_hat)
        t_hat_norm = np.linalg.norm(t_hat)
        if t_hat_norm < 1e-9:
            t_hat = np.array([0,1,0])
        else:
            t_hat /= t_hat_norm
        n_hat = np.cross(r_hat, t_hat)

        thrust_vec = t_mag * (
            math.cos(pitch) * math.cos(yaw) * t_hat +
            math.cos(pitch) * math.sin(yaw) * n_hat +
            math.sin(pitch) * r_hat
        )

        # Integrate one step
        arr = self.state.to_array()
        arr_new = self.integrator.step_rk4(arr, thrust_vec, self.dt, self.state.time)
        self.state = SpacecraftState.from_array(arr_new,
                                                time=self.state.time + self.dt,
                                                dry_mass=self.dry_mass)
        self._step_count += 1
        self._trajectory.append(self.state)

        obs = self._get_obs()

        # Terminal conditions
        alt = self.state.radius - self.planet.radius
        crashed  = alt < 0
        escaped  = self.state.radius > 10 * (self.planet.radius + self.target_altitude)
        overheated = self.state.heat_load > self.heat_limit
        out_of_fuel = self.state.fuel_mass < 1.0
        timeout  = self._step_count >= self.max_steps

        # Success: within 5% of target orbit, low eccentricity, stable
        alt_err  = abs(alt - self.target_altitude) / (self.target_altitude + 1e-9)
        ecc      = obs[3]
        success  = (alt_err < 0.05 and ecc < 0.05 and not crashed)

        terminated = crashed or escaped or overheated or success
        truncated  = timeout or out_of_fuel

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
        }

        return obs, reward, terminated, truncated, info

    def get_trajectory(self) -> list[SpacecraftState]:
        return self._trajectory
