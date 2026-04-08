"""
science_ops_env.py — Post-insertion science operations RL environment.

The agent controls a spacecraft already in orbit and must maximise science
value over a fixed mission lifetime. This is the most physically rich
environment in ExoRL: it uses power, communications, ground track
coverage, atmospheric spectroscopy, and habitability scoring together.

Episode structure
-----------------
    Duration  : N_ORBITS orbital periods (default 60)
    Time step : 1 orbital period per step
    Actions   : altitude change, inclination change, observation mode, downlink mode

At each step the agent chooses:
    [0] dalt   ∈ [-1,1] → maps to altitude change ΔR ∈ [-50,+50] km
                           (executed as Hohmann transfer, costs ΔV from fuel budget)
    [1] dinc   ∈ [-1,1] → maps to inclination change ΔI ∈ [-5,+5] deg
                           (very expensive! costs ΔV ≈ 2v_circ sin(ΔI/2))
    [2] observe ∈ [-1,1] → > 0 means instruments ON, ≤ 0 means instruments OFF
    [3] downlink ∈ [-1,1] → > 0 means transmitting, ≤ 0 means storing data

Observation vector (16 floats)
-------------------------------
    [0]  altitude_norm         alt / alt_max
    [1]  inclination_norm      inc / π
    [2]  power_margin_norm     (P_avg - P_bus) / P_max
    [3]  eclipse_frac          fraction of orbit in shadow
    [4]  data_buffer_norm      buffer / buffer_max  [0-1]
    [5]  downlink_rate_norm    rate / rate_max
    [6]  coverage_frac         cumulative surface coverage [0-1]
    [7]  time_remaining_norm   steps_left / N_ORBITS
    [8]  fuel_norm             fuel_remaining / fuel_init
    [9]  solar_flux_norm       flux / 1361  (Earth reference)
    [10] planet_radius_norm    R / R_earth
    [11] planet_gravity_norm   g / 9.81
    [12] planet_hab_score      habitability score [0-1]
    [13] tsm_norm              TSM / 100 (capped)
    [14] earth_dist_norm       d_earth / 10 AU
    [15] orbital_period_norm   T_orb / 7200 s (2hr reference)

Reward
------
    Per step:
        + science_value × observe_active
        - power_penalty   (if power budget exceeded)
        - overflow_penalty (if data buffer overflows)
        - manoeuvre_cost  (ΔV used for orbit changes)

    End of episode:
        + coverage_bonus × cumulative_coverage × hab_score
        + downlinked_bonus × total_data_downlinked / data_capacity

Usage
-----
    from exorl.core.science_ops_env import ScienceOpsEnv

    env = ScienceOpsEnv(planet_preset="mars")
    obs, info = env.reset()

    while True:
        action = agent.act(obs)
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            break

    print(info["total_coverage"])
    print(info["total_data_Gb"])
    print(info["science_score"])
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_OK = True
except ImportError:
    GYM_OK = False

    class gym:
        class Env:
            pass

    class spaces:
        @staticmethod
        def Box(*a, **kw):
            pass


# ── Constants ──────────────────────────────────────────────────────────────────
AU = 1.495_978_707e11
R_EARTH = 6.371e6
G_EARTH = 9.81
G_SI = 6.674_30e-11

# ── Default mission parameters ─────────────────────────────────────────────────
DEFAULT_N_ORBITS = 60  # episode length in orbital periods
DEFAULT_ALT_MIN_KM = 200.0  # minimum safe altitude [km]
DEFAULT_ALT_MAX_KM = 1200.0  # maximum science altitude [km]
DEFAULT_INC_MIN_DEG = 0.0  # minimum inclination
DEFAULT_INC_MAX_DEG = 180.0  # maximum inclination
DEFAULT_FUEL_KG = 200.0  # onboard propellant [kg]  (for manoeuvres)
DEFAULT_DRY_MASS_KG = 800.0  # dry spacecraft mass [kg]
DEFAULT_ISP_S = 220.0  # thruster Isp [s]  (monoprop typical)
DEFAULT_PANEL_AREA = 10.0  # solar panel area [m²]
DEFAULT_BUS_POWER = 300.0  # spacecraft bus power demand [W]
DEFAULT_BUFFER_GB = 50.0  # onboard data buffer [Gb]
DEFAULT_EARTH_DIST = 1.524  # default Earth distance [AU]  (Mars-like)
OBS_DIM = 16


class ScienceOpsEnv(gym.Env if GYM_OK else object):
    """
    Post-insertion science operations environment.

    The agent controls a spacecraft in orbit around a planet and must
    choose altitude, inclination, observation, and downlink modes to
    maximise the total science value returned to Earth.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        # Planet selection
        planet_preset: Optional[str] = "earth",
        randomize_planet: bool = False,
        generator_seed: int = 0,
        # Orbit parameters
        initial_altitude_km: float = 400.0,
        initial_inc_deg: float = 98.0,
        alt_min_km: float = DEFAULT_ALT_MIN_KM,
        alt_max_km: float = DEFAULT_ALT_MAX_KM,
        # Mission duration
        n_orbits: int = DEFAULT_N_ORBITS,
        # Resources
        fuel_kg: float = DEFAULT_FUEL_KG,
        dry_mass_kg: float = DEFAULT_DRY_MASS_KG,
        isp_s: float = DEFAULT_ISP_S,
        panel_area_m2: float = DEFAULT_PANEL_AREA,
        bus_power_W: float = DEFAULT_BUS_POWER,
        buffer_Gb: float = DEFAULT_BUFFER_GB,
        # Communications
        earth_dist_au: float = DEFAULT_EARTH_DIST,
        antenna_diam_m: float = 1.0,
        # Rewards
        r_science_per_orbit: float = 1.0,
        r_power_penalty: float = -0.5,
        r_overflow_penalty: float = -1.0,
        r_manoeuvre_penalty: float = -2.0,  # per km/s ΔV
        r_coverage_bonus: float = 50.0,
        r_data_bonus: float = 20.0,
        seed: int = None,
    ):
        self.planet_preset = planet_preset
        self.randomize_planet = randomize_planet
        self.generator_seed = generator_seed

        self.initial_alt_m = initial_altitude_km * 1e3
        self.initial_inc_deg = initial_inc_deg
        self.alt_min_m = alt_min_km * 1e3
        self.alt_max_m = alt_max_km * 1e3

        self.n_orbits = n_orbits
        self.fuel_init = fuel_kg
        self.dry_mass = dry_mass_kg
        self.isp = isp_s
        self.panel_area = panel_area_m2
        self.bus_power = bus_power_W
        self.buffer_max_Gb = buffer_Gb
        self.earth_dist_au = earth_dist_au
        self.antenna_diam = antenna_diam_m

        self.r_science = r_science_per_orbit
        self.r_power_pen = r_power_penalty
        self.r_overflow_pen = r_overflow_penalty
        self.r_manoeuvre_pen = r_manoeuvre_penalty
        self.r_coverage_bonus = r_coverage_bonus
        self.r_data_bonus = r_data_bonus

        self._rng = np.random.RandomState(seed)

        # Episode state
        self._planet = None
        self._alt_m = self.initial_alt_m
        self._inc_deg = self.initial_inc_deg
        self._fuel_kg = fuel_kg
        self._buffer_Gb = 0.0
        self._step = 0
        self._coverage = 0.0
        self._total_data_Gb = 0.0
        self._total_science = 0.0
        self._power_model = None
        self._comms_model = None
        self._hab_score = 0.0
        self._tsm = 0.0

        if GYM_OK:
            self.observation_space = spaces.Box(
                low=np.zeros(OBS_DIM, dtype=np.float32),
                high=np.full(OBS_DIM, 10.0, dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=np.full(4, -1.0, dtype=np.float32),
                high=np.full(4, 1.0, dtype=np.float32),
                dtype=np.float32,
            )

    # ── Planet loading ─────────────────────────────────────────────────────────

    def _load_planet(self):
        from exorl.core.generator import PRESETS, PlanetGenerator
        from exorl.core.interior import interior_from_bulk_density
        from exorl.core.star import star_sun

        if self.randomize_planet:
            gen = PlanetGenerator(seed=self.generator_seed + self._episode_count)
            p = gen.generate(
                atmosphere_enabled=True,
                oblateness_enabled=True,
                magnetic_field_enabled=True,
            )
        else:
            p = PRESETS[self.planet_preset or "earth"]()

        p.interior = interior_from_bulk_density(p.mean_density)
        sun = star_sun()
        p.star_context = sun
        if not getattr(p, "orbital_distance_m", None):
            p.orbital_distance_m = self.earth_dist_au * AU
        return p

    def _compute_hab_tsm(self):
        """Pre-compute habitability score and TSM for this planet."""
        try:
            from exorl.core.habitability import assess_habitability

            ha = assess_habitability(
                self._planet, self._planet.star_context, self._planet.orbital_distance_m
            )
            self._hab_score = float(ha.overall_score)
        except Exception:
            self._hab_score = 0.3

        try:
            from exorl.core.observation import transmission_spectroscopy_metric

            tsm = transmission_spectroscopy_metric(
                self._planet, self._planet.star_context, self._planet.orbital_distance_m
            )
            self._tsm = min(float(tsm), 100.0)
        except Exception:
            self._tsm = 10.0

    def _build_resource_models(self):
        """Build power and comms models for current orbit."""
        from exorl.core.comms import AntennaConfig, CommsModel
        from exorl.core.power import PowerModel, SolarPanelConfig

        panel = SolarPanelConfig(area_m2=self.panel_area, efficiency=0.28)
        self._power_model = PowerModel(
            self._planet,
            altitude_m=self._alt_m,
            inclination_deg=self._inc_deg,
            panel=panel,
            orbital_dist_m=getattr(self._planet, "orbital_distance_m", AU),
        )
        ant = AntennaConfig(
            diameter_m=self.antenna_diam,
            frequency_GHz=8.4,
            tx_power_W=50.0,
        )
        self._comms_model = CommsModel(
            self._planet,
            antenna=ant,
            earth_dist_au=self.earth_dist_au,
        )

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if not hasattr(self, "_episode_count"):
            self._episode_count = 0
        self._episode_count += 1

        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self._planet = self._load_planet()
        self._alt_m = self.initial_alt_m
        self._inc_deg = self.initial_inc_deg
        self._fuel_kg = self.fuel_init
        self._buffer_Gb = 0.0
        self._step = 0
        self._coverage = 0.0
        self._total_data_Gb = 0.0
        self._total_science = 0.0

        self._compute_hab_tsm()
        self._build_resource_models()

        obs = self._get_obs()
        info = self._make_info()
        return obs, info

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        dalt_km = float(action[0]) * 50.0  # ±50 km altitude change
        dinc_deg = float(action[1]) * 5.0  # ±5° inclination change
        observing = float(action[2]) > 0.0
        downlinking = float(action[3]) > 0.0

        reward = 0.0

        # ── Manoeuvre cost (Hohmann altitude change) ──────────────────────────
        new_alt = np.clip(self._alt_m + dalt_km * 1e3, self.alt_min_m, self.alt_max_m)
        dv_alt = self._hohmann_dv(self._alt_m, new_alt)

        # Inclination change ΔV = 2 v_circ sin(Δi/2)
        new_inc = np.clip(
            self._inc_deg + dinc_deg, DEFAULT_INC_MIN_DEG, DEFAULT_INC_MAX_DEG
        )
        v_circ = self._planet.circular_orbit_speed(self._alt_m)
        dv_inc = 2 * v_circ * math.sin(math.radians(abs(dinc_deg) / 2))

        dv_total = dv_alt + dv_inc
        if dv_total > 0:
            # Tsiolkovsky: propellant consumed
            g0 = 9.80665
            m_i = self._fuel_kg + self.dry_mass
            m_f = m_i * math.exp(-dv_total / (self.isp * g0))
            dm = m_i - m_f
            self._fuel_kg = max(0.0, self._fuel_kg - dm)
            reward += self.r_manoeuvre_pen * (dv_total / 1000)  # per km/s

            # Apply manoeuvre
            self._alt_m = float(new_alt)
            self._inc_deg = float(new_inc)
            self._build_resource_models()

        # ── Power accounting ──────────────────────────────────────────────────
        instrument_power = 150.0 if observing else 0.0
        downlink_power = 80.0 if downlinking else 0.0
        total_demand = self.bus_power + instrument_power + downlink_power

        if not self._power_model.is_power_positive(total_demand):
            reward += self.r_power_pen
            observing = False  # instruments cut off
            downlinking = False

        # ── Observation: science data generation ─────────────────────────────
        if observing:
            # Coverage increment per orbit depends on altitude and inclination
            dcov = self._coverage_increment_per_orbit()
            self._coverage = min(1.0, self._coverage + dcov)

            # Science value = coverage_rate × TSM_factor × hab_weight
            tsm_factor = math.log1p(self._tsm) / math.log1p(100)
            sci_value = self.r_science * dcov * (1 + tsm_factor) * (1 + self._hab_score)
            self._total_science += sci_value
            reward += sci_value

            # Data generation: 1 Gb per orbit at reference science rate
            data_per_orbit = 1.0 * (1 + tsm_factor)  # Gb/orbit
            overflow = max(0.0, self._buffer_Gb + data_per_orbit - self.buffer_max_Gb)
            self._buffer_Gb = min(self._buffer_Gb + data_per_orbit, self.buffer_max_Gb)
            if overflow > 0:
                reward += self.r_overflow_pen * overflow

        # ── Downlink: drain data buffer ───────────────────────────────────────
        if downlinking:
            T_orb_s = self._orbital_period_s()
            rate_Gbps = self._comms_model.downlink_rate_bps / 1e9
            # Contact window ≈ 20% of orbit period (visibility fraction)
            contact_s = T_orb_s * 0.20
            downlinked = min(self._buffer_Gb, rate_Gbps * contact_s)
            self._buffer_Gb = max(0.0, self._buffer_Gb - downlinked)
            self._total_data_Gb += downlinked

        # ── Step counter ──────────────────────────────────────────────────────
        self._step += 1
        done = (self._step >= self.n_orbits) or (self._fuel_kg <= 0)
        trunc = False

        # ── Episode-end bonus ─────────────────────────────────────────────────
        if done:
            reward += self.r_coverage_bonus * self._coverage * self._hab_score
            reward += self.r_data_bonus * (
                self._total_data_Gb / max(self.buffer_max_Gb, 1.0)
            )

        obs = self._get_obs()
        info = self._make_info()
        info.update(
            {
                "observing": observing,
                "downlinking": downlinking,
                "dv_used_m_s": dv_total,
                "coverage_this_step": dcov if observing else 0.0,
            }
        )
        return obs, float(reward), bool(done), bool(trunc), info

    # ── Physics helpers ────────────────────────────────────────────────────────

    def _orbital_period_s(self) -> float:
        r = self._planet.radius + self._alt_m
        return 2 * math.pi * math.sqrt(r**3 / self._planet.mu)

    def _hohmann_dv(self, alt1_m: float, alt2_m: float) -> float:
        """Total ΔV for a two-burn Hohmann transfer [m/s]."""
        r1 = self._planet.radius + alt1_m
        r2 = self._planet.radius + alt2_m
        if abs(r1 - r2) < 1:
            return 0.0
        mu = self._planet.mu
        dv1, dv2 = self._planet.hohmann_delta_v(alt1_m, alt2_m)
        return abs(dv1) + abs(dv2)

    def _coverage_increment_per_orbit(self) -> float:
        """
        Surface coverage fraction added per orbital period.

        Higher altitude → wider swath width → more coverage per orbit.
        More inclination away from 0° or 180° → better coverage distribution.
        Coverage saturates toward 1.0 as the surface is fully mapped.

        Simplified model: swath_fraction ∝ (altitude / R_planet)^0.5
        """
        alt_frac = self._alt_m / self._planet.radius
        swath = 0.012 * math.sqrt(alt_frac)  # fraction of circumference per pass
        inc_factor = abs(math.sin(math.radians(self._inc_deg)))
        inc_factor = max(0.1, inc_factor)  # polar orbits cover more
        remaining = 1.0 - self._coverage
        return min(swath * inc_factor, remaining * 0.5)

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[0] = self._alt_m / self.alt_max_m
        obs[1] = self._inc_deg / 180.0
        obs[2] = (
            min(
                max(
                    (self._power_model.avg_power_W - self.bus_power)
                    / max(self._power_model.peak_power_W, 1.0),
                    -1.0,
                ),
                1.0,
            )
            * 0.5
            + 0.5
        )
        obs[3] = self._power_model.eclipse_fraction
        obs[4] = self._buffer_Gb / max(self.buffer_max_Gb, 1.0)
        obs[5] = min(self._comms_model.downlink_rate_Mbps / 100.0, 1.0)
        obs[6] = self._coverage
        obs[7] = max(0.0, 1.0 - self._step / self.n_orbits)
        obs[8] = self._fuel_kg / max(self.fuel_init, 1.0)
        obs[9] = min(self._power_model.solar_flux_W_m2 / 1361.0, 5.0)
        obs[10] = self._planet.radius / R_EARTH
        obs[11] = self._planet.surface_gravity / G_EARTH
        obs[12] = self._hab_score
        obs[13] = min(self._tsm / 100.0, 1.0)
        obs[14] = min(self.earth_dist_au / 10.0, 1.0)
        obs[15] = min(self._orbital_period_s() / 7200.0, 2.0)

        return obs.clip(0.0, 10.0)

    def _make_info(self) -> dict:
        return {
            "planet": self._planet.name,
            "altitude_km": self._alt_m / 1e3,
            "inclination_deg": self._inc_deg,
            "fuel_kg": self._fuel_kg,
            "buffer_Gb": self._buffer_Gb,
            "coverage": self._coverage,
            "total_data_Gb": self._total_data_Gb,
            "total_science": self._total_science,
            "hab_score": self._hab_score,
            "tsm": self._tsm,
            "step": self._step,
            "science_score": self._total_science * (1 + self._coverage),
        }
