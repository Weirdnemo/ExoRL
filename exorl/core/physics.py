"""
physics.py — Spacecraft dynamics & orbital mechanics simulator.
Integrates equations of motion around any Planet object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from exorl.core.planet import Planet


# ── Spacecraft state ──────────────────────────────────────────────────────────
@dataclass
class SpacecraftState:
    # Position in planet-centred inertial frame [m]
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    # Velocity [m/s]
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    # Propulsion
    mass: float = 1000.0  # wet mass [kg]
    dry_mass: float = 300.0  # dry mass [kg]
    # Thermal
    heat_load: float = 0.0  # [J/m²] accumulated heating
    # Time
    time: float = 0.0  # mission elapsed time [s]

    # ── Convenience ───────────────────────────────────────────────────────────
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])

    @property
    def radius(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    @property
    def fuel_mass(self) -> float:
        return max(0.0, self.mass - self.dry_mass)

    @property
    def altitude(self, planet_radius: float = 6.371e6) -> float:
        """Rough altitude — physics engine injects planet radius properly."""
        return self.radius - planet_radius

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.vx,
                self.vy,
                self.vz,
                self.mass,
                self.heat_load,
            ]
        )

    @classmethod
    def from_array(
        cls, arr: np.ndarray, time: float = 0, dry_mass: float = 300
    ) -> "SpacecraftState":
        return cls(
            x=arr[0],
            y=arr[1],
            z=arr[2],
            vx=arr[3],
            vy=arr[4],
            vz=arr[5],
            mass=arr[6],
            heat_load=arr[7],
            time=time,
            dry_mass=dry_mass,
        )

    @classmethod
    def circular_orbit(
        cls,
        planet: Planet,
        altitude: float,
        inclination: float = 0.0,
        wet_mass: float = 1000.0,
        dry_mass: float = 300.0,
        **kwargs,
    ) -> "SpacecraftState":
        """Factory: place spacecraft in circular orbit at given altitude."""
        r = planet.radius + altitude
        v = planet.circular_orbit_speed(altitude)
        inc = math.radians(inclination)
        return cls(
            x=r,
            y=0,
            z=0,
            vx=0,
            vy=v * math.cos(inc),
            vz=v * math.sin(inc),
            mass=wet_mass,
            dry_mass=dry_mass,
            **kwargs,
        )


# ── Thruster model ────────────────────────────────────────────────────────────
@dataclass
class ThrusterConfig:
    max_thrust: float = 500.0  # [N]
    Isp: float = 320.0  # specific impulse [s]
    enabled: bool = True

    @property
    def exhaust_velocity(self) -> float:
        return self.Isp * 9.80665  # ve = Isp * g₀


# ── Aerodynamics config ───────────────────────────────────────────────────────
@dataclass
class AeroConfig:
    enabled: bool = True
    Cd: float = 2.2  # drag coefficient
    reference_area: float = 10.0  # [m²]
    heat_flux_coeff: float = 1.83e-4  # Sutton-Graves constant (approx)


# ── RK4/RK45 integrator ───────────────────────────────────────────────────────
class OrbitalIntegrator:
    """
    Fixed-step RK4 or adaptive-step RK45 integrator.
    Handles gravity (+ J2), thrust, aerodynamic drag, and aeroheating.
    """

    def __init__(
        self,
        planet: Planet,
        thruster: ThrusterConfig = None,
        aero: AeroConfig = None,
        method: str = "RK4",  # "RK4" or "RK45"
    ):
        self.planet = planet
        self.thruster = thruster or ThrusterConfig()
        self.aero = aero or AeroConfig()
        self.method = method

    def _derivatives(
        self,
        state: np.ndarray,
        thrust_vec: np.ndarray,  # [N] in inertial frame
        time: float,
    ) -> np.ndarray:
        """
        State vector: [x, y, z, vx, vy, vz, mass, heat_load]
        Returns d(state)/dt
        """
        x, y, z, vx, vy, vz, mass, heat_load = state
        pos = np.array([x, y, z])
        vel = np.array([vx, vy, vz])
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        altitude = r - self.planet.radius

        # ── Gravity ──────────────────────────────────────────────────────────
        gx, gy, gz = self.planet.gravity_vector_J2((x, y, z))
        accel = np.array([gx, gy, gz])

        # ── Thrust ───────────────────────────────────────────────────────────
        dmass_dt = 0.0
        if self.thruster.enabled and mass > self.thruster.exhaust_velocity * 1e-6:
            thrust_mag = np.linalg.norm(thrust_vec)
            if thrust_mag > 0:
                accel += thrust_vec / mass
                ve = self.thruster.exhaust_velocity
                dmass_dt = -thrust_mag / ve

        # ── Aerodynamic drag ─────────────────────────────────────────────────
        dheat_dt = 0.0
        if self.aero.enabled and self.planet.atmosphere.enabled and altitude < 1e6:
            rho = self.planet.atmosphere.density_at_altitude(max(altitude, 0))
            if rho > 1e-12 and v > 0:
                drag_mag = 0.5 * rho * v**2 * self.aero.Cd * self.aero.reference_area
                drag_dir = -vel / v
                accel += (drag_mag / mass) * drag_dir
                # Stagnation heat flux  q = k * sqrt(rho) * v³
                dheat_dt = self.aero.heat_flux_coeff * math.sqrt(rho) * v**3

        return np.array([vx, vy, vz, accel[0], accel[1], accel[2], dmass_dt, dheat_dt])

    def step_rk4(
        self,
        state_arr: np.ndarray,
        thrust_vec: np.ndarray,
        dt: float,
        time: float,
    ) -> np.ndarray:
        """Single RK4 step."""
        k1 = self._derivatives(state_arr, thrust_vec, time)
        k2 = self._derivatives(state_arr + dt / 2 * k1, thrust_vec, time + dt / 2)
        k3 = self._derivatives(state_arr + dt / 2 * k2, thrust_vec, time + dt / 2)
        k4 = self._derivatives(state_arr + dt * k3, thrust_vec, time + dt)
        return state_arr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def propagate(
        self,
        initial_state: SpacecraftState,
        duration: float,
        dt: float,
        thrust_schedule: Optional[list[tuple[float, float, np.ndarray]]] = None,
    ) -> list[SpacecraftState]:
        """
        Propagate orbit for `duration` seconds with step `dt`.

        thrust_schedule: list of (t_start, t_end, thrust_vector [N])
        Returns list of SpacecraftState snapshots (one per step).
        """
        arr = initial_state.to_array()
        dry = initial_state.dry_mass
        t = initial_state.time
        history = [initial_state]

        steps = int(duration / dt)
        for i in range(steps):
            # Look up thrust at current time
            thrust = np.zeros(3)
            if thrust_schedule:
                for t0, t1, tvec in thrust_schedule:
                    if t0 <= t < t1:
                        thrust = tvec
                        break

            arr = self.step_rk4(arr, thrust, dt, t)
            t += dt

            sc = SpacecraftState.from_array(arr, time=t, dry_mass=dry)

            # Crash detection
            if sc.radius < self.planet.radius:
                history.append(sc)
                break

            history.append(sc)

        return history


# ── Orbital elements ──────────────────────────────────────────────────────────
def state_to_orbital_elements(state: SpacecraftState, mu: float) -> dict:
    """
    Convert Cartesian state to classical orbital elements.
    Returns dict with a, e, i, RAAN, argp, nu (all in SI / degrees).
    """
    r_vec = state.position
    v_vec = state.velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Node vector
    K = np.array([0, 0, 1])
    n_vec = np.cross(K, h_vec)
    n = np.linalg.norm(n_vec)

    # Eccentricity vector
    e_vec = ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)

    # Semi-major axis
    eps = v**2 / 2 - mu / r  # specific orbital energy
    a = -mu / (2 * eps) if abs(eps) > 1e-10 else float("inf")

    # Inclination
    i = math.degrees(math.acos(max(-1, min(1, h_vec[2] / h)))) if h > 0 else 0

    # RAAN
    RAAN = 0.0
    if n > 0:
        RAAN = math.degrees(math.acos(max(-1, min(1, n_vec[0] / n))))
        if n_vec[1] < 0:
            RAAN = 360 - RAAN

    # Argument of periapsis
    argp = 0.0
    if n > 0 and e > 1e-8:
        argp = math.degrees(math.acos(max(-1, min(1, np.dot(n_vec, e_vec) / (n * e)))))
        if e_vec[2] < 0:
            argp = 360 - argp

    # True anomaly
    nu = 0.0
    if e > 1e-8:
        nu = math.degrees(math.acos(max(-1, min(1, np.dot(e_vec, r_vec) / (e * r)))))
        if np.dot(r_vec, v_vec) < 0:
            nu = 360 - nu

    return {
        "semi_major_axis_m": a,
        "eccentricity": e,
        "inclination_deg": i,
        "RAAN_deg": RAAN,
        "arg_periapsis_deg": argp,
        "true_anomaly_deg": nu,
        "specific_energy_J_kg": eps,
        "angular_momentum_m2_s": h,
        "altitude_periapsis_m": a * (1 - e) - (state.radius - state.altitude),
        "altitude_apoapsis_m": a * (1 + e) - (state.radius - state.altitude),
    }
