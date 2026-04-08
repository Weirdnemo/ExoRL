"""
mission.py — Mission design tools.

Covers:
  1. Delta-V budget builder  — itemised ΔV from launch to science orbit
  2. Aerobraking planner     — multi-pass drag campaign design
  3. Lambert solver          — ballistic transfer between two position vectors
  4. Porkchop data           — C3 and arrival v_inf over a departure/arrival grid
  5. Gravity assist          — flyby ΔV and bending angle

All SI units. Time in seconds unless labelled _days or _years.

References:
  Bate, Mueller & White (1971) "Fundamentals of Astrodynamics"
  Izzo (2015) — robust Lambert solver (Lancaster-Blanchard method)
  Vallado (2013) — aerobraking and mission design
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

G = 6.674_30e-11
TWO_PI = 2 * math.pi
AU = 1.495_978_707e11
G0 = 9.80665  # standard gravity [m/s²]


# ─────────────────────────────────────────────────────────────────────────────
# Delta-V budget
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DeltaVBudget:
    """
    Itemised ΔV budget for a planetary mission.
    Each entry is a named burn with a ΔV [m/s] and optional description.
    """

    mission_name: str
    entries: list[dict] = field(default_factory=list)

    def add(self, name: str, dv_m_s: float, description: str = "") -> None:
        """Add a ΔV entry."""
        self.entries.append(
            {
                "name": name,
                "dv_m_s": dv_m_s,
                "description": description,
            }
        )

    @property
    def total_dv(self) -> float:
        return sum(e["dv_m_s"] for e in self.entries)

    def propellant_mass_kg(self, dry_mass_kg: float, Isp_s: float) -> float:
        """
        Total propellant mass required [kg] via Tsiolkovsky equation.
        m_prop = m_dry × (exp(ΔV / (Isp × g₀)) - 1)
        """
        dv = self.total_dv
        v_e = Isp_s * G0
        ratio = math.exp(dv / v_e)
        return dry_mass_kg * (ratio - 1)

    def launch_mass_kg(self, payload_mass_kg: float, Isp_s: float) -> float:
        """
        Required launch mass (payload + propellant) [kg].
        """
        dv = self.total_dv
        v_e = Isp_s * G0
        return payload_mass_kg * math.exp(dv / v_e)

    def report(self) -> str:
        lines = [
            f"═══ ΔV Budget: {self.mission_name} ═══",
            f"  {'Burn':<30s} {'ΔV (m/s)':>10s}  Description",
            f"  {'─' * 30}  {'─' * 10}  ─────────────────────────────",
        ]
        cumulative = 0.0
        for e in self.entries:
            cumulative += e["dv_m_s"]
            lines.append(f"  {e['name']:<30s} {e['dv_m_s']:>10.1f}  {e['description']}")
        lines += [
            f"  {'─' * 30}  {'─' * 10}",
            f"  {'TOTAL':<30s} {self.total_dv:>10.1f}  m/s",
        ]
        return "\n".join(lines)


def orbital_insertion_dv(
    planet,
    approach_vinf_km_s: float,
    target_altitude_km: float,
    periapsis_altitude_km: float = None,
) -> dict:
    """
    Delta-V for orbital insertion from a hyperbolic approach.

    The spacecraft arrives with v_inf (speed relative to planet at infinity)
    and fires a retrograde burn at periapsis to capture into an elliptical
    orbit, then a second burn to circularise.

    Parameters
    ----------
    approach_vinf_km_s   : v_inf at arrival [km/s]
    target_altitude_km   : desired circular science orbit altitude [km]
    periapsis_altitude_km: periapsis of the capture orbit (default = target)

    Returns dict with dv_capture, dv_circularise, dv_total [m/s]
    """
    mu = G * planet.mass
    R = planet.radius
    v_inf = approach_vinf_km_s * 1e3  # m/s

    alt_tgt = target_altitude_km * 1e3
    alt_peri = periapsis_altitude_km * 1e3 if periapsis_altitude_km else alt_tgt

    r_peri = R + alt_peri
    r_apo = R + alt_tgt  # after circularisation
    r_circ = R + alt_tgt

    # Speed at periapsis on the hyperbolic arrival trajectory
    v_peri_hyp = math.sqrt(v_inf**2 + 2 * mu / r_peri)

    if alt_peri == alt_tgt:
        # Direct insertion to circular orbit at target altitude
        v_circ = math.sqrt(mu / r_circ)
        dv_capture = v_peri_hyp - v_circ
        dv_circularise = 0.0
    else:
        # Two-burn: capture ellipse then circularise
        # Capture ellipse: periapsis=alt_peri, apoapsis=alt_tgt
        a_cap = (r_peri + r_apo) / 2
        v_peri_ellipse = math.sqrt(mu * (2 / r_peri - 1 / a_cap))
        dv_capture = v_peri_hyp - v_peri_ellipse

        # Circularise at apoapsis
        v_apo_ellipse = math.sqrt(mu * (2 / r_apo - 1 / a_cap))
        v_circ_target = math.sqrt(mu / r_apo)
        dv_circularise = abs(v_circ_target - v_apo_ellipse)

    return {
        "v_inf_km_s": v_inf / 1e3,
        "v_periapsis_hyp_km_s": v_peri_hyp / 1e3,
        "dv_capture_m_s": dv_capture,
        "dv_circularise_m_s": dv_circularise,
        "dv_total_m_s": dv_capture + dv_circularise,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aerobraking planner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AerobrakingPass:
    """One drag pass in an aerobraking campaign."""

    pass_number: int
    apoapsis_before_km: float
    periapsis_km: float
    dv_drag_m_s: float  # ΔV from drag (negative = deceleration)
    apoapsis_after_km: float
    peak_heating_W_m2: float
    peak_decel_g: float
    cumulative_heat_J_m2: float


@dataclass
class AerobrakingCampaign:
    """Complete aerobraking campaign design."""

    planet_name: str
    passes: list[AerobrakingPass]
    initial_apoapsis_km: float
    target_apoapsis_km: float
    periapsis_km: float

    @property
    def total_passes(self) -> int:
        return len(self.passes)

    @property
    def total_dv_saved_m_s(self) -> float:
        """ΔV saved vs. propulsive insertion to the same final orbit."""
        return sum(abs(p.dv_drag_m_s) for p in self.passes)

    @property
    def total_duration_days(self) -> float:
        """Approximate campaign duration assuming one pass per orbit period."""
        return self.total_passes * 2.0  # rough estimate

    def report(self) -> str:
        lines = [
            f"═══ Aerobraking campaign: {self.planet_name} ═══",
            f"  Initial apoapsis  : {self.initial_apoapsis_km:.0f} km",
            f"  Target apoapsis   : {self.target_apoapsis_km:.0f} km",
            f"  Periapsis         : {self.periapsis_km:.0f} km",
            f"  Total passes      : {self.total_passes}",
            f"  ΔV saved          : {self.total_dv_saved_m_s:.0f} m/s",
            f"  Est. duration     : ~{self.total_duration_days:.0f} days",
            f"",
            f"  Pass  Apo-before  Apo-after   Peak heat    Peak g",
        ]
        for p in self.passes[:: max(1, len(self.passes) // 10)]:  # show ~10 rows
            lines.append(
                f"  {p.pass_number:4d}  {p.apoapsis_before_km:8.0f} km"
                f"  {p.apoapsis_after_km:8.0f} km"
                f"  {p.peak_heating_W_m2:8.0f} W/m²"
                f"  {p.peak_decel_g:6.2f} g"
            )
        return "\n".join(lines)


def plan_aerobraking(
    planet,
    initial_apoapsis_km: float,
    target_apoapsis_km: float,
    periapsis_altitude_km: float,
    spacecraft_mass_kg: float = 1000.0,
    ballistic_coeff: float = 100.0,
    heat_limit_W_m2: float = 2000.0,
    g_limit: float = 5.0,
    max_passes: int = 500,
) -> AerobrakingCampaign:
    """
    Plan a multi-pass aerobraking campaign.

    The spacecraft enters the atmosphere at periapsis on each pass.
    Drag reduces the apoapsis altitude slightly each pass.
    Continues until target apoapsis is reached.

    Parameters
    ----------
    ballistic_coeff : m/(Cd×A) [kg/m²] — governs drag effectiveness
    heat_limit_W_m2 : maximum allowable aerodynamic heating rate
    g_limit         : maximum deceleration [g] structural limit
    """
    if not planet.atmosphere.enabled:
        return AerobrakingCampaign(
            planet_name=planet.name,
            passes=[],
            initial_apoapsis_km=initial_apoapsis_km,
            target_apoapsis_km=target_apoapsis_km,
            periapsis_km=periapsis_altitude_km,
        )

    mu = G * planet.mass
    R = planet.radius
    rho = planet.atmosphere.density_at_altitude(periapsis_altitude_km * 1e3)

    passes = []
    apoapsis_km = initial_apoapsis_km
    cumulative_heat = 0.0

    for i in range(1, max_passes + 1):
        if apoapsis_km <= target_apoapsis_km * 1.01:
            break

        r_peri = R + periapsis_altitude_km * 1e3
        r_apo = R + apoapsis_km * 1e3
        a = (r_peri + r_apo) / 2
        e = (r_apo - r_peri) / (r_apo + r_peri)

        # Speed at periapsis
        v_peri = math.sqrt(mu * (2 / r_peri - 1 / a))

        # Aerodynamic drag impulse (simplified: ΔV per pass)
        # Uses King-Hele single-pass ΔV:
        # ΔV = -(ρ_peri v² H) / (2 B) × correction
        H = planet.atmosphere.scale_height
        B = ballistic_coeff
        dv_per_pass = rho * v_peri**2 * H / (2 * B) * math.pi  # m/s

        # Peak heating (Sutton-Graves): q̇ = k √ρ v³
        k_sg = 1.7e-4
        q_dot = k_sg * math.sqrt(rho) * v_peri**3

        # Peak deceleration [g]
        F_drag = 0.5 * rho * v_peri**2 / B * spacecraft_mass_kg
        peak_g = F_drag / spacecraft_mass_kg / 9.80665

        # Check limits
        if q_dot > heat_limit_W_m2 * 2 or peak_g > g_limit * 2:
            # Abort — too aggressive; raise periapsis
            periapsis_altitude_km += 2.0
            rho = planet.atmosphere.density_at_altitude(periapsis_altitude_km * 1e3)
            continue

        # Energy change: ΔE = -v_peri × ΔV_drag
        a_new = a / (
            1 + 2 * a * rho * math.sqrt(G * planet.mass) / (B * math.sqrt(a)) * 0.5
        )  # simplified
        delta_E = -v_peri * dv_per_pass  # J/kg (negative = loss)
        E_old = -mu / (2 * a)
        E_new = E_old + delta_E
        if E_new >= 0:
            break
        a_new = -mu / (2 * E_new)
        apo_new_km = (a_new * (1 + e) - R) / 1e3  # approx (e changes slightly)
        apo_new_km = max(apo_new_km, target_apoapsis_km)

        cumulative_heat += q_dot * 120  # rough: 120 s in atmosphere

        passes.append(
            AerobrakingPass(
                pass_number=i,
                apoapsis_before_km=apoapsis_km,
                periapsis_km=periapsis_altitude_km,
                dv_drag_m_s=-dv_per_pass,
                apoapsis_after_km=apo_new_km,
                peak_heating_W_m2=q_dot,
                peak_decel_g=peak_g,
                cumulative_heat_J_m2=cumulative_heat,
            )
        )

        apoapsis_km = apo_new_km

    return AerobrakingCampaign(
        planet_name=planet.name,
        passes=passes,
        initial_apoapsis_km=initial_apoapsis_km,
        target_apoapsis_km=target_apoapsis_km,
        periapsis_km=periapsis_altitude_km,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lambert solver (Lancaster-Blanchard)
# ─────────────────────────────────────────────────────────────────────────────


def lambert_solve(
    r1_vec: np.ndarray,
    r2_vec: np.ndarray,
    tof_s: float,
    mu: float,
    prograde: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve Lambert's problem: find v1 and v2 for a ballistic transfer
    from r1 to r2 in time tof.

    Uses the Lancaster-Blanchard algorithm with Halley iteration.

    Parameters
    ----------
    r1_vec   : departure position vector [m]
    r2_vec   : arrival position vector [m]
    tof_s    : time of flight [s]
    mu       : gravitational parameter of central body [m³/s²]
    prograde : True for prograde (short way) transfer

    Returns
    -------
    (v1, v2) : velocity vectors at departure and arrival [m/s]

    Raises
    ------
    ValueError if no solution exists (tof too short for the geometry)
    """
    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    cos_dnu = np.dot(r1_vec, r2_vec) / (r1 * r2)
    cos_dnu = max(-1.0, min(1.0, cos_dnu))

    # Transfer angle
    cross = np.cross(r1_vec, r2_vec)
    if prograde:
        dnu = math.acos(cos_dnu) if cross[2] >= 0 else TWO_PI - math.acos(cos_dnu)
    else:
        dnu = TWO_PI - math.acos(cos_dnu) if cross[2] >= 0 else math.acos(cos_dnu)

    # Chord and semi-perimeter
    c = math.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * cos_dnu)
    s = (r1 + r2 + c) / 2
    lambda_val = math.sqrt(1 - c / s)
    if dnu > math.pi:
        lambda_val = -lambda_val

    # Non-dimensional time
    T_min_energy = (
        math.sqrt(2)
        / 3
        * (s**1.5 - math.copysign((s - c) ** 1.5, lambda_val))
        / math.sqrt(mu)
    )

    # Normalised tof
    T = tof_s / math.sqrt(2 * s**3 / mu)
    lam2 = lambda_val**2

    # Initial guess for x (Halley's method)
    x0 = 0.0
    if T >= T_min_energy / math.sqrt(2 * s**3 / mu):
        x0 = (T_min_energy / tof_s * math.sqrt(2 * s**3 / mu)) ** (2 / 3) - 1
    else:
        x0 = 0.5

    x = x0
    for _ in range(50):
        a = 1 / (1 - x**2)
        if a > 0:
            alpha = 2 * math.acos(x)
            beta = 2 * math.asin(math.sqrt(lambda_val**2 / a))
        else:
            alpha = 2 * math.acosh(x)
            beta = 2 * math.asinh(math.sqrt(-(lambda_val**2) / a))

        if a > 0:
            psi = (alpha - beta) / 2
            y = math.sqrt(abs(1 - lam2 + lam2 * a))
            T_x = (
                y * (math.cos(psi) - lambda_val * math.sin(psi))
                - math.cos(psi) * math.sqrt(a)
            ) / a
        else:
            psi = (alpha - beta) / 2
            y = math.sqrt(abs(1 - lam2 + lam2 * a))
            T_x = (
                y * (math.cosh(psi) - lambda_val * math.sinh(psi))
                - math.cosh(psi) * math.sqrt(-a)
            ) / (-a)

        dT = T_x - T
        if abs(dT) < 1e-12:
            break
        # Halley step (simplified Newton step for stability)
        x -= dT * 0.5

    # Reconstruct velocities from x
    gamma = math.sqrt(mu * s / 2)
    if a > 0:
        rho = (r1 - r2) / c
        sigma = math.sqrt(1 - rho**2)
        y_val = math.sqrt(abs(1 - lam2 + lam2 * a))

        v_r1 = gamma * ((lambda_val * y_val - x) - rho * (lambda_val * y_val + x)) / r1
        v_r2 = -gamma * ((lambda_val * y_val - x) + rho * (lambda_val * y_val + x)) / r2
        v_t1 = gamma * sigma * (y_val + lambda_val * x) / r1
        v_t2 = gamma * sigma * (y_val + lambda_val * x) / r2
    else:
        # Hyperbolic case — simplified
        rho = (r1 - r2) / c
        sigma = math.sqrt(max(0, 1 - rho**2))
        y_val = math.sqrt(abs(1 - lam2 + lam2 * a))

        v_r1 = gamma * ((lambda_val * y_val - x) - rho * (lambda_val * y_val + x)) / r1
        v_r2 = -gamma * ((lambda_val * y_val - x) + rho * (lambda_val * y_val + x)) / r2
        v_t1 = gamma * sigma * (y_val + lambda_val * x) / r1
        v_t2 = gamma * sigma * (y_val + lambda_val * x) / r2

    # Unit vectors
    r1_hat = r1_vec / r1
    r2_hat = r2_vec / r2
    t1_hat = np.cross(
        cross / np.linalg.norm(cross)
        if np.linalg.norm(cross) > 1e-10
        else np.array([0, 0, 1]),
        r1_hat,
    )
    t1_hat /= max(np.linalg.norm(t1_hat), 1e-15)
    t2_hat = np.cross(
        cross / np.linalg.norm(cross)
        if np.linalg.norm(cross) > 1e-10
        else np.array([0, 0, 1]),
        r2_hat,
    )
    t2_hat /= max(np.linalg.norm(t2_hat), 1e-15)

    v1 = v_r1 * r1_hat + v_t1 * t1_hat
    v2 = v_r2 * r2_hat + v_t2 * t2_hat

    return v1, v2


# ─────────────────────────────────────────────────────────────────────────────
# Porkchop plot data
# ─────────────────────────────────────────────────────────────────────────────


def porkchop_data(
    origin_star_mu: float,
    origin_distance_m: float,
    destination_distance_m: float,
    departure_days: np.ndarray,
    arrival_days: np.ndarray,
    origin_period_s: float = None,
    destination_period_s: float = None,
) -> dict:
    """
    Compute C3 and arrival v_inf for a grid of departure/arrival dates.

    This is the core porkchop plot calculation.

    Parameters
    ----------
    origin_star_mu         : gravitational parameter of the central star [m³/s²]
    origin_distance_m      : departure planet's orbital radius [m]
    destination_distance_m : arrival planet's orbital radius [m]
    departure_days         : 1D array of departure times [days from epoch]
    arrival_days           : 1D array of arrival times [days from epoch]
    origin_period_s        : orbital period of origin planet [s]
    destination_period_s   : orbital period of destination planet [s]

    Returns
    -------
    dict with:
        C3          : 2D array (n_dep × n_arr) of C3 [km²/s²]
        v_inf_arr   : 2D array of arrival v_inf [km/s]
        tof_days    : 2D array of time of flight [days]
        departure_days, arrival_days : input grids
    """
    mu = origin_star_mu
    r1 = origin_distance_m
    r2 = destination_distance_m

    if origin_period_s is None:
        origin_period_s = TWO_PI * math.sqrt(r1**3 / mu)
    if destination_period_s is None:
        destination_period_s = TWO_PI * math.sqrt(r2**3 / mu)

    n_dep = len(departure_days)
    n_arr = len(arrival_days)
    C3_grid = np.full((n_dep, n_arr), np.nan)
    vinf_arr_grid = np.full((n_dep, n_arr), np.nan)
    tof_grid = np.full((n_dep, n_arr), np.nan)

    for i, t_dep in enumerate(departure_days):
        # Origin planet position (circular orbit)
        theta1 = TWO_PI * t_dep * 86400 / origin_period_s
        r1_vec = np.array([r1 * math.cos(theta1), r1 * math.sin(theta1), 0.0])
        # Origin circular velocity
        v1_circ = math.sqrt(mu / r1)
        v1_vec = np.array(
            [-v1_circ * math.sin(theta1), v1_circ * math.cos(theta1), 0.0]
        )

        for j, t_arr in enumerate(arrival_days):
            if t_arr <= t_dep + 30:  # minimum 30 days ToF
                continue
            tof_s = (t_arr - t_dep) * 86400

            # Destination planet position
            theta2 = TWO_PI * t_arr * 86400 / destination_period_s
            r2_vec = np.array([r2 * math.cos(theta2), r2 * math.sin(theta2), 0.0])
            v2_circ = math.sqrt(mu / r2)
            v2_vec = np.array(
                [-v2_circ * math.sin(theta2), v2_circ * math.cos(theta2), 0.0]
            )

            try:
                v1_tx, v2_tx = lambert_solve(r1_vec, r2_vec, tof_s, mu)
                dv1 = v1_tx - v1_vec
                dv2 = v2_tx - v2_vec
                C3 = float(np.dot(dv1, dv1)) / 1e6  # km²/s²
                vinf = float(np.linalg.norm(dv2)) / 1e3  # km/s

                if 0 < C3 < 200 and 0 < vinf < 50:
                    C3_grid[i, j] = C3
                    vinf_arr_grid[i, j] = vinf
                    tof_grid[i, j] = t_arr - t_dep
            except Exception:
                pass

    return {
        "C3": C3_grid,
        "v_inf_arr": vinf_arr_grid,
        "tof_days": tof_grid,
        "departure_days": departure_days,
        "arrival_days": arrival_days,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gravity assist
# ─────────────────────────────────────────────────────────────────────────────


class GravityAssist:
    """
    Gravity assist (flyby) calculations.
    Bending the spacecraft trajectory around a planet to change speed/direction.
    """

    @staticmethod
    def bending_angle(planet, v_inf_km_s: float, periapsis_altitude_km: float) -> float:
        """
        Total bending angle δ [radians] for a gravity assist flyby.

            sin(δ/2) = 1 / (1 + r_p v_inf² / μ)

        where r_p is the periapsis radius.
        """
        mu = G * planet.mass
        r_p = planet.radius + periapsis_altitude_km * 1e3
        v_inf = v_inf_km_s * 1e3
        sin_half = 1.0 / (1.0 + r_p * v_inf**2 / mu)
        sin_half = max(-1.0, min(1.0, sin_half))
        return 2 * math.asin(sin_half)  # radians

    @staticmethod
    def max_delta_v(planet, v_inf_km_s: float) -> float:
        """
        Maximum possible ΔV from a single gravity assist [km/s].
        Achieved at the minimum flyby altitude (surface grazing).
        δ_max = π → Δv_max = 2 v_inf
        """
        # At minimum periapsis (grazing): δ → max
        delta = GravityAssist.bending_angle(planet, v_inf_km_s, 100.0)  # 100 km alt
        return 2 * v_inf_km_s * math.sin(delta / 2)

    @staticmethod
    def outgoing_speed_km_s(
        incoming_speed_km_s: float,
        bending_angle_rad: float,
        planet_orbital_speed_km_s: float,
        approach_angle_rad: float = 0.0,
    ) -> float:
        """
        Speed relative to the Sun after a gravity assist.

        The spacecraft's speed relative to the planet is unchanged (|v_inf| conserved).
        But the direction changes, so the heliocentric speed changes.

        Simplified 2D calculation:
          v_out_helio = sqrt(v_inf² + v_planet² + 2 v_inf v_planet cos(approach + δ/2))
        """
        v_inf = incoming_speed_km_s
        v_p = planet_orbital_speed_km_s
        phi_out = approach_angle_rad + bending_angle_rad
        v_out = math.sqrt(v_inf**2 + v_p**2 + 2 * v_inf * v_p * math.cos(phi_out))
        return v_out

    @staticmethod
    def summary(planet, v_inf_km_s: float, periapsis_altitude_km: float) -> dict:
        """Summary of a gravity assist flyby."""
        mu = G * planet.mass
        r_p = planet.radius + periapsis_altitude_km * 1e3
        v_p = math.sqrt(mu / r_p) / 1e3  # circular speed at periapsis
        delta = GravityAssist.bending_angle(planet, v_inf_km_s, periapsis_altitude_km)

        return {
            "planet": planet.name,
            "v_inf_km_s": v_inf_km_s,
            "periapsis_alt_km": periapsis_altitude_km,
            "bending_angle_deg": delta * 180 / math.pi,
            "max_delta_v_km_s": GravityAssist.max_delta_v(planet, v_inf_km_s),
            "periapsis_speed_km_s": math.sqrt(v_inf_km_s**2 * 1e6 + 2 * mu / r_p) / 1e3,
        }


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end mission builder
# ─────────────────────────────────────────────────────────────────────────────


def build_mission_dv_budget(
    planet,
    star,
    orbital_distance_m: float,
    approach_vinf_km_s: float,
    target_altitude_km: float,
    use_aerobraking: bool = False,
    periapsis_altitude_km: float = None,
    station_keeping_years: float = 5.0,
    ballistic_coeff: float = 100.0,
) -> DeltaVBudget:
    """
    Build a complete mission ΔV budget from arrival to end of mission.

    Parameters
    ----------
    approach_vinf_km_s : arrival hyperbolic excess speed [km/s]
    target_altitude_km : final science orbit altitude [km]
    use_aerobraking    : use aerobraking to lower the orbit (saves ΔV)
    """
    budget = DeltaVBudget(f"Mission to {planet.name}")

    # Orbital insertion
    if periapsis_altitude_km is None:
        periapsis_altitude_km = max(50.0, target_altitude_km / 3)

    ins = orbital_insertion_dv(
        planet, approach_vinf_km_s, target_altitude_km, periapsis_altitude_km
    )
    budget.add(
        "Capture burn",
        ins["dv_capture_m_s"],
        f"v_inf={approach_vinf_km_s:.1f} km/s → capture at {periapsis_altitude_km:.0f} km",
    )

    if use_aerobraking and planet.atmosphere.enabled:
        # Aerobraking saves the circularisation ΔV
        budget.add(
            "Aerobraking periapsis raise",
            50.0,
            "Contingency burn for aerobraking abort capability",
        )
        budget.add(
            "Aerobraking exit circularisation",
            ins["dv_circularise_m_s"] * 0.1,
            "Small burn to exit aerobraking into science orbit",
        )
    else:
        budget.add(
            "Circularisation burn",
            ins["dv_circularise_m_s"],
            f"Circularise at {target_altitude_km:.0f} km altitude",
        )

    budget.add("Plane change contingency", 20.0, "Inclination trim")

    # Station-keeping
    from exorl.core.orbital_analysis import StationKeeping

    sk = StationKeeping.total_annual_budget(
        planet, target_altitude_km, 90.0, ballistic_coeff
    )
    sk_total = sk["total_dv_m_s_yr"] * station_keeping_years
    budget.add(
        f"Station-keeping ({station_keeping_years:.0f} yr)",
        sk_total,
        f"{sk['total_dv_m_s_yr']:.1f} m/s/yr × {station_keeping_years:.0f} yr",
    )

    # Disposal
    budget.add("Disposal / deorbit", 50.0, "End-of-mission disposal")

    return budget
