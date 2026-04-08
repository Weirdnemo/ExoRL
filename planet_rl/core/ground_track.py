"""
ground_track.py — Ground track propagation and surface coverage analysis.

Answers the science operations questions:
  - What is the ground track of this orbit over 1 day?
  - What fraction of the surface is observed within 7 days?
  - How often does the spacecraft pass over site X?
  - Are there persistent observation gaps?
  - How does swath width affect coverage time?

Uses a rotating-planet model — the surface rotates under the orbit.
Ignores J2 precession for short simulations (< a few days); includes
it for long-duration coverage maps.

All SI units. Latitude/longitude in degrees.

References:
  Vallado (2013) Ch. 11 — ground track geometry
  Larson & Wertz (1999) — coverage analysis methods
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
G      = 6.674_30e-11
TWO_PI = 2 * math.pi
DEG    = math.pi / 180   # deg → rad
RAD    = 180 / math.pi   # rad → deg


# ─────────────────────────────────────────────────────────────────────────────
# Core ground track propagator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroundTrackPoint:
    """One point on the ground track."""
    time_s:    float
    lat_deg:   float    # geodetic latitude [-90, 90]
    lon_deg:   float    # longitude [-180, 180]
    altitude_m: float


def propagate_ground_track(planet,
                            altitude_m: float,
                            inclination_deg: float,
                            duration_s: float,
                            dt_s: float = 60.0,
                            raan_deg: float = 0.0,
                            arg_perigee_deg: float = 0.0,
                            true_anomaly_deg: float = 0.0,
                            eccentricity: float = 0.0,
                            include_j2: bool = True
                            ) -> list[GroundTrackPoint]:
    """
    Propagate the sub-satellite ground track.

    For a circular orbit (e=0, default), the spacecraft position in the
    orbital plane rotates at mean motion n. The Earth (planet) rotates at
    Ω_rot = 2π / T_rotation underneath.

    With J2 included, the RAAN drifts at dΩ/dt (nodal precession) which
    shifts the ground track westward each orbit for prograde orbits.

    Parameters
    ----------
    planet            : Planet object
    altitude_m        : orbit altitude [m] (mean for elliptical orbits)
    inclination_deg   : orbital inclination [°]
    duration_s        : propagation duration [s]
    dt_s              : output timestep [s] (default 60 s)
    raan_deg          : initial RAAN [°]
    include_j2        : apply J2 nodal precession (default True)

    Returns
    -------
    list of GroundTrackPoint
    """
    mu  = G * planet.mass
    R_p = planet.radius
    a   = R_p + altitude_m
    n   = math.sqrt(mu / a**3)   # mean motion [rad/s]
    T   = TWO_PI / n             # orbital period [s]

    inc = inclination_deg * DEG
    raan = raan_deg * DEG

    # Planet rotation rate
    Omega_rot = TWO_PI / planet.rotation_period   # rad/s

    # J2 nodal precession rate
    dOmega_dt = 0.0
    if include_j2:
        J2 = 0.0
        if hasattr(planet, "derived_J2"):
            J2 = planet.derived_J2()
        elif planet.oblateness.enabled:
            J2 = planet.oblateness.J2
        if J2 > 0:
            p = a * (1 - eccentricity**2)
            dOmega_dt = -(3/2) * n * J2 * (R_p / p)**2 * math.cos(inc)

    points = []
    steps  = int(duration_s / dt_s) + 1

    for i in range(steps):
        t = i * dt_s

        # True anomaly (circular: θ = n×t; elliptical: approximate)
        if eccentricity < 1e-6:
            theta = (true_anomaly_deg * DEG + n * t) % TWO_PI
        else:
            # Eccentric anomaly via Newton's method
            M = (true_anomaly_deg * DEG + n * t) % TWO_PI
            E = M
            for _ in range(50):
                E_new = M + eccentricity * math.sin(E)
                if abs(E_new - E) < 1e-10:
                    break
                E = E_new
            sin_nu = math.sqrt(1 - eccentricity**2) * math.sin(E) / (1 - eccentricity * math.cos(E))
            cos_nu = (math.cos(E) - eccentricity) / (1 - eccentricity * math.cos(E))
            theta  = math.atan2(sin_nu, cos_nu) % TWO_PI

        # Current RAAN (drifting due to J2)
        Omega_t = (raan + dOmega_dt * t) % TWO_PI

        # Argument of latitude u = omega + theta
        u = (arg_perigee_deg * DEG + theta) % TWO_PI

        # Sub-satellite point in ECI frame
        # Using the orbital mechanics spherical trig formula:
        # sin(lat) = sin(i) sin(u)
        sin_lat = math.sin(inc) * math.sin(u)
        sin_lat = max(-1.0, min(1.0, sin_lat))  # clamp for floating point
        lat_rad = math.asin(sin_lat)

        # Longitude in the rotating planet frame
        # ECI longitude of sub-satellite point:
        delta_lambda = math.atan2(
            math.cos(inc) * math.sin(u),
            math.cos(u)
        )
        lon_eci_rad = Omega_t + delta_lambda
        # Subtract planet rotation (Greenwich angle advances with time)
        lon_planet_rad = (lon_eci_rad - Omega_rot * t) % TWO_PI
        # Normalise to [-π, π]
        if lon_planet_rad > math.pi:
            lon_planet_rad -= TWO_PI

        points.append(GroundTrackPoint(
            time_s    = t,
            lat_deg   = lat_rad * RAD,
            lon_deg   = lon_planet_rad * RAD,
            altitude_m = a - R_p,   # circular orbit approximation
        ))

    return points


# ─────────────────────────────────────────────────────────────────────────────
# Coverage analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoverageMap:
    """
    2D grid [lat, lon] counting number of times each surface cell was
    observed by a spacecraft pass within swath_width_km.

    Grid: lat from -90 to +90, lon from -180 to +180.
    Cell size: lat_res_deg × lon_res_deg.
    """
    lat_res_deg: float
    lon_res_deg: float
    grid: np.ndarray         # shape (n_lat, n_lon) — integer count
    duration_s: float
    swath_width_km: float

    @property
    def n_lat(self) -> int:
        return self.grid.shape[0]

    @property
    def n_lon(self) -> int:
        return self.grid.shape[1]

    def lat_edges(self) -> np.ndarray:
        return np.linspace(-90, 90, self.n_lat + 1)

    def lon_edges(self) -> np.ndarray:
        return np.linspace(-180, 180, self.n_lon + 1)

    def lat_centres(self) -> np.ndarray:
        edges = self.lat_edges()
        return (edges[:-1] + edges[1:]) / 2

    def lon_centres(self) -> np.ndarray:
        edges = self.lon_edges()
        return (edges[:-1] + edges[1:]) / 2

    def coverage_fraction(self) -> float:
        """Fraction of surface cells with at least one observation."""
        # Weight by cos(lat) to account for spherical geometry
        lats = self.lat_centres()
        weights = np.abs(np.cos(np.radians(lats)))[:, np.newaxis]
        weighted_observed = (self.grid > 0).astype(float) * weights
        total_weight = weights.sum() * self.n_lon
        return float(weighted_observed.sum() / total_weight)

    def max_gap_latitude_deg(self) -> float:
        """
        The latitude of the widest unobserved band (observation gap).
        Returns the latitude with the lowest average observation count.
        """
        lats = self.lat_centres()
        mean_per_lat = self.grid.mean(axis=1)
        idx = int(np.argmin(mean_per_lat))
        return float(lats[idx])

    def revisit_time_days(self, target_lat: float, target_lon: float,
                           pass_times_s: list[float]) -> float:
        """
        Mean time between passes over a specific lat/lon [days].
        Requires the list of pass times at that point (from find_passes).
        """
        if len(pass_times_s) < 2:
            return float("inf")
        gaps = [pass_times_s[i+1] - pass_times_s[i]
                for i in range(len(pass_times_s) - 1)]
        return float(np.mean(gaps)) / 86400

    def summary(self) -> str:
        cov  = self.coverage_fraction() * 100
        days = self.duration_s / 86400
        return (
            f"Coverage map  ({days:.1f} days, swath={self.swath_width_km:.0f} km)\n"
            f"  Grid         : {self.n_lat}×{self.n_lon} cells "
            f"({self.lat_res_deg}°×{self.lon_res_deg}°)\n"
            f"  Coverage     : {cov:.1f}%  of surface area observed\n"
            f"  Max gap lat  : {self.max_gap_latitude_deg():.1f}°\n"
            f"  Max count    : {int(self.grid.max())}\n"
            f"  Mean count   : {self.grid.mean():.2f}\n"
        )


def compute_coverage_map(planet,
                          ground_track: list[GroundTrackPoint],
                          swath_width_km: float,
                          lat_res_deg: float = 2.0,
                          lon_res_deg: float = 2.0) -> CoverageMap:
    """
    Build a coverage map from a pre-computed ground track.

    For each ground track point, marks all cells within swath_width_km/2
    of the sub-satellite point as observed.

    Parameters
    ----------
    swath_width_km : total swath width (one-sided: width/2 on each side) [km]
    """
    n_lat = int(180 / lat_res_deg)
    n_lon = int(360 / lon_res_deg)
    grid  = np.zeros((n_lat, n_lon), dtype=np.int32)

    half_swath_m = swath_width_km * 1e3 / 2

    # Convert half-swath to angular radius [rad]
    half_swath_rad = half_swath_m / planet.radius

    for pt in ground_track:
        lat_rad = pt.lat_deg * DEG
        lon_rad = pt.lon_deg * DEG

        # Which lat cells are within range?
        lat_min = (pt.lat_deg - half_swath_rad * RAD)
        lat_max = (pt.lat_deg + half_swath_rad * RAD)
        lat_min = max(-90, lat_min)
        lat_max = min( 90, lat_max)

        # Convert to grid indices
        i_min = int((lat_min + 90) / lat_res_deg)
        i_max = int((lat_max + 90) / lat_res_deg) + 1
        i_min = max(0, min(n_lat - 1, i_min))
        i_max = max(0, min(n_lat,     i_max))

        for i in range(i_min, i_max):
            cell_lat = (-90 + (i + 0.5) * lat_res_deg) * DEG
            # At this latitude, compute allowable longitude range
            dlat = abs(cell_lat - lat_rad)
            if dlat > half_swath_rad:
                continue
            dlon_rad = math.acos(max(-1.0, min(1.0,
                (math.cos(half_swath_rad) - math.cos(dlat)) /
                (math.sin(dlat + 1e-12) * math.sin(lat_rad + 1e-12) + 1e-12)
            ))) if dlat > 1e-6 else half_swath_rad / max(abs(math.cos(lat_rad)), 0.01)

            lon_min_deg = pt.lon_deg - dlon_rad * RAD
            lon_max_deg = pt.lon_deg + dlon_rad * RAD

            j_min = int((lon_min_deg + 180) / lon_res_deg)
            j_max = int((lon_max_deg + 180) / lon_res_deg) + 1
            # Wrap around
            for j_raw in range(j_min, j_max + 1):
                j = j_raw % n_lon
                grid[i, j] = min(grid[i, j] + 1, 32767)

    return CoverageMap(
        lat_res_deg   = lat_res_deg,
        lon_res_deg   = lon_res_deg,
        grid          = grid,
        duration_s    = ground_track[-1].time_s if ground_track else 0.0,
        swath_width_km = swath_width_km,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pass finder  — when does the spacecraft visit a specific point?
# ─────────────────────────────────────────────────────────────────────────────

def find_passes(ground_track: list[GroundTrackPoint],
                target_lat_deg: float,
                target_lon_deg: float,
                radius_km: float = 500.0) -> list[dict]:
    """
    Find all passes within radius_km of a surface target point.

    Returns a list of dicts, each describing one pass:
        time_s      : time of closest approach [s]
        distance_km : closest approach distance [km]
        lat_deg, lon_deg : sub-satellite point at closest approach
    """
    passes = []
    in_range = False
    current_pass = []

    for pt in ground_track:
        dist = _great_circle_distance_km(
            target_lat_deg, target_lon_deg,
            pt.lat_deg, pt.lon_deg,
            planet_radius_m=6.371e6   # use Earth as reference for angle calc
        )
        if dist <= radius_km:
            in_range = True
            current_pass.append((pt, dist))
        else:
            if in_range and current_pass:
                # Find closest point in the pass
                best = min(current_pass, key=lambda x: x[1])
                passes.append({
                    "time_s":      best[0].time_s,
                    "distance_km": best[1],
                    "lat_deg":     best[0].lat_deg,
                    "lon_deg":     best[0].lon_deg,
                })
                current_pass = []
            in_range = False

    if in_range and current_pass:
        best = min(current_pass, key=lambda x: x[1])
        passes.append({
            "time_s":      best[0].time_s,
            "distance_km": best[1],
            "lat_deg":     best[0].lat_deg,
            "lon_deg":     best[0].lon_deg,
        })
    return passes


def _great_circle_distance_km(lat1: float, lon1: float,
                                lat2: float, lon2: float,
                                planet_radius_m: float = 6.371e6) -> float:
    """Haversine great-circle distance [km]."""
    phi1, phi2 = lat1 * DEG, lat2 * DEG
    dphi = (lat2 - lat1) * DEG
    dlam = (lon2 - lon1) * DEG
    a = (math.sin(dphi/2)**2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return planet_radius_m * c / 1e3


# ─────────────────────────────────────────────────────────────────────────────
# High-level convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def coverage_analysis(planet,
                       altitude_km: float,
                       inclination_deg: float,
                       swath_width_km: float,
                       duration_days: float,
                       dt_s: float = 120.0,
                       lat_res_deg: float = 2.0,
                       lon_res_deg: float = 2.0) -> CoverageMap:
    """
    Run a complete coverage analysis and return a CoverageMap.

    Parameters
    ----------
    swath_width_km : total swath (e.g. 100 km covers ±50 km either side)
    duration_days  : simulation length
    dt_s           : propagation timestep (smaller = more accurate, slower)
    """
    track = propagate_ground_track(
        planet,
        altitude_m      = altitude_km * 1e3,
        inclination_deg = inclination_deg,
        duration_s      = duration_days * 86400,
        dt_s            = dt_s,
        include_j2      = True,
    )
    return compute_coverage_map(
        planet, track, swath_width_km,
        lat_res_deg=lat_res_deg, lon_res_deg=lon_res_deg
    )


def time_to_full_coverage_days(planet,
                                 altitude_km: float,
                                 inclination_deg: float,
                                 swath_width_km: float,
                                 target_coverage: float = 0.95,
                                 max_days: int = 60,
                                 dt_s: float = 120.0) -> Optional[float]:
    """
    Find how many days until target_coverage (default 95%) of the surface
    has been observed at least once.

    Returns None if coverage not reached within max_days.
    """
    # Build the full track at once and check incrementally
    track = propagate_ground_track(
        planet,
        altitude_m      = altitude_km * 1e3,
        inclination_deg = inclination_deg,
        duration_s      = max_days * 86400,
        dt_s            = dt_s,
        include_j2      = True,
    )

    # Check coverage at each day boundary
    day_idx = int(86400 / dt_s)
    for d in range(1, max_days + 1):
        subset = track[:d * day_idx]
        if not subset:
            continue
        cov = compute_coverage_map(planet, subset, swath_width_km,
                                    lat_res_deg=2.0, lon_res_deg=2.0)
        if cov.coverage_fraction() >= target_coverage:
            return float(d)
    return None


def mean_revisit_time_days(planet,
                             altitude_km: float,
                             inclination_deg: float,
                             swath_width_km: float,
                             duration_days: float = 30.0,
                             n_sample_sites: int = 20) -> float:
    """
    Estimate mean revisit time [days] by sampling random surface sites.

    Averages across n_sample_sites evenly distributed latitude points.
    """
    import random
    track = propagate_ground_track(
        planet,
        altitude_m      = altitude_km * 1e3,
        inclination_deg = inclination_deg,
        duration_s      = duration_days * 86400,
        dt_s            = 120.0,
        include_j2      = True,
    )

    revisit_times = []
    random.seed(42)
    for _ in range(n_sample_sites):
        lat = random.uniform(-70, 70)
        lon = random.uniform(-180, 180)
        passes = find_passes(track, lat, lon, radius_km=swath_width_km / 2)
        if len(passes) >= 2:
            intervals = [
                passes[i+1]["time_s"] - passes[i]["time_s"]
                for i in range(len(passes) - 1)
            ]
            revisit_times.append(sum(intervals) / len(intervals) / 86400)

    return float(sum(revisit_times) / len(revisit_times)) if revisit_times else float("inf")
