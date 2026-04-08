"""
soi.py — Sphere of influence transitions for patched-conic interplanetary trajectories.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

G      = 6.674_30e-11
M_SUN  = 1.989e30
MU_SUN = G * M_SUN
AU     = 1.495_978_707e11

def laplace_soi_radius(planet_mass_kg, orbital_radius_m, star_mass_kg=M_SUN):
    """r_SOI = a*(m_planet/m_star)^(2/5). Earth~924Mm, Mars~577Mm."""
    return orbital_radius_m * (planet_mass_kg / star_mass_kg) ** 0.4

def hill_sphere_radius(planet_mass_kg, orbital_radius_m, star_mass_kg=M_SUN, eccentricity=0.0):
    """r_Hill = a(1-e)*(m_planet/(3*m_star))^(1/3)."""
    return (orbital_radius_m*(1-eccentricity)
            * (planet_mass_kg/(3.0*star_mass_kg))**(1.0/3.0))


class SphereOfInfluence:
    """SOI geometry and frame transforms for one planet."""

    def __init__(self, planet_mass_kg, orbital_radius_m, planet_name="Planet",
                 star_mass_kg=M_SUN):
        self.planet_mass    = planet_mass_kg
        self.orbital_radius = orbital_radius_m
        self.planet_name    = planet_name
        self.mu_planet      = G * planet_mass_kg
        self.r_laplace      = laplace_soi_radius(planet_mass_kg, orbital_radius_m, star_mass_kg)
        self.r_hill         = hill_sphere_radius(planet_mass_kg, orbital_radius_m, star_mass_kg)

    @classmethod
    def from_planet(cls, planet, orbital_distance_m, star_mass_kg=M_SUN):
        return cls(planet.mass, orbital_distance_m, planet.name, star_mass_kg)

    def is_inside(self, sc_helio_pos, planet_helio_pos):
        return float(np.linalg.norm(np.asarray(sc_helio_pos) - np.asarray(planet_helio_pos))) < self.r_laplace

    def to_planet_frame(self, sc_helio_pos, sc_helio_vel, planet_helio_pos, planet_helio_vel):
        """Heliocentric → planet-centred frame."""
        r = np.asarray(sc_helio_pos) - np.asarray(planet_helio_pos)
        v = np.asarray(sc_helio_vel) - np.asarray(planet_helio_vel)
        return r, v

    def to_helio_frame(self, sc_planet_pos, sc_planet_vel, planet_helio_pos, planet_helio_vel):
        """Planet-centred → heliocentric frame."""
        return (np.asarray(sc_planet_pos) + np.asarray(planet_helio_pos),
                np.asarray(sc_planet_vel) + np.asarray(planet_helio_vel))

    def arrival_vinf(self, sc_helio_vel, planet_helio_vel):
        return float(np.linalg.norm(np.asarray(sc_helio_vel) - np.asarray(planet_helio_vel)))

    def arrival_vinf_vec(self, sc_helio_vel, planet_helio_vel):
        return np.asarray(sc_helio_vel) - np.asarray(planet_helio_vel)

    def report(self):
        return (f"SOI — {self.planet_name}\n"
                f"  Laplace: {self.r_laplace/1e6:.0f} Mm ({self.r_laplace/AU:.4f} AU)\n"
                f"  Hill:    {self.r_hill/1e6:.0f} Mm ({self.r_hill/AU:.4f} AU)\n")


@dataclass
class HyperbolicOrbit:
    """Geometry of a hyperbolic approach/departure trajectory."""
    v_inf_m_s:       float
    periapsis_alt_m: float
    planet_radius_m: float
    mu_planet:       float

    @property
    def periapsis_radius_m(self): return self.planet_radius_m + self.periapsis_alt_m

    @property
    def semi_major_axis_m(self): return -self.mu_planet / self.v_inf_m_s**2

    @property
    def eccentricity(self):
        return 1.0 + self.periapsis_radius_m / abs(self.semi_major_axis_m)

    @property
    def speed_at_periapsis_m_s(self):
        return math.sqrt(self.v_inf_m_s**2 + 2.0*self.mu_planet/self.periapsis_radius_m)

    @property
    def turn_angle_rad(self): return 2.0*math.asin(1.0/self.eccentricity)

    @property
    def turn_angle_deg(self): return math.degrees(self.turn_angle_rad)

    def soi_entry_true_anomaly(self, soi_radius_m):
        a = abs(self.semi_major_axis_m); e = self.eccentricity
        p = a*(e**2-1)
        cos_nu = max(-1.0, min(1.0, (p/soi_radius_m - 1.0)/e))
        return -math.acos(cos_nu)

    def time_to_periapsis_from_soi(self, soi_radius_m):
        nu = self.soi_entry_true_anomaly(soi_radius_m)
        e = self.eccentricity; a = abs(self.semi_major_axis_m)
        cos_nu = math.cos(nu)
        F = math.acosh(max(1.0, (e+cos_nu)/(1+e*cos_nu)))
        n = math.sqrt(self.mu_planet/a**3)
        return (e*math.sinh(F)-F)/n

    def report(self):
        return (f"  v∞            : {self.v_inf_m_s/1e3:.3f} km/s\n"
                f"  Periapsis alt : {self.periapsis_alt_m/1e3:.0f} km\n"
                f"  Peri speed    : {self.speed_at_periapsis_m_s/1e3:.3f} km/s\n"
                f"  Eccentricity  : {self.eccentricity:.4f}\n"
                f"  Turn angle    : {self.turn_angle_deg:.2f}°\n")


@dataclass
class HyperbolicDeparture:
    """Departure hyperbola design from a circular parking orbit."""
    v_inf_m_s:       float
    parking_alt_m:   float
    planet_radius_m: float
    mu_planet:       float

    @property
    def parking_radius_m(self): return self.planet_radius_m + self.parking_alt_m

    @property
    def v_circular_parking(self):
        return math.sqrt(self.mu_planet/self.parking_radius_m)

    @property
    def v_periapsis_hyperbola(self):
        return math.sqrt(self.v_inf_m_s**2 + 2.0*self.mu_planet/self.parking_radius_m)

    @property
    def delta_v_m_s(self): return self.v_periapsis_hyperbola - self.v_circular_parking

    @property
    def c3_km2_s2(self): return (self.v_inf_m_s/1e3)**2

    def report(self):
        return (f"═══ Departure ═══\n"
                f"  Parking alt   : {self.parking_alt_m/1e3:.0f} km\n"
                f"  v∞            : {self.v_inf_m_s/1e3:.3f} km/s\n"
                f"  C3            : {self.c3_km2_s2:.2f} km²/s²\n"
                f"  Departure ΔV  : {self.delta_v_m_s:.0f} m/s\n")


@dataclass
class HyperbolicArrival:
    """Capture manoeuvre design at arrival planet."""
    v_inf_m_s:       float
    periapsis_alt_m: float
    planet_radius_m: float
    mu_planet:       float
    target_alt_m:    float = None

    @property
    def periapsis_radius_m(self): return self.planet_radius_m + self.periapsis_alt_m

    @property
    def v_periapsis_hyperbola(self):
        return math.sqrt(self.v_inf_m_s**2 + 2.0*self.mu_planet/self.periapsis_radius_m)

    @property
    def dv_capture_m_s(self):
        if self.target_alt_m is not None:
            r_apo = self.planet_radius_m + self.target_alt_m
            a_cap = (self.periapsis_radius_m + r_apo)/2.0
            v_ell = math.sqrt(self.mu_planet*(2.0/self.periapsis_radius_m - 1.0/a_cap))
        else:
            v_ell = math.sqrt(self.mu_planet/self.periapsis_radius_m)
        return self.v_periapsis_hyperbola - v_ell

    @property
    def dv_circularise_m_s(self):
        if self.target_alt_m is None: return 0.0
        r_apo = self.planet_radius_m + self.target_alt_m
        a_cap = (self.periapsis_radius_m + r_apo)/2.0
        v_apo = math.sqrt(self.mu_planet*(2.0/r_apo - 1.0/a_cap))
        v_circ= math.sqrt(self.mu_planet/r_apo)
        return abs(v_circ - v_apo)

    @property
    def dv_total_m_s(self): return self.dv_capture_m_s + self.dv_circularise_m_s

    def report(self):
        tgt = f"{self.target_alt_m/1e3:.0f} km" if self.target_alt_m else "periapsis"
        return (f"═══ Arrival capture ═══\n"
                f"  Arrival v∞    : {self.v_inf_m_s/1e3:.3f} km/s\n"
                f"  Periapsis alt : {self.periapsis_alt_m/1e3:.0f} km\n"
                f"  Target orbit  : {tgt}\n"
                f"  Capture ΔV    : {self.dv_capture_m_s:.0f} m/s\n"
                f"  Circularise ΔV: {self.dv_circularise_m_s:.0f} m/s\n"
                f"  Total ΔV      : {self.dv_total_m_s:.0f} m/s\n")


def gravity_assist_turn(v_inf_m_s, mu_planet, periapsis_alt_m, planet_radius_m):
    """Turn angle δ [rad] for gravity assist. sin(δ/2) = 1/(1 + r_p*v∞²/μ)."""
    r_p = planet_radius_m + periapsis_alt_m
    return 2.0*math.asin(max(-1,min(1, 1.0/(1.0 + r_p*v_inf_m_s**2/mu_planet))))


@dataclass
class SOIEvent:
    """Record of an SOI crossing event."""
    time_s:           float
    planet_name:      str
    event_type:       str   # "entry" or "exit"
    sc_helio_pos:     np.ndarray
    sc_helio_vel:     np.ndarray
    planet_helio_pos: np.ndarray
    planet_helio_vel: np.ndarray
    v_inf_m_s:        float

    @property
    def distance_km(self):
        return float(np.linalg.norm(self.sc_helio_pos - self.planet_helio_pos))/1e3


class SOITransitionDetector:
    """Detects SOI boundary crossings during trajectory propagation."""

    def __init__(self, sois):
        self.sois        = sois
        self._last_inside = {s.planet_name: None for s in sois}

    def check(self, time_s, sc_helio_pos, sc_helio_vel,
               planet_positions, planet_velocities):
        events = []
        for soi in self.sois:
            name = soi.planet_name
            if name not in planet_positions: continue
            p_pos = planet_positions[name]
            p_vel = planet_velocities.get(name, np.zeros(3))
            inside_now  = soi.is_inside(sc_helio_pos, p_pos)
            inside_last = self._last_inside[name]
            if inside_last is not None and inside_now != inside_last:
                events.append(SOIEvent(
                    time_s=time_s,
                    planet_name=name,
                    event_type="entry" if inside_now else "exit",
                    sc_helio_pos=np.asarray(sc_helio_pos).copy(),
                    sc_helio_vel=np.asarray(sc_helio_vel).copy(),
                    planet_helio_pos=np.asarray(p_pos).copy(),
                    planet_helio_vel=np.asarray(p_vel).copy(),
                    v_inf_m_s=soi.arrival_vinf(sc_helio_vel, p_vel),
                ))
            self._last_inside[name] = inside_now
        return events


def patched_conic_budget(departure_planet_mass, departure_planet_radius,
                          departure_parking_alt, arrival_planet_mass,
                          arrival_planet_radius, arrival_periapsis_alt,
                          arrival_target_alt, vinf_departure_m_s,
                          vinf_arrival_m_s, dep_name="Departure", arr_name="Arrival"):
    """Complete ΔV budget for an interplanetary transfer."""
    dep = HyperbolicDeparture(vinf_departure_m_s, departure_parking_alt,
                               departure_planet_radius, G*departure_planet_mass)
    arr = HyperbolicArrival(vinf_arrival_m_s, arrival_periapsis_alt,
                             arrival_planet_radius, G*arrival_planet_mass,
                             target_alt_m=arrival_target_alt)
    return {
        "departure_planet":    dep_name,
        "arrival_planet":      arr_name,
        "dv_departure_m_s":    dep.delta_v_m_s,
        "dv_capture_m_s":      arr.dv_capture_m_s,
        "dv_circularise_m_s":  arr.dv_circularise_m_s,
        "dv_total_m_s":        dep.delta_v_m_s + arr.dv_total_m_s,
        "c3_km2_s2":           dep.c3_km2_s2,
        "vinf_departure_km_s": vinf_departure_m_s/1e3,
        "vinf_arrival_km_s":   vinf_arrival_m_s/1e3,
        "departure_obj":       dep,
        "arrival_obj":         arr,
    }
