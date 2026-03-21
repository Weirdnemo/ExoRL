"""
launch_window.py — Launch window analysis for interplanetary missions.
Porkchop plot data, optimal window finder, and RL decision space.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from core.heliocentric import LambertSolver, planet_state, MU_SUN, AU

G = 6.674_30e-11


def synodic_period_days(period1_days, period2_days):
    """T_syn = 1/|1/T1 - 1/T2|. Earth-Mars: 780 days."""
    if abs(period1_days - period2_days) < 1e-6: return float("inf")
    return 1.0 / abs(1.0/period1_days - 1.0/period2_days)


def orbital_period_days(orbital_radius_m, mu_star=MU_SUN):
    """Orbital period [days] for circular orbit."""
    return 2*math.pi*math.sqrt(orbital_radius_m**3/mu_star)/86400


def compute_transfer(dep_radius_m, arr_radius_m, dep_day, arr_day,
                     mu_star=MU_SUN, dep_phase=0.0, arr_phase=0.0, prograde=True):
    """Single Lambert transfer: returns (c3, vinf_arr, tof_days, valid)."""
    tof_days = arr_day - dep_day
    if tof_days <= 0: return float("nan"), float("nan"), tof_days, False
    tof_s = tof_days * 86400
    r1v, v1p = planet_state(dep_radius_m, dep_day*86400, mu_star, dep_phase)
    r2v, v2p = planet_state(arr_radius_m, arr_day*86400, mu_star, arr_phase)
    solver = LambertSolver(mu_star)
    v1s, v2s = solver.solve(r1v, r2v, tof_s, prograde=prograde)
    if v1s is None: return float("nan"), float("nan"), tof_days, False
    vd = float(np.linalg.norm(v1s - v1p))
    va = float(np.linalg.norm(v2s - v2p))
    if math.isnan(vd) or math.isnan(va) or vd>60000 or va>60000:
        return float("nan"), float("nan"), tof_days, False
    return (vd/1e3)**2, va/1e3, tof_days, True


@dataclass
class LaunchWindow:
    """A specific launch opportunity."""
    departure_day:    float
    arrival_day:      float
    tof_days:         float
    c3_km2_s2:        float
    vinf_dep_km_s:    float
    vinf_arr_km_s:    float
    departure_planet: str = "Departure"
    arrival_planet:   str = "Arrival"

    @property
    def vinf_dep_m_s(self): return self.vinf_dep_km_s * 1e3

    @property
    def vinf_arr_m_s(self): return self.vinf_arr_km_s * 1e3

    def total_dv(self, dep_parking_alt, dep_mass, dep_radius,
                  arr_peri_alt, arr_target_alt, arr_mass, arr_radius):
        from core.soi import patched_conic_budget
        return patched_conic_budget(dep_mass, dep_radius, dep_parking_alt,
            arr_mass, arr_radius, arr_peri_alt, arr_target_alt,
            self.vinf_dep_m_s, self.vinf_arr_m_s,
            self.departure_planet, self.arrival_planet)

    def report(self):
        return (f"=== Launch window: {self.departure_planet} -> {self.arrival_planet} ===\n"
                f"  Departure day  : {self.departure_day:.0f}\n"
                f"  Arrival day    : {self.arrival_day:.0f}\n"
                f"  ToF            : {self.tof_days:.1f} days\n"
                f"  C3             : {self.c3_km2_s2:.2f} km2/s2\n"
                f"  Departure v_inf: {self.vinf_dep_km_s:.3f} km/s\n"
                f"  Arrival v_inf  : {self.vinf_arr_km_s:.3f} km/s\n")


@dataclass
class PorkchopData:
    """Full porkchop grid: C3 and v_inf_arr over departure x arrival dates."""
    departure_days:     np.ndarray
    arrival_days:       np.ndarray
    c3:                 np.ndarray   # (n_dep, n_arr)
    vinf_arr:           np.ndarray
    tof:                np.ndarray
    valid:              np.ndarray   
    departure_planet:   str = "Departure"
    arrival_planet:     str = "Arrival"
    departure_radius_m: float = AU
    arrival_radius_m:   float = 1.524 * AU

    @classmethod
    def compute(cls, dep_radius_m, arr_radius_m, departure_days, arrival_days,
                mu_star=MU_SUN, dep_phase=0.0, arr_phase=0.0,
                min_tof_days=30.0, dep_name="Departure", arr_name="Arrival"):
        nd, na = len(departure_days), len(arrival_days)
        c3g = np.full((nd,na), np.nan); vg = np.full((nd,na), np.nan)
        tg  = np.full((nd,na), np.nan); vl = np.zeros((nd,na), dtype=bool)
        solver = LambertSolver(mu_star)
        for i, td in enumerate(departure_days):
            r1v, v1p = planet_state(dep_radius_m, td*86400, mu_star, dep_phase)
            for j, ta in enumerate(arrival_days):
                tof_d = ta - td
                if tof_d < min_tof_days: continue
                r2v, v2p = planet_state(arr_radius_m, ta*86400, mu_star, arr_phase)
                v1s, v2s = solver.solve(r1v, r2v, tof_d*86400)
                if v1s is None: continue
                vd = float(np.linalg.norm(v1s-v1p)); va = float(np.linalg.norm(v2s-v2p))
                if math.isnan(vd) or math.isnan(va) or vd>60000 or va>60000: continue
                c3g[i,j] = (vd/1e3)**2; vg[i,j] = va/1e3; tg[i,j] = tof_d; vl[i,j] = True
        return cls(np.asarray(departure_days), np.asarray(arrival_days),
                   c3g, vg, tg, vl, dep_name, arr_name, dep_radius_m, arr_radius_m)

    def best_window(self, max_c3=30.0, max_vinf_arr=8.0, min_tof_days=60.0, metric="c3"):
        mask = self.valid & (self.c3<=max_c3) & (self.vinf_arr<=max_vinf_arr) & (self.tof>=min_tof_days)
        if not mask.any(): return None
        score = np.where(mask, self.c3 if metric=="c3" else self.vinf_arr if metric=="vinf_arr" else self.c3+self.vinf_arr**2, np.inf)
        i, j = np.unravel_index(np.argmin(score), score.shape)
        return LaunchWindow(float(self.departure_days[i]), float(self.arrival_days[j]),
            float(self.tof[i,j]), float(self.c3[i,j]), float(math.sqrt(self.c3[i,j])),
            float(self.vinf_arr[i,j]), self.departure_planet, self.arrival_planet)

    def windows_in_range(self, max_c3=20.0, max_vinf_arr=6.0, min_tof_days=60.0):
        mask = self.valid & (self.c3<=max_c3) & (self.vinf_arr<=max_vinf_arr) & (self.tof>=min_tof_days)
        wins = []
        for i in range(len(self.departure_days)):
            for j in range(len(self.arrival_days)):
                if mask[i,j]:
                    wins.append(LaunchWindow(
                        float(self.departure_days[i]), float(self.arrival_days[j]),
                        float(self.tof[i,j]), float(self.c3[i,j]),
                        float(math.sqrt(self.c3[i,j])), float(self.vinf_arr[i,j]),
                        self.departure_planet, self.arrival_planet))
        wins.sort(key=lambda w: w.c3_km2_s2)
        return wins

    def cost_at(self, dep_day, arr_day):
        c3, va, tof, ok = compute_transfer(self.departure_radius_m, self.arrival_radius_m, dep_day, arr_day)
        if not ok: return None
        return LaunchWindow(dep_day, arr_day, tof, c3, math.sqrt(c3), va,
                            self.departure_planet, self.arrival_planet)

    def min_c3(self):
        v = self.c3[self.valid]; return float(v.min()) if len(v)>0 else float("nan")

    def min_vinf_arr(self):
        v = self.vinf_arr[self.valid]; return float(v.min()) if len(v)>0 else float("nan")

    def summary(self):
        n = int(self.valid.sum())
        return (f"Porkchop: {self.departure_planet} -> {self.arrival_planet}\n"
                f"  Grid     : {len(self.departure_days)}x{len(self.arrival_days)} ({n} valid)\n"
                f"  Min C3   : {self.min_c3():.2f} km2/s2\n"
                f"  Min vinf : {self.min_vinf_arr():.2f} km/s\n")


class LaunchDecisionSpace:
    """Discretised launch window decision space for RL agents."""

    def __init__(self, dep_radius_m, arr_radius_m, n_dep=20, n_arr=20,
                 window_start_day=0.0, window_duration_days=780.0,
                 min_tof_days=60.0, max_tof_days=500.0,
                 mu_star=MU_SUN, dep_phase=0.0, arr_phase=0.0):
        self.r_dep = dep_radius_m; self.r_arr = arr_radius_m
        self.n_dep = n_dep; self.n_arr = n_arr
        self.min_tof = min_tof_days; self.max_tof = max_tof_days
        self.departure_days = np.linspace(window_start_day, window_start_day+window_duration_days, n_dep)
        arr_min = window_start_day + min_tof_days
        arr_max = window_start_day + window_duration_days + max_tof_days
        self.arrival_days = np.linspace(arr_min, arr_max, n_arr)
        self._pc = PorkchopData.compute(dep_radius_m, arr_radius_m,
            self.departure_days, self.arrival_days, mu_star=mu_star,
            dep_phase=dep_phase, arr_phase=arr_phase, min_tof_days=min_tof_days)

    @property
    def porkchop(self): return self._pc

    def cost(self, dep_idx, arr_idx):
        di = max(0, min(self.n_dep-1, dep_idx)); ai = max(0, min(self.n_arr-1, arr_idx))
        if not bool(self._pc.valid[di,ai]):
            return {"valid":False,"c3":float("inf"),"vinf_dep":float("inf"),
                    "vinf_arr":float("inf"),"tof_days":float("nan")}
        c3 = float(self._pc.c3[di,ai])
        return {"valid":True, "c3":c3, "vinf_dep":float(math.sqrt(c3)),
                "vinf_arr":float(self._pc.vinf_arr[di,ai]),
                "tof_days":float(self._pc.tof[di,ai])}

    def observation(self, dep_idx, arr_idx):
        cost = self.cost(dep_idx, arr_idx)
        if not cost["valid"]:
            return np.array([dep_idx/self.n_dep, arr_idx/self.n_arr, 1.,1.,1.,0.], dtype=np.float32)
        return np.array([dep_idx/self.n_dep, arr_idx/self.n_arr,
            min(cost["c3"]/30.,1.), min(cost["vinf_arr"]/8.,1.),
            min(cost["tof_days"]/self.max_tof,1.), 1.], dtype=np.float32)

    def best_action(self):
        score = np.where(self._pc.valid, self._pc.c3, np.inf)
        i,j = np.unravel_index(np.argmin(score), score.shape)
        return int(i), int(j)

    def reward(self, dep_idx, arr_idx, max_c3=30., max_vinf_arr=8.):
        cost = self.cost(dep_idx, arr_idx)
        if not cost["valid"]: return -1.0
        return -0.5*(min(cost["c3"]/max_c3,1.) + min(cost["vinf_arr"]/max_vinf_arr,1.))

    def summary(self):
        bi, bj = self.best_action(); best = self.cost(bi,bj)
        syn = synodic_period_days(orbital_period_days(self.r_dep), orbital_period_days(self.r_arr))
        return (f"LaunchDecisionSpace\n"
                f"  Grid: {self.n_dep}x{self.n_arr}  Synodic: {syn:.0f}d\n"
                f"  Best C3: {best['c3']:.2f} km2/s2  v_inf_arr: {best['vinf_arr']:.3f} km/s\n")