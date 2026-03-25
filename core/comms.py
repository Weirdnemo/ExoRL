"""
comms.py — Spacecraft communications and data budget.

Models the downlink data rate from a spacecraft in planetary orbit to Earth,
as a function of planet-Earth distance, antenna parameters, and frequency band.

Physical model (link budget simplified)
----------------------------------------
    FSPL = (4π d f / c)²     free-space path loss
    P_rx = EIRP × G_rx / FSPL   received power
    SNR  = P_rx / (k_B × T_sys × B)
    Rate = B × log2(1 + SNR)    Shannon capacity [bps]

Simplified scaling (for RL use)
---------------------------------
    Rate ∝ EIRP × G_rx × B × c² / (4π f d)²

    The key dependence is Rate ∝ 1/d². Distance d is the Earth-to-planet
    distance which varies with the synodic cycle.

References
----------
    Wertz & Larson — Space Mission Engineering (2011)
    Haynes — DSN Telecommunications Link Design Handbook

Usage
-----
    from core.comms import CommsModel, AntennaConfig

    antenna = AntennaConfig(diameter_m=0.5, frequency_GHz=8.4)
    comms   = CommsModel(planet, antenna, earth_dist_au=1.524)

    print(comms.downlink_rate_bps)         # current downlink rate
    print(comms.downlink_rate_Mbps)        # in Mbps
    print(comms.data_volume_per_pass_Mb(contact_duration_s=600))
    print(comms.light_travel_time_s)       # one-way signal delay
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
C_LIGHT     = 2.998e8         # m/s
K_BOLTZMANN = 1.380649e-23    # J/K
AU          = 1.495_978_707e11  # m

# DSN Deep Space Network typical parameters
DSN_G_RX_dBi   = 68.0    # DSN 34m dish gain at X-band [dBi]
DSN_T_SYS_K    = 20.0    # System noise temperature [K]

# Reference link: MRO (Mars Reconnaissance Orbiter) at closest approach
# Achieved ~6 Mbps with 3m dish at X-band (8.4 GHz), ~0.5 AU, 100W TWTA
MRO_RATE_MBPS   = 6.0
MRO_DIST_AU     = 0.5
MRO_DISH_M      = 3.0
MRO_POWER_W     = 100.0
MRO_FREQ_GHZ    = 8.4


# ─────────────────────────────────────────────────────────────────────────────
# Antenna configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AntennaConfig:
    """
    Spacecraft high-gain antenna parameters.

    Attributes
    ----------
    diameter_m      : antenna dish diameter [m]
    frequency_GHz   : carrier frequency [GHz]
    tx_power_W      : transmitter power [W]
    efficiency      : antenna efficiency (0–1)
    pointing_loss_dB: pointing error loss [dB]
    """
    diameter_m      : float = 1.0
    frequency_GHz   : float = 8.4      # X-band (standard for deep space)
    tx_power_W      : float = 50.0     # W
    efficiency      : float = 0.65
    pointing_loss_dB: float = 0.5

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / (self.frequency_GHz * 1e9)

    @property
    def gain_linear(self) -> float:
        """Antenna gain (linear, dimensionless)."""
        eta = self.efficiency
        D   = self.diameter_m
        lam = self.wavelength_m
        G   = eta * (math.pi * D / lam) ** 2
        # Apply pointing loss
        G_pointing = 10 ** (-self.pointing_loss_dB / 10)
        return G * G_pointing

    @property
    def eirp_W(self) -> float:
        """Effective Isotropic Radiated Power [W]."""
        return self.tx_power_W * self.gain_linear

    @property
    def beamwidth_deg(self) -> float:
        """3dB beamwidth [degrees] — used for pointing budget."""
        return 70 * self.wavelength_m / self.diameter_m


# ─────────────────────────────────────────────────────────────────────────────
# Communications model
# ─────────────────────────────────────────────────────────────────────────────

class CommsModel:
    """
    Downlink data rate from a spacecraft in planetary orbit to Earth.

    Parameters
    ----------
    planet          : Planet object
    antenna         : AntennaConfig for the spacecraft HGA
    earth_dist_au   : current planet-Earth distance [AU]
    ground_ant_diam : DSN receiving dish diameter [m] (default: 34m)
    bandwidth_MHz   : channel bandwidth [MHz]
    """

    def __init__(self,
                 planet,
                 antenna:          AntennaConfig = None,
                 earth_dist_au:    float         = 1.0,
                 ground_ant_diam:  float         = 34.0,
                 bandwidth_MHz:    float         = 50.0,
                 system_noise_K:   float         = DSN_T_SYS_K):

        self.planet           = planet
        self.antenna          = antenna or AntennaConfig()
        self.earth_dist_m     = earth_dist_au * AU
        self.ground_ant_diam  = ground_ant_diam
        self.bandwidth_Hz     = bandwidth_MHz * 1e6
        self.system_noise_K   = system_noise_K

    # ── Ground station gain ───────────────────────────────────────────────────

    @property
    def ground_gain_linear(self) -> float:
        """DSN receiving dish gain (linear)."""
        lam = self.antenna.wavelength_m
        D   = self.ground_ant_diam
        return 0.65 * (math.pi * D / lam) ** 2

    # ── Link budget ───────────────────────────────────────────────────────────

    @property
    def free_space_path_loss(self) -> float:
        """Free-space path loss (linear, not dB)."""
        f   = self.antenna.frequency_GHz * 1e9
        d   = self.earth_dist_m
        return (4 * math.pi * d * f / C_LIGHT) ** 2

    @property
    def received_power_W(self) -> float:
        """Power received at the ground station [W]."""
        return (self.antenna.eirp_W
                * self.ground_gain_linear
                / self.free_space_path_loss)

    @property
    def snr_linear(self) -> float:
        """Signal-to-noise ratio (linear)."""
        noise = K_BOLTZMANN * self.system_noise_K * self.bandwidth_Hz
        return self.received_power_W / max(noise, 1e-30)

    @property
    def downlink_rate_bps(self) -> float:
        """
        Shannon channel capacity [bps] — theoretical maximum.
        In practice, coded systems achieve ~75% of Shannon capacity.
        """
        rate_shannon = self.bandwidth_Hz * math.log2(1 + self.snr_linear)
        return rate_shannon * 0.75   # coding efficiency

    @property
    def downlink_rate_Mbps(self) -> float:
        return self.downlink_rate_bps / 1e6

    @property
    def downlink_rate_kbps(self) -> float:
        return self.downlink_rate_bps / 1e3

    # ── Derived quantities ────────────────────────────────────────────────────

    @property
    def light_travel_time_s(self) -> float:
        """One-way signal travel time [s]."""
        return self.earth_dist_m / C_LIGHT

    @property
    def round_trip_delay_s(self) -> float:
        """Round-trip command/response delay [s]."""
        return 2 * self.light_travel_time_s

    def data_volume_per_pass_Mb(self, contact_duration_s: float) -> float:
        """
        Total data volume downlinked in one ground station contact [Mb].

        contact_duration_s : duration of ground station visibility [s].
        """
        return self.downlink_rate_bps * contact_duration_s / 1e6

    def min_distance_for_rate_au(self, target_rate_bps: float) -> float:
        """
        Maximum distance at which the given data rate is achievable [AU].
        Solves FSPL equation for d.
        """
        # Rate ∝ log2(1 + EIRP × G_rx / (FSPL × k_B × T × B))
        # Solve numerically
        for d_au in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
            old_dist = self.earth_dist_m
            self.earth_dist_m = d_au * AU
            rate = self.downlink_rate_bps
            self.earth_dist_m = old_dist
            if rate <= target_rate_bps:
                return d_au
        return float("inf")

    def contact_windows_per_day(self, altitude_m: float,
                                 ground_station_lat: float = 0.0,
                                 ground_station_lon: float = 0.0) -> float:
        """
        Approximate number of ground station contact windows per day.
        Simple geometric estimate: depends on orbital period and ground station
        visibility cone.
        """
        T_orb = 2 * math.pi * math.sqrt(
            (self.planet.radius + altitude_m) ** 3 / self.planet.mu)
        T_day = 86400.0
        n_per_day = T_day / T_orb
        # Average visibility fraction per pass for a ground station at given lat
        vis_fraction = math.cos(math.radians(ground_station_lat)) * 0.15
        return n_per_day * vis_fraction

    # ── Data budget helpers ───────────────────────────────────────────────────

    def daily_downlink_capacity_Gb(self, contact_s_per_day: float = 3600) -> float:
        """
        Total downlink capacity per day [Gb] for a given contact time budget.

        contact_s_per_day : total contact time with ground stations [s/day]
        """
        return self.downlink_rate_bps * contact_s_per_day / 1e9

    def data_rate_vs_distance(self, distances_au) -> list[float]:
        """
        Compute downlink rate [Mbps] at a sequence of distances.
        Useful for plotting rate variation over the synodic cycle.
        """
        rates = []
        old_dist = self.earth_dist_m
        for d in distances_au:
            self.earth_dist_m = d * AU
            rates.append(self.downlink_rate_Mbps)
        self.earth_dist_m = old_dist
        return rates

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"CommsModel  ({self.planet.name}, d={self.earth_dist_m/AU:.3f} AU)",
            f"  Antenna: {self.antenna.diameter_m:.1f}m  "
            f"f={self.antenna.frequency_GHz:.1f}GHz  "
            f"P_tx={self.antenna.tx_power_W:.0f}W",
            f"  EIRP           : {10*math.log10(self.antenna.eirp_W):.1f} dBW",
            f"  Path loss      : {10*math.log10(self.free_space_path_loss):.1f} dB",
            f"  SNR            : {10*math.log10(max(self.snr_linear,1e-30)):.1f} dB",
            f"  Downlink rate  : {self.downlink_rate_Mbps:.3f} Mbps",
            f"  Light delay    : {self.light_travel_time_s:.1f} s  "
            f"({self.round_trip_delay_s/60:.1f} min RTT)",
            f"  Downlink/day   : {self.daily_downlink_capacity_Gb():.3f} Gb/day (1hr contact)",
        ]
        return "\n".join(lines)

    # ── RL observation vector ─────────────────────────────────────────────────

    def obs_vector(self, buffer_fill_frac: float = 0.0) -> "np.ndarray":
        """
        3-element normalised observation vector for RL agents:
          [rate_norm, delay_norm, buffer_urgency]
        """
        import numpy as np
        rate_norm   = min(self.downlink_rate_Mbps / 100.0, 1.0)
        delay_norm  = min(self.light_travel_time_s / 2400.0, 1.0)  # 0–40 min
        urgency     = float(buffer_fill_frac)
        return np.array([rate_norm, delay_norm, urgency], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: Earth-planet distance over synodic cycle
# ─────────────────────────────────────────────────────────────────────────────

def earth_planet_distance_au(planet_dist_au: float,
                              phase_angle_deg: float) -> float:
    """
    Earth-planet distance [AU] at a given solar phase angle.

    Uses the law of cosines in the heliocentric plane.
    phase_angle_deg = 0   → opposition (closest approach)
    phase_angle_deg = 180 → conjunction (farthest)
    """
    theta = math.radians(phase_angle_deg)
    d2 = (1.0**2 + planet_dist_au**2
          - 2 * 1.0 * planet_dist_au * math.cos(theta))
    return math.sqrt(max(d2, 1e-10))


def synodic_distance_range_au(planet_dist_au: float) -> tuple[float, float]:
    """
    (min, max) Earth-planet distance over one synodic period [AU].

    min = |d_planet - 1|  (opposition)
    max = d_planet + 1    (conjunction)
    """
    return (abs(planet_dist_au - 1.0), planet_dist_au + 1.0)
