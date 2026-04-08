"""
planet_io.py — Planet serialisation, deserialisation, and fingerprinting.

Every RL experiment and science run should log the planet fingerprint so
results can be exactly reproduced.

Usage
-----
    from exorl.core.planet_io import planet_to_json, planet_from_json, planet_fingerprint

    # Save a planet
    json_str = planet_to_json(planet)
    with open("planet_config.json", "w") as f:
        f.write(json_str)

    # Reload it — guaranteed identical physics
    planet2 = planet_from_json(json_str)

    # Log the fingerprint in experiment metadata
    print(planet_fingerprint(planet))   # "4771b11296a4e6ca"

    # Or attach directly to the planet
    planet.to_json()            # → JSON string
    Planet.from_json(json_str)  # → Planet  (class method)
    planet.fingerprint          # → 16-char hex string

Schema
------
    Version 1.0 — all fields from planet.py dataclasses, enums stored as names.
    The fingerprint is the first 16 hex chars of SHA-256 of the canonical JSON
    (sorted keys, no interior/star context — only the generatable fields).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Optional

# ── Schema version ────────────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _enum_name(val) -> str:
    """Convert an Enum value to its string name, or pass through primitives."""
    if hasattr(val, "name"):
        return val.name
    return val


def _config_to_dict(cfg) -> Optional[dict]:
    """Serialise a dataclass config to a plain dict."""
    if cfg is None:
        return None
    d = {}
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        d[f.name] = _enum_name(val)
    return d


def _config_from_dict(cls, d: dict):
    """Restore a dataclass config from a plain dict, resolving enum strings."""
    if d is None:
        return cls()
    kwargs = {}
    hints = {f.name: f for f in dataclasses.fields(cls)}
    for key, val in d.items():
        if key not in hints:
            continue
        field = hints[key]
        # Resolve enum strings back to enum values
        ftype = field.type if isinstance(field.type, type) else None
        if ftype is not None and issubclass(ftype, object):
            # Try to resolve via module globals
            pass
        kwargs[key] = _resolve_enum(cls, key, val)
    return cls(**kwargs)


def _resolve_enum(parent_cls, field_name: str, val):
    """
    If val is a string that matches an enum member, return the enum value.
    Otherwise return val unchanged.
    """
    if not isinstance(val, str):
        return val
    try:
        import exorl.core.planet as _pm

        field_obj = {f.name: f for f in dataclasses.fields(parent_cls)}.get(field_name)
        if field_obj is None:
            return val
        hint = field_obj.type
        # Resolve string type hints
        if isinstance(hint, str):
            obj = getattr(_pm, hint, None)
            if obj is not None and hasattr(obj, "__mro__"):
                hint = obj
        if isinstance(hint, type) and issubclass(hint, __import__("enum").Enum):
            return hint[val]
    except Exception:
        pass
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def planet_to_dict(planet) -> dict:
    """
    Convert a Planet to a JSON-serialisable dict.

    Only the generatable fields are included — interior, star_context, and
    orbital_distance_m are runtime-attached and not part of the canonical
    definition. Regenerate them after loading.
    """
    from exorl.core.planet import Planet

    return {
        "schema_version": SCHEMA_VERSION,
        "name": planet.name,
        "radius_m": planet.radius,
        "mass_kg": planet.mass,
        "rotation_enabled": planet.rotation_enabled,
        "rotation_period_s": planet.rotation_period,
        "atmosphere": _config_to_dict(planet.atmosphere),
        "terrain": _config_to_dict(planet.terrain),
        "magnetic_field": _config_to_dict(planet.magnetic_field),
        "oblateness": _config_to_dict(planet.oblateness),
        "moons": _config_to_dict(planet.moons),
    }


def planet_to_json(planet, indent: int = 2) -> str:
    """
    Serialise a Planet to a JSON string.

    The output includes a 'fingerprint' field — the first 16 hex chars of
    SHA-256 of the canonical JSON (with fingerprint omitted).
    Use this in experiment logs to identify the exact planet used.
    """
    d = planet_to_dict(planet)
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    fp = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    d["fingerprint"] = fp
    return json.dumps(d, indent=indent, sort_keys=True)


def planet_fingerprint(planet) -> str:
    """
    Return the 16-char hex fingerprint for a planet.
    Identical planets always produce the same fingerprint.
    """
    d = planet_to_dict(planet)
    raw = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def planet_from_dict(d: dict):
    """
    Restore a Planet from a dict (as returned by planet_to_dict or parsed JSON).
    """
    from exorl.core.planet import (
        AtmosphereComposition,
        AtmosphereConfig,
        MagneticFieldConfig,
        MagneticFieldStrength,
        MoonConfig,
        OblatenessConfig,
        Planet,
        TerrainConfig,
        TerrainType,
    )

    def _atm(raw):
        if not raw:
            return AtmosphereConfig()
        return AtmosphereConfig(
            enabled=raw.get("enabled", True),
            composition=AtmosphereComposition[raw["composition"]]
            if "composition" in raw
            else AtmosphereComposition.EARTH_LIKE,
            scale_height=raw.get("scale_height", 8500.0),
            surface_pressure=raw.get("surface_pressure", 101325.0),
            surface_density=raw.get("surface_density", 1.225),
            surface_temp=raw.get("surface_temp", 288.0),
            lapse_rate=raw.get("lapse_rate", 0.0065),
            drag_coeff_multiplier=raw.get("drag_coeff_multiplier", 1.0),
            wind_enabled=raw.get("wind_enabled", False),
            wind_speed_mps=raw.get("wind_speed_mps", 0.0),
            wind_direction_deg=raw.get("wind_direction_deg", 0.0),
        )

    def _terrain(raw):
        if not raw:
            return TerrainConfig()
        return TerrainConfig(
            enabled=raw.get("enabled", True),
            terrain_type=TerrainType[raw["terrain_type"]]
            if "terrain_type" in raw
            else TerrainType.FLAT,
            max_elevation=raw.get("max_elevation", 8848.0),
            min_elevation=raw.get("min_elevation", -10_935.0),
            roughness=raw.get("roughness", 0.5),
            seed=raw.get("seed", 42),
        )

    def _mag(raw):
        if not raw:
            return MagneticFieldConfig()
        return MagneticFieldConfig(
            enabled=raw.get("enabled", True),
            strength=MagneticFieldStrength[raw["strength"]]
            if "strength" in raw
            else MagneticFieldStrength.MEDIUM,
            tilt_deg=raw.get("tilt_deg", 11.5),
            radiation_belt_enabled=raw.get("radiation_belt_enabled", True),
            inner_belt_altitude=raw.get("inner_belt_altitude", 2_000_000.0),
            outer_belt_altitude=raw.get("outer_belt_altitude", 20_000_000.0),
        )

    def _obl(raw):
        if not raw:
            return OblatenessConfig()
        return OblatenessConfig(
            enabled=raw.get("enabled", True),
            J2=raw.get("J2", 1.08263e-3),
            J3=raw.get("J3", -2.53e-6),
            flattening=raw.get("flattening", 3.35e-3),
        )

    def _moons(raw):
        if not raw:
            return MoonConfig()
        return MoonConfig(
            enabled=raw.get("enabled", True),
            count=raw.get("count", 1),
            mass_fraction=raw.get("mass_fraction", 0.0123),
            orbit_radius=raw.get("orbit_radius", 384_400_000.0),
        )

    schema = d.get("schema_version", "1.0")
    if schema not in ("1.0",):
        raise ValueError(f"Unknown planet JSON schema version: {schema!r}")

    p = Planet(
        name=d.get("name", "Unknown"),
        radius=float(d.get("radius_m", 6_371_000.0)),
        mass=float(d.get("mass_kg", 5.972e24)),
        rotation_enabled=bool(d.get("rotation_enabled", True)),
        rotation_period=float(d.get("rotation_period_s", 86_164.1)),
        atmosphere=_atm(d.get("atmosphere")),
        terrain=_terrain(d.get("terrain")),
        magnetic_field=_mag(d.get("magnetic_field")),
        oblateness=_obl(d.get("oblateness")),
        moons=_moons(d.get("moons")),
    )
    return p


def planet_from_json(json_str: str):
    """Deserialise a Planet from a JSON string produced by planet_to_json."""
    d = json.loads(json_str)
    return planet_from_dict(d)


def save_planet(planet, path: str) -> str:
    """
    Save planet to a .json file.  Returns the fingerprint.

    Example
    -------
        fp = save_planet(earth, "experiment_planets/earth.json")
        print(f"Saved Earth  fingerprint={fp}")
    """
    js = planet_to_json(planet)
    with open(path, "w") as f:
        f.write(js)
    return planet_fingerprint(planet)


def load_planet(path: str):
    """Load a Planet from a .json file."""
    with open(path) as f:
        return planet_from_json(f.read())


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch Planet with to_json / from_json / fingerprint
# Called automatically on import of this module.
# ─────────────────────────────────────────────────────────────────────────────


def _patch_planet():
    from exorl.core.planet import Planet

    def to_json(self, indent=2) -> str:
        """Serialise this planet to a JSON string. See planet_io.planet_to_json."""
        return planet_to_json(self, indent=indent)

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialise a Planet from a JSON string. See planet_io.planet_from_json."""
        return planet_from_json(json_str)

    @property
    def fingerprint(self) -> str:
        """16-char hex identifier for this exact planet configuration."""
        return planet_fingerprint(self)

    Planet.to_json = to_json
    Planet.from_json = from_json
    Planet.fingerprint = fingerprint


_patch_planet()
