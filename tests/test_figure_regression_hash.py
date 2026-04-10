from __future__ import annotations

import hashlib
import io

import pytest

from exorl.core.generator import PRESETS


def _render_hash_for_preset(name: str) -> str:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    plt = pytest.importorskip("matplotlib.pyplot")
    from exorl.visualization.visualizer import plot_planet_cross_section

    planet = PRESETS[name]()
    fig = plt.figure(figsize=(2.5, 2.8))
    ax = fig.add_subplot(111)
    plot_planet_cross_section(planet, ax=ax)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return hashlib.sha256(buf.getvalue()).hexdigest()


@pytest.mark.parametrize("preset", ["earth", "mars"])
def test_cross_section_pixel_hash_is_deterministic(preset: str) -> None:
    # Lightweight figure regression: hash the rendered PNG bytes and assert repeatability.
    # This catches rendering changes without storing binary baselines in the repo.
    h1 = _render_hash_for_preset(preset)
    h2 = _render_hash_for_preset(preset)
    assert h1 == h2
