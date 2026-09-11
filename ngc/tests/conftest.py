import sys
from pathlib import Path
import jax
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from pitch_tracking import build_case


@pytest.fixture
def case():
    return build_case()
