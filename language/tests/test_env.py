"""T0.2: the environment can import sparsimony and its dependencies.

sparsimony imports deepspeed at module load time (parametrization/fake_sparsity.py,
dst/static.py, parametrization/dfsb.py), so a missing deepspeed breaks the whole
package, not just the ZeRO-3 code path it is needed for.
"""
import pytest


def test_deepspeed_importable():
    import deepspeed  # noqa: F401


def test_sparsimony_api_importable():
    from sparsimony import gmp, rigl, static
    from sparsimony import set as sp_set

    assert all(callable(f) for f in (rigl, sp_set, gmp, static))


def test_sparsifier_classes_importable():
    from sparsimony.dst.gmp import GMP  # noqa: F401
    from sparsimony.dst.rigl import RigL  # noqa: F401
    from sparsimony.dst.set import SET  # noqa: F401
    from sparsimony.dst.static import StaticMagnitudeSparsifier  # noqa: F401


def test_sparsimony_resolves_to_vendored_copy():
    import sparsimony

    assert "vision/sparsimony" in sparsimony.__file__


@pytest.mark.parametrize("name", ["torch", "numpy", "tiktoken", "wandb", "tqdm"])
def test_training_deps_importable(name):
    __import__(name)
