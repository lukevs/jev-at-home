from types import SimpleNamespace

import pytest

from jev_at_home.backends import load_evaluator
from jev_at_home.schemas import Backend


def test_transformers_does_not_import_optional_mlx(monkeypatch) -> None:
    evaluator = object()
    monkeypatch.setattr(
        "jev_at_home.backends.TransformersEvaluator.load",
        lambda name, device: evaluator,
    )
    monkeypatch.setattr(
        "jev_at_home.backends.import_module",
        lambda name: pytest.fail("the default backend must not require MLX"),
    )
    assert load_evaluator("test", Backend.TRANSFORMERS, "cpu") is evaluator


def test_mlx_rejects_non_apple_device_before_loading() -> None:
    with pytest.raises(ValueError, match="Apple GPU"):
        load_evaluator("test", Backend.MLX, "cuda")


def test_mlx_reports_missing_extra(monkeypatch) -> None:
    def fail_import(name):
        raise ModuleNotFoundError("No module named mlx", name="mlx")

    monkeypatch.setattr("jev_at_home.backends.import_module", fail_import)
    with pytest.raises(ValueError, match="uv sync --extra mlx"):
        load_evaluator("test", Backend.MLX)


def test_mlx_loads_requested_model(monkeypatch) -> None:
    models = []

    def load_model(name):
        models.append(name)
        return evaluator

    evaluator = object()
    module = SimpleNamespace(MLXEvaluator=SimpleNamespace(load=load_model))
    monkeypatch.setattr("jev_at_home.backends.import_module", lambda name: module)
    assert load_evaluator("requested-model", Backend.MLX) is evaluator
    assert models == ["requested-model"]
