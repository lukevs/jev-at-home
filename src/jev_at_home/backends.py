"""Load the requested inference runtime."""

from importlib import import_module

from jev_at_home.evaluator import Evaluator, TransformersEvaluator
from jev_at_home.schemas import Backend, Device


def load_evaluator(
    model_name: str, backend: Backend, device: Device | None = None
) -> Evaluator:
    """Load model weights with an explicit, optional Apple MLX backend."""

    match backend:
        case Backend.TRANSFORMERS:
            return TransformersEvaluator.load(model_name, device)
        case Backend.MLX:
            if device not in (None, "mps"):
                raise ValueError("the MLX backend requires an Apple GPU")
            try:
                module = import_module("jev_at_home.mlx_evaluator")
            except ModuleNotFoundError as error:
                if error.name not in ("mlx", "mlx.core", "mlx_lm"):
                    raise
                raise ValueError(
                    "install the Apple Silicon backend with uv sync --extra mlx"
                ) from error
            return module.MLXEvaluator.load(model_name)
    raise ValueError(f"unsupported backend: {backend}")
