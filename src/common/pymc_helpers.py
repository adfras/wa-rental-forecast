"""Shared wrappers around PyMC sampling to insulate against API shifts."""
from __future__ import annotations

from typing import Any

import pymc as pm


def _parse_version(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in raw.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


_MIN_SAFE_NUTS_VERSION = (5, 25, 0)


def sample_nuts(
    *,
    draws: int,
    tune: int,
    chains: int | None = None,
    cores: int | None = None,
    target_accept: float = 0.9,
    max_treedepth: int = 10,
    random_seed: int | None = None,
    progressbar: bool = True,
    **kwargs: Any,
) -> Any:
    """Call :func:`pm.sample` using an instantiated NUTS step.

    Passing the ``target_accept`` directly to :func:`pm.sample` began raising in
    PyMC >= 5.25 when an explicit ``step`` argument is supplied. Centralising
    the call ensures both nowcast and forecast stay compatible with future
    PyMC releases.
    """
    # Instantiate the NUTS step with the tuning arguments so that PyMC never
    # needs to infer keyword placement (behaviour changed in 5.25).
    step = pm.NUTS(target_accept=target_accept, max_treedepth=max_treedepth)

    sample_kwargs: dict[str, Any] = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": cores,
        "step": step,
        "random_seed": random_seed,
        "progressbar": progressbar,
    }
    sample_kwargs.update(kwargs)

    try:
        return pm.sample(**sample_kwargs)
    except ValueError as exc:
        message = str(exc)
        if "Invalid key" in message and "step_kwargs" in message:
            raise RuntimeError(
                "PyMC rejected the sampler kwargs (likely due to an upstream API change). "
                "Update `sample_nuts` or pin PyMC to >= "
                f"{'.'.join(map(str, _MIN_SAFE_NUTS_VERSION))} (current: {pm.__version__})."
            ) from exc
        raise
