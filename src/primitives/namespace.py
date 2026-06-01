"""Build the execution namespace for running a program on one observation.

The returned dict is passed as the globals to ``exec()`` when a discovered
program is run. It contains every primitive (by name), with the data-access
primitives bound as closures to a specific observation, plus ``np``/``math``
so programs can use standard numpy/math directly.
"""
from __future__ import annotations

import math

import numpy as np

from . import functions as F

# Pure (observation-independent) primitives, exposed by name.
_PURE_PRIMITIVES = {
    "elementwise_max": F.elementwise_max,
    "elementwise_min": F.elementwise_min,
    "elementwise_sum": F.elementwise_sum,
    "elementwise_product": F.elementwise_product,
    "elementwise_division": F.elementwise_division,
    "matrix_scalar_multiplication": F.matrix_scalar_multiplication,
    "elementwise_log": F.elementwise_log,
    "elementwise_exponentiate": F.elementwise_exponentiate,
    "min_pixel_distance_to_mask": F.min_pixel_distance_to_mask,
    "get_average": F.get_average,
}

# Maps env primitive name -> manifest env column name.
_ENV_PRIMITIVES = {
    "get_temperature": "temperature",
    "get_precipitation": "precipitation",
    "get_nightlight_intensity": "nightlight",
    "get_elevation": "elevation",
}


def create_namespace(obs_index: int, dataset, data_dir: str) -> dict:
    """Create the exec() namespace for running a program on one observation.

    Returns a dict with all primitives injected by name (data-access ones bound
    to ``obs_index``) plus ``np`` and ``math``.
    """
    ns: dict = {}

    # numpy / math for direct use by programs (e.g. np.logical_and, math.log).
    ns["np"] = np
    ns["numpy"] = np
    ns["math"] = math

    # Pure math/logic primitives.
    ns.update(_PURE_PRIMITIVES)

    # Observation-scoped data-access primitives.
    ns["segment"] = F.make_segment(obs_index, dataset)
    ns["get_satellite_image"] = F.make_get_satellite_image(obs_index, dataset, data_dir)

    # Environment getters only exist for benchmarks that carry env variables.
    if dataset.env is not None:
        for prim_name, col in _ENV_PRIMITIVES.items():
            ns[prim_name] = F.make_env_getter(obs_index, dataset, col)

    return ns
