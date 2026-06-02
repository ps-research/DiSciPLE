"""Primitive function implementations for DiSciPLE programs.

Two kinds of primitives:

* **Pure math/logic primitives** (`elementwise_*`, `min_pixel_distance_to_mask`,
  `get_average`, ...) -- stateless module-level functions.
* **Data-access primitives** (`segment`, `get_satellite_image`, env getters) --
  built per-observation by the ``make_*`` closure factories in this module,
  capturing that observation's masks / env values / id.

Every primitive is defensively robust: the evolutionary loop generates many
buggy programs, and a primitive that raises would abort an otherwise-scoreable
program. On bad input each primitive returns a sensible neutral default
(zeros array or ``0.0``) instead of raising.
"""
from __future__ import annotations

import math  # noqa: F401  (exposed in the exec namespace)
from pathlib import Path

import numpy as np
import scipy.ndimage

# Synonyms seen in the paper's discovered programs that do not exactly match an
# OSM concept name. With GRAFT (open-vocab) any prompt yields a mask; with our
# precomputed 42 OSM masks we map known variants to their canonical concept.
# Anything not covered (e.g. 'poverty', 'education', 'health') falls through to
# an all-zero mask -- a deliberate, documented OSM-vs-GRAFT difference.
SYNONYMS = {
    "road": "highway",
    "roads": "highway",
    "street": "highway",
    "tree": "forest",
    "trees": "forest",
    "forests": "forest",
    "buildings": "non-residential buildings",
    "water": "lake",
    "stream": "river",
    "grass": "park",
    "vegetation": "forest",
}

MASK_SHAPE = (224, 224)


def _zeros_like(*args) -> np.ndarray | float:
    """Fallback value matching the shape of the first array-like argument."""
    for a in args:
        arr = np.asarray(a)
        if arr.ndim > 0:
            return np.zeros(arr.shape, dtype=float)
    return 0.0


# --------------------------------------------------------------------------- #
# Pure math / logic primitives                                                #
# --------------------------------------------------------------------------- #
def elementwise_max(matrix1, matrix2):
    try:
        return np.maximum(matrix1, matrix2)
    except Exception:
        return _zeros_like(matrix1, matrix2)


def elementwise_min(matrix1, matrix2):
    try:
        return np.minimum(matrix1, matrix2)
    except Exception:
        return _zeros_like(matrix1, matrix2)


def elementwise_sum(matrix1, matrix2):
    try:
        return matrix1 + matrix2
    except Exception:
        return _zeros_like(matrix1, matrix2)


def elementwise_product(matrix1, matrix2):
    try:
        return matrix1 * matrix2
    except Exception:
        return _zeros_like(matrix1, matrix2)


def elementwise_division(matrix1, matrix2):
    """Safe element-wise division: 0 where the denominator is 0."""
    try:
        m1 = np.asarray(matrix1, dtype=float)
        m2 = np.asarray(matrix2, dtype=float)
        return np.divide(m1, m2, out=np.zeros_like(m1, dtype=float), where=m2 != 0)
    except Exception:
        return _zeros_like(matrix1, matrix2)


def matrix_scalar_multiplication(matrix, scalar):
    try:
        return matrix * scalar
    except Exception:
        return _zeros_like(matrix)


def elementwise_log(matrix):
    """Element-wise log, clipped to avoid log(0)."""
    try:
        return np.log(np.clip(matrix, 1e-10, None))
    except Exception:
        return _zeros_like(matrix)


def elementwise_exponentiate(matrix, base):
    try:
        return np.power(base, matrix)
    except Exception:
        return _zeros_like(matrix)


def min_pixel_distance_to_mask(mask):
    """Distance from each pixel to the nearest mask=1 pixel (EDT of 1-mask)."""
    try:
        m = np.asarray(mask)
        dist = scipy.ndimage.distance_transform_edt(1 - (m > 0).astype(np.uint8))
        # An all-zero mask yields large finite distances; guard any inf/nan.
        return np.nan_to_num(dist, nan=0.0, posinf=float(max(m.shape)), neginf=0.0)
    except Exception:
        return _zeros_like(mask)


def get_average(segmented_image):
    """Average pixel value of a (segmented) image."""
    try:
        return float(np.mean(segmented_image))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Data-access primitive closure factories (observation-scoped)                #
# --------------------------------------------------------------------------- #
def make_segment(obs_index: int, dataset):
    """Closure: segment(im, text_prompt) -> (224,224) uint8 mask for this obs.

    Three-tier lookup: exact concept name -> synonym map -> all-zeros.
    The ``im`` argument is ignored (perception is precomputed OSM masks).
    """
    concepts = set(dataset.concepts)

    def segment(im, text_prompt="trees"):
        key = str(text_prompt).strip().lower()
        concept = None
        if key in concepts:
            concept = key
        elif key in SYNONYMS and SYNONYMS[key] in concepts:
            concept = SYNONYMS[key]
        if concept is None:
            return np.zeros(MASK_SHAPE, dtype=np.uint8)
        return dataset.masks[concept][obs_index].astype(np.uint8).copy()

    return segment


def make_get_satellite_image(obs_index: int, dataset, data_dir: str):
    """Closure: get_satellite_image(location) -> RGB (224,224,3), lazy-loaded.

    Loaded images are memoized on the dataset (keyed by observation index) so
    repeated evaluations across the evolutionary loop don't re-read from disk.
    """
    img_path = Path(data_dir) / dataset.name / "images" / f"{dataset.ids[obs_index]}.npy"

    def get_satellite_image(location):
        cache = getattr(dataset, "_satimg_cache", None)
        if cache is None:
            cache = {}
            setattr(dataset, "_satimg_cache", cache)
        if obs_index in cache:
            return cache[obs_index]
        try:
            arr = np.load(img_path)
        except Exception:
            arr = np.zeros((*MASK_SHAPE, 3), dtype=np.uint8)
        cache[obs_index] = arr
        return arr

    return get_satellite_image


def make_env_getter(obs_index: int, dataset, var_name: str):
    """Closure: get_<env>(location) -> float scalar for this observation."""
    def getter(location):
        try:
            return float(dataset.env[var_name][obs_index])
        except Exception:
            return 0.0

    return getter
