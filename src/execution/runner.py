"""Safe execution of LLM-generated programs over a benchmark dataset.

``execute_program`` runs a program string on every observation, reducing each
returned feature tuple to a scalar feature vector. It is hardened against the
many buggy programs the evolutionary loop generates: a restricted builtins set,
a wall-clock timeout (SIGALRM), and broad exception handling. On any failure it
returns ``ExecutionResult(success=False, ...)`` rather than raising.

Note: ``execute_program`` returns an :class:`ExecutionResult` (which carries the
extracted feature matrix). The downstream :func:`evaluator.evaluate_program`
consumes it and builds the full ``ProgramResult`` (fitness, scores, OLS weights).
"""
from __future__ import annotations

import re
import signal
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Matches whole-line `import ...` / `from ... import ...` statements. Generated
# programs habitually import numpy etc., but our namespace pre-injects np/math/
# primitives and the restricted builtins have no __import__ -- so we strip
# import lines. Programs that referenced a stripped module (e.g. skimage) then
# fail gracefully via NameError at call time (rejected), matching the paper's
# import-free discovered programs.
_IMPORT_LINE = re.compile(r"^[ \t]*(?:import|from)[ \t]+\S.*$", re.MULTILINE)


def strip_imports(code: str) -> str:
    """Remove top-of-line import statements from a program string."""
    return _IMPORT_LINE.sub("", code)

from src.data.loader import BenchmarkDataset
from src.primitives import functions as F
from src.primitives.namespace import create_namespace

# Whitelisted builtins available to executed programs. Deliberately excludes
# open/os/sys/import/eval/exec and other unsafe names.
_SAFE_BUILTIN_NAMES = [
    "range", "len", "int", "float", "str", "bool", "list", "tuple", "dict",
    "zip", "enumerate", "min", "max", "abs", "sum", "round", "print",
    "isinstance", "type",
]
import builtins as _builtins  # noqa: E402

SAFE_BUILTINS = {name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES}
# Needed implicitly by some expressions / exceptions.
SAFE_BUILTINS["True"] = True
SAFE_BUILTINS["False"] = False
SAFE_BUILTINS["None"] = None
SAFE_BUILTINS["Exception"] = Exception


@dataclass
class ExecutionResult:
    success: bool
    features: np.ndarray | None   # (N, n_features) float, or None on failure
    n_features: int
    error_msg: str | None


def _raise_timeout(signum, frame):
    raise TimeoutError("program execution timed out")


def _reduce_feature(el):
    """Reduce one returned feature element to a scalar float (None if invalid)."""
    try:
        if isinstance(el, np.ndarray):
            return float(np.mean(el))
        if isinstance(el, np.generic):     # numpy scalar (incl. np.bool_)
            return float(el)
        if isinstance(el, (int, float, bool)):
            return float(el)
    except Exception:
        return None
    return None


def preload_images(dataset: BenchmarkDataset, data_dir: str) -> list:
    """Load all RGB images for a benchmark into memory (ordered by dataset.ids)."""
    img_dir = Path(data_dir) / dataset.name / "images"
    imgs = []
    for obs_id in dataset.ids:
        try:
            imgs.append(np.load(img_dir / f"{obs_id}.npy"))
        except Exception:
            imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
    return imgs


def _bind_observation(ns: dict, i: int, dataset: BenchmarkDataset, data_dir: str) -> None:
    """Rebind the observation-scoped data-access primitives for observation i.

    The exec'd ``estimator`` looks these up in its globals (``ns``) at call time,
    so updating them in place re-scopes the program to a new observation without
    re-exec'ing the source.
    """
    ns["segment"] = F.make_segment(i, dataset)
    ns["get_satellite_image"] = F.make_get_satellite_image(i, dataset, data_dir)
    if dataset.env is not None:
        ns["get_temperature"] = F.make_env_getter(i, dataset, "temperature")
        ns["get_precipitation"] = F.make_env_getter(i, dataset, "precipitation")
        ns["get_nightlight_intensity"] = F.make_env_getter(i, dataset, "nightlight")
        ns["get_elevation"] = F.make_env_getter(i, dataset, "elevation")


def execute_program(
    program_str: str,
    dataset: BenchmarkDataset,
    data_dir: str,
    benchmark_name: str,
    timeout_seconds: int = 30,
    image_cache: list | None = None,
    obs_indices=None,
) -> ExecutionResult:
    """Execute ``program_str`` over the observations and extract features.

    Returns an ``ExecutionResult``. ``image_cache`` (a list of preloaded images
    aligned to ``dataset.ids``) lets the evolutionary loop avoid re-reading
    images every evaluation; if None, images are loaded here for population_density.
    ``obs_indices`` (an iterable of integer indices) restricts execution to a
    subset of observations (e.g. train-only during evolution); the returned
    feature matrix is aligned to that subset's order. If None, all observations.
    """
    is_population = benchmark_name == "population_density"

    # Build the execution namespace once and exec the source a single time.
    ns = create_namespace(0, dataset, data_dir)
    ns["__builtins__"] = SAFE_BUILTINS
    program_str = strip_imports(program_str)
    try:
        exec(program_str, ns)
    except Exception as e:  # syntax / definition-time error
        return ExecutionResult(False, None, 0, f"exec error: {type(e).__name__}: {e}")

    estimator = ns.get("estimator")
    if not callable(estimator):
        return ExecutionResult(False, None, 0, "no callable 'estimator' defined")

    # Pre-load images for population_density (argument to estimator(image)).
    images = None
    if is_population:
        images = image_cache if image_cache is not None else preload_images(dataset, data_dir)

    indices = range(len(dataset.ids)) if obs_indices is None else list(obs_indices)
    feats_rows: list[list[float]] = []
    n_features: int | None = None

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(int(timeout_seconds))
    try:
        for i in indices:
            _bind_observation(ns, i, dataset, data_dir)
            arg = images[i] if is_population else (float(dataset.lat[i]), float(dataset.lon[i]))
            ret = estimator(arg)

            if not isinstance(ret, (tuple, list)):
                return ExecutionResult(False, None, 0,
                                       f"estimator did not return a tuple (got {type(ret).__name__})")
            row = [_reduce_feature(el) for el in ret]
            if len(row) == 0 or any(v is None for v in row):
                return ExecutionResult(False, None, 0, "invalid/empty feature in return tuple")
            if n_features is None:
                n_features = len(row)
            elif len(row) != n_features:
                return ExecutionResult(False, None, 0, "inconsistent number of features across observations")
            feats_rows.append(row)
    except TimeoutError:
        return ExecutionResult(False, None, 0, f"timeout after {timeout_seconds}s")
    except Exception as e:
        return ExecutionResult(False, None, 0, f"runtime error: {type(e).__name__}: {e}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    features = np.asarray(feats_rows, dtype=float)
    if not np.isfinite(features).all():
        return ExecutionResult(False, None, n_features or 0, "non-finite feature values")

    return ExecutionResult(True, features, n_features or 0, None)
