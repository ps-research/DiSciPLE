"""Step 2 primitive-library diagnostic.

Validates the primitive implementations and the exec namespace:
  1. segment exact-match returns a {0,1} (224,224) mask; unknown concept -> zeros.
  2. math ops (max/sum/division incl. div-by-zero, log incl. log-of-zero) work.
  3. min_pixel_distance_to_mask: 0 at mask pixels, positive elsewhere; all-zero
     mask -> all large positive (no NaN/inf).
  4. get_average(segment(...)) returns a float in [0, 1].
  5. env getters return floats in expected ranges (on an env benchmark).
  6. create_namespace returns a dict with all expected primitive names.
  7. exec() integration: a trivial program runs in the namespace.

Prints PASS/FAIL per check and exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import load_benchmark  # noqa: E402
from src.primitives import create_namespace  # noqa: E402
from src.primitives import functions as F  # noqa: E402


class Reporter:
    def __init__(self) -> None:
        self.ok = True

    def check(self, label: str, passed: bool, detail: str = "") -> bool:
        status = "PASS" if passed else "FAIL"
        line = f"  [{status}] {label}"
        if detail:
            line += f" -- {detail}"
        print(line)
        if not passed:
            self.ok = False
        return passed


def main() -> int:
    config = load_config(ROOT / "configs" / "default.yaml")
    data_dir = ROOT / config.paths.data_dir

    rep = Reporter()

    # ---- population_density: segment / math / distance / get_average ------- #
    print("\n==================== population_density ====================")
    pop = load_benchmark("population_density", config)
    ns = create_namespace(0, pop, str(data_dir))
    seg = ns["segment"]

    # 1. segment
    hw = seg(None, "highway")
    rep.check(
        "segment('highway') -> (224,224) mask in {0,1}",
        hw.shape == (224, 224) and set(np.unique(hw)).issubset({0, 1}),
        f"shape={hw.shape} uniq={np.unique(hw).tolist()}",
    )
    nz = seg(None, "nonexistent_concept_xyz")
    rep.check(
        "segment(unknown) -> all zeros, no crash",
        nz.shape == (224, 224) and not nz.any(),
        f"sum={int(nz.sum())}",
    )
    # bonus: synonym mapping ('road' -> 'highway')
    syn = seg(None, "road")
    rep.check(
        "segment('road') synonym-maps to 'highway'",
        np.array_equal(syn, hw),
        f"equal_to_highway={np.array_equal(syn, hw)}",
    )

    # 2. math ops
    a = np.array([[1.0, 2.0], [3.0, 0.0]])
    b = np.array([[0.0, 2.0], [1.0, 4.0]])
    try:
        mx = F.elementwise_max(a, b)
        sm = F.elementwise_sum(a, b)
        dv = F.elementwise_division(a, b)  # includes division by zero (b[0,0]=0)
        lg = F.elementwise_log(b)          # includes log of zero (b[0,0]=0)
        ok = (
            np.array_equal(mx, np.array([[1, 2], [3, 4]]))
            and np.array_equal(sm, np.array([[1, 4], [4, 4]]))
            and dv[0, 0] == 0.0 and np.isclose(dv[1, 1], 0.0)
            and np.isfinite(lg).all()
        )
        rep.check("math ops (max/sum/div-by-zero/log-of-zero) correct & finite",
                  bool(ok), f"div[0,0]={dv[0,0]} log finite={np.isfinite(lg).all()}")
    except Exception as e:
        rep.check("math ops run without exceptions", False, repr(e))

    # 3. distance transform
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[100:105, 100:105] = 1
    dist = F.min_pixel_distance_to_mask(mask)
    rep.check(
        "min_pixel_distance_to_mask: 0 at mask, >0 elsewhere",
        np.isclose(dist[102, 102], 0.0) and dist[0, 0] > 0 and np.isfinite(dist).all(),
        f"d[102,102]={dist[102,102]:.2f} d[0,0]={dist[0,0]:.2f}",
    )
    dist0 = F.min_pixel_distance_to_mask(np.zeros((224, 224), dtype=np.uint8))
    rep.check(
        "all-zero mask -> all large positive, no NaN/inf",
        np.isfinite(dist0).all() and (dist0 > 0).all(),
        f"min={dist0.min():.2f} max={dist0.max():.2f}",
    )

    # 4. get_average
    avg = ns["get_average"](seg(None, "highway"))
    rep.check("get_average(segment('highway')) in [0,1]",
              isinstance(avg, float) and 0.0 <= avg <= 1.0, f"avg={avg:.4f}")

    # ---- env benchmark (poverty): env getters ------------------------------ #
    print("\n==================== poverty (env) ====================")
    pov = load_benchmark("poverty", config)
    ns_env = create_namespace(0, pov, str(data_dir))
    env_specs = {
        "get_temperature": (0, 255),
        "get_precipitation": (0, 255),
        "get_elevation": (0, 255),
        "get_nightlight_intensity": (0, 1),
    }
    for fn_name, (lo, hi) in env_specs.items():
        val = ns_env[fn_name](None)
        rep.check(
            f"{fn_name}() float in [{lo},{hi}]",
            isinstance(val, float) and lo <= val <= hi,
            f"val={val:.4f}",
        )

    # ---- namespace completeness -------------------------------------------- #
    print("\n==================== namespace ====================")
    expected_pop = {
        "elementwise_max", "elementwise_min", "elementwise_sum",
        "elementwise_product", "elementwise_division",
        "matrix_scalar_multiplication", "elementwise_log",
        "elementwise_exponentiate", "min_pixel_distance_to_mask",
        "get_average", "segment", "get_satellite_image", "np", "math",
    }
    rep.check("population namespace has all expected names",
              expected_pop.issubset(set(ns)),
              f"missing={sorted(expected_pop - set(ns))}")
    expected_env = {"get_temperature", "get_precipitation",
                    "get_elevation", "get_nightlight_intensity"}
    rep.check("poverty namespace has env getters",
              expected_env.issubset(set(ns_env)),
              f"missing={sorted(expected_env - set(ns_env))}")
    rep.check("population namespace has NO env getters",
              not (expected_env & set(ns)),
              f"unexpected={sorted(expected_env & set(ns))}")

    # ---- exec integration smoke test --------------------------------------- #
    print("\n==================== exec integration ====================")
    prog = "feature1 = get_average(segment(im, 'highway'))\nresult = (feature1,)"
    g = create_namespace(0, pop, str(data_dir))
    g["im"] = None
    try:
        exec(prog, g)
        res = g["result"]
        rep.check("exec'd program yields tuple of one float",
                  isinstance(res, tuple) and len(res) == 1 and isinstance(res[0], float),
                  f"result={res}")
    except Exception as e:
        rep.check("exec'd program runs without error", False, repr(e))

    print("\n" + "=" * 52)
    if rep.ok:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
