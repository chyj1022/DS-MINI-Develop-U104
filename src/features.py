"""Features available by cycle 100; no target or later-cycle inputs."""
import numpy as np
from itertools import combinations

FEATURE_SETS = {
    "variance": ["log10_dq_variance"],
    "early": ["log10_dq_variance", "qd_slope_2_100", "charge_time_mean_2_6"],
    "early_ir": ["log10_dq_variance", "qd_slope_2_100", "charge_time_mean_2_6", "ir_change_100_2"],
    "early_temperature": ["log10_dq_variance", "qd_slope_2_100", "charge_time_mean_2_6", "tavg_mean_2_100"],
}
CORE = "log10_dq_variance"
AUXILIARY = ["qd_slope_2_100", "charge_time_mean_2_6", "ir_change_100_2", "tavg_mean_2_100"]
# Full ablation: the core stays fixed while all four auxiliary features vary.
for size in range(1, 5):
    for subset in combinations(AUXILIARY, size):
        values = [CORE, *subset]
        if values not in FEATURE_SETS.values():
            name = "full" if size == 4 else "plus_" + "_".join(str(AUXILIARY.index(s)+1) for s in subset)
            FEATURE_SETS[name] = values
ALL_FEATURES = list(dict.fromkeys(f for fs in FEATURE_SETS.values() for f in fs))


def extract_features(summary, q10, q100, voltage):
    cycle = summary["cycle"]
    if len(np.unique(cycle)) != len(cycle):
        raise ValueError("Duplicate cycle numbers")
    if not (len(q10) == len(q100) == len(voltage) == 1000):
        raise ValueError("Expected 1000 voltage-aligned samples")
    if not all(np.isfinite(a).all() for a in [q10, q100, voltage]):
        raise ValueError("Non-finite Qdlin/voltage curve")
    if not (np.all(np.diff(voltage) > 0) or np.all(np.diff(voltage) < 0)):
        raise ValueError("Voltage axis must be monotonic")
    if not np.allclose(np.sort(voltage), np.linspace(2., 3.5, 1000), atol=1e-6):
        raise ValueError("Unexpected voltage range: do not compare unaligned curves")
    early = (cycle >= 2) & (cycle <= 100)
    q = summary["QDischarge"]
    valid_q = np.isfinite(q) & (q > 0) & (q <= 1.2)
    valid = early & valid_q
    variance = float(np.var(q100 - q10))
    if variance <= 0:
        raise ValueError("Zero variance in delta-Q")
    def at(name, k):
        values = summary[name][cycle == k]
        return float(values[0]) if len(values) == 1 and np.isfinite(values[0]) else np.nan
    ir2, ir100 = at("IR", 2), at("IR", 100)
    ct = summary["chargetime"][(cycle >= 2) & (cycle <= 6)]
    ct = ct[np.isfinite(ct) & (ct > 0)]
    t = summary["Tavg"][early]
    t = t[np.isfinite(t)]
    return {
        "log10_dq_variance": np.log10(variance),
        "qd_slope_2_100": float(np.polyfit(cycle[valid], q[valid], 1)[0]) if valid.sum() >= 3 else np.nan,
        "charge_time_mean_2_6": float(np.mean(ct)) if len(ct) else np.nan,
        "ir_change_100_2": ir100 - ir2 if ir100 > 0 and ir2 > 0 else np.nan,
        "tavg_mean_2_100": float(np.mean(t)) if len(t) else np.nan,
        "early_qc_flags": int(np.sum(early & ~valid_q)),
    }
