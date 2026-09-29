"""Registered contrasts, the equivalence test and descriptive summaries of the final run."""

from __future__ import annotations

import csv
import gzip
import math
from statistics import NormalDist

import numpy as np

from .experiment import paired_contrasts, registered_contrasts
from .simulator import ENDPOINT_FIELDS, PERIOD_FIELDS


def summarize(values, z, identity=False):
    """Normal-theory interval for the mean of seed-bundle contrasts."""
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / math.sqrt(n)
    half_width = z * se
    if identity and not np.array_equal(values, np.zeros(n)):
        raise ValueError("identical policies must give zero contrasts")
    status = "structural_identity" if identity else ("estimated" if se > 0 else "undefined_zero_variance")
    return dict(
        n=n,
        mean=mean,
        sd=sd,
        se=se,
        lower=mean - half_width,
        upper=mean + half_width,
        half_width=half_width,
        status=status,
        reject_zero=abs(mean) > half_width if status == "estimated" else None,
    )


def analyze(study, endpoints, selected):
    identity = selected["dynamic"] == selected["static"]
    vectors = paired_contrasts(study, endpoints, registered_contrasts(study))
    half_widths = study.config["sample_size"]["half_width"]

    registered = {}
    for family, entries in study.config["contrasts"].items():
        for entry in entries:
            row = summarize(vectors[entry["id"]], study.family_z(family), identity)
            row.update(
                family=family,
                target_half_width=half_widths[family],
                target_exceeded=row["half_width"] > half_widths[family],
            )
            registered[entry["id"]] = row

    spec = study.config["equivalence"]
    primary = study.config["contrasts"]["A"][0]["id"]
    equivalence = summarize(vectors[primary], NormalDist().inv_cdf(1 - spec["alpha"]), identity)
    equivalence.pop("reject_zero")
    equivalence.update(margin=spec["margin"], alpha=spec["alpha"], confidence_level=1 - 2 * spec["alpha"])
    if equivalence["status"] == "estimated":
        equivalence["equivalent"] = equivalence["lower"] > -spec["margin"] and equivalence["upper"] < spec["margin"]
    else:
        equivalence["equivalent"] = None

    descriptive = {}
    z95 = NormalDist().inv_cdf(0.975)
    for name, v in paired_contrasts(study, endpoints, study.config["descriptive_contrasts"]).items():
        row = summarize(v, z95, identity)
        row.pop("reject_zero")
        descriptive[name] = row

    cells = []
    for mi, mode in enumerate(study.modes):
        for si, scenario in enumerate(study.scenario_names):
            for fi, field in enumerate(ENDPOINT_FIELDS):
                static, dynamic = endpoints[:, mi, si, 0, fi], endpoints[:, mi, si, 1, fi]
                cells.append(
                    dict(
                        mode=mode,
                        scenario=scenario,
                        endpoint=field,
                        static_mean=float(np.mean(static)),
                        dynamic_mean=float(np.mean(dynamic)),
                        paired_difference=float(np.mean(dynamic - static)),
                    )
                )

    return dict(
        bundles=len(endpoints),
        selected=selected,
        registered=registered,
        equivalence=equivalence,
        descriptive_contrasts=descriptive,
        cell_means=cells,
    )


def write_world_endpoints(study, endpoints, seeds, path):
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bundle_id", "seed", "mode", "scenario", "policy", *ENDPOINT_FIELDS])
        for b in range(len(endpoints)):
            for mi, mode in enumerate(study.modes):
                for si, scenario in enumerate(study.scenario_names):
                    for pi, policy in enumerate(study.policies):
                        writer.writerow([b, seeds[b], mode, scenario, policy, *endpoints[b, mi, si, pi].tolist()])


def read_world_endpoints(study, path):
    with gzip.open(path, "rt", newline="") as f:
        rows = list(csv.DictReader(f))
    bundles = 1 + max(int(r["bundle_id"]) for r in rows)
    shape = (bundles, len(study.modes), len(study.scenario_names), len(study.policies), len(ENDPOINT_FIELDS))
    endpoints = np.full(shape, np.nan)
    seeds = [""] * bundles
    for r in rows:
        b = int(r["bundle_id"])
        seeds[b] = r["seed"]
        mode, scenario = study.modes.index(r["mode"]), study.scenario_names.index(r["scenario"])
        endpoints[b, mode, scenario, study.policies.index(r["policy"])] = [float(r[k]) for k in ENDPOINT_FIELDS]
    if np.isnan(endpoints).any():
        raise ValueError("incomplete endpoint table")
    return endpoints, seeds


def period_summaries(study, periods):
    """Per-period summaries of the baseline mode used by Figures 2b and 3."""
    base = study.modes.index("baseline")
    rmse = np.sqrt(periods[:, base, :, :, :, PERIOD_FIELDS.index("mse")])
    difference = (rmse[:, :, 1, :] - rmse[:, :, 0, :]).mean(axis=1)
    horizon = difference.shape[1]
    by_period = dict(
        period=list(range(1, horizon + 1)),
        mean=difference.mean(axis=0).tolist(),
        se=(difference.std(axis=0, ddof=1) / np.sqrt(len(difference))).tolist(),
    )

    reserve_left = periods[:, base, :, :, :, PERIOD_FIELDS.index("reserve_left")]
    calibration = np.cumsum(periods[:, base, :, :, :, PERIOD_FIELDS.index("calibration")], axis=-1)
    resources = []
    for reserve in study.config["world"]["reserves"]:
        columns = [j for j, s in enumerate(study.scenario_names) if study.reserve_of(s) == reserve]
        for pi, policy in enumerate(study.policies):
            left = reserve_left[:, columns, pi, :].mean(axis=(0, 1))
            spent = calibration[:, columns, pi, :].mean(axis=(0, 1))
            for t in range(horizon):
                resources.append(
                    dict(
                        initial_reserve=reserve,
                        policy=policy,
                        period=t + 1,
                        reserve_left=float(left[t]),
                        cumulative_calibration=float(spent[t]),
                    )
                )
    return by_period, resources
