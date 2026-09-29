"""Development search, policy selection, pilot sizing and the final evaluation run.

Worlds are grouped into seed bundles. Bundle ``i`` of a phase uses the seed ``(base << 32) | i`` and
is evaluated under every mode, scenario and policy, so all comparisons are paired within bundles.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .simulator import ENDPOINT_FIELDS, PERIOD_FIELDS, Policy, Scenario, run_world

RESERVES = ("3", "6", "12")


class Study:
    def __init__(self, config):
        self.config = config
        world = config["world"]
        self.scenarios = {}
        for reserve in world["reserves"]:
            for size in world["calibration_sizes"]:
                for hazard in world["hazards"]:
                    self.scenarios[f"r{reserve}_c{size}_h{hazard:g}"] = dict(
                        reserve=reserve,
                        calibration_size=size,
                        hazard=hazard,
                        horizon=world["horizon"],
                        rehearsal=world["rehearsal"],
                    )
        # Lexicographic scenario order fixes the summation order of every average.
        self.scenario_names = sorted(self.scenarios)
        self.modes = list(config["modes"])
        self.policies = list(config["policies"])

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def scenario(self, name, mode="baseline"):
        m = self.config["modes"][mode]
        return Scenario(
            **self.scenarios[name], diagnostic_gain=m["diagnostic_gain"], filter_increment=m["filter_increment"]
        )

    def reserve_of(self, name):
        return self.scenarios[name]["reserve"]

    def seeds(self, phase, bundles):
        base = self.config["seeds"][phase]
        return [(base << 32) | int(i) for i in bundles]

    def family_z(self, family):
        tests = self.config["sample_size"]["tests_per_family"][family]
        return NormalDist().inv_cdf(1 - self.config["sample_size"]["alpha_per_family"] / (2 * tests))


# Development: every (gamma, z) pair on every scenario, baseline mode only.


def development_bundle(study, seed):
    d = study.config["development"]
    out = np.empty((len(d["gamma_grid"]), len(d["z_grid"]), len(study.scenario_names), len(ENDPOINT_FIELDS)))
    for gi, gamma in enumerate(d["gamma_grid"]):
        for zi, z in enumerate(d["z_grid"]):
            for si, name in enumerate(study.scenario_names):
                result = run_world(seed, study.scenario(name), Policy(z=z, gamma=gamma), label=name)
                out[gi, zi, si] = [result.endpoints[k] for k in ENDPOINT_FIELDS]
    return out


def select_policies(study, endpoints):
    """Choose z per initial reserve for each gamma, then the gamma with the lowest mean RMSE.

    The static policy is the gamma = 0 profile; exact ties go to the lowest z and the lowest gamma.
    """
    d = study.config["development"]
    rmse = endpoints[..., ENDPOINT_FIELDS.index("rmse")]
    mean_over_bundles = np.mean(rmse, axis=0)
    by_reserve = {r: [j for j, s in enumerate(study.scenario_names) if study.reserve_of(s) == int(r)] for r in RESERVES}

    profiles = []
    for gi, gamma in enumerate(d["gamma_grid"]):
        z_by_reserve, indices, scores = {}, {}, []
        for r in RESERVES:
            z_scores = np.mean(mean_over_bundles[gi][:, by_reserve[r]], axis=1)
            zi = int(np.argmin(z_scores))
            z_by_reserve[r], indices[r] = d["z_grid"][zi], zi
            scores.append(float(z_scores[zi]))
        profiles.append(dict(gamma=gamma, z_by_reserve=z_by_reserve, indices=indices, score=float(np.mean(scores))))

    static = profiles[0]
    dynamic = profiles[int(np.argmin([p["score"] for p in profiles]))]
    selected = {
        "static": dict(gamma=static["gamma"], z_by_reserve=static["z_by_reserve"]),
        "dynamic": dict(gamma=dynamic["gamma"], z_by_reserve=dynamic["z_by_reserve"]),
    }

    anchors = [d["z_grid"].index(z) for z in d["z_anchors"]]
    coarse_static = {}
    for r in RESERVES:
        z_scores = np.mean(mean_over_bundles[0][:, by_reserve[r]], axis=1)
        zi = min(anchors, key=lambda j: (z_scores[j], j))
        coarse_static[r] = dict(z=d["z_grid"][zi], rmse=float(z_scores[zi]))

    scores = dict(
        profiles=profiles,
        coarse_static_development=coarse_static,
        mean_rmse_by_gamma_z_scenario=mean_over_bundles.tolist(),
    )
    return selected, scores


def selection_stability(study, endpoints):
    """Bootstrap frequencies of the selected parameters (descriptive; selection itself is fixed)."""
    d = study.config["development"]
    rng = np.random.default_rng(np.random.SeedSequence([study.config["seeds"]["development"], d["bootstrap_stream"]]))
    extremes = (d["z_grid"][0], d["z_grid"][-1])
    counts = Counter()
    for _ in range(d["bootstrap_replicates"]):
        sample = rng.integers(0, len(endpoints), size=len(endpoints))
        selected, _ = select_policies(study, endpoints[sample])
        for name, p in selected.items():
            counts[f"{name}:gamma:{p['gamma']}"] += 1
            for r, z in p["z_by_reserve"].items():
                counts[f"{name}:R{r}:z:{z}"] += 1
                counts[f"{name}:R{r}:boundary"] += int(z in extremes)
            counts[f"{name}:any_z_boundary"] += int(any(z in extremes for z in p["z_by_reserve"].values()))
    replicates = d["bootstrap_replicates"]
    return dict(replicates=replicates, frequencies={k: v / replicates for k, v in sorted(counts.items())})


# Pilot and final evaluation: selected policies under all modes and scenarios.


def policy_for(selected, name, reserve):
    p = selected[name]
    return Policy(z=p["z_by_reserve"][str(reserve)], gamma=p["gamma"])


def evaluation_bundle(study, seed, bundle, selected, trace=False):
    shape = (len(study.modes), len(study.scenario_names), len(study.policies))
    endpoints = np.empty(shape + (len(ENDPOINT_FIELDS),))
    periods = np.empty(shape + (study.config["world"]["horizon"], len(PERIOD_FIELDS)))
    events, responses = [], []
    for mi, mode in enumerate(study.modes):
        for si, name in enumerate(study.scenario_names):
            scenario = study.scenario(name, mode)
            cache = {}
            for pi, policy_name in enumerate(study.policies):
                policy = policy_for(selected, policy_name, scenario.reserve)
                if policy not in cache:
                    cache[policy] = run_world(seed, scenario, policy, label=name, trace=trace)
                result = cache[policy]
                endpoints[mi, si, pi] = [result.endpoints[k] for k in ENDPOINT_FIELDS]
                periods[mi, si, pi] = [[row[k] for k in PERIOD_FIELDS] for row in result.periods]
                tag = dict(bundle_id=bundle, mode=mode, policy=policy_name, seed=str(seed))
                events.extend({**row, **tag} for row in result.replacements)
                responses.extend({**row, **tag} for row in result.responses)
    return endpoints, periods, events, responses


def paired_contrasts(study, endpoints, definitions):
    """Seed-bundle vectors of RMSE contrasts; each term averages RMSE over its scenarios."""
    rmse = endpoints[..., ENDPOINT_FIELDS.index("rmse")]
    vectors = {}
    for entry in definitions:
        v = np.zeros(len(endpoints))
        for mode, policy, reserve, coefficient in entry["terms"]:
            mi, pi = study.modes.index(mode), study.policies.index(policy)
            columns = [
                j for j, s in enumerate(study.scenario_names) if reserve is None or study.reserve_of(s) == reserve
            ]
            v += coefficient * np.mean(rmse[:, mi, :, pi][:, columns], axis=1)
        vectors[entry["id"]] = v
    return vectors


def registered_contrasts(study):
    return [entry for entries in study.config["contrasts"].values() for entry in entries]


def pilot_standard_deviations(study, endpoints):
    vectors = paired_contrasts(study, endpoints, registered_contrasts(study))
    return {k: float(np.std(v, ddof=1)) for k, v in vectors.items()}


def final_sample_size(study, pilot_sd):
    """N = round_to * ceil(max(minimum, max_c (z_c * sd_c / h_c)^2) / round_to)."""
    spec = study.config["sample_size"]
    required = {}
    for family, entries in study.config["contrasts"].items():
        for entry in entries:
            required[entry["id"]] = (study.family_z(family) * pilot_sd[entry["id"]] / spec["half_width"][family]) ** 2
    raw = max(spec["minimum"], *required.values())
    n = int(spec["round_to"] * math.ceil(raw / spec["round_to"]))
    return n, required


# Storage: one directory per block of bundles with an .npz array file.


def block_name(ids):
    return f"block_{ids[0]:06d}_{ids[-1] + 1:06d}"


def _run_block(args):
    config, phase, ids, selected, directory = args
    study = Study(config)
    seeds = study.seeds(phase, ids)
    trace_bundles = set(config["execution"]["trace_bundles"])
    target = Path(directory) / block_name(ids)
    target.mkdir(parents=True, exist_ok=False)
    arrays = dict(ids=np.asarray(ids, dtype=np.int64), seeds=np.asarray([str(s) for s in seeds], dtype="U20"))
    if phase == "development":
        arrays["endpoints"] = np.asarray([development_bundle(study, s) for s in seeds])
    else:
        endpoints, periods, events, responses = [], [], [], []
        for bundle, seed in zip(ids, seeds, strict=True):
            e, p, ev, resp = evaluation_bundle(study, seed, bundle, selected, trace=bundle in trace_bundles)
            endpoints.append(e)
            periods.append(p)
            events += ev
            responses += resp
        arrays["endpoints"] = np.asarray(endpoints)
        arrays["periods"] = np.asarray(periods)
        for name, rows in (("events.json.gz", events), ("responses.json.gz", responses)):
            with gzip.open(target / name, "wt") as f:
                json.dump(rows, f, allow_nan=False)
    np.savez_compressed(target / "data.npz", **arrays)
    return target.name


def run_phase(study, phase, bundles, directory, selected=None, workers=4):
    size = 1 if phase == "development" else study.config["execution"]["evaluation_block_size"]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    blocks = [list(range(a, min(a + size, bundles))) for a in range(0, bundles, size)]
    todo = [ids for ids in blocks if not (directory / block_name(ids) / "data.npz").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_block, (study.config, phase, ids, selected, str(directory))) for ids in todo]
        for future in as_completed(futures):
            print(f"{phase}: {future.result()}", flush=True)


def load_blocks(directory, key="endpoints"):
    blocks = sorted(Path(directory).iterdir(), key=lambda p: int(p.name.split("_")[1]))
    arrays = []
    for block in blocks:
        with np.load(block / "data.npz", allow_pickle=False) as data:
            arrays.append(data[key])
    return np.concatenate(arrays, axis=0)
