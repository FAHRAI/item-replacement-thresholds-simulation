import json
from pathlib import Path

import numpy as np
import pytest

from replacement_thresholds import analysis, experiment

ROOT = Path(__file__).resolve().parents[1]
STUDY = experiment.Study.load(ROOT / "config" / "study.json")
RESULTS = ROOT / "results"
SELECTED = json.loads((RESULTS / "selection.json").read_text())["selected"]


def test_simulator_reproduces_recorded_final_worlds():
    recorded = np.load(Path(__file__).parent / "data" / "final_block_000000_000016.npz")
    for bundle in (0, 1):
        seed = int(recorded["seeds"][bundle])
        endpoints, periods, _, _ = experiment.evaluation_bundle(STUDY, seed, bundle, SELECTED)
        np.testing.assert_allclose(endpoints, recorded["endpoints"][bundle], rtol=1e-12, atol=0)
        np.testing.assert_allclose(periods, recorded["periods"][bundle], rtol=1e-12, atol=0)


def test_analysis_reproduces_released_results():
    endpoints, _ = analysis.read_world_endpoints(STUDY, RESULTS / "world_endpoints.csv.gz")
    released = json.loads((RESULTS / "analysis.json").read_text())
    assert json.loads(json.dumps(analysis.analyze(STUDY, endpoints, SELECTED))) == released


def test_reported_primary_result():
    result = json.loads((RESULTS / "analysis.json").read_text())
    primary = result["registered"]["dynamic_minus_static"]
    assert result["bundles"] == 768
    assert round(primary["mean"], 8) == -0.00015258
    assert (round(primary["lower"], 8), round(primary["upper"], 8)) == (-0.0002257, -0.00007946)
    assert result["equivalence"]["equivalent"] is True


def test_sample_size_follows_from_pilot():
    sizing = json.loads((RESULTS / "sample_size.json").read_text())
    n, required = experiment.final_sample_size(STUDY, sizing["pilot_sd"])
    assert n == sizing["final_bundles"] == 768
    assert required == sizing["required"]


@pytest.mark.skipif(not (ROOT / "data" / "development").exists(), reason="development records not downloaded")
def test_selection_reproduces_from_development_records():
    endpoints = experiment.load_blocks(ROOT / "data" / "development")
    selected, scores = experiment.select_policies(STUDY, endpoints)
    released = json.loads((RESULTS / "selection.json").read_text())
    assert selected == released["selected"]
    assert scores["profiles"] == released["profiles"]
