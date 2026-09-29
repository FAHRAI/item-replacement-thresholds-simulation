# Static and resource-dependent item-replacement thresholds

Code, configuration and results for the paper

> O. Kholodniak, O. Prokhorov. *Static and Resource-Dependent Item-Replacement Thresholds: Accuracy and Resource
> Use in Simulated Repeated Assessment.* Manuscript under review, 2026.

## The study in brief

An assessment system repeatedly reuses a small pool of items. Items become familiar or compromised with exposure, so
the system must decide when to replace them; each replacement spends calibration responses and one item from a finite
reserve.

- **World.** 48 learners, 6 active items, 96 responses per period, 30 periods. Each item becomes compromised after a
  geometric number of exposures. 18 scenarios cross the initial reserve (3, 6, 12), the calibration cost (12, 24, 48
  responses) and the compromise hazard (0.001, 0.008).
- **Policies.** An item is replaced when its standardized residual sum exceeds a threshold. The static policy uses a
  threshold fixed by the initial reserve; the dynamic policy scales it by remaining time and remaining reserve,
  `z * ((T - t) / (R_t + 1) / (T / (R_0 + 1))) ** gamma`. Both are searched on the same 48-point grid, so the static
  policy is the `gamma = 0` case of the dynamic one.
- **Design.** 128 development seed bundles select the policies, 256 pilot bundles set the final sample size, and 768
  final bundles are used for inference. Every bundle is evaluated under both policies, all scenarios and three modes
  (baseline, zero diagnostic practice, low filter increment), so all comparisons are paired.
- **Main result.** The dynamic-minus-static difference in probability RMSE is −0.00015258 (97.5% interval
  [−0.00022570, −0.00007946]); the 95% interval lies within the equivalence margin of ±0.001.

The study protocol, simulation code and configuration were fixed and fingerprinted with SHA-256 before any
development, pilot or final data were generated. The timestamped records are available from the authors on request.

## Repository layout

| Path | Contents |
|---|---|
| `src/replacement_thresholds/simulator.py` | One simulated world: learners, items, compromise, calibration, replacement rule |
| `src/replacement_thresholds/experiment.py` | Development search, policy selection, pilot sizing, evaluation runs, storage |
| `src/replacement_thresholds/analysis.py` | Registered contrasts, equivalence test, descriptive and per-period summaries |
| `src/replacement_thresholds/figures.py` | Figures 1–3 of the paper |
| `config/study.json` | Scenarios, modes, grids, seeds, precision targets and contrast definitions |
| `results/` | Outputs reported in the paper (see below) |
| `tests/` | Reproduction checks against recorded worlds and released results |

## Reproducing the results

The recorded run used Python 3.12 and NumPy 2.3.5 on macOS (arm64). Other platforms may differ in the last digits;
the tests use a relative tolerance of 1e-12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[figures,test]"
pytest
python -m replacement_thresholds figures
```

`pytest` re-simulates two recorded final bundles (108 worlds each) and recomputes the analysis from
`results/world_endpoints.csv.gz`. To start from the raw records, download the data archive (see [Data](#data)) into
`data/` and run

```bash
python -m replacement_thresholds select    # policies and selection stability from the development records
python -m replacement_thresholds pilot     # pilot standard deviations and the final sample size
python -m replacement_thresholds analyze   # results/ from the final records
```

Commands skip blocks that already exist. A full re-simulation runs `develop`, `select`, `pilot`, `final` and `analyze`
on an empty `data/` directory; use `--workers` to set the number of processes.

## Results

| File | Contents |
|---|---|
| `selection.json` | Selected static and dynamic policies and the development scores for every `(gamma, z)` pair |
| `selection_stability.json` | Bootstrap frequencies of the selected parameters (descriptive) |
| `sample_size.json` | Pilot standard deviations, required sample sizes and the final number of bundles |
| `analysis.json` | Registered contrasts (families A, C, D), the equivalence test, descriptive contrasts and cell means |
| `world_endpoints.csv.gz` | Endpoints of all 82,944 final worlds |
| `period_difference.json` | Mean dynamic-minus-static RMSE by period (Figure 2b) |
| `resources.json` | Mean remaining reserve and cumulative calibration responses by period (Figure 3) |

## Data

The raw development, pilot and final records (`data/`, 143 MB: one `.npz` file per block with seeds, endpoints and
per-period values, plus replacement events and a response trace for bundle 0) are archived on Zenodo:
https://doi.org/10.5281/zenodo.XXXXXXX

## Citation

See `CITATION.cff`.

## License

Code: MIT. Results and data: CC BY 4.0.
