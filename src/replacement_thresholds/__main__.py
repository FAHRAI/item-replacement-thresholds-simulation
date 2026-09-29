"""Command-line entry point: ``python -m replacement_thresholds <command>``.

Commands, in the order the study was run:

    develop   simulate the development bundles over the full (gamma, z) grid
    select    choose the static and dynamic policies and write their bootstrap stability
    pilot     simulate the pilot bundles and derive the final sample size
    final     simulate the final bundles
    analyze   registered contrasts, equivalence test and descriptive summaries
    figures   Figures 1–3 from the files in results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import analysis, experiment

ROOT = Path.cwd()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _read(path):
    return json.loads(path.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="replacement_thresholds")
    parser.add_argument("command", choices=["develop", "select", "pilot", "final", "analyze", "figures"])
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "study.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    study = experiment.Study.load(args.config)
    data, results = args.data, args.results

    if args.command == "develop":
        experiment.run_phase(
            study, "development", study.config["development"]["bundles"], data / "development", workers=args.workers
        )

    elif args.command == "select":
        endpoints = experiment.load_blocks(data / "development")
        selected, scores = experiment.select_policies(study, endpoints)
        _write(results / "selection.json", dict(selected=selected, **scores))
        _write(results / "selection_stability.json", experiment.selection_stability(study, endpoints))

    elif args.command == "pilot":
        selected = _read(results / "selection.json")["selected"]
        experiment.run_phase(
            study, "pilot", study.config["pilot"]["bundles"], data / "pilot", selected=selected, workers=args.workers
        )
        sd = experiment.pilot_standard_deviations(study, experiment.load_blocks(data / "pilot"))
        n, required = experiment.final_sample_size(study, sd)
        _write(results / "sample_size.json", dict(pilot_sd=sd, required=required, final_bundles=n))

    elif args.command == "final":
        selected = _read(results / "selection.json")["selected"]
        n = _read(results / "sample_size.json")["final_bundles"]
        experiment.run_phase(study, "final", n, data / "final", selected=selected, workers=args.workers)

    elif args.command == "analyze":
        selected = _read(results / "selection.json")["selected"]
        endpoints = experiment.load_blocks(data / "final")
        seeds = experiment.load_blocks(data / "final", "seeds").tolist()
        _write(results / "analysis.json", analysis.analyze(study, endpoints, selected))
        analysis.write_world_endpoints(study, endpoints, seeds, results / "world_endpoints.csv.gz")
        by_period, resources = analysis.period_summaries(study, experiment.load_blocks(data / "final", "periods"))
        _write(results / "period_difference.json", by_period)
        _write(results / "resources.json", resources)

    elif args.command == "figures":
        from . import figures

        figures.make_all(results, args.figures)


if __name__ == "__main__":
    main()
