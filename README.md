# Pipeline Forecasting Engine

Point-in-time correct sales-pipeline forecasting: calibrated deal
probabilities, hierarchical (shrunk) rep win rates, correlated aggregation
with honest intervals, scenario what-ifs, and rep-level roll-ups — with the
classic stage-weighted forecast built as a baseline and measured against.

[![CI](https://github.com/syedjafri06193/Pipeline-Forecasting-Engine-proj/actions/workflows/ci.yml/badge.svg)](https://github.com/syedjafri06193/Pipeline-Forecasting-Engine-proj/actions/workflows/ci.yml)

## Highlights

- **Leakage prevented structurally** — one function reads historical state, and a property test covers every feature; three planted leaks prove the test bites
- **Shrinkage for rep win rates** — 29% lower error against the true rate than raw per-rep rates
- **Correlated Monte Carlo aggregation** — 80% intervals that actually cover ~80%, versus 16% when deals are assumed independent
- **Walk-forward backtest** across six quarters and four horizons, with calibration (ECE), Brier and CRPS
- **Measured critique of the design spec** — four defects found and quantified, not just asserted

All figures come from simulated data where the ground truth is known by construction.

## Quick start

Requires Python 3.10+.

```bash
cd v1
pip install -r requirements.txt -e .
python -m pytest -q                 # 82 tests
python -m pfe.cli backtest          # the model comparison table
python -m pfe.cli forecast          # one quarter, explained
python -m pfe.cli reps              # rep roll-ups with intervals
```

The full write-up of results is in [`v1/README.md`](v1/README.md).

## Repository layout

```
.
├── 1—Documentation/   all documentation (index in its README)
│   ├── design-guide.md     full design & build guide (also as PDF)
│   └── v1/                 methodology, backtest results, data availability, spec notes
└── v1/
    ├── src/pfe/       synth, snapshot jobs, point-in-time reads, features, models,
    │                  aggregation, scenarios, backtest harness, CLI
    └── tests/         82 tests incl. leakage property tests and measured findings
```

## Documentation

All documentation lives in [`1—Documentation/`](1—Documentation/README.md).

## Versions and feedback

| Version | Summary | Feedback |
|---|---|---|
| [v1](v1/) | Point-in-time features, calibrated hazard + cohort models, shrinkage, correlated aggregation, backtest | — |

Add a row per version as new iterations land.
