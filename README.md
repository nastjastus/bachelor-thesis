# Inter-case feature encodings for remaining time prediction

Code for my bachelor thesis, "Assessing the Impact of Inter-Case Feature
Encodings for Remaining Time Prediction in Predictive Process Monitoring"
(Anastasia Stus, TUM).

The pipeline computes six inter-case encodings (E1-E6), combines them in all 64
ways, and evaluates each combination with three tree-based models (Random Forest,
CART, XGBoost) under a temporal and a random split. It runs on nine real-world
logs and three synthetic ones.

## Setup

Needs Python 3.9+. I use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running it

Pick the log and split in `config.py` (`ACTIVE` and `SPLIT_MODE`), then run
`python pipeline.py` from the project root. Results land in
`results/<log>_<split>/` as one CSV per model plus a combined `all_results.csv`.

Run everything from the project root: the scripts use relative `data/` paths and
import `config.py` from there.

## What's where

- `config.py` - all settings (log, split, model and encoding parameters)
- `pipeline.py` - the main pipeline (load log, features, train, evaluate, save)
- `evaluate.py` - trains the configurations and computes MAE/RMSE
- `models/` - the three models
- `intercase_encodings/` - the six encodings E1-E6
- `simulators/` - generate the synthetic logs
- `analysis/` - preprocessing and the scripts behind the thesis tables/figures
- `data/` - event logs (see below)
- `results/` - pipeline output

`simulator_e2.py` and `simulator_e2_e5.py` in the root were exploratory and are
not part of the thesis.

## Encodings

- E1 `open_cases_at_time` - how many cases are open at that moment
- E2 `resource_load_at_time` - cases on the same resource
- E3 `peer_cases_in_window` - how many cases started recently
- E4 `avg_delay_in_window` - current speed vs. the historical average
- E5 `queue_length_at_activity` - backlog at a given activity
- E6 `batch_indicator` + `batch_size` - batch co-occurrence

## Synthetic logs

Each simulator writes one log to `data/` and isolates a single mechanism:

- `simulators/simulator_e1_e3.py` -> `synthetic_e1_e3.xes` (system load, E1/E3)
- `simulators/simulator_e4.py` -> `synthetic_e4.xes` (load-independent delay, E4)
- `simulators/simulator_e6.py` -> `synthetic_e6.xes` (batching, E6)

They build on the simulator from the supervising chair (Mustroph, Kunkler &
Rinderle-Ma, ref [14] in the thesis), which is public and not copied in here:
https://github.com/ProbabilisticSuffixPredictionLab/Probabilistic_Suffix_Prediction_U-ED-LSTM_pub

## Analysis scripts

These reproduce the tables and figures. Run them from the project root:

- `marginal_contribution.py` - Table 4
- `make_bars.py` - Figure 3
- `make_heatmap.py` - Figure 4
- `stat_tests.py` - Table 5 (Wilcoxon, Friedman/Nemenyi)
- `check_drift.py` - Table 9
- `check_censoring.py` - censoring correlation (Section 4.5)
- `inspect_ends.py <log>` - Table 8
- `case_duration.py` - Table 2
- `truncate_helpdesk.py` - builds the truncated Helpdesk log
- `filter_bpic12.py` - builds the BPIC2012-W log
- `convert_csv_to_xes.py` - CSV to XES conversion

The first four read `analysis/Ergebnisse_aller_Logs.xlsx`, the collected pipeline
results.

## Event logs

The synthetic logs come from the simulators. The real logs are public benchmarks
and are not in the repo. Download them and put them into `data/` with the file
names `config.py` expects:

| Log | File in data/ | DOI |
|-----|---------------|-----|
| Sepsis | `Sepsis Cases - Event Log.xes` | https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460 |
| BPIC2011 | `hospital_log.xes` | https://doi.org/10.4121/uuid:d9769f3d-0ab0-4fb8-803b-0d1120ffcf54 |
| BPIC2012 | `financial_log.xes` | https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f |
| BPIC2017 | `BPI Challenge 2017 - Offer log.xes` | https://doi.org/10.4121/12705737.v2 |
| BPIC2020 Domestic Decl. | `DomesticDeclarations.xes` | https://doi.org/10.4121/uuid:3f422315-ed9d-4882-891f-e180b5b4feb5 |
| Road Traffic Fines | `Road_Traffic_Fine_Management_Process.xes.gz` | https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5 |
| Production | `production.xes` | https://doi.org/10.4121/uuid:68726926-5ac5-4fab-b873-ee76ea412399 |
| Helpdesk | `helpdesk.xes` | https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb |

`bpic2012_w.xes` and `helpdesk_resolved.xes` are built from the downloaded logs
with `filter_bpic12.py` and `truncate_helpdesk.py`.
