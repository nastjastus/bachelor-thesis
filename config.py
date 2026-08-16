"""
config.py

Central configuration file for the pipeline.
This is the only place that needs to be changed when you want to:
  - use a different event log
  - try different parameters
  - enable/disable different encodings
"""

from pathlib import Path

# All available logs. Switch between them via ACTIVE below.
# Each entry: path, res_col (resource column), exclude (resources to exclude),
# end_acts (end activities for the complete-case filter), end_mode.
LOGS = {
    "sepsis":       dict(path="data/Sepsis Cases - Event Log.xes",
                         res_col="org:group", exclude=set(), end_acts=["Release A", "Release B", "Release C", "Release D",
                         "Release E", "Return ER"], end_mode="last"),
    "bpic2011":     dict(path="data/hospital_log.xes",
                         res_col="org:group", exclude=set(), end_acts=None, end_mode="last"),
    "production":   dict(path="data/production.xes",
                         res_col="org:resource", exclude=set(), end_acts=["Final Inspection Q.C.", "Packing"], end_mode="last"),
    "helpdesk":     dict(path="data/helpdesk.xes",
                         res_col="org:resource", exclude=set(), end_acts=["Closed"], end_mode="last"),
    "helpdesk_resolved": dict(path="data/helpdesk_resolved.xes",
                          res_col="org:resource", exclude=set(), end_acts=["Resolve ticket"], end_mode="last"),
    "bpic2012_w":   dict(path="data/bpic2012_w.xes",
                         res_col="org:resource", exclude=set(), end_acts=None, end_mode="last"),
    "bpic2012_all": dict(path="data/financial_log.xes",
                         res_col="org:resource", exclude=set(), end_acts=["A_APPROVED", "A_DECLINED", "A_CANCELLED"],
                     end_mode="contains"),
    "domestic":     dict(path="data/DomesticDeclarations.xes",
                         res_col="org:resource", exclude=set(), end_acts=["Payment Handled", "Declaration REJECTED by EMPLOYEE"], 
                         end_mode="last"),
    "rtf":          dict(path="data/Road_Traffic_Fine_Management_Process.xes.gz",
                         res_col="org:resource", exclude=set(), end_acts=["Payment", "Send for Credit Collection",
                      "Notify Result Appeal to Offender"], end_mode="last"),
    "bpic2017":     dict(path="data/BPI Challenge 2017 - Offer log.xes",
                         res_col="org:resource",
                         exclude={"User_1", "User_126", "User_130", "User_137",
                                  "User_138", "User_143", "User_144"}, end_acts=["O_Cancelled", "O_Accepted", "O_Refused"], 
                                  end_mode="last"),
    "syn_e4":       dict(path="data/synthetic_e4.xes",
                         res_col="org:resource", exclude=set(), end_acts=None, end_mode="last"),
    "syn_e1_e3":    dict(path="data/synthetic_e1_e3.xes",
                         res_col="org:resource", exclude=set(), end_acts=None, end_mode="last"),
    "syn_e6":       dict(path="data/synthetic_e6.xes",
                         res_col="org:resource", exclude=set(), end_acts=None, end_mode="last"),
    "syn_e2_e5":    dict(path="data/synthetic_e2_e5.xes",
                         res_col="org:resource", exclude=set(), end_acts=None, end_mode="last"),
    "syn_e2":       dict(path="data/synthetic_e2.xes",
                         res_col="org:resource", exclude=set(), end_acts=None, end_mode="last"),
}

ACTIVE = "helpdesk"

# Split
SPLIT_MODE          = "temporal"   # "temporal" | "random_case"
#SPLIT_MODE          = "random_case" 

TRAIN_RATIO         = 0.8
RANDOM_STATE        = 42
FILTER_COMPLETE     = True         # only cases with an end activity
DROP_CROSSING_CASES = True         # temporal only: train cases crossing the split point

_cfg = LOGS[ACTIVE]
LOG_PATH = Path(_cfg["path"])
RES_COL = _cfg["res_col"]
EXCLUDE_RESOURCES = _cfg["exclude"]
END_ACTIVITIES = _cfg["end_acts"]
END_MODE = _cfg.get("end_mode", "last")

RESULT_DIR = (f"results/{LOG_PATH.stem}_{SPLIT_MODE}"
              + ("_complete" if FILTER_COMPLETE else ""))

print(f"[CONFIG] {ACTIVE}: {LOG_PATH.name} | RES_COL={RES_COL} | "
      f"{len(EXCLUDE_RESOURCES)} resources excluded | Split={SPLIT_MODE} | "
      f"complete={FILTER_COMPLETE} | → {RESULT_DIR}")

# Column names (standard XES). RES_COL is set per log from LOGS[ACTIVE].
CASE_COL = "case:concept:name"  # case ID
ACT_COL  = "concept:name"       # activity name
TS_COL   = "time:timestamp"

# Encoding parameters
WINDOW_DAYS      = 7    # time window for E3 and E4 (in days)
BATCH_WINDOW_MIN = 5    # time window for E6 (in minutes)

# Random Forest parameters
RF_PARAMS = {
    "n_estimators"    : 100,
    "max_depth"       : 10,
    "min_samples_leaf": 20,
    "n_jobs"          : -1,
    "random_state"    : RANDOM_STATE,
}