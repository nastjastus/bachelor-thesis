"""
config.py

Zentrale Konfigurationsdatei für die Pipeline.
Hier ist die einzige Stelle an der was verändert werden muss wenn man:
  - einen anderen Event Log verwenden will
  - andere Parameter ausprobieren will
  - andere Encodings aktivieren/deaktivieren will
"""

from pathlib import Path

# Log Pfad
# Hier den Pfad zu der .xes Datei eintragen
LOG_PATH = Path("data/BPI Challenge 2017 - Offer log.xes")

# Spaltennamen
# Standard XES Spaltennamen
# Werden nur geändern wenn der Log andere Namen hat
CASE_COL = "case:concept:name"
ACT_COL  = "concept:name"
TS_COL   = "time:timestamp"
RES_COL  = "org:resource"

# Train/Test Split
TRAIN_RATIO  = 0.8
RANDOM_STATE = 42

# Encoding Parameter
WINDOW_DAYS      = 7    # Zeitfenster für E3 und E4 (in Tagen)
BATCH_WINDOW_MIN = 5    # Zeitfenster für E6 (in Minuten)

# Log-spezifische Einstellungen
# Muss für jeden neuen Log manuell überprüft und angepasst werden
# Automatische Systeme die keinen echten Ressourcen-Wettbewerb darstellen
# Leer lassen wenn nicht bekannt: set()
EXCLUDE_RESOURCES = {"User_1"}  # BPI 2017: automatisches Stornierungssystem

# Random Forest Parameter
RF_PARAMS = {
    "n_estimators"    : 100,
    "max_depth"       : 10,
    "min_samples_leaf": 20,
    "n_jobs"          : -1,
    "random_state"    : RANDOM_STATE,
}