# =====================================================================================#
# ward_select.py - picks which ward every file uses, before anything else loads
# =====================================================================================#
#two wards exist:
#   paeds   - the paediatric ward: 4 beds, a marker at every bedside, a charging dock (default)
#   classic - the original 6 x 4 m ward with one bedside marker (all results in RESULTS.md)
#
#choose with --ward on any script, or the WARD environment variable:
#   python run_delivery.py --ward classic
#   python Analysis/run_trials.py --ward classic
#Build_scene.py imports this, so every script sees the same choice. it removes --ward from
#the command line so each script's own argparse never sees it, and stores the choice in
#os.environ so worker processes (multiprocessing) inherit it

import os
import sys

WARDS = ("paeds", "classic")

for i, arg in enumerate(list(sys.argv)):
    if arg == "--ward" and i + 1 < len(sys.argv):
        os.environ["WARD"] = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        break
    if arg.startswith("--ward="):
        os.environ["WARD"] = arg.split("=", 1)[1]
        del sys.argv[i]
        break

WARD = os.environ.get("WARD", "paeds").strip().lower()
if WARD not in WARDS:
    raise SystemExit(f"Unknown ward '{WARD}'. Choose one of: {', '.join(WARDS)}")
os.environ["WARD"] = WARD
