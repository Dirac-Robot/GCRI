import sys
from gcri.main import main as run_unit
from gcri.main_planner import main as run_planner


def cli_entry():
    if len(sys.argv) > 1 and sys.argv[1] == 'plan':
        sys.argv.pop(1)
        run_planner()
    else:
        run_unit()
