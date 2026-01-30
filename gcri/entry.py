import sys
from gcri.config import scope
from gcri.launch_gcri import main as run_unit
from gcri.launch_planner import main as run_planner


@scope
def main_entry(config):
    args = sys.argv[1:]
    if args and args[0] == 'cli':
        args.pop(0)
        sys.argv.pop(1)
    if args and args[0] == 'plan':
        run_planner()
    else:
        run_unit()

