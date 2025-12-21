import sys
import subprocess
import time
import os
from loguru import logger
from gcri.config import scope
from gcri.main import main as run_unit
from gcri.main_planner import main as run_planner


def launch_dashboard(config):
    dashboard_cfg = getattr(config, 'dashboard', {})
    if not dashboard_cfg.get('enabled', False):
        logger.error("Dashboard is disabled in config.")
        return

    host = dashboard_cfg.get('host', '127.0.0.1')
    port = str(dashboard_cfg.get('port', 8000))
    frontend_url = dashboard_cfg.get('frontend_url', f'http://{host}:{port}')
    
    logger.info(f"🚀 Launching GCRI Dashboard at {frontend_url}...")
    
    env = os.environ.copy()
    project_root = config.project_dir

    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "gcri.dashboard.backend.main:app", "--host", host, "--port", port],
            cwd=project_root,
            env=env
        )
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped.")


@scope
def main_entry(config):
    args = sys.argv[1:]

    # gcri cli ...
    if args and args[0] == 'cli':
        args.pop(0) # remove 'cli'
        # Adjust sys.argv so internal parsers work if needed
        sys.argv.pop(1) 
        
        # gcri cli plan
        if args and args[0] == 'plan':
            # run_planner usually doesn't take args for 'plan' keyword but let's be safe
            # sys.argv is now ['gcri', 'plan', ...]
            run_planner()
        else:
            # gcri cli
            run_unit()
    
    # gcri ... (Web Mode)
    else:
        # Ignore args for now, just launch dashboard
        launch_dashboard(config)
