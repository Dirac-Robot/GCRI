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
        return None

    host = dashboard_cfg.get('host', '127.0.0.1')
    port = str(dashboard_cfg.get('port', 8000))
    frontend_url = dashboard_cfg.get('frontend_url', f'http://{host}:{port}')

    # Path to dashboard backend main module
    # Assuming running from project root where GCRI package is located
    # We need to run uvicorn as a module.
    # "GCRI/dashboard/backend/main.py" -> "dashboard.backend.main:app"
    # usage: python -m uvicorn ...
    
    logger.info(f"🚀 Launching GCRI Dashboard at {frontend_url}...")
    
    # We must ensure the current directory allows importing 'dashboard'
    env = os.environ.copy()
    project_root = config.project_dir
    # Add 'GCRI' to pythonpath if needed? 
    # Actually, dashboard is in GCRI/dashboard. if we run from GCRI/, 'dashboard' is top level provided we have __init__.py or namespace package.
    # But dashboard is NOT a package unless I added __init__.py.
    # I should add __init__.py to dashboard and dashboard/backend.

    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "gcri.dashboard.backend.main:app", "--host", host, "--port", port, "--log-level", "error"],
        cwd=project_root, # Run from project root
        env=env,
        stderr=subprocess.DEVNULL, # Suppress uvicorn logs in main terminal
        stdout=subprocess.DEVNULL
    )
    time.sleep(1) # Give it a moment
    return process


def cli_entry():
    config = scope()
    dashboard_process = launch_dashboard(config)

    try:
        if len(sys.argv) > 1 and sys.argv[1] == 'plan':
            sys.argv.pop(1)
            run_planner()
        else:
            run_unit()
    finally:
        if dashboard_process:
            logger.info("🛑 Shutting down dashboard...")
            dashboard_process.terminate()
