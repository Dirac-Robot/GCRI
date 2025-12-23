import os
import json
import requests

from dotenv import load_dotenv
from loguru import logger

from gcri.config import scope
from gcri.graphs.gcri_unit import GCRI
from gcri.tools.cli import get_input


@scope
def main(config):
    load_dotenv()

    dashboard_cfg = getattr(config, 'dashboard', {})
    if dashboard_cfg.get('enabled', False):
        try:
            host = dashboard_cfg.get('host', '127.0.0.1')
            port = dashboard_cfg.get('port', 8000)
            url = f'http://{host}:{port}/api/log'

            def http_sink(message):
                try:
                    record = json.loads(message)
                    requests.post(url, json={'record': record}, timeout=0.1)
                except Exception:
                    pass

            logger.add(http_sink, serialize=True, level='INFO')
            logger.info(f'📡 Dashboard Logging Enabled: {url}')
        except Exception as e:
            logger.error(f'Failed to initialize dashboard sink: {e}')

    worker = GCRI(config)
    logger.info('🤖 GCRI Single Worker Started.')
    logger.info('- Press [Ctrl+C] during input to EXIT.')
    logger.info('- Press [Ctrl+C] during task to ABORT task.')
    logger.info('- Type "q" to quit.\n')
    result = None
    while True:
        try:
            try:
                command = get_input('🧩 Write task directly or path to task is contained: ')
            except KeyboardInterrupt:
                logger.info('\n👋 Exiting GCRI Worker...')
                break
            if not command:
                continue
            elif command.lower() in ('/q', '/quit', '/exit'):
                logger.info('👋 Exiting GCRI Worker...')
                break
            elif command.lower() in ('/r', '/retry'):
                if result is None:
                    logger.warning('⚠️ No previous state found in memory. Please run a task first.')
                    continue
                logger.info('🔄 Retrying with last state.')
                task = result
            elif os.path.exists(command):
                with open(command) as f:
                    task = f.read()
            else:
                task = command
            try:
                result = worker(task)
                logger.info('🎉 Final Output:')
                if result.get('final_output'):
                    logger.info(result['final_output'])
                else:
                    logger.warning('Task finished without definitive final output.')
            except KeyboardInterrupt:
                logger.warning('\n🛑 Task aborted by user (Ctrl+C). Returning to prompt...')
                continue
        except Exception as e:
            logger.error(f'(!) Task is failed with error: {e}')
