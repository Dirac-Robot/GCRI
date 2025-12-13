import os
import sys

from dotenv import load_dotenv
from loguru import logger

from gcri.config import scope
from gcri.graphs.gcri_unit import GCRI


@scope
def main(config):
    load_dotenv()
    worker = GCRI(config)
    logger.info("🤖 GCRI Single Worker Started.")
    logger.info("   - Press [Ctrl+C] during input to EXIT.")
    logger.info("   - Press [Ctrl+C] during task to ABORT task.")
    logger.info("   - Type 'q' to quit.\n")
    while True:
        try:
            logger.info('🧩 Write task directly or path to task is contained: ')
            try:
                command = sys.stdin.buffer.readline().decode('utf-8', errors='ignore').strip()
            except KeyboardInterrupt:
                logger.info('\n👋 Exiting GCRI Worker...')
                break
            if not command:
                continue
            if command.lower() in ('q', 'quit', 'exit'):
                logger.info('👋 Exiting GCRI Worker...')
                break
            if os.path.exists(command):
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