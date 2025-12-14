import os
import sys

from dotenv import load_dotenv
from loguru import logger

from gcri.config import scope
from gcri.graphs.planner import GCRIMetaPlanner


@scope
def main(config):
    load_dotenv()
    planner = GCRIMetaPlanner(config)
    logger.info("🤖 GCRI Meta Planner Started.")
    logger.info("- Press [Ctrl+C] during input to EXIT.")
    logger.info("- Press [Ctrl+C] during task to ABORT task.")
    logger.info("- Type 'q' to quit.\n")
    result = None
    while True:
        try:
            logger.info('🧩 Write task directly or path to task is contained: ')
            try:
                command = sys.stdin.buffer.readline().decode('utf-8', errors='ignore').strip()
            except KeyboardInterrupt:
                logger.info('\n👋 Exiting GCRI Planner...')
                break
            if not command:
                continue
            elif command.lower() in ('/q', '/quit', '/exit'):
                logger.info('👋 Exiting GCRI Planner...')
                break
            elif command.lower() == ('/r', '/retry'):
                if result is None:
                    logger.warning("⚠️ No previous state.")
                    continue
                task = result
            elif os.path.exists(command):
                with open(command) as f:
                    task = f.read()
            else:
                task = command
            try:
                result = planner(task)
                logger.info('🎉 Final Output:')
                if result.get('final_answer'):
                    logger.info(result['final_answer'])
                else:
                    logger.warning('No final answer provided.')
            except KeyboardInterrupt:
                logger.warning('\n🛑 Task aborted by user (Ctrl+C). Returning to prompt...')
                continue
        except Exception as e:
            logger.error(f'(!) Executing planning is failed with error: {e}')