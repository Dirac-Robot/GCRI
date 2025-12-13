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
    while True:
        try:
            logger.info('🧩 Write task directly or path to task is contained: ')
            command = sys.stdin.buffer.readline().decode('utf-8', errors='ignore').strip()
            if os.path.exists(command):
                with open(command) as f:
                    task = f.read()
            else:
                task = command
            result = planner(task)
            logger.info('🎉 Final Output:')
            logger.info(result['final_answer'])
        except Exception as e:
            logger.error(f'(!) Executing planning is failed with error: {e}')
