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
    while True:
        logger.info('🧩 Write task directly or path to task is contained: ')
        command = sys.stdin.buffer.readline().decode('utf-8', errors='ignore').strip()
        if os.path.exists(command):
            with open(command) as f:
                task = f.read()
        else:
            task = command
        result = worker(task)
        logger.info('🎉 Final Output:')
        logger.info(result['final_output'])
