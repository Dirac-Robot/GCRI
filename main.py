from dotenv import load_dotenv
from loguru import logger

from gcri.config import scope
from gcri.graphs.base import GCRIGraph


@scope
def main(config):
    # task = input('Task: ')
    workflow = GCRIGraph(config)
    with open('./tasks/multiobjects.txt') as f:
        task = f.read()
    logger.info(f'🧩 Task: {task}')
    final_result = workflow(task)
    logger.info('🎉 Final Output:')
    logger.info(final_result['final_output'])
    logger.info('🎉 Compressed Output:')
    logger.info(final_result['compressed_output'])


if __name__ == '__main__':
    load_dotenv()
    main()
