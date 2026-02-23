import os

from dotenv import load_dotenv
from loguru import logger

from gcri.config import scope
from gcri.graphs.gcri_unit import GCRI
from gcri.tools.cli import get_input


@scope
def main(config):
    load_dotenv()
    worker = GCRI(config)
    logger.info('🤖 GCRI Single Worker Started.')

    initial_task = getattr(config, 'initial_task', None)
    if initial_task:
        if os.path.exists(initial_task):
            with open(initial_task) as f:
                initial_task = f.read()
                
        logger.info(f'📝 Running initial task: {initial_task[:100]}...')
        try:
            result = worker(initial_task)
            logger.info('🎉 Final Output:')
            if result.get('final_output'):
                logger.info(result['final_output'])
            else:
                logger.warning('Task finished without definitive final output.')
        except Exception as e:
            logger.error(f'(!) Task failed with error: {e}')
        return

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


if __name__ == '__main__':
    main()
