from dotenv import load_dotenv
load_dotenv()

from gcri.config import scope
from gcri.graphs.gcri_unit import GCRI
from loguru import logger


@scope
def run_test(config):
    config.run_dir = '/Users/vanta/Documents/GCRI/workspace/.gcri'
    
    worker = GCRI(config)
    task = '월드 모델에 대해 AI 전문가에게 설명하는 1장 미만의 리포트를 작성하라'
    
    logger.info(f'Running task: {task}')
    result = worker(task)
    
    logger.info('🎉 Final Output:')
    if result.get('final_output'):
        logger.info(result['final_output'])
    else:
        logger.warning('Task finished without definitive final output.')
    
    return result


if __name__ == '__main__':
    run_test()
